# P0 CRITICAL BUGS - Why DeepCritical Produces Garbage Results

**Date:** November 27, 2025
**Status:** CRITICAL - App is functionally useless
**Severity:** P0 (Blocker)

## TL;DR

The app produces garbage because:
1. **BioRxiv search doesn't work** - returns random papers
2. **Free tier LLM is too dumb** - can't identify drugs
3. **Query construction is naive** - no optimization for PubMed/CT.gov syntax
4. **Loop terminates too early** - 5 iterations isn't enough

---

## P0-001: BioRxiv Search is Fundamentally Broken

**File:** `src/tools/biorxiv.py:248-286`

**The Problem:**
The bioRxiv API **DOES NOT SUPPORT KEYWORD SEARCH**.

The code does this:
```python
# Fetch recent papers (last 90 days, first 100 papers)
url = f"{self.BASE_URL}/{self.server}/{interval}/0/json"
# Then filter client-side for keywords
```

**What Actually Happens:**
1. Fetches the first 100 papers from medRxiv in the last 90 days (chronological order)
2. Filters those 100 random papers for query keywords
3. Returns whatever garbage matches

**Result:** For "Long COVID medications", you get random papers like:
- "Calf muscle structure-function adaptations"
- "Work-Life Balance of Ophthalmologists During COVID"

These papers contain "COVID" somewhere but have NOTHING to do with Long COVID treatments.

**Root Cause:** The `/0/json` pagination only returns 100 papers. You'd need to paginate through ALL papers (thousands) to do proper keyword filtering.

**Fix Options:**
1. **Remove BioRxiv entirely** - It's unusable without proper search API
2. **Use a different preprint aggregator** - Europe PMC has preprints WITH search
3. **Add pagination** - Fetch all papers (slow, expensive)
4. **Use Semantic Scholar API** - Has preprints and proper search

---

## P0-002: Free Tier LLM Cannot Perform Drug Identification

**File:** `src/agent_factory/judges.py:153-211`

**The Problem:**
Without an API key, the app uses `HFInferenceJudgeHandler` with:
- Llama 3.1 8B Instruct
- Mistral 7B Instruct

These are **7-8 billion parameter models**. They cannot:
- Reliably parse complex biomedical abstracts
- Identify drug candidates from scientific text
- Generate structured JSON output consistently
- Reason about mechanism of action

**Evidence of Failure:**
```python
# From MockJudgeHandler - the honest fallback when LLM fails
drug_candidates=[
    "Drug identification requires AI analysis",
    "Enter API key above for full results",
]
```

The team KNEW the free tier can't identify drugs and added this message.

**Root Cause:** Drug repurposing requires understanding:
- Drug mechanisms
- Disease pathophysiology
- Clinical trial phases
- Statistical significance

This requires GPT-4 / Claude Sonnet class models (100B+ parameters).

**Fix Options:**
1. **Require API key** - No free tier, be honest
2. **Use larger HF models** - Llama 70B or Mixtral 8x7B (expensive on free tier)
3. **Hybrid approach** - Use free tier for search, require paid for synthesis

---

## P0-003: PubMed Query Not Optimized

**File:** `src/tools/pubmed.py:54-71`

**The Problem:**
The query is passed directly to PubMed without optimization:
```python
search_params = self._build_params(
    db="pubmed",
    term=query,  # Raw user query!
    retmax=max_results,
    sort="relevance",
)
```

**What User Enters:** "What medications show promise for Long COVID?"

**What PubMed Receives:** `What medications show promise for Long COVID?`

**What PubMed Should Receive:**
```
("long covid"[Title/Abstract] OR "post-COVID"[Title/Abstract] OR "PASC"[Title/Abstract])
AND (drug[Title/Abstract] OR treatment[Title/Abstract] OR medication[Title/Abstract] OR therapy[Title/Abstract])
AND (clinical trial[Publication Type] OR randomized[Title/Abstract])
```

**Root Cause:** No query preprocessing or medical term expansion.

**Fix Options:**
1. **Add query preprocessor** - Extract medical entities, expand synonyms
2. **Use MeSH terms** - PubMed's controlled vocabulary for better recall
3. **LLM query generation** - Use LLM to generate optimized PubMed query

---

## P0-004: Loop Terminates Too Early

**File:** `src/app.py:42-45` and `src/utils/models.py`

**The Problem:**
```python
config = OrchestratorConfig(
    max_iterations=5,
    max_results_per_tool=10,
)
```

5 iterations is not enough to:
1. Search multiple variations of the query
2. Gather enough evidence for the Judge to synthesize
3. Refine queries based on initial results

**Evidence:** The user's output shows "Max Iterations Reached" with only 6 sources.

**Root Cause:** Conservative defaults to avoid API costs, but makes app useless.

**Fix Options:**
1. **Increase default to 10-15** - More iterations = better results
2. **Dynamic termination** - Stop when confidence > threshold, not iteration count
3. **Parallel query expansion** - Run more queries per iteration

---

## P0-005: No Query Understanding Layer

**Files:** `src/orchestrator.py`, `src/tools/search_handler.py`

**The Problem:**
There's no NLU (Natural Language Understanding) layer. The system:
1. Takes raw user query
2. Passes directly to search tools
3. No entity extraction
4. No intent classification
5. No query expansion

For drug repurposing, you need to extract:
- **Disease:** "Long COVID" → [Long COVID, PASC, Post-COVID syndrome, chronic COVID]
- **Drug intent:** "medications" → [drugs, treatments, therapeutics, interventions]
- **Evidence type:** "show promise" → [clinical trials, efficacy, RCT]

**Root Cause:** No preprocessing pipeline between user input and search execution.

**Fix Options:**
1. **Add entity extraction** - Use BioBERT or PubMedBERT for medical NER
2. **Add query expansion** - Use medical ontologies (UMLS, MeSH)
3. **LLM preprocessing** - Use LLM to generate search strategy before searching

---

## P0-006: ClinicalTrials.gov Results Not Filtered

**File:** `src/tools/clinicaltrials.py`

**The Problem:**
ClinicalTrials.gov returns ALL matching trials including:
- Withdrawn trials
- Terminated trials
- Not yet recruiting
- Observational studies (not interventional)

For drug repurposing, you want:
- Interventional studies
- Phase 2+ (has safety/efficacy data)
- Completed or with results

**Root Cause:** No filtering of trial metadata.

---

## Summary: Why This App Produces Garbage

```
User Query: "What medications show promise for Long COVID?"
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ NO QUERY PREPROCESSING                                       │
│ - No entity extraction                                       │
│ - No synonym expansion                                       │
│ - No medical term normalization                              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ BROKEN SEARCH LAYER                                          │
│ - PubMed: Raw query, no MeSH, gets 1 result                 │
│ - BioRxiv: Returns random papers (API doesn't support search)│
│ - ClinicalTrials: Returns all trials, no filtering          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ GARBAGE EVIDENCE                                             │
│ - 6 papers, most irrelevant                                  │
│ - "Calf muscle adaptations" (mentions COVID once)            │
│ - "Ophthalmologist work-life balance"                        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ DUMB JUDGE (Free Tier)                                       │
│ - Llama 8B can't identify drugs from garbage                 │
│ - JSON parsing fails                                         │
│ - Falls back to "Drug identification requires AI analysis"   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ LOOP HITS MAX (5 iterations)                                 │
│ - Never finds enough good evidence                           │
│ - Never synthesizes anything useful                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
    GARBAGE OUTPUT
```

---

## What Would Make This Actually Work

### Minimum Viable Fix (1-2 days)

1. **Remove BioRxiv** - It doesn't work
2. **Require API key** - Be honest that free tier is useless
3. **Add basic query preprocessing** - Strip question words, expand COVID synonyms
4. **Increase iterations to 10**

### Proper Fix (1-2 weeks)

1. **Query Understanding Layer**
   - Medical NER (BioBERT/SciBERT)
   - Query expansion with MeSH/UMLS
   - Intent classification (drug discovery vs mechanism vs safety)

2. **Optimized Search**
   - PubMed: Proper query syntax with MeSH terms
   - ClinicalTrials: Filter by phase, status, intervention type
   - Replace BioRxiv with Europe PMC (has preprints + search)

3. **Evidence Ranking**
   - Score by publication type (RCT > cohort > case report)
   - Score by journal impact factor
   - Score by recency
   - Score by citation count

4. **Proper LLM Pipeline**
   - Use GPT-4 / Claude for synthesis
   - Structured extraction of: drug, mechanism, evidence level, effect size
   - Multi-step reasoning: identify → validate → rank → synthesize

---

## The Hard Truth

Building a drug repurposing agent that works is HARD. The state of the art is:

- **Drug2Disease (IBM)** - Uses knowledge graphs + ML
- **COVID-KG (Stanford)** - Dedicated COVID knowledge graph
- **Literature Mining at scale (PubMed)** - Millions of papers, not 10

This hackathon project is fundamentally a **search wrapper with an LLM prompt**. That's not enough.

To make it useful:
1. Either scope it down (e.g., "find clinical trials for X disease")
2. Or invest serious engineering in the NLU + search + ranking pipeline
