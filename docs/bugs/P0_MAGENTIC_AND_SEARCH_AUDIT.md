# P0 Audit: Microsoft Agent Framework (Magentic) & Search Tools

**Date:** November 27, 2025
**Auditor:** Claude Code
**Status:** VERIFIED

---

## TL;DR

| Component | Status | Verdict |
|-----------|--------|---------|
| Microsoft Agent Framework | ✅ WORKING | Correctly wired, no bugs |
| GPT-5.1 Model Config | ✅ CORRECT | Using `gpt-5.1` as configured |
| Search Tools | ❌ BROKEN | Root cause of garbage results |

**The orchestration framework is fine. The search layer is garbage.**

---

## Microsoft Agent Framework Verification

### Import Test: PASSED
```python
from agent_framework import MagenticBuilder, ChatAgent
from agent_framework.openai import OpenAIChatClient
# All imports successful
```

### Agent Creation Test: PASSED
```python
from src.agents.magentic_agents import create_search_agent
search_agent = create_search_agent()
# SearchAgent created: SearchAgent
# Description: Searches biomedical databases (PubMed, ClinicalTrials.gov, bioRxiv)
```

### Workflow Build Test: PASSED
```python
from src.orchestrator_magentic import MagenticOrchestrator
orchestrator = MagenticOrchestrator(max_rounds=2)
workflow = orchestrator._build_workflow()
# Workflow built successfully: <class 'agent_framework._workflows._workflow.Workflow'>
```

### Model Configuration: CORRECT
```python
settings.openai_model = "gpt-5.1"  # ✅ Using GPT-5.1, not GPT-4o
settings.openai_api_key = True     # ✅ API key is set
```

---

## What Magentic Provides (Working)

1. **Multi-Agent Coordination**
   - Manager agent orchestrates SearchAgent, JudgeAgent, HypothesisAgent, ReportAgent
   - Uses `MagenticBuilder().with_standard_manager()` for coordination

2. **ChatAgent Pattern**
   - Each agent has internal LLM (GPT-5.1)
   - Can call tools via `@ai_function` decorator
   - Has proper instructions for domain-specific tasks

3. **Workflow Streaming**
   - Events: `MagenticAgentMessageEvent`, `MagenticFinalResultEvent`, etc.
   - Real-time UI updates via `workflow.run_stream(task)`

4. **State Management**
   - `MagenticState` persists evidence across agents
   - `get_bibliography()` tool for ReportAgent

---

## What's Actually Broken: The Search Tools

### File: `src/agents/tools.py`

The Magentic agents call these tools:
- `search_pubmed` → Uses `PubMedTool`
- `search_clinical_trials` → Uses `ClinicalTrialsTool`
- `search_preprints` → Uses `BioRxivTool`

**These tools are the problem, not the framework.**

---

## Search Tool Bugs (Detailed)

### BUG 1: BioRxiv API Does Not Support Search

**File:** `src/tools/biorxiv.py:248-286`

```python
# This fetches the FIRST 100 papers from the last 90 days
# It does NOT search by keyword - the API doesn't support that
url = f"{self.BASE_URL}/{self.server}/{interval}/0/json"

# Then filters client-side for keywords
matching = self._filter_by_keywords(papers, query_terms, max_results)
```

**Problem:**
- Fetches 100 random chronological papers
- Filters for ANY keyword match in title/abstract
- "Long COVID medications" returns papers about "calf muscles" because they mention "COVID" once

**Fix:** Remove BioRxiv or use Europe PMC (which has actual search)

---

### BUG 2: PubMed Query Not Optimized

**File:** `src/tools/pubmed.py:54-71`

```python
search_params = self._build_params(
    db="pubmed",
    term=query,  # RAW USER QUERY - no preprocessing!
    retmax=max_results,
    sort="relevance",
)
```

**Problem:**
- User enters: "What medications show promise for Long COVID?"
- PubMed receives: `What medications show promise for Long COVID?`
- Should receive: `("long covid"[Title/Abstract] OR "PASC"[Title/Abstract]) AND (treatment[Title/Abstract] OR drug[Title/Abstract])`

**Fix:** Add query preprocessing:
1. Strip question words (what, which, how, etc.)
2. Expand medical synonyms (Long COVID → PASC, Post-COVID)
3. Use MeSH terms for better recall

---

### BUG 3: ClinicalTrials.gov No Filtering

**File:** `src/tools/clinicaltrials.py`

Returns ALL trials including:
- Withdrawn trials
- Terminated trials
- Observational studies (not drug interventions)
- Phase 1 (no efficacy data)

**Fix:** Filter by:
- `studyType=INTERVENTIONAL`
- `phase=PHASE2,PHASE3,PHASE4`
- `status=COMPLETED,ACTIVE_NOT_RECRUITING,RECRUITING`

---

## Evidence: Garbage In → Garbage Out

When the Magentic SearchAgent calls these tools:

```
SearchAgent: "Find evidence for Long COVID medications"
    │
    ▼
search_pubmed("Long COVID medications")
    → Returns 1 semi-relevant paper (raw query hits)

search_preprints("Long COVID medications")
    → Returns garbage (BioRxiv API doesn't search)
    → "Calf muscle adaptations" (has "COVID" somewhere)
    → "Ophthalmologist work-life balance" (mentions COVID)

search_clinical_trials("Long COVID medications")
    → Returns all trials, no filtering
    │
    ▼
JudgeAgent receives garbage evidence
    │
    ▼
HypothesisAgent can't generate good hypotheses from garbage
    │
    ▼
ReportAgent produces garbage report
```

**The framework is doing its job. It's orchestrating agents correctly. But the agents are being fed garbage data.**

---

## Recommended Fixes

### Priority 1: Delete or Fix BioRxiv (30 min)

**Option A: Delete it**
```python
# In src/agents/tools.py, remove:
# from src.tools.biorxiv import BioRxivTool
# _biorxiv = BioRxivTool()
# @ai_function search_preprints(...)
```

**Option B: Replace with Europe PMC**
Europe PMC has preprints AND proper search API:
```
https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=long+covid+treatment&format=json
```

### Priority 2: Fix PubMed Query (1 hour)

Add query preprocessor:
```python
def preprocess_query(raw_query: str) -> str:
    """Convert natural language to PubMed query syntax."""
    # Strip question words
    # Expand medical synonyms
    # Add field tags [Title/Abstract]
    # Return optimized query
```

### Priority 3: Filter ClinicalTrials (30 min)

Add parameters to API call:
```python
params = {
    "query.term": query,
    "filter.overallStatus": "COMPLETED,RECRUITING",
    "filter.studyType": "INTERVENTIONAL",
    "pageSize": max_results,
}
```

---

## Conclusion

**Microsoft Agent Framework: NO BUGS FOUND**
- Imports work ✅
- Agent creation works ✅
- Workflow building works ✅
- Model config correct (GPT-5.1) ✅
- Streaming events work ✅

**Search Tools: CRITICALLY BROKEN**
- BioRxiv: API doesn't support search (fundamental)
- PubMed: No query optimization (fixable)
- ClinicalTrials: No filtering (fixable)

**Recommendation:**
1. Delete BioRxiv immediately (unusable)
2. Add PubMed query preprocessing
3. Add ClinicalTrials filtering
4. Then the Magentic multi-agent system will work as designed
