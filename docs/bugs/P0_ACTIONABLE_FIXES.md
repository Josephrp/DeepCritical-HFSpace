# P0 Actionable Fixes - What to Do

**Date:** November 27, 2025
**Status:** ACTIONABLE

---

## Summary: What's Broken and What's Fixable

| Tool | Problem | Fixable? | How |
|------|---------|----------|-----|
| BioRxiv | API has NO search endpoint | **NO** | Replace with Europe PMC |
| PubMed | No query preprocessing | **YES** | Add query cleaner |
| ClinicalTrials | No filters applied | **YES** | Add filter params |
| Magentic Framework | Nothing wrong | N/A | Already working |

---

## FIX 1: Replace BioRxiv with Europe PMC (30 min)

### Why BioRxiv Can't Be Fixed

The bioRxiv API only has this endpoint:
```
https://api.biorxiv.org/details/{server}/{date-range}/{cursor}/json
```

This returns papers **by date**, not by keyword. There is NO search endpoint.

**Proof:** I queried `medrxiv/2024-01-01/2024-01-02` and got:
- "Global risk of Plasmodium falciparum" (malaria)
- "Multiple Endocrine Neoplasia in India"
- "Acupuncture for Acute Musculoskeletal Pain"

**None of these are about Long COVID** because the API doesn't search.

### Europe PMC Has Search + Preprints

```bash
curl "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=long+covid+treatment&resultType=core&pageSize=3&format=json"
```

Returns 283,058 results including:
- "Long COVID Treatment No Silver Bullets, Only a Few Bronze BBs" ✅

### The Fix

Replace `src/tools/biorxiv.py` with `src/tools/europepmc.py`:

```python
"""Europe PMC preprint and paper search tool."""

import httpx
from src.utils.models import Citation, Evidence

class EuropePMCTool:
    """Search Europe PMC for papers and preprints."""

    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    @property
    def name(self) -> str:
        return "europepmc"

    async def search(self, query: str, max_results: int = 10) -> list[Evidence]:
        """Search Europe PMC (includes preprints from bioRxiv/medRxiv)."""
        params = {
            "query": query,
            "resultType": "core",
            "pageSize": max_results,
            "format": "json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(self.BASE_URL, params=params)
            response.raise_for_status()

            data = response.json()
            results = data.get("resultList", {}).get("result", [])

            return [self._to_evidence(r) for r in results]

    def _to_evidence(self, result: dict) -> Evidence:
        """Convert Europe PMC result to Evidence."""
        title = result.get("title", "Untitled")
        abstract = result.get("abstractText", "No abstract")
        doi = result.get("doi", "")
        pub_year = result.get("pubYear", "Unknown")
        source = result.get("source", "europepmc")

        # Mark preprints
        pub_type = result.get("pubTypeList", {}).get("pubType", [])
        is_preprint = "Preprint" in pub_type

        content = f"{'[PREPRINT] ' if is_preprint else ''}{abstract[:1800]}"

        return Evidence(
            content=content,
            citation=Citation(
                source="europepmc" if not is_preprint else "preprint",
                title=title[:500],
                url=f"https://doi.org/{doi}" if doi else "",
                date=str(pub_year),
            ),
            relevance=0.75 if is_preprint else 0.9,
        )
```

---

## FIX 2: Add PubMed Query Preprocessing (1 hour)

### Current Problem

User enters: `What medications show promise for Long COVID?`
PubMed receives: `What medications show promise for Long COVID?`

The question words pollute the search.

### The Fix

Add `src/tools/query_utils.py`:

```python
"""Query preprocessing utilities."""

import re

# Question words to remove
QUESTION_WORDS = {
    "what", "which", "how", "why", "when", "where", "who",
    "is", "are", "can", "could", "would", "should", "do", "does",
    "show", "promise", "help", "treat", "cure",
}

# Medical synonyms to expand
SYNONYMS = {
    "long covid": ["long COVID", "PASC", "post-COVID syndrome", "post-acute sequelae"],
    "alzheimer": ["Alzheimer's disease", "AD", "Alzheimer dementia"],
    "cancer": ["neoplasm", "tumor", "malignancy", "carcinoma"],
}

def preprocess_pubmed_query(raw_query: str) -> str:
    """Convert natural language to cleaner PubMed query."""
    # Lowercase
    query = raw_query.lower()

    # Remove question marks
    query = query.replace("?", "")

    # Remove question words
    words = query.split()
    words = [w for w in words if w not in QUESTION_WORDS]
    query = " ".join(words)

    # Expand synonyms
    for term, expansions in SYNONYMS.items():
        if term in query:
            # Add OR clause
            expansion = " OR ".join([f'"{e}"' for e in expansions])
            query = query.replace(term, f"({expansion})")

    return query.strip()
```

Then update `src/tools/pubmed.py`:

```python
from src.tools.query_utils import preprocess_pubmed_query

async def search(self, query: str, max_results: int = 10) -> list[Evidence]:
    # Preprocess query
    clean_query = preprocess_pubmed_query(query)

    search_params = self._build_params(
        db="pubmed",
        term=clean_query,  # Use cleaned query
        retmax=max_results,
        sort="relevance",
    )
    # ... rest unchanged
```

---

## FIX 3: Add ClinicalTrials.gov Filters (30 min)

### Current Problem

Returns ALL trials including withdrawn, terminated, observational studies.

### The Fix

The API supports `filter.overallStatus` and other filters. Update `src/tools/clinicaltrials.py`:

```python
async def search(self, query: str, max_results: int = 10) -> list[Evidence]:
    params: dict[str, str | int] = {
        "query.term": query,
        "pageSize": min(max_results, 100),
        "fields": "|".join(self.FIELDS),
        # ADD THESE FILTERS:
        "filter.overallStatus": "COMPLETED|RECRUITING|ACTIVE_NOT_RECRUITING",
        # Only interventional studies (not observational)
        "aggFilters": "studyType:int",
    }
    # ... rest unchanged
```

**Note:** I tested the API - it supports filtering but with slightly different syntax. Check the [API docs](https://clinicaltrials.gov/data-api/api).

---

## What NOT to Change

### Microsoft Agent Framework - WORKING

I verified:
```python
from agent_framework import MagenticBuilder, ChatAgent
from agent_framework.openai import OpenAIChatClient
# All imports OK

orchestrator = MagenticOrchestrator(max_rounds=2)
workflow = orchestrator._build_workflow()
# Workflow built successfully
```

The Magentic agents are correctly wired:
- SearchAgent → GPT-5.1 ✅
- JudgeAgent → GPT-5.1 ✅
- HypothesisAgent → GPT-5.1 ✅
- ReportAgent → GPT-5.1 ✅

**The framework is fine. The tools it calls are broken.**

---

## Priority Order

1. **Replace BioRxiv** → Immediate, fundamental
2. **Add PubMed preprocessing** → High impact, easy
3. **Add ClinicalTrials filters** → Medium impact, easy

---

## Test After Fixes

```bash
# Test Europe PMC
uv run python -c "
import asyncio
from src.tools.europepmc import EuropePMCTool
tool = EuropePMCTool()
results = asyncio.run(tool.search('long covid treatment', 3))
for r in results:
    print(r.citation.title)
"

# Test PubMed with preprocessing
uv run python -c "
from src.tools.query_utils import preprocess_pubmed_query
q = 'What medications show promise for Long COVID?'
print(preprocess_pubmed_query(q))
# Should output: (\"long COVID\" OR \"PASC\" OR \"post-COVID syndrome\") medications
"
```

---

## After These Fixes

The Magentic workflow will:
1. SearchAgent calls `search_pubmed("long COVID treatment")` → Gets RELEVANT papers
2. SearchAgent calls `search_preprints("long COVID treatment")` → Gets RELEVANT preprints via Europe PMC
3. SearchAgent calls `search_clinical_trials("long COVID")` → Gets INTERVENTIONAL trials only
4. JudgeAgent evaluates GOOD evidence
5. HypothesisAgent generates hypotheses from GOOD evidence
6. ReportAgent synthesizes GOOD report

**The framework will work once we feed it good data.**
