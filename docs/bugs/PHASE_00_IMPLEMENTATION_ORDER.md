# Phase 00: Implementation Order & Summary

**Total Effort:** 5-8 hours
**Parallelizable:** Yes (all 3 phases are independent)

---

## Executive Summary

The DeepCritical drug repurposing agent produces garbage results because the search tools are broken:

| Tool | Problem | Fix |
|------|---------|-----|
| BioRxiv | API doesn't support search | Replace with Europe PMC |
| PubMed | Raw queries, no preprocessing | Add query cleaner |
| ClinicalTrials | No filtering | Add status/type filters |

**The Microsoft Agent Framework (Magentic) is working correctly.** The orchestration layer is fine. The data layer is broken.

---

## Phase Specs

| Phase | Title | Effort | Priority | Dependencies |
|-------|-------|--------|----------|--------------|
| **01** | [Replace BioRxiv with Europe PMC](./PHASE_01_REPLACE_BIORXIV.md) | 2-3 hrs | P0 | None |
| **02** | [PubMed Query Preprocessing](./PHASE_02_PUBMED_QUERY_PREPROCESSING.md) | 2-3 hrs | P0 | None |
| **03** | [ClinicalTrials Filtering](./PHASE_03_CLINICALTRIALS_FILTERING.md) | 1-2 hrs | P1 | None |

---

## Recommended Execution Order

Since all phases are independent, they can be done in parallel by different developers.

**If doing sequentially, order by impact:**

1. **Phase 01** - BioRxiv is completely broken (returns random papers)
2. **Phase 02** - PubMed is partially broken (returns suboptimal results)
3. **Phase 03** - ClinicalTrials returns too much noise

---

## TDD Workflow (Per Phase)

```
1. Write failing tests
2. Run tests (confirm they fail)
3. Implement fix
4. Run tests (confirm they pass)
5. Run ALL tests (confirm no regressions)
6. Manual verification
7. Commit
```

---

## Verification After All Phases

After completing all 3 phases, run this integration test:

```bash
# Full system test
uv run python -c "
import asyncio
from src.tools.europepmc import EuropePMCTool
from src.tools.pubmed import PubMedTool
from src.tools.clinicaltrials import ClinicalTrialsTool

async def test_all():
    query = 'long covid treatment'

    print('=== Europe PMC (Preprints) ===')
    epmc = EuropePMCTool()
    results = await epmc.search(query, 2)
    for r in results:
        print(f'  - {r.citation.title[:60]}...')

    print()
    print('=== PubMed ===')
    pm = PubMedTool()
    results = await pm.search(query, 2)
    for r in results:
        print(f'  - {r.citation.title[:60]}...')

    print()
    print('=== ClinicalTrials.gov ===')
    ct = ClinicalTrialsTool()
    results = await ct.search(query, 2)
    for r in results:
        print(f'  - {r.citation.title[:60]}...')

asyncio.run(test_all())
"
```

**Expected:** All results should be relevant to "long covid treatment"

---

## Test Magentic Integration

After all phases are complete, test the full Magentic workflow:

```bash
# Test Magentic mode (requires OPENAI_API_KEY)
uv run python -c "
import asyncio
from src.orchestrator_magentic import MagenticOrchestrator

async def test_magentic():
    orchestrator = MagenticOrchestrator(max_rounds=3)

    print('Running Magentic workflow...')
    async for event in orchestrator.run('What drugs show promise for Long COVID?'):
        print(f'[{event.type}] {event.message[:100]}...')

asyncio.run(test_magentic())
"
```

---

## Files Changed (All Phases)

| File | Phase | Action |
|------|-------|--------|
| `src/tools/europepmc.py` | 01 | CREATE |
| `tests/unit/tools/test_europepmc.py` | 01 | CREATE |
| `src/agents/tools.py` | 01 | MODIFY |
| `src/tools/search_handler.py` | 01 | MODIFY |
| `src/tools/biorxiv.py` | 01 | DELETE |
| `tests/unit/tools/test_biorxiv.py` | 01 | DELETE |
| `src/tools/query_utils.py` | 02 | CREATE |
| `tests/unit/tools/test_query_utils.py` | 02 | CREATE |
| `src/tools/pubmed.py` | 02 | MODIFY |
| `src/tools/clinicaltrials.py` | 03 | MODIFY |
| `tests/unit/tools/test_clinicaltrials.py` | 03 | MODIFY |

---

## Success Criteria (Overall)

- [ ] All unit tests pass
- [ ] All integration tests pass (real APIs)
- [ ] Query "What drugs show promise for Long COVID?" returns relevant results from all 3 sources
- [ ] Magentic workflow produces a coherent research report
- [ ] No regressions in existing functionality

---

## Related Documentation

- [P0 Critical Bugs](./P0_CRITICAL_BUGS.md) - Root cause analysis
- [P0 Magentic Audit](./P0_MAGENTIC_AND_SEARCH_AUDIT.md) - Framework verification
- [P0 Actionable Fixes](./P0_ACTIONABLE_FIXES.md) - Fix summaries
