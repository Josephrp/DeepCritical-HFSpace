# Phase 01: Replace BioRxiv with Europe PMC

**Priority:** P0 - Critical
**Effort:** 2-3 hours
**Dependencies:** None

---

## Problem Statement

The BioRxiv API does not support keyword search. It only returns papers by date range, resulting in completely irrelevant results for any query.

## Success Criteria

- [ ] `search_preprints("long covid treatment")` returns papers actually about Long COVID
- [ ] All existing tests pass
- [ ] New tests cover Europe PMC integration

---

## TDD Implementation Order

### Step 1: Write Failing Test

**File:** `tests/unit/tools/test_europepmc.py`

```python
"""Unit tests for Europe PMC tool."""

import pytest
from unittest.mock import AsyncMock, patch

from src.tools.europepmc import EuropePMCTool
from src.utils.models import Evidence


@pytest.mark.unit
class TestEuropePMCTool:
    """Tests for EuropePMCTool."""

    @pytest.fixture
    def tool(self):
        return EuropePMCTool()

    def test_tool_name(self, tool):
        assert tool.name == "europepmc"

    @pytest.mark.asyncio
    async def test_search_returns_evidence(self, tool):
        """Test that search returns Evidence objects."""
        mock_response = {
            "resultList": {
                "result": [
                    {
                        "id": "12345",
                        "title": "Long COVID Treatment Study",
                        "abstractText": "This study examines treatments for Long COVID.",
                        "doi": "10.1234/test",
                        "pubYear": "2024",
                        "source": "MED",
                        "pubTypeList": {"pubType": ["research-article"]},
                    }
                ]
            }
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.get.return_value.json.return_value = mock_response
            mock_instance.get.return_value.raise_for_status = lambda: None

            results = await tool.search("long covid treatment", max_results=5)

            assert len(results) == 1
            assert isinstance(results[0], Evidence)
            assert "Long COVID Treatment Study" in results[0].citation.title

    @pytest.mark.asyncio
    async def test_search_marks_preprints(self, tool):
        """Test that preprints are marked correctly."""
        mock_response = {
            "resultList": {
                "result": [
                    {
                        "id": "PPR12345",
                        "title": "Preprint Study",
                        "abstractText": "Abstract text",
                        "doi": "10.1234/preprint",
                        "pubYear": "2024",
                        "source": "PPR",
                        "pubTypeList": {"pubType": ["Preprint"]},
                    }
                ]
            }
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.get.return_value.json.return_value = mock_response
            mock_instance.get.return_value.raise_for_status = lambda: None

            results = await tool.search("test", max_results=5)

            assert "[PREPRINT]" in results[0].content
            assert results[0].citation.source == "preprint"

    @pytest.mark.asyncio
    async def test_search_empty_results(self, tool):
        """Test handling of empty results."""
        mock_response = {"resultList": {"result": []}}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.get.return_value.json.return_value = mock_response
            mock_instance.get.return_value.raise_for_status = lambda: None

            results = await tool.search("nonexistent query xyz", max_results=5)

            assert results == []


@pytest.mark.integration
class TestEuropePMCIntegration:
    """Integration tests with real API."""

    @pytest.mark.asyncio
    async def test_real_api_call(self):
        """Test actual API returns relevant results."""
        tool = EuropePMCTool()
        results = await tool.search("long covid treatment", max_results=3)

        assert len(results) > 0
        # At least one result should mention COVID
        titles = " ".join([r.citation.title.lower() for r in results])
        assert "covid" in titles or "sars" in titles
```

### Step 2: Implement Europe PMC Tool

**File:** `src/tools/europepmc.py`

```python
"""Europe PMC search tool - replaces BioRxiv."""

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.exceptions import SearchError
from src.utils.models import Citation, Evidence


class EuropePMCTool:
    """
    Search Europe PMC for papers and preprints.

    Europe PMC indexes:
    - PubMed/MEDLINE articles
    - PMC full-text articles
    - Preprints from bioRxiv, medRxiv, ChemRxiv, etc.
    - Patents and clinical guidelines

    API Docs: https://europepmc.org/RestfulWebService
    """

    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    @property
    def name(self) -> str:
        return "europepmc"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def search(self, query: str, max_results: int = 10) -> list[Evidence]:
        """
        Search Europe PMC for papers matching query.

        Args:
            query: Search keywords
            max_results: Maximum results to return

        Returns:
            List of Evidence objects
        """
        params = {
            "query": query,
            "resultType": "core",
            "pageSize": min(max_results, 100),
            "format": "json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()

                data = response.json()
                results = data.get("resultList", {}).get("result", [])

                return [self._to_evidence(r) for r in results[:max_results]]

            except httpx.HTTPStatusError as e:
                raise SearchError(f"Europe PMC API error: {e}") from e
            except httpx.RequestError as e:
                raise SearchError(f"Europe PMC connection failed: {e}") from e

    def _to_evidence(self, result: dict[str, Any]) -> Evidence:
        """Convert Europe PMC result to Evidence."""
        title = result.get("title", "Untitled")
        abstract = result.get("abstractText", "No abstract available.")
        doi = result.get("doi", "")
        pub_year = result.get("pubYear", "Unknown")

        # Get authors
        author_list = result.get("authorList", {}).get("author", [])
        authors = [a.get("fullName", "") for a in author_list[:5] if a.get("fullName")]

        # Check if preprint
        pub_types = result.get("pubTypeList", {}).get("pubType", [])
        is_preprint = "Preprint" in pub_types
        source_db = result.get("source", "europepmc")

        # Build content
        preprint_marker = "[PREPRINT - Not peer-reviewed] " if is_preprint else ""
        content = f"{preprint_marker}{abstract[:1800]}"

        # Build URL
        if doi:
            url = f"https://doi.org/{doi}"
        elif result.get("pmid"):
            url = f"https://pubmed.ncbi.nlm.nih.gov/{result['pmid']}/"
        else:
            url = f"https://europepmc.org/article/{source_db}/{result.get('id', '')}"

        return Evidence(
            content=content[:2000],
            citation=Citation(
                source="preprint" if is_preprint else "europepmc",
                title=title[:500],
                url=url,
                date=str(pub_year),
                authors=authors,
            ),
            relevance=0.75 if is_preprint else 0.9,
        )
```

### Step 3: Update Magentic Tools

**File:** `src/agents/tools.py` - Replace biorxiv import:

```python
# REMOVE:
# from src.tools.biorxiv import BioRxivTool
# _biorxiv = BioRxivTool()

# ADD:
from src.tools.europepmc import EuropePMCTool
_europepmc = EuropePMCTool()

# UPDATE search_preprints function:
@ai_function
async def search_preprints(query: str, max_results: int = 10) -> str:
    """Search Europe PMC for preprints and papers.

    Use this tool to find the latest research including preprints
    from bioRxiv, medRxiv, and peer-reviewed papers.

    Args:
        query: Search terms (e.g., "long covid treatment")
        max_results: Maximum results to return (default 10)

    Returns:
        Formatted list of papers with abstracts and links
    """
    state = get_magentic_state()

    results = await _europepmc.search(query, max_results)
    if not results:
        return f"No papers found for: {query}"

    new_count = state.add_evidence(results)

    output = [f"Found {len(results)} papers ({new_count} new stored):\n"]
    for i, r in enumerate(results[:max_results], 1):
        title = r.citation.title
        date = r.citation.date
        source = r.citation.source
        content_clean = r.content[:300].replace("\n", " ")
        url = r.citation.url

        output.append(f"{i}. **{title}**")
        output.append(f"   Source: {source} | Date: {date}")
        output.append(f"   {content_clean}...")
        output.append(f"   URL: {url}\n")

    return "\n".join(output)
```

### Step 4: Update Search Handler (Simple Mode)

**File:** `src/tools/search_handler.py` - Update imports:

```python
# REMOVE:
# from src.tools.biorxiv import BioRxivTool

# ADD:
from src.tools.europepmc import EuropePMCTool
```

### Step 5: Delete Old BioRxiv Tests

```bash
# After all new tests pass:
rm tests/unit/tools/test_biorxiv.py
```

---

## Verification

```bash
# Run new tests
uv run pytest tests/unit/tools/test_europepmc.py -v

# Run integration test (real API)
uv run pytest tests/unit/tools/test_europepmc.py::TestEuropePMCIntegration -v

# Run all tests to ensure no regressions
uv run pytest tests/unit/ -v

# Manual verification
uv run python -c "
import asyncio
from src.tools.europepmc import EuropePMCTool
tool = EuropePMCTool()
results = asyncio.run(tool.search('long covid treatment', 3))
for r in results:
    print(f'- {r.citation.title}')
"
```

---

## Files Changed

| File | Action |
|------|--------|
| `src/tools/europepmc.py` | CREATE |
| `tests/unit/tools/test_europepmc.py` | CREATE |
| `src/agents/tools.py` | MODIFY (replace biorxiv import) |
| `src/tools/search_handler.py` | MODIFY (replace biorxiv import) |
| `src/tools/biorxiv.py` | DELETE (after verification) |
| `tests/unit/tools/test_biorxiv.py` | DELETE (after verification) |

---

## Rollback Plan

If issues arise:
1. Revert `src/agents/tools.py` to use BioRxivTool
2. Revert `src/tools/search_handler.py`
3. Keep `europepmc.py` for future use
