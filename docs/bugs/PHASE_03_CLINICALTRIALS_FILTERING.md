# Phase 03: ClinicalTrials.gov Filtering

**Priority:** P1 - High
**Effort:** 1-2 hours
**Dependencies:** None (can run parallel with Phase 01 & 02)

---

## Problem Statement

ClinicalTrials.gov returns ALL matching trials including:
- Withdrawn/Terminated trials (no useful data)
- Observational studies (not drug interventions)
- Phase 1 trials (safety only, no efficacy)

For drug repurposing, we need interventional studies with efficacy data.

## Success Criteria

- [ ] Only interventional studies returned
- [ ] Withdrawn/terminated trials filtered out
- [ ] Phase information included in results
- [ ] All existing tests pass
- [ ] New tests cover filtering

---

## TDD Implementation Order

### Step 1: Write Failing Tests

**File:** `tests/unit/tools/test_clinicaltrials.py` - Add filter tests:

```python
"""Unit tests for ClinicalTrials.gov tool."""

import pytest
from unittest.mock import patch, MagicMock

from src.tools.clinicaltrials import ClinicalTrialsTool
from src.utils.models import Evidence


@pytest.mark.unit
class TestClinicalTrialsTool:
    """Tests for ClinicalTrialsTool."""

    @pytest.fixture
    def tool(self):
        return ClinicalTrialsTool()

    def test_tool_name(self, tool):
        assert tool.name == "clinicaltrials"

    @pytest.mark.asyncio
    async def test_search_uses_filters(self, tool):
        """Test that search applies status and type filters."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"studies": []}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            await tool.search("test query", max_results=5)

            # Verify filters were applied
            call_args = mock_get.call_args
            params = call_args.kwargs.get("params", call_args[1].get("params", {}))

            # Should filter for active/completed studies
            assert "filter.overallStatus" in params
            assert "COMPLETED" in params["filter.overallStatus"]
            assert "RECRUITING" in params["filter.overallStatus"]

            # Should filter for interventional studies
            assert "filter.studyType" in params
            assert "INTERVENTIONAL" in params["filter.studyType"]

    @pytest.mark.asyncio
    async def test_search_returns_evidence(self, tool):
        """Test that search returns Evidence objects."""
        mock_study = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678",
                    "briefTitle": "Metformin for Long COVID Treatment",
                },
                "statusModule": {
                    "overallStatus": "COMPLETED",
                    "startDateStruct": {"date": "2023-01-01"},
                },
                "descriptionModule": {
                    "briefSummary": "A study examining metformin for Long COVID symptoms.",
                },
                "designModule": {
                    "phases": ["PHASE2", "PHASE3"],
                },
                "conditionsModule": {
                    "conditions": ["Long COVID", "PASC"],
                },
                "armsInterventionsModule": {
                    "interventions": [{"name": "Metformin"}],
                },
            }
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {"studies": [mock_study]}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            results = await tool.search("long covid metformin", max_results=5)

            assert len(results) == 1
            assert isinstance(results[0], Evidence)
            assert "Metformin" in results[0].citation.title
            assert "PHASE2" in results[0].content or "Phase" in results[0].content

    @pytest.mark.asyncio
    async def test_search_includes_phase_info(self, tool):
        """Test that phase information is included in content."""
        mock_study = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678",
                    "briefTitle": "Test Study",
                },
                "statusModule": {
                    "overallStatus": "RECRUITING",
                    "startDateStruct": {"date": "2024-01-01"},
                },
                "descriptionModule": {
                    "briefSummary": "Test summary.",
                },
                "designModule": {
                    "phases": ["PHASE3"],
                },
                "conditionsModule": {"conditions": ["Test"]},
                "armsInterventionsModule": {"interventions": []},
            }
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {"studies": [mock_study]}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            results = await tool.search("test", max_results=5)

            # Phase should be in content
            assert "PHASE3" in results[0].content or "Phase 3" in results[0].content

    @pytest.mark.asyncio
    async def test_search_empty_results(self, tool):
        """Test handling of empty results."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"studies": []}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            results = await tool.search("nonexistent xyz 12345", max_results=5)
            assert results == []


@pytest.mark.integration
class TestClinicalTrialsIntegration:
    """Integration tests with real API."""

    @pytest.mark.asyncio
    async def test_real_api_returns_interventional(self):
        """Test that real API returns interventional studies."""
        tool = ClinicalTrialsTool()
        results = await tool.search("long covid treatment", max_results=3)

        # Should get results
        assert len(results) > 0

        # Results should mention interventions or treatments
        all_content = " ".join([r.content.lower() for r in results])
        has_intervention = (
            "intervention" in all_content
            or "treatment" in all_content
            or "drug" in all_content
            or "phase" in all_content
        )
        assert has_intervention
```

### Step 2: Update ClinicalTrials Tool

**File:** `src/tools/clinicaltrials.py` - Add filters:

```python
"""ClinicalTrials.gov search tool using API v2."""

import asyncio
from typing import Any, ClassVar

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.exceptions import SearchError
from src.utils.models import Citation, Evidence


class ClinicalTrialsTool:
    """Search tool for ClinicalTrials.gov.

    Note: Uses `requests` library instead of `httpx` because ClinicalTrials.gov's
    WAF blocks httpx's TLS fingerprint. The `requests` library is not blocked.
    See: https://clinicaltrials.gov/data-api/api
    """

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    # Fields to retrieve
    FIELDS: ClassVar[list[str]] = [
        "NCTId",
        "BriefTitle",
        "Phase",
        "OverallStatus",
        "Condition",
        "InterventionName",
        "StartDate",
        "BriefSummary",
    ]

    # Status filter: Only active/completed studies with potential data
    STATUS_FILTER = "COMPLETED|ACTIVE_NOT_RECRUITING|RECRUITING|ENROLLING_BY_INVITATION"

    # Study type filter: Only interventional (drug/treatment studies)
    STUDY_TYPE_FILTER = "INTERVENTIONAL"

    @property
    def name(self) -> str:
        return "clinicaltrials"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def search(self, query: str, max_results: int = 10) -> list[Evidence]:
        """Search ClinicalTrials.gov for interventional studies.

        Args:
            query: Search query (e.g., "metformin alzheimer")
            max_results: Maximum results to return (max 100)

        Returns:
            List of Evidence objects from clinical trials
        """
        params: dict[str, str | int] = {
            "query.term": query,
            "pageSize": min(max_results, 100),
            "fields": "|".join(self.FIELDS),
            # FILTERS - Only interventional, active/completed studies
            "filter.overallStatus": self.STATUS_FILTER,
            "filter.studyType": self.STUDY_TYPE_FILTER,
        }

        try:
            # Run blocking requests.get in a separate thread for async compatibility
            response = await asyncio.to_thread(
                requests.get,
                self.BASE_URL,
                params=params,
                headers={"User-Agent": "DeepCritical-Research-Agent/1.0"},
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            studies = data.get("studies", [])
            return [self._study_to_evidence(study) for study in studies[:max_results]]

        except requests.HTTPError as e:
            raise SearchError(f"ClinicalTrials.gov API error: {e}") from e
        except requests.RequestException as e:
            raise SearchError(f"ClinicalTrials.gov request failed: {e}") from e

    def _study_to_evidence(self, study: dict[str, Any]) -> Evidence:
        """Convert a clinical trial study to Evidence."""
        # Navigate nested structure
        protocol = study.get("protocolSection", {})
        id_module = protocol.get("identificationModule", {})
        status_module = protocol.get("statusModule", {})
        desc_module = protocol.get("descriptionModule", {})
        design_module = protocol.get("designModule", {})
        conditions_module = protocol.get("conditionsModule", {})
        arms_module = protocol.get("armsInterventionsModule", {})

        nct_id = id_module.get("nctId", "Unknown")
        title = id_module.get("briefTitle", "Untitled Study")
        status = status_module.get("overallStatus", "Unknown")
        start_date = status_module.get("startDateStruct", {}).get("date", "Unknown")

        # Get phase (might be a list)
        phases = design_module.get("phases", [])
        phase = phases[0] if phases else "Not Applicable"

        # Get conditions
        conditions = conditions_module.get("conditions", [])
        conditions_str = ", ".join(conditions[:3]) if conditions else "Unknown"

        # Get interventions
        interventions = arms_module.get("interventions", [])
        intervention_names = [i.get("name", "") for i in interventions[:3]]
        interventions_str = ", ".join(intervention_names) if intervention_names else "Unknown"

        # Get summary
        summary = desc_module.get("briefSummary", "No summary available.")

        # Build content with key trial info
        content = (
            f"{summary[:500]}... "
            f"Trial Phase: {phase}. "
            f"Status: {status}. "
            f"Conditions: {conditions_str}. "
            f"Interventions: {interventions_str}."
        )

        return Evidence(
            content=content[:2000],
            citation=Citation(
                source="clinicaltrials",
                title=title[:500],
                url=f"https://clinicaltrials.gov/study/{nct_id}",
                date=start_date,
                authors=[],  # Trials don't have traditional authors
            ),
            relevance=0.85,  # Trials are highly relevant for repurposing
        )
```

---

## Verification

```bash
# Run clinicaltrials tests
uv run pytest tests/unit/tools/test_clinicaltrials.py -v

# Run integration test (real API)
uv run pytest tests/unit/tools/test_clinicaltrials.py::TestClinicalTrialsIntegration -v

# Run all tests
uv run pytest tests/unit/ -v

# Manual verification
uv run python -c "
import asyncio
from src.tools.clinicaltrials import ClinicalTrialsTool

tool = ClinicalTrialsTool()
results = asyncio.run(tool.search('long covid treatment', 3))

for r in results:
    print(f'Title: {r.citation.title}')
    print(f'Content: {r.content[:200]}...')
    print()
"
```

---

## Files Changed

| File | Action |
|------|--------|
| `src/tools/clinicaltrials.py` | MODIFY (add filters) |
| `tests/unit/tools/test_clinicaltrials.py` | MODIFY (add filter tests) |

---

## API Filter Reference

ClinicalTrials.gov API v2 supports these filters:

| Parameter | Values | Purpose |
|-----------|--------|---------|
| `filter.overallStatus` | COMPLETED, RECRUITING, etc. | Trial status |
| `filter.studyType` | INTERVENTIONAL, OBSERVATIONAL | Study design |
| `filter.phase` | PHASE1, PHASE2, PHASE3, PHASE4 | Trial phase |
| `filter.geo` | Country codes | Geographic filter |

See: https://clinicaltrials.gov/data-api/api
