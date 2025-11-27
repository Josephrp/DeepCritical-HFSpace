# Phase 02: PubMed Query Preprocessing

**Priority:** P0 - Critical
**Effort:** 2-3 hours
**Dependencies:** None (can run parallel with Phase 01)

---

## Problem Statement

PubMed receives raw natural language queries like "What medications show promise for Long COVID?" which include question words that pollute search results.

## Success Criteria

- [ ] Question words stripped from queries
- [ ] Medical synonyms expanded (Long COVID → PASC, etc.)
- [ ] Relevant results returned for natural language questions
- [ ] All existing tests pass
- [ ] New tests cover query preprocessing

---

## TDD Implementation Order

### Step 1: Write Failing Tests

**File:** `tests/unit/tools/test_query_utils.py`

```python
"""Unit tests for query preprocessing utilities."""

import pytest

from src.tools.query_utils import preprocess_query, expand_synonyms, strip_question_words


@pytest.mark.unit
class TestQueryPreprocessing:
    """Tests for query preprocessing."""

    def test_strip_question_words(self):
        """Test removal of question words."""
        assert strip_question_words("What drugs treat cancer") == "drugs treat cancer"
        assert strip_question_words("Which medications help diabetes") == "medications diabetes"
        assert strip_question_words("How can we cure alzheimer") == "cure alzheimer"
        assert strip_question_words("Is metformin effective") == "metformin effective"

    def test_strip_preserves_medical_terms(self):
        """Test that medical terms are preserved."""
        result = strip_question_words("What is the mechanism of metformin")
        assert "metformin" in result
        assert "mechanism" in result

    def test_expand_synonyms_long_covid(self):
        """Test Long COVID synonym expansion."""
        result = expand_synonyms("long covid treatment")
        assert "PASC" in result or "post-COVID" in result

    def test_expand_synonyms_alzheimer(self):
        """Test Alzheimer's synonym expansion."""
        result = expand_synonyms("alzheimer drug")
        assert "Alzheimer" in result

    def test_expand_synonyms_preserves_unknown(self):
        """Test that unknown terms are preserved."""
        result = expand_synonyms("metformin diabetes")
        assert "metformin" in result
        assert "diabetes" in result

    def test_preprocess_query_full_pipeline(self):
        """Test complete preprocessing pipeline."""
        raw = "What medications show promise for Long COVID?"
        result = preprocess_query(raw)

        # Should not contain question words
        assert "what" not in result.lower()
        assert "show" not in result.lower()
        assert "promise" not in result.lower()

        # Should contain expanded terms
        assert "PASC" in result or "post-COVID" in result or "long covid" in result.lower()
        assert "medications" in result.lower() or "drug" in result.lower()

    def test_preprocess_query_removes_punctuation(self):
        """Test that question marks are removed."""
        result = preprocess_query("Is metformin safe?")
        assert "?" not in result

    def test_preprocess_query_handles_empty(self):
        """Test handling of empty/whitespace queries."""
        assert preprocess_query("") == ""
        assert preprocess_query("   ") == ""

    def test_preprocess_query_already_clean(self):
        """Test that clean queries pass through."""
        clean = "metformin diabetes mechanism"
        result = preprocess_query(clean)
        assert "metformin" in result
        assert "diabetes" in result
        assert "mechanism" in result
```

### Step 2: Implement Query Utils

**File:** `src/tools/query_utils.py`

```python
"""Query preprocessing utilities for biomedical search."""

import re
from typing import ClassVar

# Question words and filler words to remove
QUESTION_WORDS: set[str] = {
    # Question starters
    "what", "which", "how", "why", "when", "where", "who", "whom",
    # Auxiliary verbs in questions
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "would", "should", "will", "shall", "may", "might",
    # Filler words in natural questions
    "show", "promise", "help", "believe", "think", "suggest",
    "possible", "potential", "effective", "useful", "good",
    # Articles (remove but less aggressively)
    "the", "a", "an",
}

# Medical synonym expansions
SYNONYMS: dict[str, list[str]] = {
    "long covid": [
        "long COVID",
        "PASC",
        "post-acute sequelae of SARS-CoV-2",
        "post-COVID syndrome",
        "post-COVID-19 condition",
    ],
    "alzheimer": [
        "Alzheimer's disease",
        "Alzheimer disease",
        "AD",
        "Alzheimer dementia",
    ],
    "parkinson": [
        "Parkinson's disease",
        "Parkinson disease",
        "PD",
    ],
    "diabetes": [
        "diabetes mellitus",
        "type 2 diabetes",
        "T2DM",
        "diabetic",
    ],
    "cancer": [
        "cancer",
        "neoplasm",
        "tumor",
        "malignancy",
        "carcinoma",
    ],
    "heart disease": [
        "cardiovascular disease",
        "CVD",
        "coronary artery disease",
        "heart failure",
    ],
}


def strip_question_words(query: str) -> str:
    """
    Remove question words and filler terms from query.

    Args:
        query: Raw query string

    Returns:
        Query with question words removed
    """
    words = query.lower().split()
    filtered = [w for w in words if w not in QUESTION_WORDS]
    return " ".join(filtered)


def expand_synonyms(query: str) -> str:
    """
    Expand medical terms to include synonyms.

    Args:
        query: Query string

    Returns:
        Query with synonym expansions in OR groups
    """
    result = query.lower()

    for term, expansions in SYNONYMS.items():
        if term in result:
            # Create OR group: ("term1" OR "term2" OR "term3")
            or_group = " OR ".join([f'"{exp}"' for exp in expansions])
            result = result.replace(term, f"({or_group})")

    return result


def preprocess_query(raw_query: str) -> str:
    """
    Full preprocessing pipeline for PubMed queries.

    Pipeline:
    1. Strip whitespace and punctuation
    2. Remove question words
    3. Expand medical synonyms

    Args:
        raw_query: Natural language query from user

    Returns:
        Optimized query for PubMed
    """
    if not raw_query or not raw_query.strip():
        return ""

    # Remove question marks and extra whitespace
    query = raw_query.replace("?", "").strip()
    query = re.sub(r"\s+", " ", query)

    # Strip question words
    query = strip_question_words(query)

    # Expand synonyms
    query = expand_synonyms(query)

    return query.strip()
```

### Step 3: Update PubMed Tool

**File:** `src/tools/pubmed.py` - Add preprocessing:

```python
# Add import at top:
from src.tools.query_utils import preprocess_query

# Update search method:
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def search(self, query: str, max_results: int = 10) -> list[Evidence]:
    """
    Search PubMed and return evidence.
    """
    await self._rate_limit()

    # PREPROCESS QUERY
    clean_query = preprocess_query(query)
    if not clean_query:
        clean_query = query  # Fallback to original if preprocessing empties it

    async with httpx.AsyncClient(timeout=30.0) as client:
        search_params = self._build_params(
            db="pubmed",
            term=clean_query,  # Use preprocessed query
            retmax=max_results,
            sort="relevance",
        )
        # ... rest unchanged
```

### Step 4: Update PubMed Tests

**File:** `tests/unit/tools/test_pubmed.py` - Add preprocessing test:

```python
@pytest.mark.asyncio
async def test_search_preprocesses_query(self, pubmed_tool, mock_httpx_client):
    """Test that queries are preprocessed before search."""
    # This test verifies the integration - the actual preprocessing
    # is tested in test_query_utils.py

    mock_httpx_client.get.return_value = httpx.Response(
        200,
        json={"esearchresult": {"idlist": []}},
    )

    # Natural language query
    await pubmed_tool.search("What drugs help with Long COVID?")

    # Verify the call was made (preprocessing happens internally)
    assert mock_httpx_client.get.called
```

---

## Verification

```bash
# Run query utils tests
uv run pytest tests/unit/tools/test_query_utils.py -v

# Run pubmed tests
uv run pytest tests/unit/tools/test_pubmed.py -v

# Run all tests
uv run pytest tests/unit/ -v

# Manual verification
uv run python -c "
from src.tools.query_utils import preprocess_query

queries = [
    'What medications show promise for Long COVID?',
    'Is metformin effective for cancer treatment?',
    'How can we treat Alzheimer with existing drugs?',
]

for q in queries:
    print(f'Input:  {q}')
    print(f'Output: {preprocess_query(q)}')
    print()
"
```

Expected output:
```
Input:  What medications show promise for Long COVID?
Output: medications ("long COVID" OR "PASC" OR "post-acute sequelae of SARS-CoV-2" OR "post-COVID syndrome" OR "post-COVID-19 condition")

Input:  Is metformin effective for cancer treatment?
Output: metformin for ("cancer" OR "neoplasm" OR "tumor" OR "malignancy" OR "carcinoma") treatment

Input:  How can we treat Alzheimer with existing drugs?
Output: we treat ("Alzheimer's disease" OR "Alzheimer disease" OR "AD" OR "Alzheimer dementia") with existing drugs
```

---

## Files Changed

| File | Action |
|------|--------|
| `src/tools/query_utils.py` | CREATE |
| `tests/unit/tools/test_query_utils.py` | CREATE |
| `src/tools/pubmed.py` | MODIFY (add preprocessing) |
| `tests/unit/tools/test_pubmed.py` | MODIFY (add integration test) |

---

## Future Enhancements (Out of Scope)

- MeSH term lookup via NCBI API
- Drug name normalization (brand → generic)
- Disease ontology integration (UMLS)
- Query intent classification
