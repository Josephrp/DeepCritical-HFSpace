"""Unit tests for PubMed tool."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.tools.pubmed import PubMedTool

# Sample PubMed XML response for mocking
SAMPLE_PUBMED_XML = """<?xml version="1.0" ?>
<PubmedArticleSet>
    <PubmedArticle>
        <MedlineCitation>
            <PMID>12345678</PMID>
            <Article>
                <ArticleTitle>Metformin in Alzheimer's Disease: A Systematic Review</ArticleTitle>
                <Abstract>
                    <AbstractText>Metformin shows neuroprotective properties...</AbstractText>
                </Abstract>
                <AuthorList>
                    <Author>
                        <LastName>Smith</LastName>
                        <ForeName>John</ForeName>
                    </Author>
                </AuthorList>
                <Journal>
                    <JournalIssue>
                        <PubDate>
                            <Year>2024</Year>
                            <Month>01</Month>
                        </PubDate>
                    </JournalIssue>
                </Journal>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
</PubmedArticleSet>
"""


class TestPubMedTool:
    """Tests for PubMedTool."""

    @pytest.mark.asyncio
    async def test_search_returns_evidence(self, mocker):
        """PubMedTool should return Evidence objects from search."""
        # Mock the HTTP responses
        mock_search_response = MagicMock()
        mock_search_response.json.return_value = {"esearchresult": {"idlist": ["12345678"]}}
        mock_search_response.raise_for_status = MagicMock()

        mock_fetch_response = MagicMock()
        mock_fetch_response.text = SAMPLE_PUBMED_XML
        mock_fetch_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[mock_search_response, mock_fetch_response])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        # Act
        tool = PubMedTool()
        results = await tool.search("metformin alzheimer")

        # Assert
        assert len(results) == 1
        assert results[0].citation.source == "pubmed"
        assert "Metformin" in results[0].citation.title
        assert "12345678" in results[0].citation.url

    @pytest.mark.asyncio
    async def test_search_empty_results(self, mocker):
        """PubMedTool should return empty list when no results."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"esearchresult": {"idlist": []}}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        tool = PubMedTool()
        results = await tool.search("xyznonexistentquery123")

        assert results == []

    def test_parse_pubmed_xml(self):
        """PubMedTool should correctly parse XML."""
        tool = PubMedTool()
        results = tool._parse_pubmed_xml(SAMPLE_PUBMED_XML)

        assert len(results) == 1
        assert results[0].citation.source == "pubmed"
        assert "Smith John" in results[0].citation.authors

    @pytest.mark.asyncio
    async def test_search_preprocesses_query(self, mocker):
        """Test that queries are preprocessed before search."""
        mock_search_response = MagicMock()
        mock_search_response.json.return_value = {"esearchresult": {"idlist": []}}
        mock_search_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_search_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        tool = PubMedTool()
        await tool.search("What drugs help with Long COVID?")

        # Verify call args
        call_args = mock_client.get.call_args
        params = call_args[1]["params"]
        term = params["term"]

        # "what" and "help" should be stripped
        assert "what" not in term.lower()
        assert "help" not in term.lower()
        # "long covid" should be expanded
        assert "PASC" in term or "post-COVID" in term

    @pytest.mark.asyncio
    async def test_rate_limiting_enforced(self, mocker):
        """PubMedTool should enforce rate limiting between requests."""
        from unittest.mock import patch

        mock_search_response = MagicMock()
        mock_search_response.json.return_value = {"esearchresult": {"idlist": []}}
        mock_search_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_search_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        tool = PubMedTool()
        # Reset last request time to ensure rate limit is triggered
        tool._last_request_time = 0.0

        # Mock time to control elapsed time
        with patch("asyncio.get_running_loop") as mock_loop:
            loop_mock = MagicMock()
            loop_mock.time.side_effect = [0.0, 0.1]  # Only 0.1s elapsed, need 0.34s
            mock_loop.return_value = loop_mock

            # Mock sleep to verify it's called
            with patch("asyncio.sleep") as mock_sleep:
                await tool.search("test query")
                # Should sleep for at least (0.34 - 0.1) = 0.24 seconds
                mock_sleep.assert_called_once()
                call_arg = mock_sleep.call_args[0][0]
                assert call_arg >= 0.24

    @pytest.mark.asyncio
    async def test_api_key_included_in_params(self, mocker):
        """PubMedTool should include API key in params when provided."""
        mock_search_response = MagicMock()
        mock_search_response.json.return_value = {"esearchresult": {"idlist": []}}
        mock_search_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_search_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        # Test with API key
        tool = PubMedTool(api_key="test-api-key-123")
        await tool.search("test query")

        # Verify API key was included in params
        call_args = mock_client.get.call_args
        params = call_args[1]["params"]
        assert "api_key" in params
        assert params["api_key"] == "test-api-key-123"

        # Test without API key
        tool_no_key = PubMedTool(api_key=None)
        mock_client.get.reset_mock()
        await tool_no_key.search("test query")

        call_args = mock_client.get.call_args
        params = call_args[1]["params"]
        assert "api_key" not in params

    @pytest.mark.asyncio
    async def test_handles_429_rate_limit(self, mocker):
        """PubMedTool should raise RateLimitError on 429 response."""
        import httpx

        from src.utils.exceptions import RateLimitError

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Rate limit", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        tool = PubMedTool()
        with pytest.raises(RateLimitError, match="rate limit exceeded"):
            await tool.search("test query")

    @pytest.mark.asyncio
    async def test_handles_500_server_error(self, mocker):
        """PubMedTool should raise SearchError on 500 response."""
        import httpx

        from src.utils.exceptions import SearchError

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        tool = PubMedTool()
        with pytest.raises(SearchError, match="PubMed search failed"):
            await tool.search("test query")

    @pytest.mark.asyncio
    async def test_handles_network_timeout(self, mocker):
        """PubMedTool should handle network timeout errors."""
        import httpx

        from src.utils.exceptions import SearchError

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        tool = PubMedTool()
        # Should be retried by tenacity, but eventually raise SearchError
        with pytest.raises(SearchError):
            await tool.search("test query")

    def test_parse_empty_xml(self):
        """PubMedTool should handle empty XML gracefully."""
        tool = PubMedTool()
        empty_xml = '<?xml version="1.0" ?><PubmedArticleSet></PubmedArticleSet>'
        results = tool._parse_pubmed_xml(empty_xml)
        assert results == []

    def test_parse_malformed_xml(self):
        """PubMedTool should raise SearchError on malformed XML."""
        from src.utils.exceptions import SearchError

        tool = PubMedTool()
        malformed_xml = "<not>valid</xml>"
        with pytest.raises(SearchError, match="Failed to parse PubMed XML"):
            tool._parse_pubmed_xml(malformed_xml)

    def test_parse_article_without_abstract(self):
        """PubMedTool should skip articles without abstracts."""
        tool = PubMedTool()
        xml_no_abstract = """<?xml version="1.0" ?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>12345678</PMID>
                    <Article>
                        <ArticleTitle>Test Article</ArticleTitle>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        results = tool._parse_pubmed_xml(xml_no_abstract)
        # Should return empty list because article has no abstract
        assert results == []

    def test_parse_article_without_title(self):
        """PubMedTool should skip articles without titles."""
        tool = PubMedTool()
        xml_no_title = """<?xml version="1.0" ?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>12345678</PMID>
                    <Article>
                        <Abstract>
                            <AbstractText>Some abstract text</AbstractText>
                        </Abstract>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        results = tool._parse_pubmed_xml(xml_no_title)
        # Should return empty list because article has no title
        assert results == []
