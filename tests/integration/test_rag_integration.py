"""Integration tests for RAG integration.

These tests require OPENAI_API_KEY and may make real API calls.
Marked with @pytest.mark.integration to skip in unit test runs.
"""

import pytest

from src.services.llamaindex_rag import get_rag_service
from src.tools.rag_tool import create_rag_tool
from src.tools.search_handler import SearchHandler
from src.tools.tool_executor import execute_agent_task
from src.utils.config import settings
from src.utils.models import AgentTask, Citation, Evidence


@pytest.mark.integration
class TestRAGServiceIntegration:
    """Integration tests for LlamaIndexRAGService."""

    @pytest.mark.asyncio
    async def test_rag_service_ingest_and_retrieve(self):
        """RAG service should ingest and retrieve evidence."""
        if not settings.openai_api_key:
            pytest.skip("OPENAI_API_KEY required for RAG integration tests")

        # Create RAG service
        rag_service = get_rag_service(collection_name="test_integration")

        # Create sample evidence
        evidence_list = [
            Evidence(
                content="Metformin is a first-line treatment for type 2 diabetes. It works by reducing glucose production in the liver and improving insulin sensitivity.",
                citation=Citation(
                    source="pubmed",
                    title="Metformin Mechanism of Action",
                    url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                    date="2024-01-15",
                    authors=["Smith J", "Johnson M"],
                ),
                relevance=0.9,
            ),
            Evidence(
                content="Recent studies suggest metformin may have neuroprotective effects in Alzheimer's disease models.",
                citation=Citation(
                    source="pubmed",
                    title="Metformin and Neuroprotection",
                    url="https://pubmed.ncbi.nlm.nih.gov/12345679/",
                    date="2024-02-20",
                    authors=["Brown K", "Davis L"],
                ),
                relevance=0.85,
            ),
        ]

        # Ingest evidence
        rag_service.ingest_evidence(evidence_list)

        # Retrieve evidence
        results = rag_service.retrieve("metformin diabetes", top_k=2)

        # Assert
        assert len(results) > 0
        assert any("metformin" in r["text"].lower() for r in results)
        assert all("text" in r for r in results)
        assert all("metadata" in r for r in results)

        # Cleanup
        rag_service.clear_collection()

    @pytest.mark.asyncio
    async def test_rag_service_query(self):
        """RAG service should synthesize responses from ingested evidence."""
        if not settings.openai_api_key:
            pytest.skip("OPENAI_API_KEY required for RAG integration tests")

        rag_service = get_rag_service(collection_name="test_query")

        # Ingest evidence
        evidence_list = [
            Evidence(
                content="Python is a high-level programming language known for its simplicity and readability.",
                citation=Citation(
                    source="pubmed",
                    title="Python Programming",
                    url="https://example.com/python",
                    date="2024",
                    authors=["Author"],
                ),
            )
        ]
        rag_service.ingest_evidence(evidence_list)

        # Query
        response = rag_service.query("What is Python?", top_k=1)

        assert isinstance(response, str)
        assert len(response) > 0
        assert "python" in response.lower()

        # Cleanup
        rag_service.clear_collection()


@pytest.mark.integration
class TestRAGToolIntegration:
    """Integration tests for RAGTool."""

    @pytest.mark.asyncio
    async def test_rag_tool_search(self):
        """RAGTool should search RAG service and return Evidence objects."""
        if not settings.openai_api_key:
            pytest.skip("OPENAI_API_KEY required for RAG integration tests")

        # Create RAG service and ingest evidence
        rag_service = get_rag_service(collection_name="test_rag_tool")
        evidence_list = [
            Evidence(
                content="Machine learning is a subset of artificial intelligence.",
                citation=Citation(
                    source="pubmed",
                    title="ML Basics",
                    url="https://example.com/ml",
                    date="2024",
                    authors=["ML Expert"],
                ),
            )
        ]
        rag_service.ingest_evidence(evidence_list)

        # Create RAG tool
        tool = create_rag_tool(rag_service=rag_service)

        # Search
        results = await tool.search("machine learning", max_results=5)

        # Assert
        assert len(results) > 0
        assert all(isinstance(e, Evidence) for e in results)
        assert results[0].citation.source == "rag"
        assert (
            "machine learning" in results[0].content.lower()
            or "artificial intelligence" in results[0].content.lower()
        )

        # Cleanup
        rag_service.clear_collection()

    @pytest.mark.asyncio
    async def test_rag_tool_empty_collection(self):
        """RAGTool should return empty list when collection is empty."""
        if not settings.openai_api_key:
            pytest.skip("OPENAI_API_KEY required for RAG integration tests")

        rag_service = get_rag_service(collection_name="test_empty")
        rag_service.clear_collection()  # Ensure empty

        tool = create_rag_tool(rag_service=rag_service)
        results = await tool.search("any query")

        assert results == []


@pytest.mark.integration
class TestRAGAgentIntegration:
    """Integration tests for RAGAgent in tool executor."""

    @pytest.mark.asyncio
    async def test_rag_agent_execution(self):
        """RAGAgent should execute and return ToolAgentOutput."""
        if not settings.openai_api_key:
            pytest.skip("OPENAI_API_KEY required for RAG integration tests")

        # Setup: Ingest evidence into RAG
        rag_service = get_rag_service(collection_name="test_rag_agent")
        evidence_list = [
            Evidence(
                content="Deep learning uses neural networks with multiple layers.",
                citation=Citation(
                    source="pubmed",
                    title="Deep Learning",
                    url="https://example.com/dl",
                    date="2024",
                    authors=["DL Researcher"],
                ),
            )
        ]
        rag_service.ingest_evidence(evidence_list)

        # Execute RAGAgent task
        task = AgentTask(
            agent="RAGAgent",
            query="deep learning",
            gap="Need information about deep learning",
        )

        result = await execute_agent_task(task)

        # Assert
        assert result.output
        assert "deep learning" in result.output.lower() or "neural network" in result.output.lower()
        assert len(result.sources) > 0

        # Cleanup
        rag_service.clear_collection()


@pytest.mark.integration
class TestRAGSearchHandlerIntegration:
    """Integration tests for RAG in SearchHandler."""

    @pytest.mark.asyncio
    async def test_search_handler_with_rag(self):
        """SearchHandler should work with RAG tool included."""
        if not settings.openai_api_key:
            pytest.skip("OPENAI_API_KEY required for RAG integration tests")

        # Setup: Create RAG service and ingest some evidence
        rag_service = get_rag_service(collection_name="test_search_handler")
        evidence_list = [
            Evidence(
                content="Test evidence for search handler integration.",
                citation=Citation(
                    source="pubmed",
                    title="Test Evidence",
                    url="https://example.com/test",
                    date="2024",
                    authors=["Tester"],
                ),
            )
        ]
        rag_service.ingest_evidence(evidence_list)

        # Create SearchHandler with RAG
        handler = SearchHandler(
            tools=[],  # No other tools
            include_rag=True,
            auto_ingest_to_rag=False,  # Don't auto-ingest (already has data)
        )

        # Execute search
        result = await handler.execute("test evidence", max_results_per_tool=5)

        # Assert
        assert result.total_found > 0
        assert "rag" in result.sources_searched
        assert any(e.citation.source == "rag" for e in result.evidence)

        # Cleanup
        rag_service.clear_collection()

    @pytest.mark.asyncio
    async def test_search_handler_auto_ingest(self):
        """SearchHandler should auto-ingest evidence into RAG."""
        if not settings.openai_api_key:
            pytest.skip("OPENAI_API_KEY required for RAG integration tests")

        # Create empty RAG service
        rag_service = get_rag_service(collection_name="test_auto_ingest")
        rag_service.clear_collection()

        # Create mock tool that returns evidence
        from unittest.mock import AsyncMock

        mock_tool = AsyncMock()
        mock_tool.name = "pubmed"
        mock_tool.search = AsyncMock(
            return_value=[
                Evidence(
                    content="Evidence to be ingested",
                    citation=Citation(
                        source="pubmed",
                        title="Test",
                        url="https://example.com",
                        date="2024",
                        authors=[],
                    ),
                )
            ]
        )

        # Create handler with auto-ingest enabled
        handler = SearchHandler(
            tools=[mock_tool],
            include_rag=False,  # Don't include RAG as search tool
            auto_ingest_to_rag=True,
        )
        handler._rag_service = rag_service  # Inject RAG service

        # Execute search
        await handler.execute("test query")

        # Verify evidence was ingested
        rag_results = rag_service.retrieve("Evidence to be ingested", top_k=1)
        assert len(rag_results) > 0

        # Cleanup
        rag_service.clear_collection()


@pytest.mark.integration
class TestRAGHybridSearchIntegration:
    """Integration tests for hybrid search (RAG + database)."""

    @pytest.mark.asyncio
    async def test_hybrid_search_rag_and_pubmed(self):
        """SearchHandler should support RAG + PubMed hybrid search."""
        if not settings.openai_api_key:
            pytest.skip("OPENAI_API_KEY required for RAG integration tests")

        # Setup: Ingest evidence into RAG
        rag_service = get_rag_service(collection_name="test_hybrid")
        evidence_list = [
            Evidence(
                content="Previously collected evidence about metformin.",
                citation=Citation(
                    source="pubmed",
                    title="Previous Research",
                    url="https://example.com/prev",
                    date="2024",
                    authors=[],
                ),
            )
        ]
        rag_service.ingest_evidence(evidence_list)

        # Note: This test would require real PubMed API access
        # For now, we'll just test that the handler can be created with both tools
        from src.tools.pubmed import PubMedTool

        handler = SearchHandler(
            tools=[PubMedTool()],
            include_rag=True,
            auto_ingest_to_rag=True,
        )

        # Verify handler has both tools
        tool_names = [t.name for t in handler.tools]
        assert "pubmed" in tool_names
        assert "rag" in tool_names

        # Cleanup
        rag_service.clear_collection()
