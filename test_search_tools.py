"""Regression tests for the arxiv and tavily-python search integrations.

These tests verify that the ``arxiv_search`` and ``tavily_search`` helpers
added in recent commits work correctly and can be used by the agent to
answer questions.
"""

import os
import json
from datetime import datetime
from unittest import mock

import pytest

# --- dependency import sanity checks ---
import arxiv  # noqa: F401 – proves the dependency is installed
from tavily import TavilyClient  # noqa: F401 – proves the dependency is installed

from agent import arxiv_search, tavily_search, TOOLS, TOOL_SCHEMAS


# ===================================================================
# helpers
# ===================================================================

def _make_arxiv_result(
    title="Attention Is All You Need",
    authors=("Ashish Vaswani", "Noam Shazeer", "Niki Parmar"),
    published=None,
    entry_id="https://arxiv.org/abs/1706.03762v7",
    summary="The dominant sequence transduction models are based on complex "
            "recurrent or convolutional neural networks.",
):
    """Build a mock arxiv.Result-like object."""
    r = mock.MagicMock()
    r.title = title
    r.authors = [mock.MagicMock(name=a) for a in authors]
    for a, name in zip(r.authors, authors):
        a.name = name
    r.published = published or datetime(2017, 6, 12)
    r.entry_id = entry_id
    r.summary = summary
    return r


# ===================================================================
# arxiv_search
# ===================================================================

class TestArxivSearch:
    """Test arxiv_search with a mocked arxiv.Client."""

    @mock.patch("agent.arxiv.Client")
    @mock.patch("agent.arxiv.Search")
    def test_returns_results_for_known_paper(self, mock_search_cls, mock_client_cls):
        mock_client = mock.MagicMock()
        mock_client.results.return_value = iter([_make_arxiv_result()])
        mock_client_cls.return_value = mock_client

        result = arxiv_search("attention is all you need transformer", max_results=2)

        assert "Title: Attention Is All You Need" in result
        assert "Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar" in result
        assert "Published: 2017-06-12" in result
        assert "Abstract:" in result
        mock_search_cls.assert_called_once()

    @mock.patch("agent.arxiv.Client")
    @mock.patch("agent.arxiv.Search")
    def test_result_contains_arxiv_id(self, mock_search_cls, mock_client_cls):
        mock_client = mock.MagicMock()
        mock_client.results.return_value = iter([_make_arxiv_result()])
        mock_client_cls.return_value = mock_client

        result = arxiv_search("BERT pre-training", max_results=1)
        assert "ArXiv ID: 1706.03762v7" in result

    @mock.patch("agent.arxiv.Client")
    @mock.patch("agent.arxiv.Search")
    def test_no_results_returns_message(self, mock_search_cls, mock_client_cls):
        mock_client = mock.MagicMock()
        mock_client.results.return_value = iter([])
        mock_client_cls.return_value = mock_client

        result = arxiv_search("xyznonexistent", max_results=1)
        assert "No arXiv papers found" in result

    @mock.patch("agent.arxiv.Client")
    @mock.patch("agent.arxiv.Search")
    def test_multiple_results_separated(self, mock_search_cls, mock_client_cls):
        mock_client = mock.MagicMock()
        mock_client.results.return_value = iter([
            _make_arxiv_result(title="Paper A"),
            _make_arxiv_result(title="Paper B"),
        ])
        mock_client_cls.return_value = mock_client

        result = arxiv_search("machine learning", max_results=2)
        assert "Paper A" in result
        assert "Paper B" in result
        assert "---" in result  # separator between results

    @mock.patch("agent.arxiv.Client")
    @mock.patch("agent.arxiv.Search")
    def test_many_authors_truncated(self, mock_search_cls, mock_client_cls):
        mock_client = mock.MagicMock()
        mock_client.results.return_value = iter([
            _make_arxiv_result(authors=("A", "B", "C", "D", "E")),
        ])
        mock_client_cls.return_value = mock_client

        result = arxiv_search("test", max_results=1)
        assert "et al." in result

    @mock.patch("agent.arxiv.Client")
    @mock.patch("agent.arxiv.Search")
    def test_error_handling(self, mock_search_cls, mock_client_cls):
        mock_client = mock.MagicMock()
        mock_client.results.side_effect = Exception("HTTP 503")
        mock_client_cls.return_value = mock_client

        result = arxiv_search("anything")
        assert "arXiv search error" in result

    def test_registered_in_tools_dict(self):
        assert "arxiv_search" in TOOLS
        assert TOOLS["arxiv_search"] is arxiv_search

    def test_schema_present(self):
        names = [s["function"]["name"] for s in TOOL_SCHEMAS]
        assert "arxiv_search" in names


# ===================================================================
# tavily_search
# ===================================================================

class TestTavilySearch:
    """Test tavily_search with a mocked TavilyClient."""

    @staticmethod
    def _mock_tavily_response():
        return {
            "answer": "The capital of France is Paris.",
            "results": [
                {
                    "title": "Paris - Wikipedia",
                    "url": "https://en.wikipedia.org/wiki/Paris",
                    "content": "Paris is the capital and most populous city of France.",
                },
                {
                    "title": "France - Britannica",
                    "url": "https://www.britannica.com/place/France",
                    "content": "France, country of northwestern Europe. Its capital is Paris.",
                },
            ],
        }

    @mock.patch.dict(os.environ, {"TAVILY_API_KEY": "test-key-12345"})
    @mock.patch("agent.TavilyClient")
    def test_returns_answer_and_results(self, mock_client_cls):
        mock_client = mock.MagicMock()
        mock_client.search.return_value = self._mock_tavily_response()
        mock_client_cls.return_value = mock_client

        result = tavily_search("What is the capital of France?")

        mock_client_cls.assert_called_once_with(api_key="test-key-12345")
        mock_client.search.assert_called_once()
        assert "Paris" in result
        assert "Answer:" in result
        assert "Wikipedia" in result

    @mock.patch.dict(os.environ, {"TAVILY_API_KEY": "test-key-12345"})
    @mock.patch("agent.TavilyClient")
    def test_search_depth_forwarded(self, mock_client_cls):
        mock_client = mock.MagicMock()
        mock_client.search.return_value = self._mock_tavily_response()
        mock_client_cls.return_value = mock_client

        tavily_search("test query", search_depth="basic")

        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs["search_depth"] == "basic"

    @mock.patch.dict(os.environ, {"TAVILY_API_KEY": ""})
    def test_falls_back_to_wikipedia_without_key(self):
        result = tavily_search("Python programming language")
        assert "Tavily search error" not in result
        assert len(result) > 0

    @mock.patch.dict(os.environ, {}, clear=False)
    def test_falls_back_when_key_missing(self):
        env = os.environ.copy()
        env.pop("TAVILY_API_KEY", None)
        with mock.patch.dict(os.environ, env, clear=True):
            result = tavily_search("Python programming language")
            assert "Tavily search error" not in result

    @mock.patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    @mock.patch("agent.TavilyClient")
    def test_handles_empty_results(self, mock_client_cls):
        mock_client = mock.MagicMock()
        mock_client.search.return_value = {"answer": None, "results": []}
        mock_client_cls.return_value = mock_client

        result = tavily_search("obscure nonexistent topic")
        assert "No results found" in result

    @mock.patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    @mock.patch("agent.TavilyClient")
    def test_handles_tavily_exception(self, mock_client_cls):
        mock_client = mock.MagicMock()
        mock_client.search.side_effect = Exception("API rate limit exceeded")
        mock_client_cls.return_value = mock_client

        result = tavily_search("anything")
        assert "Tavily search error" in result

    def test_registered_in_tools_dict(self):
        assert "tavily_search" in TOOLS
        assert TOOLS["tavily_search"] is tavily_search

    def test_schema_present(self):
        names = [s["function"]["name"] for s in TOOL_SCHEMAS]
        assert "tavily_search" in names


# ===================================================================
# Integration: agent answers a question using both search tools
# ===================================================================

class TestSearchIntegration:
    """Simulate the agent calling arxiv_search and tavily_search to answer
    a question, by mocking call_hf to emit tool-call responses."""

    @mock.patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    @mock.patch("agent.TavilyClient")
    @mock.patch("agent.arxiv.Client")
    @mock.patch("agent.arxiv.Search")
    @mock.patch("agent.call_hf")
    def test_agent_uses_both_tools_to_answer(
        self, mock_call_hf, mock_arxiv_search_cls, mock_arxiv_client_cls,
        mock_tavily_cls,
    ):
        from agent import run_agent

        # --- tavily mock ---
        mock_tavily = mock.MagicMock()
        mock_tavily.search.return_value = {
            "answer": "The Transformer was introduced in 'Attention Is All You Need'.",
            "results": [{
                "title": "Attention Is All You Need",
                "url": "https://arxiv.org/abs/1706.03762",
                "content": "We propose a new simple network architecture, the Transformer.",
            }],
        }
        mock_tavily_cls.return_value = mock_tavily

        # --- arxiv mock ---
        mock_arxiv_client = mock.MagicMock()
        mock_arxiv_client.results.return_value = iter([
            _make_arxiv_result(
                title="Attention Is All You Need",
                authors=("Ashish Vaswani", "Noam Shazeer", "Niki Parmar",
                         "Jakob Uszkoreit", "Llion Jones"),
            ),
        ])
        mock_arxiv_client_cls.return_value = mock_arxiv_client

        question = "Who are the authors of 'Attention Is All You Need'?"

        # Iteration 1: LLM calls tavily_search
        tavily_tool_resp = {
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "tavily_search",
                            "arguments": json.dumps({
                                "query": "authors of Attention Is All You Need",
                            }),
                        },
                    }],
                },
            }],
        }

        # Iteration 2: LLM calls arxiv_search
        arxiv_tool_resp = {
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "arxiv_search",
                            "arguments": json.dumps({
                                "query": "Attention Is All You Need",
                                "max_results": 1,
                            }),
                        },
                    }],
                },
            }],
        }

        # Iteration 3: LLM produces final answer
        final_resp = {
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": (
                        "FINAL ANSWER: Vaswani, Shazeer, Parmar, "
                        "Uszkoreit, Jones, Gomez, Kaiser, Polosukhin"
                    ),
                },
            }],
        }

        mock_call_hf.side_effect = [tavily_tool_resp, arxiv_tool_resp, final_resp]

        answer = run_agent(question)

        assert len(answer) > 0
        assert "Vaswani" in answer
        assert mock_tavily.search.called, "tavily_search was not invoked"
        assert mock_call_hf.call_count == 3
