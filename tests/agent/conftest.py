from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.tools import WebSearchProvider
from app.config import Settings
from app.container import Container, build_container
from app.llm.embeddings import HashingEmbeddings
from tests.agent.fakes import ScriptedAgentLLM
from tests.conftest import make_settings
from tests.fakes import ScriptedLLM


@pytest.fixture
def agent_llm() -> ScriptedAgentLLM:
    return ScriptedAgentLLM()


@pytest.fixture
def container(settings: Settings, llm: ScriptedLLM, agent_llm: ScriptedAgentLLM) -> Container:
    """Agent tests run the graph inline (no background thread) for determinism."""
    return build_container(
        settings,
        llm=llm,
        agent_llm=agent_llm,
        embeddings=HashingEmbeddings(),
        agent_background=False,
    )


def agent_container(
    tmp_path: Path,
    agent_llm: ScriptedAgentLLM,
    web_search: WebSearchProvider | None = None,
    **overrides: object,
) -> Container:
    return build_container(
        make_settings(tmp_path, **overrides),
        llm=ScriptedLLM(),
        agent_llm=agent_llm,
        embeddings=HashingEmbeddings(),
        agent_background=False,
        web_search=web_search,
    )
