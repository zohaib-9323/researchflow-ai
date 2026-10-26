"""Assembly of the research StateGraph.

Edges (conditional ones are decided by app/graph/routing.py):

    START             -> classify_question
    classify_question -> retrieve_documents | finalize_response
    retrieve_documents-> generate_answer | revise_search (no results) | finalize_response
    generate_answer   -> evaluate_evidence | finalize_response (error)
    evaluate_evidence -> finalize_response (sufficient / budget exhausted / error)
                       | revise_search (insufficient or irrelevant evidence)
                       | revise_answer (unsupported claims)
    revise_search     -> retrieve_documents
    revise_answer     -> evaluate_evidence
    finalize_response -> END
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph import routing
from app.graph.nodes import GraphDependencies, ResearchNodes
from app.graph.state import ResearchState

# Upper bound on node executions per run: a safety net on top of the retry budget.
RECURSION_LIMIT = 40


def build_research_graph(
    deps: GraphDependencies, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    nodes = ResearchNodes(deps)
    graph = StateGraph(ResearchState)

    graph.add_node(routing.CLASSIFY, nodes.classify_question)
    graph.add_node(routing.RETRIEVE, nodes.retrieve_documents)
    graph.add_node(routing.GENERATE, nodes.generate_answer)
    graph.add_node(routing.EVALUATE, nodes.evaluate_evidence)
    graph.add_node(routing.REVISE_SEARCH, nodes.revise_search)
    graph.add_node(routing.REVISE_ANSWER, nodes.revise_answer)
    graph.add_node(routing.FINALIZE, nodes.finalize_response)

    graph.add_edge(START, routing.CLASSIFY)
    graph.add_conditional_edges(routing.CLASSIFY, routing.route_after_classify)
    graph.add_conditional_edges(routing.RETRIEVE, routing.route_after_retrieve)
    graph.add_conditional_edges(routing.GENERATE, routing.route_after_generate)
    graph.add_conditional_edges(routing.EVALUATE, routing.route_after_evaluate)
    graph.add_conditional_edges(routing.REVISE_SEARCH, routing.route_after_revise_search)
    graph.add_conditional_edges(routing.REVISE_ANSWER, routing.route_after_revise_answer)
    graph.add_edge(routing.FINALIZE, END)

    return graph.compile(checkpointer=checkpointer, name="research_workflow")


def thread_config(thread_id: str) -> RunnableConfig:
    """Each conversation is its own checkpoint thread."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
