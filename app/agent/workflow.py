"""Assembly of the research-agent StateGraph.

START                      -> plan_research
plan_research              -> select_next_task | finalize_research (invalid input)
select_next_task           -> execute_research_tool (task picked)
                            | verify_evidence (no task left / limit reached)
execute_research_tool      -> evaluate_research_progress
evaluate_research_progress -> revise_research_plan (needs more evidence, retry left)
                            | select_next_task
revise_research_plan       -> select_next_task
verify_evidence            -> generate_report
generate_report            -> finalize_research
finalize_research          -> END
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent import routing
from app.agent.nodes import AgentDependencies, AgentNodes
from app.agent.state import AgentLimits, AgentState


def build_agent_graph(
    deps: AgentDependencies, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    nodes = AgentNodes(deps)
    graph = StateGraph(AgentState)

    graph.add_node(routing.PLAN, nodes.plan_research)
    graph.add_node(routing.SELECT, nodes.select_next_task)
    graph.add_node(routing.EXECUTE, nodes.execute_research_tool)
    graph.add_node(routing.EVALUATE, nodes.evaluate_research_progress)
    graph.add_node(routing.REVISE, nodes.revise_research_plan)
    graph.add_node(routing.VERIFY, nodes.verify_evidence)
    graph.add_node(routing.REPORT, nodes.generate_report)
    graph.add_node(routing.FINALIZE, nodes.finalize_research)

    graph.add_edge(START, routing.PLAN)
    graph.add_conditional_edges(routing.PLAN, routing.route_after_plan)
    graph.add_conditional_edges(routing.SELECT, routing.route_after_select)
    graph.add_conditional_edges(routing.EXECUTE, routing.route_after_execute)
    graph.add_conditional_edges(routing.EVALUATE, routing.route_after_evaluate)
    graph.add_edge(routing.REVISE, routing.SELECT)
    graph.add_conditional_edges(routing.VERIFY, routing.route_after_verify)
    graph.add_edge(routing.REPORT, routing.FINALIZE)
    graph.add_edge(routing.FINALIZE, END)

    return graph.compile(checkpointer=checkpointer, name="research_agent")


def recursion_limit(limits: AgentLimits) -> int:
    """Upper bound on node executions: each task attempt is at most four nodes
    (select, execute, evaluate, revise), plus plan, the final select and the tail."""
    attempts = limits["max_tasks"] * (limits["max_task_retries"] + 1)
    return attempts * 4 + 8


def agent_config(run_id: str, limits: AgentLimits) -> RunnableConfig:
    """Each run is its own checkpoint thread."""
    return {
        "configurable": {"thread_id": f"agent-{run_id}"},
        "recursion_limit": recursion_limit(limits),
    }
