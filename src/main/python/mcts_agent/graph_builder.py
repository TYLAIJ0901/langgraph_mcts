import langgraph.graph
import langgraph
from .node.mcts_node import mcts_node
from .common.state import GraphState
from .common.config import ConfigSchema

def build_graph() -> langgraph.graph.StateGraph:
    builder = langgraph.graph.StateGraph(GraphState, config_schema=ConfigSchema)

    NAME_MCTS_NODE = "mcts_node"
    builder.add_node(NAME_MCTS_NODE, mcts_node)
    builder.add_edge(langgraph.graph.START, NAME_MCTS_NODE)
    builder.add_edge(NAME_MCTS_NODE, langgraph.graph.END)

    builder.set_entry_point(NAME_MCTS_NODE)

    return builder