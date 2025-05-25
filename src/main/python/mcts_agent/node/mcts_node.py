from ..common.state import GraphState
import langchain_core.runnables

def mcts_node(graph_state: GraphState, config: langchain_core.runnables.RunnableConfig) -> GraphState:
    """
    Based on the game_state specified in `graph_state`, run MCTS to find the best action and execute
    the best action to get the next game state and update the next game state in graph state.
    Record the executed action in the `action_records` list in the graph state.

    MCTS relayted configurations are specified in config['configurable']['mcts_config'].
    """
    raise NotImplementedError()