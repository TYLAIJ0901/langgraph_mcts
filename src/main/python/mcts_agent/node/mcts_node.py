import typing # Add typing for MCTSConfig and other types if needed directly
from ..common.state import GraphState
import langchain_core.runnables
from mcts.config import MCTSConfig
from mcts.core import MCTS
from common.base_types import BaseState, BaseAction # For type hinting game_state


def mcts_node(graph_state: GraphState, runnable_config: langchain_core.runnables.RunnableConfig) -> GraphState:
    """
    Based on the game_state specified in `graph_state`, run MCTS to find the best action and execute
    the best action to get the next game state and update the next game state in graph state.
    Record the executed action in the `action_records` list in the graph state.

    MCTS related configurations are specified in runnable_config['configurable']['mcts_config'].
    """
    game_state: BaseState = graph_state['game_state']

    # Initialize action_records if not present or None, to ensure it's a list.
    action_records: list[BaseAction] = graph_state.get('action_records') or []

    # Extract MCTS parameters from the runnable_config
    mcts_config_params = runnable_config['configurable'].get('mcts_config', {})
    # The code below handles if mcts_config_params is MCTSConfig or dict.
    # So the initial check should allow both.
    if not isinstance(mcts_config_params, (dict, MCTSConfig)):
        raise ValueError("MCTS config in runnable_config['configurable']['mcts_config'] must be a dictionary or MCTSConfig instance.")

    # Create MCTSConfig object.
    if isinstance(mcts_config_params, MCTSConfig):
        mcts_config = mcts_config_params
    else:
        # This assumes mcts_config_params contains valid keys for MCTSConfig.
        mcts_config = MCTSConfig(**mcts_config_params)

    # If the current game state is terminal, no action can be taken.
    if game_state.is_terminated:
        updated_graph_state: GraphState = {
            'game_state': game_state,
            'action_records': action_records, # Persist existing records
        }
        return updated_graph_state

    mcts_instance = MCTS(config=mcts_config)
    best_action = mcts_instance.search(initial_state=game_state)

    next_game_state = game_state # Default to current state if no action or error

    if best_action is not None:
        try:
            # Assuming get_next_state returns (next_state_obj, reward)
            next_game_state_obj, _ = game_state.get_next_state(best_action)
            next_game_state = next_game_state_obj # Update to the new state
            action_records.append(best_action)
        except Exception as e:
            # Handle potential errors during get_next_state.
            # For now, we'll let the game_state remain as is before the action attempt,
            # and the action won't be added to records.
            # Consider logging: print(f"Error getting next state for action {best_action}: {e}")
            pass

    updated_graph_state: GraphState = {
        'game_state': next_game_state,
        'action_records': action_records,
    }
    return updated_graph_state
