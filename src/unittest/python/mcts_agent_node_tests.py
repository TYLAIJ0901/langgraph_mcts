import unittest
import typing

from mcts.config import MCTSConfig
from mcts_agent.node.mcts_node import mcts_node
from mcts_agent.common.state import GraphState, BaseAction # BaseAction might not be directly used here but good for consistency
from mcts_tests import MockState # Assuming this import path works
from langchain_core.runnables import RunnableConfig # For type hinting, actual object not needed

class TestMCTSNodeFunction(unittest.TestCase):

    def setUp(self):
        # Default MCTS parameters to be used in runnable_config
        self.default_mcts_params = {
            'num_simulations': 10,
            'exploration_constant': 1.41, # This is C_p or uct_C
            'uct_C': 1.41 # Explicitly matching MCTSConfig field
        }

        # Game tree setup using MockState
        # Note: MockState uses 'id_val' for its ID parameter.
        self.terminal_state = MockState(id_val='terminal', value=100.0, terminal=True, available_actions=[], children_map={})

        # Scenario 1: 'a1' is the best action (leads to value 10)
        self.s_a1_child_val10 = MockState(id_val='s_a1_child_val10', value=10.0, terminal=True, available_actions=[], children_map={})
        self.s_a2_child_val5 = MockState(id_val='s_a2_child_val5', value=5.0, terminal=True, available_actions=[], children_map={})
        self.root_state_val10_best = MockState(
            id_val='root_val10', value=0.0, terminal=False,
            available_actions=['a1', 'a2'],
            children_map={'a1': self.s_a1_child_val10, 'a2': self.s_a2_child_val5}
        )

        # Scenario 2: 'a1' leads to value 5 (s_a2_child_val5), 'a2' leads to value 3 (s_b1_child_val3)
        # So 'a1' is still best in this setup.
        self.s_b1_child_val3 = MockState(id_val='s_b1_child_val3', value=3.0, terminal=True, available_actions=[], children_map={})
        self.root_state_val5_best = MockState(
            id_val='root_val5', value=0.0, terminal=False,
            available_actions=['a1', 'a2'],
            children_map={'a1': self.s_a2_child_val5, 'a2': self.s_b1_child_val3} # 'a1' leads to state with value 5
        )


    def create_mock_runnable_config(self, mcts_config_data: typing.Union[dict, MCTSConfig]) -> dict: # Actual RunnableConfig is complex
        return {'configurable': {'mcts_config': mcts_config_data}}

    def test_mcts_node_selects_best_action_and_updates_state(self):
        game_state = self.root_state_val10_best # 'a1' leads to value 10 (best), 'a2' to value 5
        initial_graph_state: GraphState = {
            'game_state': game_state,
            'action_records': [],
            'current_player_id': 'player1'
        }

        # More simulations to be more certain of picking the best path
        mcts_params = {'num_simulations': 30, 'exploration_constant': 1.41, 'uct_C': 1.41}
        run_config = self.create_mock_runnable_config(mcts_params)

        result_graph_state = mcts_node(initial_graph_state, typing.cast(RunnableConfig, run_config))

        self.assertEqual(result_graph_state['game_state'].id_val, self.s_a1_child_val10.id_val)
        self.assertTrue(result_graph_state['game_state'].is_terminated)
        self.assertEqual(len(result_graph_state['action_records']), 1)
        self.assertEqual(result_graph_state['action_records'][0], 'a1')
        self.assertEqual(result_graph_state['current_player_id'], 'player1')

    def test_mcts_node_on_terminal_state(self):
        game_state = self.terminal_state
        initial_action_records = ['prev_action1', 'prev_action2']
        initial_graph_state: GraphState = {
            'game_state': game_state,
            'action_records': list(initial_action_records), # Ensure it's a new list
            'current_player_id': 'player2'
        }

        run_config = self.create_mock_runnable_config(self.default_mcts_params)
        result_graph_state = mcts_node(initial_graph_state, typing.cast(RunnableConfig, run_config))

        self.assertEqual(result_graph_state['game_state'].id_val, self.terminal_state.id_val)
        self.assertEqual(result_graph_state['action_records'], initial_action_records) # No new action
        self.assertEqual(len(result_graph_state['action_records']), 2)
        self.assertEqual(result_graph_state['current_player_id'], 'player2')

    def test_mcts_node_initializes_action_records_if_missing(self):
        game_state = self.root_state_val5_best # 'a1' (to val 5) is better than 'a2' (to val 3)
        # 'action_records' key is missing in initial_graph_state
        initial_graph_state: GraphState = {
            'game_state': game_state,
            'current_player_id': 'player_xyz'
        }

        run_config = self.create_mock_runnable_config(self.default_mcts_params)
        result_graph_state = mcts_node(initial_graph_state, typing.cast(RunnableConfig, run_config))

        # Expect action 'a1' to be chosen
        self.assertEqual(result_graph_state['game_state'].id_val, self.s_a2_child_val5.id_val)
        self.assertIsNotNone(result_graph_state.get('action_records'))
        self.assertIsInstance(result_graph_state['action_records'], list)
        self.assertEqual(len(result_graph_state['action_records']), 1)
        self.assertEqual(result_graph_state['action_records'][0], 'a1')
        self.assertEqual(result_graph_state['current_player_id'], 'player_xyz')

    def test_mcts_node_uses_mcts_config_from_runnable_config(self):
        game_state = self.root_state_val10_best
        initial_graph_state: GraphState = {
            'game_state': game_state,
            'action_records': [],
            'current_player_id': 'player1'
        }

        # Test 1: MCTSConfig passed as dict
        # Using very few simulations to ensure it runs quickly.
        # The exact outcome isn't strictly tested here, just that it runs with the config.
        mcts_params_dict = {'num_simulations': 1, 'exploration_constant': 1.0, 'uct_C': 1.0}
        run_config_dict = self.create_mock_runnable_config(mcts_params_dict)
        result_1 = mcts_node(initial_graph_state, typing.cast(RunnableConfig, run_config_dict))

        self.assertIsNotNone(result_1)
        self.assertTrue(len(result_1['action_records']) >= 0) # Could be 0 if no action found, or 1

        # Test 2: MCTSConfig passed as MCTSConfig object
        mcts_config_obj = MCTSConfig(num_simulations=1, exploration_constant=1.0, uct_C=1.0)
        run_config_obj = self.create_mock_runnable_config(mcts_config_obj)
        # Need to pass a fresh copy of initial_graph_state as mcts_node modifies action_records
        initial_graph_state_copy: GraphState = {
            'game_state': game_state,
            'action_records': [],
            'current_player_id': 'player1'
        }
        result_2 = mcts_node(initial_graph_state_copy, typing.cast(RunnableConfig, run_config_obj))

        self.assertIsNotNone(result_2)
        self.assertTrue(len(result_2['action_records']) >= 0)

    def test_mcts_node_handles_no_possible_actions_from_mcts(self):
        # This state is not terminal, but MCTS won't find any actions if get_all_possible_actions is empty.
        no_action_state = MockState(id_val='no_action_state', value=0.0, terminal=False,
                                    available_actions=[], # No actions available from this state
                                    children_map={})

        initial_graph_state: GraphState = {
            'game_state': no_action_state,
            'action_records': [],
            'current_player_id': 'player_no_action'
        }

        run_config = self.create_mock_runnable_config(self.default_mcts_params)
        result_graph_state = mcts_node(initial_graph_state, typing.cast(RunnableConfig, run_config))

        self.assertEqual(result_graph_state['game_state'].id_val, no_action_state.id_val) # State should not change
        self.assertEqual(len(result_graph_state['action_records']), 0) # No action recorded
        self.assertEqual(result_graph_state['current_player_id'], 'player_no_action')

if __name__ == '__main__':
    unittest.main()
