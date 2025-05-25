import unittest
import typing
import math

from mcts.config import MCTSConfig
from mcts.core import MCTS, Node
from common.base_types import BaseState, BaseAction

# 1. Mock Action Type
MockAction = str # Using str directly as suggested

# 2. Mock BaseState Implementation (MockState)
class MockState(BaseState[MockAction]):
    def __init__(self,
                 id_val: str,  # renamed from 'id' to avoid conflict with builtin
                 value: float,
                 terminal: bool,
                 available_actions: list[MockAction],
                 children_map: typing.Optional[typing.Dict[MockAction, 'MockState']] = None):
        self.id_val = id_val
        self.value = value
        self.terminal = terminal
        self.available_actions = available_actions
        self.children_map: typing.Dict[MockAction, 'MockState'] = children_map if children_map else {}

    def reset(self, *args, **kwargs):
        # Not strictly needed for these tests if we always initialize to a specific state.
        pass

    def get_next_state(self, action: MockAction) -> tuple['MockState', float]:
        if action not in self.children_map:
            raise ValueError(f"Action '{action}' not valid for state '{self.id_val}'")
        next_s = self.children_map[action]
        return next_s, 0.0 # Returning 0.0 reward for transitions as specified

    def evaluate(self) -> float:
        return self.value

    @property
    def is_terminated(self) -> bool:
        return self.terminal

    def get_all_possible_actions(self) -> list[MockAction]:
        return self.available_actions

    def __hash__(self) -> int:
        return hash(self.id_val)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MockState):
            return False
        return self.id_val == other.id_val

    def __repr__(self) -> str:
        return f"MockState(id='{self.id_val}', val={self.value}, term={self.terminal}, actions={self.available_actions})"


# 3. Test Class TestMCTSNode
class TestMCTSNode(unittest.TestCase):
    def setUp(self):
        self.config = MCTSConfig(uct_C=1.414) # uct_C is exploration_constant in MCTSConfig
        self.mock_state_non_terminal = MockState(id_val="s1", value=0.5, terminal=False, available_actions=["a1", "a2"])
        self.mock_state_terminal = MockState(id_val="s_term", value=1.0, terminal=True, available_actions=[])

    def test_node_creation(self):
        node = Node(state=self.mock_state_non_terminal, config=self.config)
        self.assertEqual(node.state, self.mock_state_non_terminal)
        self.assertIsNone(node.parent)
        self.assertIsNone(node.action_taken)
        self.assertEqual(node.children, [])
        self.assertEqual(node.visit_count, 0)
        self.assertEqual(node.total_reward, 0.0)
        self.assertEqual(node.config, self.config)
        self.assertFalse(node.is_terminated)

        node_terminal = Node(state=self.mock_state_terminal, config=self.config)
        self.assertTrue(node_terminal.is_terminated)

        # Test config inheritance
        child_node = Node(state=self.mock_state_non_terminal, parent=node)
        self.assertEqual(child_node.config, self.config)


    def test_average_reward(self):
        node = Node(state=self.mock_state_non_terminal, config=self.config)
        self.assertEqual(node.average_reward, 0.0)

        node.visit_count = 10
        node.total_reward = 5.0
        self.assertEqual(node.average_reward, 0.5)

        node.visit_count = 0 # Reset
        node.total_reward = 0 # Reset
        self.assertEqual(node.average_reward, 0.0)


    def test_uct_value_unvisited(self):
        node = Node(state=self.mock_state_non_terminal, config=self.config)
        self.assertEqual(node.uct_value(parent_visit_count=10), float('inf'))

    def test_uct_value_visited(self):
        node = Node(state=self.mock_state_non_terminal, config=self.config)
        node.visit_count = 5
        node.total_reward = 2.5 # avg_reward = 0.5
        parent_visits = 20

        # UCT = avg_reward + uct_C * sqrt(log(parent_visits) / node_visits)
        # UCT = 0.5 + 1.414 * sqrt(log(20) / 5)
        # log(20) approx 2.9957
        # sqrt(2.9957 / 5) = sqrt(0.59914) approx 0.7740
        # 0.5 + 1.414 * 0.7740 = 0.5 + 1.0945 approx
        # Expected = 1.5945
        expected_uct = 0.5 + self.config.uct_C * math.sqrt(math.log(parent_visits) / node.visit_count)
        self.assertAlmostEqual(node.uct_value(parent_visit_count=parent_visits), expected_uct, places=4)


# 4. Test Class TestMCTSAlgorithm
class TestMCTSAlgorithm(unittest.TestCase):
    def setUp(self):
        self.config = MCTSConfig(num_simulations=50, uct_C=1.414) # Low sims for speed

        # Define a simple game tree
        self.s_a1_done = MockState(id_val="A1", value=10.0, terminal=True, available_actions=[])
        self.s_b1_done = MockState(id_val="B1", value=5.0, terminal=True, available_actions=[])

        self.s_a = MockState(id_val="A", value=0, terminal=False, available_actions=["done_A"], children_map={"done_A": self.s_a1_done})
        self.s_b = MockState(id_val="B", value=0, terminal=False, available_actions=["done_B"], children_map={"done_B": self.s_b1_done})

        self.root_state = MockState(id_val="root", value=0, terminal=False,
                                    available_actions=["go_A", "go_B"],
                                    children_map={"go_A": self.s_a, "go_B": self.s_b})

        self.mcts = MCTS(self.config)

    def test_search_on_terminal_initial_state(self):
        terminal_state = MockState(id_val="terminal", value=1.0, terminal=True, available_actions=[])
        action = self.mcts.search(terminal_state)
        self.assertIsNone(action)

    def test_search_simple_deterministic_tree(self):
        # root_state -> s_a (via 'go_A') -> s_a1_done (value 10)
        # root_state -> s_b (via 'go_B') -> s_b1_done (value 5)
        # With enough simulations, MCTS should pick 'go_A'

        # Modify config for this test to ensure it picks the best path
        # A higher number of simulations might be needed if randomness causes issues,
        # but for this deterministic tree, 50 should be okay.
        # If the values were closer or tree deeper, might need more.

        best_action = self.mcts.search(self.root_state)
        self.assertEqual(best_action, "go_A")

    def test_expand_node(self):
        # root_state has actions ["go_A", "go_B"]
        root_node = Node(state=self.root_state, config=self.config)

        self.assertEqual(len(root_node.children), 0)

        # First expansion
        child1 = self.mcts._expand_node(root_node)
        self.assertIsNotNone(child1)
        self.assertEqual(len(root_node.children), 1)
        self.assertIn(child1.action_taken, ["go_A", "go_B"])
        self.assertIs(child1.parent, root_node)

        # Second expansion
        child2 = self.mcts._expand_node(root_node)
        self.assertIsNotNone(child2)
        self.assertEqual(len(root_node.children), 2)
        self.assertNotEqual(child1.action_taken, child2.action_taken) # Ensure different actions chosen
        self.assertIs(child2.parent, root_node)

        # Try to expand again (fully expanded)
        same_node = self.mcts._expand_node(root_node)
        self.assertIs(same_node, root_node) # Should return the node itself
        self.assertEqual(len(root_node.children), 2) # No new children

    def test_simulate_random_playout(self):
        # s0 -> s1 (action 'x') -> s2 (terminal, value 20)
        s2_term = MockState(id_val="s2_term", value=20.0, terminal=True, available_actions=[])
        s1_inter = MockState(id_val="s1_inter", value=0, terminal=False, available_actions=["to_s2"], children_map={"to_s2": s2_term})
        s0_start = MockState(id_val="s0_start", value=0, terminal=False, available_actions=["to_s1"], children_map={"to_s1": s1_inter})

        node_s0 = Node(state=s0_start, config=self.config)

        # Since the playout is deterministic for this mock state chain:
        # s0_start -> "to_s1" -> s1_inter -> "to_s2" -> s2_term (evaluates to 20.0)
        reward = self.mcts._simulate_random_playout(node_s0)
        self.assertEqual(reward, 20.0)

        # Test with an already terminal node
        node_s2_term = Node(state=s2_term, config=self.config)
        reward_terminal = self.mcts._simulate_random_playout(node_s2_term)
        self.assertEqual(reward_terminal, 20.0)


    def test_backpropagate(self):
        # Create a chain: root -> child -> grandchild
        grandchild_state = MockState(id_val="gc_s", value=1, terminal=True, available_actions=[])
        child_state = MockState(id_val="c_s", value=0, terminal=False, available_actions=["to_gc"], children_map={"to_gc": grandchild_state})
        root_s = MockState(id_val="r_s", value=0, terminal=False, available_actions=["to_c"], children_map={"to_c": child_state})

        root_n = Node(state=root_s, config=self.config)
        child_n = Node(state=child_state, parent=root_n, action_taken="to_c", config=self.config)
        grandchild_n = Node(state=grandchild_state, parent=child_n, action_taken="to_gc", config=self.config)

        reward_val = 1.0
        self.mcts._backpropagate(grandchild_n, reward_val)

        # Check grandchild
        self.assertEqual(grandchild_n.visit_count, 1)
        self.assertEqual(grandchild_n.total_reward, reward_val)
        # Check child
        self.assertEqual(child_n.visit_count, 1)
        self.assertEqual(child_n.total_reward, reward_val)
        # Check root
        self.assertEqual(root_n.visit_count, 1)
        self.assertEqual(root_n.total_reward, reward_val)

        # Second backpropagation from another simulation (e.g. different reward through grandchild)
        reward_val_2 = 0.5
        self.mcts._backpropagate(grandchild_n, reward_val_2)
        self.assertEqual(grandchild_n.visit_count, 2)
        self.assertEqual(grandchild_n.total_reward, reward_val + reward_val_2)
        self.assertEqual(child_n.visit_count, 2)
        self.assertEqual(child_n.total_reward, reward_val + reward_val_2)
        self.assertEqual(root_n.visit_count, 2)
        self.assertEqual(root_n.total_reward, reward_val + reward_val_2)


    def test_selection_logic_respects_uct(self):
        root_node = Node(state=self.root_state, config=self.config) # root_state has 'go_A', 'go_B'

        # Manually create children to control their stats
        # Child 1 (c1) - action 'go_A' leading to state self.s_a
        c1 = Node(state=self.s_a, parent=root_node, action_taken="go_A", config=self.config)
        # Child 2 (c2) - action 'go_B' leading to state self.s_b
        c2 = Node(state=self.s_b, parent=root_node, action_taken="go_B", config=self.config)

        root_node.children = [c1, c2]

        # Set stats such that c2 has higher UCT
        # UCT = avg_reward + C * sqrt(log(N_parent) / N_child)
        # Let C = 1.414 (as per self.config)
        # N_parent (root_node.visit_count) must be sum of children visits (or more)

        c1.visit_count = 5
        c1.total_reward = 2.0 # avg_reward_c1 = 0.4

        c2.visit_count = 2
        c2.total_reward = 1.0 # avg_reward_c2 = 0.5

        root_node.visit_count = c1.visit_count + c2.visit_count # 7

        # UCT_c1 = 0.4 + 1.414 * sqrt(log(7) / 5) = 0.4 + 1.414 * sqrt(1.946 / 5) = 0.4 + 1.414 * sqrt(0.3892)
        #          = 0.4 + 1.414 * 0.6238 = 0.4 + 0.882 = 1.282
        # UCT_c2 = 0.5 + 1.414 * sqrt(log(7) / 2) = 0.5 + 1.414 * sqrt(1.946 / 2) = 0.5 + 1.414 * sqrt(0.973)
        #          = 0.5 + 1.414 * 0.9864 = 0.5 + 1.394 = 1.894
        # So, c2 should be selected.

        # Ensure root_node itself is not terminal and has children
        self.assertFalse(root_node.is_terminated)
        self.assertTrue(len(root_node.children) > 0)

        # The _select_promising_node logic: if node is not fully expanded, it returns itself.
        # To test UCT selection among children, we need to ensure the node *thinks* it's fully expanded.
        # This means len(root_node.children) >= len(root_node.state.get_all_possible_actions())
        # self.root_state has actions ["go_A", "go_B"], so 2 actions.
        # We have 2 children c1, c2. So it is "fully expanded" in terms of having children for all initial actions.

        selected_node = self.mcts._select_promising_node(root_node)

        # _select_promising_node will traverse down until an unexpanded or terminal node.
        # If c1 and c2 are not terminal and have no children of their own (yet),
        # and are not fully expanded (i.e. their own children list < their own state's possible actions),
        # then select_promising_node would pick one (c2 due to UCT) and then try to expand it.
        # If _select_promising_node returns c2, it means c2 was chosen for further action.
        self.assertIs(selected_node, c2,
                      f"Expected c2 to be selected. UCT_c1={c1.uct_value(root_node.visit_count)}, UCT_c2={c2.uct_value(root_node.visit_count)}")

# 5. Main execution block
if __name__ == '__main__':
    unittest.main()
