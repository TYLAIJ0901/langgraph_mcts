import math
import random # Ensure random is imported
import typing

from src.main.python.common.base_types import BaseState, BaseAction
from src.main.python.mcts.config import MCTSConfig

# TYPE_CHECKING block might be needed if MCTS refers to Node and Node refers to MCTS later in the same file.
# For now, Node is defined first.
# if typing.TYPE_CHECKING:
#     pass

class Node(typing.Generic[BaseAction]):
    def __init__(self, state: BaseState, parent: typing.Optional['Node[BaseAction]'] = None,
                 action_taken: typing.Optional[BaseAction] = None, config: MCTSConfig = None):
        # Critical Assumption: The 'state' object is expected to have a method 'get_all_possible_actions() -> list[BaseAction]'
        # This is necessary for MCTS operations like expansion and determining if a node is fully expanded.
        self.state: BaseState = state
        self.parent: typing.Optional['Node[BaseAction]'] = parent
        self.action_taken: typing.Optional[BaseAction] = action_taken # Action that led to this node

        self.children: list['Node[BaseAction]'] = []

        self.visit_count: int = 0
        self.total_reward: float = 0.0

        if config is None:
            if parent is None:
                # This should ideally not happen if MCTS class always creates root with config.
                raise ValueError("MCTSConfig must be provided to the root node or be inheritable from a parent.")
            self.config: MCTSConfig = parent.config
        else:
            self.config: MCTSConfig = config

    @property
    def average_reward(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count

    @property
    def is_terminated(self) -> bool:
        # Assuming BaseState has an 'is_terminated' property or method
        # The provided snippet uses self.state.is_terminated, implying it's a property.
        # If it were a method, it would be self.state.is_terminated().
        # Let's stick to the snippet's direct attribute access.
        return self.state.is_terminated

    def uct_value(self, parent_visit_count: int) -> float:
        if self.visit_count == 0:
            # Encourage exploration of unvisited nodes
            return float('inf')
        
        # Q(s,a) is the average reward of the current node (exploitation term)
        exploitation_term = self.average_reward
        
        # C_p * sqrt(ln(N(s_parent)) / N(s,a)) is the exploration term
        # N(s_parent) is parent_visit_count
        # N(s,a) is self.visit_count
        if parent_visit_count == 0: # Should not happen if root is visited once before children are explored
            # Avoid math domain error for log(0) or log(<1) if parent_visit_count is unusually low.
            # log(1) is 0, so if parent_visit_count is 1, exploration term becomes 0 unless handled.
            # A common practice is to ensure log argument is > 1 or handle small N(s_parent) carefully.
            # If parent_visit_count is 1 (e.g. just after root creation, exploring its direct children),
            # log(1) = 0, making exploration term 0. Some implementations add a small epsilon or use log(parent_visit_count + 1).
            # For now, standard log:
            if parent_visit_count < 1: parent_visit_count = 1 # Defensive

        exploration_term = self.config.uct_C * math.sqrt(
            math.log(parent_visit_count) / self.visit_count
        )
        return exploitation_term + exploration_term

    def __repr__(self) -> str:
        return (f"Node(action={self.action_taken}, visits={self.visit_count}, "
                f"avg_reward={self.average_reward:.2f}, children={len(self.children)}, "
                f"terminated={self.is_terminated})")

class MCTS(typing.Generic[BaseAction]):
    def __init__(self, config: MCTSConfig):
        self.config = config

    def search(self, initial_state: BaseState) -> typing.Optional[BaseAction]:
        if not hasattr(initial_state, 'get_all_possible_actions'):
            raise AttributeError("The initial_state object must have a method 'get_all_possible_actions'.")
        
        if not hasattr(initial_state, 'is_terminated'):
            raise AttributeError("The initial_state object must have a property or method 'is_terminated'.")

        if not hasattr(initial_state, 'get_next_state'):
            raise AttributeError("The initial_state object must have a method 'get_next_state'.")

        if not hasattr(initial_state, 'evaluate'):
            raise AttributeError("The initial_state object must have a method 'evaluate'.")


        root_node: Node[BaseAction] = Node(state=initial_state, config=self.config)

        if root_node.is_terminated:
            return None # Initial state is already terminal, no action to take.

        for i in range(self.config.num_simulations):
            promising_node = self._select_promising_node(root_node)

            expanded_node_for_simulation = promising_node
            if not promising_node.is_terminated:
                expanded_node_for_simulation = self._expand_node(promising_node)
            
            simulation_reward = self._simulate_random_playout(expanded_node_for_simulation)
            
            self._backpropagate(expanded_node_for_simulation, simulation_reward)

        best_child = None
        highest_visits = -1

        if not root_node.children:
            return None 

        for child in root_node.children:
            if child.visit_count > highest_visits:
                highest_visits = child.visit_count
                best_child = child
        
        return best_child.action_taken if best_child else None

    def _select_promising_node(self, node: Node[BaseAction]) -> Node[BaseAction]:
        current_node = node
        while not current_node.is_terminated:
            if not hasattr(current_node.state, 'get_all_possible_actions'):
                 raise AttributeError(f"State {type(current_node.state)} lacks 'get_all_possible_actions'")

            possible_actions = current_node.state.get_all_possible_actions()
            if not possible_actions: 
                break 

            if len(current_node.children) < len(possible_actions):
                return current_node
            
            if not current_node.children: 
                 break 

            best_child = None
            max_uct = -float('inf')
            for child_node in current_node.children:
                uct = child_node.uct_value(parent_visit_count=current_node.visit_count)
                if uct > max_uct:
                    max_uct = uct
                    best_child = child_node
            
            if best_child is None: 
                break 
            current_node = best_child
        
        return current_node

    def _expand_node(self, node: Node[BaseAction]) -> Node[BaseAction]:
        if node.is_terminated: 
            return node 

        if not hasattr(node.state, 'get_all_possible_actions'):
            raise AttributeError(f"State {type(node.state)} lacks 'get_all_possible_actions'")

        possible_actions = node.state.get_all_possible_actions()
        tried_actions = {child.action_taken for child in node.children}

        for action in possible_actions:
            if action not in tried_actions:
                # Assuming get_next_state returns a tuple (next_state, reward_from_action)
                # The reward from action is not used here, only the next_state
                next_state, _ = node.state.get_next_state(action)
                new_child_node = Node(state=next_state, parent=node, action_taken=action, config=node.config)
                node.children.append(new_child_node)
                return new_child_node 

        return node 

    def _simulate_random_playout(self, node: Node[BaseAction]) -> float:
        current_state = node.state 
        
        # Use max_simulation_depth from config if available, otherwise a default.
        # The problem description does not specify adding max_simulation_depth to MCTSConfig,
        # so I'm using a default value here as shown in the snippet.
        # Consider adding it to MCTSConfig for more flexibility.
        max_simulation_depth = getattr(self.config, 'max_simulation_depth', 100) 

        for _ in range(max_simulation_depth):
            if current_state.is_terminated:
                break
            
            if not hasattr(current_state, 'get_all_possible_actions'):
                raise AttributeError(f"State {type(current_state)} in sim lacks 'get_all_possible_actions'")

            available_actions = current_state.get_all_possible_actions()
            if not available_actions:
                break 
            
            random_action = random.choice(available_actions)
            # Assuming get_next_state returns (next_state, reward_from_action)
            current_state, _ = current_state.get_next_state(random_action)
        
        # Assuming BaseState has an evaluate() method that returns the value of the terminal state.
        return current_state.evaluate()

    def _backpropagate(self, node: Node[BaseAction], reward: float):
        temp_node = node
        while temp_node is not None:
            temp_node.visit_count += 1
            temp_node.total_reward += reward
            temp_node = temp_node.parent
