"""
Monte Carlo Tree Search agent for strategic planning in Snake AI
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import copy
import time

from game.game_state import GameState, Direction
from utils.logger import setup_logger

class MCTSNode:
    """Node in the Monte Carlo Tree Search"""
    
    def __init__(self, state: GameState, parent: Optional['MCTSNode'] = None, 
                 action: Optional[int] = None):
        self.state = copy.deepcopy(state)
        self.parent = parent
        self.action = action
        
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior_probability = 0.0
        
        # For progressive widening
        self.expanded_actions = set()
        
    @property
    def average_value(self) -> float:
        """Get average value of this node"""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return len(self.children) == 0
    
    @property
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been expanded"""
        return len(self.expanded_actions) == 4  # 4 possible actions
    
    def ucb_score(self, c_puct: float = 1.0) -> float:
        """Calculate UCB score for action selection"""
        if self.visits == 0:
            return float('inf')
        
        # UCB1 formula with exploration
        exploitation = self.average_value
        exploration = c_puct * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration
    
    def add_child(self, action: int, child_state: GameState) -> 'MCTSNode':
        """Add a child node"""
        child = MCTSNode(child_state, parent=self, action=action)
        self.children[action] = child
        self.expanded_actions.add(action)
        return child
    
    def update(self, value: float):
        """Update node statistics"""
        self.visits += 1
        self.value_sum += value
    
    def select_best_child(self, c_puct: float = 1.0) -> 'MCTSNode':
        """Select best child using UCB"""
        best_child = None
        best_score = -float('inf')
        
        for child in self.children.values():
            score = child.ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def select_most_visited_child(self) -> 'MCTSNode':
        """Select child with most visits"""
        if not self.children:
            return None
        
        return max(self.children.values(), key=lambda child: child.visits)

class MCTSAgent:
    """Monte Carlo Tree Search agent for Snake AI"""
    
    def __init__(self, simulations: int = 1000, c_puct: float = 1.0, 
                 max_depth: int = 50, config=None):
        self.simulations = simulations
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.config = config
        self.logger = setup_logger('mcts_agent')
        
        # Neural network for value estimation (optional)
        self.value_network = None
        self.policy_network = None
        
        # Statistics
        self.search_times = []
        self.tree_sizes = []
        
    def set_networks(self, value_network=None, policy_network=None):
        """Set neural networks for guided search"""
        self.value_network = value_network
        self.policy_network = policy_network
    
    def get_action(self, game_state: GameState) -> Tuple[int, Dict]:
        """Get best action using MCTS"""
        start_time = time.time()
        
        # Create root node
        root = MCTSNode(game_state)
        
        # Run simulations
        for simulation in range(self.simulations):
            self._run_simulation(root)
        
        # Select best action
        best_child = root.select_most_visited_child()
        best_action = best_child.action if best_child else 0
        
        # Collect statistics
        search_time = time.time() - start_time
        self.search_times.append(search_time)
        
        tree_size = self._count_nodes(root)
        self.tree_sizes.append(tree_size)
        
        # Prepare info
        info = {
            'search_time': search_time,
            'tree_size': tree_size,
            'simulations': self.simulations,
            'best_value': best_child.average_value if best_child else 0,
            'root_visits': root.visits,
            'action_values': {
                action: child.average_value 
                for action, child in root.children.items()
            },
            'action_visits': {
                action: child.visits 
                for action, child in root.children.items()
            }
        }
        
        return best_action, info
    
    def _run_simulation(self, root: MCTSNode):
        """Run one MCTS simulation"""
        path = []
        node = root
        
        # Selection: traverse down the tree
        while not node.is_leaf and not node.state.game_over:
            if not node.is_fully_expanded:
                # Expansion: add a new child
                node = self._expand_node(node)
                path.append(node)
                break
            else:
                # Selection: choose best child
                node = node.select_best_child(self.c_puct)
                path.append(node)
        
        # Simulation: random rollout or neural network evaluation
        if node.state.game_over:
            value = self._evaluate_terminal_state(node.state)
        else:
            value = self._simulate_random_game(node.state)
        
        # Backpropagation: update all nodes in path
        for node in reversed(path):
            node.update(value)
        
        # Update root
        root.update(value)
    
    def _expand_node(self, node: MCTSNode) -> MCTSNode:
        """Expand a node by adding a new child"""
        available_actions = [
            action for action in range(4) 
            if action not in node.expanded_actions
        ]
        
        if not available_actions:
            return node
        
        # Choose action to expand
        if self.policy_network:
            # Use policy network to guide expansion
            action = self._select_action_with_policy(node.state, available_actions)
        else:
            # Random expansion
            action = np.random.choice(available_actions)
        
        # Create child state
        child_state = copy.deepcopy(node.state)
        reward, done = child_state.step(action)
        
        # Add child
        child = node.add_child(action, child_state)
        
        # Set prior probability if using policy network
        if self.policy_network:
            child.prior_probability = self._get_action_probability(node.state, action)
        
        return child
    
    def _simulate_random_game(self, initial_state: GameState) -> float:
        """Simulate a random game from the given state"""
        state = copy.deepcopy(initial_state)
        total_reward = 0
        steps = 0
        max_steps = 200  # Prevent infinite games
        
        while not state.game_over and steps < max_steps:
            # Random action
            action = np.random.randint(4)
            reward, done = state.step(action)
            total_reward += reward
            steps += 1
        
        # Enhanced evaluation
        value = self._evaluate_state(state, total_reward, steps)
        
        return value
    
    def _evaluate_state(self, state: GameState, total_reward: float, steps: int) -> float:
        """Evaluate a game state"""
        if self.value_network:
            # Use neural network for evaluation
            state_tensor = self._state_to_tensor(state)
            with torch.no_grad():
                value = self.value_network(state_tensor).item()
            return value
        
        # Heuristic evaluation
        base_value = total_reward
        
        # Bonus for survival
        survival_bonus = steps * 0.01
        
        # Bonus for score
        score_bonus = state.score * 10
        
        # Penalty for getting stuck
        if steps > 0:
            efficiency = state.score / steps
            efficiency_bonus = efficiency * 5
        else:
            efficiency_bonus = 0
        
        # Distance to food penalty/bonus
        if state.snake:
            head_x, head_y = state.snake[0]
            food_x, food_y = state.food
            distance = abs(head_x - food_x) + abs(head_y - food_y)
            distance_penalty = distance * 0.1
        else:
            distance_penalty = 0
        
        total_value = base_value + survival_bonus + score_bonus + efficiency_bonus - distance_penalty
        
        return total_value
    
    def _evaluate_terminal_state(self, state: GameState) -> float:
        """Evaluate a terminal game state"""
        if state.game_over:
            # Penalty for dying, bonus for score
            death_penalty = -10
            score_bonus = state.score * 10
            return death_penalty + score_bonus
        else:
            return self._evaluate_state(state, 0, state.steps)
    
    def _select_action_with_policy(self, state: GameState, available_actions: List[int]) -> int:
        """Select action using policy network"""
        if not self.policy_network:
            return np.random.choice(available_actions)
        
        state_tensor = self._state_to_tensor(state)
        
        with torch.no_grad():
            action_probs = self.policy_network(state_tensor)
            
        # Filter to available actions
        available_probs = [action_probs[action].item() for action in available_actions]
        available_probs = np.array(available_probs)
        available_probs = available_probs / np.sum(available_probs)  # Normalize
        
        # Sample from distribution
        chosen_idx = np.random.choice(len(available_actions), p=available_probs)
        return available_actions[chosen_idx]
    
    def _get_action_probability(self, state: GameState, action: int) -> float:
        """Get probability of an action from policy network"""
        if not self.policy_network:
            return 0.25  # Uniform probability
        
        state_tensor = self._state_to_tensor(state)
        
        with torch.no_grad():
            action_probs = self.policy_network(state_tensor)
            return action_probs[action].item()
    
    def _state_to_tensor(self, state: GameState):
        """Convert game state to tensor for neural networks"""
        # This should match the input format expected by your networks
        import torch
        compact_state = state.get_compact_state()
        return torch.FloatTensor(compact_state).unsqueeze(0)
    
    def _count_nodes(self, root: MCTSNode) -> int:
        """Count total nodes in the tree"""
        count = 1
        for child in root.children.values():
            count += self._count_nodes(child)
        return count
    
    def get_statistics(self) -> Dict:
        """Get MCTS statistics"""
        recent_times = self.search_times[-100:] if self.search_times else [0]
        recent_sizes = self.tree_sizes[-100:] if self.tree_sizes else [0]
        
        return {
            'avg_search_time': np.mean(recent_times),
            'avg_tree_size': np.mean(recent_sizes),
            'simulations': self.simulations,
            'c_puct': self.c_puct,
            'max_depth': self.max_depth,
            'total_searches': len(self.search_times)
        }
    
    def analyze_position(self, game_state: GameState) -> Dict:
        """Analyze a position using MCTS"""
        root = MCTSNode(game_state)
        
        # Run simulations
        for _ in range(self.simulations):
            self._run_simulation(root)
        
        # Analyze results
        analysis = {
            'total_simulations': root.visits,
            'estimated_value': root.average_value,
            'action_analysis': {}
        }
        
        for action, child in root.children.items():
            action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
            analysis['action_analysis'][action_names[action]] = {
                'visits': child.visits,
                'average_value': child.average_value,
                'visit_percentage': child.visits / root.visits * 100,
                'ucb_score': child.ucb_score(self.c_puct)
            }
        
        return analysis

class AlphaBetaAgent:
    """Alpha-Beta pruning agent for comparison"""
    
    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth
        self.nodes_evaluated = 0
        
    def get_action(self, game_state: GameState) -> Tuple[int, Dict]:
        """Get best action using Alpha-Beta search"""
        self.nodes_evaluated = 0
        start_time = time.time()
        
        _, best_action = self._alpha_beta(
            game_state, self.max_depth, -float('inf'), float('inf'), True
        )
        
        search_time = time.time() - start_time
        
        info = {
            'search_time': search_time,
            'nodes_evaluated': self.nodes_evaluated,
            'max_depth': self.max_depth
        }
        
        return best_action if best_action is not None else 0, info
    
    def _alpha_beta(self, state: GameState, depth: int, alpha: float, 
                   beta: float, maximizing: bool) -> Tuple[float, Optional[int]]:
        """Alpha-Beta search algorithm"""
        self.nodes_evaluated += 1
        
        if depth == 0 or state.game_over:
            return self._evaluate_position(state), None
        
        best_action = None
        
        if maximizing:
            max_eval = -float('inf')
            
            for action in range(4):
                child_state = copy.deepcopy(state)
                reward, done = child_state.step(action)
                
                eval_score, _ = self._alpha_beta(child_state, depth - 1, alpha, beta, False)
                eval_score += reward
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            
            return max_eval, best_action
        
        else:
            min_eval = float('inf')
            
            for action in range(4):
                child_state = copy.deepcopy(state)
                reward, done = child_state.step(action)
                
                eval_score, _ = self._alpha_beta(child_state, depth - 1, alpha, beta, True)
                eval_score += reward
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            
            return min_eval, best_action
    
    def _evaluate_position(self, state: GameState) -> float:
        """Evaluate position heuristically"""
        if state.game_over:
            return -1000 + state.score * 10
        
        # Basic heuristics
        score_value = state.score * 100
        
        if state.snake:
            head_x, head_y = state.snake[0]
            food_x, food_y = state.food
            
            # Distance to food
            food_distance = abs(head_x - food_x) + abs(head_y - food_y)
            distance_value = -food_distance * 2
            
            # Space availability
            free_spaces = self._count_free_spaces(state)
            space_value = free_spaces * 0.5
            
            # Avoid walls and self
            danger_penalty = self._calculate_danger(state) * -10
            
            return score_value + distance_value + space_value + danger_penalty
        
        return score_value
    
    def _count_free_spaces(self, state: GameState) -> int:
        """Count free spaces around the snake head"""
        if not state.snake:
            return 0
        
        head_x, head_y = state.snake[0]
        free_count = 0
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x, y = head_x + dx, head_y + dy
            
            if (0 <= x < state.width and 0 <= y < state.height and 
                (x, y) not in state.snake):
                free_count += 1
        
        return free_count
    
    def _calculate_danger(self, state: GameState) -> int:
        """Calculate immediate danger level"""
        if not state.snake:
            return 0
        
        head_x, head_y = state.snake[0]
        danger_count = 0
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x, y = head_x + dx, head_y + dy
            
            if (x < 0 or x >= state.width or y < 0 or y >= state.height or 
                (x, y) in state.snake):
                danger_count += 1
        
        return danger_count
