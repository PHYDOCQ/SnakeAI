"""
Advanced experience replay mechanisms for Snake AI training
"""

import numpy as np
import random
from typing import Tuple, List, NamedTuple, Optional
from collections import deque
import torch

class Experience(NamedTuple):
    """Single experience tuple"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

class SumTree:
    """Sum tree data structure for prioritized experience replay"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.empty(capacity, dtype=object)
        self.data_pointer = 0
        self.size = 0
    
    def add(self, priority: float, data: Experience):
        """Add new experience with priority"""
        tree_idx = self.data_pointer + self.capacity - 1
        
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
    
    def update(self, tree_idx: int, priority: float):
        """Update priority of experience"""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Propagate change up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def get_leaf(self, value: float) -> Tuple[int, float, Experience]:
        """Retrieve experience based on priority value"""
        parent_idx = 0
        
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            
            if value <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                value -= self.tree[left_child_idx]
                parent_idx = right_child_idx
        
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    @property
    def total_priority(self) -> float:
        """Get total priority (root of tree)"""
        return self.tree[0]

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6
        
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size: int, beta: Optional[float] = None) -> Tuple:
        """Sample batch of experiences"""
        if beta is None:
            beta = self.beta
        
        experiences = []
        indices = []
        weights = []
        
        priority_segment = self.tree.total_priority / batch_size
        
        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            value = random.uniform(a, b)
            
            idx, priority, experience = self.tree.get_leaf(value)
            
            if experience is not None:
                # Calculate importance sampling weight
                sampling_prob = priority / self.tree.total_priority
                weight = (self.tree.size * sampling_prob) ** (-beta)
                
                experiences.append(experience)
                indices.append(idx)
                weights.append(weight)
        
        # Normalize weights
        if weights:
            max_weight = max(weights)
            weights = [w / max_weight for w in weights]
        
        # Convert to arrays
        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences])
        
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities of sampled experiences"""
        for idx, priority in zip(indices, priorities):
            priority = max(priority, self.epsilon)
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority ** self.alpha)
    
    def __len__(self) -> int:
        return self.tree.size

class HindsightExperienceReplay:
    """Hindsight Experience Replay for sparse rewards"""
    
    def __init__(self, buffer: PrioritizedReplayBuffer, strategy: str = "future", k: int = 4):
        self.buffer = buffer
        self.strategy = strategy
        self.k = k  # Number of additional goals to sample
        self.episode_buffer = []
    
    def add_episode(self, episode_experiences: List[Experience]):
        """Add full episode and generate HER experiences"""
        # Add original experiences
        for exp in episode_experiences:
            self.buffer.add(exp.state, exp.action, exp.reward, exp.next_state, exp.done)
        
        # Generate HER experiences
        her_experiences = self._generate_her_experiences(episode_experiences)
        
        for exp in her_experiences:
            self.buffer.add(exp.state, exp.action, exp.reward, exp.next_state, exp.done)
    
    def _generate_her_experiences(self, episode: List[Experience]) -> List[Experience]:
        """Generate HER experiences from episode"""
        her_experiences = []
        
        for t, experience in enumerate(episode):
            # Sample goals based on strategy
            if self.strategy == "future":
                # Sample from future states in the episode
                future_indices = list(range(t + 1, len(episode)))
                goal_indices = random.sample(
                    future_indices, 
                    min(self.k, len(future_indices))
                )
            elif self.strategy == "episode":
                # Sample from any state in the episode
                goal_indices = random.sample(
                    range(len(episode)), 
                    min(self.k, len(episode))
                )
            else:
                goal_indices = []
            
            for goal_idx in goal_indices:
                goal_state = episode[goal_idx].next_state
                
                # Create new experience with alternative goal
                new_reward = self._compute_reward(experience.next_state, goal_state)
                new_done = self._is_success(experience.next_state, goal_state)
                
                her_exp = Experience(
                    state=self._add_goal_to_state(experience.state, goal_state),
                    action=experience.action,
                    reward=new_reward,
                    next_state=self._add_goal_to_state(experience.next_state, goal_state),
                    done=new_done
                )
                
                her_experiences.append(her_exp)
        
        return her_experiences
    
    def _compute_reward(self, achieved_state: np.ndarray, goal_state: np.ndarray) -> float:
        """Compute reward for achieving goal"""
        # For Snake game, goal could be reaching a certain position
        # This is a simplified implementation
        distance = np.linalg.norm(achieved_state[:2] - goal_state[:2])
        return 1.0 if distance < 0.1 else -0.1
    
    def _is_success(self, achieved_state: np.ndarray, goal_state: np.ndarray) -> bool:
        """Check if goal was achieved"""
        distance = np.linalg.norm(achieved_state[:2] - goal_state[:2])
        return distance < 0.1
    
    def _add_goal_to_state(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Add goal information to state"""
        # Concatenate goal to state
        return np.concatenate([state, goal[:2]])

class CuriosityDrivenReplay:
    """Curiosity-driven experience replay for exploration"""
    
    def __init__(self, buffer: PrioritizedReplayBuffer, curiosity_weight: float = 0.1):
        self.buffer = buffer
        self.curiosity_weight = curiosity_weight
        self.state_visits = {}
        self.prediction_errors = deque(maxlen=10000)
    
    def add_with_curiosity(self, state: np.ndarray, action: int, reward: float,
                          next_state: np.ndarray, done: bool, prediction_error: float = 0.0):
        """Add experience with curiosity bonus"""
        
        # Calculate intrinsic reward based on state novelty
        state_key = self._state_to_key(state)
        visit_count = self.state_visits.get(state_key, 0)
        self.state_visits[state_key] = visit_count + 1
        
        # Curiosity bonus decreases with visit count
        novelty_bonus = 1.0 / (1.0 + visit_count)
        
        # Prediction error bonus
        self.prediction_errors.append(prediction_error)
        normalized_error = prediction_error / (np.std(self.prediction_errors) + 1e-8)
        prediction_bonus = min(normalized_error, 5.0)  # Cap the bonus
        
        # Combined intrinsic reward
        intrinsic_reward = self.curiosity_weight * (novelty_bonus + prediction_bonus)
        total_reward = reward + intrinsic_reward
        
        self.buffer.add(state, action, total_reward, next_state, done)
    
    def _state_to_key(self, state: np.ndarray) -> tuple:
        """Convert state to hashable key"""
        # Discretize state for counting
        discretized = np.round(state * 10).astype(int)
        return tuple(discretized)

class MultiStepReplayBuffer:
    """Multi-step experience replay buffer"""
    
    def __init__(self, buffer: PrioritizedReplayBuffer, n_step: int = 3, gamma: float = 0.99):
        self.buffer = buffer
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)
    
    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool):
        """Add experience with n-step returns"""
        
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) == self.n_step:
            # Calculate n-step return
            n_step_reward = 0
            gamma_power = 1
            
            for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
                n_step_reward += gamma_power * r
                gamma_power *= self.gamma
                
                if d:  # Episode ended
                    break
            
            # Get first state and last next_state
            first_state, first_action, _, _, _ = self.n_step_buffer[0]
            _, _, _, last_next_state, last_done = self.n_step_buffer[-1]
            
            # Add to main buffer
            self.buffer.add(first_state, first_action, n_step_reward, last_next_state, last_done)

class DiversityReplayBuffer:
    """Replay buffer that promotes diverse experiences"""
    
    def __init__(self, capacity: int, diversity_weight: float = 0.1):
        self.capacity = capacity
        self.diversity_weight = diversity_weight
        self.experiences = deque(maxlen=capacity)
        self.state_embeddings = []
        
    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool):
        """Add experience with diversity consideration"""
        
        experience = Experience(state, action, reward, next_state, done)
        
        # Calculate diversity score
        diversity_score = self._calculate_diversity(state)
        
        # Weighted random insertion
        if len(self.experiences) < self.capacity:
            self.experiences.append(experience)
            self.state_embeddings.append(state)
        else:
            # Replace experience with low diversity
            replace_prob = diversity_score * self.diversity_weight
            if random.random() < replace_prob:
                idx = random.randint(0, len(self.experiences) - 1)
                self.experiences[idx] = experience
                self.state_embeddings[idx] = state
    
    def _calculate_diversity(self, state: np.ndarray) -> float:
        """Calculate diversity score for state"""
        if not self.state_embeddings:
            return 1.0
        
        # Calculate minimum distance to existing states
        distances = [
            np.linalg.norm(state - existing_state) 
            for existing_state in self.state_embeddings
        ]
        
        min_distance = min(distances)
        
        # Normalize to [0, 1]
        diversity_score = min(min_distance / 10.0, 1.0)
        
        return diversity_score
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample diverse batch"""
        if len(self.experiences) < batch_size:
            return list(self.experiences)
        
        return random.sample(list(self.experiences), batch_size)
    
    def __len__(self) -> int:
        return len(self.experiences)

class AdaptiveReplayBuffer:
    """Adaptive replay buffer that adjusts sampling based on learning progress"""
    
    def __init__(self, buffer: PrioritizedReplayBuffer):
        self.buffer = buffer
        self.learning_progress = deque(maxlen=1000)
        self.adaptation_rate = 0.1
        
    def add_learning_signal(self, loss: float, q_value_change: float):
        """Add learning progress signal"""
        progress = abs(q_value_change) / (loss + 1e-8)
        self.learning_progress.append(progress)
    
    def adaptive_sample(self, batch_size: int) -> Tuple:
        """Sample batch with adaptive strategy"""
        
        if not self.learning_progress:
            return self.buffer.sample(batch_size)
        
        # Calculate recent learning progress
        recent_progress = np.mean(list(self.learning_progress)[-100:])
        
        # Adjust beta based on learning progress
        if recent_progress > 1.0:  # Good learning
            beta = min(1.0, self.buffer.beta + self.adaptation_rate)
        else:  # Poor learning
            beta = max(0.1, self.buffer.beta - self.adaptation_rate)
        
        self.buffer.beta = beta
        
        return self.buffer.sample(batch_size, beta)

class EpisodeReplayBuffer:
    """Buffer for storing and sampling complete episodes"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.episodes = deque(maxlen=capacity)
        
    def add_episode(self, episode: List[Experience]):
        """Add complete episode"""
        self.episodes.append(episode)
    
    def sample_episodes(self, num_episodes: int) -> List[List[Experience]]:
        """Sample complete episodes"""
        if len(self.episodes) < num_episodes:
            return list(self.episodes)
        
        return random.sample(list(self.episodes), num_episodes)
    
    def sample_sequences(self, batch_size: int, sequence_length: int) -> List[List[Experience]]:
        """Sample sequences from episodes"""
        sequences = []
        
        for _ in range(batch_size):
            if not self.episodes:
                break
                
            episode = random.choice(self.episodes)
            
            if len(episode) >= sequence_length:
                start_idx = random.randint(0, len(episode) - sequence_length)
                sequence = episode[start_idx:start_idx + sequence_length]
                sequences.append(sequence)
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.episodes)
