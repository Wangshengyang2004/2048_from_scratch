import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import deque, namedtuple

# Define a named tuple for experiences
Experience = namedtuple('Experience', 
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """
    Replay buffer for storing and sampling experience tuples
    """
    
    def __init__(self, capacity: int, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the buffer
        
        Args:
            capacity: Maximum number of experiences to store in the buffer
            device: Device to store tensors on
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.device = device
    
    def add(self, 
            state: torch.Tensor, 
            action: int, 
            reward: float, 
            next_state: torch.Tensor, 
            done: bool) -> None:
        """
        Add a new experience to the buffer
        
        Args:
            state: Current state tensor
            action: Action taken
            reward: Reward received
            next_state: Next state tensor
            done: Whether the episode ended
        """
        # Create the experience tuple and add it to the buffer
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of experiences from the buffer
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        # Sample random indices
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # Get the experiences at the sampled indices
        experiences = [self.buffer[i] for i in indices]
        
        # Stack the experiences into batches
        states = torch.stack([e.state for e in experiences]).to(self.device)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.int64).to(self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32).to(self.device)
        next_states = torch.stack([e.next_state for e in experiences]).to(self.device)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.bool).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return the current size of the buffer"""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for DQN
    """
    
    def __init__(self, 
                 capacity: int, 
                 alpha: float = 0.6, 
                 beta: float = 0.4, 
                 beta_increment: float = 0.001,
                 epsilon: float = 1e-6,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the buffer
        
        Args:
            capacity: Maximum number of experiences to store in the buffer
            alpha: Priority exponent parameter
            beta: Importance sampling correction exponent
            beta_increment: Increment to beta per sampling
            epsilon: Small constant added to priorities to avoid zero priority
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.device = device
        
        # Initialize the buffer
        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add(self, 
            state: torch.Tensor, 
            action: int, 
            reward: float, 
            next_state: torch.Tensor, 
            done: bool) -> None:
        """
        Add a new experience to the buffer
        
        Args:
            state: Current state tensor
            action: Action taken
            reward: Reward received
            next_state: Next state tensor
            done: Whether the episode ended
        """
        # Create the experience tuple
        experience = Experience(state, action, reward, next_state, done)
        
        # Get max priority for new experience
        max_priority = np.max(self.priorities) if self.size > 0 else 1.0
        
        # Add experience to buffer
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            # Replace the existing experience at the current position
            self.buffer[self.position] = experience
        
        # Update priority
        self.priorities[self.position] = max_priority
        
        # Increment position
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of experiences from the buffer
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        # If buffer is not full, sample from available experiences
        sample_size = min(batch_size, self.size)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= np.sum(probs)
        
        # Sample indices based on priorities
        indices = np.random.choice(self.size, sample_size, replace=False, p=probs)
        
        # Get the experiences at the sampled indices
        experiences = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** -self.beta
        weights /= np.max(weights)  # Normalize weights
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Stack the experiences into batches
        states = torch.stack([e.state for e in experiences]).to(self.device)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.int64).to(self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32).to(self.device)
        next_states = torch.stack([e.next_state for e in experiences]).to(self.device)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.bool).to(self.device)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities of sampled transitions
        
        Args:
            indices: Indices of sampled experiences
            td_errors: TD errors of sampled experiences
        """
        # Calculate new priorities (add epsilon to avoid zero priority)
        priorities = np.abs(td_errors) + self.epsilon
        
        # Update priorities for each experience
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self) -> int:
        """Return the current size of the buffer"""
        return self.size 