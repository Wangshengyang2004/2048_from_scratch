import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import time
from typing import Optional, Dict, List, Tuple, Union

from agent.utils.networks import get_dqn_network
from agent.utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent:
    """
    Deep Q-Network (DQN) agent for 2048 game.
    Implements both standard DQN and Double DQN variants.
    """
    
    def __init__(self, 
                 state_dim: Tuple[int, int, int] = (12, 4, 4),
                 action_dim: int = 4,
                 hidden_channels: int = 128,
                 num_blocks: int = 4,
                 fc_size: int = 512,
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.999,
                 target_update_freq: int = 1000,
                 batch_size: int = 128,
                 buffer_size: int = 100000,
                 double_dqn: bool = False,
                 dueling: bool = False,
                 prioritized_replay: bool = False,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 per_epsilon: float = 1e-6,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 jit_compile: bool = True):
        """
        Initialize the DQN agent
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.double_dqn = double_dqn
        self.dueling = dueling
        self.prioritized_replay = prioritized_replay
        
        # Initialize Q-networks
        self.q_network = get_dqn_network(
            dueling=dueling,
            jit_compile=jit_compile,
            input_channels=state_dim[0],
            board_size=state_dim[1],
            num_actions=action_dim,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            fc_size=fc_size
        ).to(device)
        
        self.target_network = get_dqn_network(
            dueling=dueling,
            jit_compile=jit_compile,
            input_channels=state_dim[0],
            board_size=state_dim[1],
            num_actions=action_dim,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            fc_size=fc_size
        ).to(device)
        
        # Copy parameters from Q-network to target network
        self.update_target_network()
        
        # Freeze target network
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        if prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_size,
                alpha=alpha,
                beta=beta,
                epsilon=per_epsilon,
                device=device
            )
        else:
            self.replay_buffer = ReplayBuffer(
                capacity=buffer_size,
                device=device
            )
        
        # Initialize training metrics
        self.train_steps = 0
        self.losses = []
        self.q_values = []
    
    @torch.no_grad()
    def select_action(self, 
                      state: torch.Tensor, 
                      valid_actions: Optional[torch.Tensor] = None,
                      training: bool = True) -> torch.Tensor:
        """
        Select an action based on the current state using epsilon-greedy policy
        
        Args:
            state: Current state tensor [channels, height, width]
            valid_actions: Tensor of valid actions [action_space] or None
            training: Whether the agent is in training mode (uses epsilon-greedy) or evaluation mode
            
        Returns:
            Selected action as a tensor
        """
        # Add batch dimension if needed
        if state.dim() == 3:
            state = state.unsqueeze(0)
        
        if training and np.random.random() < self.epsilon:
            # Random action
            if valid_actions is not None and valid_actions.sum() > 0:
                # Sample from valid actions
                valid_indices = torch.nonzero(valid_actions, as_tuple=True)[0]
                random_idx = torch.randint(0, len(valid_indices), (1,), device=self.device)
                return valid_indices[random_idx]
            else:
                # Completely random action
                return torch.randint(0, self.action_dim, (1,), device=self.device)
        else:
            # Greedy action from Q-network
            q_values = self.q_network(state)
            
            if valid_actions is not None and valid_actions.sum() > 0:
                # Set invalid actions to very low values
                invalid_mask = ~valid_actions
                q_values[0][invalid_mask] = float('-inf')
            
            return torch.argmax(q_values, dim=1)
    
    @torch.no_grad()
    def batch_act(self, 
                  states: torch.Tensor, 
                  valid_actions_batch: Optional[torch.Tensor] = None,
                  training: bool = True) -> torch.Tensor:
        """
        Select actions for a batch of states
        
        Args:
            states: Batch of states [batch_size, channels, height, width]
            valid_actions_batch: Batch of valid action masks [batch_size, action_space]
            training: Whether the agent is in training mode
            
        Returns:
            Batch of selected actions [batch_size]
        """
        batch_size = states.shape[0]
        actions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        # Get Q-values for all states
        q_values = self.q_network(states)
        
        # For each state in the batch
        for i in range(batch_size):
            if training and np.random.random() < self.epsilon:
                # Random action
                if valid_actions_batch is not None and valid_actions_batch[i].sum() > 0:
                    # Sample from valid actions
                    valid_indices = torch.nonzero(valid_actions_batch[i], as_tuple=True)[0]
                    random_idx = torch.randint(0, len(valid_indices), (1,), device=self.device)
                    actions[i] = valid_indices[random_idx]
                else:
                    # Completely random action
                    actions[i] = torch.randint(0, self.action_dim, (1,), device=self.device)
            else:
                # Greedy action from Q-values
                if valid_actions_batch is not None and valid_actions_batch[i].sum() > 0:
                    # Set invalid actions to very low values
                    q_vals = q_values[i].clone()
                    invalid_mask = ~valid_actions_batch[i]
                    q_vals[invalid_mask] = float('-inf')
                    actions[i] = torch.argmax(q_vals)
                else:
                    actions[i] = torch.argmax(q_values[i])
        
        return actions
    
    def store_experience(self, 
                         state: torch.Tensor, 
                         action: torch.Tensor, 
                         reward: float, 
                         next_state: torch.Tensor, 
                         done: bool) -> None:
        """
        Store an experience in the replay buffer
        
        Args:
            state: Current state tensor
            action: Action taken
            reward: Reward received
            next_state: Next state tensor
            done: Whether the episode ended
        """
        # Extract scalar value if action is a tensor
        action_val = action.item() if isinstance(action, torch.Tensor) else action
        
        # Store in replay buffer
        self.replay_buffer.add(state, action_val, reward, next_state, done)
    
    def update(self) -> float:
        """
        Update the Q-network parameters
        
        Returns:
            The loss value
        """
        # Skip if not enough experiences in buffer
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        if self.prioritized_replay:
            # Sample with priorities
            states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)
        else:
            # Sample uniformly
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            weights = torch.ones_like(rewards)  # Equal weights
        
        # Calculate current Q-values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            # Calculate next Q-values from target network
            next_q_values = self.target_network(next_states)
            
            if self.double_dqn:
                # Double DQN: select actions using the Q-network, but evaluate using the target network
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q_values = next_q_values.gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN: select actions using the target network
                next_q_values = next_q_values.max(dim=1)[0]
            
            # Calculate target Q-values
            target_q_values = rewards + self.gamma * next_q_values * (~dones)
        
        # Calculate loss
        td_errors = target_q_values - q_values
        loss = (weights * td_errors.pow(2)).mean()
        
        # Update priorities in PER if used
        if self.prioritized_replay:
            self.replay_buffer.update_priorities(
                indices=indices.cpu().numpy(), 
                td_errors=td_errors.detach().cpu().numpy()
            )
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target network if it's time
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Log metrics
        self.losses.append(loss.item())
        self.q_values.append(q_values.mean().item())
        
        return loss.item()
    
    def update_target_network(self) -> None:
        """Update the target network with the parameters from the Q-network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path: str) -> None:
        """Save the agent state to a file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model, optimizer and training state
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_steps': self.train_steps,
            'epsilon': self.epsilon,
            'losses': self.losses,
            'q_values': self.q_values,
            'args': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'double_dqn': self.double_dqn,
                'dueling': self.dueling,
                'gamma': self.gamma,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'target_update_freq': self.target_update_freq,
                'batch_size': self.batch_size,
                'prioritized_replay': self.prioritized_replay
            }
        }, path)
    
    def load(self, path: str) -> None:
        """Load the agent state from a file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model parameters
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.train_steps = checkpoint['train_steps']
        self.epsilon = checkpoint['epsilon']
        self.losses = checkpoint['losses']
        self.q_values = checkpoint['q_values']
        
        # Load and verify args
        args = checkpoint['args']
        assert args['state_dim'] == self.state_dim, "State dimensions don't match"
        assert args['action_dim'] == self.action_dim, "Action dimensions don't match"
        
        # Update parameters from saved args if needed
        self.double_dqn = args['double_dqn']
        self.dueling = args['dueling']
        self.gamma = args['gamma']
        self.epsilon_end = args['epsilon_end']
        self.epsilon_decay = args['epsilon_decay']
        self.target_update_freq = args['target_update_freq']
        self.batch_size = args['batch_size']
        self.prioritized_replay = args['prioritized_replay']


class DoubleDQNAgent(DQNAgent):
    """Double DQN agent wrapper for convenience"""
    
    def __init__(self, **kwargs):
        super().__init__(double_dqn=True, **kwargs)


class DuelingDQNAgent(DQNAgent):
    """Dueling DQN agent wrapper for convenience"""
    
    def __init__(self, **kwargs):
        super().__init__(dueling=True, **kwargs) 