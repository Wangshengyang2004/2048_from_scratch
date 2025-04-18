import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from typing import Optional, Dict, List, Tuple, Union, Any

from agent.utils.networks import get_policy_network


class REINFORCEAgent:
    """
    REINFORCE agent with baseline for 2048 game.
    """
    
    def __init__(self,
                 state_dim: Tuple[int, int, int] = (12, 4, 4),
                 action_dim: int = 4,
                 hidden_channels: int = 128,
                 num_blocks: int = 4,
                 fc_size: int = 512,
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 jit_compile: bool = True):
        """
        Initialize the REINFORCE agent
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Initialize policy network
        self.policy_network = get_policy_network(
            jit_compile=jit_compile,
            input_channels=state_dim[0],
            board_size=state_dim[1],
            num_actions=action_dim,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            fc_size=fc_size
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Initialize episode buffer for storing trajectories
        self.reset_episode_buffer()
        
        # Initialize training metrics
        self.train_steps = 0
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
    
    def reset_episode_buffer(self) -> None:
        """Reset the episode buffer for a new episode"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    @torch.no_grad()
    def select_action(self, 
                      state: torch.Tensor, 
                      valid_actions: Optional[torch.Tensor] = None,
                      training: bool = True) -> torch.Tensor:
        """
        Select an action based on the current state
        
        Args:
            state: Current state tensor [channels, height, width]
            valid_actions: Tensor of valid actions [action_space] or None
            training: Whether the agent is in training mode
            
        Returns:
            Selected action as a tensor
        """
        # Add batch dimension if needed
        if state.dim() == 3:
            state = state.unsqueeze(0)
        
        # Get policy logits and value from the network
        policy_logits, value = self.policy_network(state)
        
        if valid_actions is not None and valid_actions.sum() > 0:
            # Apply mask for valid actions
            invalid_mask = ~valid_actions
            policy_logits[0][invalid_mask] = float('-inf')
        
        # Calculate action probabilities
        probs = F.softmax(policy_logits, dim=-1)
        
        if training:
            # Sample action from the distribution
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            
            # Store episode data if training
            self.states.append(state.squeeze(0))
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(value.squeeze(-1))
            
            return action
        else:
            # Choose action with highest probability during evaluation
            return torch.argmax(probs, dim=-1)
    
    def store_reward(self, reward: float, done: bool) -> None:
        """
        Store a reward and done flag from the environment
        
        Args:
            reward: Reward received
            done: Whether the episode ended
        """
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self) -> Dict[str, float]:
        """
        Update the policy network parameters based on the collected trajectory
        
        Returns:
            Dictionary of training metrics
        """
        # Return zeros if the episode buffer is empty
        if len(self.states) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        # Convert episode data to tensors
        states = torch.stack(self.states)
        actions = torch.cat(self.actions)
        log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.values)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones, dtype=torch.bool, device=self.device)
        
        # Calculate returns with discount
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate advantage (returns - values)
        advantages = returns - values
        
        # Calculate policy loss (negative for gradient ascent)
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Calculate value loss (MSE)
        value_loss = F.mse_loss(values, returns)
        
        # Forward pass to get current policy distribution
        policy_logits, _ = self.policy_network(states)
        probs = F.softmax(policy_logits, dim=-1)
        
        # Calculate entropy bonus
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        
        # Calculate total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Optimize the network
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # Log metrics
        self.train_steps += 1
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        self.entropies.append(entropy.item())
        
        # Clear episode buffer
        self.reset_episode_buffer()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item()
        }
    
    def save(self, path: str) -> None:
        """Save the agent state to a file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model, optimizer and training state
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_steps': self.train_steps,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropies': self.entropies,
            'args': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef
            }
        }, path)
    
    def load(self, path: str) -> None:
        """Load the agent state from a file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model parameters
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.train_steps = checkpoint['train_steps']
        self.policy_losses = checkpoint['policy_losses']
        self.value_losses = checkpoint['value_losses']
        self.entropies = checkpoint['entropies']
        
        # Load and verify args
        args = checkpoint['args']
        assert args['state_dim'] == self.state_dim, "State dimensions don't match"
        assert args['action_dim'] == self.action_dim, "Action dimensions don't match"
        
        # Update parameters from saved args if needed
        self.gamma = args['gamma']
        self.entropy_coef = args['entropy_coef']
        self.value_coef = args['value_coef'] 