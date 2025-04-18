import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from typing import Optional, Dict, List, Tuple, Union, Any

from agent.utils.networks import get_policy_network


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for 2048 game.
    """
    
    def __init__(self,
                 state_dim: Tuple[int, int, int] = (12, 4, 4),
                 action_dim: int = 4,
                 hidden_channels: int = 128,
                 num_blocks: int = 4,
                 fc_size: int = 512,
                 learning_rate: float = 0.0003,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 update_epochs: int = 4,
                 batch_size: int = 64,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 jit_compile: bool = True):
        """
        Initialize the PPO agent
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
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
        self.approx_kl_divs = []
        self.clip_fractions = []
    
    def reset_episode_buffer(self) -> None:
        """Reset the episode buffer for a new episode"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
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
    
    def _compute_gae(self, 
                     rewards: torch.Tensor, 
                     values: torch.Tensor, 
                     dones: torch.Tensor,
                     next_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: Rewards tensor [T]
            values: Values tensor [T]
            dones: Dones tensor [T]
            next_value: Next value tensor [1]
            
        Returns:
            returns: Returns tensor [T]
            advantages: Advantages tensor [T]
        """
        T = len(rewards)
        advantages = torch.zeros(T, dtype=torch.float32, device=self.device)
        returns = torch.zeros(T, dtype=torch.float32, device=self.device)
        
        # Initialize with next_value (bootstrapped value of the final state)
        gae = torch.zeros(1, dtype=torch.float32, device=self.device)
        
        # Compute GAE in reverse order
        for t in reversed(range(T)):
            # If done, the next value is 0
            if t == T - 1:
                next_val = next_value * (~dones[t])
            else:
                next_val = values[t + 1] * (~dones[t])
            
            # Compute delta (TD error)
            delta = rewards[t] + self.gamma * next_val - values[t]
            
            # Compute GAE
            gae = delta + self.gamma * self.gae_lambda * (~dones[t]) * gae
            
            # Store advantage and return
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return returns, advantages
    
    def update(self) -> Dict[str, float]:
        """
        Update the policy network parameters based on the collected trajectory
        
        Returns:
            Dictionary of training metrics
        """
        # Return zeros if the episode buffer is empty
        if len(self.states) == 0:
            return {
                "policy_loss": 0.0, 
                "value_loss": 0.0, 
                "entropy": 0.0, 
                "approx_kl": 0.0,
                "clip_fraction": 0.0
            }
        
        # Convert episode data to tensors
        states = torch.stack(self.states)
        actions = torch.cat(self.actions)
        old_log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.values)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones, dtype=torch.bool, device=self.device)
        
        # Compute returns and advantages
        with torch.no_grad():
            # Get the value of the last state (next_value)
            if len(self.states) > 0:
                _, next_value = self.policy_network(self.states[-1].unsqueeze(0))
                next_value = next_value.squeeze(-1)
            else:
                next_value = torch.zeros(1, device=self.device)
            
            returns, advantages = self._compute_gae(rewards, values, dones, next_value)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        total_clip_fraction = 0
        
        # Prepare data indices
        indices = np.arange(len(states))
        
        for epoch in range(self.update_epochs):
            # Shuffle indices
            np.random.shuffle(indices)
            
            # Create mini-batches
            for start_idx in range(0, len(indices), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # Create batch tensors
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                policy_logits, value_preds = self.policy_network(batch_states)
                
                # Calculate new log probabilities
                probs = F.softmax(policy_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                
                # Calculate entropy
                entropy = dist.entropy().mean()
                
                # Calculate policy ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_advantages
                
                # Calculate policy loss (negative for gradient ascent)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss (clipped)
                value_preds = value_preds.squeeze(-1)
                value_loss = F.mse_loss(value_preds, batch_returns)
                
                # Calculate total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Optimize the network
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
                
                # Calculate metrics
                with torch.no_grad():
                    # Approximate KL divergence
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    
                    # Clipping fraction
                    clip_fraction = ((ratio < 1.0 - self.clip_range) | (ratio > 1.0 + self.clip_range)).float().mean().item()
                
                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_approx_kl += approx_kl
                total_clip_fraction += clip_fraction
        
        # Calculate average metrics
        n_batches = len(indices) // self.batch_size + (1 if len(indices) % self.batch_size != 0 else 0)
        n_updates = n_batches * self.update_epochs
        
        avg_policy_loss = total_policy_loss / n_updates
        avg_value_loss = total_value_loss / n_updates
        avg_entropy = total_entropy / n_updates
        avg_approx_kl = total_approx_kl / n_updates
        avg_clip_fraction = total_clip_fraction / n_updates
        
        # Log metrics
        self.train_steps += 1
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropies.append(avg_entropy)
        self.approx_kl_divs.append(avg_approx_kl)
        self.clip_fractions.append(avg_clip_fraction)
        
        # Clear episode buffer
        self.reset_episode_buffer()
        
        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "approx_kl": avg_approx_kl,
            "clip_fraction": avg_clip_fraction
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
            'approx_kl_divs': self.approx_kl_divs,
            'clip_fractions': self.clip_fractions,
            'args': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_range': self.clip_range,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef,
                'max_grad_norm': self.max_grad_norm,
                'update_epochs': self.update_epochs,
                'batch_size': self.batch_size
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
        self.approx_kl_divs = checkpoint.get('approx_kl_divs', [])
        self.clip_fractions = checkpoint.get('clip_fractions', [])
        
        # Load and verify args
        args = checkpoint['args']
        assert args['state_dim'] == self.state_dim, "State dimensions don't match"
        assert args['action_dim'] == self.action_dim, "Action dimensions don't match"
        
        # Update parameters from saved args if needed
        self.gamma = args['gamma']
        self.gae_lambda = args['gae_lambda']
        self.clip_range = args['clip_range']
        self.entropy_coef = args['entropy_coef']
        self.value_coef = args['value_coef']
        self.max_grad_norm = args['max_grad_norm']
        self.update_epochs = args['update_epochs']
        self.batch_size = args['batch_size'] 