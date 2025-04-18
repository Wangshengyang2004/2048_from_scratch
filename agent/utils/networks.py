import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional, Union


class ConvBlock(nn.Module):
    """
    Convolutional block with batch normalization and activation
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3, 
                 stride: int = 1, 
                 padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block for the neural network
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class Policy2048Network(nn.Module):
    """
    Neural network architecture for 2048 policy-based agents.
    This network outputs both policy logits and value estimates.
    """
    def __init__(self, 
                 input_channels: int = 12, 
                 board_size: int = 4, 
                 num_actions: int = 4,
                 hidden_channels: int = 128,
                 num_blocks: int = 4,
                 fc_size: int = 512):
        super().__init__()
        
        # Input preprocessing layer
        self.conv_in = ConvBlock(input_channels, hidden_channels)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_channels) for _ in range(num_blocks)]
        )
        
        # Shared feature extraction
        self.conv_out = ConvBlock(hidden_channels, 64)
        self.flatten_size = 64 * board_size * board_size
        
        # Policy head
        self.policy_fc1 = nn.Linear(self.flatten_size, fc_size)
        self.policy_fc2 = nn.Linear(fc_size, num_actions)
        
        # Value head
        self.value_fc1 = nn.Linear(self.flatten_size, fc_size)
        self.value_fc2 = nn.Linear(fc_size, 1)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            x: The input tensor representing the state [batch_size, channels, height, width]
            
        Returns:
            policy_logits: The action probabilities [batch_size, num_actions]
            value: The state value estimates [batch_size, 1]
        """
        # Process input
        x = self.conv_in(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Shared feature extraction
        x = self.conv_out(x)
        x = x.view(-1, self.flatten_size)
        
        # Policy head
        policy = F.relu(self.policy_fc1(x))
        policy_logits = self.policy_fc2(policy)
        
        # Value head
        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)
        
        return policy_logits, value


class DQN2048Network(nn.Module):
    """
    Neural network architecture for 2048 DQN-based agents.
    This network outputs Q-values for each action.
    """
    def __init__(self, 
                 input_channels: int = 12, 
                 board_size: int = 4, 
                 num_actions: int = 4,
                 hidden_channels: int = 128,
                 num_blocks: int = 4,
                 fc_size: int = 512,
                 dueling: bool = False):
        super().__init__()
        
        self.dueling = dueling
        
        # Input preprocessing layer
        self.conv_in = ConvBlock(input_channels, hidden_channels)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_channels) for _ in range(num_blocks)]
        )
        
        # Feature extraction
        self.conv_out = ConvBlock(hidden_channels, 64)
        self.flatten_size = 64 * board_size * board_size
        
        if dueling:
            # Advantage stream
            self.advantage_fc1 = nn.Linear(self.flatten_size, fc_size)
            self.advantage_fc2 = nn.Linear(fc_size, num_actions)
            
            # Value stream
            self.value_fc1 = nn.Linear(self.flatten_size, fc_size)
            self.value_fc2 = nn.Linear(fc_size, 1)
        else:
            # Standard DQN layers
            self.fc1 = nn.Linear(self.flatten_size, fc_size)
            self.fc2 = nn.Linear(fc_size, fc_size)
            self.fc3 = nn.Linear(fc_size, num_actions)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: The input tensor representing the state [batch_size, channels, height, width]
            
        Returns:
            q_values: Q-values for each action [batch_size, num_actions]
        """
        # Process input
        x = self.conv_in(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Feature extraction
        x = self.conv_out(x)
        x = x.view(-1, self.flatten_size)
        
        if self.dueling:
            # Advantage stream
            advantage = F.relu(self.advantage_fc1(x))
            advantage = self.advantage_fc2(advantage)
            
            # Value stream
            value = F.relu(self.value_fc1(x))
            value = self.value_fc2(value)
            
            # Combine value and advantage
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            # Standard DQN forward
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            q_values = self.fc3(x)
        
        return q_values


def get_policy_network(jit_compile: bool = False, **kwargs) -> nn.Module:
    """
    Get a policy network instance
    
    Args:
        jit_compile: Whether to JIT compile the network
        **kwargs: Arguments to pass to the network constructor
    
    Returns:
        A policy network instance
    """
    if jit_compile:
        # Create JIT-compiled network with custom parameters
        network = Policy2048Network(**kwargs)
        return torch.jit.script(network)
    else:
        return Policy2048Network(**kwargs)


def get_dqn_network(dueling: bool = False, jit_compile: bool = False, **kwargs) -> nn.Module:
    """
    Get a DQN network instance
    
    Args:
        dueling: Whether to use a dueling network architecture
        jit_compile: Whether to JIT compile the network
        **kwargs: Arguments to pass to the network constructor
    
    Returns:
        A DQN network instance
    """
    if jit_compile:
        # Create JIT-compiled network with custom parameters
        network = DQN2048Network(dueling=dueling, **kwargs)
        return torch.jit.script(network)
    else:
        return DQN2048Network(dueling=dueling, **kwargs) 