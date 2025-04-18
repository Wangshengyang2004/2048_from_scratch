# 2048 Reinforcement Learning Environment

This project implements a gym-compatible 2048 game environment with PyTorch and Pygame. It includes multiple reinforcement learning algorithms to solve the 2048 game.

## Features

- **Efficient 2048 Game Environment**: Pure PyTorch implementation for high performance
- **Pygame Renderer**: Visual display of the game
- **Vectorized Environment**: For parallel training
- **JIT Compilation**: For optimized performance
- **Multiple RL Algorithms**:
  - Random policy (baseline)
  - Deep Q-Network (DQN)
  - Double DQN
  - Dueling DQN
  - REINFORCE with baseline (Policy Gradient)
  - Proximal Policy Optimization (PPO)

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Pygame
- Gym
- Matplotlib
- NumPy

## Installation

```bash
pip install torch numpy pygame gym matplotlib
```

## Project Structure

```
.
├── env/                 # Game environment
│   ├── game_2048.py     # 2048 game logic
│   ├── renderer.py      # Pygame renderer
│   └── env_2048.py      # Gym-compatible environment
├── agent/               # Agent implementations
│   ├── random/          # Random agent
│   ├── policies/        # RL policies (DQN, REINFORCE, PPO)
│   └── utils/           # Utilities (networks, replay buffer)
├── train.py             # Training script
├── eval.py              # Evaluation script
└── eval_all.py          # Comparative evaluation
```

## Usage

### Manual Play

You can play the game manually to test the environment:

```bash
python -c "from env import Env2048; env = Env2048(render_mode='human'); env.manual_play()"
```

### Training

Train an agent using the `train.py` script:

```bash
# Train a DQN agent
python train.py --agent dqn --num_episodes 1000 --output_dir results

# Train a PPO agent
python train.py --agent ppo --num_episodes 1000 --output_dir results

# Train with rendering
python train.py --agent reinforce --render
```

#### Options

- `--agent`: Agent type (random, dqn, ddqn, dueling_dqn, reinforce, ppo)
- `--num_episodes`: Number of episodes to train for
- `--max_steps`: Maximum steps per episode
- `--render`: Render the environment during training
- `--device`: Device to use (cuda or cpu)
- `--vector_env`: Use vectorized environment (DQN only)
- `--num_envs`: Number of environments in vectorized env
- `--output_dir`: Directory to save results

### Evaluation

Evaluate a trained agent using the `eval.py` script:

```bash
# Evaluate a trained DQN model
python eval.py --agent dqn --model_path results/dqn/dqn_model_final.pt --num_episodes 10

# Render the evaluation
python eval.py --agent ppo --model_path results/ppo/policy_model_final.pt --render
```

### Comparing Agents

Compare all trained agents using the `eval_all.py` script:

```bash
python eval_all.py --model_dir results --num_episodes 20
```

This will evaluate all trained agents and generate comparison plots in the `comparison_results` directory.

## Custom Implementation

The project is designed to be extensible. You can:

1. Add new RL algorithms by creating new agent classes
2. Modify the environment parameters in `env/game_2048.py`
3. Customize the neural network architecture in `agent/utils/networks.py`

## Performance Tips

- Use CUDA device for faster training if available
- Enable JIT compilation for neural networks
- Use the vectorized environment for DQN-based agents

## License

MIT 