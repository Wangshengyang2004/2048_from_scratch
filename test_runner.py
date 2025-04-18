import os
import sys
import time
import torch
from env import Game2048, Env2048, PygameRenderer
from agent import RandomAgent

print("Testing 2048 RL Environment")
print("==========================")

# Check PyTorch availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Test game logic
print("\nTesting game logic...")
game = Game2048()
print(f"Board size: {game.size}x{game.size}")
print(f"Initial board:\n{game.board}")

# Test actions
print("\nTesting actions...")
actions = ["UP", "RIGHT", "DOWN", "LEFT"]
for i, action_name in enumerate(actions):
    print(f"Taking action: {action_name}")
    board, reward, done, info = game.step(i)
    print(f"Reward: {reward}, Done: {done}")
    print(f"Max tile: {info['max_tile']}")
    print(f"Board:\n{board}")

# Test gym environment
print("\nTesting gym environment...")
env = Env2048()
state, _ = env.reset()
print(f"State shape: {state.shape}")

# Test random agent
print("\nTesting random agent...")
agent = RandomAgent()
for i in range(5):
    action = agent.select_action(state, env.get_valid_actions())
    state, reward, done, truncated, info = env.step(action)
    print(f"Step {i+1}: Action: {action.item()}, Reward: {reward:.2f}, Max tile: {info['max_tile']}")

# Test pygame renderer if available
try:
    print("\nTesting pygame renderer (for 3 seconds)...")
    renderer = PygameRenderer(game)
    renderer.render()
    print("Pygame renderer is working. Waiting 3 seconds...")
    time.sleep(3)
    renderer.close()
except Exception as e:
    print(f"Error testing pygame renderer: {e}")

print("\nAll tests completed!")
print("To run the environment with manual control, use:")
print("    python -c \"from env import Env2048; env = Env2048(render_mode='human'); env.manual_play()\"")
print("\nTo train an agent, use:")
print("    python train.py --agent dqn --num_episodes 100")
print("\nTo evaluate an agent, use:")
print("    python eval.py --agent dqn --model_path results/dqn/dqn_model_final.pt --render") 