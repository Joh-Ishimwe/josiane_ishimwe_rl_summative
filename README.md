
# Agrika: Smart Agricultural Fleet Management with Reinforcement Learning

## Project Overview
This project implements a sophisticated reinforcement learning system for managing agricultural tractor fleets in Rwanda. The system optimizes maintenance scheduling, resource allocation, and operational decisions to maximize productivity while minimizing costs and equipment downtime.

## Environment Features
- **Rich State Space**: 15-dimensional observation including tractor conditions, weather forecasts, and seasonal demands
- **Exhaustive Action Space**: 27 discrete actions covering all possible fleet management combinations
- **3D Visualization**: Professional PyBullet-based rendering with real-time state updates
- **Realistic Dynamics**: Weather effects, seasonal patterns, and breakdown probabilities

## Implemented Algorithms
1. **Deep Q-Network (DQN)** - Value-based learning
2. **Proximal Policy Optimization (PPO)** - Policy gradient method
3. **Advantage Actor-Critic (A2C)** - Actor-critic method  
4. **REINFORCE** - Basic policy gradient approach

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Train all models
python main.py --mode train --algorithm all

# Train specific algorithm
python main.py --mode train --algorithm dqn

# Run 3D visualization
python main.py --mode visualize
```

## Project Structure
```
josiane_ishimwe_rl_summative/
├── environment/          # Custom environment implementation
├── training/            # Training scripts for all algorithms
├── models/              # Saved model files
├── logs/                # Training logs and tensorboard data
└── main.py              # Main entry point
```

## Results
[Add your results after training]

## Author
Josiane Ishimwe - Software Engineering Student, African Leadership University