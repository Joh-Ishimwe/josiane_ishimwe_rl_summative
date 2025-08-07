# ============================================================================
# FILE: main.py
# ============================================================================

import argparse
from training.dqn_training import train_dqn
from training.ppo_training import train_ppo
from training.a2c_training import train_a2c
from training.reinforce_training import train_reinforce
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from environment.agrika_env import AgrikaTractorFleetEnv
from environment.agrika_3d_viz import Agrika3DVisualizer
import os

def main():
    parser = argparse.ArgumentParser(description="Agrika RL Training and Evaluation")
    parser.add_argument("--mode", choices=["train", "evaluate", "visualize"], 
                       default="train", help="Mode to run")
    parser.add_argument("--algorithm", choices=["dqn", "ppo", "a2c", "reinforce", "all"],
                       default="all", help="Algorithm to train/evaluate")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        if args.algorithm == "dqn" or args.algorithm == "all":
            print("=== Training DQN ===")
            train_dqn()
            
        if args.algorithm == "ppo" or args.algorithm == "all":
            print("=== Training PPO ===")
            train_ppo()
            
        if args.algorithm == "a2c" or args.algorithm == "all":
            print("=== Training A2C ===")
            train_a2c()
            
        if args.algorithm == "reinforce" or args.algorithm == "all":
            print("=== Training REINFORCE ===")
            train_reinforce()
    
    elif args.mode == "visualize":
        print("Starting 3D visualization...")
        env = AgrikaTractorFleetEnv()
        visualizer = Agrika3DVisualizer(env)
        visualizer.initialize_visualization()
        
        # Run visualization with random actions
        obs, info = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            visualizer.render_frame(obs, action, reward, info)
            if terminated:
                break
        
        visualizer.close()

if __name__ == "__main__":
    main()