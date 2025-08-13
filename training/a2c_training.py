import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, PPO  # Added PPO for commented section
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'environment')))
from agrika_env import EnhancedAgrikaTractorFleetEnv  # Updated to Enhanced environment


class TrainingCallback(BaseCallback):
    """Custom callback to track training metrics and detect convergence"""
    def __init__(self, log_dir, window_size=100, reward_threshold=200, std_threshold=40):
        super().__init__(verbose=1)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.window_size = window_size
        self.reward_threshold = reward_threshold
        self.std_threshold = std_threshold
        self.episode_count = 0
        os.makedirs(log_dir, exist_ok=True)

    def _on_step(self):
        # Handle vectorized environments
        dones = self.locals.get('dones', [False])
        infos = self.locals.get('infos', [{}])
        
        # Check if any environment finished an episode
        for i, (done, info) in enumerate(zip(dones, infos)):
            if done and 'episode' in info:
                self.episode_count += 1
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Check convergence
                if len(self.episode_rewards) >= self.window_size:
                    mean_reward = np.mean(self.episode_rewards[-self.window_size:])
                    std_reward = np.std(self.episode_rewards[-self.window_size:])
                    
                    # Log metrics
                    with open(f"{self.log_dir}/training_metrics.txt", "a") as f:
                        f.write(f"Episode {self.episode_count}: Mean Reward = {mean_reward:.2f}, "
                               f"Std = {std_reward:.2f}, Length = {episode_length}\n")
                    
                    # Check for convergence
                    if mean_reward >= self.reward_threshold and std_reward <= self.std_threshold:
                        print(f"Converged at episode {self.episode_count}: "
                              f"Mean Reward = {mean_reward:.2f}, Std = {std_reward:.2f}")
                        return False  # Stop training
        
        return True

def train_a2c_optimized():
    """Train A2C with optimized hyperparameters"""
    # Create directories
    os.makedirs("models/pg/a2c", exist_ok=True)
    os.makedirs("logs/a2c", exist_ok=True)
    
    print("🚀 Starting A2C Training with Optimized Hyperparameters")
    
    # Create vectorized environment for better sample efficiency
    def make_env():
        def _init():
            env = EnhancedAgrikaTractorFleetEnv()  # Updated to Enhanced environment
            return env
        return _init
    
    # Use 4 parallel environments
    n_envs = 4
    env = make_vec_env(make_env, n_envs=n_envs)
    env = Monitor(env, filename="logs/a2c/monitor.csv")
    
    # A2C with optimized hyperparameters
    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,      # Slightly higher than PPO for faster learning
        n_steps=256,               # Shorter rollouts than PPO for on-policy updates
        gamma=0.99,                # High discount for long-term planning
        gae_lambda=1.0,            # No GAE smoothing for A2C (keeps it simple)
        ent_coef=0.01,             # Entropy regularization for exploration
        vf_coef=0.25,              # Lower value function coefficient
        max_grad_norm=0.5,         # Gradient clipping
        rms_prop_eps=1e-5,         # RMSprop epsilon
        use_rms_prop=True,         # Use RMSprop optimizer (typical for A2C)
        use_sde=False,             # No state-dependent exploration
        sde_sample_freq=-1,
        normalize_advantage=True,   # Normalize advantages
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Deep networks
            activation_fn=torch.nn.ReLU,
            ortho_init=True,
            log_std_init=0.0
        ),
        tensorboard_log="logs/a2c",
        verbose=1,
        device="auto"
    )
    
    print(f"A2C Model created with {n_envs} parallel environments")
    print(f"Policy network: {model.policy}")
    
    
    # Setup evaluation
    eval_env = Monitor(EnhancedAgrikaTractorFleetEnv(), filename="logs/a2c/eval_monitor.csv")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/pg/a2c/",
        log_path="logs/a2c/",
        eval_freq=10000,  # Evaluate every 10k steps
        deterministic=True,
        render=False,
        n_eval_episodes=10,
        verbose=1
    )
    
    # Setup training callback for convergence detection
    training_callback = TrainingCallback(log_dir="logs/a2c")
    
    # Train the model
    total_timesteps = 400000  # A2C typically needs more steps than PPO
    print(f"Training A2C for {total_timesteps} timesteps...")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, training_callback],
            progress_bar=True,
            tb_log_name="a2c_optimized"
        )
        
        # Save final model
        model.save("models/pg/a2c/final_a2c_model")
        
        # Generate learning curve
        plot_learning_curve("logs/a2c", "A2C Learning Curve", "models/pg/a2c/a2c_learning_curve.png")
        
        print("✅ A2C training completed!")
        
        # Final evaluation
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
        print(f"Final evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
        
        # Test generalization on different scenarios
        print("\n🧪 Testing generalization...")
        test_scenarios = [
            {'max_weather_changes': 2, 'breakdown_multiplier': 0.5},
            {'max_weather_changes': 8, 'breakdown_multiplier': 1.5}
        ]
        
        for i, scenario in enumerate(test_scenarios):
            print(f"Scenario {i+1}: {scenario}")
            test_env = EnhancedAgrikaTractorFleetEnv()
            test_rewards = []
            for _ in range(10):
                try:
                    obs, info = test_env.reset(options=scenario)
                except TypeError:
                    obs, info = test_env.reset()
                    if hasattr(test_env, 'curriculum_params'):
                        test_env.curriculum_params = scenario
                
                episode_reward = 0
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                test_rewards.append(episode_reward)
            print(f"  Mean reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
            test_env.close()
        
        env.close()
        eval_env.close()
        return model
        
    except Exception as e:
        print(f"❌ A2C training failed: {str(e)}")
        env.close()
        eval_env.close()
        raise

def plot_learning_curve(log_dir, title, save_path):
    """Plot learning curve from training logs"""
    try:
        results = load_results(log_dir)
        if len(results) == 0:
            print(f"No data found in {log_dir}/monitor.csv")
            return
        
        x, y = ts2xy(results, 'timesteps')
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Episode rewards
        plt.subplot(2, 2, 1)
        plt.plot(x, y, alpha=0.6, color='blue', label="Episode Rewards")
        
        # Add moving average
        if len(y) > 50:
            window = min(100, len(y) // 10)
            moving_avg = np.convolve(y, np.ones(window)/window, mode='valid')
            x_avg = x[window-1:]
            plt.plot(x_avg, moving_avg, color='red', linewidth=2, label=f"Moving Average ({window})")
        
        plt.xlabel('Timesteps')
        plt.ylabel('Episode Reward')
        plt.title(f'{title} - Rewards')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Episode lengths
        plt.subplot(2, 2, 2)
        episode_lengths = results['l'].values
        plt.plot(range(len(episode_lengths)), episode_lengths, alpha=0.6, color='green')
        if len(episode_lengths) > 50:
            window = min(50, len(episode_lengths) // 10)
            moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(episode_lengths)), moving_avg, color='red', linewidth=2)
        plt.xlabel('Episodes')
        plt.ylabel('Episode Length')
        plt.title('Episode Lengths')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Reward distribution
        plt.subplot(2, 2, 3)
        plt.hist(y, bins=50, alpha=0.7, color='purple')
        plt.xlabel('Episode Reward')
        plt.ylabel('Frequency')
        plt.title('Reward Distribution')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Training progress (reward vs episodes)
        plt.subplot(2, 2, 4)
        episodes = range(len(y))
        plt.plot(episodes, y, alpha=0.6, color='orange')
        if len(y) > 50:
            window = min(50, len(y) // 10)
            moving_avg = np.convolve(y, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(y)), moving_avg, color='red', linewidth=2)
        plt.xlabel('Episodes')
        plt.ylabel('Episode Reward')
        plt.title('Training Progress')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Learning curve saved to {save_path}")
        
    except Exception as e:
        print(f"Error generating learning curve: {str(e)}")

if __name__ == "__main__":
    model = train_a2c_optimized()