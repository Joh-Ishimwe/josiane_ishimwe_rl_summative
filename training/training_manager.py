# test_training_setup.py
# Test script to verify environment works with Stable Baselines3

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'environment')))
from agrika_env import AgrikaTractorFleetEnv



def test_environment_compatibility():
    """Test if your environment is compatible with Stable Baselines3"""
    
    print("=== Testing Environment Compatibility ===")
    
    # Create environment
    env = AgrikaTractorFleetEnv()
    
    # Check environment using SB3's checker
    try:
        check_env(env, warn=True)
        print("✅ Environment passed SB3 compatibility check!")
    except Exception as e:
        print(f"❌ Environment compatibility issue: {e}")
        return False
    
    # Test basic environment functions
    try:
        obs, info = env.reset()
        print(f"✅ Reset successful: obs shape {obs.shape}")
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Step successful: reward {reward:.2f}")
        
        print(f"✅ Action space: {env.action_space}")
        print(f"✅ Observation space: {env.observation_space}")
        
    except Exception as e:
        print(f"❌ Environment function error: {e}")
        return False
    
    return True

def test_algorithm_creation():
    """Test creating RL algorithms with your environment"""
    
    print("\n=== Testing Algorithm Creation ===")
    
    env = AgrikaTractorFleetEnv()
    
    # Test DQN
    try:
        dqn_model = DQN("MlpPolicy", env, verbose=0)
        print("✅ DQN model created successfully")
    except Exception as e:
        print(f"❌ DQN creation failed: {e}")
        return False
    
    # Test PPO
    try:
        ppo_model = PPO("MlpPolicy", env, verbose=0)
        print("✅ PPO model created successfully")
    except Exception as e:
        print(f"❌ PPO creation failed: {e}")
        return False
    
    # Test A2C
    try:
        a2c_model = A2C("MlpPolicy", env, verbose=0)
        print("✅ A2C model created successfully")
    except Exception as e:
        print(f"❌ A2C creation failed: {e}")
        return False
    
    return True

def test_short_training():
    """Test a very short training session"""
    
    print("\n=== Testing Short Training Session ===")
    
    env = AgrikaTractorFleetEnv()
    
    try:
        # Create DQN model
        model = DQN("MlpPolicy", env, verbose=1)
        
        # Train for just 1000 steps to test
        print("Training DQN for 1000 steps...")
        model.learn(total_timesteps=1000)
        
        # Test the trained model
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        print(f"✅ Training successful! Model predicts action: {action}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def main():
    """Run all tests"""
    
    print("🚀 Testing Agrika RL Setup")
    print("=" * 50)
    
    # Test 1: Environment compatibility
    if not test_environment_compatibility():
        print("❌ Environment compatibility test failed!")
        return
    
    # Test 2: Algorithm creation
    if not test_algorithm_creation():
        print("❌ Algorithm creation test failed!")
        return
    
    # Test 3: Short training
    if not test_short_training():
        print("❌ Training test failed!")
        return
    
    print("\n🎉 ALL TESTS PASSED!")
    print("Your Agrika environment is ready for full training!")
    print("\nNext steps:")
    print("1. Set up complete project structure")
    print("2. Run full training sessions")
    print("3. Implement hyperparameter tuning")
    print("4. Create performance analysis")

if __name__ == "__main__":
    main()