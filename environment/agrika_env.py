# environment/agrika_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import random

class AgrikaTractorFleetEnv(gym.Env):
    """
    Agricultural Tractor Fleet Management Environment for Agrika
    
    Environment:
    - Runs for 7 days (week)
    - Tracks 3 tractors
    - Daily conditions: Weather (Rainy/Dry), Demand (High/Low/Moderate)
    
    State Space (13 dimensions):
    - Tractor 1: [hours_used, condition, days_since_maintenance] (3)
    - Tractor 2: [hours_used, condition, days_since_maintenance] (3)
    - Tractor 3: [hours_used, condition, days_since_maintenance] (3)
    - Weather: [today, next_day] (2, 0=Rainy, 1=Dry)
    - Season: [day_in_season, working_demand] (2, demand: 0=Low, 1=Moderate, 2=High)
    
    Action Space: 27 discrete actions (3^3 combinations)
    - For each tractor: [OPERATE (0), MAINTAIN (1), REST (2)]
    
    Reward Function:
    - High rewards for meeting high demand in dry season
    - Penalties for operating in rainy season
    - Bonuses for timely maintenance
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Environment parameters
        self.season_length = 7  # 7-day week
        self.num_tractors = 3
        self.max_hours_per_day = 8
        self.breakdown_threshold = 100  # hours before high breakdown risk
        
        # State space: 13 dimensions
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([150, 100, 30, 150, 100, 30, 150, 100, 30, 1, 1, 60, 2]),
            dtype=np.float32
        )
        
        # Action space: 27 discrete actions (3^3)
        self.action_space = spaces.Discrete(27)
        
        # Weather types: 0=Rainy, 1=Dry
        self.weather_types = ['Rainy', 'Dry']
        
        # Demand types: 0=Low, 1=Moderate, 2=High
        self.demand_types = ['Low', 'Moderate', 'High']
        
        # Performance tracking
        self.episode_stats = {
            'total_productivity': 0.0,
            'total_maintenance_cost': 0.0,
            'breakdown_count': 0,
            'efficiency_score': 0.0
        }
        
        # Store current tractor actions for reward calculation
        self.current_tractor_actions = [0, 0, 0]
        
        self.reset()
        self.render_mode = render_mode
    
    def decode_action(self, action: int) -> Tuple[int, int, int]:
        """Convert discrete action to individual tractor actions"""
        tractor3_action = action % 3
        tractor2_action = (action // 3) % 3
        tractor1_action = (action // 9) % 3
        return tractor1_action, tractor2_action, tractor3_action
    
    def reset(self, seed=None, options=None):
        if options:
            max_weather_changes = options.get('max_weather_changes', 6)
            breakdown_multiplier = options.get('breakdown_multiplier', 1.0)
        
        # Initialize tractors with slight randomness
        self.tractors = np.array([
            [random.uniform(0, 5), random.uniform(85, 100), 0.0],  # Tractor 1
            [random.uniform(0, 5), random.uniform(85, 100), 0.0],  # Tractor 2
            [random.uniform(0, 5), random.uniform(85, 100), 0.0]   # Tractor 3
        ])
        
        # Initialize weather (Rainy or Dry)
        self.weather = np.random.randint(0, 2, size=2)  # Today, next day
        
        # Season progress
        self.current_day = 0
        self.working_demand = self._calculate_working_demand()
        
        # Reset episode stats
        self.episode_stats = {
            'total_productivity': 0.0,
            'total_maintenance_cost': 0.0,
            'breakdown_count': 0,
            'efficiency_score': 0.0
        }
        
        # Reset current actions
        self.current_tractor_actions = [0, 0, 0]
        
        return self._get_observation(), {}
    
    def _calculate_working_demand(self) -> float:
        """Calculate working demand based on season (0=Low, 1=Moderate, 2=High)"""
        if self.current_day <= 20:  # Planting (high demand)
            return 2  # High
        elif 20 < self.current_day <= 40:  # Growing (low demand)
            return 0  # Low
        else:  # Harvest (moderate demand)
            return 1  # Moderate
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        obs = np.concatenate([
            self.tractors.flatten(),  # 9 dimensions
            self.weather.astype(np.float32),  # 2 dimensions
            [self.current_day, self.working_demand]  # 2 dimensions
        ])
        return obs.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        tractor_actions = self.decode_action(action)
        self.current_tractor_actions = list(tractor_actions)
        
        # Daily metrics
        daily_productivity = 0.0
        daily_maintenance_cost = 0.0
        daily_breakdown_penalty = 0.0
        
        # Process each tractor
        for i, tractor_action in enumerate(tractor_actions):
            if tractor_action == 0:  # OPERATE
                productivity, breakdown = self._operate_tractor(i)
                daily_productivity += productivity
                if breakdown:
                    daily_breakdown_penalty += 200
                    self.episode_stats['breakdown_count'] += 1
                    
            elif tractor_action == 1:  # MAINTAIN
                cost = self._maintain_tractor(i)
                daily_maintenance_cost += cost
                
            elif tractor_action == 2:  # REST
                self._rest_tractor(i)
        
        # Update environment
        self._update_environment()
        
        # Update episode stats
        self.episode_stats['total_productivity'] += daily_productivity
        self.episode_stats['total_maintenance_cost'] += daily_maintenance_cost
        
        # Calculate reward
        reward = self._calculate_reward(
            daily_productivity, daily_maintenance_cost, daily_breakdown_penalty
        )
        
        # Check termination
        terminated = self.current_day >= self.season_length
        
        if terminated:
            self.episode_stats['efficiency_score'] = (
                self.episode_stats['total_productivity'] /
                max(self.episode_stats['total_maintenance_cost'], 1)
            )
        
        info = {
            'day': self.current_day,
            'productivity': daily_productivity,
            'maintenance_cost': daily_maintenance_cost,
            'breakdowns': self.episode_stats['breakdown_count'],
            'weather': self.weather_types[self.weather[0]],
            'demand': self.demand_types[int(self.working_demand)],
            'episode_stats': self.episode_stats.copy() if terminated else None
        }
        
        return self._get_observation(), reward, terminated, False, info
    
    def _operate_tractor(self, tractor_id: int) -> Tuple[float, bool]:
        """Operate a tractor and return productivity and breakdown status"""
        # Weather impact
        weather_multiplier = 0.3 if self.weather[0] == 0 else 1.2  # Rainy: 0.3, Dry: 1.2
        
        # Calculate productivity based on condition
        condition_factor = max(0.3, self.tractors[tractor_id, 1] / 100)
        base_hours = min(self.max_hours_per_day, 8 + condition_factor * 4)
        actual_productivity = base_hours * weather_multiplier * condition_factor
        
        # Update tractor state
        self.tractors[tractor_id, 0] += base_hours
        degradation = random.uniform(1.5, 4.0) * (1 + 0.5 * (self.weather[0] == 0))  # Worse in rain
        self.tractors[tractor_id, 1] = max(0, self.tractors[tractor_id, 1] - degradation)
        self.tractors[tractor_id, 2] += 1
        
        # Breakdown check
        breakdown_prob = self._calculate_breakdown_probability(tractor_id)
        breakdown = random.random() < breakdown_prob
        
        if breakdown:
            self.tractors[tractor_id, 1] = max(0, self.tractors[tractor_id, 1] - 25)
            actual_productivity *= 0.4
            
        return actual_productivity, breakdown
    
    def _maintain_tractor(self, tractor_id: int) -> float:
        """Perform maintenance on a tractor"""
        condition = self.tractors[tractor_id, 1]
        base_cost = 50 + (100 - condition) * 1.5
        
        # Improve condition
        condition_improvement = min(40, 100 - condition)
        self.tractors[tractor_id, 1] = min(100, condition + condition_improvement)
        self.tractors[tractor_id, 2] = 0
        
        return base_cost
    
    def _rest_tractor(self, tractor_id: int):
        """Rest a tractor"""
        self.tractors[tractor_id, 2] += 1
        # Minimal wear during rest
        if random.random() < 0.05:
            self.tractors[tractor_id, 1] = max(0, self.tractors[tractor_id, 1] - 0.5)
    
    def _calculate_breakdown_probability(self, tractor_id: int) -> float:
        """Calculate breakdown probability"""
        hours_used = self.tractors[tractor_id, 0]
        condition = self.tractors[tractor_id, 1]
        days_since_maintenance = self.tractors[tractor_id, 2]
        
        prob = 0.005  # Base probability
        
        # Hours factor
        if hours_used > self.breakdown_threshold:
            prob += 0.03 * (hours_used - self.breakdown_threshold) / 20
            
        # Condition factor
        if condition < 50:
            prob += 0.08 * (50 - condition) / 50
            
        # Maintenance factor
        if days_since_maintenance > 15:
            prob += 0.02 * (days_since_maintenance - 15) / 15
            
        # Weather impact
        prob *= 1.5 if self.weather[0] == 0 else 0.8  # Rainy: 1.5x, Dry: 0.8x
        
        return min(prob, 0.4)
    
    def _update_environment(self):
        """Update environment state"""
        self.current_day += 1
        
        # Update weather with some persistence
        if random.random() < 0.7:  # 70% chance weather persists
            self.weather[0] = self.weather[1]
            self.weather[1] = random.randint(0, 1)  # New next day
        else:  # Weather changes
            self.weather = np.random.randint(0, 2, size=2)
            
        self.working_demand = self._calculate_working_demand()
    
    def _calculate_reward(self, productivity: float, maintenance_cost: float, 
                         breakdown_penalty: float) -> float:
        """Reward function emphasizing demand, weather, and maintenance"""
        # Base productivity reward
        productivity_reward = productivity * 12
        
        # Demand-based reward
        target_productivity = {0: 10, 1: 20, 2: 30}[int(self.working_demand)]  # Low, Moderate, High
        fulfillment_ratio = min(productivity / max(target_productivity, 1), 1.5)
        demand_bonus = fulfillment_ratio * 50
        
        # Dry season bonus for high demand
        if self.weather[0] == 1 and self.working_demand == 2:  # Dry + High demand
            demand_bonus *= 2  # Double reward for meeting high demand in dry season
        
        # Rainy season penalty for operating
        rainy_penalty = 0
        active_tractors = sum(1 for i in range(3) if self.current_tractor_actions[i] == 0)
        if self.weather[0] == 0 and active_tractors > 0:  # Rainy + operating
            rainy_penalty = active_tractors * 30  # Penalty per operating tractor
        
        # Timely maintenance bonus
        maintenance_bonus = 0
        if maintenance_cost > 0:
            # Count tractors being maintained with good condition
            maintained_tractors = sum(1 for i in range(3) if self.current_tractor_actions[i] == 1)
            avg_condition_maintained = np.mean([self.tractors[i, 1] for i in range(3) 
                                              if self.current_tractor_actions[i] == 1])
            if avg_condition_maintained > 50:  # Maintained before condition too low
                maintenance_bonus = 20 * maintained_tractors
        
        # Total penalties
        total_penalty = maintenance_cost + breakdown_penalty + rainy_penalty
        
        total_reward = productivity_reward + demand_bonus + maintenance_bonus - total_penalty
        
        return total_reward
    
    def render(self, mode='human'):
        """Render environment status"""
        if mode == 'human':
            print(f"\n{'='*50}")
            print(f"Day {self.current_day}/{self.season_length} - Season Progress: {self.current_day/self.season_length*100:.1f}%")
            print(f"Weather: {self.weather_types[self.weather[0]]} | Next Day: {self.weather_types[self.weather[1]]}")
            print(f"Working Demand: {self.demand_types[int(self.working_demand)]} | Season Phase: {self._get_season_phase()}")
            print(f"{'='*50}")
            
            print("TRACTOR FLEET STATUS:")
            for i in range(3):
                hours, condition, days_maint = self.tractors[i]
                status = self._get_tractor_status(condition)
                print(f"  🚜 Tractor {i+1}: {hours:.1f}h | {condition:.1f}% condition ({status}) | {days_maint:.0f} days since maintenance")
            
            print(f"\nEPISODE STATISTICS:")
            print(f"  Total Productivity: {self.episode_stats['total_productivity']:.1f}")
            print(f"  Maintenance Costs: ${self.episode_stats['total_maintenance_cost']:.2f}")
            print(f"  Breakdowns: {self.episode_stats['breakdown_count']}")
            
            if self.current_day >= self.season_length:
                print(f"  Efficiency Score: {self.episode_stats['efficiency_score']:.2f}")
                print("🏁 SEASON COMPLETED!")
    
    def _get_season_phase(self) -> str:
        """Get current season phase"""
        if self.current_day <= 20:
            return "Planting"
        elif self.current_day <= 40:
            return "Growing"
        else:
            return "Harvest"
    
    def _get_tractor_status(self, condition: float) -> str:
        """Get tractor status based on condition"""
        if condition > 80:
            return "Excellent"
        elif condition > 60:
            return "Good"
        elif condition > 40:
            return "Fair"
        elif condition > 20:
            return "Poor"
        else:
            return "Critical"

# Test the environment
if __name__ == "__main__":
    env = AgrikaTractorFleetEnv()
    obs, info = env.reset()
    
    print("🌾 AGRIKA TRACTOR FLEET MANAGEMENT SYSTEM")
    print("Environment Test")
    print(f"State Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    # Test with intelligent actions
    for step in range(10):
        # Sample intelligent action (prefer maintenance when condition is low)
        avg_condition = np.mean(env.tractors[:, 1])
        if avg_condition < 50:
            action = 13  # All maintain (1,1,1 in base 3)
        elif env.weather[0] == 0:  # Rainy: prefer rest
            action = 26  # All rest (2,2,2 in base 3)
        else:
            action = env.action_space.sample()
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\n--- Step {step + 1} ---")
        print(f"Action: {action} -> {env.decode_action(action)}")
        print(f"Reward: {reward:.2f}")
        env.render()
        
        if terminated:
            break