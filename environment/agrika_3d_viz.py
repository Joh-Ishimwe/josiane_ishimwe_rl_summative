import pybullet as p
import pybullet_data
import numpy as np
import time
import math
from agrika_env import AgrikaTractorFleetEnv

class Agrika3DVisualizer:
    """
    Advanced 3D Visualization for Agrika Tractor Fleet Management
    Using PyBullet for high-quality rendering
    """
    
    def __init__(self, env):
        self.env = env
        self.physics_client = None
        self.tractor_ids = []
        self.field_markers = []
        self.ui_elements = {}
        self.weather_particles = []
        
        # Colors for different states
        self.colors = {
            'healthy': [0, 1, 0, 1],      # Green
            'warning': [1, 1, 0, 1],      # Yellow  
            'critical': [1, 0, 0, 1],     # Red
            'maintenance': [0, 0, 1, 1],  # Blue
            'operating': [0, 1, 1, 1],    # Cyan
            'resting': [0.5, 0.5, 0.5, 1] # Gray
        }
        
    def initialize_visualization(self):
        """Initialize PyBullet 3D environment"""
        # Connect to PyBullet in GUI mode
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Configure visualization
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        
        # Set up physics
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        
        # Create the farm environment
        self._create_farm_environment()
        self._create_tractors()
        self._setup_camera()
        self._create_ui_elements()
        
    def _create_farm_environment(self):
        """Create the 3D farm environment"""
        # Load ground plane (farm field)
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(self.plane_id, -1, rgbaColor=[0.4, 0.6, 0.2, 1])  # Green field
        
        # Create field boundaries and sections
        field_size = 20
        section_size = field_size / 4
        
        # Create field section markers
        for i in range(5):
            for j in range(5):
                x = (i - 2) * section_size
                y = (j - 2) * section_size
                
                # Create small marker cubes
                marker_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.1])
                marker_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.1],
                                                   rgbaColor=[0.8, 0.4, 0.1, 1])
                marker_id = p.createMultiBody(0, marker_shape, marker_visual, [x, y, 0.1])
                self.field_markers.append(marker_id)
        
        # Create farm buildings (simple structures)
        self._create_farm_buildings()
        
    def _create_farm_buildings(self):
        """Create farm buildings for context"""
        # Barn
        barn_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[3, 2, 2])
        barn_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[3, 2, 2],
                                         rgbaColor=[0.6, 0.3, 0.1, 1])  # Brown
        self.barn_id = p.createMultiBody(0, barn_shape, barn_visual, [15, 0, 2])
        
        # Storage shed
        shed_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[2, 1.5, 1.5])
        shed_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[2, 1.5, 1.5],
                                         rgbaColor=[0.5, 0.5, 0.5, 1])  # Gray
        self.shed_id = p.createMultiBody(0, shed_shape, shed_visual, [12, 8, 1.5])
        
    def _create_tractors(self):
        """Create 3D tractor models"""
        for i in range(3):
            # Create tractor body (main chassis)
            chassis_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.5, 0.8, 0.6])
            chassis_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[1.5, 0.8, 0.6],
                                               rgbaColor=self.colors['healthy'])
            
            # Position tractors in different locations
            x_pos = -8 + i * 4
            y_pos = -8 + i * 2
            
            tractor_id = p.createMultiBody(1000, chassis_shape, chassis_visual, 
                                         [x_pos, y_pos, 0.6])
            
            # Add wheels to tractors
            self._add_wheels_to_tractor(tractor_id, x_pos, y_pos)
            
            self.tractor_ids.append(tractor_id)
            
    def _add_wheels_to_tractor(self, tractor_id, x_pos, y_pos):
        """Add wheels to tractor for realistic appearance"""
        wheel_positions = [
            [1.2, 0.9, 0],    # Front right
            [1.2, -0.9, 0],   # Front left
            [-1.2, 0.9, 0],   # Rear right
            [-1.2, -0.9, 0]   # Rear left
        ]
        
        for i, (dx, dy, dz) in enumerate(wheel_positions):
            wheel_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.4, height=0.3)
            wheel_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.4, length=0.3,
                                             rgbaColor=[0.1, 0.1, 0.1, 1])  # Black wheels
            
            wheel_id = p.createMultiBody(10, wheel_shape, wheel_visual,
                                       [x_pos + dx, y_pos + dy, 0.4])
            
            # Create constraint to attach wheel to tractor
            p.createConstraint(tractor_id, -1, wheel_id, -1, p.JOINT_FIXED,
                             [0, 0, 0], [dx, dy, dz], [0, 0, 0])
    
    def _setup_camera(self):
        """Setup camera for optimal viewing"""
        # Set camera position for good overview
        p.resetDebugVisualizerCamera(
            cameraDistance=25,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0]
        )
        
    def _create_ui_elements(self):
        """Create UI elements for displaying information"""
        # These will be text overlays showing environment state
        self.ui_elements = {
            'day': None,
            'weather': None,
            'demand': None,
            'tractors': [None, None, None]
        }
        
    def update_visualization(self, obs, action, reward, info):
        """Update the 3D visualization based on current state"""
        
        # Update tractor colors and positions based on their state
        self._update_tractor_states(obs, action)
        
        # Update weather effects
        self._update_weather_effects(info.get('weather', 'Sunny'))
        
        # Update text information
        self._update_text_display(obs, reward, info)
        
        # Step physics simulation
        p.stepSimulation()
        
    def _update_tractor_states(self, obs, action):
        """Update tractor visual states based on their condition and actions"""
        
        # Decode action for each tractor
        tractor_actions = self.env.decode_action(action)
        
        for i in range(3):
            # Get tractor state from observation
            tractor_start_idx = i * 3
            hours_used = obs[tractor_start_idx]
            condition = obs[tractor_start_idx + 1]
            days_since_maintenance = obs[tractor_start_idx + 2]
            
            # Determine tractor color based on condition and action
            if tractor_actions[i] == 1:  # MAINTAIN
                color = self.colors['maintenance']
            elif tractor_actions[i] == 2:  # REST
                color = self.colors['resting']
            elif tractor_actions[i] == 0:  # OPERATE
                if condition > 70:
                    color = self.colors['healthy']
                elif condition > 40:
                    color = self.colors['warning']
                else:
                    color = self.colors['critical']
            
            # Update tractor color
            p.changeVisualShape(self.tractor_ids[i], -1, rgbaColor=color)
            
            # Add movement animation for operating tractors
            if tractor_actions[i] == 0:  # Operating
                current_pos, current_orn = p.getBasePositionAndOrientation(self.tractor_ids[i])
                # Small oscillation to show operation
                new_x = current_pos[0] + 0.1 * math.sin(time.time() * 2)
                p.resetBasePositionAndOrientation(self.tractor_ids[i], 
                                                [new_x, current_pos[1], current_pos[2]], 
                                                current_orn)
    
    def _update_weather_effects(self, weather):
        """Add visual weather effects"""
        # Clear previous weather particles
        for particle in self.weather_particles:
            p.removeBody(particle)
        self.weather_particles.clear()
        
        # Add weather-specific visual effects
        if weather == 'Rainy':
            self._add_rain_effect()
        elif weather == 'Stormy':
            self._add_storm_effect()
        elif weather == 'Sunny':
            # Change lighting/sky color
            pass
            
    def _add_rain_effect(self):
        """Add rain particle effects"""
        for _ in range(20):
            x = np.random.uniform(-15, 15)
            y = np.random.uniform(-15, 15)
            z = np.random.uniform(8, 12)
            
            raindrop_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
            raindrop_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.05,
                                                 rgbaColor=[0.6, 0.8, 1, 0.7])
            
            raindrop_id = p.createMultiBody(0.1, raindrop_shape, raindrop_visual, [x, y, z])
            self.weather_particles.append(raindrop_id)
    
    def _add_storm_effect(self):
        """Add storm effects (darker particles, more intense)"""
        for _ in range(30):
            x = np.random.uniform(-20, 20)
            y = np.random.uniform(-20, 20)
            z = np.random.uniform(10, 15)
            
            storm_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
            storm_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.1,
                                             rgbaColor=[0.3, 0.3, 0.4, 0.8])
            
            storm_id = p.createMultiBody(0.2, storm_shape, storm_visual, [x, y, z])
            self.weather_particles.append(storm_id)
    
    def _update_text_display(self, obs, reward, info):
        """Update text information overlay"""
        # Add debug text showing key information
        text_lines = [
            f"Day: {info.get('day', 0)}/60",
            f"Weather: {info.get('weather', 'Unknown')}",
            f"Reward: {reward:.2f}",
            f"Crop Demand: {obs[14]:.1f}",
            "",
            "Tractor Status:",
        ]
        
        for i in range(3):
            tractor_start_idx = i * 3
            condition = obs[tractor_start_idx + 1]
            text_lines.append(f"T{i+1}: {condition:.1f}% condition")
        
        # Display text (PyBullet text rendering is limited, but functional)
        display_text = "\\n".join(text_lines)
        
    def render_frame(self, obs, action, reward, info):
        """Render a single frame of the visualization"""
        self.update_visualization(obs, action, reward, info)
        
        # Add small delay for visualization
        time.sleep(0.1)
        
    def close(self):
        """Clean up visualization"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)

# Test the 3D visualization
def test_3d_visualization():
    """Test the 3D visualization with random actions"""
    
    # Create environment and visualizer
    env = AgrikaTractorFleetEnv()
    visualizer = Agrika3DVisualizer(env)
    
    # Initialize visualization
    visualizer.initialize_visualization()
    
    # Reset environment
    obs, info = env.reset()
    
    print("3D Visualization Test Started!")
    print("Close the PyBullet window to stop the test.")
    
    try:
        # Run simulation for several steps
        for step in range(50):
            # Take random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update visualization
            visualizer.render_frame(obs, action, reward, info)
            
            # Print step info
            print(f"Step {step + 1}: Action {action} -> {env.decode_action(action)}, "
                  f"Reward: {reward:.2f}, Weather: {info['weather']}")
            
            if terminated:
                print("Season completed!")
                break
                
    except KeyboardInterrupt:
        print("Test interrupted by user")
    
    finally:
        # Clean up
        visualizer.close()
        print("3D Visualization test completed!")

if __name__ == "__main__":
    test_3d_visualization()