"""
Main Integration Module for Autonomous Forklift System (Modified)
================================================================

This module integrates all the individual components of the autonomous forklift system
described in the thesis "Algorithms for Autonomous Forklift Driving with 3D Camera".

Modified to run without Open3D and external dependencies, this version uses simulated
data and simplified algorithms to demonstrate the core concepts.
"""

import os
import logging
import time
import json
import random
import threading
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/forklift_system.log", mode='w'),
        logging.StreamHandler()
    ]
)

class SimulatedPointCloud:
    """
    A simplified point cloud representation for simulation purposes.
    """
    def __init__(self, num_points=1000, pallet_type="EUR-Pallet"):
        self.pallet_type = pallet_type
        
        # Generate standard pallet dimensions
        if pallet_type == "EUR-Pallet":
            self.width = 1.2  # meters
            self.depth = 0.8  # meters
        elif pallet_type == "CP-Pallet":
            self.width = 1.0  # meters
            self.depth = 1.2  # meters
        else:
            self.width = 1.0 + random.random() * 0.5  # meters
            self.depth = 0.8 + random.random() * 0.5  # meters
        
        self.height = 0.15 + random.random() * 0.05  # meters
        
        # Add some random variation to dimensions
        self.width += random.normalvariate(0, 0.02)
        self.depth += random.normalvariate(0, 0.02)
        
        # Generate pocket coordinates
        pocket_spacing = self.width * 0.5
        self.pockets = [
            {"x": -pocket_spacing/2, "y": 0.0, "z": 0.12, "width": 0.2, "depth": 0.2},
            {"x": pocket_spacing/2, "y": 0.0, "z": 0.12, "width": 0.2, "depth": 0.2}
        ]
        
        # Simulate point data (simplified)
        self.dimensions = {
            "length": self.width,
            "width": self.depth,
            "height": self.height
        }
        
        # Add some noise to simulate real point cloud
        self.tilt_angle = random.normalvariate(0, 2.0)  # Random tilt in degrees
        
        # Flag to indicate if this is a simulated point cloud
        self.is_simulation = True

class SimulatedDepthCamera:
    """Simplified depth camera for simulation."""
    
    def __init__(self):
        """Initialize the simulated depth camera."""
        self.is_running = False
        self.frame_count = 0
        self.dynamic_mode = False
        self.move_probability = 0.3  # Probability of movement in dynamic mode
        
        # Create initial point cloud
        self.current_point_cloud = SimulatedPointCloud()
        
        logging.info("Simulated depth camera initialized")
    
    def start(self):
        """Start the depth camera stream."""
        self.is_running = True
        logging.info("Depth camera started")
    
    def stop(self):
        """Stop the depth camera stream."""
        self.is_running = False
        logging.info("Depth camera stopped")
    
    def set_dynamic_mode(self, enabled):
        """Enable/disable dynamic mode with movement."""
        self.dynamic_mode = enabled
        logging.info(f"Dynamic mode {'enabled' if enabled else 'disabled'}")
    
    def capture_frame(self):
        """Capture a frame from the depth camera."""
        if not self.is_running:
            return None
        
        # Increment frame count
        self.frame_count += 1
        
        # In dynamic mode, sometimes move the pallet slightly
        if self.dynamic_mode and random.random() < self.move_probability:
            # Move pallet position slightly
            for pocket in self.current_point_cloud.pockets:
                pocket["x"] += random.normalvariate(0, 0.01)
                pocket["y"] += random.normalvariate(0, 0.01)
            
            # Adjust tilt slightly
            self.current_point_cloud.tilt_angle += random.normalvariate(0, 0.5)
            
            logging.debug("Point cloud updated in dynamic mode")
        
        # Add some random noise to simulate sensor variation
        noisy_cloud = SimulatedPointCloud(
            pallet_type=self.current_point_cloud.pallet_type
        )
        
        # Copy dimensions with small variations
        noisy_cloud.width = self.current_point_cloud.width + random.normalvariate(0, 0.005)
        noisy_cloud.depth = self.current_point_cloud.depth + random.normalvariate(0, 0.005)
        noisy_cloud.height = self.current_point_cloud.height + random.normalvariate(0, 0.005)
        noisy_cloud.tilt_angle = self.current_point_cloud.tilt_angle + random.normalvariate(0, 0.2)
        
        # Update dimensions
        noisy_cloud.dimensions = {
            "length": noisy_cloud.width,
            "width": noisy_cloud.depth,
            "height": noisy_cloud.height
        }
        
        # Copy pocket positions with small variations
        noisy_cloud.pockets = []
        for pocket in self.current_point_cloud.pockets:
            noisy_pocket = pocket.copy()
            noisy_pocket["x"] += random.normalvariate(0, 0.003)
            noisy_pocket["y"] += random.normalvariate(0, 0.003)
            noisy_pocket["z"] += random.normalvariate(0, 0.002)
            noisy_cloud.pockets.append(noisy_pocket)
        
        return noisy_cloud

class PointCloudProcessor:
    """Process point clouds to extract pallet information."""
    
    def __init__(self):
        """Initialize the point cloud processor."""
        pass
    
    def segment_pallet_pockets(self, point_cloud):
        """
        Segment pallet pockets from the point cloud.
        
        In a real implementation, this would use DBSCAN clustering.
        For simulation, we just use the predefined pockets.
        
        Args:
            point_cloud: Simulated point cloud
            
        Returns:
            List of pocket data
        """
        if not point_cloud or not hasattr(point_cloud, 'pockets'):
            return []
        
        # In simulation mode, just return the predefined pockets with added metadata
        pocket_results = []
        for i, pocket in enumerate(point_cloud.pockets):
            pocket_with_metadata = pocket.copy()
            
            # Add confidence and direction information
            pocket_with_metadata["alpha"] = 90.0 + random.normalvariate(0, 1.0)  # degrees
            pocket_with_metadata["confidence"] = 0.8 + random.random() * 0.15
            pocket_with_metadata["normal"] = [0, 0, 1]  # Pointing upward
            
            pocket_results.append(pocket_with_metadata)
            
            logging.info(f"Segmented pocket {i+1} at position ({pocket['x']:.2f}, {pocket['y']:.2f}, {pocket['z']:.2f})")
        
        return pocket_results
    
    def analyze_pallet_dimensions(self, point_cloud):
        """
        Analyze pallet dimensions and safety parameters.
        
        Args:
            point_cloud: Point cloud data
            
        Returns:
            dict: Dimension analysis results
        """
        if not point_cloud:
            return None
        
        # Get dimensions from the point cloud
        if hasattr(point_cloud, 'dimensions'):
            length = point_cloud.dimensions["length"]
            width = point_cloud.dimensions["width"]
            height = point_cloud.dimensions["height"]
            tilt_angle = point_cloud.tilt_angle
        else:
            # Default dimensions
            length = 1.2  # meters
            width = 0.8  # meters
            height = 0.15  # meters
            tilt_angle = random.normalvariate(0, 2.0)  # degrees
        
        # Safety checks
        is_standard_size = (abs(length - 1.2) < 0.1 and abs(width - 0.8) < 0.1)
        is_height_safe = height < 2.0  # 2 meters max height
        is_tilt_safe = abs(tilt_angle) < 5.0  # 5 degrees max tilt
        overall_safe = is_height_safe and is_tilt_safe
        
        return {
            "dimensions": {
                "length": float(length),
                "width": float(width),
                "height": float(height)
            },
            "is_standard_size": bool(is_standard_size),
            "is_height_safe": bool(is_height_safe),
            "tilt_angle": float(tilt_angle),
            "is_tilt_safe": bool(is_tilt_safe),
            "overall_safe": bool(overall_safe)
        }
    
    def recognize_pallet_type(self, point_cloud):
        """
        Recognize pallet type from point cloud.
        
        In a real implementation, this would use SVM.
        For simulation, we use predefined types.
        
        Args:
            point_cloud: Point cloud data
            
        Returns:
            dict: Pallet type information
        """
        if not point_cloud:
            return None
        
        # In simulation, use the predefined type
        if hasattr(point_cloud, 'pallet_type'):
            pallet_type = point_cloud.pallet_type
            confidence = 0.9
        else:
            # Default to EUR-Pallet with lower confidence
            pallet_type = "EUR-Pallet"
            confidence = 0.7
        
        return {
            "pallet_type": pallet_type,
            "confidence": confidence,
            "dimensions": point_cloud.dimensions if hasattr(point_cloud, 'dimensions') else None
        }

class ForkPositionCalculator:
    """Calculate optimal fork positions for pallet pickup."""
    
    def __init__(self):
        """Initialize the fork position calculator."""
        pass
    
    def calculate_fork_positioning(self, pocket_coordinates, pallet_dimensions):
        """
        Calculate optimal fork positioning based on pocket coordinates.
        
        Args:
            pocket_coordinates: List of pocket coordinate dictionaries
            pallet_dimensions: Pallet dimension information
            
        Returns:
            dict: Fork positioning parameters
        """
        if not pocket_coordinates or len(pocket_coordinates) < 1:
            return None
        
        try:
            # Calculate center point between pockets
            if len(pocket_coordinates) >= 2:
                # Use the two pockets with highest confidence
                pockets = sorted(pocket_coordinates, key=lambda p: p.get("confidence", 0), reverse=True)[:2]
                
                # Calculate center point and approach angle
                x_values = [p["x"] for p in pockets]
                y_values = [p["y"] for p in pockets]
                z_values = [p["z"] for p in pockets]
                
                center_x = sum(x_values) / len(x_values)
                center_y = sum(y_values) / len(y_values)
                center_z = sum(z_values) / len(z_values)
                
                # Get approach angle from the first pocket
                approach_angle = pockets[0].get("alpha", 90.0)
                
                # Calculate fork spacing based on pocket distance
                if len(x_values) >= 2:
                    fork_spacing = abs(x_values[0] - x_values[1])
                else:
                    # Default spacing for Euro pallet
                    fork_spacing = 0.8  # meters
            else:
                # Only one pocket detected, use it with default parameters
                center_x = pocket_coordinates[0]["x"]
                center_y = pocket_coordinates[0]["y"]
                center_z = pocket_coordinates[0]["z"]
                approach_angle = pocket_coordinates[0].get("alpha", 90.0)
                fork_spacing = 0.8  # meters
            
            # Calculate insertion depth based on pallet dimensions
            if pallet_dimensions and "dimensions" in pallet_dimensions:
                insertion_depth = min(0.9, pallet_dimensions["dimensions"]["width"] * 0.8)
            else:
                insertion_depth = 0.7  # meters, default
            
            # Generate a confidence value
            if len(pocket_coordinates) >= 2:
                confidence = 0.9  # High confidence with two pockets
            else:
                confidence = 0.7  # Moderate confidence with one pocket
            
            # Return the positioning parameters
            return {
                "position": {
                    "x": float(center_x),
                    "y": float(center_y),
                    "z": float(center_z)
                },
                "approach_angle": float(approach_angle),
                "fork_spacing": float(fork_spacing),
                "insertion_depth": float(insertion_depth),
                "confidence": float(confidence)
            }
            
        except Exception as e:
            logging.error(f"Error calculating fork positioning: {str(e)}")
            return None

class CommunicationSimulator:
    """Simulate communication with PLC and other systems."""
    
    def __init__(self):
        """Initialize the communication simulator."""
        self.last_data = None
        self.connected = False
        self.messages = []
    
    def connect(self):
        """Simulate connecting to PLC."""
        self.connected = True
        logging.info("Connected to simulated PLC")
    
    def disconnect(self):
        """Simulate disconnecting from PLC."""
        self.connected = False
        logging.info("Disconnected from simulated PLC")
    
    def send_data(self, data):
        """
        Simulate sending data to PLC.
        
        Args:
            data: Data to send
            
        Returns:
            bool: Success status
        """
        if not self.connected:
            logging.warning("Cannot send data: Not connected")
            return False
        
        # Simulate network delay
        time.sleep(0.1)
        
        # Store the data
        self.last_data = data
        self.messages.append({
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
        
        logging.info(f"Sent data to PLC: {json.dumps(data, indent=2)}")
        return True

class AutonomousForkliftSystem:
    """
    Main class integrating all components of the autonomous forklift system.
    This class orchestrates the entire workflow from capturing point cloud data
    to sending positioning commands to the forklift controller.
    """
    
    def __init__(self, use_simulation=True):
        """
        Initialize the autonomous forklift system.
        
        Args:
            use_simulation: Whether to use simulated data (always True in this version)
        """
        # Always use simulation in this version
        self.use_simulation = True
        
        # System parameters
        self.max_weight = 1500  # kg, as per EPAL standards
        self.max_height = 2.0   # meters, maximum safe height
        self.max_tilt = 5.0     # degrees, maximum allowable tilt
        
        # Create component instances
        self.depth_camera = SimulatedDepthCamera()
        self.point_cloud_processor = PointCloudProcessor()
        self.fork_position_calculator = ForkPositionCalculator()
        self.communication = CommunicationSimulator()
        
        # Initialize the system state
        self.is_running = False
        self.current_point_cloud = None
        self.detection_thread = None
        self.stop_event = threading.Event()
        
        # Ensure required directories exist
        os.makedirs('logs', exist_ok=True)
        
        # System state and metrics
        self.system_state = {
            "status": "initialized",
            "last_detection_time": None,
            "detection_count": 0,
            "successful_alignments": 0,
            "error_count": 0
        }
        
        logging.info("Autonomous Forklift System initialized")
    
    def start(self):
        """
        Start the autonomous forklift system.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.is_running:
            logging.warning("System is already running")
            return False
        
        try:
            # Print system startup banner
            logging.info("=" * 80)
            logging.info("   Autonomous Forklift Driving System with 3D Camera - Starting Up")
            logging.info("=" * 80)
            
            # Start the depth camera
            self.depth_camera.start()
            self.depth_camera.set_dynamic_mode(True)
            
            # Connect to PLC (simulated)
            self.communication.connect()
            
            # Start the detection thread
            self.stop_event.clear()
            self.detection_thread = threading.Thread(target=self._detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            self.is_running = True
            self.system_state["status"] = "running"
            
            logging.info("Autonomous Forklift System started successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start system: {str(e)}")
            self._cleanup()
            return False
    
    def stop(self):
        """
        Stop the autonomous forklift system.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.is_running:
            logging.warning("System is not running")
            return False
        
        try:
            logging.info("Stopping Autonomous Forklift System...")
            
            # Signal detection thread to stop
            self.stop_event.set()
            
            # Wait for detection thread to terminate
            if self.detection_thread and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=5.0)
            
            # Disconnect from PLC
            self.communication.disconnect()
            
            # Stop depth camera
            self.depth_camera.stop()
            
            self.is_running = False
            self.system_state["status"] = "stopped"
            
            logging.info("Autonomous Forklift System stopped successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error stopping system: {str(e)}")
            return False
    
    def _detection_loop(self):
        """Main detection loop running in a separate thread."""
        logging.info("Starting detection loop")
        
        cycle_count = 0
        
        while not self.stop_event.is_set():
            try:
                cycle_count += 1
                logging.info("=" * 50)
                logging.info(f"Detection cycle #{cycle_count}")
                
                # 1. Capture point cloud
                self.current_point_cloud = self.depth_camera.capture_frame()
                
                if not self.current_point_cloud:
                    logging.warning("Failed to capture point cloud")
                    time.sleep(1)
                    continue
                
                # Update detection metrics
                self.system_state["last_detection_time"] = time.time()
                self.system_state["detection_count"] += 1
                
                # 2. Analyze pallet dimensions and safety
                pallet_dimensions = self.point_cloud_processor.analyze_pallet_dimensions(self.current_point_cloud)
                
                if not pallet_dimensions:
                    logging.warning("Failed to analyze pallet dimensions")
                    time.sleep(1)
                    continue
                
                logging.info(f"Pallet dimensions: {json.dumps(pallet_dimensions, indent=2)}")
                
                if not pallet_dimensions["overall_safe"]:
                    logging.warning("Pallet dimensions or orientation unsafe for operation")
                    self._send_safety_alert(pallet_dimensions)
                    time.sleep(1)
                    continue
                
                # 3. Recognize pallet type
                pallet_type = self.point_cloud_processor.recognize_pallet_type(self.current_point_cloud)
                
                if pallet_type:
                    logging.info(f"Detected pallet type: {pallet_type['pallet_type']} (confidence: {pallet_type['confidence']:.2f})")
                
                # 4. Segment pallet pockets
                segmented_pockets = self.point_cloud_processor.segment_pallet_pockets(self.current_point_cloud)
                
                if not segmented_pockets:
                    logging.warning("No pallet pockets detected")
                    time.sleep(1)
                    continue
                
                logging.info(f"Detected {len(segmented_pockets)} pallet pockets")
                
                # 5. Calculate optimal fork positioning
                fork_positioning = self.fork_position_calculator.calculate_fork_positioning(
                    segmented_pockets, pallet_dimensions
                )
                
                if not fork_positioning:
                    logging.warning("Failed to calculate fork positioning")
                    time.sleep(1)
                    continue
                
                logging.info(f"Fork positioning: {json.dumps(fork_positioning, indent=2)}")
                
                # 6. Send positioning data to PLC
                positioning_data = {
                    "fork_positioning": fork_positioning,
                    "pallet_dimensions": pallet_dimensions,
                    "pallet_type": pallet_type,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Send data via communication manager
                send_success = self.communication.send_data(positioning_data)
                
                if send_success:
                    logging.info("Successfully communicated positioning data to PLC")
                    
                    # Simulate fork movement and pickup
                    self._simulate_pickup_operation(fork_positioning)
                else:
                    logging.warning("Failed to communicate with PLC")
                    self.system_state["error_count"] += 1
                
                # Wait before next cycle
                time.sleep(2)
                
            except Exception as e:
                logging.error(f"Error in detection loop: {str(e)}")
                self.system_state["error_count"] += 1
                time.sleep(1)
    
    def _simulate_pickup_operation(self, fork_positioning):
        """
        Simulate a pickup operation based on fork positioning.
        
        Args:
            fork_positioning: Fork positioning data
        """
        # Calculate success probability based on positioning confidence
        success_probability = fork_positioning["confidence"] * 0.9
        
        # Simulate success/failure
        if random.random() < success_probability:
            logging.info("Pickup operation successful")
            self.system_state["successful_alignments"] += 1
        else:
            logging.warning("Pickup operation failed")
            self.system_state["error_count"] += 1
    
    def _send_safety_alert(self, pallet_dimensions):
        """
        Send a safety alert for unsafe pallet dimensions.
        
        Args:
            pallet_dimensions: Pallet dimension data
        """
        # Prepare safety alert message
        safety_issues = []
        
        if not pallet_dimensions["is_height_safe"]:
            safety_issues.append(f"Height exceeds maximum: {pallet_dimensions['dimensions']['height']:.2f}m > {self.max_height}m")
        
        if not pallet_dimensions["is_tilt_safe"]:
            safety_issues.append(f"Tilt exceeds maximum: {pallet_dimensions['tilt_angle']:.2f}° > {self.max_tilt}°")
        
        alert_data = {
            "alert_type": "safety_violation",
            "timestamp": datetime.now().isoformat(),
            "safety_issues": safety_issues,
            "pallet_dimensions": pallet_dimensions
        }
        
        # Send alert via communication manager
        self.communication.send_data(alert_data)
        
        logging.warning(f"Safety alert sent: {', '.join(safety_issues)}")
    
    def _cleanup(self):
        """Clean up resources."""
        logging.info("Cleaning up resources")
        
        # Stop the depth camera if it's running
        if hasattr(self, 'depth_camera'):
            self.depth_camera.stop()
        
        # Disconnect from PLC
        if hasattr(self, 'communication'):
            self.communication.disconnect()
    
    def get_system_state(self):
        """
        Get the current state of the system.
        
        Returns:
            dict: Current system state
        """
        # Update the system state
        self.system_state["current_point_cloud"] = (
            "Available" if self.current_point_cloud is not None else "None"
        )
        
        self.system_state["communication_status"] = self.communication.connected
        
        return self.system_state


# Example usage
if __name__ == "__main__":
    print("\n======== Autonomous Forklift System - Modified Implementation ========\n")
    print("This simulation demonstrates the algorithms described")
    print("in the thesis 'Algorithms for Autonomous Forklift Driving with 3D Camera'")
    print("without requiring external hardware or complex libraries.\n")
    
    # Create and start the system
    forklift_system = AutonomousForkliftSystem(use_simulation=True)
    
    try:
        # Start the system
        if forklift_system.start():
            print("\nSystem is running. Press Ctrl+C to stop.\n")
            
            # Run for a limited time
            max_runtime = 30  # seconds
            start_time = time.time()
            
            while time.time() - start_time < max_runtime:
                # Print system state every 5 seconds
                if int(time.time() - start_time) % 5 == 0:
                    state = forklift_system.get_system_state()
                    print(f"\nSystem state: {state['status']}")
                    print(f"Detections: {state['detection_count']}")
                    print(f"Successful alignments: {state['successful_alignments']}")
                    print(f"Errors: {state['error_count']}")
                
                time.sleep(1)
            
            # Stop the system
            forklift_system.stop()
            print("\nSimulation completed successfully\n")
            
        else:
            print("\nFailed to start the system\n")
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        forklift_system.stop()
    except Exception as e:
        print(f"\nError: {str(e)}")
        forklift_system.stop()
    
    print("\n======== Simulation Complete ========\n")