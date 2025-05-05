"""
Main Integration Module for Autonomous Forklift System
======================================================

This module integrates all components of the autonomous forklift system
as described in the thesis "Algorithms for Autonomous Forklift Driving
with 3D Camera."

The system combines:
1. Convolutional Neural Networks (CNN) for pallet pocket detection
2. DBSCAN for point cloud segmentation
3. RANSAC for robust coordinate calculation
4. ICP for alignment of forks
5. Communication protocols (TCP/IP)
"""

import numpy as np
import os
import time
import logging
import json
import socket
import threading
from datetime import datetime

# Import custom modules
from forklift_cnn_model import PalletDetectionCNN
from forklift_point_cloud_segmentation import segment_pallet_pockets, analyze_pallet_dimensions
from forklift_ransac_icp import calculate_icp, calculate_optimal_fork_position
from forklift_shape_recognition import recognize_pallet_type
import forklift_communication as comm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MAX_WEIGHT_EURO_PALLET = 1500  # As per EPAL standards (kg)
REFERENCE_PALLET_PATH = 'data/reference_euro_pallet.ply'
TARGET_CLOUD_PATH = 'data/fork_target.ply'

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
            use_simulation: Whether to use simulated data instead of a physical camera
        """
        self.use_simulation = use_simulation
        self.running = False
        self.detection_thread = None
        self.depth_stream = None
        self.cnn_model = PalletDetectionCNN()
        self.communication_manager = comm.CommunicationManager()
        
        # System state
        self.system_state = {
            "status": "initializing",
            "pallet_detected": False,
            "last_detection_time": None,
            "fork_positioning": None,
            "pallet_dimensions": None
        }
        
        # Initialize necessary directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('output', exist_ok=True)
    
    def start(self):
        """
        Start the autonomous forklift system.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            # Initialize the depth camera or simulation
            if not self._initialize_depth_stream():
                return False
            
            # Load the CNN model
            if not self.cnn_model.load():
                logging.error("Failed to load CNN model.")
                return False
            
            # Start the communication manager
            self.communication_manager.start_communication_threads()
            
            # Start the detection loop
            self.running = True
            self.detection_thread = threading.Thread(target=self._detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            self.system_state["status"] = "running"
            logging.info("Autonomous forklift system started successfully.")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start autonomous forklift system: {e}")
            self._cleanup()
            return False
    
    def stop(self):
        """
        Stop the autonomous forklift system.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        try:
            self.running = False
            
            if self.detection_thread and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=2.0)
            
            self._cleanup()
            
            self.system_state["status"] = "stopped"
            logging.info("Autonomous forklift system stopped successfully.")
            return True
            
        except Exception as e:
            logging.error(f"Error while stopping autonomous forklift system: {e}")
            return False
    
    def _initialize_depth_stream(self):
        """Initialize the depth camera or simulation."""
        try:
            if self.use_simulation:
                from forklift_simulator import SimulatedDepthCamera
                self.depth_stream = SimulatedDepthCamera()
                self.depth_stream.start()
                logging.info("Simulated depth camera initialized")
                self.depth_stream.set_dynamic_mode(True)
            else:
                # Use real camera (Orbbec Gemini 2)
                import openni2
                openni2.initialize()
                dev = openni2.Device.open_any()
                self.depth_stream = dev.create_depth_stream()
                self.depth_stream.start()
                logging.info("Orbbec Gemini 2 camera initialized successfully.")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize depth stream: {e}")
            return False
    
    def _cleanup(self):
        """Clean up resources when shutting down."""
        if self.depth_stream:
            self.depth_stream.stop()
            logging.info("Depth camera stopped")
        
        self.communication_manager.stop_communication_threads()
        logging.info("System shutdown complete")
    
    def _capture_point_cloud(self):
        """
        Capture a point cloud from the depth camera or simulation.
        
        Returns:
            Point cloud data or None if failed
        """
        try:
            if self.use_simulation:
                # Simulated point cloud
                return self.depth_stream.capture_frame()
            else:
                # Real camera point cloud
                import open3d as o3d
                
                # Read frame from depth camera
                frame = self.depth_stream.read_frame()
                depth_data = np.array(frame.get_buffer_as_uint16()).reshape((frame.height, frame.width))
                
                # Convert depth data to point cloud
                points = []
                for i in range(depth_data.shape[0]):
                    for j in range(depth_data.shape[1]):
                        z = depth_data[i, j] / 1000.0  # Convert to meters
                        if 0 < z < 5.0:  # Filter valid depth points
                            x = (j - depth_data.shape[1] / 2) * z / frame.focal_length
                            y = (i - depth_data.shape[0] / 2) * z / frame.focal_length
                            points.append([x, y, z])
                
                # Create Open3D point cloud
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
                return point_cloud
                
        except Exception as e:
            logging.error(f"Error capturing point cloud: {e}")
            return None
    
    def _detection_loop(self):
        """Main detection loop running in a separate thread."""
        cycle_count = 0
        
        while self.running:
            cycle_count += 1
            logging.info(f"=== Processing cycle {cycle_count} ===")
            
            # Step 1: Capture point cloud
            point_cloud = self._capture_point_cloud()
            if point_cloud is None:
                logging.warning("Failed to capture point cloud. Retrying...")
                time.sleep(1)
                continue
            
            # Step 2: Analyze pallet dimensions (safety check)
            pallet_dimensions = analyze_pallet_dimensions(point_cloud)
            logging.info(f"Pallet dimensions: {json.dumps(pallet_dimensions, indent=2)}")
            
            # Step 3: Check if pallet dimensions are safe
            if not pallet_dimensions["overall_safe"]:
                self._send_safety_alert(pallet_dimensions)
                time.sleep(1)
                continue
            
            # Step 4: Segment pallet pockets
            pockets = segment_pallet_pockets(point_cloud)
            if not pockets or len(pockets) < 2:
                logging.warning(f"Insufficient pallet pockets detected: {len(pockets) if pockets else 0}")
                time.sleep(1)
                continue
            
            # Log pocket positions
            for i, pocket in enumerate(pockets[:2]):  # We only need the first two pockets
                center = np.mean(pocket, axis=0)
                logging.info(f"Segmented pocket {i+1} at position ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
            
            logging.info(f"Detected {len(pockets)} pallet pockets")
            
            # Step 5: Recognize pallet type
            pallet_type = recognize_pallet_type(point_cloud)
            
            # Step 6: Calculate optimal fork positioning
            fork_positioning = calculate_optimal_fork_position(pockets, pallet_dimensions)
            logging.info(f"Fork positioning: {json.dumps(fork_positioning, indent=2)}")
            
            # Step 7: Send data to PLC
            data_to_send = {
                "fork_positioning": fork_positioning,
                "pallet_dimensions": pallet_dimensions,
                "timestamp": datetime.now().isoformat()
            }
            
            self.communication_manager.send_data(data_to_send, protocol="TCP/IP")
            
            # Update system state
            self.system_state["pallet_detected"] = True
            self.system_state["last_detection_time"] = datetime.now().isoformat()
            self.system_state["fork_positioning"] = fork_positioning
            self.system_state["pallet_dimensions"] = pallet_dimensions
            
            # Simulate pickup operation for visualization/testing
            if self.use_simulation:
                self._simulate_pickup_operation(fork_positioning)
            
            # Delay before next cycle
            time.sleep(2)
    
    def _simulate_pickup_operation(self, fork_positioning):
        """
        Simulate a pickup operation based on fork positioning.
        
        Args:
            fork_positioning: Fork positioning data
        """
        # Simple simulation of pickup success based on fork positioning
        if (abs(fork_positioning["approach_angle"] - 90) < 5 and 
            abs(fork_positioning["position"]["x"]) < 0.05 and
            abs(fork_positioning["position"]["y"]) < 0.05):
            logging.info("Pickup operation successful!")
        else:
            logging.warning("Pickup operation failed!")
    
    def _send_safety_alert(self, pallet_dimensions):
        """
        Send a safety alert for unsafe pallet dimensions.
        
        Args:
            pallet_dimensions: Pallet dimension data
        """
        alert = {
            "type": "safety_alert",
            "timestamp": datetime.now().isoformat(),
            "message": "Unsafe pallet detected",
            "details": pallet_dimensions
        }
        
        self.communication_manager.send_data(alert, protocol="TCP/IP")
        logging.warning(f"Safety alert sent: {json.dumps(alert, indent=2)}")
    
    def get_system_state(self):
        """
        Get the current state of the system.
        
        Returns:
            dict: Current system state
        """
        return self.system_state.copy()


if __name__ == "__main__":
    try:
        # Create and start the autonomous forklift system
        system = AutonomousForkliftSystem(use_simulation=True)
        
        if not system.start():
            logging.error("Failed to start autonomous forklift system.")
            exit(1)
        
        # Simulated pallet weight (in a real system, this would come from sensors)
        pallet_weight = 1200  # kg
        
        # Check pallet weight
        if pallet_weight > MAX_WEIGHT_EURO_PALLET:
            logging.error(f"Pallet weight ({pallet_weight} kg) exceeds the maximum allowable weight ({MAX_WEIGHT_EURO_PALLET} kg).")
            logging.warning("Task reset due to excessive weight.")
            system.stop()
            exit(1)
            
        logging.info(f"Pallet weight ({pallet_weight} kg) is within the allowable limit.")
        
        # Keep the main thread running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("User interrupted. Shutting down...")
        
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        
    finally:
        if 'system' in locals():
            system.stop()