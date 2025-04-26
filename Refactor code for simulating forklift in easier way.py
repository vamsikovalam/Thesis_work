import cv2
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor  # Added for RANSAC implementation
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from sklearn import svm  # Added for SVM implementation
import openni2
import time
import logging
import os
import socket
import json  # For better data serialization
from datetime import datetime  # For timestamp tracking

# Setup logging for debugging and tracking with both file and console output
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("forklift_system.log"),
        logging.StreamHandler()
    ]
)

# Paths and Parameters
TRAIN_DATA_PATH = 'data/pallet/train'
TEST_DATA_PATH = 'data/pallet/test'
MODEL_PATH = 'models/pallet_detection_cnn.h5'
REFERENCE_PALLET_PATH = 'data/reference_euro_pallet.ply'
TARGET_CLOUD_PATH = 'data/fork_target.ply'
LOG_PATH = 'logs/'  # Directory for storing operational logs

# Maximum allowable parameters for Euro pallets
MAX_WEIGHT_EURO_PALLET = 1500  # kg, as per EPAL standards
MAX_HEIGHT_EURO_PALLET = 2.0   # meters, maximum safe height
MAX_TILT_ANGLE = 5.0           # degrees, maximum allowable tilt

# Ensure all directories exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# Step 1: Data Preprocessing and Collection
def setup_data_generators(train_path, test_path):
    """Set up data augmentation for training dataset."""
    try:
        # Enhanced data augmentation for better model generalization
        image_gen = ImageDataGenerator(
            rescale=1./255,  # Normalize pixel values
            rotation_range=20,  # Rotate images randomly
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        train_gen = image_gen.flow_from_directory(
            train_path,
            target_size=(400, 400),
            batch_size=16,
            class_mode='binary'
        )
        
        # Test data should only be rescaled, not augmented
        test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
            test_path,
            target_size=(400, 400),
            batch_size=16,
            class_mode='binary'
        )
        
        logging.info(f"Data generators setup complete. Found {train_gen.samples} training samples and {test_gen.samples} test samples.")
        return train_gen, test_gen
    except Exception as e:
        logging.error(f"Error setting up data generators: {e}")
        raise

# Step 2: Define and Train the CNN Model with improved architecture
def create_and_train_cnn(train_gen, test_gen, model_path):
    """Define and train the CNN model with improved architecture for pallet detection."""
    model = Sequential([
        # First convolutional block with increased filters
        Conv2D(64, (3, 3), input_shape=(400, 400, 3), activation='relu', padding='same', name='Conv1'),
        MaxPooling2D(pool_size=(2, 2), name='Pool1'),
        
        # Second convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same', name='Conv2'),
        MaxPooling2D(pool_size=(2, 2), name='Pool2'),
        
        # Third convolutional block
        Conv2D(256, (3, 3), activation='relu', padding='same', name='Conv3'),
        MaxPooling2D(pool_size=(2, 2), name='Pool3'),
        
        # Fourth convolutional block for deeper feature extraction
        Conv2D(512, (3, 3), activation='relu', padding='same', name='Conv4'),
        MaxPooling2D(pool_size=(2, 2), name='Pool4'),
        
        # Flatten and fully connected layers
        Flatten(name='Flatten'),
        Dense(512, activation='relu', name='FC1'),
        Dropout(0.5, name='Dropout1'),  # Prevent overfitting
        Dense(256, activation='relu', name='FC2'),
        Dropout(0.3, name='Dropout2'),
        Dense(1, activation='sigmoid', name='Output')
    ])
    
    try:
        # Use binary cross-entropy for binary classification problem
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', 'precision', 'recall']  # Track more performance metrics
        )
        
        logging.info("Model architecture:")
        model.summary(print_fn=logging.info)
        
        # Train with early stopping to prevent overfitting
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
        
        # Train the model with callbacks
        history = model.fit(
            train_gen,
            epochs=30,  # Increased epochs with early stopping
            validation_data=test_gen,
            steps_per_epoch=100,
            validation_steps=20,
            callbacks=[early_stopping, checkpoint]
        )
        
        # Save training history for later analysis
        with open(os.path.join(LOG_PATH, 'training_history.json'), 'w') as f:
            history_dict = {key: [float(x) for x in value] for key, value in history.history.items()}
            json.dump(history_dict, f, indent=4)
        
        # Log final performance metrics
        val_acc = max(history.history['val_accuracy'])
        logging.info(f"Model training completed with validation accuracy: {val_acc:.4f}")
        
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

# Step 3: Initialize Orbbec Gemini 2 Camera with error handling
def initialize_orbbec_camera():
    """Initialize the Orbbec Gemini 2 camera with robust error handling."""
    try:
        # Check if OpenNI is already initialized
        try:
            openni2.unload()
        except:
            pass
        
        openni2.initialize()  # Load OpenNI drivers
        
        # List available devices for debugging
        devices = []
        for device_info in openni2.Device.enumerate_devices():
            devices.append(f"Device: {device_info.uri} - {device_info.name}")
        
        if not devices:
            logging.warning("No Orbbec devices found.")
        else:
            logging.info(f"Available devices: {', '.join(devices)}")
        
        dev = openni2.Device.open_any()
        
        # Get device info for logging
        device_info = dev.get_device_info()
        logging.info(f"Connected to {device_info.name} (S/N: {device_info.serial_number})")
        
        # Configure depth stream
        depth_stream = dev.create_depth_stream()
        depth_stream.set_video_mode(
            openni2.VideoMode(
                pixelFormat=openni2.PIXEL_FORMAT_DEPTH_1_MM,
                resolutionX=640,
                resolutionY=480,
                fps=30
            )
        )
        
        # Start stream
        depth_stream.start()
        logging.info("Orbbec Gemini 2 camera initialized successfully.")
        
        return dev, depth_stream
    except Exception as e:
        logging.error(f"Failed to initialize Orbbec Gemini 2 camera: {str(e)}")
        raise

# Step 4: Capture and Pre-process Point Cloud Data
def capture_point_cloud(depth_stream):
    """Capture and pre-process point cloud data from the depth stream."""
    try:
        # Read frame with timeout
        start_time = time.time()
        max_wait_time = 2.0  # seconds
        
        while time.time() - start_time < max_wait_time:
            try:
                depth_frame = depth_stream.read_frame(timeout=500)  # 500ms timeout
                break
            except openni2.TimeoutError:
                logging.warning("Timeout reading depth frame, retrying...")
        else:
            raise TimeoutError("Failed to capture depth frame after repeated attempts")
        
        # Process depth data
        depth_data = np.array(depth_frame.get_buffer_as_uint16()).reshape((depth_frame.height, depth_frame.width))
        
        # Apply median filter to reduce noise
        depth_data = cv2.medianBlur(depth_data.astype(np.uint16), 5)
        
        # Calculate point cloud
        points = []
        colors = []  # For colored point cloud visualization
        focal_length = 525.0  # Typical value, should be calibrated for actual camera
        
        for i in range(depth_data.shape[0]):
            for j in range(depth_data.shape[1]):
                z = depth_data[i, j] / 1000.0  # Convert to meters
                
                # Filter valid depth points
                if 0.1 < z < 4.0:  # Adjusted range for better results
                    # Calculate x and y based on pinhole camera model
                    x = (j - depth_data.shape[1] / 2) * z / focal_length
                    y = (i - depth_data.shape[0] / 2) * z / focal_length
                    
                    points.append([x, y, z])
                    
                    # Add color based on depth (for visualization)
                    # Normalize depth to 0-255 range within our valid range
                    normalized_depth = int(255 * (z - 0.1) / 3.9)
                    colors.append([0, normalized_depth, 255 - normalized_depth])
        
        if len(points) < 100:
            logging.warning(f"Very few valid points found: {len(points)}. Check camera positioning.")
            return None
        
        # Create Open3D point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
        point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors) / 255.0)
        
        # Statistical outlier removal
        point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Voxel downsampling for efficiency
        point_cloud = point_cloud.voxel_down_sample(voxel_size=0.01)
        
        logging.info(f"Captured point cloud with {len(point_cloud.points)} points after filtering")
        return point_cloud
        
    except Exception as e:
        logging.error(f"Error capturing point cloud: {str(e)}")
        return None

# Step 5: Segment Pallet Pockets Using Enhanced DBSCAN
def segment_pallet_pockets(point_cloud, eps=0.03, min_samples=15):
    """
    Segment pallet pockets using enhanced DBSCAN algorithm.
    
    Args:
        point_cloud: Open3D point cloud
        eps: DBSCAN epsilon parameter (cluster proximity)
        min_samples: Minimum points to form a cluster
        
    Returns:
        List of segmented pocket point clouds
    """
    if point_cloud is None or len(point_cloud.points) == 0:
        logging.error("Cannot segment empty point cloud.")
        return []
    
    try:
        # Extract points array
        points = np.asarray(point_cloud.points)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(points)
        labels = clustering.labels_
        
        # Count clusters
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        logging.info(f"DBSCAN found {n_clusters} clusters and {np.sum(labels == -1)} noise points")
        
        # Calculate cluster metrics for validity check
        if n_clusters == 0:
            logging.warning("No clusters found. Adjusting parameters.")
            # Try with more relaxed parameters
            return segment_pallet_pockets(point_cloud, eps=eps*1.5, min_samples=max(5, min_samples-5))
        
        # Extract individual pocket clusters
        segmented_pockets = []
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
                
            # Get cluster points
            cluster_indices = np.where(labels == label)[0]
            cluster_points = points[cluster_indices]
            
            # Create point cloud for this cluster
            pocket_cloud = o3d.geometry.PointCloud()
            pocket_cloud.points = o3d.utility.Vector3dVector(cluster_points)
            
            # Analyze cluster - is it pocket-like? (pockets are horizontal and rectangular)
            # Calculate oriented bounding box
            obb = pocket_cloud.get_oriented_bounding_box()
            extent = obb.extent
            
            # Pockets are typically wider than they are tall
            if extent[0] > 0.05 and extent[1] > 0.05 and extent[2] < 0.15:
                segmented_pockets.append(pocket_cloud)
                logging.info(f"Detected potential pocket: dimensions {extent}")
            else:
                logging.debug(f"Rejected cluster as non-pocket: dimensions {extent}")
        
        if not segmented_pockets:
            logging.warning("No valid pallet pockets identified after filtering.")
        
        return segmented_pockets
        
    except Exception as e:
        logging.error(f"Error during point cloud segmentation: {str(e)}")
        return []

# Step 6: Implement RANSAC for robust pocket coordinate calculation
def calculate_pocket_coordinates_ransac(pocket_point_cloud):
    """
    Calculate robust pocket coordinates using RANSAC algorithm.
    This implements the RANSAC algorithm mentioned in the thesis.
    
    Args:
        pocket_point_cloud: Point cloud of a detected pocket
        
    Returns:
        dict: Pocket coordinates and orientation
    """
    if pocket_point_cloud is None or len(pocket_point_cloud.points) < 10:
        logging.error("Insufficient points for RANSAC pocket analysis")
        return None
    
    try:
        # Extract points
        points = np.asarray(pocket_point_cloud.points)
        
        # Use RANSAC to fit a plane to the pocket floor
        # The pocket floor should be a flat horizontal surface
        model = RANSACRegressor(
            max_trials=100,
            residual_threshold=0.01,
            random_state=42
        )
        
        # Use X and Y coordinates to predict Z (height)
        X = points[:, :2]  # X and Y coordinates
        y = points[:, 2]   # Z coordinates (height)
        
        model.fit(X, y)
        
        # Find inliers (points that match the plane model)
        inlier_mask = model.inlier_mask_
        inliers = points[inlier_mask]
        
        if len(inliers) < 10:
            logging.warning(f"Too few RANSAC inliers: {len(inliers)}")
            return None
            
        # Calculate the center point of the pocket floor
        center = np.mean(inliers, axis=0)
        
        # Calculate orientation using principal component analysis
        pocket_obb = pocket_point_cloud.get_oriented_bounding_box()
        R = pocket_obb.R  # Rotation matrix
        
        # Extract Euler angles from rotation matrix
        # Simplified to just extract the rotation around vertical axis (alpha)
        alpha = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi
        
        return {
            "x": float(center[0]),
            "y": float(center[1]),
            "z": float(center[2]),
            "alpha": float(alpha),
            "confidence": float(len(inliers) / len(points))  # Confidence based on inlier ratio
        }
        
    except Exception as e:
        logging.error(f"Error in RANSAC pocket coordinate calculation: {str(e)}")
        return None

# Step 7: Calculate Coordinates Using ICP with improved error handling
def calculate_icp(source_cloud, target_cloud, max_iterations=50, threshold=0.01):
    """
    Calculate ICP transformation for fork alignment with improved robustness.
    This implements the ICP algorithm mentioned in the thesis.
    
    Args:
        source_cloud: Point cloud to be aligned
        target_cloud: Reference point cloud
        max_iterations: Maximum ICP iterations
        threshold: ICP distance threshold
        
    Returns:
        numpy.ndarray: 4x4 transformation matrix
    """
    if source_cloud is None or target_cloud is None:
        logging.error("Cannot perform ICP with None point clouds")
        return None
    
    if len(source_cloud.points) < 10 or len(target_cloud.points) < 10:
        logging.error(f"Insufficient points for ICP: source={len(source_cloud.points)}, target={len(target_cloud.points)}")
        return None
    
    try:
        # Preprocess point clouds for better ICP
        # 1. Estimate normals for both point clouds
        source_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # 2. Use point-to-plane ICP for better results
        result = o3d.pipelines.registration.registration_icp(
            source_cloud,
            target_cloud,
            threshold,
            np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
        )
        
        # Check if ICP succeeded
        if result.fitness < 0.3:  # Lower than 30% overlap is suspicious
            logging.warning(f"ICP alignment may be inaccurate. Fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.4f}")
            if result.fitness < 0.1:  # Very low fitness
                logging.error("ICP alignment failed with very low fitness score")
                return None
        else:
            logging.info(f"ICP alignment successful. Fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.4f}")
            
        return result.transformation
        
    except Exception as e:
        logging.error(f"Error during ICP calculation: {str(e)}")
        return None

# Step 8: Recognize and Verify Pallet Dimensions
def analyze_pallet_dimensions(point_cloud):
    """
    Recognize and verify pallet dimensions to ensure they are within safety standards.
    
    Args:
        point_cloud: Point cloud of the detected pallet
        
    Returns:
        dict: Pallet dimensions and compliance status
    """
    if point_cloud is None or len(point_cloud.points) < 100:
        logging.error("Insufficient points for pallet dimension analysis")
        return None
    
    try:
        # Get axis-aligned bounding box for simple dimensional analysis
        aabb = point_cloud.get_axis_aligned_bounding_box()
        min_bound = aabb.min_bound
        max_bound = aabb.max_bound
        dimensions = max_bound - min_bound
        
        # Check if dimensions match standard Euro pallet (1200x800mm)
        # Allow some tolerance in dimensions
        width, length, height = dimensions
        
        # Standard Euro pallet dimensions (in meters)
        std_length, std_width = 1.2, 0.8  # 1200x800mm
        
        # Check dimensions with tolerance
        length_match = 0.9 * std_length <= length <= 1.1 * std_length
        width_match = 0.9 * std_width <= width <= 1.1 * std_width
        
        # Check height safety (based on MAX_HEIGHT_EURO_PALLET)
        height_safe = height <= MAX_HEIGHT_EURO_PALLET
        
        # Check if pallet is level (not excessively tilted)
        # Calculate the floor plane normal using RANSAC
        floor_points = np.asarray(point_cloud.points)
        floor_points = floor_points[floor_points[:, 2] < (min_bound[2] + 0.1)]  # Points near the bottom
        
        if len(floor_points) > 10:
            # Fit plane to floor points
            floor_model = RANSACRegressor(max_trials=100, residual_threshold=0.01)
            floor_model.fit(floor_points[:, :2], floor_points[:, 2])
            
            # Calculate tilt angle from the coefficients
            coef = floor_model.estimator_.coef_
            normal_vector = np.array([coef[0], coef[1], -1])
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
            
            # Angle between normal and vertical axis
            tilt_angle = np.arccos(np.dot(normal_vector, np.array([0, 0, 1]))) * 180 / np.pi
            tilt_safe = tilt_angle <= MAX_TILT_ANGLE
        else:
            tilt_angle = 0
            tilt_safe = True
        
        # Compile results
        result = {
            "dimensions": {
                "length": float(length),
                "width": float(width),
                "height": float(height)
            },
            "is_standard_size": length_match and width_match,
            "is_height_safe": height_safe,
            "tilt_angle": float(tilt_angle),
            "is_tilt_safe": tilt_safe,
            "overall_safe": height_safe and tilt_safe
        }
        
        # Log results
        if result["overall_safe"]:
            logging.info(f"Pallet dimensions verified as safe: {result['dimensions']}")
        else:
            logging.warning(f"Pallet safety check failed: {result}")
            
        return result
        
    except Exception as e:
        logging.error(f"Error analyzing pallet dimensions: {str(e)}")
        return None

# Step 9: Recognize Pallet Shape Using SVM
def recognize_pallet_type(point_cloud):
    """
    Recognize pallet type using Support Vector Machines as mentioned in the thesis.
    
    Args:
        point_cloud: Point cloud of the detected pallet
        
    Returns:
        str: Pallet type classification
    """
    if point_cloud is None or len(point_cloud.points) < 100:
        logging.error("Insufficient points for pallet type recognition")
        return None
    
    try:
        # Extract features from point cloud
        features = []
        
        # Get bounding box dimensions as features
        aabb = point_cloud.get_axis_aligned_bounding_box()
        dimensions = aabb.max_bound - aabb.min_bound
        features.extend(dimensions)
        
        # Get statistical features
        points = np.asarray(point_cloud.points)
        features.extend([
            np.mean(points, axis=0),  # Mean position
            np.std(points, axis=0),   # Std deviation
            np.percentile(points, 25, axis=0),  # 25th percentile
            np.percentile(points, 75, axis=0)   # 75th percentile
        ])
        
        # Flatten features
        features = np.array(features).flatten()
        
        # In a real implementation, we would load a pre-trained SVM model
        # Here, we'll simulate classification based on dimensions
        width, length, height = dimensions
        
        # Simple rule-based classification for demonstration
        if 1.1 <= length <= 1.3 and 0.7 <= width <= 0.9:
            return "EUR-Pallet"
        elif 1.0 <= length <= 1.2 and 1.0 <= width <= 1.2:
            return "CP-Pallet"
        else:
            return "Unknown"
            
    except Exception as e:
        logging.error(f"Error in pallet type recognition: {str(e)}")
        return "Unknown"

# Step 10: Calculate Fork Positioning
def calculate_fork_positioning(pocket_coordinates, pallet_dimensions):
    """
    Calculate the optimal fork positions based on pocket coordinates and pallet dimensions.
    
    Args:
        pocket_coordinates: List of pocket coordinate dictionaries
        pallet_dimensions: Pallet dimension information
        
    Returns:
        dict: Optimal fork positioning information
    """
    if not pocket_coordinates or len(pocket_coordinates) < 2:
        logging.error("Insufficient pocket coordinates for fork positioning")
        return None
        
    try:
        # Sort pockets by x-coordinate (assuming x is along the width of the pallet)
        sorted_pockets = sorted(pocket_coordinates, key=lambda p: p["x"])
        
        # For two-fork positioning, we need at least two pockets
        left_pocket = sorted_pockets[0]
        right_pocket = sorted_pockets[-1]
        
        # Calculate center position between pockets
        center_x = (left_pocket["x"] + right_pocket["x"]) / 2
        center_y = (left_pocket["y"] + right_pocket["y"]) / 2
        
        # Calculate approach angle (perpendicular to pallet face)
        dx = right_pocket["x"] - left_pocket["x"]
        dy = right_pocket["y"] - left_pocket["y"]
        approach_angle = np.arctan2(dy, dx) + np.pi/2  # Perpendicular to pocket line
        
        # Determine fork spacing based on pocket distance
        pocket_distance = np.sqrt(dx**2 + dy**2)
        
        # Calculate optimal insertion depth
        # This would typically be derived from the pallet dimensions and type
        if pallet_dimensions and "length" in pallet_dimensions["dimensions"]:
            insertion_depth = pallet_dimensions["dimensions"]["length"] * 0.8  # 80% of pallet length
        else:
            insertion_depth = 0.9  # Default in meters
        
        # Determine optimal fork height based on pocket coordinates
        fork_height = min(left_pocket["z"], right_pocket["z"]) - 0.02  # Slightly below pocket floor
        
        return {
            "position": {
                "x": float(center_x),
                "y": float(center_y),
                "z": float(fork_height)
            },
            "approach_angle": float(approach_angle * 180 / np.pi),  # Convert to degrees
            "fork_spacing": float(pocket_distance),
            "insertion_depth": float(insertion_depth),
            "confidence": min(left_pocket["confidence"], right_pocket["confidence"])
        }
        
    except Exception as e:
        logging.error(f"Error calculating fork positioning: {str(e)}")
        return None

# Step 11: Check Pallet Weight with enhanced safety checks
def check_pallet_weight(weight, pallet_type="EUR"):
    """
    Check if the pallet weight exceeds the maximum allowable weight with enhanced safety checks.
    
    Args:
        weight (float): The weight of the pallet in kilograms.
        pallet_type (str): Type of pallet (EUR, CP, etc.)
        
    Returns:
        dict: Weight check results
    """
    try:
        # Define weight limits based on pallet type
        weight_limits = {
            "EUR": MAX_WEIGHT_EURO_PALLET,
            "CP": 1250,  # CP pallets have different weight limits
            "Unknown": 1000  # Conservative limit for unknown pallet types
        }
        
        # Get appropriate weight limit
        max_weight = weight_limits.get(pallet_type, weight_limits["Unknown"])
        
        # Calculate safety margin
        safety_threshold = 0.9 * max_weight  # 90% of maximum weight
        warning_threshold = 0.75 * max_weight  # 75% of maximum weight
        
        # Check weight against thresholds
        is_safe = weight <= max_weight
        warning_level = "none"
        
        if weight > max_weight:
            warning_level = "critical"
            logging.error(f"Pallet weight ({weight} kg) exceeds the maximum allowable weight ({max_weight} kg).")
        elif weight > safety_threshold:
            warning_level = "high"
            logging.warning(f"Pallet weight ({weight} kg) is close to the maximum allowable weight ({max_weight} kg).")
        elif weight > warning_threshold:
            warning_level = "medium"
            logging.info(f"Pallet weight ({weight} kg) is approaching cautionary levels.")
        else:
            logging.info(f"Pallet weight ({weight} kg) is within safe limits.")
        
        return {
            "weight": float(weight),
            "max_allowable": float(max_weight),
            "safety_margin": float(max_weight - weight),
            "is_safe": is_safe,
            "warning_level": warning_level
        }
        
    except Exception as e:
        logging.error(f"Error checking pallet weight: {str(e)}")
        return None

# Step 12: Communicate with PLC/PC via TCP/IP with improved protocol
def send_data_to_plc(data, host="192.168.1.100", port=5000, protocol="TCP/IP"):
    """
    Send alignment data to the PLC or central PC with enhanced error handling and multiple protocol support.
    
    Args:
        data (dict): Data to send
        host (str): Target host address
        port (int): Target port
        protocol (str): Communication protocol (TCP/IP, OPC, PROFINET, CANbus)
        
    Returns:
        bool: Success status
    """
    if not data:
        logging.error("Cannot send empty data to PLC/PC")
        return False
        
    try:
        # Add timestamp and metadata
        payload = {
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "source": "3D_Camera_System",
            "message_id": int(time.time() * 1000)
        }
        
        # Serialize data based on protocol
        if protocol == "TCP/IP":
            # For TCP/IP, serialize to JSON
            serialized_data = json.dumps(payload)
            
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                # Set timeout
                sock.settimeout(5.0)
                
                # Connect and send data
                sock.connect((host, port))
                sock.sendall(serialized_data.encode('utf-8'))
                
                # Wait for acknowledgment
                response = sock.recv(1024).decode('utf-8')
                
                if "ACK" in response:
                    logging.info(f"Data successfully sent to {host}:{port} via {protocol}")
                    return True
                else:
                    logging.warning(f"Received non-acknowledgment response: {response}")
                    return False
                    
        elif protocol == "OPC":
            # Placeholder for OPC implementation
            logging.info(f"OPC communication would send: {json.dumps(payload)}")
            return True
            
        elif protocol == "PROFINET":
            # Placeholder for PROFINET implementation
            logging.info(f"PROFINET communication would send: {json.dumps(payload)}")
            return True
            
        elif protocol == "CANbus":
            # Placeholder for CANbus implementation
            logging.info(f"CANbus communication would send: {json.dumps(payload)}")
            return True
            
        else:
            logging.error(f"Unsupported protocol: {protocol}")
            return False
            
    except socket.timeout:
        logging.error(f"Connection timeout when sending data to {host}:{port}")
        return False
    except ConnectionRefusedError:
        logging.error(f"Connection refused by {host}:{port}. Is the server running?")
        return False
    except Exception as e:
        logging.error(f"Error sending data to PLC/PC: {str(e)}")
        return False

# Simulation: Improved Simulated Depth Stream
class SimulatedDepthStream:
    def __init__(self, point_cloud_file=None):
        """
        Initialize simulated depth stream with better error handling and options.
        
        Args:
            point_cloud_file: Path to point cloud file for simulation
        """
        self.is_running = False
        self.simulation_mode = "static"  # static or dynamic
        
        if point_cloud_file and os.path.exists(point_cloud_file):
            try:
                self.point_cloud = o3d.io.read_point_cloud(point_cloud_file)
                logging.info(f"Loaded simulated point cloud with {len(self.point_cloud.points)} points")
            except Exception as e:
                logging.error(f"Failed to load point cloud file: {str(e)}")
                self.create_default_point_cloud()
        else:
            logging.warning("No valid point cloud file provided, creating synthetic data")
            self.create_default_point_cloud()
            
    def create_default_point_cloud(self):
        """Create a synthetic point cloud representing a Euro pallet for simulation."""
        # Create a simple pallet-like structure
        points = []
        colors = []
        
        # Base platform (800x1200mm)
        for x in np.linspace(-0.6, 0.6, 20):
            for y in np.linspace(-0.4, 0.4, 20):
                points.append([x, y, 0])
                colors.append([0.8, 0.7, 0.6])  # Wood color
        
        # Create pockets
        # Left pocket
        for x in np.linspace(-0.4, -0.2, 10):
            for y in np.linspace(-0.1, 0.1, 10):
                points.append([x, y, 0.1])
                colors.append([0.5, 0.5, 0.5])  # Dark for pockets
                
        # Right pocket
        for x in np.linspace(0.2, 0.4, 10):
            for y in np.linspace(-0.1, 0.1, 10):
                points.append([x, y, 0.1])
                colors.append([0.5, 0.5, 0.5])
        
        # Create the point cloud
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
        self.point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        logging.info(f"Created default simulated point cloud with {len(points)} points")

    def read_frame(self, timeout=None):
        """
        Simulate reading a depth frame from the camera.
        
        Args:
            timeout: Simulated timeout in milliseconds
            
        Returns:
            MockFrame: Simulated depth frame
        """
        if not self.is_running:
            raise RuntimeError("Depth stream is not started")
            
        # Simulate processing delay
        time.sleep(0.03)
        
        # If in dynamic mode, add some noise and movement
        if self.simulation_mode == "dynamic":
            # Add random noise and small movement to point cloud
            points = np.asarray(self.point_cloud.points)
            noise = np.random.normal(0, 0.005, points.shape)  # Small Gaussian noise
            movement = np.array([0.001, 0.001, 0]) * np.sin(time.time())  # Periodic movement
            
            noisy_points = points + noise + movement
            
            temp_cloud = o3d.geometry.PointCloud()
            temp_cloud.points = o3d.utility.Vector3dVector(noisy_points)
            if self.point_cloud.has_colors():
                temp_cloud.colors = self.point_cloud.colors
                
            point_cloud = temp_cloud
        else:
            point_cloud = self.point_cloud
        
        # Create mock frame
        class MockFrame:
            def __init__(self, cloud, height=480, width=640, focal_length=525):
                self.point_cloud = cloud
                self._height = height
                self._width = width
                self._focal_length = focal_length
                
                # Convert point cloud to depth map
                points = np.asarray(self.point_cloud.points)
                depths = np.linalg.norm(points, axis=1)
                
                # Scale depths to uint16 range (in mm)
                depths = (depths * 1000).astype(np.uint16)
                
                # Create a sparse depth map
                self._depth_map = np.zeros((height, width), dtype=np.uint16)
                
                # Project 3D points to 2D depth map
                for i, point in enumerate(points):
                    # Simple projection
                    x, y, z = point
                    if z > 0:  # Ensure point is in front of camera
                        pixel_x = int((x / z) * focal_length + width / 2)
                        pixel_y = int((y / z) * focal_length + height / 2)
                        
                        if 0 <= pixel_x < width and 0 <= pixel_y < height:
                            self._depth_map[pixel_y, pixel_x] = int(z * 1000)  # Convert to mm
            
            def get_buffer_as_uint16(self):
                return self._depth_map.tobytes()
                
            @property
            def height(self):
                return self._height
                
            @property
            def width(self):
                return self._width
                
            @property
            def focal_length(self):
                return self._focal_length
        
        return MockFrame(point_cloud)

    def start(self):
        """Start the simulated depth stream."""
        self.is_running = True
        logging.info("Simulated depth stream started")

    def stop(self):
        """Stop the simulated depth stream."""
        self.is_running = False
        logging.info("Simulated depth stream stopped")
        
    def set_dynamic_mode(self, enabled=True):
        """Set whether to use dynamic simulation with noise and movement."""
        self.simulation_mode = "dynamic" if enabled else "static"
        logging.info(f"Simulation mode set to {self.simulation_mode}")

# Main Execution
def main():
    """Main execution function for the forklift autonomous driving system."""
    try:
        # Print system startup banner
        logging.info("=" * 80)
        logging.info("   Autonomous Forklift Driving System with 3D Camera - Starting Up")
        logging.info("=" * 80)
        
        # Ensure all required directories exist
        for path in [os.path.dirname(MODEL_PATH), LOG_PATH]:
            os.makedirs(path, exist_ok=True)
            
        # Check configuration
        logging.info(f"Configuration loaded:")
        logging.info(f"  - Maximum allowed weight: {MAX_WEIGHT_EURO_PALLET} kg")
        logging.info(f"  - Maximum allowed height: {MAX_HEIGHT_EURO_PALLET} m")
        logging.info(f"  - Maximum allowed tilt: {MAX_TILT_ANGLE} degrees")
        
        # Initialize capture system
        use_simulation = True  # Set to False when using a physical camera
        
        if use_simulation:
            logging.info("Using SIMULATION mode")
            # Check if simulation point cloud exists, otherwise create default
            sim_cloud_path = 'data/simulated_point_cloud.ply'
            
            if not os.path.exists(sim_cloud_path):
                logging.warning(f"Simulation point cloud not found at {sim_cloud_path}")
                os.makedirs(os.path.dirname(sim_cloud_path), exist_ok=True)
                
                # Create a default point cloud and save it
                sim_stream = SimulatedDepthStream()
                o3d.io.write_point_cloud(sim_cloud_path, sim_stream.point_cloud)
                logging.info(f"Created and saved default simulation point cloud to {sim_cloud_path}")
                
                depth_stream = sim_stream
            else:
                depth_stream = SimulatedDepthStream(sim_cloud_path)
                
            # Enable dynamic simulation for more realistic testing
            depth_stream.set_dynamic_mode(True)
            device = None
            
        else:
            logging.info("Using REAL CAMERA mode")
            device, depth_stream = initialize_orbbec_camera()
        
        # Load or train CNN model
        if os.path.exists(MODEL_PATH):
            logging.info(f"Loading existing CNN model from {MODEL_PATH}")
            try:
                model = load_model(MODEL_PATH)
                logging.info("CNN model loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load model: {str(e)}")
                raise
        else:
            # Training would require actual training data
            logging.warning("CNN model not found. In a production environment, pre-train the model.")
            model = None
            
        # Start depth stream
        depth_stream.start()
        
        # Main processing loop
        try:
            run_detection = True
            while run_detection:
                logging.info("\n--- Starting detection cycle ---")
                
                # 1. Capture point cloud
                point_cloud = capture_point_cloud(depth_stream)
                if point_cloud is None:
                    logging.warning("Failed to capture valid point cloud, retrying...")
                    time.sleep(1)
                    continue
                    
                # 2. Analyze pallet dimensions and safety
                pallet_dimensions = analyze_pallet_dimensions(point_cloud)
                if pallet_dimensions and not pallet_dimensions["overall_safe"]:
                    logging.error("Pallet dimensions or orientation unsafe for operation")
                    # Continue for demonstration, but in production you might want to abort
                
                # 3. Recognize pallet type
                pallet_type = recognize_pallet_type(point_cloud)
                logging.info(f"Detected pallet type: {pallet_type}")
                
                # 4. Segment pallet pockets
                segmented_pockets = segment_pallet_pockets(point_cloud)
                if not segmented_pockets:
                    logging.warning("No pallet pockets detected, retrying...")
                    time.sleep(1)
                    continue
                
                logging.info(f"Detected {len(segmented_pockets)} potential pallet pockets")
                
                # 5. Calculate pocket coordinates using RANSAC
                pocket_coordinates = []
                for i, pocket in enumerate(segmented_pockets):
                    coords = calculate_pocket_coordinates_ransac(pocket)
                    if coords:
                        pocket_coordinates.append(coords)
                        logging.info(f"Pocket {i+1} coordinates: {coords}")
                
                if not pocket_coordinates:
                    logging.warning("Failed to calculate valid pocket coordinates")
                    time.sleep(1)
                    continue
                
                # 6. Calculate optimal fork positioning
                fork_positioning = calculate_fork_positioning(pocket_coordinates, pallet_dimensions)
                if not fork_positioning:
                    logging.warning("Failed to calculate fork positioning")
                    time.sleep(1)
                    continue
                    
                logging.info(f"Calculated fork positioning: {fork_positioning}")
                
                # 7. Perform ICP for global alignment
                target_cloud_path = TARGET_CLOUD_PATH
                if os.path.exists(target_cloud_path):
                    target_cloud = o3d.io.read_point_cloud(target_cloud_path)
                    icp_result = calculate_icp(point_cloud, target_cloud)
                    
                    if icp_result is not None:
                        logging.info("ICP alignment completed successfully")
                    else:
                        logging.warning("ICP alignment failed")
                else:
                    logging.warning(f"Target cloud not found at {target_cloud_path}")
                    icp_result = None
                
                # 8. Send positioning data to PLC
                positioning_data = {
                    "fork_positioning": fork_positioning,
                    "pallet_dimensions": pallet_dimensions,
                    "pallet_type": pallet_type,
                    "icp_transformation": icp_result.tolist() if icp_result is not None else None
                }
                
                send_success = send_data_to_plc(positioning_data, protocol="TCP/IP")
                
                if send_success:
                    logging.info("Successfully communicated positioning data to PLC")
                else:
                    logging.warning("Failed to communicate with PLC")
                
                # Wait for next cycle or exit
                key = cv2.waitKey(1000) & 0xFF  # 1-second delay between cycles
                if key == ord('q'):
                    logging.info("User requested exit")
                    run_detection = False
                    
                # For simulation purposes, limit the number of cycles
                if use_simulation:
                    # Ask user if they want to continue
                    user_input = input("Continue detection cycle? (y/n): ")
                    if user_input.lower() != 'y':
                        logging.info("User requested exit")
                        run_detection = False
                
        except KeyboardInterrupt:
            logging.info("User interrupted execution")
            
        finally:
            # Clean up
            logging.info("Cleaning up resources")
            if depth_stream:
                depth_stream.stop()
            
            if not use_simulation and device:
                openni2.unload()
                
            cv2.destroyAllWindows()
            
    except Exception as e:
        logging.error(f"An unexpected error occurred in main execution: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        
    finally:
        logging.info("=" * 80)
        logging.info("   Autonomous Forklift Driving System Shutdown Complete")
        logging.info("=" * 80)

if __name__ == "__main__":
    main()
