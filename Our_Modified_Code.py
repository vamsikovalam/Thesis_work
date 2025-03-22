import cv2
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import openni2
import time
import logging
import os
import socket

# Setup logging for debugging and tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths and Parameters
TRAIN_DATA_PATH = 'data/pallet/train'
TEST_DATA_PATH = 'data/pallet/test'
MODEL_PATH = 'models/pallet_detection_cnn.h5'
REFERENCE_PALLET_PATH = 'data/reference_euro_pallet.ply'
TARGET_CLOUD_PATH = 'data/fork_target.ply'

# Maximum allowable weight for Euro pallets (in kg)
MAX_WEIGHT_EURO_PALLET = 1500  # As per EPAL standards

# Step 1: Data Preprocessing and Collection
def setup_data_generators(train_path, test_path):
    """Set up data augmentation for training dataset."""
    try:
        image_gen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        train_gen = image_gen.flow_from_directory(
            train_path,
            target_size=(400, 400),
            batch_size=16,
            class_mode='binary'
        )
        test_gen = image_gen.flow_from_directory(
            test_path,
            target_size=(400, 400),
            batch_size=16,
            class_mode='binary'
        )
        return train_gen, test_gen
    except Exception as e:
        logging.error(f"Error setting up data generators: {e}")
        exit(1)

# Step 2: Define and Train the CNN Model
def create_and_train_cnn(train_gen, test_gen, model_path):
    """Define and train the CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(400, 400, 3), activation='relu', name='Conv1'),
        MaxPooling2D(pool_size=(2, 2), name='Pool1'),
        Conv2D(64, (3, 3), activation='relu', name='Conv2'),
        MaxPooling2D(pool_size=(2, 2), name='Pool2'),
        Conv2D(128, (3, 3), activation='relu', name='Conv3'),
        MaxPooling2D(pool_size=(2, 2), name='Pool3'),
        Flatten(name='Flatten'),
        Dense(128, activation='relu', name='FC1'),
        Dropout(0.5, name='Dropout1'),
        Dense(1, activation='sigmoid', name='Output')
    ])
    try:
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        logging.info("Model summary:")
        model.summary()
        model.fit(train_gen, epochs=20, validation_data=test_gen, steps_per_epoch=100, validation_steps=12)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        logging.info(f"Model saved at {model_path}")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        exit(1)

# Step 3: Initialize Orbbec Gemini 2 Camera
def initialize_orbbec_camera():
    """Initialize the Orbbec Gemini 2 camera."""
    try:
        openni2.initialize()  # Load OpenNI drivers
        dev = openni2.Device.open_any()  # Connect to the Orbbec camera
        depth_stream = dev.create_depth_stream()
        depth_stream.start()
        logging.info("Orbbec Gemini 2 camera initialized successfully.")
        return depth_stream
    except Exception as e:
        logging.error(f"Failed to initialize Orbbec Gemini 2 camera: {e}")
        exit(1)

# Step 4: Capture Point Cloud Data
def capture_point_cloud(depth_stream):
    """Capture point cloud data from the Orbbec Gemini 2 camera."""
    try:
        depth_frame = depth_stream.read_frame()
        depth_data = np.array(depth_frame.get_buffer_as_uint16()).reshape((depth_frame.height, depth_frame.width))
    except Exception as e:
        logging.error(f"Failed to capture depth frame: {e}")
        return None

    points = []
    for i in range(depth_data.shape[0]):
        for j in range(depth_data.shape[1]):
            z = depth_data[i, j] / 1000.0  # Convert to meters
            if 0 < z < 5.0:  # Filter valid depth points within a reasonable range
                x = (j - depth_data.shape[1] / 2) * z / depth_frame.focal_length
                y = (i - depth_data.shape[0] / 2) * z / depth_frame.focal_length
                points.append([x, y, z])

    if len(points) == 0:
        logging.warning("No valid points found in depth data.")
        return None

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    return point_cloud

# Step 5: Segment Pallet Pockets Using DBSCAN
def segment_pallet_pockets(point_cloud, eps=0.05, min_samples=10):
    """Segment pallet pockets using DBSCAN."""
    if point_cloud is None:
        logging.error("Point cloud is None. Cannot proceed with segmentation.")
        return []

    points = np.asarray(point_cloud.points)
    if len(points) == 0:
        logging.error("Point cloud is empty. Exiting segmentation.")
        return []

    try:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_

        unique_labels = set(labels)
        segmented_pockets = [points[labels == label] for label in unique_labels if label != -1]
        return segmented_pockets
    except Exception as e:
        logging.error(f"Error during point cloud segmentation: {e}")
        return []

# Step 6: Calculate Coordinates Using Iterative Closest Point (ICP)
def calculate_icp(source_cloud, target_cloud, threshold=0.02):
    """Calculate ICP transformation for fork alignment."""
    if source_cloud is None or target_cloud is None:
        logging.error("Source or target cloud is None. Cannot proceed with ICP.")
        return None

    try:
        transformation = o3d.pipelines.registration.registration_icp(
            source_cloud, target_cloud, threshold, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        ).transformation
        return transformation
    except Exception as e:
        logging.error(f"Error during ICP calculation: {e}")
        return None

# Step 7: Recognize Pallet Shape Using Reference Model
def recognize_pallet_shape(point_cloud, reference_path):
    """Recognize pallet shape using point cloud registration."""
    if point_cloud is None or not os.path.exists(reference_path):
        logging.error(f"Reference pallet point cloud file does not exist: {reference_path}")
        return None

    reference_pallet = o3d.io.read_point_cloud(reference_path)

    try:
        reg_p2p = o3d.pipelines.registration.registration_icp(
            point_cloud, reference_pallet, 0.05, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        return reg_p2p.transformation
    except Exception as e:
        logging.error(f"Error during pallet shape recognition: {e}")
        return None

# Step 8: Check Pallet Weight
def check_pallet_weight(weight):
    """
    Check if the pallet weight exceeds the maximum allowable weight.
    
    Args:
        weight (float): The weight of the pallet in kilograms.
    
    Returns:
        bool: True if the weight is within limits, False otherwise.
    """
    if weight > MAX_WEIGHT_EURO_PALLET:
        logging.error(f"Pallet weight ({weight} kg) exceeds the maximum allowable weight ({MAX_WEIGHT_EURO_PALLET} kg).")
        logging.warning("Task reset due to excessive weight.")
        return False
    logging.info(f"Pallet weight ({weight} kg) is within the allowable limit.")
    return True

# Step 9: Communicate with PLC/PC via TCP/IP
def send_data_to_plc(data, host="192.168.1.100", port=5000):
    """Send alignment data to the PLC or central PC via TCP/IP."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            sock.sendall(data.encode('utf-8'))
            logging.info(f"Data sent to {host}:{port}: {data}")
    except Exception as e:
        logging.error(f"Error communicating with PLC/PC: {e}")

# Simulation: Simulate Depth Stream Without Physical Camera
class SimulatedDepthStream:
    def __init__(self, point_cloud_file):
        self.point_cloud = o3d.io.read_point_cloud(point_cloud_file)

    def read_frame(self):
        class MockFrame:
            def get_buffer_as_uint16(self):
                points = np.asarray(self.point_cloud.points)
                depth_map = np.linalg.norm(points, axis=1).reshape((480, 640)).astype(np.uint16)
                return depth_map.tobytes()

            @property
            def height(self):
                return 480

            @property
            def width(self):
                return 640

            @property
            def focal_length(self):
                return 500  # Simulated focal length

        return MockFrame()

    def start(self):
        pass

    def stop(self):
        pass

# Main Execution
if __name__ == "__main__":
    try:
        # Ensure paths exist
        if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
            logging.error(f"Data paths do not exist: {TRAIN_DATA_PATH}, {TEST_DATA_PATH}")
            exit(1)

        if not os.path.exists(REFERENCE_PALLET_PATH):
            logging.error(f"Reference pallet path does not exist: {REFERENCE_PALLET_PATH}")
            exit(1)

        if not os.path.exists(TARGET_CLOUD_PATH):
            logging.error(f"Target cloud path does not exist: {TARGET_CLOUD_PATH}")
            exit(1)

        # Initialize Orbbec Gemini 2 camera or use simulation
        use_simulation = True  # Set to False if using a physical camera
        if use_simulation:
            depth_stream = SimulatedDepthStream('data/simulated_point_cloud.ply')
        else:
            depth_stream = initialize_orbbec_camera()

        # Load or train CNN model
        if not os.path.exists(MODEL_PATH):
            logging.info("Training new CNN model...")
            train_gen, test_gen = setup_data_generators(TRAIN_DATA_PATH, TEST_DATA_PATH)
            create_and_train_cnn(train_gen, test_gen, MODEL_PATH)
        model = load_model(MODEL_PATH)

        # Simulated pallet weight (replace with actual weight sensor reading)
        pallet_weight = float(input("Enter the pallet weight (in kg): "))

        # Check pallet weight
        if not check_pallet_weight(pallet_weight):
            logging.error("Task aborted due to excessive pallet weight.")
            exit(1)

        # Continuous loop for real-time processing
        while True:
            # Capture point cloud
            point_cloud = capture_point_cloud(depth_stream)
            if point_cloud is None:
                logging.warning("Failed to capture point cloud. Retrying...")
                continue

            # Segment pallet pockets
            segmented_pockets = segment_pallet_pockets(point_cloud)
            if len(segmented_pockets) == 0:
                logging.warning("No pallet pockets found during segmentation.")
                continue

            # Perform ICP and shape recognition
            target_cloud = o3d.io.read_point_cloud(TARGET_CLOUD_PATH)
            icp_transformation = calculate_icp(point_cloud, target_cloud)
            if icp_transformation is not None:
                logging.info("ICP Transformation Matrix:")
                logging.info(icp_transformation)

            pallet_transformation = recognize_pallet_shape(point_cloud, REFERENCE_PALLET_PATH)
            if pallet_transformation is not None:
                logging.info("Pallet Recognition Transformation Matrix:")
                logging.info(pallet_transformation)

            # Send data to PLC/PC
            alignment_data = {
                "x": icp_transformation[0, 3],
                "y": icp_transformation[1, 3],
                "z": icp_transformation[2, 3],
                "alpha": np.arctan2(icp_transformation[1, 0], icp_transformation[0, 0])
            }
            send_data_to_plc(str(alignment_data))

            # Exit on key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    finally:
        if depth_stream is not None:
            logging.info("Stopping depth stream.")
            depth_stream.stop()
        openni2.unload()