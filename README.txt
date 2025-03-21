DEVELOPING AN ALGORITHM FOR AUTONOMOUS FORKLIFT BY USING ORBBEC GEMINI 3D-CAMERA

Main Objective - Developing an algorithm for AGV forklift by using 3D camera in most economical way with best efficiency.

1.Introduction:
 	The aim of this project is to develop an autonomous driving system for an industrial forklift, integrated with an advanced 3D camera. This innovative solution aims to optimize loading and unloading operations, improving operational efficiency and reducing the need for human intervention. In an industrial context where speed and precision are crucial, such an autonomous system not only increases productivity, but also significantly contributes to  safety, reducing the risk of accidents and operational errors.

The system is designed to:
 • Identify the pallet: Recognizing the distinctive characteristics of the pallet and the load.
 • Check the load: Ensuring that the load complies with the overlap specifications.
 • Position the forks yourself: Automatically adjusting the position of the forklift forks for safe and effective lifting.
 • Moving the load autonomously.



2.By Using these Methods:

	 The effectiveness of the system depends largely on the algorithms implemented. The main algorithms used in the system are described in detail below, along with their advantages and specific applications:

	2.1. Pallet Pocket Detection

 		• Convolutional Neural Networks (CNN): CNNs are used to train the system on a large dataset of pallet images, allowing accurate identification of the structural features of the pockets. This approach is particularly effective in visual recognition  due to the ability of CNNs to learn and generalize complex patterns.

How it works:
 ▪ Training: CNNs are trained on a large dataset of pallet images,  including various pocket configurations.
 ▪ Feature Extraction: During training, CNNs learn to identify and  generalize the structural features of pockets, such as edges, corners,  and contours.

Advantages:
 ▪ Adaptation: Ability to adapt to new pallet patterns thanks to machine learning
 ▪ Robustness: High robustness under variable lighting and visual noise conditions.

Specific Application: CNN is able to automatically recognize pockets even in the presence of variable loads or obstacles.


	2.2.Point Cloud Segmentation: 
		
		Algorithms like DBSCAN are used to separate pockets  from other visual elements, providing a clear view of the placement areas. Segmentation is crucial to ensure that the system focuses only on the areas that are relevant for alignment.

How it works:
 ▪ Creating the Point Cloud: The camera acquires a 3D point cloud representing the surrounding environment.
 ▪ Separation: DBSCAN segments pallet pockets from other visual elements, providing a clear view of placement areas.

Advantages:
 ▪ Reliable Recognition: Identification of pockets even in the presence of variable loads or obstacles.
 ▪ Robustness: Works well in complex environments where data may be incomplete or noisy.

Application Example: Automatic pocket recognition even in the presence of variable loads or obstacles.



3. Calculating the Coordinates (x, y, z, α) for the Fork Alignment
 
	 • ICP (Iterative Closest Point): This algorithm is essential to align the camera  coordinates with those of the pallet, improving the alignment of the forks with  respect to the pockets. The ICP works iteratively to optimize the accuracy of the 
alignment, progressively reducing the error.

	 • RANSAC: Used to compute pocket coordinates robustly, even in the presence of noise in the data. RANSAC is particularly effective in handling situations where the 
observed data is incomplete or imprecise.
 
Advantages: Increase alignment robustness in real and complex environments.


 4. Recognition of the shape and dimensions of the load
 
	• Point Cloud Registration: This process determines the volume and shape of the load by comparing the acquired point cloud with predefined reference models. Point cloud registration is essential to evaluate whether the load complies with safety and  stability requirements.



5.Interaction and Data Flow

	The 3D camera, once integrated into the system, transmits the detected data through the chosen  communication bus. Here is an example of the data flow in the system:
 
1.Data Collection: The camera scans the pallet and identifies the location and  dimensions of the load. The data is processed and formatted.

2.Data Transmission: Using one of the communication protocols (TCP/IP, OPC, CANbus or  PROFINET), the data is sent to the PLC and the central computer.

3.Processing and Control: The PLC receives the data and processes it in real time.  Based on this data, the PLC can send commands to the forklift for fork positioning 
and load control.

4.Feedback and Monitoring: The system can also transmit feedback to the central PC  for operation monitoring and performance analysis, ensuring that all functions are operating as intended.


6. Conclusion
 
	In summary, the forklift autonomous driving system developed with the help of a 3D  camera represents a significant step towards optimizing industrial operations. With an  approach focused on safety, efficiency and continuous innovation, this system not only improves productivity but also contributes to a safer and more dynamic working  environment. The challenges encountered during the development process provided valuable learning opportunities, while the future potential of the system promises to redefine the logistics and freight handling landscape.