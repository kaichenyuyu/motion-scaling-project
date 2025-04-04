#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud
from ambf_msgs.msg import RigidBodyState
from geometry_msgs.msg import Point32, Pose, Point
from std_msgs.msg import Float32
import numpy as np
import math
import matplotlib.pyplot as plt
import tf.transformations as tf

class ClosestPointDistPlotter:
    def __init__(self, targets_topic: str, track_1_topic: str, track_2_topic: str):
        # Initialize the ROS node
        rospy.init_node('closest_point_dist_plotter', anonymous=True)
        
        # Set phantom transformation from YAML file first
        # These values should match the phantom position and orientation in phantom.yaml
        self.phantom_position = np.array([0.0564242899, 0.2649616003, 0.7395174503])
        # Convert from roll, pitch, yaw to quaternion
        roll, pitch, yaw = 0.0, 0.0, 0.87266469
        self.phantom_quaternion = tf.quaternion_from_euler(roll, pitch, yaw)
        # Create transformation matrix
        self.phantom_transform = tf.compose_matrix(
            translate=self.phantom_position,
            angles=tf.euler_from_quaternion(self.phantom_quaternion)
        )
        # Inverse transform (to convert world coordinates to phantom coordinates)
        self.phantom_inverse_transform = tf.inverse_matrix(self.phantom_transform)
        
        # Latest values from each topic
        self.targets = None
        self.track_1 = None
        self.track_2 = None
        
        # For plotting
        self.times = []
        self.track_1_dist_data = []
        self.track_2_dist_data = []
        self.start_time = rospy.get_time()
        
        # Setup matplotlib for interactive real-time plotting
        self.setup_plotting()
        
        # Subscribers for topics - add these AFTER initializing transforms
        self.sub_targets = rospy.Subscriber(targets_topic, PointCloud, self.targets_callback)
        self.sub_track_1 = rospy.Subscriber(track_1_topic, RigidBodyState, self.track_1_callback)
        self.sub_track_2 = rospy.Subscriber(track_2_topic, RigidBodyState, self.track_2_callback)
    
    def targets_callback(self, msg):
        # Store the target points directly as they're already in phantom frame
        self.targets = np.array([[p.x, p.y, p.z] for p in msg.points])

    def track_1_callback(self, msg):
        position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        # Transform PSM1 position from world frame to phantom frame
        self.track_1 = self.transform_to_phantom_frame(position)

    def track_2_callback(self, msg):
        position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        # Transform PSM2 position from world frame to phantom frame
        self.track_2 = self.transform_to_phantom_frame(position)
    
    def transform_to_phantom_frame(self, position):
        """Transform a position from world frame to phantom frame"""
        # Convert to homogeneous coordinates
        position_homogeneous = np.append(position, 1.0)
        # Apply inverse transform to get position in phantom's local frame
        local_position_homogeneous = np.dot(self.phantom_inverse_transform, position_homogeneous)
        # Return to 3D coordinates
        return local_position_homogeneous[:3]

    def find_closest_point_and_dist(self, targets, source):        
        # Compute Euclidean distances from each point to the reference point
        distances = np.linalg.norm(targets - source, axis=1)
        
        # Find the index of the minimum distance
        min_index = np.argmin(distances)
        
        # Retrieve the closest point and its corresponding distance
        closest_point = targets[min_index]
        min_distance = distances[min_index]
        
        return closest_point, min_distance
    
    def setup_plotting(self):
        # Create figure for plotting distance only
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Title and labels
        self.fig.suptitle('PSM Distance from Figure-8 Tube')
        
        # Setup for distance plot
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Distance (m)')
        self.ax.set_ylim(0, 1.0)
        self.ax.grid(True)
        
        # Create empty line objects
        self.line_psm1_dist, = self.ax.plot([], [], 'r-', label='PSM1 Distance')
        self.line_psm2_dist, = self.ax.plot([], [], 'b-', label='PSM2 Distance')
        
        # Add legend
        self.ax.legend(loc='upper right')
        
        # Set tight layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show(block=False)
    
    def update_plot(self):
        # Only update if all required values have been received
        if self.targets is not None and self.track_1 is not None and self.track_2 is not None:
            current_time = rospy.get_time() - self.start_time

            _, min_dist_1 = self.find_closest_point_and_dist(targets=self.targets, source=self.track_1)
            _, min_dist_2 = self.find_closest_point_and_dist(targets=self.targets, source=self.track_2)

            # Append new data points
            self.times.append(current_time)
            self.track_1_dist_data.append(min_dist_1)
            self.track_2_dist_data.append(min_dist_2)

            print("MIN DIST 1: ", min_dist_1)
            print("MIN DIST 2:", min_dist_2)

            # Update the plot
            self.line_psm1_dist.set_data(self.times, self.track_1_dist_data)
            self.line_psm2_dist.set_data(self.times, self.track_2_dist_data)
            self.ax.relim()            # Recompute the data limits
            self.ax.autoscale_view()   # Autoscale the view to the new data

            plt.draw()
            plt.pause(0.001)

if __name__ == '__main__':
    try:
        figure_8_topic = '/ambf/env/World/figure_8'
        psm1_toolyawlink_topic = '/ambf/env/psm1/toolyawlink/State'
        psm2_toolyawlink_topic = '/ambf/env/psm2/toolyawlink/State'

        psm_tracker = ClosestPointDistPlotter(figure_8_topic, psm1_toolyawlink_topic, psm2_toolyawlink_topic)
        rate = rospy.Rate(10)  # 10 Hz update rate
        
        # Main loop: update the plot in the main thread
        while not rospy.is_shutdown():
            psm_tracker.update_plot()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
