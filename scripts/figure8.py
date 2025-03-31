import rospy
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from std_msgs.msg import Float32
import numpy as np
import math
import time

# Function to calculate figure-8 coordinates at parameter t
def calculate_figure8(t):
    # Classic lemniscate (figure-8) formula
    x = figure8_radius * math.sin(t) / (1 + math.cos(t)**2)
    y = figure8_radius * math.sin(t) * math.cos(t) / (1 + math.cos(t)**2)
    return x, y


# ROS
rospy.init_node('static_figure8_pc')

# AMBF will have a default PC listener at '/ambf/env/World/point_cloud'
topics_names_param = '/ambf/env/World/point_cloud_topics'
figure_8_topic = '/ambf/env/World/figure_8'
figure_8_radius_topic = f'{figure_8_topic}/radius'

pc_topics = rospy.get_param(topics_names_param, [])
print('Existing Topics AMBF is listening to for Point Cloud:')
print(pc_topics)

pc_topics.append(figure_8_topic)
rospy.set_param(topics_names_param, pc_topics)
print(f'Adding topic {figure_8_topic} via the ROS Param server')

pub = rospy.Publisher(figure_8_topic, PointCloud, queue_size=10)
size_pub = rospy.Publisher(figure_8_radius_topic, Float32, queue_size=10)

msg = PointCloud()
msg.header.frame_id = 'phantomBODY phantom'


# PC
size_msg = Float32()
size_msg.data = 10.0  # Point size
figure8_radius = 0.04 
num_points = 18000
for i in range(num_points):
    msg.points.append(Point32())


# Create an evenly distributed figure-8 curve using arc length parameterization
# We'll use a lookup table approach to get more uniform spacing
total_length = 0
step_size = 0.01
parameter_values = []
accumulated_lengths = []

# Calculate total length and sample points
t = 0
prev_x, prev_y = calculate_figure8(0)

while t <= 2*np.pi:
    # Figure-8 curve coordinates
    x, y = calculate_figure8(t)
    
    # Add to the total length except for the first point
    if t > 0:
        segment_length = math.sqrt((x - prev_x)**2 + (y - prev_y)**2)
        total_length += segment_length
    
    parameter_values.append(t)
    accumulated_lengths.append(total_length)
    
    prev_x, prev_y = x, y
    t += step_size

# Normalize accumulated lengths
if total_length > 0:
    accumulated_lengths = [l/total_length for l in accumulated_lengths]


# Generate evenly spaced points along the curve
point_idx = 0
for i in range(num_points):
    # Target arc length position (normalized)
    target_length = i / (num_points - 1)
    
    # Find the closest precomputed point
    closest_idx = min(range(len(accumulated_lengths)), 
                     key=lambda i: abs(accumulated_lengths[i] - target_length))
    
    # Get the parameter value
    t = parameter_values[closest_idx]
    x, y = calculate_figure8(t)
    z = 0.02  # Fixed z position

    msg.points[point_idx].x = x
    msg.points[point_idx].y = y
    msg.points[point_idx].z = z

    point_idx += 1
    if point_idx >= num_points:
        break


# Fill any remaining points (if we didn't use all 18000)
for i in range(point_idx, num_points):
    msg.points[i].x = msg.points[point_idx-1].x
    msg.points[i].y = msg.points[point_idx-1].y
    msg.points[i].z = msg.points[point_idx-1].z


# Main loop
rate = rospy.Rate(10)
while not rospy.is_shutdown():
    # Update timestamp and publish
    msg.header.stamp = rospy.Time.now()
    pub.publish(msg)
    size_pub.publish(size_msg)
    
    rate.sleep()
