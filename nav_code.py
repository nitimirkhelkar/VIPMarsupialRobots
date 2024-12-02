#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

# Parameters
MOTHER_ROBOT_WIDTH = 0.22  # Width of TurtleBot in meters
BABY_ROBOT_WIDTH = 0.11  # Width of baby robot in meters

SAFE_MARGIN = 0.2  # Extra margin to avoid obstacles
MOTHER_MIN_GAP_SIZE = MOTHER_ROBOT_WIDTH + SAFE_MARGIN
BABY_MIN_GAP_SIZE = BABY_ROBOT_WIDTH + SAFE_MARGIN

# Spike = 2 consecutive points with a large distance between them, indicating 2 obstacles
# Length of obstacle + wiggle room to consider it a spike
# Diameter of cylindrical obstacles = 0.3 m but added some extra margin
SPIKE_SIZE = 0.4  # Minimum size of a spike to be considered an obstacle

class LocalMappingGapFinder(Node):
    def __init__(self):
        super().__init__('local_mapping_gap_finder')
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.process_scan,
            10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info("Node initialized and listening to /scan")
        
        # Previous gap angle to compare changes
        self.prev_gap_angle = None


    def process_scan(self, scan):
        ranges = np.array(scan.ranges)
        angle_increment = scan.angle_increment
        angle_min = scan.angle_min

        # Replace invalid readings (Inf or 0.0) with a large number (max range)
        max_range = scan.range_max
        ranges = np.where((ranges == 0) | (np.isinf(ranges)), max_range, ranges)

        # Calculate angles corresponding to laser readings
        angles = angle_min + np.arange(len(ranges)) * angle_increment

        # Identify gaps by checking distances between consecutive points
        gaps = self.find_gaps(ranges, angles)

        # Check if there is a gap large enough for the mother and/or baby robot to navigate through
        for gap in gaps:
            gap_size, gap_center = gap
            if gap_size >= MOTHER_MIN_GAP_SIZE:
                # Ignore small changes in gap angle
                if self.prev_gap_angle is None or abs(gap_center - self.prev_gap_angle) > 0.1:  # 0.1 rad threshold
                    self.prev_gap_angle = gap_center
                    self.get_logger().info(f"Navigable gap found: size={gap_size:.2f}m, center angle={gap_center:.2f}rad")
                    self.move_towards_gap(gap_center)
                return
            elif gap_size >= BABY_MIN_GAP_SIZE:
                self.get_logger().info(f"Baby robot can navigate through this gap: size={gap_size:.2f}m, center angle={gap_center:.2f}rad")
        
        # If no navigable gap is found, rotate in place
        self.rotate_in_place()


    def find_gaps(self, ranges, angles):
        gaps = []
        start_idx = None
        spike1 = False
        spike2 = False
        for i in range(len(ranges) - 1):
            # Check if current point and next point are far apart
            if abs(ranges[i] - ranges[i + 1]) > SPIKE_SIZE:
                if spike1 != False:
                    spike2 = True
                    continue
                elif start_idx is None:
                    start_idx = i
                    spike1 = True
            else:
                if spike2 == False:
                    continue
                elif start_idx is not None:
                    end_idx = i

                    # Calculate gap size and center angle
                    # Getting cartesian coordinates of start and end points
                    start_x = ranges[start_idx] * np.cos(angles[start_idx])
                    start_y = ranges[start_idx] * np.sin(angles[start_idx])

                    end_x = ranges[end_idx] * np.cos(angles[end_idx])
                    end_y = ranges[end_idx] * np.sin(angles[end_idx])

                    # Euclidean distance between start and end points
                    gap_size = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)

                    gap_center = angles[(start_idx + end_idx) // 2]
                    gaps.append((gap_size, gap_center))
                    start_idx = None

                    spike1 = False
                    spike2 = False

        return gaps

    def move_towards_gap(self, gap_angle):
        cmd = Twist()

        # Dynamically adjust angular_kp based on gap angle magnitude
        if abs(gap_angle) > 1.0:  # Larger angles need faster turning
            angular_kp = 1.0
        elif abs(gap_angle) < 0.3:  # Small angles = smoother, slower turning
            angular_kp = 0.6
        else:
            angular_kp = 0.8

        # Proportional control for angular velocity
        cmd.angular.z = angular_kp * gap_angle

        # Clamp angular velocity to a reasonable range to prevent spinning too fast
        max_angular_speed = 1.0  # rad/s
        cmd.angular.z = max(-max_angular_speed, min(max_angular_speed, cmd.angular.z))

        # Set constant forward velocity
        cmd.linear.x = 0.2

        # Publish the velocity command
        self.cmd_vel_pub.publish(cmd)


    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

    def rotate_in_place(self):
        # Rotates the robot in place to search for new gaps.
        self.get_logger().info("No navigable gap found. Rotating in place to search for gaps.")
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.5  # Rotate in place at a fixed speed
        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = LocalMappingGapFinder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

