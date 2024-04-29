#!/usr/bin/env python3

import rospy
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import MapMetaData, OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion, Vector3, Point, PoseStamped, PoseWithCovarianceStamped
from rrt import RRT, RRTVertex
import numpy as np
from typing import Optional
import actionlib
from gazebo_ros_link_attacher.srv import Attach, AttachRequest
from gazebo_msgs.srv import SetLinkProperties, SetLinkPropertiesRequest

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

def create_sphere_marker(position: Point = Point(x=0, y=0), color: ColorRGBA = ColorRGBA(r=1, a=1)) -> Marker:
  marker: Marker = Marker()
  marker.header.frame_id = 'map'
  marker.type = Marker.SPHERE
  marker.pose.position = position
  marker.pose.orientation = Quaternion(w=1)
  marker.color = color
  marker.scale = Vector3(x=0.1, y=0.1, z=0.1)

  return marker

if __name__ == '__main__':
  rospy.init_node('multi_planner')

  # Hard coded start and goal
  start = Point(x=0, y=0)
  goal = Point(x=4, y=-2)

  # Visualization marker publishers
  start_marker_publisher = rospy.Publisher('/multi_planner/visualization/start', Marker, queue_size=10, latch=True)
  goal_marker_publisher = rospy.Publisher('/multi_planner/visualization/goal', Marker, queue_size=10, latch=True)

  start_marker = create_sphere_marker(start)
  goal_marker = create_sphere_marker(goal, color=ColorRGBA(g=1, a=1))

  start_marker_publisher.publish(start_marker)
  goal_marker_publisher.publish(goal_marker)

  forward_rrt_publisher = rospy.Publisher('/multi_planner/visualization/forward_rrt', Marker, queue_size=10, latch=True)
  reverse_rrt_publisher = rospy.Publisher('/multi_planner/visualization/reverse_rrt', Marker, queue_size=10, latch=True)

  exchange_point_publisher = rospy.Publisher('/multi_planner/visualization/exchange_point', Marker, queue_size=10, latch=True)

  # Read inflated costmaps
  mother_costmap: Optional[OccupancyGrid] = rospy.wait_for_message('/mother_inflation/costmap/costmap', OccupancyGrid)
  baby_costmap: Optional[OccupancyGrid] = rospy.wait_for_message('/baby_inflation/costmap/costmap', OccupancyGrid)

  # Initialize move_base clients
  mother_move_base_client = actionlib.SimpleActionClient('/mother/move_base', MoveBaseAction)
  mother_move_base_client.wait_for_server()

  baby_move_base_client = actionlib.SimpleActionClient('/baby/move_base', MoveBaseAction)
  baby_move_base_client.wait_for_server()

  # Initialize simulation attach/detach service
  attach_client = rospy.ServiceProxy('/link_attacher_node/attach', Attach)
  attach_client.wait_for_service()

  detach_client = rospy.ServiceProxy('/link_attacher_node/detach', Attach)
  detach_client.wait_for_service()

  # Initialize simulation link property service
  link_property_client = rospy.ServiceProxy('/gazebo/set_link_properties', SetLinkProperties)
  link_property_client.wait_for_service()

  # Initialize baby initial pose guess publisher
  baby_initial_pose_publisher = rospy.Publisher('/baby/initialpose', PoseWithCovarianceStamped, queue_size=1, latch=False)
  
  if mother_costmap is not None and baby_costmap is not None:
    map_metadata: MapMetaData = mother_costmap.info

    min_x = map_metadata.origin.position.x
    min_y = map_metadata.origin.position.y

    max_x = min_x + map_metadata.resolution * map_metadata.height
    max_y = min_y + map_metadata.resolution * map_metadata.width

    # Initialize RRTs
    forward_rrt = RRT(start=(start.x, start.y), goal=(goal.x, goal.y), x_range=(min_x, max_x), y_range=(min_y, max_y), occupancy_grid=mother_costmap, delta=0.2)
    reverse_rrt = RRT(start=(goal.x, goal.y), goal=(start.x, start.y), x_range=(min_x, max_x), y_range=(min_y, max_y), occupancy_grid=baby_costmap, delta=0.2)

    exchange_point: Optional[Point] = None
    best_exchange_goal_distance = np.inf

    current_rrt = forward_rrt

    for i in range(10000):
      random_vertex = current_rrt.random_vertex()
      stepped_vertex = current_rrt.extend(random_vertex)
      
      connected_vertex: Optional[RRTVertex] = None

      if stepped_vertex is not None:

        if current_rrt is forward_rrt:
          connected_vertex = reverse_rrt.connect(stepped_vertex)
          current_rrt = reverse_rrt
        else:
          connected_vertex = forward_rrt.connect(stepped_vertex)
          current_rrt = forward_rrt
        
        
        if connected_vertex is not None and connected_vertex == stepped_vertex and connected_vertex.distance(RRTVertex((goal.x, goal.y))) < best_exchange_goal_distance:
          best_exchange_goal_distance = connected_vertex.distance(RRTVertex((goal.x, goal.y)))
          exchange_point = Point(x=connected_vertex.point[0], y=connected_vertex.point[1])

      forward_rrt_publisher.publish(forward_rrt.get_marker_visualization(color=ColorRGBA(r=1, a=1)))
      reverse_rrt_publisher.publish(reverse_rrt.get_marker_visualization(color=ColorRGBA(g=1, a=1)))

      if exchange_point is not None:
        exchange_point_publisher.publish(create_sphere_marker(exchange_point, color=ColorRGBA(b=1, a=1)))

    
    if exchange_point is not None:
      
      # Attach baby
      attach_request = AttachRequest()
      attach_request.model_name_1 = "mother"
      attach_request.link_name_1 = "base_footprint"
      attach_request.model_name_2 = "baby"
      attach_request.link_name_2 = "base_footprint"

      attach_client.call(attach_request)

      # Navigate mother to exchange point
      move_base_goal = MoveBaseGoal()
      move_base_goal.target_pose = PoseStamped()
      move_base_goal.target_pose.header.frame_id = 'map'
      move_base_goal.target_pose.pose.position = exchange_point
      move_base_goal.target_pose.pose.orientation = Quaternion(w=1)

      mother_move_base_client.send_goal_and_wait(move_base_goal)

      # Detach baby
      attach_request = AttachRequest()
      attach_request.model_name_1 = "mother"
      attach_request.link_name_1 = "base_footprint"
      attach_request.model_name_2 = "baby"
      attach_request.link_name_2 = "base_footprint"

      detach_client.call(attach_request)

      # Enable baby gravity
      link_property_request = SetLinkPropertiesRequest()
      link_property_request.link_name = "baby::base_footprint"
      link_property_request.mass = 0.94
      link_property_request.ixx = 0.01
      link_property_request.iyy = 0.01
      link_property_request.gravity_mode = True
      link_property_client.call(link_property_request)

      link_property_request = SetLinkPropertiesRequest()
      link_property_request.link_name = "baby::wheel_left_link"
      link_property_request.mass = 0.03
      link_property_request.gravity_mode = True
      link_property_client.call(link_property_request)

      link_property_request = SetLinkPropertiesRequest()
      link_property_request.link_name = "baby::wheel_right_link"
      link_property_request.mass = 0.03
      link_property_request.gravity_mode = True
      link_property_client.call(link_property_request)

      rospy.logdebug("Baby detached")

      # initial_pose = PoseWithCovarianceStamped()
      # initial_pose.header.frame_id = 'map'
      # initial_pose.pose.pose.position = exchange_point
      # initial_pose.pose.pose.orientation = Quaternion(w=1)
      # baby_initial_pose_publisher.publish(initial_pose)

      # Navigate baby to goal
      move_base_goal = MoveBaseGoal()
      move_base_goal.target_pose = PoseStamped()
      move_base_goal.target_pose.header.frame_id = 'map'
      move_base_goal.target_pose.pose.position = goal
      move_base_goal.target_pose.pose.orientation = Quaternion(w=1)

      baby_move_base_client.send_goal(move_base_goal)


      move_base_goal = MoveBaseGoal()
      move_base_goal.target_pose = PoseStamped()
      move_base_goal.target_pose.header.frame_id = 'map'
      move_base_goal.target_pose.pose.position = start
      move_base_goal.target_pose.pose.orientation = Quaternion(w=1)

      mother_move_base_client.send_goal_and_wait(move_base_goal)



