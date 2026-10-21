#!/usr/bin/env python3
import math
import rclpy
import numpy as np
import time

from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from nav2_msgs.action import NavigateToPose

from tf2_ros import Buffer, TransformListener
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import Odometry
from action_msgs.msg import GoalStatus

# Import custom message
from tracker_msgs_team_14.msg import TrackerReport
from std_srvs.srv import Trigger

class TrackerTwoTB4(Node):

    def __init__(self):
        super().__init__('tracker_two_tb4')

        self.declare_parameter('chaser')         
        self.declare_parameter('enemy')          
        self.declare_parameter('global_frame') 

        self.chaser = self.get_parameter('chaser').value
        self.enemy = self.get_parameter('enemy').value
        self.global_frame = self.get_parameter('global_frame').value

        nav_action = self.chaser + '/navigate_to_pose'

        self.H = np.eye(4)  

        # --- Distance and robot parameters ---
        self.robot_radius = 0.17                    # Robot radius (0.17 m)
        self.enemy_offset = 0.5 - self.robot_radius # Target offset behind enemy (0.33 m; increase slightly e.g. to 0.36 m if collision risk occurs)
        self.enemy_fov = 40.0                       # Half FOV of the enemy in degrees (assuming 80° total)
        self.enemy_fov_offset = 20.0                # Buffer zone in degrees

        # --- Parameters for Evasive Behaviour ---
        self.is_evading = False                     # Current state (False = Chaser mode, True = Evasive mode)
        self.evade_distance = 0.9                   # Lateral escape distance away from enemy [m]
        self.evade_duration = 2.5                   # Minimum escape duration [s] before switching back to Chaser mode
        self.evade_start_time = 0.0                 # Timestamp when evasive maneuver started

        self.last_goal_x = 0.0   
        self.last_goal_y = 0.0   
        self.last_goal_yaw = 0.0 
        self.rel_dist_enemy_in_chaser  = 0.0    
        self.rel_orient_enemy_in_chaser = 0.0   

        self.goal_handle = None
        self.pending_goal = False

        self.chaser_base = self.chaser + '/base_link'
        self.enemy_base = self.enemy + '/base_link'

        self.have_goal = False

        # Flexible ID extraction for simulation (tb3_0) and hardware (tb407)
        self.my_id = int(self.chaser.split('_')[-1] if '_' in self.chaser else self.chaser[3:])
        self.opp_id = int(self.enemy.split('_')[-1] if '_' in self.enemy else self.enemy[3:])
        
        # Subscriber for the camera image
        self.tracker_image_sub = self.create_subscription(
            CompressedImage,
            '/' + self.chaser + '/oakd/rgb/preview/image_raw/compressed',
            self.tracker_image_callback,
            qos_profile_sensor_data
        )

        # Publisher for custom report message
        self.tracker_report_pub = self.create_publisher(
            TrackerReport,
            '/' + self.chaser + '/tracker_report',
            10
        )

        # Publisher for separate RViz image visualization
        self.tracker_report_img_pub = self.create_publisher(
            CompressedImage,
            '/' + self.chaser + '/tracker_report_img/compressed',
            10
        )

        self.tracker_image = None
       
        # TF Buffer, Listener, and Broadcaster
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)   
        self.tf_broadcaster = TransformBroadcaster(self)                
      
        self.nav_client = ActionClient(self, NavigateToPose, nav_action)

        self.timer = self.create_timer(0.05, self.update_goal) # 20 Hz
        self.get_logger().info("Tracker Two TB4 initialized (with Evasive Behaviour)")

    def tracker_image_callback(self, msg):
        self.tracker_image = msg 

    def homogeneous_T(self, pos, yaw):
        H = np.eye(4)
        c_yaw = np.cos(yaw)
        s_yaw = np.sin(yaw)

        H[0:2, 0:2] = np.array([[c_yaw, -s_yaw],
                                [s_yaw,  c_yaw]])
        H[0:3, 3] = pos  

        return H

    def sign(self, k):
        if k < 0:
            return -1
        else:
            return 1

    def update_goal(self):
        try:
            # Lookup enemy pose in global frame
            enemy_tf = self.tf_buffer.lookup_transform(self.global_frame, self.enemy_base, rclpy.time.Time())
        except Exception as e:
            return
        
        try:
            # Lookup enemy pose in chaser frame
            rel_tf_enemy_in_chaser = self.tf_buffer.lookup_transform(self.chaser_base, self.enemy_base, rclpy.time.Time())
        except Exception as e:
            return

        try:
            # Lookup chaser pose in enemy frame
            rel_tf_chaser_in_enemy = self.tf_buffer.lookup_transform(self.enemy_base, self.chaser_base, rclpy.time.Time())
        except Exception as e:
            return        

        enemy_x = enemy_tf.transform.translation.x
        enemy_y = enemy_tf.transform.translation.y
        
        opp_q = [
            enemy_tf.transform.rotation.x,
            enemy_tf.transform.rotation.y,
            enemy_tf.transform.rotation.z,
            enemy_tf.transform.rotation.w
        ]
        
        _, _, opp_yaw = euler_from_quaternion(opp_q)

        rel_x_enemy_in_chaser = rel_tf_enemy_in_chaser.transform.translation.x
        rel_y_enemy_in_chaser = rel_tf_enemy_in_chaser.transform.translation.y
        rel_q_enemy_in_chaser = rel_tf_enemy_in_chaser.transform.rotation

        _, _, rel_yaw_enemy_in_chaser = euler_from_quaternion([
            rel_q_enemy_in_chaser.x,
            rel_q_enemy_in_chaser.y, 
            rel_q_enemy_in_chaser.z, 
            rel_q_enemy_in_chaser.w
        ])

        # Calculate relative distance (compensated for robot radius) and orientation
        raw_dist = math.sqrt(rel_x_enemy_in_chaser**2 + rel_y_enemy_in_chaser**2)
        self.rel_dist_enemy_in_chaser = max(0.0, raw_dist - self.robot_radius)
        self.rel_orient_enemy_in_chaser = math.degrees(rel_yaw_enemy_in_chaser)

        rel_x_chaser_in_enemy = rel_tf_chaser_in_enemy.transform.translation.x
        rel_y_chaser_in_enemy = rel_tf_chaser_in_enemy.transform.translation.y

        # --- FOV check & state transition (Evasive vs. Chaser) ---
        angle_to_chaser_deg = math.degrees(math.atan2(rel_y_chaser_in_enemy, rel_x_chaser_in_enemy))
        fov_limit = self.enemy_fov + self.enemy_fov_offset
        now = time.time()

        if abs(angle_to_chaser_deg) <= fov_limit:
            if not self.is_evading:
                self.get_logger().warn("Inside enemy FOV! Starting evasive behaviour...")
                self.is_evading = True
                self.have_goal = False  # Force immediate transmission of the escape goal
            # Reset timer while still inside the danger zone
            self.evade_start_time = now
        else:
            # Once outside FOV (+ 10° hysteresis) and minimum escape time has elapsed -> return to Chaser mode
            if self.is_evading and (now - self.evade_start_time > self.evade_duration) and abs(angle_to_chaser_deg) > (fov_limit + 10.0):
                self.get_logger().info("Safe zone reached! Switching back to chaser mode.")
                self.is_evading = False
                self.have_goal = False  # Force immediate transmission of the tracking goal

        # --- Goal calculation depending on active mode ---
        H_enemy_in_map = self.homogeneous_T(np.array([enemy_x, enemy_y, 0.0]), opp_yaw)

        if self.is_evading:
            # EVASIVE MODE:
            # Use sign() to determine which side (left = +1, right = -1) offers the fastest exit from the FOV
            side = self.sign(rel_y_chaser_in_enemy)
            
            # Escape point is placed laterally outside the FOV (y = side * evade_distance) and slightly behind the front (x = -0.2)
            # Orientation points laterally away from the enemy (side * pi/2) so the robot drives straight away
            evade_yaw_rel = side * (math.pi / 2.0)
            H_offset = self.homogeneous_T(
                np.array([-0.2, side * self.evade_distance, 0.0]),
                evade_yaw_rel
            )
        else:
            # CHASER MODE:
            # Standard tracking goal directly behind the enemy
            H_offset = self.homogeneous_T(
                np.array([-self.enemy_offset, 0.0, 0.0]),
                0.0
            )
        
        H_d_in_map = H_enemy_in_map @ H_offset 

        goal_x = H_d_in_map[0, 3]
        goal_y = H_d_in_map[1, 3]
        
        R_d_in_map = R.from_matrix(H_d_in_map[0:3, 0:3]) 
        _, _, goal_yaw = R_d_in_map.as_euler('xyz', degrees=False)

        # Send goal if deviation is > 0.1m OR after a mode switch (not self.have_goal)
        goal_shift = math.sqrt((goal_x - self.last_goal_x)**2 + (goal_y - self.last_goal_y)**2)

        if (not self.have_goal) or goal_shift > 0.1:
            sent = self.send_goal(goal_x, goal_y, goal_yaw)
            if sent:
                self.have_goal = True
                self.last_goal_x = goal_x
                self.last_goal_y = goal_y
                self.last_goal_yaw = goal_yaw

        # Transform Broadcaster for RViz visualization of the current goal
        goal_q = quaternion_from_euler(0, 0, goal_yaw)
        goal_tf = TransformStamped()
        goal_tf.header.stamp = self.get_clock().now().to_msg()
        goal_tf.header.frame_id = self.global_frame
        goal_tf.child_frame_id = self.chaser + '/goal_pose'
        goal_tf.transform.translation.x = float(goal_x)
        goal_tf.transform.translation.y = float(goal_y)
        goal_tf.transform.rotation.x = float(goal_q[0])
        goal_tf.transform.rotation.y = float(goal_q[1])
        goal_tf.transform.rotation.z = float(goal_q[2])
        goal_tf.transform.rotation.w = float(goal_q[3])
        self.tf_broadcaster.sendTransform(goal_tf)

    def send_goal(self, goal_x, goal_y, goal_yaw):
        if not self.nav_client.server_is_ready():
            return False

        goal_pose = NavigateToPose.Goal()
        goal_pose.pose.header.frame_id = self.global_frame
        goal_pose.pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.pose.position.x = goal_x
        goal_pose.pose.pose.position.y = goal_y
        goal_pose.pose.pose.position.z = 0.0

        goal_q = quaternion_from_euler(0, 0, goal_yaw)
        goal_pose.pose.pose.orientation.x = goal_q[0]
        goal_pose.pose.pose.orientation.y = goal_q[1]
        goal_pose.pose.pose.orientation.z = goal_q[2]
        goal_pose.pose.pose.orientation.w = goal_q[3]

        self._send_goal_future = self.nav_client.send_goal_async(goal_pose)
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        self.pending_goal = True

        return True

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.pending_goal = False
            return

        self.goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        status = future.result().status     
        self.pending_goal = False

        if status == GoalStatus.STATUS_SUCCEEDED:
            # If we only reached the escape point -> do NOT take a picture, switch back to Chaser mode instead
            if self.is_evading:
                self.get_logger().info("Escape point reached! Switching back to chaser mode.")
                self.is_evading = False
                self.have_goal = False
                return

            self.get_logger().info(f"Goal succeeded! Publishing report at distance: {self.rel_dist_enemy_in_chaser:.2f} m")
            
            if self.tracker_image is None:
                self.get_logger().info("Goal succeeded but no image available.")
                return

            # Capture & Report logic (only in Chaser mode)
            report = TrackerReport()
            report.my_id = self.my_id
            report.opp_id = self.opp_id
            report.image = self.tracker_image
            report.rel_dist = float(self.rel_dist_enemy_in_chaser)       
            report.rel_orient = float(self.rel_orient_enemy_in_chaser)   

            self.tracker_report_pub.publish(report)
            self.tracker_report_img_pub.publish(self.tracker_image)

        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn("Goal aborted")
            self.have_goal = False
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warn("Goal canceled")
            self.have_goal = False

def main(args=None):
    rclpy.init(args=args)
    node = TrackerTwoTB4()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()