#!/usr/bin/env python3
import rospy
from sss_object_detection.consts import ObjectID
from vision_msgs.msg import ObjectHypothesisWithPose, Detection2DArray, Detection2D
import tf2_ros
import tf2_geometry_msgs
from smarc_msgs.msg import GotoWaypoint
from nav_msgs.msg import Odometry

import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor


class sss_detection_listener:
    def __init__(self, robot_name):

        self.rope_pose = []

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.current_pose = None
        self.odom_sub = rospy.Subscriber(
            '/{}/dr/odom'.format(robot_name), Odometry,
            self.update_pose)

        self.detection_topic = '/{}/payload/sidescan/detection_hypothesis'.format(
            robot_name)
        self.detection_sub = rospy.Subscriber(self.detection_topic, Detection2DArray,
                                         self.detection_callback)
        self.waypoint_topic = '/{}/ctrl/goto_waypoint'.format(robot_name)
        self.waypoint_topic_type = GotoWaypoint #ROS topic type
        self.waypoint_pub = rospy.Publisher(self.waypoint_topic, self.waypoint_topic_type,
                queue_size=5)
       
        print(self.waypoint_topic)

    def wait_for_transform(self, from_frame, to_frame):
        """Wait for transform from from_frame to to_frame"""
        trans = None
        while trans is None:
            try:
                trans = self.tf_buffer.lookup_transform(
                    to_frame, from_frame, rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as error:
                print('Failed to transform. Error: {}'.format(error))
        return trans

    def transform_pose(self, pose, from_frame, to_frame):
        trans = self.wait_for_transform(from_frame=from_frame,
                                         to_frame=to_frame)
        pose_transformed = tf2_geometry_msgs.do_transform_pose(pose, trans)
        return pose_transformed

    def update_pose(self, msg):
        # might need to transform pose to another frame
        to_frame = 'utm'
        transformed_pose = self.transform_pose(msg.pose, from_frame=msg.header.frame_id, to_frame=to_frame)
        self.current_pose = transformed_pose
        #print('Current pose:')
        #print(type(self.current_pose))

    def detection_callback(self, msg):
        # Assume each Detection2DArray only contains one Detection2D message
        # Further assume each Detection2D only contains one ObjectHypothesisWithPose
        for detection in msg.detections:
            object_hypothesis = detection.results[0]
            object_frame_id = msg.header.frame_id[1:]
            # Pose msg
            object_pose = object_hypothesis.pose
            detection_confidence = object_hypothesis.score
            object_id = object_hypothesis.id

            to_frame = 'utm'
            object_pose = self.transform_pose(object_pose, from_frame=object_frame_id, to_frame=to_frame)

            if object_id == ObjectID.ROPE.value:
                print('Detected rope at frame {}, pose {}, with confidence {}'.format(
                    object_frame_id, object_pose, detection_confidence))
                # Do whatever you want to do
                self.rope_pose.append(object_pose)
                self.publish_waypoint()
            if object_id == ObjectID.BUOY.value:
                print('Detected buoy at frame {}, pose {}, with confidence {}'.format(
                    object_frame_id, object_pose, detection_confidence))
            if object_id == ObjectID.NADIR.value:
                pass
                #print('Detected nadir at frame {}, pose {}, with confidence {}'.format(
                #    object_frame_id, object_pose, detection_confidence))

    def publish_waypoint(self):
        msg = GotoWaypoint()
        #msg.waypoint_pose.pose.position.x = x
        msg.waypoint_pose = self.rope_pose[-1]
        msg.waypoint_pose.header.stamp = rospy.Time(0)
        msg.goal_tolerance = 2
        self.waypoint_pub.publish(msg)
        print('Publishing waypoint: {}'.format(msg))


def main():
    rospy.init_node('sss_detection_listener', anonymous=True)
    rospy.Rate(5)  # ROS Rate at 5Hz

    robot_name_param = '~robot_name'
    if rospy.has_param(robot_name_param):
        robot_name = rospy.get_param(robot_name_param)
        print('Getting robot_name = {} from param server'.format(robot_name))
    else:
        robot_name = 'sam'
        print('{} param not found in param server.\n'.format(robot_name_param))
        print('Setting robot_name = {} default value.'.format(robot_name))

    print('entering ssss_detection_listner...')
    
    listner = sss_detection_listener(robot_name)

    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == '__main__':
    main()
