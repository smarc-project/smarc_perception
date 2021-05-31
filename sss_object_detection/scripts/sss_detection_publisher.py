#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import Pose
from smarc_msgs.msg import Sidescan
from vision_msgs.msg import ObjectHypothesisWithPose, Detection2DArray, Detection2D

from consts import ObjectID, Side

class sss_detector:
    def __init__(self,
                 robot_name):
        # Object height below water [m]
        self.object_height = 2
        self.robot_name = robot_name
        self.sidescan_sub = rospy.Subscriber('/{}/payload/sidescan'.format(robot_name), Sidescan,
                self._sidescan_callback)
        self.detection_pub = rospy.Publisher(
            '/{}/payload/sidescan/detection_hypothesis'.format(robot_name),
            Detection2DArray, queue_size=2)
        self.pub_frame_id = '/{}/base_link'.format(robot_name)
        # TODO: implement detectors: CPD and neural network
        self.detector = Detector()

    def _sidescan_callback(self, msg):
        channels = {Side.PORT: msg.port_channel, Side.STARBOARD:
                msg.starbord_channel}
        for channel_id, channel in channels.items():
            # TODO: normalize ping
            ping = np.array(bytearray(channel), dtype=np.ubyte)
            # TODO: make detector.detect(ping) return
            #       dict[ObjectID: {pos: int, confidence: float}]
            detection_res = self.detector.detect(ping)

            if ObjectID.BUOY or ObjectID.ROPE in detection_res:
                detection_msg = self._construct_detection_msg(detection_res,
                        channel_id, msg.decimation, msg.header.stamp)
                self.detection_pub.publish(detection_msg)

    def _construct_detection_msg(self, detection_res, channel_id, decimation,
            stamp):
        detection_array_msg = Detection2DArray()
        detection_array_msg.header.frame_id = self.pub_frame_id
        detection_array_msg.header.stamp = stamp

        for object_id, detection in detection_res.items():
            detection_msg = Detection2D()
            detection_msg.header = detection_array_msg.header

            object_hypothesis = ObjectHypothesisWithPose()
            object_hypothesis.id = object_id
            object_hypothesis.score = detection['confidence']
            object_hypothesis.pose.pose = self._detection_pos_to_pose(detection['pos'], channel_id, decimation)

            detection_msg.results.append(object_hypothesis)
            detection_array_msg.detections.append(detection_msg)
        return detection_array_msg

    def _detection_to_pose(self, pos, channel_id, decimation):
        """Given detected pos (index in the sidescan ping), channel_id
        (Side.PORT or Side.STARBOARD) and decimation, return the constructed
        pose for the detection"""
        detected_pose = Pose()
        hypothenus = pos * decimation
        distance = (hypothenus**2 - self.object_height**2)**.5
        detected_pose.position.y = distance
        if channel_id == Side.PORT:
            detected_pose.position.y *= -1
        return detected_pose



def main():
    rospy.init_node('sss_detection_publisher', anonymous=True)
    rospy.Rate(5)  # ROS Rate at 5Hz

    robot_name_param = '~robot_name'
    if rospy.has_param(robot_name_param):
        robot_name = rospy.get_param(robot_name_param)
        print('Getting robot_name = {} from param server'.format(robot_name))
    else:
        robot_name = 'sam'
        print('{} param not found in param server.\n'.format(robot_name_param))
        print('Setting robot_name = {} default value.'.format(robot_name))

    detector = sss_detector(robot_name=robot_name)

    while not rospy.is_shutdown():
        rospy.spin()


if __name__ == '__main__':
    main()
