#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from smarc_msgs.msg import Sidescan
from vision_msgs.msg import ObjectHypothesisWithPose, Detection2DArray, Detection2D
from cv_bridge import CvBridge, CvBridgeError

from consts import ObjectID, Side
from cpd_detector import CPDetector

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
        #TODO: implement neural network detector
        self.detector = CPDetector()

        # Detection visualization
        self.bridge = CvBridge
        self.sidescan_image = np.zeros((2000, 500, 3))
        self.detection_image = np.zeros_like(self.sidescan_image)
        self.sidescan_image_pub = rospy.Publisher(
                '/{}/payload/sidescan/image'.format(robot_name),
                Image, queue_size=2)
        self.detection_image_pub = rospy.Publisher(
                '/{}/payload/sidescan/detection_hypothesis_image'.format(robot_name),
                Image, queue_size=2)
        self.detection_colors = {ObjectID.NADIR: (255, 255, 0), #yellow
                                 ObjectID.BUOY: (0, 0, 255), #blue
                                 ObjectID.ROPE: (255, 0, 0) #red
                                 }


    def _sidescan_callback(self, msg):
        channel_to_np = lambda channel: np.array(bytearray(channel),
                dtype=np.ubyte)
        channels = {Side.PORT: channel_to_np(msg.port_channel), Side.STARBOARD:
                channel_to_np(msg.starbord_channel)}

        # Update sidescan image
        self.sidescan_image[1:, :, :] = self.sidescan_image[:-1, :]
        self.sidescan_image[0, :, :] = np.concatenate([np.flip(channels[Side.PORT], axis=0),
            channels[Side.STARBOARD]])
        self.detection_image = self.sidescan_image.copy()

        for channel_id, channel in channels.items():
            # TODO: normalize ping
            ping = channel

            detection_res = self.detector.detect(ping)

            if detection_res:
                # Publish detection message
                detection_msg = self._construct_detection_msg(detection_res,
                        channel_id, msg.decimation, msg.header.stamp)
                self.detection_pub.publish(detection_msg)

                # Update detection image
                self._update_detection_image(self, detection_res)

        self._publish_sidescan_and_detection_images()

    def _update_detection_image(self, detection_res):
        for object_id, detection in detection_res:
            self.detection_image[0, detection['pos'], :] = self.detection_colors[object_id]


    def _publish_sidescan_and_detection_images(self):
        try:
            self.sidescan_image_pub.publish(self.bridge.cv2_to_imgmsg(self.sidescan_image,
                "passthrough"))
            self.detection_image_pub.publish(self.bridge.cv2_to_imgmsg(self.detection_image,
                "passthrough"))
        except CvBridgeError as error:
            print('Error converting numpy array to img msg: {}'.format(error))


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
