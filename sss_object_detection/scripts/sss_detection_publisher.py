#!/usr/bin/env python

import argparse
import rospy
import numpy as np
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from smarc_msgs.msg import Sidescan
from vision_msgs.msg import ObjectHypothesisWithPose, Detection2DArray, Detection2D
from cv_bridge import CvBridge, CvBridgeError

from consts import ObjectID, Side
from cpd_detector import CPDetector


class sss_detector:
    def __init__(self, robot_name, water_depth=15, object_height=2):
        # Object height below water [m]
        self.object_height = object_height
        self.vehicle_z_pos = 0
        self.robot_name = robot_name
        self.sidescan_sub = rospy.Subscriber(
            '/{}/payload/sidescan'.format(robot_name), Sidescan,
            self._sidescan_callback)
        self.detection_pub = rospy.Publisher(
            '/{}/payload/sidescan/detection_hypothesis'.format(robot_name),
            Detection2DArray,
            queue_size=2)
        self.pub_frame_id = '/{}/base_link'.format(robot_name)
        self.odom_sub = rospy.Subscriber('/{}/dr/odom/z'.format(robot_name),
                                         Float64, self._update_vehicle_z_pos)
        #TODO: implement neural network detector
        self.detector = CPDetector()
        #TODO: remove temporary hard-coded resolution value (should be read and
        self.resolution = 0.05
        self.channel_size = 1000
        self.water_depth = water_depth

        # Detection visualization
        self.bridge = CvBridge()
        self.sidescan_image = np.zeros((500, self.channel_size * 2, 3),
                                       dtype=np.uint8)
        self.detection_image = np.zeros_like(self.sidescan_image,
                                             dtype=np.uint8)
        self.sidescan_image_pub = rospy.Publisher(
            '/{}/payload/sidescan/image'.format(robot_name),
            Image,
            queue_size=2)
        self.detection_image_pub = rospy.Publisher(
            '/{}/payload/sidescan/detection_hypothesis_image'.format(
                robot_name),
            Image,
            queue_size=2)
        self.detection_colors = {
            ObjectID.NADIR: (255, 255, 0),  #yellow
            ObjectID.BUOY: (0, 255, 255),  #blue
            ObjectID.ROPE: (255, 0, 0)  #red
        }

    def _update_vehicle_z_pos(self, msg):
        self.vehicle_z_pos = msg.data

    def _sidescan_callback(self, msg):
        channel_to_np = lambda channel: np.array(bytearray(channel),
                                                 dtype=np.uint8)
        channels = {
            Side.PORT: channel_to_np(msg.port_channel),
            Side.STARBOARD: channel_to_np(msg.starboard_channel)
        }

        # Update sidescan image
        self.sidescan_image[1:, :, :] = self.sidescan_image[:-1, :, :]
        for i in range(3):
            self.sidescan_image[0, :, i] = np.concatenate([
                np.flip(channels[Side.PORT], axis=0), channels[Side.STARBOARD]
            ])
        self.detection_image[1:, :, :] = self.detection_image[:-1, :, :]
        self.detection_image[0, :, :] = self.sidescan_image[0, :, :]

        for channel_id, channel in channels.items():
            # TODO: normalize ping
            ping = channel

            detection_res = self.detector.detect(ping)

            if detection_res:
                # Publish detection message
                detection_msg = self._construct_detection_msg_and_update_detection_image(
                    detection_res, channel_id, msg.header.stamp)
                if len(detection_msg.detections) > 0:
                    self.detection_pub.publish(detection_msg)

        self._publish_sidescan_and_detection_images()

    def _publish_sidescan_and_detection_images(self):
        try:
            self.sidescan_image_pub.publish(
                self.bridge.cv2_to_imgmsg(self.sidescan_image, "passthrough"))
            self.detection_image_pub.publish(
                self.bridge.cv2_to_imgmsg(self.detection_image, "rgb8"))
        except CvBridgeError as error:
            print('Error converting numpy array to img msg: {}'.format(error))

    def _construct_detection_msg_and_update_detection_image(
            self, detection_res, channel_id, stamp):
        if channel_id == Side.PORT:
            multiplier = -1
        else:
            multiplier = 1

        detection_array_msg = Detection2DArray()
        detection_array_msg.header.frame_id = self.pub_frame_id
        detection_array_msg.header.stamp = stamp

        for object_id, detection in detection_res.items():
            detection_msg = Detection2D()
            detection_msg.header = detection_array_msg.header

            object_hypothesis = ObjectHypothesisWithPose()
            object_hypothesis.id = object_id.value
            object_hypothesis.score = detection['confidence']
            object_hypothesis.pose.pose = self._detection_to_pose(
                detection['pos'], channel_id)

            # Filter out object detection outliers
            if object_id != ObjectID.NADIR and abs(
                    object_hypothesis.pose.pose.position.y) > self.water_depth:
                continue
            else:
                pos = self.channel_size + multiplier * detection['pos']
                self.detection_image[
                    0,
                    max(pos -
                        20, 0):min(pos + 20, self.channel_size *
                                   2), :] = self.detection_colors[object_id]

            detection_msg.results.append(object_hypothesis)
            detection_array_msg.detections.append(detection_msg)
        return detection_array_msg

    def _detection_to_pose(self, pos, channel_id):
        """Given detected pos (index in the sidescan ping), channel_id
        (Side.PORT or Side.STARBOARD) and resolution, return the constructed
        pose for the detection"""
        detected_pose = Pose()
        hypothenus = pos * self.resolution
        height_diff = self.object_height - self.vehicle_z_pos
        distance = (hypothenus**2 - height_diff**2)**.5
        detected_pose.position.y = distance
        if channel_id == Side.PORT:
            detected_pose.position.y *= -1
        return detected_pose


def main():
    args = parse_arg()
    print(args)
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

    detector = sss_detector(robot_name=robot_name,
                            object_height=args.object_height,
                            water_depth=args.water_depth)

    while not rospy.is_shutdown():
        rospy.spin()


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object-height',
                        help='Rope height under water [m]',
                        default=2,
                        type=float)
    parser.add_argument('--water-depth',
                        help='Approximate depth of the water [m]',
                        default=15,
                        type=float)
    return parser.parse_args()


if __name__ == '__main__':
    main()
