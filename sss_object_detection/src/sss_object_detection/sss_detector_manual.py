import rospy
import numpy as np
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from smarc_msgs.msg import Sidescan
from vision_msgs.msg import ObjectHypothesisWithPose, Detection2DArray, Detection2D
from cv_bridge import CvBridge, CvBridgeError

from sss_object_detection.consts import ObjectID, Side
from visualization_msgs.msg import Marker, MarkerArray


class SSSDetector_manual:
    """
    Class that provides manually determined detections as well as data associations if they are provided
    Detections were determined by visual examination of the sss data and the terminating jounction of each rope section
    was selected.

    The DA is sent in the confidence field of each detection and in the score field of the ObjectHypothesisWithPose()
    that is published.
    """
    def __init__(self, robot_name, water_depth=15, object_height=0):
        print('Starting manual sss detector')
        # Object height below water [m]
        self.object_height = object_height
        self.vehicle_z_pos = 0
        self.robot_name = robot_name

        # TODO: Remove hard coded detections or allow them to be specified in launch
        # The below data is [[index of return (starting at 0), index of the detection]]
        self.buoy_targets = [[7106, 1092], [6456, 1064],
                             [5570, 956], [4894, 943],
                             [4176, 956], [3506, 924],
                             [2356, 911], [1753, 949],
                             [1037, 941], [384, 943]]

        # there is a bug in the sss that causes the channels to get flipped
        self.detection_flipped = [1, 1, 0, 0, 0, 0, 0, 0, 1, 1]

        #
        self.detection_seq_ids = [78107, 78757, 79643, 80319, 81037, 81707, 82857, 83460, 84176, 84829]

        self.buoy_associations = [3, 2, 0, 5, 4, 1, 1, 4, 3, 2]

        self.sidescan_sub = rospy.Subscriber(
            '/{}/payload/sidescan'.format(robot_name), Sidescan,
            self._sidescan_callback)

        self.detection_pub = rospy.Publisher(
            '/{}/payload/sidescan/detection_hypothesis'.format(robot_name),
            Detection2DArray,
            queue_size=2)

        self.pub_frame_id = '{}/base_link'.format(robot_name)
        self.odom_sub = rospy.Subscriber('/{}/dr/odom/z'.format(robot_name),
                                         Float64, self._update_vehicle_z_pos)

        # TODO: remove temporary hard-coded resolution value (should be read and
        self.resolution = 0.05
        self.channel_size = 1000
        self.water_depth = water_depth

        # Detection visualization
        self.bridge = CvBridge()
        self.sidescan_image = np.zeros((500, self.channel_size * 2, 3),
                                       dtype=np.uint8)
        self.detection_image = np.zeros_like(self.sidescan_image,
                                             dtype=np.uint8)

        # Visualization publishers
        # images
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
            ObjectID.NADIR: (255, 255, 0),  # yellow
            ObjectID.BUOY: (0, 255, 255),  # blue
            ObjectID.ROPE: (255, 0, 0)  # red
        }
        # rviz markers
        # marker publisher
        self.detection_markers = MarkerArray()
        self.detection_count = 0

        self.marker_pub = rospy.Publisher(f'/{robot_name}/real/marked_detections', MarkerArray, queue_size=10)
        self.marker_duration = 10000
        self.marker_scale = 1.0
        self.marker_alpha = 0.5

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

        # The manual approach uses the seq ids
        current_seq_id = msg.header.seq
        if current_seq_id in self.detection_seq_ids:
            print(f'Manual detection - {current_seq_id}')
            detection_index = self.detection_seq_ids.index(current_seq_id)
            buoy_index = self.buoy_targets[detection_index][1]

            # Find the side of the detection
            # Find the distance of the detection
            if buoy_index < self.channel_size:
                # Handle the direction taking into account the sss channel flipping bug
                if self.detection_flipped[detection_index]:
                    channel_id = Side.STARBOARD
                else:
                    channel_id = Side.PORT

                buoy_range = self.channel_size - 1 - buoy_index
            else:
                # Handle the direction taking into account the sss channel flipping bug
                if self.detection_flipped[detection_index]:
                    channel_id = Side.PORT
                else:
                    channel_id = Side.STARBOARD

                buoy_range = buoy_index - self.channel_size

            buoy_da = self.buoy_associations[detection_index]

            detection_res = {ObjectID.BUOY: {'pos': buoy_range,
                                             'confidence': buoy_da}}


            detection_msg = self._construct_detection_msg_and_update_detection_image(
                detection_res, channel_id, msg.header.stamp)
            if len(detection_msg.detections) > 0:
                print(f'Publishing detection - {current_seq_id}')
                self._publish_detection_marker(detection_msg, msg.header)
                self.detection_pub.publish(detection_msg)

        self._publish_sidescan_and_detection_images()
    def _publish_detection_marker(self, detection_message, message_header):
        if len(detection_message.detections) == 0:
            return

        for detection in detection_message.detections:
            for result in detection.results:
                marker = Marker()
                marker.header.frame_id = 'sam/base_link'  # detection.header.frame_id
                marker.header.stamp.secs = message_header.stamp.secs
                marker.header.stamp.nsecs = message_header.stamp.nsecs
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.id = self.detection_count
                marker.lifetime = rospy.Duration(self.marker_duration)
                marker.pose.position.x = result.pose.pose.position.x
                marker.pose.position.y = result.pose.pose.position.y
                marker.pose.position.z = 0.0
                marker.pose.orientation.x = 0
                marker.pose.orientation.y = 0
                marker.pose.orientation.z = 0
                marker.pose.orientation.w = 1
                marker.scale.x = self.marker_scale
                marker.scale.y = self.marker_scale
                marker.scale.z = self.marker_scale
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = self.marker_alpha

                self.detection_markers.markers.append(marker)
                self.detection_count += 1

            print('Publishing detection markers')
            self.marker_pub.publish(self.detection_markers)

    def _publish_sidescan_and_detection_images(self):
        try:
            self.sidescan_image_pub.publish(
                self.bridge.cv2_to_imgmsg(self.sidescan_image, "passthrough"))
            self.detection_image_pub.publish(
                self.bridge.cv2_to_imgmsg(self.detection_image, "passthrough"))
        except CvBridgeError as error:
            print('Error converting numpy array to img msg: {}'.format(error))

    def _construct_detection_msg_and_update_detection_image(
            self, detection_res, channel_id, stamp):
        if channel_id == Side.PORT:
            multiplier = -1
        else:
            multiplier = 1

        detection_array_msg = Detection2DArray()
        # TODO: Check that this is the proper frame, base_link != SSS frame
        detection_array_msg.header.frame_id = self.pub_frame_id
        detection_array_msg.header.stamp = stamp

        for object_id, detection in detection_res.items():
            detection_msg = Detection2D()
            detection_msg.header = detection_array_msg.header

            object_hypothesis = ObjectHypothesisWithPose()
            object_hypothesis.id = object_id.value
            object_hypothesis.score = detection['confidence']
            object_hypothesis.pose.pose = self._detection_to_pose(detection['pos'], channel_id)

            # Filter out object detection outliers
            if abs(object_hypothesis.pose.pose.position.y) > self.water_depth:
                continue
            else:
                pos = self.channel_size + multiplier * detection['pos']
                self.detection_image[
                0,
                max(pos -
                    10, 0):min(pos + 10, self.channel_size *
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
        distance = (hypothenus ** 2 - height_diff ** 2) ** .5
        detected_pose.position.y = distance
        # base link point forward snd left
        if channel_id == Side.STARBOARD:
            detected_pose.position.y *= -1
        return detected_pose
