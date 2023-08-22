import os
import rospy
import numpy as np
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image
from smarc_msgs.msg import Sidescan
from vision_msgs.msg import ObjectHypothesisWithPose, Detection2DArray, Detection2D
from cv_bridge import CvBridge, CvBridgeError

from sss_object_detection.consts import ObjectID, Side
from sss_object_detection.cpd_detector import CPDetector


class SSSDetector_image_proc:
    """
    Class that provides manually determined detections as well as data associations if they are provided
    Detections were determined by visual examination of the sss data and the terminating jounction of each rope section
    was selected.

    The DA is sent in the confidence field of each detection and in the score field of the ObjectHypothesisWithPose()
    that is published.
    """

    def __init__(self, robot_name, water_depth=15, object_height=0,
                 buoy_path=None, rope_port_path=None, rope_star_path=None):
        print('Starting image processing sss detector')
        print("Processing is currently done offline!")
        # Object height below water [m]
        self.object_depth = object_height  # object_height
        self.buoy_depth = 0
        self.vehicle_z_pos = 0
        self.robot_name = robot_name

        # TODO: currently changing to work with new image processing detections
        # This is one of the output files from the image processing script.
        # It is formatted as an array with the following columns [ orig index | seq ID | orig cross index ]
        self.buoy_detections = np.genfromtxt(buoy_path, delimiter=',').astype(int)

        if rope_port_path is not None and os.path.exists(rope_port_path):
            self.rope_port_detections = np.genfromtxt(rope_port_path, delimiter=',').astype(int)
            self.use_rope_port = True
        else:
            self.use_rope_port = False

        if rope_star_path is not None and os.path.exists(rope_star_path):
            self.rope_star_detections = np.genfromtxt(rope_star_path, delimiter=',').astype(int)
            self.use_rope_star = True
        else:
            self.use_rope_star = False

        # TODO: Remove hard coded detections or allow them to be specified in launch
        # The below data is [[index of return (starting at 0), index of the detection]]
        # self.buoy_targets = [[7106, 1092], [6456, 1064],
        #                      [5570, 956], [4894, 943],
        #                      [4176, 956], [3506, 924],
        #                      [2356, 911], [1753, 949],
        #                      [1037, 941], [384, 943]]

        # there is a bug in the sss that causes the channels to get flipped
        # self.detection_flipped = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

        # self.detection_seq_ids = [78107, 78757, 79643, 80319, 81037, 81707, 82857, 83460, 84176, 84829]

        # self.buoy_associations = [3, 2, 0, 5, 4, 1, 1, 4, 3, 2]

        # This is still needed as the rad SSS data needs to be flipped for the visualization
        self.flipped_regions = [[77606, 79385]]

        self.channels_to_detect = [Side.PORT]

        # Real detector - CPD
        self.detector = CPDetector()
        self.detector_max_nadir = 125

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

        # Check if the channels need to flipped
        flip_channels = False
        seq_id = msg.header.seq
        if self.flipped_regions is not None and len(self.flipped_regions) > 0:
            for region in self.flipped_regions:
                if region[0] <= seq_id <= region[1]:
                    flip_channels = True
                    break

        # Unpack the channel data from the msg, and flip if needed
        if flip_channels:
            channels = {
                Side.PORT: np.array(bytearray(msg.starboard_channel), dtype=np.uint8),
                Side.STARBOARD: np.array(bytearray(msg.port_channel), dtype=np.uint8)
            }
        else:
            channels = {
                Side.PORT: np.array(bytearray(msg.port_channel), dtype=np.uint8),
                Side.STARBOARD: np.array(bytearray(msg.starboard_channel), dtype=np.uint8)
            }

        # Update sidescan image
        self.sidescan_image[1:, :, :] = self.sidescan_image[:-1, :, :]
        for i in range(3):
            self.sidescan_image[0, :, i] = np.concatenate([
                np.flip(channels[Side.PORT], axis=0), channels[Side.STARBOARD]
            ])
        self.detection_image[1:, :, :] = self.detection_image[:-1, :, :]
        self.detection_image[0, :, :] = self.sidescan_image[0, :, :]

        # TODO: I want to combine the detections into a hybrid of cbd and manual

        # The manual approach uses the seq ids
        current_seq_id = msg.header.seq
        if current_seq_id in self.buoy_detections[:, 1]:

            detection_index = np.where(self.buoy_detections[:, 1] == current_seq_id)[0][0]
            print(f'IMG PROC BUOY DETECTION - Seq ID: {current_seq_id}  detection index: {detection_index}')
            buoy_index = self.buoy_detections[detection_index, 2]

            # Find the side of the detection
            # Find the distance of the detection
            if buoy_index < self.channel_size:
                # Handle the direction taking into account the sss channel flipping bug
                # if self.detection_flipped[detection_index]:
                #     channel_id = Side.STARBOARD
                # else:
                #     channel_id = Side.PORT
                channel_id = Side.PORT
                buoy_range = self.channel_size - 1 - buoy_index
            else:
                # Handle the direction taking into account the sss channel flipping bug
                # if self.detection_flipped[detection_index]:
                #     channel_id = Side.PORT
                # else:
                #     channel_id = Side.STARBOARD

                channel_id = Side.STARBOARD
                buoy_range = buoy_index - self.channel_size

            # Manual associations,
            # buoy_da = self.buoy_associations[detection_index]

            # TODO check how confidence is used by the graph builder
            # Setting confidence to -ObjectID.BUOY.value will force data association
            detection_res = {ObjectID.BUOY: {'pos': buoy_range,
                                             'confidence': -ObjectID.BUOY.value}}

            detection_msg = self._construct_detection_msg_and_update_detection_image(
                detection_res, channel_id, msg.header.stamp, detection_type=ObjectID.BUOY)
            if len(detection_msg.detections) > 0:
                print(f'Publishing buoy detection - {current_seq_id}')
                self._publish_detection_marker(detection_msg, msg.header)
                self.detection_pub.publish(detection_msg)

        # Perform rope detections when no buoy is detected
        else:
            # # Find the nadir
            # detected_nadir_ind = self.detector.detect_nadir(channels[Side.PORT],
            #                                                 channels[Side.STARBOARD])
            #
            # detection_limit_ind = min(detected_nadir_ind, self.detector_max_nadir)
            #
            # for channel_id, channel in channels.items():
            #     if channel_id in self.channels_to_detect:
            #         ping = channel
            #         detection_res = self.detector.detect_rope(ping, detection_limit_ind)
            #         if detection_res:
            #             detection_msg = self._construct_detection_msg_and_update_detection_image(
            #                 detection_res, channel_id, msg.header.stamp)
            #             if len(detection_msg.detections) > 0:
            #                 self.detection_pub.publish(detection_msg)

            # TODO use img process rope detector instead of cpd
            # PORT
            if self.use_rope_port:
                if current_seq_id in self.rope_port_detections[:, 1]:
                    detection_index = np.where(self.rope_port_detections[:, 1] == current_seq_id)[0][0]
                    rope_index = self.rope_port_detections[detection_index, 2]
                    channel_id = Side.PORT
                    detection_res = {ObjectID.ROPE: {
                        'pos': rope_index,
                        'confidence': 0.5
                    }}

                    detection_msg = self._construct_detection_msg_and_update_detection_image(
                        detection_res, channel_id, msg.header.stamp, detection_type=ObjectID.ROPE)
                    if len(detection_msg.detections) > 0:
                        self.detection_pub.publish(detection_msg)

            # STARBOARD
            if self.use_rope_star:
                if current_seq_id in self.rope_star_detections[:, 1]:
                    detection_index = np.where(self.rope_star_detections[:, 1] == current_seq_id)[0][0]
                    rope_index = self.rope_star_detections[detection_index, 2]
                    channel_id = Side.STARBOARD
                    detection_res = {ObjectID.ROPE: {
                        'pos': rope_index,
                        'confidence': 0.5
                    }}

                    detection_msg = self._construct_detection_msg_and_update_detection_image(
                        detection_res, channel_id, msg.header.stamp, detection_type=ObjectID.ROPE)
                    if len(detection_msg.detections) > 0:
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
            self, detection_res, channel_id, stamp, detection_type):

        if detection_type not in list(ObjectID):
            return Detection2DArray()

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
            detection_pose = self._detection_to_pose(detection['pos'], channel_id, detection_type)

            # Catch Error
            if detection_pose is None:
                continue
            else:
                object_hypothesis.pose.pose = detection_pose

            # Filter out object detection outliers
            if abs(object_hypothesis.pose.pose.position.y) > self.water_depth:
                continue
            else:
                pos = self.channel_size + multiplier * detection['pos']

                # Mark detection image with the detections
                color = self.detection_colors[object_id]
                if object_id == ObjectID.BUOY:
                    thickness = 5
                else:
                    thickness = 1
                self.detection_image[0:thickness, max(pos - 10, 0):min(pos + 10, self.channel_size * 2), :] = color

            detection_msg.results.append(object_hypothesis)
            detection_array_msg.detections.append(detection_msg)
        return detection_array_msg

    def _detection_to_pose(self, pos, channel_id, detection_type):
        """Given detected pos (index in the sidescan ping),
        channel_id (Side.PORT or Side.STARBOARD) and resolution,
        type: rope or buoy, determines assumed depth
        return the constructed pose for the detection"""

        # NOTE: The rope detection allowance is currently not being used as it was cause issues with the detections that
        # were being added.

        # rope leeway allows the detected distance to be less than the height difference
        #leeway_ratio = 0.05
        detected_pose = Pose()
        hypotenuse = pos * self.resolution

        # Adjust assumed depth of target based on type of detection
        if detection_type == ObjectID.BUOY:
            target_depth = self.buoy_depth
        else:
            target_depth = self.object_depth

        height_diff = target_depth - self.vehicle_z_pos

        # Check for distance errors
        # For ropes allow some leeway
        # if height_diff == 0:
        #     height_diff_ratio = 1
        # else:
        #     height_diff_ratio = (hypotenuse-height_diff)/height_diff
        #
        # if self.vehicle_z_pos > target_depth:
        #     height_diff_ratio = 1  # Only make allowances if the SAM is above the target

        # Allowance case
        # if detection_type == ObjectID.ROPE and 0 >= height_diff_ratio >= -abs(leeway_ratio):
        #     print(f"DETECTION ALLOWANCE: {detection_type}\n"
        #           f"Current depth: {self.vehicle_z_pos}\n"
        #           f"Target depth: {target_depth}\n"
        #           f"Height difference: {height_diff}\n"
        #           f"Hypotenuse: {hypotenuse}")
        #
        #     distance = 0
        #
        #     return None

        # Error case - will prevent detection
        if abs(hypotenuse) <= abs(height_diff):
            # print(f"DETECTION ERROR: {detection_type}\n"
            #       f"Hypotenuse < height difference\n"
            #       f"Current depth: {self.vehicle_z_pos}\n"
            #       f"Target depth: {target_depth}\n"
            #       f"Height difference: {height_diff}\n"
            #       f"Hypotenuse: {hypotenuse}")

            return None

        else:
            distance = (hypotenuse ** 2 - height_diff ** 2) ** .5

        detected_pose.position.y = distance
        # base link point forward and left
        if channel_id == Side.STARBOARD:
            detected_pose.position.y *= -1
        return detected_pose
