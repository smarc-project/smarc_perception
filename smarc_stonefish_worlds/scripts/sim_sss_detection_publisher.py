#!/usr/bin/env python

import rospy
import numpy as np
import tf
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PoseStamped, PointStamped
from tf.transformations import euler_from_quaternion
import copy
import math


class sim_sss_detector:
    """A mock SSS object detector for simulation. Only objects within the
    detection_range of the vehicle will be detectable."""
    def __init__(self, detection_range=8, buoy_radius=0.20):
        self.detection_range = detection_range
        self.buoy_radius = buoy_radius
        self.prev_pose = None
        self.current_pose = None
        self.yaw = None
        self.frame_id = None
        self.marked_positions = {}

        self.tf_listener = tf.TransformListener()
        self.odom_sub = rospy.Subscriber('/sam/dr/odom', Odometry,
                                         self._update_pose)
        self.marked_pos_sub = rospy.Subscriber('/sam/sim/marked_positions',
                                               MarkerArray,
                                               self._update_marked_positions)
        self.pub = rospy.Publisher('/sam/sim/sidescan/detection',
                                   Detection2DArray,
                                   queue_size=2)
        self.pub_detected_markers = rospy.Publisher(
            '/sam/sim/sidescan/detected_markers', Marker, queue_size=2)

    def _update_marked_positions(self, msg):
        """Update marked_positions based on the MarkerArray msg received."""
        if len(self.marked_positions) > 0:
            return
        for marker in msg.markers:
            self.marked_positions[f'{marker.ns}/{marker.id}'] = marker
        print(
            f'There are {len(self.marked_positions)} number of marked positions'
        )

    def _update_pose(self, msg):
        """Update prev_pose and current_pose according to the odom msg received"""
        if not self.prev_pose:
            self.prev_pose = msg.pose.pose
            self.current_pose = msg.pose.pose

        self.frame_id = msg.header.frame_id
        self.prev_pose = self.current_pose
        self.current_pose = msg.pose.pose

        markers_in_range = self.get_markers_in_detection_range()
        heading = self.calculate_heading()

        if len(markers_in_range) > 0:
            print(
                f'{len(markers_in_range)} markers are within detection range: {markers_in_range}'
            )
        for marker in markers_in_range:
            cos_sim = self.calculate_marker_cosine_angle(heading, marker)
            detectable = cos_sim <= self.buoy_radius

            if detectable:
                print(f'\t{marker} is within detection angle! Cos = {cos_sim}')
                self._publish_marker_detection(marker)

    def _publish_marker_detection(self, marker):
        """Publish detected marker"""
        detected_marker = copy.deepcopy(self.marked_positions[marker])
        detected_marker.header.stamp = rospy.Time.now()
        detected_marker.ns = f'detected_{detected_marker.ns}'
        detected_marker.color = ColorRGBA(0, 1, 0, 1)
        detected_marker.lifetime.secs = 1
        self.pub_detected_markers.publish(detected_marker)

    def _get_position_differences(self, position1, position2):
        dx = position1.x - position2.x
        dy = position1.y - position2.y
        dz = position1.z - position2.z
        return dx, dy, dz

    def _normalize_vector(self, position_array):
        """Given an np.ndarray, return the normalized equivalent"""
        norm = np.linalg.norm(position_array)
        if norm > 0:
            position_array = position_array / norm
        return position_array

    def calculate_heading(self):
        """Calculate a normalized heading vector using current orientation"""
        quaternion = self.current_pose.orientation
        (_, pitch, yaw) = euler_from_quaternion(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        heading = np.array([
            math.cos(yaw) * math.cos(pitch),
            math.sin(yaw) * math.cos(pitch),
            math.sin(pitch)
        ]).reshape(-1, 1)
        heading = self._normalize_vector(heading)
        return heading

    def _calculate_distance_to_position(self, position):
        """Calculate the distance between current_pose.position and the given position"""
        dx, dy, dz = self._get_position_differences(position,
                                                    self.current_pose.position)
        return (dx**2 + dy**2 + dz**2)**.5

    def _get_vec_to_position(self, position, normalized=True):
        """Return vector from current_pose.position to the given position"""
        dx, dy, dz = self._get_position_differences(position,
                                                    self.current_pose.position)
        vec_to_position = np.array([dx, dy, dz]).reshape(-1, 1)

        if normalized:
            vec_to_position = self._normalize_vector(
                position_array=vec_to_position)
        return vec_to_position

    def _construct_pose_stamped_from_marker_msg(self, marker):
        marker_pose_stamped = PoseStamped()
        marker_pose_stamped.pose = marker.pose
        marker_pose_stamped.header.stamp = rospy.Time.now()
        marker_pose_stamped.header.frame_id = marker.header.frame_id
        return marker_pose_stamped

    def _get_distance_to_marker(self, marker):
        """Return distance from the marker to current_pose"""
        marker_pose_stamped = self._construct_pose_stamped_from_marker_msg(
            marker)
        marker_transformed = self.tf_listener.transformPose(
            self.frame_id, marker_pose_stamped)

        distance = self._calculate_distance_to_position(
            marker_transformed.pose.position)
        return distance

    def get_markers_in_detection_range(self):
        """Returns a list of markers within detection_range relative to
        self.current_pose"""
        markers_in_range = []
        for marker_name, marker in self.marked_positions.items():
            distance = self._get_distance_to_marker(marker)
            if distance < self.detection_range:
                markers_in_range.append(marker_name)
        return markers_in_range

    def calculate_marker_cosine_angle(self, heading, marker):
        """Calculate the cosine between the heading and the marker position.
        Used to determine whether the marker is observable:
        A marker is observable if the magnitude of the projection of the vector
        from self.current_pose.position onto the heading vector <= the marker's radius."""
        marker_pose_stamped = self._construct_pose_stamped_from_marker_msg(
            self.marked_positions[marker])
        marker_transformed = self.tf_listener.transformPose(
            self.frame_id, marker_pose_stamped)
        vec_to_marker_position = self._get_vec_to_position(
            marker_transformed.pose.position, normalized=True)
        cos_heading_marker = np.dot(heading.reshape(1, -1),
                                    vec_to_marker_position.reshape(-1,
                                                                   1))[0][0]
        return abs(cos_heading_marker)


def main():
    rospy.init_node('sim_sss_detection_publisher', anonymous=True)
    rospy.Rate(5)  # ROS Rate at 5Hz

    detector = sim_sss_detector()
    while not rospy.is_shutdown():
        rospy.spin()


if __name__ == '__main__':
    main()
