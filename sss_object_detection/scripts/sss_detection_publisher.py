#!/usr/bin/env python3
import rospy
from sss_object_detection import consts
from sss_object_detection.sss_detector import SSSDetector


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
    object_height = rospy.get_param('~object_height')
    water_depth = rospy.get_param('~water_depth')

    detector = SSSDetector(robot_name=robot_name,
                           object_height=object_height,
                           water_depth=water_depth)

    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == '__main__':
    main()
