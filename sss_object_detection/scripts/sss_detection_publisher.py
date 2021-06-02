#!/usr/bin/env python3
import argparse
import rospy
from sss_object_detection import consts
from sss_object_detection.sss_detector import SSSDetector


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

    detector = SSSDetector(robot_name=robot_name,
                           object_height=args.object_height,
                           water_depth=args.water_depth)

    while not rospy.is_shutdown():
        rospy.spin()


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object-height',
                        help='Object height under water [m]',
                        default=0,
                        type=float)
    parser.add_argument('--water-depth',
                        help='Approximate depth of the water [m]',
                        default=15,
                        type=float)
    return parser.parse_args()


if __name__ == '__main__':
    main()
