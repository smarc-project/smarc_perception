#!/usr/bin/env python3
import rospy
from sss_object_detection import consts
from sss_object_detection.sss_detector_image_processing import SSSDetector_image_proc


def main():
    rospy.init_node('sss_detection_publisher', anonymous=True)
    rospy.Rate(5)  # ROS Rate at 5Hz

    robot_name_param = '~robot_name'
    if rospy.has_param(robot_name_param):
        robot_name = rospy.get_param(robot_name_param)
        print(f'Getting robot_name = {robot_name} from param server')
    else:
        robot_name = 'sam'
        print('{} param not found in param server.\n'.format(robot_name_param))
        print(f'Setting robot_name = {robot_name} (default value)')

    object_height = rospy.get_param('~object_height', 1.50)
    water_depth = rospy.get_param('~water_depth', 15)

    buoy_path = rospy.get_param('buoy_detections_path',
                                '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/image_process_buoys.csv')

    rope_port_path = rospy.get_param('rope_port_detections_path',
                                     '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/image_process_ropes_port.csv')

    rope_star_path = rospy.get_param('rope_star_detections_path',
                                     '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/image_process_ropes_star.csv')

    detector = SSSDetector_image_proc(robot_name=robot_name,
                                      object_height=object_height,
                                      water_depth=water_depth,
                                      buoy_path=buoy_path,
                                      rope_port_path=rope_port_path,
                                      rope_star_path=rope_star_path)

    while not rospy.is_shutdown():
        rospy.spin()


if __name__ == '__main__':
    main()
