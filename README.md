# smarc_perception
Perception algorithms for underwater sonars and cameras

## Installation
This package is strongly coupled with the other packages in [smarc-project](https://github.com/smarc-project). For installation of SMaRC software packages, see [smarc-project/rosinstall](https://github.com/smarc-project/rosinstall).

Each folder in this project is a stand-alone ROS package, with the package name being the same as the folder name.

## Existing ROS packages in SMaRC perception
### [sss_object_detection](https://github.com/smarc-project/smarc_perception/tree/noetic-devel/sss_object_detection)
This package implements sidescan sonar object detection, more specifically bouy and rope detection, for SMaRC simulations and vehicles. The detection results are published as [Detection2DArray](http://docs.ros.org/en/lunar/api/vision_msgs/html/msg/Detection2DArray.html) messages at the topic `{robot_name}/payload/sidescan/detection_hypothesis` in the frame `{robot-name}/base_link`.

Currently, the algorithms for the simulated and real-world detection are completely disjoint. This is because of the simulated sidescan sonar signals looking very different than the real-world signals, as described in [this stonefish issue](https://github.com/patrykcieslak/stonefish/issues/21). As mentioned in the issue, part of the problem is most likely the rolling of the vehicle model of [SAM](https://github.com/smarc-project/smarc_stonefish_sims/tree/noetic-devel/sam_stonefish_sim).

#### Simulated SSS detection
Due to the abovementioned issues with the simulated sidescan, the SSS detection for the simulation ignores the simulated sidescan topics entirely, but instead rely on the [marked_pos_publisher](https://github.com/smarc-project/smarc_utils/tree/noetic-devel/marked_pos_publisher) from the [smarc_utils](https://github.com/smarc-project/smarc_utils) repo to publish the groundtruth position of marked positions (currently the buoy positions for the algae farm scenario).

To publish simulated detections, the ROS node [sim_sss_detection_publisher](https://github.com/smarc-project/smarc_perception/blob/noetic-devel/sss_object_detection/scripts/sim_sss_detection_publisher.py) subscribes to the groundtruth poses of the marked positions, compute the range and angle between the simulated vehicle and all marked positions, and then publish the positions of the markers that should be visible by the vehicle as detection results.

##### Running simulated SSS detection
The launch file for the simulated detection is located at [sim_sss_buoy_detection.launch](https://github.com/smarc-project/smarc_perception/blob/noetic-devel/sss_object_detection/launch/sim_sss_buoy_detection.launch). To launch the simulated detection, run the following: 
```
roslaunch sss_object_detection sim_sss_buoy_detection.launch
```

#### Real SSS detection
The real SSS detection subscribes to the [Sidescan message](https://github.com/smarc-project/smarc_msgs/blob/noetic-devel/msg/Sidescan.msg) published at the topic `/{robot_name}/payload/sidescan` and uses a 1D sliding window change point detection method to detect buoys and ropes. The computation of the detection message based on the vehicle pose is implemented in [sss_detector.py](https://github.com/smarc-project/smarc_perception/blob/noetic-devel/sss_object_detection/src/sss_object_detection/sss_detector.py), which uses the CPDetector in [cpd_detector](https://github.com/smarc-project/smarc_perception/blob/noetic-devel/sss_object_detection/src/sss_object_detection/cpd_detector.py) for the actual 1D change detection.

##### Running real SSS detection
The launch file for the simulated detection is located at [sss_detection.launch](https://github.com/smarc-project/smarc_perception/blob/noetic-devel/sss_object_detection/launch/sss_detection.launch). To launch the real detection, run the following: 
```
roslaunch sss_object_detection sss_detection.launch
```
