## Introduction
This repository contains the code for the paper **"Semi-Supervised Online Continual Learning for 3D Object Detection in Mobile Robotics."**

## Abstract
Continual learning addresses the challenge of acquiring and retaining knowledge over time across multiple tasks and environments. Previous research primarily focuses on offline settings where models learn through increasing tasks from samples paired with ground truth annotations. In this work, we focus on an unsolved, challenging, yet practical scenario: specifically, semi-supervised online continual learning in autonomous driving and mobile robotics. In our settings, models are tasked with learning new distributions from streaming unlabeled samples and performing 3D object detection as soon as the LiDAR point cloud arrives. Additionally, we conducted experiments on both the KITTI dataset, our newly built IUSL dataset, and the Canadian Adverse Driving Conditions (CADC) dataset. The results indicate that our method achieves a balance between rapid adaptation and knowledge retention, showcasing its effectiveness in the dynamic and complex environment of autonomous driving and mobile robotics.
## Dataset
- The collected IUSL dataset can be get through [Baidu Netdisk](https://pan.baidu.com/s/1Zu4pUBITQt-lA0LHD7FCIg?pwd=iusl) code: iusl.


## How to Build & Run

The ros1 branch uses the following implementations:

- The powerful streaming learning classifier **AMF**, implemented by [river](https://github.com/online-ml/river).
- **Patchwork++** to remove ground points ([repository](https://github.com/url-kaist/patchwork-plusplus)).
- The pretrained **YOLOv8** as the image detector ([repository](https://github.com/ultralytics/ultralytics)).

### ROS1 IUSL Bag Pipeline

This branch is configured for ROS Noetic and the IUSL bag:

```bash
/media/ros/SSData/dataset/iusl/sensor_fusion_data/2023-04-25-16-37-35.bag
```

The recommended pipeline uses handcrafted LiDAR clustering features, YOLOv8 image detections, and the Python online random forest node. The online forest is pre-trained from offline initial samples before it starts consuming online feature callbacks:

```bash
online_forests_ros/data/initial_samples_iusl_bag_2023_04_25.jsonl
```

After startup, online samples are generated from `/point_cloud_features_global/features_global`. Prediction and training run asynchronously: one thread publishes `/online_random_forest/rf_label`, while another thread consumes online samples and updates the replay buffer. The replay buffer is fixed-size per class and uses reservoir sampling. Current defaults are:

```text
person label: 1
unknown label: 9
rf_n_estimators: 25
replay_buffer_size_per_class: 512
replay_samples_per_class: 2
online_training_queue_size: 256
```

### Steps to Run

1. Clone the repository:
   ```bash
   mkdir -p ~/ocl3d_ws/src
   cd ~/ocl3d_ws/src
   git clone -b ros1 https://github.com/npu-ius-lab/OCL3D.git
   ```

2. Navigate to the workspace and build:
   ```bash
   cd ~/ocl3d_ws
   source /opt/ros/noetic/setup.bash
   catkin_make --source src/OCL3D
   ```

3. Start the handcrafted-feature IUSL pipeline:
   ```bash
   source /opt/ros/noetic/setup.bash
   source ~/ocl3d_ws/devel/setup.bash
   roslaunch ~/ocl3d_ws/src/OCL3D/launch/efficient_online_learning_iusl_bag.launch rviz:=true
   ```

4. Play the bag in another terminal:
   ```bash
   source /opt/ros/noetic/setup.bash
   rosbag play --clock /media/ros/SSData/dataset/iusl/sensor_fusion_data/2023-04-25-16-37-35.bag
   ```

5. Optional FAST-LIO launch:
   ```bash
   source /opt/ros/noetic/setup.bash
   source ~/ocl3d_ws/devel/setup.bash
   roslaunch ~/ocl3d_ws/src/OCL3D/launch/efficient_online_learning_iusl_bag_fastlio.launch rviz:=true
   ```

Useful visualization topics include:

```text
/autoware_tracker/cluster/objects
/autoware_tracker/cluster/image_associated_boxes
/online_random_forest/rf_label
/autoware_tracker/visualizer/forests_objects
```

You can override the online learning parameters from the launch command, for example:

```bash
roslaunch ~/ocl3d_ws/src/OCL3D/launch/efficient_online_learning_iusl_bag.launch \
  rviz:=true \
  replay_samples_per_class:=2 \
  replay_buffer_size_per_class:=512
```

Before running the shell scripts in this repository, check that their workspace paths match your local checkout.

## Important Note:
**Patchwork++ cannot be compiled with OCL3D in the same workspace. Please place them in two separate workspaces.**

# Cite
```
@article{liu2024semi,
  title={Semi-Supervised Online Continual Learning for 3D Object Detection in Mobile Robotics},
  author={Liu, Binhong and Yao, Dexin and Yang, Rui and Yan, Zhi and Yang, Tao},
  journal={Journal of Intelligent \& Robotic Systems},
  volume={110},
  number={4},
  pages={1--16},
  year={2024},
  publisher={Springer}
}
```
