## Introduction
This repository contains the code for the paper **"Semi-Supervised Online Continual Learning for 3D Object Detection in Mobile Robotics."**

## Abstract
Continual learning addresses the challenge of acquiring and retaining knowledge over time across multiple tasks and environments. Previous research primarily focuses on offline settings where models learn through increasing tasks from samples paired with ground truth annotations. In this work, we focus on an unsolved, challenging, yet practical scenario: specifically, semi-supervised online continual learning in autonomous driving and mobile robotics. In our settings, models are tasked with learning new distributions from streaming unlabeled samples and performing 3D object detection as soon as the LiDAR point cloud arrives. Additionally, we conducted experiments on both the KITTI dataset, our newly built IUSL dataset, and the Canadian Adverse Driving Conditions (CADC) dataset. The results indicate that our method achieves a balance between rapid adaptation and knowledge retention, showcasing its effectiveness in the dynamic and complex environment of autonomous driving and mobile robotics.

## How to Build & Run

We use the following implementations:

- **PointPillars** from [mmdetection3d v1.0.0rc6](https://github.com/open-mmlab/mmdetection3d).
- The powerful streaming learning classifier **AMF**, implemented by [river](https://github.com/online-ml/river).
- **Patchwork++** to remove ground points ([repository](https://github.com/url-kaist/patchwork-plusplus)).
- The pretrained **YOLOv8** as the image detector ([repository](https://github.com/ultralytics/ultralytics)).

### Steps to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/npu-ius-lab/OCL3D.git catkin_ws/src
2. Navigate to the workspace and build:
cd catkin_ws
catkin_make
3. We provide two scripts to run the OCL3D method:
Run OCL3D on the KITTI tracking dataset using PointNet features and PointPillars:
./run_kitti_pillars_pointnet.sh
Run OCL3D on the IUSL dataset using handcrafted features:
./run_iusl_hand.sh
