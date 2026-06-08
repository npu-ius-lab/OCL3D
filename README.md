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

### Steps to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/npu-ius-lab/OCL3D.git catkin_ws/src
   ```

2. Navigate to the workspace and build:
   ```bash
   cd catkin_ws
   catkin_make
   ```

3. Run OCL3D on the IUSL dataset using handcrafted LiDAR clustering features:
     ```bash
     ./run_iusl_hand.sh
     ```
   Before running the script, please carefully check the ROS workspace path for each package.
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

