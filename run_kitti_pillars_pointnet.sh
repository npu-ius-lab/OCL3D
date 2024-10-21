#!/bin/bash
while getopts ":s:b:c:" opt; do
  case $opt in
    s)
      kitti_scence="$OPTARG"
      ;;
    b)
      arg2="$OPTARG"
      ;;
    c)
      arg3="$OPTARG"
      ;;
    \?)
      echo "未知参数"
      exit 1
      ;;
  esac
done


echo "running kitti scence $kitti_scence"

gnome-terminal -t "start_ros" -x bash -c "roscore;  exec bash"
sleep 1s
gnome-terminal -t "start_feature" -x bash -c "source ~/anaconda3/bin/activate; conda activate py38torch; source ~/online_learning_git/devel/setup.bash;roslaunch point_cloud_features pointnet_features.launch;  exec bash"
sleep 30s

gnome-terminal -t "start_pp" -x bash -c "source ~/anaconda3/bin/activate; conda activate mmdet3d; source ~/online_learning_git/devel/setup.bash;roslaunch pointpillars_ros pointpillars.launch output:=false scence:=0012;  exec bash"
sleep 10s

gnome-terminal -t "start_imf" -x bash -c "source ~/anaconda3/bin/activate; conda activate py38torch; source ~/online_learning_git/devel/setup.bash;roslaunch online_forests_ros online_forests_ros_kitti.launch scence:="$kitti_scence";  exec bash"
sleep 5s

gnome-terminal -t "start_patchwork" -x bash -c "source ~/deep_learning/patchwork_ws/devel/setup.bash;roslaunch patchworkpp demo.launch;  exec bash"
sleep 2s


gnome-terminal -t "start_online" -x bash -c "source ~/anaconda3/bin/activate; conda activate torch; source ~/online_learning_git/devel/setup.bash;roslaunch src/launch/efficient_online_learning_pillars_pointnet.launch scence:="$kitti_scence";exec bash"


