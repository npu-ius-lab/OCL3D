#!/usr/bin/env bash
set -euo pipefail

BAG_PATH="${BAG_PATH:-/media/ros/SSData/dataset/iusl/sensor_fusion_data/2023-04-25-16-37-35.bag}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_SETUP="${WORKSPACE_SETUP:-$(cd "$REPO_DIR/.." && pwd)/devel/setup.bash}"
SAVE_DIR="${SAVE_DIR:-$HOME/ocl3d_imf_workdir/iusl_bag_2023_04_25}"
RVIZ="${RVIZ:-false}"
RATE="${RATE:-0.5}"

if [[ ! -f "$BAG_PATH" ]]; then
  echo "Bag file not found: $BAG_PATH" >&2
  exit 1
fi

if [[ ! -f "$WORKSPACE_SETUP" ]]; then
  echo "Workspace setup not found: $WORKSPACE_SETUP" >&2
  echo "Build from the catkin workspace root first, for example: catkin_make" >&2
  exit 1
fi

source "$WORKSPACE_SETUP"

cleanup() {
  jobs -pr | xargs -r kill
}
trap cleanup EXIT

roscore &
sleep 2

roslaunch "$REPO_DIR/launch/efficient_online_learning_iusl_bag.launch" rviz:="$RVIZ" save_dir:="$SAVE_DIR" &
sleep 4

rosbag play --clock -r "$RATE" "$BAG_PATH"
