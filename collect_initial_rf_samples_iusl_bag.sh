#!/usr/bin/env bash
set -euo pipefail

BAG_PATH="${BAG_PATH:-/media/ros/SSData/dataset/iusl/sensor_fusion_data/2023-04-25-16-37-35.bag}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_SETUP="${WORKSPACE_SETUP:-$(cd "$REPO_DIR/.." && pwd)/devel/setup.bash}"
OUTPUT_FILE="${OUTPUT_FILE:-$HOME/ocl3d_imf_workdir/iusl_bag_2023_04_25/initial_samples.jsonl}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-100}"
MAX_FRAMES="${MAX_FRAMES:-600}"
RATE="${RATE:-1.0}"

if [[ ! -f "$BAG_PATH" ]]; then
  echo "Bag file not found: $BAG_PATH" >&2
  exit 1
fi

if [[ ! -f "$WORKSPACE_SETUP" ]]; then
  echo "Workspace setup not found: $WORKSPACE_SETUP" >&2
  echo "Build from the catkin workspace root first, for example: catkin_make --source OCL3D" >&2
  exit 1
fi

source "$WORKSPACE_SETUP"

cleanup() {
  jobs -pr | xargs -r kill
}
trap cleanup EXIT

if ! rostopic list >/dev/null 2>&1; then
  roscore &
  sleep 2
fi

roslaunch "$REPO_DIR/launch/collect_initial_rf_samples_iusl_bag.launch" \
  output_file:="$OUTPUT_FILE" \
  samples_per_class:="$SAMPLES_PER_CLASS" \
  max_frames:="$MAX_FRAMES" &

sleep 4
rosbag play --clock -r "$RATE" "$BAG_PATH"
