
while getopts ":s:n:c:" opt; do
  case $opt in
    s)
      iusl_scence="$OPTARG"
      ;;
    n)
      noise_scence="$OPTARG"
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

echo "running iusl scence 0001,noise scence $noise_scence"
gnome-terminal -t "start_ros" -x bash -c "roscore;  exec bash"
sleep 1s



gnome-terminal -t "start_imf" -x bash -c "source ~/anaconda3/bin/activate; conda activate py38torch; source ~/online_learning_git/devel/setup.bash;roslaunch online_forests_ros online_forests_ros.launch kitti:=false scence:=0001 noise:=$noise_scence;  exec bash"
sleep 5s

gnome-terminal -t "start_patchwork" -x bash -c "source ~/deep_learning/patchwork_ws/devel/setup.bash;roslaunch patchworkpp demo_iusl.launch;  exec bash"
sleep 2s


gnome-terminal -t "start_online" -x bash -c "source ~/anaconda3/bin/activate; conda activate torch; source ~/online_learning_git/devel/setup.bash;roslaunch src/launch/efficient_online_learning_iusl.launch scence:=0001 visibility:="$noise_scence";exec bash"

















































# gnome-terminal -t "start_railway_detect" -x bash -c "source /opt/ros/melodic/setup.bash; source /home/ros/anaconda3/bin/activate; source /home/ros/cv_bridge_ws/devel/setup.bash;
# source /home/ros/aeb_ws_new/devel/setup.bash;roslaunch /home/ros/aeb_ws_new/src/launch/railway_detect_final.launch; exec bash"
