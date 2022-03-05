#! /usr/bin/bash
mkdir -p "$1"
cd "$1"

youtube-dl "$2" -o "$5"

ffmpeg -i "$5".mkv -ss "$3" -t "$4" -async 1 -strict -2 "$5"_cut.mkv 

mkdir frames_"$5"
ffmpeg -i "$5"_cut.mkv frames_"$5"/frames_%010d.png
touch frames_"$5"/images.csv

ssim
roscd esim_ros
python scripts/generate_stamps_file.py -i "$1"/frames -r 1200.0

rosrun esim_ros esim_node \
 --data_source=2 \
 --path_to_output_bag="$1"/bag/out_"$5".bag \
 --path_to_data_folder="$1"/frames \
 --ros_publisher_frame_rate=60 \
 --exposure_time_ms=10.0 \
 --use_log_image=1 \
 --log_eps=0.1 \
 --contrast_threshold_pos=0.15 \
 --contrast_threshold_neg=0.15

gnome-terminal --noclose
ssim
rosrun dvs_renderer dvs_renderer events:=/cam0/events

gnome-terminal --noclose
ssim
rqt_image_view /dvs_rendering