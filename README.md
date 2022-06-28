# Event-Based Velocity Prediction with Spiking Neural Networks (SNNs)

## Abstract 
Neuromorphic computing uses very-large-scale integration (VLSI) systems with the goal of replicating neurobiological structures and signal conductance mechanisms. Neuromorphic processors can run spiking neural networks (SNNs) that mimic how biological neurons function, particularly by emulating the emission of electrical spikes. A key benefit of using SNNs and neuromorphic technology is the ability to optimize the size, weight, and power consumed in a system. SNNs can be trained and employed in various robotic and computer vision applications; we attempt to use event-based to create a novel approach in order to the predict velocity of objects moving in frame. Data generated in this work is recorded and simulated as event camera data using ESIM. Vicon motion tracking data provides the ground truth position and time values, from which the velocity is calculated. The SNNs developed in this work regress the velocity vector, consisting of the x, y, and z-components, while using the event data, or the list of events associated with each velocity measurement, as the input features. With the use of the novel dataset created, three SNN models were trained and then the model that minimized the loss function the most was further validated by omitting a subset of data used in the original training. The average loss, in terms of RMSE, on the test set after using the trained model on the omitted subset of data was 0.000386. Through this work, it is shown that it is possible to train an SNN on event data in order to predict the velocity of an object in view.

## Contributions

This work provides the following contributions:

* An approach to how to set up a spiking neural network to predict numeric values, as opposed to classifying data points, using regression. 
* An approach to set up a neural network with convolutional and recurrent features with event-based data.
* The open-source code and dataset, found here https://zenodo.org/record/6762053#.YrtfPjfMITs. 
* Methods to preprocess event-based data and create a custom dataset using the frameworks based in PyTorch, such as `snntorch` and `Tonic`.

## Pipeline: Data Generation to Modeling

1. Vicon motion tracking data and video are calibrated to have the same timestamps.
2. The trimmed video is put through the Event Camera Simulator (ESIM) to generate an event rosbag.
3. The event rosbag is read and the event data is outputted to a text file using `extract_events_from_rosbag.py`.
4. Various functions in `utils.py` are used to calculate the ground truth velocities and match them with the event data from the textfile. A .csv file is produced. 

## Preparing Data with ESIM

The bash script `event_simulation.sh` is provided to automate the event camera simulation, and gather the simulation data by topic from the output rosbag. You need the ESIM software set up, `youtube-dl`, and `ffmpeg` all on your Linux system. The following parameters required to run this bash script are:

1. Workspace directory
2. Video URL
3. Start time of video
4. How much time from start of video you should trim
5. Name of video

Example of running script: 

`event_simulation.sh "workspace/dataset/" "www.youtube.com/<some url>" "00:00:00" "00:00:40" "wand_1"`

## Run Videos in ESIM Manually

### Download the video

These instructions are taken from the [ESIM tutorial](https://github.com/uzh-rpg/rpg_esim/wiki/Simulating-events-from-a-video).

Steps to get started creating simulated event camera video data:

1. Create a working folder:

```
mkdir -p workspace/<project name>/
cd workspace/<project name>/
```

2. Download video from YouTube:

```
youtube-dl <video URL> -o <name>
```

3. Cut relevant part of video:

```
ffmpeg -i <video name>.mkv -ss <start timestamp> -t <end timestamp> -async 1 -strict -2 <cut video name>.mkv
```

Option to zoom in twice in case there are unecessary contents in the video: `ffmpeg -i box1.mkv -ss 00:00:08 -t 00:01:22 -async 1 -strict -2 -vf "scale=2*iw:-1, crop=iw/2:ih/2" box1_cut.mkv`

### Pre-process video for ESIM

1. Export video to `frames` folder:

```
mkdir frames
ffmpeg -i <cut video name>.mkv frames/frames_%010d.png
```

2. Create `images.csv` in the `frames` folder, which is necessary for ESIM. Each line of this file contains the image timestamp and path in the format `timestamp_in_nanoseconds,frames_xxxxx.png` (:

```
ssim
roscd esim_ros
python scripts/generate_stamps_file.py -i /workspace/<project name>/frames -r 1200.0
```

3. Open a new terminal and run `roscore`. Go back to your other terminal and run:

```
rosrun esim_ros esim_node \
 --data_source=2 \
 --path_to_output_bag=<path to bag output folder> \
 --path_to_data_folder=<path_to_frames_folder> \
 --ros_publisher_frame_rate=60 \
 --exposure_time_ms=10.0 \
 --use_log_image=1 \
 --log_eps=0.1 \
 --contrast_threshold_pos=0.15 \
 --contrast_threshold_neg=0.15
```

4. Open a new terminal to visualize simulated event data. Every time you open a new terminal, run `ssim`. Start the `dvs_renderer`:

```
rosrun dvs_renderer dvs_renderer events:=/cam0/events
```

5. You can now visualize this by opening a new terminal and running `rqt_image_view /dvs_rendering`.
 
## Open Event Camera Simulation (ESIM) with ROS Noetic (Ubuntu 20.04) and Python 3

These are installation instructions to run the ESIM on ROS Noetic on Ubuntu 20.04 with Python 3. Previous versions of working ESIM programs have been run with ROS Kinetic on Ubuntu 16.04 and ROS Melodic on Ubuntu 18.04. Credit to ESIM belongs to the [Robotics and Perception Group at the University of Zurich](https://rpg.ifi.uzh.ch/index.html). Here is the [original paper](https://rpg.ifi.uzh.ch/docs/CORL18_Rebecq.pdf) about the event camera simulator as well. These instructions were created from the original instructions given on [ESIM's Wiki](https://github.com/uzh-rpg/rpg_esim/wiki/installation) but adapted to suit ROS Noetic. Note that some of the issues may not be reproducible, and further issues faced should be documented on the Issues section of the is repository or on the Issues section of the [ESIM repository](https://github.com/uzh-rpg/rpg_esim).

### Installation

Start by ensuring that you have install ROS Noetic and Python 3. Instructions to install ROS Noetic can be found [here](http://wiki.ros.org/noetic/Installation/Ubuntu). Before moving on to creating the workspace, run the following just in case you have another version of ROS or you are unsure if you have already run these statements before (perhaps from your root directory):

```
sudo apt update
sudo rosdep init
rosdep update
rosinstall_generator desktop --rosdistro noetic --deps --tar > noetic_desktop_rosinstall
mkdir ./src
sudo pip3 install vcstool
vcs import --input noetic-desktop.rosinstall ./src
rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro noetic -y
```

Next, create a new catkin workspace that will hold contents related to this simulator. Do so by executing the following, perhaps starting in your root directory:

```
 mkdir ~/sim_ws/src
 cd ~/sim_ws
```

You may need to install the catkin tools. To do so, use

```
sudo pip3 install -U catkin_tools
./src/catkin/bin/catkin_make_isolated --install -DCMAKE_BUILD_TYPE=Release
```
If you are in the `sim_ws` (if not, `cd` into it), run `catkin_init_workspace`. This will initialize the workspace. `catkin_make` allows for a standard layout of the workspace. You can also use `cmake CMakeLists.txt`. Now run the following:

```
cd src/
git clone https://github.com/uzh-rpg/rpg_esim
vcs-import < https://github.com/shilpakancharla/rpg_esim/blob/master/dependencies.yaml
```

Install `pcl-ros` and other requirements:

```
sudo apt-get install install ros-noetic-pcl-ros
sudo apt-get install libproj-dev
sudo apt-get install libglfw3 libgl2fw3-dev
sudo apt-get install libglm-dev
```

Optionally install the trajectory server:

```
sudo apt-get install ros-noetic-hector-trajectory-server
```

Disable the following packages:

```
cd ze_oss
touch imp_3rdparty_cuda_toolkit/CATKIN_IGNORE \
      imp_app_pangolin_example/CATKIN_IGNORE \
      imp_benchmark_aligned_allocator/CATKIN_IGNORE \
      imp_bridge_pangolin/CATKIN_IGNORE \
      imp_cu_core/CATKIN_IGNORE \
      imp_cu_correspondence/CATKIN_IGNORE \
      imp_cu_imgproc/CATKIN_IGNORE \
      imp_ros_rof_denoising/CATKIN_IGNORE \
      imp_tools_cmd/CATKIN_IGNORE \
      ze_data_provider/CATKIN_IGNORE \
      ze_geometry/CATKIN_IGNORE \
      ze_imu/CATKIN_IGNORE \
      ze_trajectory_analysis/CATKIN_IGNORE
```

Now, change back into `~/sim_ws`. Run `catkin build esim_ros`. If there is an error thrown about the build space, run `catkin clean -y`, and then run the build command again. You may receive an error about `<cv_bridge/cv_bridge.h>` not being found within one of the `esim_data_provider` files. If this occurs, try the following first to make sure you have OpenCV on your system (further instructions on OpenCV installation [here](https://linuxize.com/post/how-to-install-opencv-on-ubuntu-20-04/)):

```
sudo apt install libopencv-dev python3-opencv
sudo apt-get ros-noetic-cv-bridge
sudo apt-get ros-noetic-vision-opencv
```

If you are still receiving the same error, try the following steps. 

1. Go into `~sim_ws/src/vision_opencv/cv_bridge/include/`.
2. Copy the entire `cv_bridge` folder. 
3. Go into `~sim_ws/src/rpg_esim/event_camera_simulator/esim_data_provider/include/`.
4. Paste the `cv_bridge` folder at this level.

Rerun `catkin build esim_ros`. The installation should succeed. 

Make an alias for the workspace so you can source it easily:

```
echo "source ~/sim_ws/devel/setup.bash" >> ~/setupeventsim.sh
chmod +x ~/setupeventsim.sh
```

In your `.bashrc` file, add the following line: `alias ssim='source ~/setupeventsim.sh'`. Now, whenever you type `ssim` into your terminal this will initialize the simulator workspace. You need to run `bash` first if you are staying the same terminal after editing the `.bashrc` file. 

### Running ESIM

This set of instructions uses the Planar Renderer. You may choose to use another renderer. Run the following commands for Planar Renderer:

```
roscd esim_ros
roslaunch esim_ros esim.launch config:=cfg/example.conf
```

To visualize the output of the simulator, open `rviz` with a new terminal window:

```
roscd esim_visualization
rviz -d cfg/esim.rviz
```

You can also use `rqt`:

```
roscd esim_visualization
rqt --perspective-file cfg/esim.perspective
```
