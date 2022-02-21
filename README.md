# Motion Segmentation with Spiking Neural Networks (SNNs)

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

### Running ESIM
