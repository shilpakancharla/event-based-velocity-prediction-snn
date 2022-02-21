# Motion Segmentation with Spiking Neural Networks (SNNs)

## Open Event Camera Simulation (ESIM) with ROS Noetic (Ubuntu 20.04) and Python 3

These are installation instructions to run the ESIM on ROS Noetic on Ubuntu 20.04 with Python 3. Previous versions of working ESIM programs have been run with ROS Kinetic on Ubuntu 16.04 and ROS Melodic on Ubuntu 18.04. Credit to ESIM belongs to the [Robotics and Perception Group at the University of Zurich](https://rpg.ifi.uzh.ch/index.html). Here is the [original paper](https://rpg.ifi.uzh.ch/docs/CORL18_Rebecq.pdf) about the event camera simulator as well. These instructions were created from the original instructions given on [ESIM's Wiki](https://github.com/uzh-rpg/rpg_esim/wiki/installation) but adapted to suit ROS Noetic. Note that some of the issues may not be reproducible, and further issues faced should be documented on the Issues section of the is repository or on the Issues section of the [ESIM repository](https://github.com/uzh-rpg/rpg_esim).

### Installation

Start by ensuring that you have install ROS Noetic and Python 3. Instructions to install ROS Noetic can be found [here](http://wiki.ros.org/noetic/Installation/Ubuntu). Before moving on to creating the workspace, run the following just in case you have another version of ROS or you are unsure if you have already run these statements before (perhaps from your root directory):

```
sudo rosdep init
rosdep update
rosinstall_generator desktop --rosdistro noetic --deps --tar > noetic_desktop_rosinstall
mkdir ./src
vcs import --input noetic-desktop.rosinstall ./src
rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro noetic -y
```

Next, create a new catkin workspace that will hold contents related to this simulator. Do so by executing the following, perhaps starting in your root directory:

```
 mkdir ~/sim_ws
 cd ~/sim_ws
```

### Running ESIM
