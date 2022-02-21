FROM osrf/ros:kinetic-desktop
ENV ROS_DISTRO=kinetic
ENV WDIR=/sim_ws

RUN echo 'hello word'

RUN echo 'source /opt/ros/$ROS_DISTRO/setup.bash' >> ~/.bashrc &&\
    echo 'echo "Sourced $ROS_DISTRO"' >> ~/.bashrc

RUN mkdir -p $WDIR/src
WORKDIR $WDIR
RUN apt-get update &&\
    apt-get install git &&\
    apt-get install -y ros-kinetic-catkin python-catkin-tools 
RUN catkin init
RUN catkin config --extend /opt/ros/kinetic --cmake-args -DCMAKE_BUILD_TYPE=Release
RUN apt-get install -y python-vcstool
WORKDIR $WDIR/src
RUN git clone https://github.com/shilpakancharla/rpg_esim.git
RUN vcs-import < rpg_esim/dependencies.yaml
RUN apt-get install -y ros-kinetic-pcl-ros libproj-dev
RUN apt-get install -y libglfw3 libglfw3-dev
RUN apt-get install -y libglm-dev
RUN apt-get install -y ros-kinetic-hector-trajectory-server
WORKDIR $WDIR/src/ze_oss
RUN touch imp_3rdparty_cuda_toolkit/CATKIN_IGNORE \
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
RUN catkin build esim_ros
RUN echo "source ~/sim_ws/devel/setup.bash" >> ~/setupeventsim.sh
RUN chmod +x ~/setupeventsim.sh
RUN echo 'alias ssim="source ~/setupeventsim.sh"' >> ~/.bashrc
RUN echo 'echo "created alias"' >> ~/.bashrc
RUN echo 'bash'

# sudo docker build -t esim .

# sudo docker images

# sudo docker system prune

# sudo docker run -it -v $(pwd)/shared:/shared esim /bin/bash

# sudo docker ps

# sudo docker exec -it [generated name here] /bin/bash

# sudo rocker --devices /dev/dri/card0 --x11 --volume $(pwd)/shared:/shared -- esim
