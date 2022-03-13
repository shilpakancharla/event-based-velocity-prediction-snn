import os
import cv2
import math
import h5py
import yaml
import bagpy
import tarfile
import subprocess
import numpy as np
import pandas as pd 
from bagpy import bagreader
from cv_bridge import CvBridge

"""
    Take images in a rosbag file and convert them to .png images.

    @param filename: name of rosbag
    @param topic: name of topic where images are recorded
    @param dest: destination folder where images are stored
"""
def bag_to_png(filename, topic, dest):
    bagfile = filename + ".bag"
    bag = rosbag.Bag(bagfile)
    image_topic = bag.read_messages(topic)
    for k, b in enumerate(image_topic):
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(b.message, b.message.encoding)
        cv_image.astype(np.uint8)
        filepath = dest + '/' + str(b.timestamp) + '.png'
        if not os.path.exists(filepath): # If the image is already in the folder, do not create it again
            cv2.imwrite(filepath, cv_image)
    bag.close()
    print(filename + " images processing complete.")

"""
    Organize event data from rosbag.

    @param event: string of event data from rosbag
    @return x: x-coordinate of an event
    @return y: y-coordinate of an event
    @return secs: seconds timestamp
    @return nsecs: nanoseconds timestamp
    @return polarity: Boolean value; whether there was a change in the pixels
"""
def parse_events(event):
  x = []
  y = []
  secs = []
  nsecs = []
  polarity = []
  tokens = event.splitlines()
  for t in tokens:
    if 'x: ' in t:
      temp_t = t
      if '[' in t: # Handle special case for beginning of array
        temp_t = temp_t.replace('[', '')
      elif 'polarity: False, ' in temp_t:
        temp_t = temp_t.replace('polarity: False, ', '') # Handle special case for polarity and x values being on same line
      elif 'polarity: True, ' in temp_t:
        temp_t = temp_t.replace('polarity: True, ', '')
      temp_t = temp_t.replace('x: ', '')
      x.append(int(temp_t))
    elif 'y: ' in t and t[0] == 'y': # Do not get lines with polarity, only y
      temp_t = t
      temp_t = temp_t.replace('y: ', '')
      y.append(int(temp_t))
    elif 'secs: ' in t and 'n' not in t:
      temp_t = t
      temp_t = temp_t.replace('secs: ', '')
      secs.append(int(temp_t))
    elif 'nsecs: ' in t:
      temp_t = t
      temp_t = temp_t.replace('nsecs: ', '')
      nsecs.append(int(temp_t))
    elif 'polarity: ' in t:
      temp_t = t
      if 'True' in temp_t:
        polarity.append(True)
      elif 'False' in temp_t:
        polarity.append(False)    
  return x, y, secs, nsecs, polarity

"""
    Unpacks the rosbag and returns a .csv file of a particular topic.

    @param rosbag_path: where the rosbag is located
    @param topic: name of topic data to convert to .csv
    @return name of topic
    @return dataframe with topic contents
"""
def unpack_rosbag(rosbag_path, topic_type):
  bag_reader_obj = bagreader(rosbag_path)
  # Output the topic table for reference
  print(bag_reader_obj.topic_table)
  # Decode messages by topic - run if you do not have the csv files saved already
  print("Creating .csv file.")
  topic = bag_reader_obj.message_by_topic(topic_type)
  print("File saved: {}".format(topic))
  # Create dataframe from the topic
  df = pd.read_csv(topic)
  return topic, df

"""
    Converting times within rosbag to human-readable seconds and aligning it with rosbag dataframe.

    @param df: Pandas dataframe of topic information
    @param time_series: column of time in seconds of dataframe
    @param ns_time_series: column of time in nanoseconds of dataframe
"""
def convert_rosbag_timestamps(df, time_series, ns_time_series):
  new_times = []
  init_t = df['header.stamp.secs'][0]
  init_ns = df['header.stamp.nsecs'][0]
  for time_, time_ns in zip(time_series, ns_time_series):
    new_time = (time_ - init_t) + ((time_ns - init_ns) / 1e9)
    new_times.append(new_time)
  # Append the series to the dataframe
  df['New Time'] = np.array(new_times).tolist()

"""
    Calculates the velocity ground truth of rosbag made into a dataframe.
    
    @param rosbag converted to dataframe
    @return dataframe with time and velocity (ground truth)
"""
def create_velocity_gt(df):
  ptr = 0
  time = []
  vel = []
  while ptr < len(df) - 1:
    delta_time = df['New Time'][ptr + 1] - df['New Time'][ptr]
    timestamp = (df['New Time'][ptr + 1] + df['New Time'][ptr]) / 2
    time.append(timestamp)
    delta_x = df['transform.translation.x'][ptr + 1] - df['transform.translation.x'][ptr]
    delta_y = df['transform.translation.y'][ptr + 1] - df['transform.translation.y'][ptr]
    delta_z = df['transform.translation.z'][ptr + 1] - df['transform.translation.z'][ptr]
    velocity = calculate_velocity(delta_time, delta_x, delta_y, delta_z)
    vel.append(velocity)
    ptr += 1
  d = {'Time': time, 'Velocity': vel}
  gt_df = pd.DataFrame(d, columns = ['Time', 'Velocity'])
  return gt_df

"""
    Calculate the velocity vector ground truth.
    
    @param delta_time: difference in time between two timestamps
    @param delta_x: difference in x position between two timestamps
    @param delta_y: difference in y position between two timestamps
    @param delta_z: difference in z position between two timestamps
    @return velocity vector
"""
def calculate_velocity(delta_time, delta_x, delta_y, delta_z):
  x_component = (delta_x / delta_time) ** 2
  y_component = (delta_y / delta_time) ** 2 
  z_component = (delta_z / delta_time) ** 2 
  return math.sqrt(x_component + y_component + z_component)

"""
    Populate the ground truth dataframe with the corresponding events. 
    
    @param gt_df: ground truth dataframe with only time and velocity
    @param event_df: dataframe with (t, x, y, p) event data
    @return ground truth dataframe with associated events
"""
def populate_gt_df(gt_df, event_df):
  ptr = 0
  event_array_to_add = []
  while ptr < len(gt_df) - 1:
    if ptr == 0:
      events = align_events_with_gt(event_df, 0.0, gt_df['Time'][ptr])
    else:
      events = align_events_with_gt(event_df, gt_df['Time'][ptr], gt_df['Time'][ptr + 1])
    event_array_to_add.append(events)
    print("Processed event group " + str(ptr) + " / " + str(len(gt_df)) + ".")
    ptr += 1
  # Concatenate to ground truth dataframe
  gt_df = pd.concat(event_array_to_add, columns = "Events")
  return gt_df

"""
    Capture the corresponding events in an interval. 
    
    @param event_file: dataframe with (t, x, y, p) event data
    @param start_time: start of an event, inclusive
    @param end_time: end of an event, exclusive
    @return array of events associated with a time interval
"""
def align_events_with_gt(event_df, start_time, end_time):
  event_arrays = []
  event_group = event_df.loc[(event_df['t'] >= start_time) & (event_df['t'] < end_time)]
  for e in event_group.iterrows(): # e is a tuple
    event = []
    event.append(e[1][0]) # t
    event.append(e[1][1]) # x
    event.append(e[1][2]) # y
    event.append(e[1][3]) # p
    event_arrays.append(event)
  return event_arrays

"""
    Prints the rosbag information/metadata.
    
    @param bag: path to rosbag file
"""
def get_rosbag_info(bag):
  info_dict = yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', bag], stdout = subprocess.PIPE).communicate()[0])
  print(info_dict)

"""
  Unzip tarfiles from source and extracted files go to a dest folder.

  @param src: location of tarfile
  @param dest: destination of unzipped files
"""
def unzip(src, dest):
  t_file = tarfile.open(src) # Open file
  t_file.extractall(dest) # Extract file  
  t_file.close()

if __name__ == "__main__":
  vicon_motion_bag = "data/2022-03-02-15-37-09_human_movement_with_wand_1.bag"
  event_filepath = "data/out_hw1.txt"

  topic_human_1, df_human_1 = unpack_rosbag(vicon_motion_bag, '/vicon/WAND/WAND')
  convert_rosbag_timestamps(df_human_1, df_human_1['header.stamp.secs'], df_human_1["header.stamp.nsecs"])
  df_human_1_mod = df_human_1[df_human_1['New Time'].between(0, 49)]

  gt_hw1 = create_velocity_gt(df_human_1_mod)
  print("Calculated ground truth velocities. Reading from event files.")

  print("Creating event dataframe.")
  event_hw1_df = pd.read_csv(event_filepath, sep = " ", skiprows = 1, index_col = False, names = ['t', 'x', 'y', 'p'])
  print("Finished creating event dataframe.")

  gt_hw1_with_events = populate_gt_df(gt_hw1, event_hw1_df)
  print("Aligned events with velocity ground truth and timestamps.")

  # Convert ground truth with events dataframe to hdf5 format and save
  print("Saving datframe to .h5 format.")
  hdf5_filepath = "data/hdf5/human_wand_1.h5"
  gt_hw1_with_events.to_hdf(hdf5_filepath, key = 'gt_hw1_with_events', mode = 'w')
  print("Finished saving dataframe to .h5 format.")