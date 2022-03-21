import os
import gc
import ast
#import cv2
import math
#import yaml
#import bagpy
#import rosbag
#import tarfile
#import subprocess
import numpy as np
import pandas as pd 
from tqdm import tqdm
#from bagpy import bagreader
#from cv_bridge import CvBridge

"""
  Author: Shilpa Kancharla
  Last updated: March 21, 2022
"""

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
  x = []
  y = []
  z = []
  while ptr < len(df) - 1:
    delta_time = df['New Time'][ptr + 1] - df['New Time'][ptr]
    timestamp = (df['New Time'][ptr + 1] + df['New Time'][ptr]) / 2
    time.append(timestamp)
    delta_x = df['transform.translation.x'][ptr + 1] - df['transform.translation.x'][ptr]
    delta_y = df['transform.translation.y'][ptr + 1] - df['transform.translation.y'][ptr]
    delta_z = df['transform.translation.z'][ptr + 1] - df['transform.translation.z'][ptr]
    result = calculate_velocity(delta_time, delta_x, delta_y, delta_z)
    x.append(result[0])
    y.append(result[1])
    z.append(result[2])
    ptr += 1
  d = {'Time': time, 'Vel_x': x, 'Vel_y': y, 'Vel_z': z}
  gt_df = pd.DataFrame.from_dict(d)
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
  result = []
  x_component = (delta_x / delta_time)
  y_component = (delta_y / delta_time)
  z_component = (delta_z / delta_time)
  result.append(x_component)
  result.append(y_component)
  result.append(z_component)
  return result

"""
    Populate the ground truth dataframe with the corresponding events. 
    
    @param gt_df: ground truth dataframe with only time and velocity
    @param event_df: dataframe with (t, x, y, p) event data
    @param start_flag: Boolean that will tell us if we are dealing with the very start of the dataframe
          and need to process from time 0.0
    @return ground truth dataframe with associated events
"""
def populate_gt_df(gt_df, event_df, start_flag):
  ptr = 0
  event_array_to_add = []
  while ptr < len(gt_df) - 1:
    if start_flag:
      if ptr == 0:
        events = align_events_with_gt(event_df, 0.0, gt_df['Time'][ptr])
      else:
        events = align_events_with_gt(event_df, gt_df['Time'][ptr], gt_df['Time'][ptr + 1])
    else:
      events = align_events_with_gt(event_df, gt_df['Time'][ptr], gt_df['Time'][ptr + 1])
    event_array_to_add.append(events) # List of group of events, or list of lists of lists
    print("Processed event group " + str(ptr) + " / " + str(len(gt_df)) + ".")
    ptr += 1
  # Concatenate to ground truth dataframe
  gt_df['Events'] = pd.Series(event_array_to_add)
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
  event_group = event_df.loc[event_df['t'].between(start_time, end_time)]
  for e in tqdm(event_group.iterrows()): # e is a tuple
    event = [] # Represents a single event, list of t, x, y, p
    event.append(e[1][0]) # t
    event.append(e[1][1]) # x
    event.append(e[1][2]) # y
    event.append(e[1][3]) # p
    event_arrays.append(event) # All events within this time range, list of lists
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

"""
  Read only the relevant section of a large .txt file.

  @param textfile_path: location of large text file
  @param start_time: start time of section of file, inclusive
  @param stop_time: end time of section of file, exclusive
  @return event dataframe of a time period
"""
def read_textfile(textfile_path, start_time, stop_time):
  t = []
  x = []
  y = []
  p = []
  with open(textfile_path) as f:
    next(f) # Skip the first line because it has sensor information
    for i, line in enumerate(f):
      tokens = line.split(' ') # Split the line into t x y p
      if start_time <= float(tokens[0]) < stop_time:
        t.append(float(tokens[0])) 
        x.append(int(tokens[1]))
        y.append(int(tokens[2]))
        p.append(int(tokens[3])) 
      elif float(tokens[0]) >= stop_time:
        break
      else:
        pass
  d = {'t': t, 'x': x, 'y': y, 'p': p}
  event_df = pd.DataFrame.from_dict(d)
  return event_df

"""
  Remove event data points that are empty (look like [[]]).
  
  @param csv_file: .csv file contain preprocessed event data
  @return dataframe with dropped contents
"""
def remove_rows_with_empty_events(csv_file):
  df = pd.read_csv(csv_file, index_col = 0)
  print("Length of original dataframe: ", len(df))
  temp_df = df.dropna(how = 'any')
  print("Length of dataframe after dropping NA values: ", len(temp_df))
  # Drop the time and other extra columns column
  temp_df = temp_df.drop('Time', axis = 1)
  # Drop the row if there are no events - convert string representation to list
  for index, e in temp_df.iterrows():
    list_rep_e = convert_to_list(e['Events']) # Event data here in tuple
    if len(list_rep_e) < 1:
      temp_df.drop(index, inplace = True) # Drop by index
  print("Length of dataframe after dropping empty events: ", len(temp_df))
  print("Shape of new dataframe: ", temp_df.shape)
  return temp_df.reset_index()

"""
  Convert a string representation of a list into a list.

  @param string_representation: event string representation read from preprocessed dataframe
  @return list representation of event data
"""
def convert_to_list(string_representation):
  list_ = ast.literal_eval(string_representation)
  return list_

if __name__ == "__main__": 
  gc.collect()

  event_filepath = "data/out_hw2.txt"
  df_hw2 = pd.read_csv('data/vicon-WAND-WAND_2.csv')
  
  convert_rosbag_timestamps(df_hw2, df_hw2['header.stamp.secs'], df_hw2["header.stamp.nsecs"])
  df_hw2_mod = df_hw2[df_hw2['New Time'].between(13, 117)] # Calibrated time
  df_hw2_mod = df_hw2_mod.reset_index() # Reset the index when we aren't starting at values of 0
  
  # Subtract the first time measurement from all the values 'New Time' (recalibration)
  df_hw2_mod_sub = df_hw2_mod # Make a copy of the old dataframe
  df_hw2_mod_sub['New Time'] = df_hw2_mod['New Time'] - df_hw2_mod['New Time'].iloc[0] 
  df_hw2_mod_sub = df_hw2_mod_sub[df_hw2_mod_sub['New Time'].between(60, 75)]
  df_hw2_mod_sub = df_hw2_mod_sub.reset_index()
  print(df_hw2_mod_sub.head())
  print(df_hw2_mod_sub.tail())
  gt_hw2 = create_velocity_gt(df_hw2_mod_sub)
  print(gt_hw2.head())
  print("Calculated ground truth velocities. Reading from event files.")

  print("Creating event dataframe.")
  #event_hw2_df = pd.read_csv(event_filepath, sep = " ", skiprows = 1, index_col = False, names = ['t', 'x', 'y', 'p'])
  event_hw2_df = read_textfile(event_filepath, 60, 75)
  print("Event dataframe start:")
  print(event_hw2_df.head())
  print("Event dataframe end:")
  print(event_hw2_df.tail())
  print("Finished creating event dataframe.")

  gt_hw2_with_events = populate_gt_df(gt_hw2, event_hw2_df, False)
  print("Aligned events with velocity ground truth and timestamps.")
  
  # Save to .csv file
  print("Saving dataframe to .csv format to access later.")
  gt_hw2_with_events.to_csv('data/processed/GT_HW2_4.csv')

  # df = pd.read_csv('data/processed/GT_HW2_1.csv')
  # print(len(df))
  # print(df.head())
  # print(df.tail(10))