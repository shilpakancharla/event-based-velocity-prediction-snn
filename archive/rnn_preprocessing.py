import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
  Author: Shilpa Kancharla
  Last updated: April 2, 2022
"""

"""
    Using the sliding window technique to generate image time series data of the event data. 
    The output will be images of 5 events accumulated over time. The window size is 1.
    
    @param event_df: dataframe of event data with velocity measurements
    @param df_name: name of dataframe for saving purposes
    @param dest: where to save the images to
    @param time_steps: determines size of window
    @param step_size: incremental value that window will slide over
"""

def create_sliding_window(event_df, df_name, dest, timesteps = 5, window_size = 1):
  ptr = 0
  event_series = event_df['Events']
  vel_series = event_df[['Vel_x', 'Vel_y', 'Vel_z']]
  vel_xyz = dict()
  while ptr < (len(event_df) - timesteps - window_size):
    # Collect the points to get x, y, polarity values over 5 events
    x, y, pol = [], [], []
    for i in range(timesteps):
      events = ast.literal_eval(event_series[ptr + i])
      for e in events:
        x.append(e[1])
        y.append(e[2])
        if int(e[3]) == 1:
          pol.append('b')
        else:
          pol.append('r')
    
    print(f"Processed {ptr} / {len(event_df)} event points.")
    
    print("Creating accumulated plot.")
    plt.scatter(x, y, color = pol, s = 0.1)
    plt.axis('off')
    # Give the image a name and save it
    img_name = '0000' + str(ptr) + '.png'
    plt.savefig(dest + img_name)
    plt.clf()
    print("Saved image " + img_name + ".")
    # Save the velocity information of the image name in a .csv file
    vel_xyz[img_name] = vel_series.loc[ptr + timesteps - window_size].tolist()
    print(f"Recorded velocity at end of window {ptr}.")
    # Increment the pointer and keep moving through the loop
    ptr += window_size
  
  # Save the dataframe
  print("Saving the dataframe to " + dest + ".")
  vel_image_df = pd.DataFrame.from_dict(vel_xyz, orient = 'index', columns = ['vel_x', 'vel_y', 'vel_z'])
  vel_image_df.to_csv(dest + df_name + '.csv')
  print("Finished saving dataframe to .csv file.")