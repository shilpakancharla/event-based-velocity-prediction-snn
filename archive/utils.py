import os
import cv2
import yaml
import rosbag
import subprocess
import numpy as np
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
    elif 'secs: ' in t and t[0] == 's':
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

if __name__ == '__main__':
    FILENAME = "people_dynamic_air_guitar"
    TOPIC = "/dvs/image_raw"
    DEST = "images/"
    bag_to_png(FILENAME, TOPIC, DEST)
