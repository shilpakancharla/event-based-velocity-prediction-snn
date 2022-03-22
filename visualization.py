import tonic 
import matplotlib.pyplot as plt 
import tonic.transforms as transforms
from mpl_toolkits import mplot3d

"""
  Author: Shilpa Kancharla
  Last updated: March 21, 2022
"""

def parse_lambda_data(event_file, tonic_flag):
    read_file = open(event_file, 'br')
    next(read_file) # Skip line with sensor information
    timestamp = []
    x = []
    y = []
    polarity = []
    number_of_lines = 100000
    for i in range(number_of_lines):
    line = read_file.readline()
    tokens = line.split(b' ')
    timestamp.append(float(tokens[0]) * 1e3) # Milliseconds
    x.append(int(tokens[1]))
    y.append(int(tokens[2]))
    if int(tokens[3]) == 1:
        if tonic_flag:
            polarity.append(True)
        else:
            polarity.append('b')
    else:
        if tonic_flag:
            polarity.append(False)
        else:
            polarity.append('r')
    return timestamp, x, y, polarity

def create_event_plot(timestamp, x, y, polarity, title):
    fig = plt.figure(figsize = (20, 20))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter3D(timestamp, x, y, color = polarity, s = 0.1)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_title(title, loc = 'left')
    plt.show()

def plot_tonic(timestamp, x, y, polarity, denoise_flag):
    events = tonic.io.make_structured_array(x, y, timestamp, polarity)
    sensor_size = (1920, 1080, 2)
    frame_transform = transforms.ToFrame(sensor_size = sensor_size, time_window = 200000)
    frames = frame_transform(events)
    if denoise_flag:
        denoise_transform = tonic.transforms.Denoise(filter_time = 10000)
        events_denoised = denoise_transform(events)
        frames_denoised = frame_transform(events_denoised)
        return frames_denoised 
    else:
        return frames

def plot_frames(frames):
    fig, axes = plt.subplots(len(frames), 1, figsize = (20, 15))
    for axis, frame in zip(axes, frames):
        axis.imshow(frame[1] - frame[0])
        axis.axis("off")
    #plt.tight_layout()
