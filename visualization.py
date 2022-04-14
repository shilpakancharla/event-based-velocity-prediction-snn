import tonic 
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import tonic.transforms as transforms
from mpl_toolkits import mplot3d
from plotly.subplots import make_subplots

"""
  Author: Shilpa Kancharla
  Last updated: April 14, 2022
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
    
def create_acc_plot(x, y, p, title):
  fig = plt.figure(figsize = (15, 15))
  plt.scatter(x, y, color = p, s = 0.1)
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title(title, loc = 'left')
  plt.gca().invert_yaxis()
  plt.show()

def plot_tonic(timestamp, x, y, polarity, denoise_flag):
  events = tonic.io.make_structured_array(x, y, timestamp, polarity)
  sensor_size = (1920, 1080, 2)
  frame_transform = transforms.ToFrame(sensor_size = sensor_size, time_window = 200000)
  frames = frame_transform(events)
  if denoise_flag:
    denoise_transform = transforms.Denoise(filter_time = 10000)
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
  plt.tight_layout()
    
def plot_rmse_history(vel_x_rmse_history, vel_y_rmse_history, vel_z_rmse_history, loss_history):
  fig, axs = plt.subplots(2, 2, figsize = (15, 15))
  fig.suptitle('Vector Component RMSE Loss Histories')
  axs[0][0].set_title('Vel X RMSE')
  axs[0][0].plot(vel_x_rmse_history)
  axs[0][1].set_title('Vel Y RMSE')
  axs[0][1].plot(vel_y_rmse_history)
  axs[1][0].set_title('Vel Z RMSE')
  axs[1][0].plot(vel_z_rmse_history)
  axs[1][1].set_title('Total RMSE')
  axs[1][1].plot(loss_history)
  plt.show()
  
def create_velocity_boxplot(df_list):
  cdf = pd.concat(df_list).drop(['index', 'Events'], axis = 1)
  cdf = cdf.rename(columns={"Vel_x": "x", "Vel_y": "y", "Vel_z": "z"})
  mdf = pd.melt(cdf, id_vars = 'Category', var_name = "Velocity Component", value_name = 'Velocity (m/s)')
  fig = go.Figure()

  fig = make_subplots(
      rows = 3, cols = 2, 
      subplot_titles=("Human with Wand 1", "Human with Wand 2", "Lambda Aerial Robot", 
                      "Box Thrown Around", "Box Sliding on Floor")
  )

  fig = px.box(mdf, x = "Category", y = "Velocity (m/s)", color = "Velocity Component",
               title = "Velocity distributions amongst all data subsets",
               width = 1050, height = 1000, boxmode = "group")

  fig.update_layout(boxgroupgap = 0.2, 
                    boxgap = 0,
                    font_size = 20,
                    legend = dict(yanchor = "top", y = 0.99, xanchor = "left", x = 0.01))

  for t in mdf['Category'].unique():
    fig.add_annotation(x = t,
                       y = mdf[mdf['Category'] == t]['Velocity (m/s)'].max(),
                       text = "No. of points: " + str(int(len(mdf[mdf['Category'] == t]) / 3)),
                       yshift = 25,
                       showarrow = False) 

  fig.show()
  fig.write_image(HOME + "fig1.png", engine = "kaleido")
