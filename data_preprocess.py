import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from scipy import interpolate

#raw_data_path = "/media/addvaluejack/F066EBC466EB8A24/dataset/"
raw_data_path = "/home/addvaluejack/dataset/"
project_path = "/home/addvaluejack/Repo/tracking_failure_detector/"

def resize_and_reshape_response(response_file_name):
  values = np.loadtxt(response_file_name, ndmin=2)
  y = np.arange(0, np.shape(values)[0], 1)
  x = np.arange(0, np.shape(values)[1], 1)
  f = interpolate.interp2d(x, y, values, kind='linear')
  new_y = np.arange(0, (np.shape(values)[0]-1), (np.shape(values)[0]-1)/19)
  if len(new_y) < 20:
    new_y = np.append(new_y, (np.shape(values)[0]-1))
  new_x = np.arange(0, (np.shape(values)[1]-1), (np.shape(values)[1]-1)/19)
  if len(new_x) < 20:
    new_x = np.append(new_x, (np.shape(values)[1]-1))
  new_values = f(new_x, new_y)
  fig, axes = plt.subplots(nrows=1, ncols=2)
  axes[0].imshow(values)
  axes[1].imshow(new_values)
  plt.show()  

  return np.ravel(new_values)

def data_preprocess():
  directory_names = listdir(raw_data_path)
  overlap = np.zeros(0)
  response = np.zeros(0)
  length = np.zeros(0)
  for directory_name in directory_names:
    directory_name = directory_name+"/"
    print(directory_name)
    file_names = listdir(raw_data_path+directory_name)
    t_overlap = np.zeros(0)
    t_response = np.zeros(0)
    t_length = 0
    # read everything from files
    for file_name in file_names:
      if file_name == "tmp":
        continue
      if file_name == "overlap.txt":
        t_overlap = np.loadtxt(raw_data_path+directory_name+file_name, ndmin=1)
      else:
        if np.shape(t_response)[0] == 0:
          t_response = [resize_and_reshape_response(raw_data_path+directory_name+file_name)]
        else:
          t_response = np.append(t_response, [resize_and_reshape_response(raw_data_path+directory_name+file_name)], axis=0)
        t_length = t_length+1
    # truncate the t_response if it is longer than t_overlap
    if t_length > np.shape(t_overlap)[0]:
      t_response = t_response[:np.shape(t_overlap)[0], :]
    # store all temporary values
    overlap = np.append(overlap, t_overlap)
    if np.shape(response)[0] == 0:
      response = t_response
    else:
      response = np.append(response, t_response, axis=0)
    length = np.append(length, np.shape(t_overlap)[0])
  # write everything into new files
  np.savetxt(project_path+"overlap.txt", overlap)
  np.savetxt(project_path+"response.txt", response)
  np.savetxt(project_path+"length.txt", length)

if __name__ == "__main__":
  data_preprocess()
