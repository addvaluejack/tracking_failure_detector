import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

INPUT_SIZE = 400
RNN_HIDDEN = 800
OUTPUT_SIZE = 1
LEARNING_RATE = 0.01

map_fn = tf.map_fn

processed_data_path = "/home/addvaluejack/Repo/tracking_failure_detector/processed_dataset/"
length = np.loadtxt(processed_data_path+"length.txt").astype(int)
nested_overlap = np.loadtxt(processed_data_path+"overlap.txt")
nested_response = np.loadtxt(processed_data_path+"response.txt")
overlap = []
response = []
t_overlap_index = 0
t_response_index = 0
for i in range(np.shape(length)[0]):
  t_overlap = nested_overlap[t_overlap_index:t_overlap_index+length[i]]
  t_response = nested_response[t_response_index:t_response_index+length[i]]
  overlap.append(t_overlap)
  response.append(t_response)
  t_overlap_index += length[i]
  t_response_index += length[i]

def generate_sample(i):
  t_length = length[i]
  t_overlap = overlap[i]
  t_response = response[i]
  x = np.empty((t_length, 1, INPUT_SIZE))
  y = np.empty((t_length, 1, OUTPUT_SIZE))
  for i in range(t_length):
    x[i, 0, :] = t_response[i, :]
    y[i, 0, :] = t_overlap[i]
  c = np.ones((1,RNN_HIDDEN))*t_overlap[0]
  h = np.ones((1,RNN_HIDDEN))*t_overlap[0]
  return x, y, c, h

inputs = tf.placeholder(tf.float32, (None, 1, INPUT_SIZE)) # time, batch_size, input_size
outputs = tf.placeholder(tf.float32, (None, 1, OUTPUT_SIZE))

cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)

initial_c_state = tf.placeholder(tf.float32, (1, RNN_HIDDEN))
initial_h_state = tf.placeholder(tf.float32, (1, RNN_HIDDEN))
initial_state = tf.nn.rnn_cell.LSTMStateTuple(c=initial_c_state, h=initial_h_state)

rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)

final_projection = lambda x: layers.fully_connected(x, num_outputs=OUTPUT_SIZE, activation_fn=None)

predicted_outputs = map_fn(final_projection, rnn_outputs)

error = tf.abs(outputs-predicted_outputs)
error = tf.reduce_mean(error)

train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)

session = tf.Session()
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
session.run(init_op)

for epoch in range(1):
  print("Epoch %d"%(epoch))
  epoch_error = 0
  for i in range(len(overlap)):
    x, y, c, h = generate_sample(i)
    epoch_error += session.run([error, train_fn], {
      inputs: x,
      outputs: y,
      initial_c_state: c,
      initial_h_state: h,
    })[0]
  epoch_error /= len(overlap) 
  print("  Train error: %.5f"%(epoch_error))

save_path = saver.save(session, processed_data_path+"../model/model.ckpt")
print("Model saved in path: %s"%save_path)
