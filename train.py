import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

processed_data_path = "/home/addvaluejack/Repo/tracking_failure_detector/processed_dataset/"

def generate_batch(batch_size):

INPUT_SIZE = 400
RNN_HIDDEN = 800
OUTPUT_SIZE = 400
LEARNING_RATE = 0.01

inputs = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))
inputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE))

cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)

batch_size = tf
