import tensorflow as tf
import keras
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from config import *


# Notice this FC networks' parameter will be sharing between local and shared location
def location_FC(inputs, hidden_size = LOCAL_FC_HIDDEN_SIZE):
    with tf.variable_scope("location_fc"):
        # local locations will be changed after a long batch period in datasets
        location_fc_output = tf.contrib.layers.fully_connected(inputs, hidden_size)
    return location_fc_output


# Maybe we can share parameters between local and shared?
def high_level_fc(inputs, hidden_size = HIGH_LEVEL_FC_HIDDEN_SIZE):
    with tf.variable_scope('high_level_fc'):
        layer1 = tf.contrib.layers.fully_connected(inputs, hidden_size)
        outputs = tf.contrib.layers.fully_connected(layer1, hidden_size)
    return outputs

# only one fusion model, no need to share parameters
def fusion_fc_layer(inputs, hidden_size = FUSION_LAYER_HIDDEN_SIZE):
    with tf.variable_scope("fusion_layer_fc"):
        outputs = tf.contrib.layers.fully_connected(inputs, hidden_size)
    return outputs

# inputs shape is [batch_size, hidden_size]
# output shape is [batch_size, label_size]
def predict_layer(inputs, hidden_size = PREDICT_LAYER_HIDDEN_SIZE):
    with tf.variable_scope("predict_layer"):
        inputs_length = inputs.get_shape()[1]
        inputs_size = inputs.get_shape()[0]
        weight = tf.get_variable("vector_weight", [inputs_length, len(Labels)], initializer = Initializer)
        bias = tf.get_variable("bias", [len(Labels), ], initializer = Initializer)
        # outputs = tf.tensordot(inputs, weight, [[1], [0]])
        outputs = tf.matmul(inputs, weight)
        outputs = tf.add(outputs, bias)

    return outputs

# inputs should be a vector from last layer's output
# because of batch training, this actually will be a matrix [batch_size, hidden_size]

# we need cancat inputs and local_inputs, so the actuall input shape is [batch_size, hidden_size1 + hidden_size2]
def get_attention_raw_score(inputs, local_inputs, batch_size, hidden_size1, hidden_size2):
    # params
    # hidden_size1: hidden_size for inputs
    # hidden_size2: hidden_size for local_inputs
    with tf.variable_scope("attention_score"):
        # concat on the hidden_size dimension
        feed_inputs = tf.concat([inputs, local_inputs], axis = 1)

        # Assuming The inputs and local_inputs has the same dimension,
        # so we have hidden_size * 2 as our first dimension of weight1
        weight1 = tf.get_variable("weight1", [hidden_size1 + hidden_size2, ATTENTION_HIDDEN_SIZE], initializer = Initializer)
        bias = tf.get_variable("bias", [ATTENTION_HIDDEN_SIZE], initializer = Initializer)
        layer1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(feed_inputs, weight1), bias))

        # U to project the vector to a value
        u = tf.get_variable("u_vecotr", [ATTENTION_HIDDEN_SIZE, ], initializer = Initializer)
        # Notice bias is always the last dimension value in wieght's shape
        bias2 = tf.get_variable("bias2", [1], initializer = Initializer)
        output = tf.add(tf.tensordot(layer1, u, [[1], [0]]), bias2)

        return output

def get_attention_raw_score_solely(inputs, batch_size, hidden_size):
    # params
    # hidden_size: hidden_size for inputs
    with tf.variable_scope("attention_score"):

        # Assuming The inputs and local_inputs has the same dimension,
        # so we have hidden_size * 2 as our first dimension of weight1
        weight1 = tf.get_variable("weight1", [hidden_size, ATTENTION_HIDDEN_SIZE], initializer = Initializer)
        bias = tf.get_variable("bias", [ATTENTION_HIDDEN_SIZE], initializer = Initializer)
        layer1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, weight1), bias))

        # U to project the vector to a value
        u = tf.get_variable("u_vecotr", [ATTENTION_HIDDEN_SIZE, ], initializer = Initializer)
        # Notice bias is always the last dimension value in wieght's shape
        bias2 = tf.get_variable("bias2", [1], initializer = Initializer)
        output = tf.add(tf.tensordot(layer1, u, [[1], [0]]), bias2)

        return output

# station_inputs shape [staion_count, batch_size, hidden_size]
# local_inputs: [batch_size, hidden_size]
def attention_layer(station_inputs, local_inputs, batch_size, hidden_size1, hidden_size2):
    # params
    # hidden_size1: hidden_size for inputs
    # hidden_size2: hidden_size for local_inputs
    with tf.variable_scope("attention_layer"):

        scores = []
        for idx, staion_input in enumerate(station_inputs):
            with tf.variable_scope("attention_score-" + str(idx)):
                # feed_inputs = tf.concat([staion_input, local_inputs], axis=1)
                # hidden_size is the hidden_size in inputs tensor (the last dimenstion tensor)
                raw_score = get_attention_raw_score(staion_input, local_inputs, batch_size, hidden_size1, hidden_size2)
                scores.append(raw_score)
        # get scores
        scores = tf.nn.softmax(scores, axis=0)
        scores_tile = tf.tile(tf.expand_dims(scores, 2), [1, 1, hidden_size1])
        t_scores = tf.reshape(scores_tile, [scores_tile.get_shape()[0], scores_tile.get_shape()[1] * scores_tile.get_shape()[2]])
        t_input = tf.reshape(station_inputs, [len(station_inputs), station_inputs[0].get_shape()[0] * station_inputs[0].get_shape()[1]])
        t_multi = tf.multiply(t_scores, t_input)
        t_split = tf.split(t_multi, hidden_size1, axis = 1)
        output = tf.reduce_sum(t_split, axis=1)
        # output = tf.reduce_sum(tf.multiply(scores, station_inputs))
        return output


# station_inputs shape [staion_count, batch_size, hidden_size]
# local_inputs: [batch_size, hidden_size]
def attention_layer_uni_input(station_inputs, batch_size, hidden_size):
    # params
    # hidden_size1: hidden_size for inputs
    # hidden_size2: hidden_size for local_inputs
    with tf.variable_scope("attention_layer_uni"):
        scores = []
        for idx, staion_input in enumerate(station_inputs):
            with tf.variable_scope("uni_attention_score-" + str(idx)):
                # hidden_size is the hidden_size in inputs tensor (the last dimenstion tensor)
                raw_score = get_attention_raw_score_solely(staion_input, batch_size, hidden_size)
                scores.append(raw_score)
        # get scores
        scores = tf.nn.softmax(scores, axis=0)
        scores_tile = tf.tile(tf.expand_dims(scores, 2), [1, 1, hidden_size])
        t_scores = tf.reshape(scores_tile, [scores_tile.get_shape()[0],
                                             scores_tile.get_shape()[1] * scores_tile.get_shape()[2]])
        t_input = tf.reshape(station_inputs,
                             [len(station_inputs), station_inputs[0].get_shape()[0] * station_inputs[0].get_shape()[1]])
        t_multi = tf.multiply(t_scores, t_input)
        t_split = tf.split(t_multi, hidden_size, axis=1)
        output = tf.reduce_sum(t_split, axis=1)

        # scores = tf.nn.softmax(scores, axis=0)
        # output = tf.reduce_sum(tf.multiply(scores, station_inputs))
        return output



class LSTM_model(object):
    def __init__(self, inputs, batch_size=BATCH_SIZE, state_size=LSTM_HIDDEN_SIZE, layer_num=2, num_steps=NUM_STEPS,
                 feature_num=AIR_FEATURE_NUM):

        if USE_GPU:
            # time major for cudnn_rnn
            self.inputs = tf.transpose(inputs, [1, 0, 2])

            self.cell = tf.contrib.cudnn_rnn.CudnnLSTM(layer_num, state_size)


            h = tf.get_variable('hidden_h', [layer_num, batch_size, state_size], initializer=UniformInitializer)
            c = tf.get_variable('hidden_c', [layer_num, batch_size, state_size], initializer=UniformInitializer)

            # self.state = self.cell.zero_state(batch_size, dtype=tf.float32)
            self.state = tf.contrib.rnn.LSTMStateTuple(h=h, c=c)

            # self.init_state = tf.get_variable('initial_state',
            #                                   [tf.random_uniform(cell.state_shape(batch_size)[0]),
            #                                    tf.random_uniform(cell.state_shape(batch_size)[1])])

            # self.init_state = (tf.contrib.rnn.LSTMStateTuple(h=self.init_state[1], c=self.init_state[0]),)
            # state = self.init_state


        else:
            self.inputs = inputs
            self.stacked_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(state_size) \
                                                        for _ in range(layer_num)])

            self.state = self.stacked_cell.zero_state(batch_size, dtype=tf.float32)

            # state = self.init_state

            # print(self.inputs.get_shape())


        #         print("lstm ", outputs)
        #     self.final_state = state

    def __call__(self, *args, **kwargs):
        if USE_GPU:
            outputs, self.state = self.cell(self.inputs, initial_state=self.state)
            outputs = tf.transpose(outputs, [1, 0, 2])
            return outputs[:, -1, :]

        else:
            outputs, self.state = tf.nn.dynamic_rnn(self.stacked_cell,
                                               self.inputs, initial_state=self.state, dtype=tf.float32)
            return outputs[:, -1, :]



