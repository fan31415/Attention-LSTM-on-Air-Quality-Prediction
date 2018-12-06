import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
import pickle
import os

from config import *


def load_data(dir = DATA_DIR):
    up_dir = dir
    files = os.listdir(up_dir)
    files.sort()
    data = []
    for file in files:
        if "pkl" not in file:
            continue
        path = os.path.join(up_dir, file)
        with open(path, "rb") as f:
            data.append(pickle.load(f))
    return data, files


# Notice this will also return actuall count
def getBatchCount(total_count, batch_size = BATCH_SIZE):
    take_count = total_count - 1
    sequence_count = (take_count - batch_size + 1)
    batch_count = sequence_count // batch_size
    actuall_count = batch_count * batch_size

    return batch_count, actuall_count

# To make fc output has the same length as lstm output, we need to enlarge the input
# input should be a vector eg. (dis, dir)


def copy_enlarge_vector(inputs, sequence_length):
    sequence= np.tile(inputs, [sequence_length, 1])
    return sequence


def generate_location_batch_data(inputs, batch_count, sequence_length):
    long_vector = copy_enlarge_vector(inputs, sequence_length)
    # Notice that batch count should consider lstm feed data's num_steps's integrity
    # batch_count, actuall_count = getBatchCount(sequence_length)
    actuall_count = batch_count * BATCH_SIZE
    long_vector = long_vector[: actuall_count]
    data = np.split(long_vector, batch_count)
    return data





# inputs will be shape [time_count, feature_number]
def generate_lstm_data(inputs, batch_size = BATCH_SIZE, num_steps = NUM_STEPS, hasLabel = False, stop_before = 0, data_scalar=None):
    # params:
    # stop_before: stop sequence before stop_before days before sequence end
    columns = list(inputs)
    feature_num = len(columns)

    row_count = inputs.shape[0]



    # Notcie weather data longer than air quality two days
    # row_count = row_count - NUM_STEPS
    # inputs = inputs[:-NUM_STEPS]
    if stop_before != 0:
        row_count = row_count - stop_before
        inputs = inputs[:-stop_before]

    take_count = row_count - 1



    # Adjust to make Y one hour later than X inherently

    # The last time data will be one of our predictions, so will not include in our input_X
    # To make input X has proper shape, I start the index from n-1
    input_X = inputs.iloc[- take_count - 1: - 1]

    # we only need the label data to be our labels
    if hasLabel:
        data_Y = inputs[Labels]
    # predict data will be start after lstm feedforwad through the first num steps data
    # Y start will be one hour later than X to be our label
        Y = data_Y.iloc[- take_count:]

    # # Normalized Data
    input_X = data_scalar.transform(input_X)

    #     Arrange X into sequence list
    sequence_X = []

    if hasLabel:
        sequence_Y = []

    sequence_count = row_count
    for i in range(take_count):
        if i > take_count - num_steps:
            sequence_count = i
            break
        sequence_X.append(input_X[i:i + num_steps])
        if hasLabel:
        # Y start already earlier than X one hour, so here we should minus 1, it will get the correspond Y for X
            sequence_Y.append(Y.values[i + num_steps - 1])

    # Make Batch
    # batch_size is the actuall training records' batch size
    batch_count, actuall_count = getBatchCount(sequence_count)
    # print(batch_count)

    # clip the margin data

    sequence_X = sequence_X[-actuall_count:]
    if hasLabel:
        sequence_Y = sequence_Y[-actuall_count:]

    X_batches = np.split(np.array(sequence_X), batch_count)
    if hasLabel:
        Y_batches = np.split(np.array(sequence_Y), batch_count)
    if hasLabel:
        return X_batches, Y_batches
    else:
        return  X_batches


def generate_weather_lstm_data(inputs, batch_size = BATCH_SIZE, num_steps = NUM_STEPS):
    columns = list(inputs)
    feature_num = len(columns)

    row_count = inputs.shape[0]

    # Notcie weather data longer than air quality two days
    row_count = row_count - NUM_STEPS
    inputs = inputs[:-NUM_STEPS]

    take_count = row_count - 1

    # Adjust to make Y one hour later than X inherently

    # The last time data will be one of our predictions, so will not include in our input_X
    # To make input X has proper shape, I start the index from n-1

    input_X = inputs.iloc[- take_count - 1: - 1]

    # Normalized Data
    scaler = StandardScaler()
    input_X = scaler.fit_transform(input_X)

    #     Arrange X into sequence list
    sequence_X = []
    sequence_count = row_count
    for i in range(take_count):
        if i > take_count - num_steps:
            sequence_count = i
            break
        sequence_X.append(input_X[i:i + num_steps])

    # Make Batch
    # batch_size is the actuall training records' batch size
    batch_count, actuall_count = getBatchCount(sequence_count)

    # clip the margin data

    sequence_X = sequence_X[-actuall_count:]

    X_batches = np.split(np.array(sequence_X), batch_count)

    return X_batches


def generate_locations_data(inputs, batch_size = BATCH_SIZE, num_steps = NUM_STEPS, data_scalar=StandardScaler()):
    columns = list(inputs)
    feature_num = len(columns)

    row_count = inputs.shape[0]


    take_count = row_count - 1


    # Adjust to make Y one hour later than X inherently

    # The last time data will be one of our predictions, so will not include in our input_X
    # To make input X has proper shape, I start the index from n-1

    input_X = inputs.iloc[- take_count - 1: - 1]

    # # Normalized Data
    input_X = data_scalar.transform(input_X)

    #     Arrange X into sequence list
    sequence_X = []
    sequence_count = row_count
    for i in range(take_count):
        if i > take_count - num_steps:
            sequence_count = i
            break
        sequence_X.append(input_X[i])

    # Make Batch
    # batch_size is the actuall training records' batch size
    batch_count, actuall_count = getBatchCount(sequence_count)

    # clip the margin data

    sequence_X = sequence_X[-actuall_count:]

    X_batches = np.split(np.array(sequence_X), batch_count)

    return X_batches


# pre-processor for single model
class PreProcessor:
    def __init__(self):
        # each model has a independent scalar
        self.local_air_scalar = StandardScaler()
        self.local_weather_scalar = StandardScaler()
        self.neighbor_air_scalar = StandardScaler()
        self.neighbor_weather_scalar = StandardScaler()
        self.neighbor_location_scalar = StandardScaler()



