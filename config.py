# Labels = ["PM2.5", "PM10", "O3"]
Labels = ["PM2.5"]

DATA_DIR = "./data/"

USE_GPU = True

LEARNING_RATE = 0.005

EARLY_STOP = 30

# the epoch for all local station training
TOTAL_EPOCH = 1

# the epoch on first dataset (2017)
FIRST_EPOCH = 0

# the epoch on second dataset (2018)
SECOND_EPOCH = 2000

BATCH_SIZE = 48 # The number of records within a batch

NUM_STEPS = 48 # the cut off steps for one time back-prop process
# TRAINING_STEPS = 10000 # the epoch number of training

# BATCH_COUNT = 1

AIR_FEATURE_NUM = 1 # 3

LOCAL_WEATHER_FEATURE_NUM = 4

LSTM_HIDDEN_SIZE = 128

ATTENTION_CHOSEN_HIDDEN_SIZE = 32

# WEATHER_STATION_NUM = 18
AIR_STATION_NUM = 35

LOCAL_FC_HIDDEN_SIZE = 64

# the first layer hidden size
HIGH_LEVEL_FC_HIDDEN_SIZE = 256

FUSION_LAYER_HIDDEN_SIZE = 1536

GRID_WEAtHER_STATION_NUM = 30

WEATHER_FEATURE_NUM = 5

ATTENTION_JUDGE_FEATURE_NUM = 6


# LOCATION_FEATURE_NUM = 2

ATTENTION_HIDDEN_SIZE = 128



PREDICT_LAYER_HIDDEN_SIZE = 100


import tensorflow as tf
Initializer = tf.keras.initializers.he_uniform()

UniformInitializer = tf.initializers.variance_scaling(distribution='uniform')
