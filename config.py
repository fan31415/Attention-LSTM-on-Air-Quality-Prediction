Labels = ["PM2.5", "PM10", "O3"]

DATA_DIR = "/Users/fanyijie/Downloads/5002Data/datapack_5002/"

TEST_DATA_DIR = "/Users/fanyijie/Downloads/5002Data/real_air_5002/"

USE_GPU = False

LEARNING_RATE = 0.005

# the epoch for all local station training
TOTAL_EPOCH = 10

# the epoch for one local station training
FIRST_EPOCH = 1

SECOND_EPOCH = 1

BATCH_SIZE = 48 # The number of records within a batch

NUM_STEPS = 48 # the cut off steps for one time back-prop process
# TRAINING_STEPS = 10000 # the epoch number of training

# BATCH_COUNT = 1

AIR_FEATURE_NUM = 3

LOCAL_WEATHER_FEATURE_NUM = 4

LSTM_HIDDEN_SIZE = 128

ATTENTION_CHOSEN_HIDDEN_SIZE = 32

# WEATHER_STATION_NUM = 18
AIR_STATION_NUM = 35

LOCAL_FC_HIDDEN_SIZE = 100
HIGH_LEVEL_FC_HIDDEN_SIZE = 200

FUSION_LAYER_HIDDEN_SIZE = 100

GRID_WEAtHER_STATION_NUM = 30

WEATHER_FEATURE_NUM = 5

ATTENTION_JUDGE_FEATURE_NUM = 6


# LOCATION_FEATURE_NUM = 2

ATTENTION_HIDDEN_SIZE = 100



PREDICT_LAYER_HIDDEN_SIZE = 100


import tensorflow as tf
Initializer = tf.keras.initializers.he_uniform()

UniformInitializer = tf.initializers.variance_scaling(distribution='uniform')
