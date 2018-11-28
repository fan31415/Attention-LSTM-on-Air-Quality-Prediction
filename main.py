import tensorflow as tf
import keras
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import os
import sys


# local lib

from model import *

from dataHelper import *

from config import *

# log = open("myprog.log","w")
# sys.stdout = log
# error_log = open("myerror.log", "w")
# sys.stderr = error_log


trainDataByStation = load_data()

print(len(trainDataByStation))

air_qualities = []

air_station_locations = []

grid_location_by_station = []

# each of its item has 30 grid weather station data
grid_weathers_around_station = []


for staion_train_data in trainDataByStation:
    air_qualities.append(staion_train_data[0])
    air_station_locations.append(staion_train_data[1])
    grid_location_by_station.append(staion_train_data[2])
    grid_weathers_around_station.append(staion_train_data[3])


# Normalize locations data

for idx, locations in enumerate(air_station_locations):
    scaler = pre_processor.air_location_scalar[idx]
    air_station_locations[idx] = scaler.fit_transform(locations[['dist', 'direction']])


for idx, locations in enumerate(grid_location_by_station):
    scaler = pre_processor.weather_location_scalar[idx]
    grid_location_by_station[idx] = scaler.fit_transform(locations[['dist', 'direction']])


# generate air lstm data
# air data is normalized in a uniform scale
air_lstm_datas= [None] * AIR_STATION_NUM
Y = [None] * AIR_STATION_NUM
# fit normalizer
all_air_data = None
for i in range(AIR_STATION_NUM):
    if all_air_data is None:
        all_air_data = air_qualities[i]
    else:
        all_air_data = pd.concat([all_air_data, air_qualities[i]], axis=0)
pre_processor.fit_transform_air_scalar(all_air_data, only_fit=True)
# generate lstm input data
for i in range(AIR_STATION_NUM):
    air_lstm_datas[i], Y[i] = generate_air_quality_lstm_data(air_qualities[i], BATCH_SIZE, NUM_STEPS)


# generate weather lstm data
weather_lstm_datas = [[None] * GRID_WEAtHER_STATION_NUM] * AIR_STATION_NUM
for i in range(AIR_STATION_NUM):
    all_weather_data = None
    for j in range(GRID_WEAtHER_STATION_NUM):
        if all_weather_data is None:
            all_weather_data = grid_weathers_around_station[i][j]
        else:
            all_weather_data = pd.concat([all_weather_data, grid_weathers_around_station[i][j]], axis=0)
    all_weather_data_normal = pre_processor.fit_transform_weather_scalar(i, all_weather_data)
    offset = 0
    for j in range(GRID_WEAtHER_STATION_NUM):
        weather_data_size_j = grid_weathers_around_station[i][j].shape[0]
        normalized_input = all_weather_data_normal[offset:weather_data_size_j+offset]
        offset += weather_data_size_j
        weather_lstm_datas[i][j] = generate_weather_lstm_data(normalized_input, BATCH_SIZE, NUM_STEPS)


# define BATCH_COUNT in global config here
# the locaton batch rely on this too
BATCH_COUNT = len(air_lstm_datas[0])
print("BATCH_C ", BATCH_COUNT)

train_length = len(air_qualities[0])-1



# generate data for air station location
air_locations_shared_fc_feed = [None] * AIR_STATION_NUM
for i in range(AIR_STATION_NUM):
    air_locations_shared_fc_feed[i] = []
    for pair in np.array(air_station_locations[i]):
        # feed_data = tf.concat([copy_enlarge_vector(pair[0], train_length), copy_enlarge_vector(pair[1], train_length)],
        #                       axis=1)
        # Notice this function has consider real count difference, so we need to plus 1 here
        air_locations_shared_fc_feed[i].append(generate_location_batch_data(pair, BATCH_COUNT, train_length+1))


# generate data for grid weather station location
weather_locations_shared_fc_feed = [None] * AIR_STATION_NUM
for i in range(AIR_STATION_NUM):
    weather_locations_shared_fc_feed[i] = []
    for pair in np.array(grid_location_by_station[i]):
        feed_data = generate_location_batch_data(pair, BATCH_COUNT, train_length+1)
        weather_locations_shared_fc_feed[i].append(feed_data)


# print(np.shape(weather_locations_shared_fc_feed))


# print(air_locations_shared_fc_feed[0][0])
# print(air_locations_shared_fc_feed[0][1])
# print(weather_locations_shared_fc_feed[0][0])
# print(weather_locations_shared_fc_feed[0][1])
#
# print(air_lstm_datas[0])
# print(air_lstm_datas[0][0])
# exit()
class FModel(object):
    def __init__(self, is_training, sess=None):
        self.global_bp_cnt = 0  # a counter to record the times running BackPropagation
        self.sess = sess
        self.batch_size = BATCH_SIZE
        self.num_steps = NUM_STEPS

        # there seems can use tf.int32?
        self.targets = tf.placeholder(tf.float32, [BATCH_SIZE, len(Labels)])

        #         self.air_lstm_inputs = tf.placeholder(tf.float32, [batch_size])
        # self.temp_targets = tf.placeholder(tf.float32, [batch_size])



        # Build Model

        self.local_air_lstm_inputs = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_STEPS, AIR_FEATURE_NUM])

        # self.air_fc_inputs = tf.placeholder(tf.float32, [AIR_STATION_NUM, BATCH_SIZE, LOCATION_FEATURE_NUM])
        self.air_fc_inputs = tf.placeholder(tf.float32, [AIR_STATION_NUM, BATCH_SIZE, LOCATION_FEATURE_NUM])
        # for i in range(AIR_STATION_NUM):
        #     self.air_fc_inputs.append()

        # self.weather_fc_inputs = tf.placeholder(tf.float32,
        #                                         [GRID_WEAtHER_STATION_NUM, BATCH_SIZE, LOCATION_FEATURE_NUM])
        self.weather_fc_inputs = tf.placeholder(tf.float32, [GRID_WEAtHER_STATION_NUM, BATCH_SIZE, LOCATION_FEATURE_NUM])
        # for i in range(GRID_WEAtHER_STATION_NUM):
        #     self.weather_fc_inputs.append()

        self.air_lstm_inputs = tf.placeholder(tf.float32, [AIR_STATION_NUM, BATCH_SIZE, NUM_STEPS, AIR_FEATURE_NUM])

        self.weather_lstm_inputs = tf.placeholder(tf.float32, [GRID_WEAtHER_STATION_NUM, BATCH_SIZE, NUM_STEPS, WEATHER_FEATURE_NUM])


        with tf.variable_scope("loccal_air_lstm", reuse=tf.AUTO_REUSE) as scope:
            self.local_air_lstm = LSTM_model(self.local_air_lstm_inputs, feature_num=AIR_FEATURE_NUM)

        # share parametes between them
        with tf.variable_scope("air_lstm", reuse=tf.AUTO_REUSE):
            self.air_lstms = []
            for i in range(AIR_STATION_NUM):
                self.air_lstms.append(LSTM_model(self.air_lstm_inputs[i], feature_num=AIR_FEATURE_NUM))
        # share parametes between them
        with tf.variable_scope("weather_lstm", reuse=tf.AUTO_REUSE):
            self.weather_lstms = []
            for i in range(GRID_WEAtHER_STATION_NUM):
                self.weather_lstms.append(LSTM_model(self.weather_lstm_inputs[i], feature_num=WEATHER_FEATURE_NUM))
        # share parametes between them
        with tf.variable_scope("location_fc", reuse=tf.AUTO_REUSE):
            # with tf.variable_scope("local"):
            #     self.local_fc_output = location_FC(self.local_fc_inputs)
            with tf.variable_scope("air"):
                self.air_location_fc_outputs = []
                for i in range(AIR_STATION_NUM):
                    self.air_location_fc_outputs.append(location_FC(self.air_fc_inputs[i]))
            with tf.variable_scope("weather"):
                self.weather_location_fc_outputs = []
                for i in range(GRID_WEAtHER_STATION_NUM):
                    self.weather_location_fc_outputs.append(location_FC(self.weather_fc_inputs[i]))

        # self.local_high_level_fc_output =  high_level_fc([self.local_fc_output, self.local_air_lstm.output])
        # Do we still need this layer?
        # self.local_high_level_fc_output = high_level_fc(self.local_air_lstm.output)
        with tf.variable_scope("air_high_level_fc", reuse = tf.AUTO_REUSE):
            self.air_high_level_fc_outputs = []
            for i in range(AIR_STATION_NUM):
                with tf.variable_scope("air_higle_level_fc"):
                    air_high_level_fc_inputs = tf.concat([self.air_location_fc_outputs[i],
                                                          self.air_lstms[i].output], axis=1)
                    self.air_high_level_fc_outputs.append(high_level_fc(air_high_level_fc_inputs))

        with tf.variable_scope("weather_high_level_fc", reuse = tf.AUTO_REUSE):
            self.weather_high_level_fc_outputs = []
            for i in range(GRID_WEAtHER_STATION_NUM):
                with tf.variable_scope("weather_high_level_fc"):
                    weather_high_level_fc_inputs = tf.concat([self.weather_location_fc_outputs[i],
                                                              self.weather_lstms[i].output], axis=1)
                    self.weather_high_level_fc_outputs.append(high_level_fc(weather_high_level_fc_inputs))

        # Is use local lstm to feed directly without fc OK?
        with tf.variable_scope("air_attention"):
            air_station_attention_output = attention_layer(self.air_high_level_fc_outputs, self.local_air_lstm.output,
                                                           BATCH_SIZE, HIGH_LEVEL_FC_HIDDEN_SIZE, LSTM_HIDDEN_SIZE)
        # Do we need local weather?
        # If we use local weather, we seems to find the relationship between local weather and other station weathers
        # Is above true under the condition that we have all the near weather?
        # Notice that, there have difference between input local and other weather here or just include local
        # weather in other weathers. Because if we input here, we will concat every other weather with local to
        # calucate each score, however if we did not input in this function but include local weather in all weather
        # outputs, every single input in score calculation is only one station weather.
        # In my view, just use all the near weather is weather
        with tf.variable_scope("weather_attention"):
            weather_station_attention_output = attention_layer_uni_input(self.weather_high_level_fc_outputs,
                                                                         BATCH_SIZE, HIGH_LEVEL_FC_HIDDEN_SIZE)

        t_air = tf.transpose(air_station_attention_output)
        t_weather = tf.transpose(weather_station_attention_output)
        fusion_fc_inputs = tf.concat([t_air, t_weather, self.local_air_lstm.output], axis = 1)
        fusion_fc_outputs =  fusion_fc_layer(fusion_fc_inputs)

        self.results = predict_layer(fusion_fc_outputs)

        # loss = tf.losses.mean_squared_error(labels=self.targets, predictions=self.results)

        self.losses = tf.square(tf.subtract(self.targets, self.results))

        # average cost
        self.cost = tf.div(tf.reduce_sum(self.losses), BATCH_SIZE)
        # self.train_op = tf.contrib.layers.optimize_loss(
        #     loss, tf.train.get_global_step(), optimizer="Adam", learning_rate=0.01)
        tf.summary.scalar('train loss', self.cost)
        self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
        self.merged_summary = tf.summary.merge_all()

        if self.sess is not None:
            self.train_writer = tf.summary.FileWriter('./logs/train/', sess.graph)
            self.test_writer = tf.summary.FileWriter('./logs/test')


def run_epoch(session, model, batch_count, train_op, output_log, step,
              air_locations_feed, weather_location_feed, air_qualities_feed,
              weather_feed, targets, station_idx, big_iter=0):
    # big_iter is marked as the current loop idx of the big loop
    total_costs = 0.0
    iters = 0

    for batch_idx in range(batch_count):

        # print(np.shape(air_locations_feed))
        # print(np.shape(air_qualities_feed))
        # print(np.shape(weather_location_feed))
        # print(np.shape(weather_feed))
        # print(np.shape(targets))
        # add global_bp_cnt
        model.global_bp_cnt += 1
        # record running metadata when firstly run a batch
        if iters == 0:
            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()
            cost, _, output, losses, summary = session.run([model.cost, train_op, model.results, model.losses, model.merged_summary],
                                       {model.local_air_lstm_inputs: air_qualities_feed[batch_idx][station_idx],
                                        # the input below is all list of batch data
                                        model.air_fc_inputs: air_locations_feed[batch_idx],
                                        model.air_lstm_inputs: air_qualities_feed[batch_idx],
                                        model.weather_fc_inputs: weather_location_feed[batch_idx],
                                        model.weather_lstm_inputs: weather_feed[batch_idx],
                                        model.targets: targets[batch_idx]})
        #         cost, _ = session.run([model.cost, train_op], {model.targets: y[0][batch_idx],
        #
        #                                                model.temp_targets: train_Y})
        else:
            cost, _, output, losses, summary = session.run(
                [model.cost, train_op, model.results, model.losses, model.merged_summary],
                {model.local_air_lstm_inputs: air_qualities_feed[batch_idx][station_idx],
                 # the input below is all list of batch data
                 model.air_fc_inputs: air_locations_feed[batch_idx],
                 model.air_lstm_inputs: air_qualities_feed[batch_idx],
                 model.weather_fc_inputs: weather_location_feed[batch_idx],
                 model.weather_lstm_inputs: weather_feed[batch_idx],
                 model.targets: targets[batch_idx]})

        model.train_writer.add_summary(summary, model.global_bp_cnt)
        iters += 1
        print(iters)
        print(cost)
        # print(losses)
        # print(output)
        # print(targets[batch_idx])

        total_costs += cost

    step += 1
    return step, total_costs



def main():
    # The only place define batch count, this should be edit accordingly by BATCH_SIZE
    # define BATCH_COUNT HERE

    sess = tf.InteractiveSession()

    # [AIR_STATION_NUM (total data by station), DATA_NUM (in one total train), BATCH_COUNT, BATCH_SIZE, FEATURE_NUM)]
    # swap axes so that we can first choose data by trained local station, then get data by bacth_idx
    np_air_locations_shared_fc_feed = np.swapaxes(np.array(air_locations_shared_fc_feed), 1, 2)
    np_weather_locations_shared_fc_feed = np.swapaxes(np.array(weather_locations_shared_fc_feed), 1, 2)
    np_air_lstm_datas = np.swapaxes(np.array(air_lstm_datas), 0, 1)
    np_weather_lstm_datas = np.swapaxes(np.array(weather_lstm_datas), 1, 2)
    np_Y = np.array(Y)
    print(np.shape(np_air_locations_shared_fc_feed))
    print(np.shape(np_weather_locations_shared_fc_feed))
    print(np.shape(np_air_lstm_datas))
    print(np.shape(np_weather_lstm_datas))
    print(np.shape(np_Y))

    train_model = FModel(True, sess=sess)

    saver = tf.train.Saver()


    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        step = 0

        for epoch_idx in range(TOTAL_EPOCH):
            print("total epoch :", epoch_idx)
            # For every station, run NUM_EPOCH baches
            for station_idx in range(AIR_STATION_NUM):
                # data are all in batches, the first dimension is batch size
                air_locations_feed = np_air_locations_shared_fc_feed[station_idx]
                weather_location_feed = np_weather_locations_shared_fc_feed[station_idx]
                # pass a array with all air lstm datas
                # the only one we can reuse
                air_qualities_feed = np_air_lstm_datas
                # weather_feed: array element length is grid_weather_station_num
                # we have to choose this data by station id, because the around weather data changed with local station
                weather_feed = np_weather_lstm_datas[station_idx]
                targets = np_Y[station_idx]

                print("targets shape")
                print(np.shape(targets))

                for i in range(NUM_EPOCH):
                    print("In iteration ", i)

                    step, total_cost = run_epoch(sess, train_model, BATCH_COUNT, train_model.train_op, True, step,
                                                 air_locations_feed, weather_location_feed, air_qualities_feed,
                                                 weather_feed, targets, station_idx, big_iter=i)
                    print("Step: ", step)
                    print(total_cost/BATCH_COUNT)
                    saver.save(sess, './my_model.model', global_step=step)


main()