import tensorflow as tf
import keras
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
import pickle
import os
import sys

from copy import deepcopy


# local lib

from model import *

from dataHelper import *

from config import *

# log = open("myprog.log","w")
# sys.stdout = log
# error_log = open("myerror.log", "w")
# sys.stderr = error_log

trainDataByStation, _ = load_data()

testDataByStation, files = load_data(dir = TEST_DATA_DIR)
testDataDict = {}

for idx, file in enumerate(files):
    name = file.split('.')[0].split('_')[2:4]
    name = "_".join(name)
    testDataDict[name] = testDataByStation[idx]

# do this due to data generalization not consist here
# for i in range(AIR_STATION_NUM):
#     trainDataByStation[i][0] = trainDataByStation[i][0][:-48]
#     for j in range(len(trainDataByStation[i][2])):
#         trainDataByStation[i][2][j] = trainDataByStation[i][2][j][:-48]


print(len(trainDataByStation))


local_air = []

local_weather = []

global_air = []

global_weather = []

global_locations = []

class Dataset:
    local_air = None
    local_weather = None
    global_weather = None
    global_air = None
    global_locations = None
    local_air_lstm_datas = None
    local_weather_lstm_datas = None
    global_air_lstm_datas = None
    global_weather_lstm_datas = None
    global_locations_datas = None
    Y = None

    test_local_weather = None
    test_global_weather = None
    test_global_locations = None
    test_local_air = None
    test_global_air = None
    test_append_local_weather = None
    test_append_global_weather = None
    test_append_location = None


dataset = Dataset()






datasets = []


for staion_train_data in trainDataByStation:
    local_air.append(staion_train_data[0])
    local_weather.append(staion_train_data[1])
    # array inside
    global_air.append(staion_train_data[2])
    global_weather.append(staion_train_data[3])

    global_locations.append(staion_train_data[4])

dataset.local_air = local_air
dataset.local_weather = local_weather
dataset.global_weather = global_weather
dataset.global_air = global_air
dataset.global_locations = global_locations


# Workround: do this due to data generalization not consist here
for i in range(AIR_STATION_NUM):
    dataset.local_air[i] = dataset.local_air[i]["2017-03-01 04:00:00":"2017-04-30 22:00:00"]
    dataset.local_weather[i] = dataset.local_weather[i]["2017-03-01 05:00:00":"2017-04-30 23:00:00"]
    for j in range(len(dataset.global_weather[i])):
        dataset.global_air[i][j] = dataset.global_air[i][j]["2017-03-01 04:00:00":"2017-04-30 22:00:00"]
        dataset.global_weather[i][j] = dataset.global_weather[i][j]["2017-03-01 04:00:00":"2017-04-30 22:00:00"]
        dataset.global_locations[i][j] = dataset.global_locations[i][j]["2017-03-01 04:00:00":"2017-04-30 22:00:00"]


datasets.append(deepcopy(dataset))

# init again

dataset = Dataset()

local_air = []

local_weather = []

global_air = []

global_weather = []

global_locations = []

for staion_train_data in trainDataByStation:
    local_air.append(staion_train_data[5])
    local_weather.append(staion_train_data[6])
    # array inside
    global_air.append(staion_train_data[7])
    global_weather.append(staion_train_data[8])

    global_locations.append(staion_train_data[9])



dataset.local_air = deepcopy(local_air)
dataset.local_weather = deepcopy(local_weather)
dataset.global_weather = deepcopy(global_weather)
dataset.global_air = deepcopy(global_air)
dataset.global_locations = deepcopy(global_locations)





# For predict
dataset.test_local_weather = []
dataset.test_global_locations = []
dataset.test_global_weather = []
dataset.test_local_air = []
dataset.test_global_air = []
dataset.test_append_global_weather = []
dataset.test_append_local_weather = []
dataset.test_append_location = []

# Workround: do this due to data generalization not consist here
for i in range(AIR_STATION_NUM):
    dataset.test_local_weather.append(local_weather[i]["2018-04-29 01:00:00":"2018-05-01 00:00:00"])

    dataset.test_local_air.append(local_air[i]["2018-04-29 00:00:00":"2018-04-30 23:00:00"])

    dataset.test_append_local_weather.append(local_weather[i]["2018-05-01 01:00:00":"2018-05-02 23:00:00"])
    dataset.test_append_local_weather[i] = dataset.test_append_local_weather[i] \
        .append(local_weather[i].iloc[-1])


    dataset.local_air[i] = dataset.local_air[i]["2018-03-01 04:00:00":"2018-04-30 23:00:00"]
    dataset.local_weather[i] = dataset.local_weather[i]["2018-03-01 05:00:00":"2018-05-01 00:00:00"]

    num = len(dataset.global_weather[i])
    dataset.test_global_weather.append([])
    dataset.test_global_locations.append([])
    dataset.test_global_air.append([])

    dataset.test_append_global_weather.append([])
    dataset.test_append_location.append([])



    for j in range(num):
        dataset.global_air[i][j] = dataset.global_air[i][j]["2018-03-01 04:00:00":"2018-04-30 23:00:00"]
        dataset.global_weather[i][j] = dataset.global_weather[i][j]["2018-03-01 04:00:00":"2018-04-30 23:00:00"]
        dataset.global_locations[i][j] = dataset.global_locations[i][j]["2018-03-01 04:00:00":"2018-04-30 23:00:00"]

        dataset.test_global_weather[i].append(global_weather[i][j]["2018-04-29 00:00:00":"2018-04-30 23:00:00"])
        dataset.test_global_locations[i].append(global_locations[i][j]["2018-04-29 00:00:00":"2018-04-30 23:00:00"])
        dataset.test_global_air[i].append(global_air[i][j]["2018-04-29 00:00:00":"2018-04-30 23:00:00"])

        dataset.test_append_location[i].append(global_locations[i][j]["2018-05-01 00:00:00":"2018-05-02 23:00:00"])
        dataset.test_append_global_weather[i].append(global_weather[i][j]["2018-05-01 00:00:00":"2018-05-02 23:00:00"])







datasets.append(deepcopy(dataset))


indexes = []

for i in range(AIR_STATION_NUM):
    num = len(dataset.global_weather[i])
    indexes.append(list(datasets[1].global_weather[i].index))

#
# print(len(datasets[0].global_locations[0]))
#
# exit()
# Normalize none list data
# for dataset in datasets:
#     # enumerate by station
#     for idx, local_air in enumerate(dataset.local_air):
#         scaler = StandardScaler()
#         dataset.local_air[idx] = scaler.fit_transform(dataset.local_air[idx])
#
#     for idx, local_weather in enumerate(dataset.local_weather):
#         scaler = StandardScaler()
#         dataset.local_weather[idx] = scaler.fit_transform(dataset.local_weather[idx])

def deal_dataset(origin_dataset):
    dataset = deepcopy(origin_dataset)
    dataset.local_air_lstm_datas = [None] * AIR_STATION_NUM
    dataset.Y = [None] * AIR_STATION_NUM
    for i in range(AIR_STATION_NUM):
        # For one whole training, we only pick one data
        dataset.local_air_lstm_datas[i], dataset.Y[i] = generate_lstm_data(dataset.local_air[i],
                                                                                       hasLabel=True)

    dataset.local_weather_lstm_datas = [None] * AIR_STATION_NUM
    for i in range(AIR_STATION_NUM):
        dataset.local_weather_lstm_datas[i] = generate_lstm_data(dataset.local_weather[i], stop_before=0)

    dataset.global_air_lstm_datas = [None] * AIR_STATION_NUM
    for i in range(AIR_STATION_NUM):
        dataset.global_air_lstm_datas[i] = []
        num = len(dataset.global_air[i])
        for j in range(num):
            dataset.global_air_lstm_datas[i].append(generate_lstm_data(dataset.global_air[i][j]))

    dataset.global_weather_lstm_datas = [None] * AIR_STATION_NUM
    for i in range(AIR_STATION_NUM):
        dataset.global_weather_lstm_datas[i] = []
        num = len(dataset.global_weather[i])
        for j in range(num):
            dataset.global_weather_lstm_datas[i].append(
                generate_lstm_data(dataset.global_weather[i][j], stop_before=0))

    dataset.global_locations_datas = [None] * AIR_STATION_NUM
    for i in range(AIR_STATION_NUM):
        dataset.global_locations_datas[i] = []
        num = len(dataset.global_weather[i])
        for j in range(num):
            dataset.global_locations_datas[i].append(generate_locations_data(dataset.global_locations[i][j]))

    return dataset

# generate local air lstm data
for idx, dataset in enumerate(datasets):
    datasets[idx] = deal_dataset(dataset)
    # datasets[idx].local_air_lstm_datas = [None] * AIR_STATION_NUM
    # datasets[idx].Y = [None] * AIR_STATION_NUM
    # for i in range(AIR_STATION_NUM):
    #     # For one whole training, we only pick one data
    #     datasets[idx].local_air_lstm_datas[i], datasets[idx].Y[i] = generate_lstm_data(dataset.local_air[i], hasLabel=True)
    #
    # datasets[idx].local_weather_lstm_datas = [None] * AIR_STATION_NUM
    # for i in range(AIR_STATION_NUM):
    #     datasets[idx].local_weather_lstm_datas[i] = generate_lstm_data(dataset.local_weather[i], stop_before=0)
    #
    #
    # datasets[idx].global_air_lstm_datas = [None] * AIR_STATION_NUM
    # for i in range(AIR_STATION_NUM):
    #     datasets[idx].global_air_lstm_datas[i] = []
    #     num = len(dataset.global_air[i])
    #     for j in range(num):
    #         datasets[idx].global_air_lstm_datas[i].append(generate_lstm_data(dataset.global_air[i][j]))
    #
    # datasets[idx].global_weather_lstm_datas = [None] * AIR_STATION_NUM
    # for i in range(AIR_STATION_NUM):
    #     datasets[idx].global_weather_lstm_datas[i]= []
    #     num = len(dataset.global_weather[i])
    #     for j in range(num):
    #         datasets[idx].global_weather_lstm_datas[i].append(generate_lstm_data(dataset.global_weather[i][j], stop_before=0))
    #
    # datasets[idx].global_locations_datas = [None] * AIR_STATION_NUM
    # for i in range(AIR_STATION_NUM):
    #     datasets[idx].global_locations_datas[i] = []
    #     num = len(dataset.global_weather[i])
    #     for j in range(num):
    #         datasets[idx].global_locations_datas[i].append(generate_locations_data(dataset.global_locations[i][j]))

# dataset_idx : [current local station number, global air station num around current local station, batch count]
# print(datasets[0].global_air_lstm_datas[0][0][0])
#
# print("test")
#
# print(datasets[0].global_air_lstm_datas[0][0][1])










# print(np.shape(weather_locations_shared_fc_feed))

#
# print(air_locations_shared_fc_feed[0][0])
# print(air_locations_shared_fc_feed[0][1])
# print(weather_locations_shared_fc_feed[0][0])
# print(weather_locations_shared_fc_feed[0][1])
#
# print(air_lstm_datas[0])
# print(air_lstm_datas[0][0])
# exit()
class FModel(object):
    def __init__(self, global_station_num, is_training):
        self.global_station_num = global_station_num
        self.batch_size = BATCH_SIZE
        self.num_steps = NUM_STEPS

        # there seems can use tf.int32?
        self.targets = tf.placeholder(tf.float32, [BATCH_SIZE, len(Labels)])

        #         self.air_lstm_inputs = tf.placeholder(tf.float32, [batch_size])
        # self.temp_targets = tf.placeholder(tf.float32, [batch_size])



        # Build Model

        self.local_air_lstm_inputs = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_STEPS, AIR_FEATURE_NUM])

        self.local_weather_lstm_inputs = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_STEPS, LOCAL_WEATHER_FEATURE_NUM])

        self.attention_chosen_inputs = tf.placeholder(tf.float32, [self.global_station_num, BATCH_SIZE, ATTENTION_JUDGE_FEATURE_NUM])

        self.air_lstm_inputs = tf.placeholder(tf.float32, [self.global_station_num, BATCH_SIZE, NUM_STEPS, AIR_FEATURE_NUM])

        self.weather_lstm_inputs = tf.placeholder(tf.float32, [self.global_station_num, BATCH_SIZE, NUM_STEPS, WEATHER_FEATURE_NUM])

        self.air_lstms = []
        self.weather_lstms = []

        self.weather_lstms_states = []
        self.air_lstms_states = []


        with tf.variable_scope("loccal_air_lstm") as scope:
            self.local_air_lstm = LSTM_model(self.local_air_lstm_inputs)

        with tf.variable_scope("local_weather_lstm"):
            self.local_weather_lstm = LSTM_model(self.local_weather_lstm_inputs)



        # share parametes between them
        with tf.variable_scope("air_lstm", reuse=tf.AUTO_REUSE):


            for i in range(self.global_station_num):
                model = LSTM_model(self.air_lstm_inputs[i])
                # if self.air_lstms_states != []:
                #     print(len(self.air_lstms_states))
                #     model.set_state(self.air_lstms_states[i])
                self.air_lstms.append(model)
                # self.air_lstms_states.append(self.air_lstms[i].state)
        # share parametes between them
        with tf.variable_scope("weather_lstm", reuse=tf.AUTO_REUSE):


            for i in range(self.global_station_num):
                model = LSTM_model(self.weather_lstm_inputs[i])
                # if self.weather_lstms_states != []:
                #     model.set_state(self.weather_lstms_states[i])
                self.weather_lstms.append(model)
                # self.weather_lstms_states.append(self.weather_lstms[i].state)



        # share parametes between them
        # with tf.variable_scope("location_fc", reuse=tf.AUTO_REUSE):
        #     # with tf.variable_scope("local"):
        #     #     self.local_fc_output = location_FC(self.local_fc_inputs)
        #     with tf.variable_scope("air"):
        #         self.air_location_fc_outputs = []
        #         for i in range(AIR_STATION_NUM):
        #             self.air_location_fc_outputs.append(location_FC(self.air_fc_inputs[i]))
        #     with tf.variable_scope("weather"):
        #         self.weather_location_fc_outputs = []
        #         for i in range(GRID_WEAtHER_STATION_NUM):
        #             self.weather_location_fc_outputs.append(location_FC(self.weather_fc_inputs[i]))

        # self.local_high_level_fc_output =  high_level_fc([self.local_fc_output, self.local_air_lstm.output])
        # Do we still need this layer?
        # self.local_high_level_fc_output = high_level_fc(self.local_air_lstm.output)
        with tf.variable_scope("high_level_fc_for_weather_and_air", reuse = tf.AUTO_REUSE):
            self.high_level_fc_outputs = []
            for i in range(self.global_station_num):
                with tf.variable_scope("higle_level_fc"):
                    # print(self.weather_lstms[i].output.get_shape())
                    # print(self.air_lstms[i].output.get_shape())
                    # print("test")
                    air_high_level_fc_inputs = tf.concat([self.weather_lstms[i].output,
                                                          self.air_lstms[i].output], axis=1)
                    self.high_level_fc_outputs.append(high_level_fc(air_high_level_fc_inputs))


        with tf.variable_scope("attention_chosen_fc", reuse = tf.AUTO_REUSE):
            self.attention_chosen_outputs = []
            for i in range(self.global_station_num):
                self.attention_chosen_outputs.append(attention_chosen_layer(self.attention_chosen_inputs[i]))

        # Is use local lstm to feed directly without fc OK?
        with tf.variable_scope("air_attention"):


            air_station_attention_output = attention_layer(self.high_level_fc_outputs, self.attention_chosen_outputs,
                                                           BATCH_SIZE, HIGH_LEVEL_FC_HIDDEN_SIZE, ATTENTION_CHOSEN_HIDDEN_SIZE)
        # Do we need local weather?
        # If we use local weather, we seems to find the relationship between local weather and other station weathers
        # Is above true under the condition that we have all the near weather?
        # Notice that, there have difference between input local and other weather here or just include local
        # weather in other weathers. Because if we input here, we will concat every other weather with local to
        # calucate each score, however if we did not input in this function but include local weather in all weather
        # outputs, every single input in score calculation is only one station weather.
        # In my view, just use all the near weather is weather
        # with tf.variable_scope("weather_attention"):
        #     weather_station_attention_output = attention_layer_uni_input(self.weather_high_level_fc_outputs,
        #                                                                  BATCH_SIZE, HIGH_LEVEL_FC_HIDDEN_SIZE)

        t_air = tf.transpose(air_station_attention_output)
        # t_weather = tf.transpose(weather_station_attention_output)


        fusion_fc_inputs = tf.concat([t_air, self.local_weather_lstm.output, self.local_air_lstm.output], axis = 1)
        fusion_fc_outputs =  fusion_fc_layer(fusion_fc_inputs)

        self.results = predict_layer(fusion_fc_outputs)

        # loss = tf.losses.mean_squared_error(labels=self.targets, predictions=self.results)

        self.losses = tf.square(tf.subtract(self.targets, self.results))

        # average cost
        self.cost = tf.div(tf.reduce_sum(self.losses), BATCH_SIZE)
        # self.train_op = tf.contrib.layers.optimize_loss(
        #     loss, tf.train.get_global_step(), optimizer="Adam", learning_rate=0.01)

        if not is_training:
            return

        self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)


def run_epoch(session, model, batch_count, train_op, output_log, step,
              np_local_air, np_local_weather, np_global_air, np_global_weather,
              np_global_location, targets):
    total_costs = 0.0
    iters = 0

    # global states, each is different air state
    # air_states = [session.run(model.air_lstms[0].initial_state)] * AIR_STATION_NUM
    # weather_states = [session.run(model.weather_lstms[0].initial_state)] * AIR_STATION_NUM
    #
    # local_weather_state = session.run(model.local_weather_lstm.initial_state)
    #
    # local_air_state = session.run(model.local_air_lstm.initial_state)

    for batch_idx in range(batch_count):

        print("In batch " + str(batch_idx))
        # print(np.shape(np_local_air))
        # print(np.shape(np_local_weather))
        # print(np.shape(np_global_air))
        # print(np.shape(np_global_weather))
        # print(np.shape(np_global_location))
        # print(np.shape(targets))

        #
        # cost, _, output, losses, local_weather_state, local_air_state, weather_states, air_states = session.run([model.cost, train_op, model.results, model.losses,
        #                                        model.local_weather_lstm.state, model.local_air_lstm.state,
        #                                                               model.weather_lstms_states, model.air_lstms_states],
        #                                {model.local_air_lstm_inputs: np_local_air[batch_idx],
        #                                 # the input below is all list of batch data
        #                                 model.local_weather_lstm_inputs: np_local_weather[batch_idx],
        #                                 model.air_lstm_inputs: np_global_air[batch_idx],
        #                                 model.weather_lstm_inputs: np_global_weather[batch_idx],
        #                                 model.attention_chosen_inputs: np_global_location[batch_idx],
        #                                 model.targets: targets[batch_idx],
        #                                 model.local_weather_lstm.initial_state: local_weather_state,
        #                                 model.local_air_lstm.initial_state: local_air_state,
        #                                 model.weather_lstms_states: weather_states,
        #                                 model.air_lstms_states: air_states})


        cost, _, output, losses = session.run(
            [model.cost, train_op, model.results, model.losses],
            {model.local_air_lstm_inputs: np_local_air[batch_idx],
             # the input below is all list of batch data
             model.local_weather_lstm_inputs: np_local_weather[batch_idx],
             model.air_lstm_inputs: np_global_air[batch_idx],
             model.weather_lstm_inputs: np_global_weather[batch_idx],
             model.attention_chosen_inputs: np_global_location[batch_idx],
             model.targets: targets[batch_idx],
             })
        #         cost, _ = session.run([model.cost, train_op], {model.targets: y[0][batch_idx],
        #
        #                                                model.temp_targets: train_Y})
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

    # [AIR_STATION_NUM (total data by station), DATA_NUM (in one total train), BATCH_COUNT, BATCH_SIZE, FEATURE_NUM)]

    # np_air_lstm_datas = np.swapaxes(np.array(air_lstm_datas), 0, 1)
    # np_weather_lstm_datas = np.swapaxes(np.array(weather_lstm_datas), 1, 2)

    for model_idx in range(AIR_STATION_NUM):



        # for datasets 0
        # initial step

        tf.reset_default_graph()

        global_station_number = len(datasets[0].global_air[model_idx])
        train_model = FModel(global_station_number, True)



        with tf.Session() as sess:


            saver = tf.train.Saver()

            sess.run(tf.global_variables_initializer())



            step = 0

            for epoch_idx in range(TOTAL_EPOCH):
                print("total epoch :", epoch_idx)




                # define BATCH_COUNT in global config here
                # the locaton batch rely on this too
                example_data = datasets[0].global_air_lstm_datas[model_idx][0]
                BATCH_COUNT = len(example_data)
                print("BATCH_C ", BATCH_COUNT)

                train_length = len(example_data) - 1

                np_local_air = np.array(datasets[0].local_air_lstm_datas[model_idx])
                np_local_weather = np.array(datasets[0].local_weather_lstm_datas[model_idx])
                # swap axes so that we can first choose data by trained local station, then get data by bacth_idx
                np_global_air = np.swapaxes(np.array(datasets[0].global_air_lstm_datas[model_idx]), 0, 1)
                np_global_weather = np.swapaxes(np.array(datasets[0].global_weather_lstm_datas[model_idx]), 0, 1)
                np_global_location = np.swapaxes(np.array(datasets[0].global_locations_datas[model_idx]), 0, 1)
                np_Y = np.array(datasets[0].Y[model_idx])

                # print(np.shape(np_local_air))
                # print(np.shape(np_local_weather))
                # print(np.shape(np_global_air))
                # print(np.shape(np_global_weather))
                # print(np.shape(np_global_location))
                # print(np.shape(np_Y))

                train_model.global_station_num = len(datasets[0].global_air[model_idx])
                # For every station, run NUM_EPOCH baches

                for i in range(FIRST_EPOCH):
                    print("In iteration ", i)

                    step, total_cost = run_epoch(sess, train_model, BATCH_COUNT, train_model.train_op, True, step,
                                                 np_local_air, np_local_weather, np_global_air, np_global_weather,
                                                 np_global_location, np_Y)
                    print("Step: ", step)
                    print(total_cost / BATCH_COUNT)


                # datasets 1

                example_data = datasets[1].global_air_lstm_datas[model_idx][0]
                BATCH_COUNT = len(example_data)
                print("BATCH_C ", BATCH_COUNT)

                train_length = len(example_data) - 1

                train_model.global_station_num = len(datasets[1].global_air[model_idx])

                np_local_air = np.array(datasets[1].local_air_lstm_datas[model_idx])
                np_local_weather = np.array(datasets[1].local_weather_lstm_datas[model_idx])
                # swap axes so that we can first choose data by trained local station, then get data by bacth_idx
                np_global_air = np.swapaxes(np.array(datasets[1].global_air_lstm_datas[model_idx]), 0, 1)
                np_global_weather = np.swapaxes(np.array(datasets[1].global_weather_lstm_datas[model_idx]), 0, 1)
                np_global_location = np.swapaxes(np.array(datasets[1].global_locations_datas[model_idx]), 0, 1)
                np_Y = np.array(datasets[1].Y[model_idx])

                for i in range(SECOND_EPOCH):
                    print("In iteration ", i)

                    step, total_cost = run_epoch(sess, train_model, BATCH_COUNT, train_model.train_op, True, step,
                                                 np_local_air, np_local_weather, np_global_air, np_global_weather,
                                                 np_global_location, np_Y)
                    print("Step: ", step)
                    print(total_cost / BATCH_COUNT)




                saver.save(sess, './my_model-' + str(model_idx) + ".model")


def predict():
    for model_idx in [8]:
        dataset = datasets[1]
        global_station_number = len(datasets[1].global_air[model_idx])
        eval_model = FModel(global_station_number, False)
        model = eval_model
        train_op = tf.no_op()

        example_data = datasets[1].global_air_lstm_datas[model_idx][0]
        BATCH_COUNT = len(example_data)
        print("BATCH_C ", BATCH_COUNT)

        saver = tf.train.Saver()

        np_local_air = np.array(datasets[1].local_air_lstm_datas[model_idx])
        np_local_weather = np.array(datasets[1].local_weather_lstm_datas[model_idx])
        # swap axes so that we can first choose data by trained local station, then get data by bacth_idx
        np_global_air = np.swapaxes(np.array(datasets[1].global_air_lstm_datas[model_idx]), 0, 1)
        np_global_weather = np.swapaxes(np.array(datasets[1].global_weather_lstm_datas[model_idx]), 0, 1)
        np_global_location = np.swapaxes(np.array(datasets[1].global_locations_datas[model_idx]), 0, 1)
        np_Y = np.array(datasets[1].Y[model_idx])

        num = len(np_global_air[0])
        print(num)




        test_local_weather = datasets[1].test_local_weather[model_idx]
        test_global_weather = datasets[1].test_global_weather[model_idx]
        test_global_location = datasets[1].test_global_locations[model_idx]
        test_local_air = datasets[1].test_local_air[model_idx]
        test_global_air = datasets[1].test_global_air[model_idx]


        future_local_air = testDataByStation[model_idx]

        future_global_air = []

        for index in indexes[model_idx]:
            future_global_air.append(testDataDict[index])

        #
        # print(np_local_weather)
        # print(np_local_air)
        # print(np_global_weather)
        # print(np_global_location)
        # print(np_global_air)


        #
        #
        # # TODO: need to change them to training scaler of themselves
        # weather_scaler = StandardScaler()
        # weather_scaler.fit_transform(test_local_weather)
        #
        # local_air_scaler = StandardScaler()
        # local_air_scaler.fit_transform(test_local_air)
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        # for i in range(len(datasets[1].test_global_locations[model_idx])):
        #     global_weather_scaler = StandardScaler()
        #     global_weather_scaler.fit_transform(datasets[1].test_global_weather[model_idx][i])
        #     location_scaler = StandardScaler()
        #     location_scaler.fit_transform(datasets[1].test_global_locations[model_idx][i])
        #     global_air_scaler = StandardScaler()
        #     global_air_scaler.fit_transform(test_global_air[model_idx][i])
        #
        # print(test_local_weather)
        # print(test_local_air)
        # print(test_global_weather[0])
        # print(test_global_location[0])
        # print(test_global_air[0])

        with tf.Session() as sess:
            saver.restore(sess, "my_model-0.model-1131")
            batch_idx = BATCH_COUNT-1
            # sess.run(eval_model.results)

            for i in range(48):
                cost, _, output, losses = sess.run(
                    [model.cost, train_op, model.results, model.losses],
                    {model.local_air_lstm_inputs: np_local_air[batch_idx],
                     # the input below is all list of batch data
                     model.local_weather_lstm_inputs: np_local_weather[batch_idx],
                     model.air_lstm_inputs: np_global_air[batch_idx],
                     model.weather_lstm_inputs: np_global_weather[batch_idx],
                     model.attention_chosen_inputs: np_global_location[batch_idx],
                     model.targets: np_Y[batch_idx],
                     })

                result = output[-1]

                dataset.local_air[model_idx].loc[len(dataset.local_air[model_idx])] = result

                dataset.local_weather[model_idx].loc[len(dataset.local_weather[model_idx])] = dataset.test_append_local_weather[model_idx].iloc[i]


                for j in range(num):
                    dataset.global_weather[model_idx][j].loc[len(dataset.global_weather[model_idx][j])] = dataset.test_append_global_weather[model_idx][j].iloc[i]
                    dataset.global_locations[model_idx][j].loc[len(dataset.global_locations[model_idx][j])] = dataset.test_append_location[model_idx][j].iloc[i]
                    dataset.global_air[model_idx][j].loc[len(dataset.global_air[model_idx][j])] = future_global_air[j].iloc[i]


                dataset = deal_dataset(dataset)

                np_local_air = np.array(dataset.local_air_lstm_datas[model_idx])
                np_local_weather = np.array(dataset.local_weather_lstm_datas[model_idx])
                # swap axes so that we can first choose data by trained local station, then get data by bacth_idx
                np_global_air = np.swapaxes(np.array(dataset.global_air_lstm_datas[model_idx]), 0, 1)
                np_global_weather = np.swapaxes(np.array(dataset.global_weather_lstm_datas[model_idx]), 0, 1)
                np_global_location = np.swapaxes(np.array(dataset.global_locations_datas[model_idx]), 0, 1)
                np_Y = np.array(dataset.Y[model_idx])


                print(result)
            # print(output)


# main()
predict()