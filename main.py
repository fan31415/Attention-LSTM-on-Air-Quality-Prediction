import tensorflow as tf
import pandas as pd
import numpy as np
from dataHelper import PreProcessor
import pickle
import os
import sys
from sklearn.model_selection import KFold

from copy import deepcopy


# local lib

from model import *

from dataHelper import *

from config import *

trainDataByStation, station_files = load_data()

testDataByStation, files = load_data(dir = TEST_DATA_DIR)
testDataDict = {}

for idx, file in enumerate(files):
    name = file.split('.')[0].split('_')[2:4]
    name = "_".join(name)
    testDataDict[name] = testDataByStation[idx]

pd.options.mode.chained_assignment = None

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

preprocessor_list = []
for model_idx in range(len(trainDataByStation)):
    print("normalization: Model %d" % model_idx)
    num_neighbor = len(datasets[0].global_air[model_idx])
    preprocessor = PreProcessor()
    # normalize data
    local_air_pack = pd.concat([datasets[0].local_air[model_idx], datasets[1].local_air[model_idx]], axis=0)
    local_weather_pack = pd.concat([datasets[0].local_weather[model_idx], datasets[1].local_weather[model_idx]], axis=0)
    preprocessor.local_air_scalar.fit(local_air_pack)
    preprocessor.local_weather_scalar.fit(local_weather_pack)
    del local_air_pack
    del local_weather_pack
    neighbor_air_pack = pd.concat([datasets[0].global_air[model_idx].iloc[0], datasets[1].global_air[model_idx].iloc[0]], axis=0)
    for neigh_idx in range(1, num_neighbor):
        neighbor_air_pack = pd.concat(
            [neighbor_air_pack, datasets[0].global_air[model_idx].iloc[neigh_idx], datasets[1].global_air[model_idx].iloc[neigh_idx]], axis=0)
    preprocessor.neighbor_air_scalar.fit(neighbor_air_pack)
    del neighbor_air_pack

    neighbor_weather_pack = pd.concat([datasets[0].global_weather[model_idx].iloc[0], datasets[1].global_weather[model_idx].iloc[0]], axis=0)
    for neigh_idx in range(1, num_neighbor):
        neighbor_weather_pack = pd.concat(
            [neighbor_weather_pack, datasets[0].global_weather[model_idx].iloc[neigh_idx], datasets[1].global_weather[model_idx].iloc[neigh_idx]], axis=0)
    preprocessor.neighbor_weather_scalar.fit(neighbor_weather_pack)
    del neighbor_weather_pack

    neighbor_location_pack = pd.concat([datasets[0].global_locations[model_idx].iloc[0], datasets[1].global_locations[model_idx].iloc[0]], axis=0)
    for neigh_idx in range(1, num_neighbor):
        neighbor_location_pack = pd.concat(
            [neighbor_location_pack, datasets[0].global_locations[model_idx].iloc[neigh_idx], datasets[1].global_locations[model_idx].iloc[neigh_idx]], axis=0)
    preprocessor.neighbor_location_scalar.fit(neighbor_location_pack)
    del neighbor_location_pack
    preprocessor_list.append(preprocessor)

indexes = []

for i in range(AIR_STATION_NUM):
    num = len(dataset.global_weather[i])
    indexes.append(list(datasets[1].global_weather[i].index))

def deal_dataset(origin_dataset):
    dataset = deepcopy(origin_dataset)
    dataset.local_air_lstm_datas = [None] * AIR_STATION_NUM
    dataset.Y = [None] * AIR_STATION_NUM
    for i in range(AIR_STATION_NUM):
        # For one whole training, we only pick one data
        dataset.local_air_lstm_datas[i], dataset.Y[i] = generate_lstm_data(dataset.local_air[i],
                                                                                       hasLabel=True, data_scalar=preprocessor_list[i].local_air_scalar)

    dataset.local_weather_lstm_datas = [None] * AIR_STATION_NUM
    for i in range(AIR_STATION_NUM):
        dataset.local_weather_lstm_datas[i] = generate_lstm_data(dataset.local_weather[i], stop_before=0, data_scalar=preprocessor_list[i].local_weather_scalar)

    dataset.global_air_lstm_datas = [None] * AIR_STATION_NUM
    for i in range(AIR_STATION_NUM):
        dataset.global_air_lstm_datas[i] = []
        num = len(dataset.global_air[i])
        for j in range(num):
            dataset.global_air_lstm_datas[i].append(generate_lstm_data(dataset.global_air[i][j], data_scalar=preprocessor_list[i].neighbor_air_scalar))

    dataset.global_weather_lstm_datas = [None] * AIR_STATION_NUM
    for i in range(AIR_STATION_NUM):
        dataset.global_weather_lstm_datas[i] = []
        num = len(dataset.global_weather[i])
        for j in range(num):
            dataset.global_weather_lstm_datas[i].append(
                generate_lstm_data(dataset.global_weather[i][j], stop_before=0, data_scalar=preprocessor_list[i].neighbor_weather_scalar))

    dataset.global_locations_datas = [None] * AIR_STATION_NUM
    for i in range(AIR_STATION_NUM):
        dataset.global_locations_datas[i] = []
        num = len(dataset.global_weather[i])
        for j in range(num):
            dataset.global_locations_datas[i].append(generate_locations_data(dataset.global_locations[i][j], data_scalar=preprocessor_list[i].neighbor_location_scalar))
    return dataset


# generate local air lstm data
for idx, dataset in enumerate(datasets):
    datasets[idx] = deal_dataset(dataset)

class FModel(object):
    def __init__(self, global_station_num, is_training, sess=None, model_id=-1):
        self.global_bp_cnt = 0  # a counter to record the times running BackPropagation
        self.global_test_cnt = 0
        self.sess = sess
        self.global_station_num = global_station_num
        self.batch_size = BATCH_SIZE
        self.num_steps = NUM_STEPS
        self.model_id = model_id

        self.targets = tf.placeholder(tf.float32, [BATCH_SIZE, len(Labels)])

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
                self.air_lstms.append(model)
        # share parametes between them
        with tf.variable_scope("weather_lstm", reuse=tf.AUTO_REUSE):


            for i in range(self.global_station_num):
                model = LSTM_model(self.weather_lstm_inputs[i])
                # if self.weather_lstms_states != []:
                #     model.set_state(self.weather_lstms_states[i])
                self.weather_lstms.append(model)
                # self.weather_lstms_states.append(self.weather_lstms[i].state)

        # self.local_high_level_fc_output =  high_level_fc([self.local_fc_output, self.local_air_lstm.output])
        # Do we still need this layer?
        # self.local_high_level_fc_output = high_level_fc(self.local_air_lstm.output)
        with tf.variable_scope("high_level_fc_for_weather_and_air", reuse = tf.AUTO_REUSE):
            self.high_level_fc_outputs = []
            for i in range(self.global_station_num):
                with tf.variable_scope("higle_level_fc"):

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

        self.mse_losses = tf.square(tf.subtract(self.targets, self.results))

        self.msape_losses = tf.div(tf.abs(tf.subtract(self.targets, self.results)),
                             tf.div(tf.add(self.targets, self.results), 2))
        self.losses = self.mse_losses

        beta = 0.2
        params = tf.trainable_variables()
        params_need_reg = []
        reg_loss = None
        for i in range(len(params)):
            # no reg in predict layer, because labels are not scaled
            if "weigh" in params[i].name and "predict_layer" not in params[i].name:
                params_need_reg.append(params[i])
                print(params[i].name)
                if reg_loss is None:
                    reg_loss = tf.nn.l2_loss(params[i])
                else:
                    reg_loss += tf.nn.l2_loss(params[i])
        # average cost
        self.cost_pure = tf.div(tf.reduce_sum(self.losses), BATCH_SIZE)
        self.cost = self.cost_pure

        self.show_loss = tf.div(tf.reduce_sum(self.msape_losses), (BATCH_SIZE*len(Labels)))
        # self.cost = tf.div(tf.reduce_sum(self.losses) + beta*reg_loss, BATCH_SIZE)
        # tf.summary.scalar('train pure cost[%s]' % self.model_id, self.cost_pure)
        tf.summary.scalar('train cost with reg[%s]' % self.model_id, self.cost)
        tf.summary.scalar('show cost (msape)[%s]' % self.model_id, self.show_loss)


        # self.train_op = tf.contrib.layers.optimize_loss(
        #     loss, tf.train.get_global_step(), optimizer="Adam", learning_rate=0.01)

        if not is_training:
            return

        self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
        self.merged_summary = tf.summary.merge_all()


        if self.sess is not None:
           self.train_writer = tf.summary.FileWriter('./logs/train/', sess.graph)
           self.test_writer = tf.summary.FileWriter('./logs/test')


def run_epoch(session, model, batch_count, train_op, output_log, step,
              np_local_air, np_local_weather, np_global_air, np_global_weather,
              np_global_location, targets, big_iter=0, is_test=False):
    total_costs = 0.0
    iters = 0

    for batch_idx in range(batch_count):
        if is_test:
            model.global_test_cnt += 1
            cost, output, losses, summary, ape_loss = session.run(
                [model.cost, model.results, model.losses, model.merged_summary, model.show_loss],
                {model.local_air_lstm_inputs: np_local_air[batch_idx],
                 # the input below is all list of batch data
                 model.local_weather_lstm_inputs: np_local_weather[batch_idx],
                 model.air_lstm_inputs: np_global_air[batch_idx],
                 model.weather_lstm_inputs: np_global_weather[batch_idx],
                 model.attention_chosen_inputs: np_global_location[batch_idx],
                 model.targets: targets[batch_idx],
                 })
            print("ape:", ape_loss)
            cost = ape_loss
            model.test_writer.add_summary(summary, model.global_test_cnt)
            iters += 1
        else:
            model.global_bp_cnt += 1
            cost, _, output, losses, summary = session.run(
                [model.cost, train_op, model.results, model.losses, model.merged_summary],
                {model.local_air_lstm_inputs: np_local_air[batch_idx],
                 # the input below is all list of batch data
                 model.local_weather_lstm_inputs: np_local_weather[batch_idx],
                 model.air_lstm_inputs: np_global_air[batch_idx],
                 model.weather_lstm_inputs: np_global_weather[batch_idx],
                 model.attention_chosen_inputs: np_global_location[batch_idx],
                 model.targets: targets[batch_idx],
                 })
            model.train_writer.add_summary(summary, model.global_bp_cnt)
            iters += 1

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
        sess = tf.InteractiveSession()

        train_model = FModel(global_station_number, True, sess=sess, model_id=model_idx)



        with tf.Session() as sess:

            saver = tf.train.Saver(max_to_keep=100)

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


                train_model.global_station_num = len(datasets[0].global_air[model_idx])

                for i in range(FIRST_EPOCH):
                    print("In iteration ", i)

                    step, total_cost = run_epoch(sess, train_model, BATCH_COUNT, train_model.train_op, True, step,
                                                 np_local_air, np_local_weather, np_global_air, np_global_weather,
                                                 np_global_location, np_Y, big_iter=i)
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

                test_err_history = np.array([])
                min_loss = float("inf")
                test_idx = [np.random.randint(0, BATCH_COUNT)]
                train_idx = [n for n in range(BATCH_COUNT) if n != test_idx[0]]
                print("train chosen:", train_idx)
                print("test chosen:", test_idx)
                for i in range(SECOND_EPOCH):
                    # train/cv generate
                    ####
                    # randomly choose 20% batches to cv, and other 80% batches to train
                    #
                    print("In iteration ", i)
                    np_local_air_train = np_local_air[train_idx]
                    np_local_weather_train = np_local_weather[train_idx]
                    np_global_air_train = np_global_air[train_idx]
                    np_global_weather_train = np_global_weather[train_idx]
                    np_global_location_train = np_global_location[train_idx]
                    np_Y_train = np_Y[train_idx]
                    np_local_air_test = np_local_air[test_idx]
                    np_local_weather_test = np_local_weather[test_idx]
                    np_global_air_test = np_global_air[test_idx]
                    np_global_weather_test = np_global_weather[test_idx]
                    np_global_location_test = np_global_location[test_idx]
                    np_Y_test = np_Y[test_idx]
                    train_batch_num = len(train_idx)
                    test_batch_num = len(test_idx)

                    # train
                    step, total_cost = run_epoch(sess, train_model, train_batch_num, train_model.train_op, True, step,
                                                 np_local_air_train, np_local_weather_train, np_global_air_train,
                                                 np_global_weather_train,
                                                 np_global_location_train, np_Y_train)

                    print("Step: ", step)
                    print("Train Cost of a batch:", total_cost / train_batch_num)

                    # test
                    step, total_cost = run_epoch(sess, train_model, test_batch_num, train_model.train_op, True, step,
                                                 np_local_air_test, np_local_weather_test, np_global_air_test,
                                                 np_global_weather_test,
                                                 np_global_location_test, np_Y_test, is_test=True)
                    print("Step: ", step)
                    print("Test Error of a batch:", total_cost / test_batch_num)

                    test_err = total_cost / test_batch_num
                    print("cur min:", min_loss)
                    if test_err < min_loss:
                        min_loss = test_err
                        saver.save(sess, './my_model-' + str(model_idx) + "-acc-" + str(test_err), global_step=step)

                    if len(test_err_history) >= EARLY_STOP:
                        if test_err >= min_loss:
                            print("Early Stop. Because loss not decrease for %d epoches" % EARLY_STOP)
                            break
                        else:
                            test_err_history = np.array([])

                    test_err_history = np.append(test_err_history, test_err)
                    # print(test_err_history)


def predict():
    err_model = []
    model_not_found = []
    os.mkdir("./result/")

    for model_idx in range(AIR_STATION_NUM):
        tf.reset_default_graph()
        dataset = datasets[1]
        global_station_number = len(datasets[1].global_air[model_idx])
        eval_model = FModel(global_station_number, False)
        model = eval_model
        train_op = tf.no_op()

        example_data = datasets[1].global_air_lstm_datas[model_idx][0]
        BATCH_COUNT = len(example_data)
        print("BATCH_C ", BATCH_COUNT)

        saver = tf.train.Saver(max_to_keep=100)

        np_local_air = np.array(datasets[1].local_air_lstm_datas[model_idx])
        np_local_weather = np.array(datasets[1].local_weather_lstm_datas[model_idx])
        # swap axes so that we can first choose data by trained local station, then get data by bacth_idx
        np_global_air = np.swapaxes(np.array(datasets[1].global_air_lstm_datas[model_idx]), 0, 1)
        np_global_weather = np.swapaxes(np.array(datasets[1].global_weather_lstm_datas[model_idx]), 0, 1)
        np_global_location = np.swapaxes(np.array(datasets[1].global_locations_datas[model_idx]), 0, 1)
        np_Y = np.array(datasets[1].Y[model_idx])

        num = len(np_global_air[0])
        print(num)


        future_global_air = []

        for index in indexes[model_idx]:
            future_global_air.append(testDataDict[index])


        with tf.Session() as sess:
            def choose_model(model_idx):
                acc_filename_dict = {}
                for filename in os.listdir("./"):
                    if ".meta" in filename and str(model_idx) in filename:
                        info = filename.split("-")
                        file_model_idx = int(info[1])
                        if model_idx != file_model_idx:
                            continue
                        acc = float(info[3])
                        acc_filename_dict[filename] = acc
                if len(acc_filename_dict) == 0:
                    return None, None
                acc_rank = sorted(acc_filename_dict.items(), key=lambda k: k[1])
                return acc_rank[0][0].split(".met")[0], acc_rank[0][1]
            model_name, acc = choose_model(model_idx)
            if model_name is None:
                model_not_found.append(model_idx)
                continue
            if acc > 0.8:
                err_model.append(model_idx)
                continue
            # saver.restore(sess, "my_model-0-acc-0.5583217740058899-40")
            saver.restore(sess, model_name)

            batch_idx = BATCH_COUNT-1
            # sess.run(eval_model.results)
            result_pd = pd.DataFrame(columns=["PM2.5", "PM10", "O3"])

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

                result_pd.loc[result_pd.shape[0], :] = result
                print(result)
            result_pd.to_csv("./result/station_" + station_files[model_idx] + ".csv")
            # print(output)
    print("error model:", err_model)
    print("model not found:", model_not_found)

# if we do training or directly predict
DoTraining = True
if (DoTraining):
    main()
else:
    predict()