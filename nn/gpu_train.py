import csv
import tensorflow as tf
import os
import scipy.io as sio
import numpy as np
from nn.td_prediction_lstm_V3 import td_prediction_lstm_V3
from nn.td_prediction_lstm_V4 import td_prediction_lstm_V4
from utils import *
from configuration import MODEL_TYPE, MAX_TRACE_LENGTH, FEATURE_NUMBER, BATCH_SIZE, GAMMA, H_SIZE, model_train_continue, FEATURE_TYPE, ITERATE_NUM, learning_rate, SPORT, save_mother_dir
tf.debugging.set_log_device_placement(True)

LOG_DIR = save_mother_dir + "/log/logging" + str(
    FEATURE_TYPE) + "_batch" + str(
    BATCH_SIZE) + "_iterate" + str(
    ITERATE_NUM) + "_lr" + str(
    learning_rate) + "_" + str(MODEL_TYPE) + "_MaxTL" + str(MAX_TRACE_LENGTH)
SAVED_NETWORK = save_mother_dir + "/log/saved_networks_feature" + str(
    FEATURE_TYPE) + "_batch" + str(
    BATCH_SIZE) + "_iterate" + str(
    ITERATE_NUM) + "_lr" + str(
    learning_rate) + "_" + str(MODEL_TYPE) + "_MaxTL" + str(MAX_TRACE_LENGTH)
DATA_STORE = "./pickles"

DIR_GAMES_ALL = os.listdir(DATA_STORE)
number_of_total_game = len(DIR_GAMES_ALL)


def write_game_average_csv(data_record):
    """
    write the cost of training
    :param data_record: the recorded cost dict
    """
    try:
        if os.path.exists(LOG_DIR + '/avg_cost_record.csv'):
            with open(LOG_DIR + '/avg_cost_record.csv', 'a') as csvfile:
                fieldnames = (data_record[0]).keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for record in data_record:
                    writer.writerow(record)
        else:
            with open(LOG_DIR + '/avg_cost_record.csv', 'w') as csvfile:
                fieldnames = (data_record[0]).keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for record in data_record:
                    writer.writerow(record)
    except:
        if os.path.exists(LOG_DIR + '/avg_cost_record2.csv'):
            with open(LOG_DIR + '/avg_cost_record.csv', 'a') as csvfile:
                fieldnames = (data_record[0]).keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for record in data_record:
                    writer.writerow(record)
        else:
            with open(LOG_DIR + '/avg_cost_record2.csv', 'w') as csvfile:
                fieldnames = (data_record[0]).keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for record in data_record:
                    writer.writerow(record)


def train_network(sess, model):
    """
    training thr neural network game by game
    :param sess: session of tf
    :param model: nn model
    :return:
    """
    game_number = 0
    global_counter = 0
    converge_flag = False

    # loading network
    saver = tf.train.Saver()
    merge = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    sess.run(tf.global_variables_initializer())

    ## Preload and resume training
    if model_train_continue:
        checkpoint = tf.train.get_checkpoint_state(SAVED_NETWORK)
        if checkpoint and checkpoint.model_checkpoint_path:
            check_point_game_number = int((checkpoint.model_checkpoint_path.split("-"))[-1])
            game_number_checkpoint = check_point_game_number % number_of_total_game
            game_number = check_point_game_number
            game_starting_point = 0
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    game_diff_record_all = []
    iteration_now=0
    ## Training loop
    while True:
        game_diff_record_dict = {}
        iteration_now +=1
        game_diff_record_dict.update({"Iteration": iteration_now})
        if converge_flag:
            break
        elif ITERATE_NUM >  ITERATE_NUM:
            break
        else:
            converge_flag = True
        
        ##Read in reward, state, and trace from files
        v_diff_record = []
        game_number += 1
        game_cost_record = []
        game_files = os.listdir(DATA_STORE)
        game_info_list=[]
        for filename in game_files:
            game_info_list.append(np.load("./pickles/"+filename,allow_pickle=True))               
        values_errors=0
        for game_number,game in enumerate(game_info_list):
            try:
                reward, episode, episode_length,event_type,final_tl,home=game[0]
            except:
                print("Values problem")
                values_errors+=1
                continue
            for game_data in game:
                reward, episode, episode_length,event_type,final_tl,home=game_data
                # s_t0 = observations[train_number]
                
                possession_number=0
                s_t0 = episode[possession_number]
                possession_number+=1
                while possession_number<len(episode):
                    # try:
                    batch_return, possession_number, s_tl = get_nba_possessesion_batch(s_t0,episode,reward,possession_number,final_tl,home,event_type,BATCH_SIZE)

                    # get the batch variables
                    s_t0_batch = [d[0] for d in batch_return]
                    s_t1_batch = [d[1] for d in batch_return]
                    r_t_batch = [d[2] for d in batch_return]
                    trace_t0_batch=[1 for i in s_t0_batch]
                    trace_t1_batch=[1 for i in s_t1_batch]
                    # trace_t0_batch = [d[3] for d in batch_return]
                    # trace_t1_batch = [d[4] for d in batch_return]
                    y_batch = []

                    [outputs_t1, readout_t1_batch] = sess.run([model.outputs, model.read_out],
                                                            feed_dict={model.trace_lengths: trace_t1_batch,
                                                                        model.rnn_input: s_t1_batch})

                    for i in range(0, len(batch_return)):
                        terminal = batch_return[i][3]
                        cut = batch_return[i][4]
                        # if terminal, only equals reward
                        if terminal or cut:
                            y_home = float((r_t_batch[i])[0])
                            y_away = float((r_t_batch[i])[1])
                            y_batch.append([y_home, y_away])
                            break
                        else:
                            y_home = float((r_t_batch[i])[0]) + GAMMA * ((readout_t1_batch[i]).tolist())[0]
                            y_away = float((r_t_batch[i])[1]) + GAMMA * ((readout_t1_batch[i]).tolist())[1]
                
                            y_batch.append([y_home, y_away])

                    # perform gradient step
                    y_batch = np.asarray(y_batch)
                    [diff, read_out, cost_out, summary_train, _] = sess.run(
                        [model.diff, model.read_out, model.cost, merge, model.train_step],
                        feed_dict={model.y: y_batch,
                                model.trace_lengths: trace_t0_batch,
                                model.rnn_input: s_t0_batch})

                    v_diff_record.append(diff)

                    # if cost_out > 0.0001:
                    #     converge_flag = False
                    global_counter += 1
                    game_cost_record.append(cost_out)
                    train_writer.add_summary(summary_train, global_step=global_counter)
                    s_t0 = s_tl

                    # print info
                    # if terminal or ((possession_number - 1) / BATCH_SIZE) % 5 == 1:
                    #     print("TIMESTEP:", possession_number, "Game:", game_number)
                    #     home_avg = sum(read_out[:, 0]) / len(read_out[:, 0])
                    #     away_avg = sum(read_out[:, 1]) / len(read_out[:, 1])
                        
                    #     print("home average:{0}, away average:{1}".format(str(home_avg), str(away_avg)
                    #                                                                         ))
                    #     print("cost of the network is" + str(cost_out))

            print("End of game:",game_number)
            # save progress after a game
        saver.save(sess, SAVED_NETWORK + '/' + SPORT + '-iteration-', global_step=iteration_now)
        v_diff_record_average = sum(v_diff_record) / len(v_diff_record)
        game_diff_record_dict.update({dir_game: v_diff_record_average})
        
                
        cost_per_game_average = sum(game_cost_record) / len(game_cost_record)
        write_game_average_csv([{"iteration": str(game_number / (number_of_total_game + 1)), "game": game_number,
                                "cost_per_game_average": cost_per_game_average}])

        game_diff_record_all.append(game_diff_record_dict)


def train_start():
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)
    if not os.path.isdir(SAVED_NETWORK):
        os.mkdir(SAVED_NETWORK)

    sess = tf.InteractiveSession()
    if MODEL_TYPE == "v3":
        nn = td_prediction_lstm_V3(FEATURE_NUMBER, H_SIZE, MAX_TRACE_LENGTH, learning_rate)
    elif MODEL_TYPE == "v4":
        nn = td_prediction_lstm_V4(FEATURE_NUMBER, H_SIZE, MAX_TRACE_LENGTH, learning_rate)
    else:
        raise ValueError("MODEL_TYPE error")
    train_network(sess, nn)


if __name__ == '__main__':
from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())
    train_start()
