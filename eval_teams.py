import csv
import pickle
import tensorflow as tf
import os
import scipy.io as sio
import numpy as np
from nn.td_prediction_lstm_V3 import td_prediction_lstm_V3
from nn.td_prediction_lstm_V4 import td_prediction_lstm_V4
from utils import *
from configuration import MODEL_TYPE, MAX_TRACE_LENGTH, FEATURE_NUMBER, BATCH_SIZE, GAMMA, H_SIZE, model_train_continue, FEATURE_TYPE, ITERATE_NUM, learning_rate, SPORT, save_mother_dir
tf.debugging.set_log_device_placement(True)

DATA_STORE = "./pickles"
DIR_GAMES_ALL = os.listdir(DATA_STORE)
number_of_total_game = len(DIR_GAMES_ALL)
model_train_continue=True
SAVED_NETWORK="./models/log/model_checkpoints_2/"
def eval_teams(sess, model):
    """
    training thr neural network game by game
    :param sess: session of tf
    :param model: nn model
    :return:
    """
    game_to_teams=load_obj("game_to_teams")
    team_q_values={}
    game_number = 0
    global_counter = 0
    converge_flag = False

    # loading network
    saver = tf.train.Saver()
    merge = tf.summary.merge_all()

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

    iteration_now=0
    ## Training loop
    iteration_now +=1
    
    num_teams=200
    ##Read in reward, state, and trace from files
    game_files = os.listdir(DATA_STORE)
    game_info_list=[]
    teams=[]
    for filename in game_files:
        game_info_list.append(np.load("./pickles/"+filename[:],allow_pickle=True))               
    print("same Length?:",len(game_info_list)==len(game_files))
    for game_number,game in enumerate(game_info_list[-num_teams:]):
        print(game_number)
        # try:
        home_team=game_to_teams["./pickles/"+game_files[-num_teams+game_number][:-4]][0]
        away_team=game_to_teams["./pickles/"+game_files[-num_teams+game_number][:-4]][1]
        if home_team not in team_q_values:
            team_q_values[home_team]={"games":0,"possesions":0,"total_value":0,"movements":0}
        if away_team not in team_q_values:
            team_q_values[away_team]={"games":0,"possesions":0,"total_value":0,"movements":0}
        team_q_values[home_team]["games"]+=1
        team_q_values[away_team]["games"]+=1
        for reward, episode, episode_length,event_type,final_tl,possession in game:
            # s_t0 = observations[train_number]
            team_q_values[home_team]["possesions"]+=1
            team_q_values[away_team]["possesions"]+=1
            possession_number=0
            s_t0 = episode[possession_number]
            possession_number+=1
            
            while possession_number<len(episode):
                # try:
                batch_return, possession_number, s_tl = get_nba_possessesion_batch(s_t0,episode,reward,possession_number,final_tl,1,event_type,BATCH_SIZE)

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
                                                        feed_dict={model.trace_lengths: trace_t0_batch,
                                                                    model.rnn_input: s_t0_batch})
                home_values=0
                away_values=0
                movements=len(readout_t1_batch)
                for home,away in readout_t1_batch:
                    home_values+=home
                    away_values+=away

                team_q_values[home_team]["total_value"]+=home_values
                team_q_values[home_team]["movements"]+=movements

                team_q_values[away_team]["total_value"]+=away_values
                team_q_values[away_team]["movements"]+=movements
    # except:
        # print("errored")
    return team_q_values

                


def train_start():

    sess = tf.InteractiveSession()
    if MODEL_TYPE == "v3":
        nn = td_prediction_lstm_V3(FEATURE_NUMBER, H_SIZE, MAX_TRACE_LENGTH, learning_rate)
    elif MODEL_TYPE == "v4":
        nn = td_prediction_lstm_V4(FEATURE_NUMBER, H_SIZE, MAX_TRACE_LENGTH, learning_rate)
    else:
        raise ValueError("MODEL_TYPE error")
    dict_object=eval_teams(sess, nn)
    save_obj(dict_object,"team_eval_dict")


if __name__ == '__main__':
    train_start()
