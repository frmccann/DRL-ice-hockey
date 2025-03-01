import csv
import pandas as pd
import numpy as np
from configuration import *
import os
import pickle
class GameProcessor:
    def __init__(self, csv_movement, csv_event, reward_map, sample_factor=10, trace_length=5):
        self.csv_movement = csv_movement
        self.df_mvmt = pd.read_csv(csv_movement)
        self.df_evt = pd.read_csv(csv_event)
        self.reward_map = reward_map
        self.sample_factor = sample_factor
        self.trace_length = trace_length

    def sample(self):
        data = []
        num_rows = self.df_mvmt.shape[0]
        indices = self.df_mvmt['radius'].to_numpy().nonzero()[0]
        for i in indices[::self.sample_factor]:
            rows = self.df_mvmt[i:i+11]
            data.append(rows)
        df = pd.concat(data)
        return df

    def rolling_average(self, num_average):
        pass

    def get_possession(observation):
        team1_pos = observation

    def process_game(self):
        self.df_mvmt = self.sample()
        episodes = self.create_episodes()
        chunks = []
        idx=0
        errors=0
        for episode in episodes:
            idx+=1
            try:
                processed_ep = self.process_episode(episode)
                if processed_ep is not None:
                    chunks.append(processed_ep)
            except Exception as e: 
                errors+=1
                print(e)
                print("errant episode")
                print('idx',idx)
        print("errors:",errors)
        return chunks


    def create_episodes(self):
        episodes = []
        event_id = self.df_mvmt['event_id'][0]
        # print('event_id', event_id)
        last_i = 0
        for (i, event) in enumerate(self.df_mvmt['event_id']):
            ## Could be an issue if back to back shots, but shots always followed by rebounds? so should be fine
            if event != event_id:
                episode = self.df_mvmt[last_i:i]
                event_id = event
                last_i = i
                episodes.append(episode)
        # print(episodes)
        return episodes
    def get_team_ids(self):
        tids = self.df_mvmt['team_id']
        return tids[1], tids[6] # Home, away

    def process_episode(self, episode):
        evt_num = episode['event_id'].iloc[0]
        df_n = self.df_evt[self.df_evt.EVENTNUM == evt_num]
        if df_n.empty:
            return None
        event = df_n['EVENTMSGTYPE'].iloc[0]
        # print("event:",event)
        reward = self.reward_map[event]
        # print("reward:",reward)
        observations = []
        actions_1 = []
        actions_2 = []
        prev_obs = np.zeros(24)
        # print('ep', episode)
        indices = range(episode.shape[0])[::11]
        tl=len(indices)
        # if tl>MAX_TRACE_LENGTH:
        #     start_idx=tl-MAX_TRACE_LENGTH
        # else:
        #     start_idx=0

        
        # if tl<MAX_TRACE_LENGTH:
        #     for i in range(0,MAX_TRACE_LENGTH-tl):
        #         observations.append(np.zeros(24))

        prev_shot_clock = 24
        possession = 0
        for i, idx in enumerate(indices):

            rows = episode[idx:idx+11]

            shot_clock = rows['shot_clock'].iloc[0]
            if np.isnan(shot_clock):
                shot_clock = prev_shot_clock
            else:
                prev_shot_clock = shot_clock
            # print('rows', rows)
            # print('ss', rows['shot_clock'])
            observation = np.zeros(24)
            observation[:2] = [
                shot_clock,
                rows['game_clock'].iloc[0],
            ]
            observation[2:13] = rows['x_loc']
            observation[13:] = rows['y_loc']
            
            # full_action = prev_obs - observation
            # full_action = full_action[2:]
            # prev_obs = observation
            # action_1 = full_action[1:6] + full_action[12:17]
            # action_2 = full_action[6:11] + full_action[17:]
            # actions_1.append(action_1)
            # actions_2.append(action_2)
            observations.append(observation)
            
            if i == 0:
                possession = get_possession(observation)
                # print('possession', p)
                
        observations, final_trace_length = padded_chunks(np.array(observations), self.trace_length)
        # return reward, np.array(observations), np.array(actions_1), np.array(actions_2), final_trace_length
        return reward, np.array(observations), len(observations), event, final_trace_length , possession

def padded_chunks(l, n):
    """Yield successive n-sized chunks from l."""
    tl = n
    out = []
    for i in range(0, len(l), n):
        if i + n > len(l):
            tl = n - (len(l) - i)
            out.append(np.pad(np.array(l[i:]), ((0, tl), (0, 0)), 'constant'))
        else:
            out.append(np.array(l[i:i + n]))
    return out, tl

def get_possession(observation):
    xs = observation[3:13]
    ys = observation[14:]

    players = np.transpose([xs, ys])
    ball = [observation[2], observation[13]]
    dists = np.mean(np.square(players - ball), 1)
    return 0 if np.argmin(dists) < 5 else 1


reward_map = {
    1: 2,
    2: 0,
    3: 1,
    4: 1,
    5: -1,
    6: -0.5,
    7:-1,
    8:0,
    9:0,
    10:0,
    11:0,
    12:0,
    13:0
}

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def test():
    movement_files = os.listdir("../nba-movement-data/data/csv/")
    event_files=os.listdir("../nba-movement-data/data/events/")
    game_to_teams={}
    game_number=0
    for movement_file,event_file in zip(movement_files[game_number:],event_files[game_number:]):
        print("game:",game_number,movement_file,event_file)
        print("progress:",game_number/len(movement_files))
        csv_movement="../nba-movement-data/data/csv/"+movement_file
        csv_event="../nba-movement-data/data/events/"+event_file
        game_number+=1
        gp = GameProcessor(csv_movement, csv_event, reward_map)
        filename="./pickles/game_"+str(game_number)
        try:
            game_to_teams[filename]=tuple(gp.get_team_ids())
            # episodes= gp.process_game()
            # np.save(filename,episodes,allow_pickle=True,fix_imports=True)
        except:
            game_number-=1
            continue
    # r, o, l = gp.process_game()[1]
    # print('r', r)
    # print('o', o
    print(game_number)
    save_obj(game_to_teams,"game_to_teams")
test()
