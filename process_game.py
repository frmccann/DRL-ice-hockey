import csv
import pandas as pd
import numpy as np
from configuration import *
class GameProcessor:
	def __init__(self, csv_movement, csv_event, reward_map, sample_factor=10):
		self.csv_movement = csv_movement
		self.df_mvmt = pd.read_csv(csv_movement)
		self.df_evt = pd.read_csv(csv_event)
		self.reward_map = reward_map
		self.sample_factor = sample_factor

	def sample(self):
		data = []
		# data = pd.DataFrame(columns = self.df_mvmt.columns)
		# print(data)
		num_rows = self.df_mvmt.shape[0]
		for i in range(num_rows)[::self.sample_factor*11]:
			# print(i, row)
			rows = self.df_mvmt[i:i+11]
			# print(i, rows)
			data.append(rows)

		df = pd.concat(data)

		return df

	def rolling_average(self, num_average):
		pass

	def process_game(self):
		self.df_mvmt = self.sample()
		episodes = self.create_episodes()
		chunks = []
		idx=0
		errors=0
		for episode in episodes:
			idx+=1
			try:
				chunks.append(self.process_episode(episode))
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

	def process_episode(self, episode):
		evt_index = episode['event_id'].iloc[0]
		event = self.df_evt['EVENTMSGTYPE'].iloc[evt_index]
		print("event:",event)
		reward = self.reward_map[event]
		print("reward:",reward)
		observations = []
		actions_1 = []
		actions_2 = []
		prev_obs = np.zeros(24)
		# print('ep', episode)
		indices = range(episode.shape[0])[::11]
		tl=len(indices)
		if tl>MAX_TRACE_LENGTH:
			start_idx=tl-MAX_TRACE_LENGTH
		else:
			start_idx=0
		for i in indices[start_idx:]:

			rows = episode[i:i+11]
			# print('rows', rows)
			# print('ss', rows['shot_clock'])
			observation = np.zeros(24)
			observation[:2] = [
				rows['shot_clock'].iloc[0],
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

		# return reward, np.array(observations), np.array(actions_1), np.array(actions_2)
		return reward, np.array(observations), len(observations),event


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


def test():
	csv_movement = '../nba-movement-data/data/csv/0021500490.csv'
	csv_event = '../nba-movement-data/data/events/0021500490.csv'
	gp = GameProcessor(csv_movement, csv_event, reward_map)
	episodes= gp.process_game()
	np.save('test',episodes,allow_pickle=True,fix_imports=True)

	# r, o, l = gp.process_game()[1]
	# print('r', r)
	# print('o', o
	
test()