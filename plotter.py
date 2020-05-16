import matplotlib.pyplot as plt
import numpy as np
import csv
from utils import *

class Plotter():
	def __init__(self, value_dict, team_csv):
		# self.values, self.teams = values
		self.team_map = read_team_csv(team_csv)
		self.value_dict = value_dict

	def compare_rankings(self, test=False):
		q_vals = []
		team_names = []
		power_rankings = []
		standings = []

		i = 0
		for team_id in self.team_map:
			name, power_ranking, standing = self.team_map[team_id]
			team_names.append(name)
			power_rankings.append(power_ranking)
			standings.append(standing)
			if test:
				q_vals.append((np.random.randint(1000), i))
			else:
				info_dict = self.value_dict[int(team_id)]
				q_val=info_dict['total_value']
				games=info_dict['games']
				possessions=info_dict["possesions"]
				movements=info_dict["movements"]
				q_vals.append((q_val/games, i))
			i += 1
		
		q_val_rankings = [t[1] for t in sorted(q_vals,reverse=True)]

		plt.xlabel('Teams')
		plt.ylabel('Ranking')
		plt.title('2015-16 NBA Team Rankings')

		# print(power_rankings)
		plt.plot(range(len(team_names)), power_rankings, label='Standing', marker='o')
		plt.plot(range(len(team_names)), q_val_rankings, label='Q Val per Game Ranking', marker='o')
		plt.plot(range(len(team_names)), standings, label='Power Ranking', marker='o')
		
		plt.xticks(range(len(team_names)), team_names, size='small', rotation='vertical')
		plt.legend()
		plt.show()
		plt.savefig('rankings_game_plot.png')

def read_team_csv(path):
	teams = {}
	with open(path) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			team_id = row['TEAM_ID']
			teams[team_id] = row['TEAM_NAME'], row['POWER_RANKING'], row['STANDING']
	return teams

team_csv = 'team_mapping.csv'
test_value_dict = load_obj("team_eval_dict")
plotter = Plotter(test_value_dict, team_csv)
# print(plotter.team_map)
plotter.compare_rankings(test=False)