import matplotlib.pyplot as plt
import qlearning
import math
import numpy as np
from ple.games.pixelcopter import Pixelcopter
from ple import PLE
import math

"""
if state['next_gate_dist_to_player'] >= gate_distance_threshold:
	chosenFeatures = list(chosenFeatures)
	chosenFeatures[2] = math.ceil(gate_distance_threshold/4)
	chosenFeatures = tuple(chosenFeatures)
  """

# vars
y_axis_min = -7
y_axis_max = 16
num_runs = 7000
print_all_scores = True
total_vals_to_plot = 400 # must be >= num_runs
displayScreen = False

# hyperparams
exploration_prob = .1
using_exp_decay = True
iteration_decay_zero = 4000
deterministic_ceiling_dist = 80
deterministic_floor_dist = 15
gate_distance_threshold = 60
danger_zone = 50

# don't delete this -- early call to plt.plot prevents later crash during plot
plt.plot([])

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
	#print 'distance_to_ceiling:', state['player_dist_to_ceil']
#	print 'distance_to_floor:', state['player_dist_to_floor']
	#print 'dist to ceil / dist to floor:', round(state['player_dist_to_ceil'] / state['player_dist_to_floor'], 2)
  	chosenFeatures = tuple([
	#    math.ceil(state['player_dist_to_ceil'] / 4),
	#    math.ceil(state['player_dist_to_floor'] / 4),
		math.ceil((state['player_y'] - state['next_gate_block_top'])),
		round(state['player_dist_to_ceil'] / state['player_dist_to_floor'], 2),
	    math.ceil(state['next_gate_dist_to_player'] / 4),
	#    math.ceil(state['next_gate_block_top'] / 4),
#	    math.ceil(state['next_gate_block_bottom'] / 4),
		np.sign(state['player_vel'])
	#    round(state['player_vel'] / 3, 1)
	])
  	featureKey = (chosenFeatures, action)
  	featureValue = 1
  	return [(featureKey, featureValue)]

class Bot():
  """
  This is our bot that will automatically play Pixelcopter.
  """

  def __init__(self, actions):
    self.actions = actions
    self.qLearning = qlearning.QLearningAlgorithm(actions=actions,
      discount=1,
      featureExtractor=identityFeatureExtractor,
      explorationProb=exploration_prob,
      decayingExpProb=using_exp_decay,
      decay_target_it=iteration_decay_zero)

  def pickAction(self, state):
    # Deterministically pick action if too close to ceiling or floor.
	#print 'dist to floor:', state['player_dist_to_floor']
#	print 'dist to ceiling:', state['player_dist_to_ceil']
	if state['player_dist_to_floor'] < deterministic_floor_dist:
#		print 'deterministic floor action\n'
		return self.actions[0]    # move up
	elif state['player_dist_to_ceil'] < deterministic_ceiling_dist:
#		print 'deterministic ceiling action\n'
		return self.actions[1]    # do nothing
	#print 'qlearning action\n'
	return self.qLearning.getAction(state)

  def incorporateFeedback(self, state, action, reward, newState):
    self.qLearning.incorporateFeedback(state, action, reward, newState)

  def printWeights(self):
    print str(self.qLearning.getWeights())
    print 'num weights: %d' % len(self.qLearning.getWeights())

############################################################
if __name__ == '__main__':
  game = Pixelcopter(width=200, height=200)
  env = PLE(game, fps=30, display_screen=displayScreen)

  agent = Bot(actions=env.getActionSet())
  env.init()

  total_reward = 0.0
  min_reward = float('inf')
  max_reward = float('-inf')

  all_episode_scores = []
  plot_episode_scores = []
  plotted_episodes = []

  for i in range(num_runs):   # should run until qvalues converge
    episode_reward = 0.0
    frames = []
    while not env.game_over():
      state = game.getGameState()
      action = agent.pickAction(state)
      reward = env.act(action)
      frames.append((state, action, reward))
      episode_reward += reward

		# Update q-values
    for f in range(len(frames)-1):
      state, action, reward = frames[f]
      nextState = frames[f+1][0]
      agent.incorporateFeedback(state, action, reward, nextState)

    if print_all_scores:
      print 'Agent score {:0.1f} reward for episode %d.'.format(episode_reward) % i
    if i % (num_runs / total_vals_to_plot) is 0:
      plot_episode_scores.append(int(episode_reward))
      plotted_episodes.append(i)

    all_episode_scores.append(int(episode_reward))
    total_reward += episode_reward
    min_reward = min(min_reward, episode_reward)
    max_reward = max(max_reward, episode_reward)
    env.reset_game()

  print 'Average score for %d runs: {:0.1f}'.format(total_reward / num_runs) % num_runs
  print 'Min score: {:0.1f}'.format(min_reward)
  print 'Max score: {:0.1f}'.format(max_reward)
  average_last_10_percent = sum(all_episode_scores[int(num_runs*.9):]) / (num_runs*.1)
  print 'Average score for last 10% of runs: {:0.1f}'.format(average_last_10_percent)

  # plot scores
  plt.plot(plotted_episodes, plot_episode_scores, 'k')
  x1, x2, _, _ = plt.axis()
  plt.axis((x1, x2, y_axis_min, y_axis_max))
  plt.ylabel("Score")
  plt.xlabel("Episode")
  plt.show()
