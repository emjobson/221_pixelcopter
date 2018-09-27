import matplotlib.pyplot as plt
import qlearning
import math
import numpy as np
from ple.games.pixelcopter import Pixelcopter
from ple import PLE
import math
import collections
import datetime

"""
if state['next_gate_dist_to_player'] >= gate_distance_threshold:
  chosenFeatures = list(chosenFeatures)
  chosenFeatures[2] = math.ceil(gate_distance_threshold/4)
  chosenFeatures = tuple(chosenFeatures)
 """

# vars
y_axis_min = -7
y_axis_max = 16
num_runs = 10000
print_all_scores = True
total_vals_to_plot = 2000 # must be <= num_runs
displayScreen = False
grouped_avg_size = num_runs / total_vals_to_plot
curr_avg_group = []


# hyperparams
exploration_prob = .1
using_exp_decay = True
iteration_decay_zero = num_runs / 2
deterministic_ceiling_dist = 60
deterministic_floor_dist = 10
gate_distance_threshold = 60
last_frames_to_neg_reward = 11
final_velocities_to_save = last_frames_to_neg_reward
deterministic_vel = -.8

end_stats = collections.defaultdict(int)
neg_end_stats = collections.defaultdict(int)

# don't delete this -- early call to plt.plot prevents later crash during plot
plt.plot([])

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    chosenFeatures = tuple([
    #    math.ceil(state['player_dist_to_ceil'] / 4),
    #    math.ceil(state['player_dist_to_floor'] / 4),
        math.ceil((state['player_dist_to_ceil'] / state['player_dist_to_floor']) / 4),
        math.ceil(state['next_gate_dist_to_player'] / 4),
        math.ceil((state['next_gate_block_bottom'] - state['player_y'])/ 4),
        round(state['player_vel'] / 3, 1)
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

      # change back to 0, 1, 1
  def pickAction(self, state):
    # Deterministically pick action if too close to ceiling or floor.
    if state['player_dist_to_floor'] < deterministic_floor_dist:
      return self.actions[0]    # move up
    elif state['player_dist_to_ceil'] < deterministic_ceiling_dist:
      return self.actions[1]    # do nothing
    elif state['player_vel'] < deterministic_vel:
        return self.actions[1]
    return self.qLearning.getAction(state)

  def incorporateFeedback(self, state, action, reward, newState):
    self.qLearning.incorporateFeedback(state, action, reward, newState)

  def printWeights(self):
    print str(self.qLearning.getWeights())
    print 'num weights: %d' % len(self.qLearning.getWeights())

############################################################
if __name__ == '__main__':
  start_time = datetime.datetime.now()
  game = Pixelcopter(width=200, height=200)
  env = PLE(game, fps=30, display_screen=displayScreen)

  agent = Bot(actions=env.getActionSet())
  env.init()

  total_reward = 0.0
  min_reward = float('inf')
  max_reward = float('-inf')
  min_vel = float('inf')
  max_vel = float('-inf')

  all_episode_scores = []
  plot_episode_scores = []
  plotted_episodes = []
  all_final_velocities = []

  for i in range(num_runs):   # should run until qvalues converge
    run_velocities = []
    episode_reward = 0.0
    frames = []
    while not env.game_over():
      state = game.getGameState()
      action = agent.pickAction(state)
      reward = env.act(action)
      frames.append((state, action, reward))
      episode_reward += reward
      run_velocities.append(state['player_vel'])
      if state['player_vel'] > max_vel:
          max_vel = state['player_vel']
      if state['player_vel'] < min_vel:
          min_vel = state['player_vel']

    if episode_reward > (last_frames_to_neg_reward - 4): # otherwise, line directly below won't work
        avg_final_vel = sum(run_velocities[(len(run_velocities) - final_velocities_to_save):]) / final_velocities_to_save
        all_final_velocities.append(avg_final_vel)

    # Update q-values by negatively rewarding the last actions that led to crashing
    for f in range(len(frames) - last_frames_to_neg_reward):
      state, action, reward = frames[f]
      nextState = frames[f+1][0]
      agent.incorporateFeedback(state, action, 1, nextState)
    for f in range(max(len(frames) - last_frames_to_neg_reward, 0), len(frames)-1):
      state, action, reward = frames[f]
      nextState = frames[f+1][0]
      agent.incorporateFeedback(state, action, -1000, nextState)

    if print_all_scores and (i % 100 == 0):
#    if print_all_scores:
      print 'Agent score {:0.1f} reward for episode %d.'.format(episode_reward) % i
    # if i % (num_runs / total_vals_to_plot) is 0:
    #   plot_episode_scores.append(int(episode_reward))
    #   plotted_episodes.append(i)
    curr_avg_group.append(episode_reward)
    if len(curr_avg_group) is grouped_avg_size:
      plot_episode_scores.append(np.mean(curr_avg_group))
      plotted_episodes.append(i)
      curr_avg_group = []


    all_episode_scores.append(int(episode_reward))
    total_reward += episode_reward
    min_reward = min(min_reward, episode_reward)
    max_reward = max(max_reward, episode_reward)
    if i > num_runs * .9:   # only count end stats for last 10% of runs
      end_stats[game.getDeath()] += 1
      if episode_reward <= 0:
        neg_end_stats[game.getDeath()] += 1
    env.reset_game()

  print 'Average score for %d runs: {:0.1f}'.format(total_reward / num_runs) % num_runs
  print 'Min score: {:0.1f}'.format(min_reward)
  print 'Max score: {:0.1f}'.format(max_reward)
  average_last_10_percent = sum(all_episode_scores[int(num_runs*.9):]) / (num_runs*.1)
  print 'Average score for last 10% of runs: {:0.1f}'.format(average_last_10_percent)
  print 'Std dev for last 10% of runs: {:0.1f}'.format(np.std(all_episode_scores[int(num_runs*.9):]))

  print 'End stats:'
  for k, v in end_stats.iteritems():
    print '%s: {:0.2f}'.format(v / (.1 * num_runs)) % k
  print 'Negative end stats:'
  num_neg_end_runs = sum(neg_end_stats.values())
  for k, v in neg_end_stats.iteritems():
    print '%s: {:0.2f}'.format(1.0 * v / num_neg_end_runs) % k
  print 'Time elapsed: %s' % str(datetime.datetime.now() - start_time)

  # plot scores
  plt.plot(plotted_episodes, plot_episode_scores, 'k')
  x1, x2, _, _ = plt.axis()
  y_axis_max = math.ceil(max(plot_episode_scores) * 1.5) #max_reward
  plt.axis((x1, x2, y_axis_min, y_axis_max))
  plt.ylabel("Score")
  plt.xlabel("Episode")
  plt.show()

  print 'min_vel:', min_vel
  print 'max_vel:', max_vel
  plt.plot(all_final_velocities)
  plt.ylabel('Average velocity for final frames before crash')
  plt.xlabel('Episode')
  plt.show()
  print 'average final crash velocity:'
  print sum(all_final_velocities) / len(all_final_velocities)
