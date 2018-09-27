import numpy as np
from ple.games.pixelcopter import Pixelcopter
from ple import PLE

class BaselineAgent():
  """
  This is our naive agent. It will make the pixel move up if the distance to the
  floor is less than a hardcoded number (essentially if the pixel is too close to
  the ground) and do nothing otherwise. The agent does not consider the blocks in
  its decision.
  """

  def __init__(self, actions):
    self.actions = actions

  def pickAction(self, state):
    if state['player_dist_to_floor'] < 50:
      return self.actions[0]  # move up
    return self.actions[1]    # do nothing

############################################################


game = Pixelcopter(width=200, height=200)
env = PLE(game, fps=30, display_screen=False)

agent = BaselineAgent(actions=env.getActionSet())
env.init()

num_runs = 100
scores = []
min_reward = float('inf')
max_reward = float('-inf')

for i in range(num_runs):
  episode_reward = 0.0
  while not env.game_over():
    state = game.getGameState()
    action = agent.pickAction(state)
    reward = env.act(action)
    episode_reward += reward
  print 'Agent score {:0.1f} reward for episode %d.'.format(episode_reward) % i
  scores.append(episode_reward)
  min_reward = min(min_reward, episode_reward)
  max_reward = max(max_reward, episode_reward)
  env.reset_game()

print 'Average score for %d runs: {:0.1f}'.format(np.mean(scores)) % num_runs
print 'Min score: {:0.1f}'.format(min_reward)
print 'Max score: {:0.1f}'.format(max_reward)
print 'Std dev: {:0.3f}'.format(np.std(scores))
