# Reminder to install dependencies:
# pip install pyobjc-core
# pip install pyobjc-framework-Quartz
# pip install keyboard
# To run: sudo python runHuman.py

import numpy as np
from ple.games.pixelcopter import Pixelcopter
from ple import PLE
import keyboard
import time

class Human():
  """
  This interprets the human player's keyboard actions. If the user presses 
  the space bar, the pixel will move up.
  """

  def __init__(self, actions):
    self.actions = actions

  def pickAction(self, state):
    if keyboard.is_pressed('space'):
      return self.actions[0]  # move up
    return self.actions[1]    # do nothing

############################################################

game = Pixelcopter(width=200, height=200)
env = PLE(game, fps=70, display_screen=True)

agent = Human(actions=env.getActionSet())
env.init()

scores = []
min_reward = float('inf')
max_reward = float('-inf')

while True:
  episode_reward = 0.0
  print 'Press s to start'
  while not keyboard.is_pressed('s'):
    continue

  while not env.game_over():
    state = game.getGameState()
    action = agent.pickAction(state)
    reward = env.act(action)
    episode_reward += reward
  print 'Player score: {:0.1f}.'.format(episode_reward) 
  scores.append(episode_reward)
  min_reward = min(min_reward, episode_reward)
  max_reward = max(max_reward, episode_reward)

  answer = raw_input('Keep playing ("y" to continue)? ')
  if answer is not 'y':
    break
  env.reset_game()

print 'Average score for %d runs: {:0.1f}'.format(np.mean(scores)) % len(scores)
print 'Min score: {:0.1f}'.format(min_reward)
print 'Max score: {:0.1f}'.format(max_reward)
print 'Std dev: {:0.3f}'.format(np.std(scores))
