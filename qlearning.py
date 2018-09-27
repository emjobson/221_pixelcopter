import math, random
from collections import defaultdict

############################################################
#       Modified from the CS 221 Blackjack Assignment.     #
############################################################

# Abstract class: an RLAlgorithm performs reinforcement learning.  All it needs
# to know is the set of available actions to take.  The simulator (see
# simulate()) will call getAction() to get an action, perform the action, and
# then provide feedback (via incorporateFeedback()) to the RL algorithm, so it can adjust
# its parameters.
class RLAlgorithm:
    # Your algorithm will be asked to produce an action given a state.
    def getAction(self, state): raise NotImplementedError("Override me")

    # We will call this function when simulating an MDP, and you should update
    # parameters.
    # If |state| is a terminal state, this function will be called with (s, a,
    # 0, None). When this function is called, it indicates that taking action
    # |action| in state |state| resulted in reward |reward| and a transition to state
    # |newState|.
    def incorporateFeedback(self, state, action, reward, newState): raise NotImplementedError("Override me")

############################################################

class QLearningAlgorithm(RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2, \
        decayingExpProb=False, decay_target_it=0):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.startExpProb = explorationProb
        self.decayingExpProb = decayingExpProb
        self.decay_target_it = decay_target_it
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if self.decayingExpProb:
            self.explorationProb -= self.startExpProb / self.decay_target_it
        if random.random() < self.explorationProb:
            return random.choice(self.actions)
        else:
    #        upQ = self.getQ(state, self.actions[0])
    #        downQ = self.getQ(state, self.actions[1])
    #        if upQ <= downQ:
    #            return self.actions[1] # default now down
    #        else:
    #            return self.actions[0]

            return max((self.getQ(state, action), action) for action in self.actions)[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        if action is None or newState is None:
            return

        prediction = self.getQ(state, action)
        features = self.featureExtractor(state, action)
        possActions = self.actions
        vOpt = float("-inf")
        for newAction in possActions:
            vOpt = max(self.getQ(newState, newAction), vOpt)
        target = reward + self.discount * vOpt
        factor = self.getStepSize() * (prediction - target)

        for f, v in features:
            self.weights[f] = self.weights.get(f, 0) - factor * v

    def getWeights(self):
        return self.weights
