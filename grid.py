import numpy as np
import matplotlib.pyplot as plt
import random

# hyperparameter
gridSize = 4
gamma = 1            # discounting rate
initialReward = -1   # reward
nIterations = 20
# up, down, right, left
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
terminationStates = [[0,0], [gridSize-1, gridSize-1]]

# action reward
def actionReward(initialPosition, action):
    # Termination state not allow any actions
    if initialPosition in terminationStates:
        return initialPosition, 0

    reward = initialReward
    # Position after one iteration
    finalPosition = np.array(initialPosition) + np.array(action)
    # Position exceed grid size
    if -1 in finalPosition or 4 in finalPosition:
        finalPosition = initialPosition

    return finalPosition, reward


# Set initial gridworld
value = np.zeros((gridSize, gridSize))
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]
# Initial state value
print("Iteration 0:")
print(value)
print("")


# Iteration
for it in range(nIterations):
    Value = np.copy(value)
    # Counting state-value function and add to state value
    for state in states:
        stateValueFunc = 0
        for action in actions:
            finalPosition, reward = actionReward(state, action)
            # v(s) = prob*(r+gamma*v(s-1))
            stateValueFunc += (1/len(actions))*(reward+(gamma*value[finalPosition[0], finalPosition[1]]))
        Value[state[0], state[1]] = stateValueFunc
    value = Value
    # print result
    if it in range(nIterations):
        print("Iteration {}:".format(it+1))
        print(value)
        print("")