import numpy as np
import matplotlib.pyplot as plt
import time
import math

# TestBed
class Testbed(object):
    # Initialize
    def __init__(self, nArms, mean, var):
        self.nArms = nArms              # Number of bandit
        # Used for normal distribution
        self.mean = mean
        self.var = var

        self.actArr = np.zeros(nArms)   # action values array
        self.optim = 0                  # optimal value for optimistic greedy
        self.reset()

    # Reset for next iteration
    def reset(self):
        # Set random value for action-value array
        self.actArr = np.random.normal(self.mean, self.var, self.nArms)
        # Choose the max value in action array
        self.optim = np.argmax(self.actArr)


# Learning Agent
class Agent(object):
    # Initialize
    def __init__(self,nArms, eProb):
        self.nArms = nArms
        self.eProb = eProb   # Epsilon probability

        self.timeStep = 0
        self.lastAction = None

        self.kAction = np.zeros(nArms)         # count of actions taken at time t
        self.rSum = np.zeros(nArms)            # Sums of rewards
        self.valEstimates = np.zeros(nArms)    # action-value estimates


    # Policy action
    def action(self):
        # Policy
        if self.eProb == 0 and self.timeStep == 0:
            self.valEstimates[:] = 5  # Greedy Q1(a)=5

        # Epsilon method
        randProb = np.random.random()  # random probability between 0-1
        if randProb < self.eProb:
            # Select random action
            a = np.random.choice(len(self.valEstimates))

        # Greedy Method
        else:
            # Find max value estimate
            maxAction = np.argmax(self.valEstimates)
            #if self.timeStep in range(2):
            #    print(self.valEstimates)
            #    print(maxAction)
            # array contain actions with max (only)
            action = np.where(self.valEstimates == np.argmax(self.valEstimates))[0]

            if len(action) == 0:
                a = maxAction
            # actions contain the same value, randomly select an action
            else:
                a = np.random.choice(action)

        self.lastAction = a
        return a


    # Update
    def update(self, reward):
        # Add 1 to the number of action taken in step
        At = self.lastAction
        #At = np.argmax(self.lastAction + 2 * ((math.log(self.timeStep / self.kAction[self.lastAction])) ** 0.5))
        # Add 1 to action selection
        self.kAction[At] += 1
        # Add reward to sum array
        self.rSum[At] += reward
        # Update action-value
        self.valEstimates[At] = self.rSum[At]/self.kAction[At]
        # Increase time step
        self.timeStep += 1

    def UBCupdate(self,reward):
        # Add 1 to the number of action taken in step
        if self.kAction[self.lastAction] == 0:
            T = 0
        else:
            T = math.log(self.timeStep) / self.kAction[self.lastAction]yyyy
        explore = 2 * math.sqrt(T)
        At = np.argmax(self.valEstimates[self.lastAction] + explore)
        # Add 1 to action selection
        self.kAction[At] += 1
        # Add reward to sum array
        self.rSum[At] += reward
        # Update action-value
        self.valEstimates[At] = self.rSum[At] / self.kAction[At]
        # Increase time step
        self.timeStep += 1


    # Reset for next iteration
    def reset(self):
        self.timeStep = 0
        self.lastAction = None
        self.kAction[:] = 0
        self.rSum[:] = 0
        self.valEstimates[:] = 0


# set running environment
class Environment(object):
    # Initialize
    def __init__(self, testbed, agents, steps, iterations):
        self.testbed = testbed
        self.agents = agents
        self.steps = steps
        self.iterations = iterations

    # Run Test
    def run(self):
        # Storing scores
        scoreArr = np.zeros((self.steps, len(self.agents)))
        # Optimal count
        optimalArr = np.zeros((self.steps, len(self.agents)))

        # Iterations
        for iIter in range(self.iterations):
            # Running progress
            if (iIter%100) == 0:
                print("Completed Iterations: ",iIter)

            # Reset
            self.testbed.reset()
            for agent in self.agents:
                agent.reset()

            for nSteps in range(self.steps):
                agtCnt = 0

                for nAgent in self.agents:
                    action = nAgent.action()
                    # Reward: normal distribution (q*(at), variance=1)
                    reward = np.random.normal(self.testbed.actArr[action], scale=1)
                    # Agent checks state
                    if self.agents == u_agents:
                        nAgent.UBCupdate(reward=reward)
                        #print("u {}".format(nAgent))
                    else:
                        nAgent.update(reward=reward)
                    # Add score in arrary
                    scoreArr[nSteps,agtCnt] += reward
                    # Check if optimal and add to array
                    if action == self.testbed.optim:
                        optimalArr[nSteps,agtCnt] += 1
                    agtCnt += 1

        # return averages
        scoreAvg = scoreArr/self.iterations
        optimlAvg = optimalArr/self.iterations

        return scoreAvg, optimlAvg



if __name__ == "__main__":
    # hyperparameter
    start_time = time.time()
    nArms = 10
    iterations = 2000
    steps = 1000

    # Setup objects to contain information about the agents, testbed, and environment
    print("Environment setup")
    testbed = Testbed(nArms=nArms,mean=0,var=1)
    agents = [Agent(nArms=nArms, eProb=0)]     # greedy
    e_agents = [Agent(nArms=nArms,eProb=0.1)]  # epsilon
    u_agents = [Agent(nArms=nArms,eProb=0)]    # ubc
    environment = Environment(testbed=testbed,agents=agents,steps=steps,iterations=iterations)
    environment1 = Environment(testbed=testbed, agents=e_agents, steps=steps, iterations=iterations)
    environment2 = Environment(testbed=testbed, agents=u_agents, steps=steps, iterations=iterations)

    # Run Environment
    print("Start running")
    g_Scores, g_Optimal = environment.run()
    e_Scores, e_Optimal = environment1.run()
    u_Scores, u_Optimal = environment2.run()
    print("Execution time: %s seconds" % (time.time() - start_time))
    #(np.size(g_Scores))
    #print(g_Scores)

    # Optimal selections over all steps
    plt.title("10-Armed TestBed - % Optimal Action")
    plt.plot(g_Optimal * 100)
    plt.plot(e_Optimal * 100)
    plt.ylim(0, 100)
    plt.ylabel('% Optimal Action')
    plt.xlabel('Steps')
    plt.legend(["optimistic greedy Q1=5 ɛ=0","Realistic ɛ-greedy Q1=0 ɛ=0.1"], loc=4)
    plt.show()

    # Average rewards over all steps
    plt.title("10-Armed TestBed - Average Rewards")
    plt.plot(u_Scores)
    plt.plot(e_Scores)
    plt.ylabel('Average Reward')
    plt.xlabel('Steps')
    plt.legend(["UBC c=2","ɛ-greedy ɛ=0.1"], loc=4)
    plt.show()

