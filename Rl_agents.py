from abc import ABC, abstractmethod
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F


class Agent(ABC):
    def __init__(self, env):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    
    @abstractmethod
    def process_transition(self, observation, action, reward, next_observation, done):
        pass

    @abstractmethod    
    def get_action(self, observation, learning):
        pass


class RandomAgent(Agent):
    def __init__(self, env):
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def process_transition(self, observation, action, reward, next_observation, done):
        pass
    
    def get_action(self, observation): #learning):
        return np.random.randint(self.number_of_action), {} # chose rundom action
        

    
class TabularQLearningAgent(Agent):
    def __init__(self, number_of_action, number_of_states, γ=1, α=0.3, ε=0.7, α_decay=0.999, ε_decay=0.999, α_min=0, ε_min=0):
        
        self.Q = np.zeros((number_of_states, number_of_action))
        self.number_of_action = number_of_action
        self.number_of_states = number_of_states
        self.γ = γ # The discount factor γ, which controls how much future rewards are valued compared to immediate rewards
        self.ε = ε # exploration rate, how often the agent takes a random action (exploration) versus the best-known action (exploitation).
        self.α = α # learning rate, get smaller and smaller at every episode 
        self.α_decay = α_decay # Multiplier to reduce α at every episode
        self.ε_decay = ε_decay # Multiplier to reduce ϵ at every episode
        self.α_min = α_min # A minimum value for α to ensure that updates don't stop entirely
        self.ε_min = ε_min # A minimum value for ϵ to ensure that the agent doesn’t stop exploring entirely

    def process_transition(self, observation, action, reward, next_observation, done):
        a, r, s, s_next = action, reward, observation, next_observation
        if done:
            max_next_q = 0
            # self.Q[s_next] = 0 # terminal states have no future rewards, 
            self.ε = max(self.ε_min, self.ε * self.ε_decay) # Multiplier to reduce ϵ at every episode
            self.α = max(self.α_min, self.α * self.α_decay)
        else:
            max_next_q = np.max(self.Q[s_next])
        
        # Q-learning update equation
        self.Q[s][a] += self.α * (r + self.γ * max_next_q - self.Q[s][a])
        # temporal difference error. It measures how much the agent's current estimate of 
        #  𝑄(𝑠,𝑎)Q(s,a) differs from the observed reward and future predictions.
        
    
    def get_action(self, observation): #learning):
        if np.random.rand() < self.ε:
            return np.array([np.random.randint(self.number_of_action)]), {} # chose rundom action
        
        a = np.argmax(self.Q[observation])
        return np.array([a]), {}
    
    
    def get_the_best_action(self, observation):
        a = np.argmax(self.Q[observation])
        return np.array([a]), {}
    




class Qnn(nn.Module):

    def __init__(self, n_input, n_output):
        super(Qnn, self).__init__()

        self.l1 = nn.Linear(n_input, 120)
        self.l2 = nn.Linear(120, 84)
        self.l3 = nn.Linear(84, n_output)

    def forward(self, input):
        x = torch.relu(self.l1(input))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        return x 
        

class DeepQlearningAgent(Agent):
    def __init__(self, number_of_action, number_of_states, γ=1, α=0.3, ε=0.7, α_decay=0.999, ε_decay=0.999, α_min=0, ε_min=0):
        
        self.qnn = Qnn(n_input=number_of_states, n_output=number_of_action)
        self.number_of_action = number_of_action
        self.number_of_states = number_of_states

        self.criterion = nn.MSELoss()

        self.ε = ε
        self.γ = γ # The discount factor γ, which controls how much future rewards are valued compared to immediate rewards
        self.optim = torch.optim.Adam(self.qnn.parameters())

    def process_transition(self, observation, action, reward, next_observation, done):
        a, r, s, s_next = action, reward, observation, next_observation


        target_q = r + (self.γ * self.qnn(F.one_hot(torch.tensor([s_next]), num_classes = self.number_of_states).squeeze(0).float()).max()) * ~done
        
        predicted_q = self.qnn(F.one_hot(torch.tensor([s]), num_classes = self.number_of_states).squeeze(0).float())[a]

        loss = self.criterion(target_q, predicted_q)
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        # Q-learning update equation
        # self.Q[s][a] += self.α * (r + self.γ * max_next_q - self.Q[s][a])
        # temporal difference error. It measures how much the agent's current estimate of 
        #  𝑄(𝑠,𝑎)Q(s,a) differs from the observed reward and future predictions.

        
    
    def get_action(self, observation): #learning):
        # 3 if learning and 
        if np.random.rand() < self.ε:
            return np.random.randint(self.number_of_action) # chose rundom action
        
        with torch.no_grad():
            o = torch.tensor([observation])
            s = F.one_hot(o, num_classes=self.number_of_states).squeeze(0)
            a = self.qnn(s.float()).argmax()
            return a

