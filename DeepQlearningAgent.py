
from Rl_agents import Agent
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import random


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory():
    def __init__(self, maxlen):
        self.queue = deque([], maxlen=maxlen)

    def append(self, *elements):
        self.queue.append(Transition(*elements))

    def sample(self, size):
        return random.sample(self.queue, size)
    
    def get_size(self):
        return len(self.queue)


class Qnn(nn.Module):

    def __init__(self, n_input, n_output, hidden=[256, 64, 16]):
        super(Qnn, self).__init__()

        l = [n_input, *hidden, n_output]
        n = len(l)
        self.NN = nn.Sequential()
        
        for i in range(1, n):
            self.NN.add_module(f"Linear{i}", nn.Linear(l[i-1], l[i]))
            if i < n - 1:
                self.NN.add_module(f'Activation{i}', nn.ReLU())


    def forward(self, input):
        return self.NN(input)
        


class DeepQlearningAgent(Agent):
    def __init__(self, 
                 number_of_action, 
                 number_of_states, 
                 γ=1, α=1e-3, ε=0.7, 
                 α_decay=0.999, ε_decay=0.999, 
                 α_min=0, ε_min=0,
                 hidden=[256, 64, 16],
                 sync_freq=20,freezing=True, 
                 double_dqn=False,
                 is_replayMemory=True, replayMemory_size=100,
                 mini_batch_size=16):
        
        self.is_replayMemory = is_replayMemory
        self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else
                    "mps" if torch.backends.mps.is_available() else
                    "cpu"
                )
        print(f"{self.device=}")

        self.politicy_qnn = Qnn(n_input=number_of_states, n_output=number_of_action, hidden=hidden).to(self.device)
        
        # Create the target network and make it identical to the policy network
        if double_dqn:
            self.target_qnn = Qnn(n_input=number_of_states, n_output=number_of_action, hidden=hidden).to(self.device)
            # self.target_qnn.load_state_dict(self.politicy_qnn.state_dict())
        # else:
        #     self.target_qnn = self.politicy_qnn


        self.optim = torch.optim.AdamW(self.politicy_qnn.parameters(), lr=α)

        self.number_of_action = number_of_action
        self.number_of_states = number_of_states

        if self.is_replayMemory:
            self.replayMemory = ReplayMemory(replayMemory_size)
            self.mini_batch_size = mini_batch_size
        
        self.criterion = nn.MSELoss() #nn.SmoothL1Loss()

        self.double_dqn = double_dqn
        self.ε_decay = ε_decay
        self.ε = ε
        self.ε_min = ε_min
        self.γ = γ # The discount factor γ, which controls how much future rewards are valued compared to immediate rewards


    def process_transition(self, state, action, reward, next_state, done):
        
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        

        if self.is_replayMemory:
            self.replayMemory.append(state, action, reward, next_state, done)
            if self.replayMemory.get_size() <= self.mini_batch_size:
                return
        
            transitions = self.replayMemory.sample(self.mini_batch_size)
            batch = Transition(*zip(*transitions))
            state_batch = torch.stack(batch.state)
            next_state_batch = torch.stack(batch.next_state)
            action_batch = torch.tensor(batch.action, device=self.device) # dtype=torch.float32
            reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)

        else:
            # bach is only one demension and we assignet as a batch 
            batch = Transition(state, action, reward, next_state, done)

            state_batch = batch.state.unsqueeze(0)
            next_state_batch = batch.next_state.unsqueeze(0)
            action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(0) # dtype=torch.float32
            reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32).unsqueeze(0)

        if self.double_dqn:
            if done:
                target_q_value = reward_batch  # If done, no future reward, just the immediate reward
                self.ε = max(self.ε_min, self.ε * self.ε_decay)
            else:
                best_actions_from_policy = self.politicy_qnn(next_state_batch).argmax(1)
                target_q_value = reward_batch + self.γ * self.target_qnn(next_state_batch).gather(1, best_actions_from_policy.unsqueeze(1)).squeeze(1)
        else:
            if done:
                target_q_value = reward_batch  # If done, no future reward, just the immediate reward
                self.ε = max(self.ε_min, self.ε * self.ε_decay)
            
            else:
                q = self.politicy_qnn(next_state_batch).max(1).values
                target_q_value = reward_batch + self.γ * q
            


        q_values = self.politicy_qnn(state_batch)
        # selected_q_values = q_values[torch.tensor([i for i in range(4)]), action_batch]
        current_q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        loss = self.criterion(current_q_value, target_q_value)
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
    
    def get_action(self, observation): #learning):
        # 3 if learning and 
        if np.random.rand() < self.ε:
            return np.random.randint(self.number_of_action), {} # chose rundom action
        
        with torch.no_grad():
            o = torch.tensor(observation, dtype=torch.float32, device=self.device)
            # self.politicy_qnn.eval()
            a = self.politicy_qnn(o).argmax()
            # self.politicy_qnn.train()  # Switch to training mode
            return a.item(), {}

