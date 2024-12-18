import numpy as np
import pandas as pd
from tqdm import tqdm
from run_and_plot_env import run_agent_in_env, test_agent
import matplotlib.pyplot as plt
import cv2 
import os
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

from Rl_agents import TabularQLearningAgent, RandomAgent, DeepQlearningAgent
from Rl_envs import SimpleCovargePathPlanningEnv, MaskedCovargePathPlanningEnv
from stable_baselines3 import A2C
from ploting import plot_graph
import networkx as nx


def plot_rewards(rewards, eps, roll):
    fig, ax = plt.subplots()
    ax.margins(x=0)
    ax.plot(rewards)
    ax.plot(pd.Series(rewards).rolling(roll).mean())
    ax2 = ax.twinx()
    ax2.plot(eps, label='epsilon', color='k')
    # ax2.plot(alpha, label='alpha', color='b')
    ax2.legend(loc='lower left')
    ax2.grid(False)
    ax2.set_ylim(0, 1)
    plt.show()


def plot_mean_reward(env, model, sample=100):
    scores = []
    for _ in range(sample):
        observation, _ = env.reset()
        score = 0
        while True:
            action, _ = model.predict(observation)
            action = action.item()
            # action = env.action_space.sample() 
            observation, reward, done, _, _ = env.step(action)
            score += reward
            if done: break
        scores.append(score)
    plt.subplots()
    plt.hist(scores, label='scores'); plt.xlabel('score')
    plt.axvline(np.mean(scores), color='k', label='mean')
    plt.legend(loc='upper right')
    print('mean score', np.mean(scores))
    plt.show()


def get_sample_action(env, model):
    actions = []

    observation, _ = env.reset()
    while True:
        action, _ = model.predict(observation)
        action = action.item()
        actions.append(action)
        observation, _, done, _, _ = env.step(action)
        if done: break
    return actions



def get_sample_path(actions):
    
    points = [(0,0)]
    grid_size = 10
    move_out_of_boundry = 0 
    move_to_obs = 0

    for action in actions:
        new_x, new_y = points[-1]
        x, y = points[-1]
        if action == 0 and x > 0:  # Move up
            new_x -= 1
        elif action == 1 and x < grid_size - 1:  # Move down
            new_x += 1
        elif action == 2 and y > 0:  # Move left
            new_y -= 1
        elif action == 3 and y < grid_size - 1:  # Move right
            new_y += 1
        
        if new_x == x and new_y == y: # move out of bandries 
            move_out_of_boundry += 1

        elif env.grid[new_x, new_y] == 1: # move to obstyckle
            move_to_obs +=1
        
        else:
            points.append((new_x, new_y))
    
    path = [point[1] + point[0] * env.grid_size for point in points]
    
    print(f"{move_out_of_boundry=}")
    print(f"{move_to_obs=}")
    print(f"{path=}")

    return path


def learn_agent(env, agent, episodes = 3000):

    rewards = []
    eps = []
    for _ in tqdm(range(episodes)):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = agent.get_action(obs)
            # action = np.random.randint(4)
            # execute this action
            action = action.item()
            n_obs, reward, done, _, _ = env.step(action)
            total_reward += reward

            agent.process_transition(obs, action, reward, n_obs, done)
            obs = n_obs

        rewards.append(total_reward)
        eps.append(agent.ε)

    plot_rewards(rewards, eps, 50)
    return env, agent


int_to_act = {0 : "Move up",
        1: "Move down",
        2: "Move left",
        3: "Move right"
        }


def get_sample_actions_Q_table(env, agent):
    actions = []

    observation, _ = env.reset()
    while True:
        action, _ = agent.get_the_best_action(observation)
        action = action.item()
        actions.append(action)
        env.render()
        print(int_to_act[action])
        import time; time.sleep(2)
        observation, _, done, _, _ = env.step(action)
        if done: break
    return actions


if __name__ == "__main__":

    env = SimpleCovargePathPlanningEnv()

    agent = TabularQLearningAgent(number_of_action=4, 
                                  number_of_states=100, 
                                  γ=1, 
                                  α=0.3, 
                                  ε=0.9,
                                  α_decay=0.999, 
                                  ε_decay=0.9999, 
                                  α_min=0, 
                                  ε_min=0)

    learn_agent(env, agent, episodes = 100)
    
    actions = get_sample_actions_Q_table(env, agent)
    path = get_sample_path(actions)

    graph = nx.Graph()
    graph.add_nodes_from([i for i in range(100)])
    plot_graph(graph, path)


    # env = MaskedCovargePathPlanningEnv() #, random_obstacles=False, coverage=0.2)
    # env.reset()

    # policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])])

    # lr_schedule = linear_schedule(0.0001)

    # model = A2C(
    #     "MultiInputPolicy", 
    #     env, 
    #     # learning_rate=0.0003, 
    #     n_steps=20, 
    #     gamma=0.999, # Determines the weight of future rewards relative to immediate rewards
    #     ent_coef=0.1,  # Higher values encourage exploration
    #     vf_coef=0.5, 
    #     max_grad_norm=0.5, 
    #     tensorboard_log="logs", 
    #     policy_kwargs=policy_kwargs
    # )

    
    # model.learn(total_timesteps=1000000000, tb_log_name="ppo_logs", log_interval=100)

    # obs, _ = env.reset()
    # done = False
    # while not done:
    #     env.render()
    #     import time ; time.sleep(2)
    #     action, _ = model.predict(obs)
    #     action = action.item()
    #     n_obs, reward, done, _, _ = env.step(action)
    #     print(F"{int_to_act[action]}, {reward=}")

    #     obs = n_obs

    # plot_mean_reward(env, model, sample=100)
    
    # actions = get_sample_action(env, model)
    # path = get_sample_path(actions)
    
    # graph = nx.Graph()
    # graph.add_nodes_from([i for i in range(100)])
    # plot_graph(graph, path)
    
    # env, agent = learn_agent(env, agent)
