import numpy as np
import pandas as pd
from tqdm import tqdm
# from run_and_plot_env import run_agent_in_env, test_agent
import matplotlib.pyplot as plt
# import os

# from Rl_agents import TabularQLearningAgent
# from MaskedCovargePathPlanningEnv import MaskedCovargePathPlanningEnv
# from SimpleCovargePathPlanningEnv import SimpleCovargePathPlanningEnv
# from stable_baselines3 import A2C
# from ploting import plot_graph
# import networkx as nx


def plot_rewards(rewards, number_of_steps_to_end, eps, roll):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    ax[0].margins(x=0)
    line1, = ax[0].plot(rewards, label='Reward', color='r')
    line2, = ax[0].plot(pd.Series(rewards).rolling(roll).mean(), label=f"Reward Mean after: {roll} epochs", color='orange')
    
    
    ax2 = ax[0].twinx()
    ax2.grid(False)
    # log_steps = np.log1p(number_of_steps_to_end)  # Log scale for number_of_steps_to_end
    line4, = ax2.plot(number_of_steps_to_end, label='Number_of_steps_to_end', )
    ax[0].set_ylabel("Rewards")
    ax2.set_ylabel("Number_of_steps_to_end")
    ax2.set_ylim(min(number_of_steps_to_end) * 0.9, max(number_of_steps_to_end) * 1.1)

    # ax2.set_ylim(0, 1)

    # Combine legends
    lines = [line1, line2, line4]
    labels = [line.get_label() for line in lines]
    ax[0].legend(lines, labels, loc='lower left')

    # Second plot (Epsilon)
    ax[1].plot(eps, label="Epsilon", color='r')
    ax[1].legend()
    ax[1].set_title("Epsilon")

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()




def plot_eps(eps):
    fig, ax = plt.subplots()
    ax.margins(x=0)
    
    # Plot rewards and rolling mean
    ax.plot(eps, label='Epsilon')
    ax.grid(False)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower left')  # Single legend in one location
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


int_to_act = {0 : "Move up",
        1: "Move down",
        2: "Move left",
        3: "Move right"
        }


def get_new_point_from_action(last_point, action, grid_size, env):
    new_x, new_y = last_point
    x, y = last_point
    if action == 0 and x > 0:  # Move up
        new_x -= 1
    elif action == 1 and x < grid_size - 1:  # Move down
        new_x += 1
    elif action == 2 and y > 0:  # Move left
        new_y -= 1
    elif action == 3 and y < grid_size - 1:  # Move right
        new_y += 1
    
    if new_x == x and new_y == y: # move out of bandries 
        return "move_out_of_boundry"
    elif env.grid[new_x, new_y] == 1: # move to obstyckle
        return "move_to_obs"
    else:
        return (new_x, new_y)


def get_sample_actions_Q_table(env, agent, starting_point = (0,0)):
    
    actions = []
    points = [starting_point]
    grid_size = env.grid_width
    move_out_of_boundry = 0 
    move_to_obs = 0
    
    observation, _ = env.reset()
    while True:
        action, _ = agent.get_the_best_action(observation)
        action = action.item()
        actions.append(action)

        next_point = get_new_point_from_action(points[-1], action, grid_size, env)
        
        if next_point == "move_out_of_boundry":
            print(next_point)
            move_out_of_boundry += 1
        elif next_point == "move_to_obs":
            print(next_point)
            move_to_obs +=1
        else:
            points.append(next_point)

        env.render()
        print(int_to_act[action])
        import time; time.sleep(0.15)
        observation, _, done, _, _ = env.step(action)
        if done: break

    path = [point[1] + point[0] * env.grid_width for point in points]
    
    print(f"{move_out_of_boundry=}")
    print(f"{move_to_obs=}")
    print(f"{path=}")
    
    return actions, path


def get_sample_actions(env, agent, starting_point = (0,0)):
    
    actions = []
    points = [starting_point]
    grid_size = 10
    move_out_of_boundry = 0 
    move_to_obs = 0
    
    observation, _ = env.reset()
    while True:
        action, _ = agent.predict(observation)
        action = action.item()
        actions.append(action)

        next_point = get_new_point_from_action(points[-1], action, grid_size)
        
        if next_point == "move_out_of_boundry":
            print(next_point)
            move_out_of_boundry += 1
        elif next_point == "move_to_obs":
            print(next_point)
            move_to_obs +=1
        else:
            points.append(next_point)

        env.render()
        print(int_to_act[action])
        import time; time.sleep(1)
        observation, _, done, _, _ = env.step(action)
        if done: break

    path = [point[1] + point[0] * env.grid_width for point in points]
    
    print(f"{move_out_of_boundry=}")
    print(f"{move_to_obs=}")
    print(f"{path=}")
    
    return actions, path



def learn_agent(env, agent, episodes=3000, plot=True, display=False):

    rewards = []
    eps = []
    number_of_steps_to_end = []


    if display:
        fig, ax = plt.subplots(figsize=(4, 15))  # Set the figure size
        im = ax.imshow(np.zeros_like(agent.Q), cmap='hot', aspect='auto', interpolation='nearest')
        plt.colorbar(im, ax=ax)
        text_annotations = []
        plt.title('Q-values Heatmap')

    for episode in tqdm(range(episodes)):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        numberofsteps=0
        while not done:
            numberofsteps+=1
            action, _ = agent.get_action(obs)
            # action = np.random.randint(4)

            action = action.item()
            next_obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            if display: 
                env.render()
                Q_values = agent.Q
                im.set_data(Q_values)  # Update the data of the heatmap
                ax.set_title(f'Q-values Heatmap (episode: {episode} moves: {numberofsteps}) \nexplor. rate ε: {agent.ε}')  # Update title to show episode

                for annotation in text_annotations:
                    annotation.remove()
                text_annotations.clear()
                for j in range(agent.Q.shape[1]):
                    for i in range(agent.Q.shape[0]):
                        annotation = plt.text(j, i, f'{agent.Q[i, j]:.5f}', ha='center', va='center', color='black', fontsize=8)
                        text_annotations.append(annotation)

                plt.draw()
                plt.pause(0.0001)
            eps.append(agent.ε)
            agent.process_transition(obs, action, reward, next_obs, done)
            obs = next_obs
        number_of_steps_to_end.append(numberofsteps)
        rewards.append(total_reward)
        
        
    if plot:
        plot_rewards(rewards, number_of_steps_to_end,eps, 30)
        # plot_eps(eps)
    return env, agent


def make_Q_table_plot(agent):
    
    plt.figure(figsize=(4, 15))  # Adjust the size of the figure (width=15, height=3)

    plt.imshow(agent.Q, cmap='viridis', aspect='auto', interpolation='nearest')

    for j in range(agent.Q.shape[1]):
        for i in range(agent.Q.shape[0]):
            plt.text(j, i, f'{agent.Q[i, j]:.5f}', ha='center', va='center', color='black', fontsize=8)
    
    np.save('Q_table.npy', agent.Q)

    plt.title("Q table")
    plt.savefig("Q_table.jpg")
    plt.show()



# if __name__ == "__main__":

#     env = SimpleCovargePathPlanningEnv()

#     agent = TabularQLearningAgent(number_of_action=4, 
#                                   number_of_states=100, 
#                                   γ=1, 
#                                   α=0.3, 
#                                   ε=0.7,
#                                   α_decay=0.999, 
#                                   ε_decay=0.999, 
#                                   α_min=0, 
#                                   ε_min=0)

#     env, agent = learn_agent(env, agent, episodes = 10)

#     make_Q_table_plot(agent)


#     env = SimpleCovargePathPlanningEnv(display=True)
#     actions, path = get_sample_actions_Q_table(env, agent)

#     graph = nx.Graph()
#     graph.add_nodes_from([i for i in range(100)])
#     plot_graph(graph, path)


    # env = MaskedCovargePathPlanningEnv() #, random_obstacles=False, coverage=0.2)
    # env.reset()

    # policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])])


    # model = A2C(
    #     "MlpPolicy", 
    #     env, 
    #     n_steps=10000, 
    #     verbose=1
    #     # gamma=0.999, # Determines the weight of future rewards relative to immediate rewards
    #     # ent_coef=0.9,  # Higher values encourage exploration
    #     # vf_coef=0.5, 
    #     # max_grad_norm=0.5, 
    #     # tensorboard_log="logs", 
    #     # policy_kwargs=policy_kwargs
    # )

    
    # model.learn(total_timesteps=10000)#, tb_log_name="ppo_logs", log_interval=100)

    # env = MaskedCovargePathPlanningEnv(display=True)
    # actions, path = get_sample_actions(env, model)


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
