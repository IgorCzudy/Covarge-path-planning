from ploting import plot_graph_two_agents, plot_matrix
from Rl_run import initialdouble_q_heatmap, updatedouble_q_heatmap
import pandas as pd 
from Rl_run import make_Q_table_plot, get_new_point_from_action
import pygame
import matplotlib.pyplot as plt
import numpy as np


int_to_act = {0: "Move up", 1: "Move down", 2: "Move left", 3: "Move right"}


def plot_double_q_table(agent1, agent2):

    fig, axes = plt.subplots(ncols=2, figsize=(8, 8))

    ax1, ax2 = axes

    im1 = ax1.imshow(
        agent1.Q, cmap="Blues", aspect="auto", interpolation="nearest"
    )

    im2 = ax2.imshow(
        agent2.Q, cmap="Blues", aspect="auto", interpolation="nearest"
    )
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)

    for agent, ax, text_annotations in zip(
        [agent1, agent2], [ax1, ax2], [[], []]
    ):
        for j in range(agent.Q.shape[1]):
            for i in range(agent.Q.shape[0]):
                annotation = ax.text(
                    j,
                    i,
                    f"{agent.Q[i, j]:.5f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )
                text_annotations.append(annotation)


    action_names = ["Move up", "Move down", "Move left", "Move right"]
    
    for ax, title in zip([ax1, ax2], ["Q-values agent 1 Heatmap", "Q-values agent 2 Heatmap"]):
        ax.set_xticks(np.arange(len(action_names)))
        ax.set_xticklabels(action_names, rotation=45, ha="right")
        ax.set_title(title)

    plt.show()

def format_title(agent, reward, action, episode):
    return (f"Q-values Heatmap (episode: {episode}\n"
            f"explor. rate ε: {agent.ε:.2f},\n"
            f"received reward: {reward},\n"
            f"action: {action})")


def plot_heatmap(agent1, agent2, q_heatmap1, q_heatmap2, text_annotations1, text_annotations2, title1, title2):
    
    updatedouble_q_heatmap(
        agent1, agent2,
        q_heatmap1, q_heatmap2,
        text_annotations1, text_annotations2,
        title1,
        title2
    )
    
    wait_for_keypress("n")

def wait_for_keypress(key="n"):
    while True:
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.KEYDOWN and event.unicode == key:
            print(f"Key '{key}' pressed! Moving to the next iteration.")
            break


def plot_reward(rewards1, rewards2, epsilons):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    rolling_mean1 = pd.Series(rewards1).rolling(50).mean()
    rolling_mean2 = pd.Series(rewards2).rolling(50).mean()
    
    ax1.plot(rewards1, alpha=0.3)
    ax1.plot(np.arange(len(rolling_mean1)), rolling_mean1, label="Reward of Agent 1")
    ax1.plot(np.arange(len(rolling_mean2)), rolling_mean2, label="Reward of Agent 2")
    ax1.plot(rewards2, alpha=0.3)
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Rewards')
    ax1.legend()
    
    ax2.plot(np.arange(len(epsilons)), epsilons, label='Epsilon', color='orange')
    ax2.set_xlabel('Moves')
    ax2.set_ylabel('Epsilon')
    ax2.legend()
    
    plt.subplots_adjust(hspace=0.6)
    plt.show()



def plot_graph(env, actions1, actions2, starting_point1_tuple=(0, 0), starting_point1_int=0, starting_point2_tuple=(4, 4), starting_point2_int=24):
    
    def compute_points(actions, start_tuple, start_int):
        points = [start_tuple]
        int_points = [start_int]
        for action in actions:
            new_point = get_new_point_from_action(points[-1], action, env.grid_width, env)
            if new_point not in ("move_out_of_boundry", "move_to_obs"):
                points.append(new_point)
                int_points.append(new_point[0] * env.grid_width + new_point[1])
        return points, int_points
    
    points1, int_points1 = compute_points(actions1, starting_point1_tuple, starting_point1_int)
    points2, int_points2 = compute_points(actions2, starting_point2_tuple, starting_point2_int)
    print(f"{points1=} || {points2=}")
    plot_graph_two_agents(int_points1, int_points2)


def run_learned(env, agent1, agent2):

    obs1, obs2, _ = env.reset()
    done = False
    actions1, actions2 = [], []
    int_to_act = {0: "Move up", 1: "Move down", 2: "Move left", 3: "Move right"}

    while not done:
        action1 = agent1.get_action(obs1)[0].item()
        action2 = agent2.get_action(obs2)[0].item()
        actions1.append(action1)
        actions2.append(action2)

        env.render()

        print(f"agent1: {int_to_act[action1]}, agent2: {int_to_act[action2]}")
        print(f"obs1: {obs1}, obs2: {obs2}")
        print("Press any key to continue")
        wait_for_keypress("n")

        next_obs1, next_obs2, reward1, reward2, done, _, _ = env.step(action1, action2)
        print(f"{reward1=}, {reward2=}")

        agent1.process_transition(obs1, action1, reward1, next_obs1, done)
        agent2.process_transition(obs2, action2, reward2, next_obs2, done)

        obs1, obs2 = next_obs1, next_obs2

    return actions1, actions2
