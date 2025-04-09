import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pygame
import gymnasium as gym
from typing import Optional, List, Dict, Tuple
from Rl_agents import Agent
from torch.utils.tensorboard import SummaryWriter


int_to_act = {0: "Move up", 1: "Move down", 2: "Move left", 3: "Move right"}


def plot_rewards(rewards, steps_to_end, eps, roll):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    ax[0].margins(x=0)
    ax[0].plot(rewards, label="Reward", color="orange")
    ax[0].plot(
        pd.Series(rewards).rolling(roll).mean(),
        label=f"Reward Mean after: {roll} epochs",
        color="r",
    )
    ax[0].set_xlabel("Number of episodes")
    ax[0].set_ylabel("Rewards")

    ax2 = ax[0].twinx()
    ax2.plot(steps_to_end, label="Number of steps of episode", color="b")
    ax2.set_ylabel("Number of steps of episode")
    ax2.set_ylim(min(steps_to_end) * 0.9, max(steps_to_end) * 1.1)

    lines = ax[0].get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax[0].legend(lines, labels, loc="lower left")

    ax[1].plot(eps, label="Epsilon", color="g")
    ax[1].set_xlabel("Number of steps")
    ax[1].set_ylabel("Epsilon")
    ax[1].legend()
    ax[1].set_title("Epsilon over Time")

    plt.tight_layout()
    plt.show()


def update_q_heatmpa(agent, im, text_annotations, title):

    im.set_data(agent.Q)
    plt.title(
        title,
    )

    for annotation in text_annotations:
        annotation.remove()
    text_annotations.clear()

    for j in range(agent.Q.shape[1]):
        for i in range(agent.Q.shape[0]):
            annotation = plt.text(
                j,
                i,
                f"{agent.Q[i, j]:.5f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
            text_annotations.append(annotation)

    plt.draw()
    plt.pause(0.1)


def initial_q_heatmap(agent):
    fig, ax = plt.subplots(figsize=(4, 15))

    im = ax.imshow(
        np.zeros_like(agent.Q), cmap="hot", aspect="auto", interpolation="nearest"
    )
    plt.colorbar(im, ax=ax)

    action_names = ["Move up", "Move down", "Move left", "Move right"]
    ax.set_xticks(np.arange(len(action_names)))
    ax.set_xticklabels(action_names, rotation=45, ha="right")

    ax.set_title("Q-values Heatmap")

    plt.ion()
    plt.show(block=False)
    plt.pause(0.1)

    return im, []


def initialdouble_q_heatmap(agent1, agent2):
    fig, axes = plt.subplots(ncols=2, figsize=(8, 8))

    ax1, ax2 = axes

    im1 = ax1.imshow(
        np.zeros_like(agent1.Q), cmap="Blues", aspect="auto", interpolation="nearest"
    )

    im2 = ax2.imshow(
        np.zeros_like(agent2.Q), cmap="Blues", aspect="auto", interpolation="nearest"
    )
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)

    action_names = ["Move up", "Move down", "Move left", "Move right"]
    
    for ax, title in zip([ax1, ax2], ["Q-values agent 1 Heatmap", "Q-values agent 2 Heatmap"]):
        ax.set_xticks(np.arange(len(action_names)))
        ax.set_xticklabels(action_names, rotation=45, ha="right")
        ax.set_title(title)

    plt.ion()
    plt.show(block=False)
    plt.pause(0.1)

    return im1, im2, [], []


def updatedouble_q_heatmap(agent1, agent2, im1, im2, text_annotations1, text_annotations2, title1, title2):
    # Update heatmaps
    im1.set_data(agent1.Q)
    im2.set_data(agent2.Q)

    ax1 = im1.axes
    ax2 = im2.axes
    ax1.set_title(title1)
    ax2.set_title(title2)

    # Clear old annotations
    for text_annotations in [text_annotations1, text_annotations2]:
        for annotation in text_annotations:
            annotation.remove()
        text_annotations.clear()

    # Add new text annotations to each heatmap
    for agent, ax, text_annotations in zip(
        [agent1, agent2], [ax1, ax2], [text_annotations1, text_annotations2]
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

    # Redraw only the affected canvas
    im1.figure.canvas.draw()
    im1.figure.canvas.flush_events()



def pygame_waiting() -> None:
    print("Press any key to continue")
    while True:  # waiting for button press
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.KEYDOWN:
            print("Key 'n' pressed! Moving to the next iteration.")
            break


def learn_agent(
    env: gym.Env,
    agent: Agent,
    writer: Optional[SummaryWriter] = None,
    episodes: int = 100,
    plot: bool = True,
    display_qtable: bool = False,
    display_pygame: bool = False,
    change_starting_point: bool = False,
    debug_mode: bool = False,
) -> Tuple[gym.Env, Agent]:

    i_eps: int = 0
    reward: bool = None
    rewards: List[int] = []
    eps: List[int] = []
    steps_to_end: List[int] = []

    if display_qtable:
        im, text_annotations = initial_q_heatmap(agent)

    for episode in tqdm(range(episodes)):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        numberofsteps = 0
        while not done:
            numberofsteps += 1

            action, _ = agent.get_action(obs)
            action = action.item()

            if display_qtable:
                if env.action_space.n  <= 4:
                    a = int_to_act[action]
                else:
                    first_agent_action = action % 4
                    secend_agent_action = action // 4
                    a = (int_to_act[first_agent_action], int_to_act[secend_agent_action])

                update_q_heatmpa(
                    agent,
                    im,
                    text_annotations,
                    title=f"""Q-values Heatmap (episode: {episode} moves: {numberofsteps})\n explor. rate ε: {agent.ε:.2f},\n received reword: {reward},\n action: {a}""",
                )
            if display_pygame:
                env.render()
            if debug_mode:
                pygame_waiting()

                if env.action_space.n == 16:
                    first_agent_action = action % 4
                    secend_agent_action = action // 4
                    print(f"Received reword: {reward}")
                    print(f"First agent action: {int_to_act[first_agent_action]}")
                    print(f"Secend agent action: {int_to_act[secend_agent_action]}")

            next_obs, reward, done, _, _ = env.step(action)
            total_reward += reward

            agent.process_transition(obs, action, reward, next_obs, done)
            obs = next_obs

            eps.append(agent.ε)
            if writer is not None:
                writer.add_scalar("eps", agent.ε, i_eps)
            i_eps += 1

        if change_starting_point:
            env.change_starting_point()

        if writer is not None:
            writer.add_scalar("Number of step in one epoch", numberofsteps, episode)
            writer.add_scalar("total reward", total_reward, episode)

        steps_to_end.append(numberofsteps)
        rewards.append(total_reward)

    if plot:
        plot_rewards(rewards, steps_to_end, eps, 30)

    return env, agent


def get_new_point_from_action(last_point: Tuple[int, int], action: int, grid_size: int, env: gym.Env):
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
    elif env.grid[new_x, new_y] == 1:  # move to obstyckle
        return "move_to_obs"
    else:
        return (new_x, new_y)


def get_sample_actions_Q_table(
    env: gym.Env,
    agent: Agent,
    starting_point: Tuple[int, int] = (0, 0),
    process_transition: bool = True,
) -> Tuple[List[int], List[int]]:

    actions: List[int] = []
    points: List[Tuple[int, int]] = [starting_point]
    grid_size: int = env.grid_width
    move_out_of_boundry: int = 0
    move_to_obs: int = 0

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
            move_to_obs += 1
        else:
            points.append(next_point)

        env.render()
        print(int_to_act[action])
        import time; time.sleep(0.15)

        next_obs, reward, done, _, _ = env.step(action)
        if process_transition:
            agent.process_transition(observation, action, reward, next_obs, done)
        observation = next_obs

        # observation, _, done, _, _ = env.step(action)
        if done:
            break

    path: List[int] = [point[1] + point[0] * env.grid_width for point in points]

    print(f"{move_out_of_boundry=}")
    print(f"{move_to_obs=}")
    print(f"{path=}")

    return actions, path


def get_sample_actions_Q_table_two_agents(
    env: gym.Env,
    agent: Agent,
    firs_starting_point: Tuple[int, int] = (0, 0),
    secend_starting_point: Tuple[int, int] = (4, 4),
    process_transition: bool = True,
) -> Tuple[List[int], List[int]]:

    first_actions: List[int] = []
    secend_actions: List[int] = []
    first_agent_points: List[Tuple[int, int]] = [firs_starting_point]
    secend_agent_points: List[Tuple[int, int]] = [secend_starting_point]

    grid_size: int = env.grid_width
    move_out_of_boundry: int = 0
    move_to_obs: int = 0

    observation, _ = env.reset()
    while True:
        action, _ = agent.get_the_best_action(observation)
        action = action.item()

        first_agent_action = action % 4
        secend_agent_action = action // 4
        first_actions.append(first_agent_action)
        secend_actions.append(secend_agent_action)
        # 0: move up, 1: move down, 2: move left, 3: move right
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        for i, (agent_position, a) in enumerate(
            [
                (first_agent_points[-1], first_actions[-1]),
                (secend_agent_points[-1], secend_actions[-1]),
            ]
        ):
            x, y = agent_position
            dx, dy = moves[a]
            new_x, new_y = x + dx, y + dy
            if not (
                0 <= new_x < grid_size and 0 <= new_y < grid_size
            ):  # move out of bandry
                print("move_out_of_boundry")
                move_out_of_boundry += 1

            elif env.grid[new_x, new_y] == 1:  # Obstacle
                print("move_to_obs")
                move_to_obs += 1

            else:
                if i == 0:
                    first_agent_points.append((new_x, new_y))
                else:
                    secend_agent_points.append((new_x, new_y))

        env.render()
        import time

        time.sleep(0.15)

        next_obs, reward, done, _, _ = env.step(action)
        if process_transition:
            agent.process_transition(observation, action, reward, next_obs, done)
        observation = next_obs
        if done:
            break

    first_agent_path: List[int] = [
        point[1] + point[0] * env.grid_width for point in first_agent_points
    ]
    secend_agent_path: List[int] = [
        point[1] + point[0] * env.grid_width for point in secend_agent_points
    ]

    print(f"{move_out_of_boundry=}")
    print(f"{move_to_obs=}")
    print(f"{first_agent_path=}")
    print(f"{secend_agent_path=}")

    return first_agent_path, secend_agent_path


def get_sample_actions(
    env: gym.Env, agent: Agent, starting_point: Tuple[int, int] = (0, 0)
) -> tuple[List[int], List[int]]:

    actions: List[int] = []
    points: List[Tuple[int, int]] = [starting_point]
    grid_size: int = 10
    move_out_of_boundry: int = 0
    move_to_obs: int = 0

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
            move_to_obs += 1
        else:
            points.append(next_point)

        env.render()
        print(int_to_act[action])

        import time

        time.sleep(1)

        observation, _, done, _, _ = env.step(action)
        if done:
            break

    path = [point[1] + point[0] * env.grid_width for point in points]

    print(f"{move_out_of_boundry=}")
    print(f"{move_to_obs=}")
    print(f"{path=}")

    return actions, path


def make_Q_table_plot(agent: Agent, save: bool = False):

    plt.figure(figsize=(4, 15))  # Adjust the size of the figure (width=15, height=3)

    plt.imshow(agent.Q, cmap="viridis", aspect="auto", interpolation="nearest")

    for j in range(agent.Q.shape[1]):
        for i in range(agent.Q.shape[0]):
            plt.text(
                j,
                i,
                f"{agent.Q[i, j]:.5f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    if save:
        np.save("Q_table.npy", agent.Q)
        plt.savefig("Q_table.jpg")

    plt.title("Q table")
    plt.show()


def plot_mean_reward(env: gym.Env, agent: Agent, sample: int = 100) -> None:
    scores: List[int] = []

    for _ in range(sample):
        observation, _ = env.reset()
        score = 0
        while True:
            action, _ = agent.predict(observation)
            action = action.item()
            observation, reward, done, _, _ = env.step(action)
            score += reward
            if done:
                break
        scores.append(score)
    plt.subplots()
    plt.hist(scores, label="scores")
    plt.xlabel("score")
    plt.axvline(np.mean(scores), color="k", label="mean")
    plt.legend(loc="upper right")
    print("mean score", np.mean(scores))
    plt.show()
