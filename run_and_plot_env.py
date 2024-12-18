import matplotlib.pylab as plt
import numpy as np 
from IPython import display
import pandas as pd 


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
    # display.display(plt.gcf())
    # display.clear_output(wait=True)
    
def run_agent_in_env(env, agent, episodes, learning=False, plot=False, plot_interval=1000, target=None, target_window=None):
    rewards = []
    eps = []
    roll = target_window or 50
    for episode in range(episodes):
        observation = env.reset()
        total_reward = 0
        done = False
        while not done :
            # Zapytajmy agenta o akcje dla aktualnego stanu
            action = agent.get_action(observation, learning)
            
            # Wykonajmy akcje
            next_observation, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Jeśli się uczymy, przekażmy przejście do agenta
            if learning:
                agent.process_transition(observation, action, reward, next_observation, done)
            
            observation = next_observation
        rewards.append(total_reward)
        eps.append(agent.ε)
        
        # Wyświetl na wykresie nagrody otrzymane po kolei w epizodach
        if plot and episode % plot_interval == 0:
            plot_rewards(rewards, eps, roll)
            
        if plot and  target is not None and len(rewards) > target_window and np.mean(rewards[-target_window:]) > target:
            print('early stopping...')
            plot_rewards(rewards, eps, roll)
            break
    return rewards    

def test_agent(env, agent, epochs, render=False):
    import seaborn as sns
    render = render or epochs <= 5
    scores = []
    for _ in range(epochs):
        observation = env.reset()
        score = 0
        for _ in range(500):
            if render: env.render()
            action = agent.get_action(observation, learning=False)
            observation, reward, done, _ = env.step(action)
            score += reward
            if done: break
        scores.append(score)
        if render: env.close()
    plt.subplots()
    plt.hist(scores, label='scores'); plt.xlabel('score')
    plt.axvline(np.mean(scores), color='k', label='mean')
    plt.legend(loc='upper right')
    print('mean score', np.mean(scores))