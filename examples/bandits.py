import numpy as np
import matplotlib.pyplot as plt
from markov.envs.n_bandits_env import NBanditsEnv
from markov.algos.value_methods import EpsilonGreedy, SoftmaxActionSelection


def train_bandit(env, num_episodes, value_method, save_checkpoints=True, data=None):
    # choose random action with prob epsilon
    # If epsilon == 0, a random action will be generated the first time and no
    # exploration will happen

    # Sum of return over all episodes
    sum_of_returns = 0.0

    # data array
    data = None
    if (save_checkpoints):
        data = np.empty((num_episodes, env.episode_len))

    episode = 0
    while episode < num_episodes:
        if (episode % 100 == 0):
            print "Episode: ", episode
        value_method.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = value_method.get_action()
            step = env.step(action)
            done = step.done

            value_method.update_values(action, step.reward)
            total_reward += step.reward

            if (save_checkpoints):
                data[episode][env.itr-1] = step.reward


        sum_of_returns += total_reward / env.episode_len
        episode += 1
        env.reset()

    #print "avg return: ", sum_of_returns / num_episodes
    return data





if __name__ == "__main__":
    num_episodes = 500
    env = NBanditsEnv(10, 1000, random_walk=True)

    use_egreedy = True

    if (use_egreedy):
        epsilons = [0, .01, .1]
        data = np.empty((len(epsilons), num_episodes, env.episode_len))
        for i in range(len(epsilons)):
            e = epsilons[i]
            egreedy = EpsilonGreedy(e, env.action_space.n, step_update=True)
            partial_data = train_bandit(env, num_episodes, egreedy, save_checkpoints=True, data=data)
            data[i,:,:] = partial_data

        data = np.mean(data, axis=1)
        plt.plot(data.T)
        plt.legend([str(x) for x in epsilons])
        plt.show()
    else:
        temps = [0.25, 0.1, 0.01]
        data = np.empty((len(temps), num_episodes, env.episode_len))
        for i in range(len(temps)):
            t = temps[i]
            softmax = SoftmaxActionSelection(t, env.action_space.n)
            partial_data = train_bandit(env, num_episodes, softmax, save_checkpoints=True, data=data)
            data[i,:,:] = partial_data

        data = np.mean(data, axis=1)
        plt.plot(data.T)
        plt.legend([str(x) for x in temps])
        plt.show()
