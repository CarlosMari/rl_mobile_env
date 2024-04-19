import sys
import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
import mobile_env 
from reinforce import REINFORCE, PiApproximationWithNN, Baseline, VApproximationWithNN
import wandb


def test_reinforce(with_baseline):
    env = gym.make("mobile-small-central-v0")
    actions_shape = (env.NUM_STATIONS+1, env.NUM_USERS)
    gamma = 1.
    lr = 3e-4

    pi = PiApproximationWithNN(
        env.observation_space.sample().shape,
        actions_shape,
        lr)

    if with_baseline:
        B = VApproximationWithNN(
            env.observation_space.shape[0],
            lr)
    else:
        B = Baseline(0.)

    return REINFORCE(env,gamma,10000,pi,B)

if __name__ == "__main__":
    wandb.init(
        project='MOBILE-ENV',
        name = 'Reinforce'
    )
    num_iter = 1

    # Test REINFORCE with baseline
    with_baseline = []
    for _ in range(num_iter):
        training_progress = test_reinforce(with_baseline=True)
        with_baseline.append(training_progress)
    with_baseline = np.mean(with_baseline,axis=0)


    # Plot the experiment result
    fig,ax = plt.subplots()
    ax.plot(np.arange(len(with_baseline)),with_baseline, label='with baseline')

    ax.set_xlabel('iteration')
    ax.set_ylabel('G_0')
    ax.legend()

    plt.show()

