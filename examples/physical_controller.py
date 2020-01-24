import gym
from gym import wrappers, logger


class PhysicalAgent:
    def __init__(self, action_space, T_ext, k):
        self.max = action_space.high[0]
        self.T_ext = T_ext
        self.k = k

    def act(self, observation):
        T, P, T_target = observation
        if T - T_target > 0.5:
            return self.max
        elif T - T_target < 0:
            return 0.
        else:
            return (self.T_ext - T_target) * self.k


def train(agent, env):
    ob = env.reset()
    t_history = []
    action = 0
    while True:
        ob, reward, done, _ = env.step(action)
        T, P, T_target = ob
        t_history.append(T)

env = gym.make('gym_thermostat:thermostat-v0')
outdir = '/tmp/physical-results'
logger.set_level(logger.INFO)
env = wrappers.Monitor(env, directory=outdir, force=True)
env.seed(0)
agent = PhysicalAgent(env.action_space, env.Text, env.k)

episode_count = 100

for i in range(episode_count):
    logger.info("episode = %d" % i)
    ob = env.reset()
    while True:
        action = agent.act(ob)
        ob, reward, done, _ = env.step(action)
        if done:
            break

env.close()