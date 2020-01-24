import gym
from gym import wrappers, logger


class OnOffAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        T, P, T_target = observation
        if T > T_target:
            return self.action_space.high[0]
        else:
            return 0.


env = gym.make('gym_thermostat:thermostat-v0')
outdir = '/tmp/on-off-results'
logger.set_level(logger.INFO)
env = wrappers.Monitor(env, directory=outdir, force=True)
env.seed(0)
agent = OnOffAgent(env.action_space)

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
