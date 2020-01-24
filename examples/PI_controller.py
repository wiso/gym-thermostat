import gym
from gym import wrappers, logger


class PIAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.p = 50.
        self.i = 3.
        self.integral_window = 100
        self.temperature_diff_history = []

    def act(self, observation):
        T, P, T_target = observation
        diff = T - T_target
        self.temperature_diff_history.append(diff)
        diff_integral = sum(self.temperature_diff_history[-self.integral_window:])
        logger.info("diff = %f integral = %f P = %f I = %f" % (diff, diff_integral, self.p * diff, self.i * diff))
        if diff > 0:
            return self.p * diff + self.i * diff_integral
        else:
            return 0.

env = gym.make('gym_thermostat:thermostat-v0')
outdir = '/tmp/pi-results'
logger.set_level(logger.INFO)
env = wrappers.Monitor(env, directory=outdir, force=True)
env.seed(0)
agent = PIAgent(env.action_space)

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
