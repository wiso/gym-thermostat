from gym.envs.registration import register

register(
    id='thermostat-v0',
    entry_point='gym_thermostat.envs:ThermostatEnv',
    max_episode_steps=1000
)
