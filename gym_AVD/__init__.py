from gym.envs.registration import register

register(
    id='AVD-v0',
    entry_point='gym_AVD.envs:AVDEnv',
)
