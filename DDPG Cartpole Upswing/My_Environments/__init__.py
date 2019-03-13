from gym.envs.registration import register

register(
    id='CartpoleDownwards-v0',
    entry_point='My_Environments.cartpole_downwards:CartPoleEnv_Downwards',
    max_episode_steps=400,
)

register(
    id='CartpoleDownwardsCont-v0',
    entry_point='My_Environments.cartpole_downwards_continuous:CartPoleDownwardsContinuous',
    max_episode_steps=400,
)
