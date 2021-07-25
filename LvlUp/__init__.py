from gym.envs.registration import register

register(
    id='lvlup-v0',
    entry_point='LvlUp.envs:LevelUpdateEnv',
)