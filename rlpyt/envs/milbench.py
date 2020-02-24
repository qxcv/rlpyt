import gym
from milbench import register_envs

from rlpyt.envs.gym import GymEnvWrapper


class MILBenchGymEnv(GymEnvWrapper):
    """Useful for constructing rlpyt environments from Gym environment names
    (as needed to, e.g., create agents/samplers/etc.). Will automatically
    register MILBench envs first."""
    def __init__(self, env_name, **kwargs):
        register_envs()
        env = gym.make(env_name)
        super().__init__(env, **kwargs)
