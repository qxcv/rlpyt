import gym

from rlpyt.envs.gym_schema import GymEnvWrapper


class MILBenchGymEnv(GymEnvWrapper):
    """Useful for constructing rlpyt environments from Gym environment names
    (as needed to, e.g., create agents/samplers/etc.). Will automatically
    register MILBench envs first."""
    def __init__(self, env_name, **kwargs):
        from milbench import register_envs
        register_envs()
        env = gym.make(env_name)
        super().__init__(env, **kwargs)
