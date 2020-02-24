"""
DQN applied to MILBench task.
"""
import torch

from rlpyt.envs.milbench import MILBenchGymEnv
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.models.dqn.atari_dqn_model import AtariDqnModel
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class MILBenchDqnModel(AtariDqnModel):
    def __init__(self, image_shape, *args, **kwargs):
        image_shape = (image_shape[2], image_shape[0], image_shape[1])
        super().__init__(image_shape, *args, **kwargs)

    def forward(self, observation, prev_action, prev_reward):
        img = observation.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        ax_order = (0, 3, 1, 2)
        flat_img = img.view(T * B, *img_shape).permute(ax_order).contiguous()
        conv_out = self.conv(flat_img)  # Fold if T dimension.
        q = self.head(conv_out.view(T * B, -1))

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, lead_dim, T, B)
        return q


def build_and_train(run_ID=0, cuda_idx=None, n_parallel=2):
    config = dict(
        algo=dict(batch_size=128),
        sampler=dict(batch_T=2, batch_B=32),
    )
    env_kwargs = dict(env_name='MoveToCorner-DebugReward-AtariStyle-v0')
    sampler = GpuSampler(
        EnvCls=MILBenchGymEnv,
        env_kwargs=env_kwargs,
        eval_env_kwargs=env_kwargs,
        CollectorCls=GpuWaitResetCollector,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
        # batch_T=4,  # Get from config.
        # batch_B=1,
        # More parallel environments for batched forward-pass.
        **config["sampler"],
    )
    algo = DQN(**config["algo"])  # Run with defaults.
    agent = AtariDqnAgent(ModelCls=MILBenchDqnModel)
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx,
                      workers_cpus=list(range(n_parallel)),
                      set_affinity=False),
    )
    name = "movetocorner_dqn"
    log_dir = name
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_ID',
                        help='run identifier (logging)',
                        type=int,
                        default=0)
    parser.add_argument('--cuda_idx',
                        help='gpu to use ',
                        type=int,
                        default=None)
    parser.add_argument('--n_parallel',
                        help='number of sampler workers',
                        type=int,
                        default=2)
    args = parser.parse_args()
    build_and_train(
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        n_parallel=args.n_parallel,
    )
