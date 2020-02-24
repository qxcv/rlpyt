
"""
A2C/PPO 
"""
import torch
import torch.nn.functional as F

from rlpyt.envs.milbench import MILBenchGymEnv
from rlpyt.models.pg.atari_ff_model import AtariFfModel
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.algos.pg.a2c import A2C
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.atari import AtariFfAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class MILBenchFfModel(AtariFfModel):
    def __init__(self, image_shape, *args, **kwargs):
        image_shape = (image_shape[2], image_shape[0], image_shape[1])
        super().__init__(image_shape, *args, **kwargs)

    def forward(self, image, prev_action, prev_reward):
        """Just like default .forward(), except we transpose channels."""
        img = image.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        # permute from [N,H,W,C] to [N,C,H,W]
        ax_order = (0, 3, 1, 2)
        flat_img = img.view(T * B, *img_shape).permute(ax_order).contiguous()
        fc_out = self.conv(flat_img)
        pi = F.softmax(self.pi(fc_out), dim=-1)
        v = self.value(fc_out).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v


def build_and_train(run_ID=0, cuda_idx=None, sample_mode="serial",
                    n_parallel=2, algo_name="a2c"):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    gpu_cpu = "CPU" if cuda_idx is None else f"GPU {cuda_idx}"
    if sample_mode == "serial":
        Sampler = SerialSampler  # (Ignores workers_cpus.)
        print(f"Using serial sampler, {gpu_cpu} for sampling and optimizing.")
    elif sample_mode == "cpu":
        Sampler = CpuSampler
        print(f"Using CPU parallel sampler (agent in workers), {gpu_cpu} for optimizing.")
    elif sample_mode == "gpu":
        Sampler = GpuSampler
        print(f"Using GPU parallel sampler (agent in master), {gpu_cpu} for sampling and optimizing.")
    elif sample_mode == "alternating":
        Sampler = AlternatingSampler
        affinity["workers_cpus"] += affinity["workers_cpus"]  # (Double list)
        affinity["alternating"] = True  # Sampler will check for this.
        print(f"Using Alternating GPU parallel sampler, {gpu_cpu} for sampling and optimizing.")

    sampler = Sampler(
        EnvCls=MILBenchGymEnv,
        env_kwargs=dict(env_name='MoveToCorner-DebugReward-AtariStyle-v0'),
        batch_T=5,  # 5 time-steps per sampler iteration.
        batch_B=16,  # 16 parallel environments.
        max_decorrelation_steps=200,
    )
    if algo_name == 'a2c':
        algo = A2C()  # Run with defaults.
    else:
        algo = PPO()
    agent = AtariFfAgent(ModelCls=MILBenchFfModel)
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e5,
        affinity=affinity,
    )
    config = dict(algo_name=algo_name)
    name = f"movetocorner_{algo_name}"
    log_dir = name
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--sample_mode', help='serial or parallel sampling',
        type=str, default='serial', choices=['serial', 'cpu', 'gpu', 'alternating'])
    parser.add_argument('--n_parallel', help='number of sampler workers', type=int, default=2)
    parser.add_argument(
        '--algo', choices=('a2c', 'ppo'), help='algorithm to use', default='a2c')
    args = parser.parse_args()
    build_and_train(
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        sample_mode=args.sample_mode,
        n_parallel=args.n_parallel,
        algo_name=args.algo,
    )
