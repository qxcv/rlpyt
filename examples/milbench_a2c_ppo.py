"""
A2C/PPO
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from rlpyt.envs.milbench import MILBenchGymEnv, MILBenchTrajInfo
from rlpyt.models.pg.atari_ff_model import AtariFfModel
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.algos.pg.a2c import A2C
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.atari import AtariFfAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.buffer import (buffer_from_example, numpify_buffer,
                                torchify_buffer)
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

ENV_NAME = 'MoveToCorner-DebugReward-AtariStyle-v0'
FPS = 40
MAX_T = 80


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


def build_and_train(run_ID=0,
                    cuda_idx=None,
                    sample_mode="serial",
                    n_parallel=2,
                    algo_name="a2c"):
    affinity = dict(cuda_idx=cuda_idx,
                    workers_cpus=list(range(n_parallel)),
                    set_affinity=False)
    gpu_cpu = "CPU" if cuda_idx is None else f"GPU {cuda_idx}"
    if sample_mode == "serial":
        Sampler = SerialSampler  # (Ignores workers_cpus.)
        print(f"Using serial sampler, {gpu_cpu} for sampling and optimizing.")
    elif sample_mode == "cpu":
        Sampler = CpuSampler
        print(
            f"Using CPU parallel sampler (agent in workers), {gpu_cpu} for optimizing."
        )
    elif sample_mode == "gpu":
        Sampler = GpuSampler
        print(
            f"Using GPU parallel sampler (agent in master), {gpu_cpu} for sampling and optimizing."
        )
    elif sample_mode == "alternating":
        Sampler = AlternatingSampler
        affinity["workers_cpus"] += affinity["workers_cpus"]  # (Double list)
        affinity["alternating"] = True  # Sampler will check for this.
        print(
            f"Using Alternating GPU parallel sampler, {gpu_cpu} for sampling and optimizing."
        )
    sampler = Sampler(
        EnvCls=MILBenchGymEnv,
        env_kwargs=dict(env_name=ENV_NAME),
        batch_T=5,  # 5 time-steps per sampler iteration.
        batch_B=16,  # 16 parallel environments.
        max_decorrelation_steps=80,
        TrajInfoCls=MILBenchTrajInfo,
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
        log_interval_steps=2e4,
        affinity=affinity,
    )
    config = dict(algo_name=algo_name)
    name = f"movetocorner_{algo_name}"
    log_dir = name
    with logger_context(log_dir,
                        run_ID,
                        name,
                        config,
                        snapshot_mode="last",
                        override_prefix=True):
        runner.train()


def test(snapshot_path, cuda_idx):
    env = MILBenchGymEnv(ENV_NAME)
    agent = AtariFfAgent(ModelCls=MILBenchFfModel)
    agent.initialize(env.spaces)
    loaded_pkl = torch.load(snapshot_path, map_location=torch.device('cpu'))
    agent.load_state_dict(loaded_pkl['agent_state_dict'])
    # TODO: CUDA support for policy, somehow
    try:
        while True:
            # set up PyTorch buffers
            init_obs = env.reset()
            init_obs = buffer_from_example(init_obs, 1)
            init_act = env.action_space.null_value()
            init_act = buffer_from_example(init_act, 1)
            init_prev_reward = np.zeros((1, ), dtype='float32')
            obs_pyt, act_pyt, prev_rew_pyt = torchify_buffer(
                (init_obs, init_act, init_prev_reward))
            slow_frames = 0

            # housekeeping
            traj_info = MILBenchTrajInfo()
            agent.reset()

            # pretend it's iteration 0 because it shouldn't matter what
            # iteration it is (I think that's just used to set epsilon for DQN,
            # but DQN doesn't seem like it should be training in eval mode)
            # TODO: on second thought, double-check that DQN really isn't doing
            # epsilon-greedy exploration in eval mode.
            agent.eval_mode(0)
            for t in range(MAX_T):
                # to regulate FPS
                frame_start = time.time()

                act_pyt, agent_info = agent.step(obs_pyt, act_pyt,
                                                 prev_rew_pyt)
                act_np = numpify_buffer(act_pyt)
                obs_np = numpify_buffer(obs_pyt)
                new_obs, new_rew, done, new_env_info = env.step(act_np[0])
                traj_info.step(obs_np[0], act_np[0], new_rew, done,
                               agent_info[0], new_env_info)
                if getattr(new_env_info, "traj_done", done):
                    traj_info_done = traj_info.terminate(new_obs)
                    print(f"Trajectory done; score is {traj_info_done.Score} "
                          f"and base reward is {traj_info_done.BaseReward}")
                    traj_info = MILBenchTrajInfo()
                    new_obs = env.reset()
                if done:
                    print("Got 'done' flag from environment")
                    act_pyt[0] = 0  # Prev_action for next step.
                    new_rew = 0.0
                    agent.reset_one(idx=0)
                obs_pyt[0] = torchify_buffer(new_obs)
                prev_rew_pyt[0] = new_rew

                env.render(mode='human')

                elapsed = time.time() - frame_start
                sleep_time = max(0, 1. / FPS - elapsed)
                time.sleep(sleep_time)
                if sleep_time == 0:
                    slow_frames += 1

            if slow_frames > MAX_T / 10:
                print(f"WARNING: below target FPS {slow_frames} slow "
                      f"frames in this rollout")

    finally:
        env.close()


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
    parser.add_argument('--sample_mode',
                        help='serial or parallel sampling',
                        type=str,
                        default='serial',
                        choices=['serial', 'cpu', 'gpu', 'alternating'])
    parser.add_argument('--n_parallel',
                        help='number of sampler workers',
                        type=int,
                        default=2)
    parser.add_argument('--algo',
                        choices=('a2c', 'ppo'),
                        help='algorithm to use',
                        default='a2c')
    parser.add_argument('--test_pol', default=None, help='test given policy')
    args = parser.parse_args()
    if args.test_pol:
        test(snapshot_path=args.test_pol, cuda_idx=args.cuda_idx)
    else:
        build_and_train(
            run_ID=args.run_ID,
            cuda_idx=args.cuda_idx,
            sample_mode=args.sample_mode,
            n_parallel=args.n_parallel,
            algo_name=args.algo,
        )
