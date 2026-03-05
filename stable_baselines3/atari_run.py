import os
import sys
sys.path.append("/home/dsi/fiskustal/talfiskus/stable-baselines3/")

import gymnasium as gym
import matplotlib.pyplot as plt

from stable_baselines3.common import results_plotter
from stable_baselines3.common.logger import configure
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.dqn.ddqn import DDQN
from stable_baselines3.dqn.dqn1 import DQN1
from stable_baselines3.common.callbacks import EvalCallback

import time
from datetime import timedelta

if __name__ == "__main__":
    # There already exists an environment generator
    # that will make and wrap atari environments correctly.
    # Here we are also multi-worker training (n_envs=4 => 4 environments)
    # get the device
    run_device = sys.argv[1]
    print("device", run_device)
    # get CF lambda from command line
    cf_lambda = float(sys.argv[2])
    print("cf_lambda", cf_lambda)
    # get theory from command line
    theory = sys.argv[3]
    print("theory", theory)
    # get the agent
    agent_name = sys.argv[4]
    print("agent_name", agent_name)
    # get the timestep from command line
    timesteps = int(sys.argv[5])
    print("timesteps", timesteps)
    # get the run number from command line
    run_number = sys.argv[6]
    print("run_number", run_number)
    # get the random seed
    random_seed = int(sys.argv[7])
    print("random_seed", random_seed)
    # get the environment from command line
    gym_env_name = sys.argv[8]
    print("gym_env_name", gym_env_name)

    # get the model name from command line
    model_name = gym_env_name + "_" + agent_name + "_atari_cf_lambda_" + str(cf_lambda) + "_timesteps_" + str(timesteps) + "_run_" + str(
        run_number) + "_seed_" + str(random_seed)
    model_name = model_name.replace("ALE/", "")
    print("model_name", model_name)

    # Full directory path
    full_log_dir = os.path.join("/home/dsi/fiskustal/talfiskus/stable-baselines3/", theory, agent_name, gym_env_name.replace("ALE/", ""))
    if not os.path.exists(full_log_dir):
        os.makedirs(full_log_dir, exist_ok=True)
    print("full_log_dir", full_log_dir)

    env = make_atari_env(gym_env_name, seed=random_seed)
    # Stack 4 frames
    env = VecFrameStack(env, n_stack=4)
    # Eval env
    eval_env = make_atari_env(gym_env_name, seed=random_seed)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_callback = EvalCallback(eval_env, eval_freq=10000, verbose=1, n_eval_episodes=100)
    tb_logger_name = f"{model_name}"
    # select agent by name
    if agent_name == 'PPO':
        model = PPO('CnnPolicy', env, verbose=2, device=run_device, cf_lambda=cf_lambda, tensorboard_log=full_log_dir, tensor_board_logger_name=tb_logger_name)
    elif agent_name == 'DQN':
        buffer_size = int(sys.argv[9])
        print("buffer_size", buffer_size)
        # tb_logger_name = f"{tb_logger_name}_buffer_{buffer_size}"
        # model = DQN('CnnPolicy', env, verbose=2, device=run_device, cf_lambda=cf_lambda, buffer_size=buffer_size, tensorboard_log=full_log_dir, tensor_board_logger_name=tb_logger_name)
        model = DQN('CnnPolicy', env, verbose=2, device=run_device, buffer_size=buffer_size,)
    elif agent_name == 'DDQN':
        buffer_size = int(sys.argv[9])
        print("buffer_size", buffer_size)
        tb_logger_name = f"{tb_logger_name}_buffer_{buffer_size}"
        model = DDQN('CnnPolicy', env, verbose=2, device=run_device, cf_lambda=cf_lambda, buffer_size=buffer_size, tensorboard_log=full_log_dir, tensor_board_logger_name=tb_logger_name)
    elif agent_name == 'DQN1':
        buffer_size = int(sys.argv[9])
        print("buffer_size", buffer_size)
        importance_sampling = bool(sys.argv[10])
        print("importance_sampling", importance_sampling)
        importance_sampling_normalize = bool(sys.argv[11])
        print("importance_sampling_normalize", importance_sampling_normalize)
        weighted_importance_sampling = bool(sys.argv[12])
        print("weighted_importance_sampling", weighted_importance_sampling)
        tb_logger_name = f"{tb_logger_name}_buffer_{buffer_size}_importance_sampling_{importance_sampling}_importance_sampling_normalize_{importance_sampling_normalize}_weighted_importance_sampling_{weighted_importance_sampling}"
        model = DQN1('CnnPolicy', env, verbose=2, device=run_device, cf_lambda=cf_lambda, buffer_size=buffer_size, tensorboard_log=full_log_dir, tensor_board_logger_name=tb_logger_name, importance_sampling=importance_sampling, importance_sampling_normalize=importance_sampling_normalize, weighted_importance_sampling=weighted_importance_sampling)

    else:
        print("ERROR - no valid agent")
    # Start measure runtime
    start_time = time.time()
    # Train model
    model.learn(total_timesteps=timesteps, callback=eval_callback) #, callback=eval_callback)
    # Record the end time
    end_time = time.time()
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    # print(f"Trained agent mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    # Save model
    # model.save(f"{full_log_dir}/trained_model")
    # plot_results([full_log_dir], timesteps, results_plotter.X_EPISODES, model_name)
    # plt.savefig(f"{full_log_dir}/plot_episodes_reward_plot.png")
    # plot_results([full_log_dir], timesteps, results_plotter.X_TIMESTEPS, model_name)
    # plt.savefig(f"{full_log_dir}/plot_timesteps_reward_plot.png")

    # Calculate the total runtime
    total_time = end_time - start_time
    # Convert total_time to hh:mm:ss format
    formatted_time = str(timedelta(seconds=int(total_time)))
    # Print the total runtime
    print(f"Total train runtime: {formatted_time}")

    # # Enjoy trained agent
    # video_length = 2000
    # vec_env = model.get_env()
    # # Record the video starting at the first step
    # vec_env = VecVideoRecorder(vec_env, f"{full_log_dir}/video",
    #                            record_video_trigger=lambda x: x == 0, video_length=video_length,
    #                            name_prefix=f"{model_name}")

    # obs = vec_env.reset()
    # for i in range(video_length):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, rewards, dones, info = vec_env.step(action)
    # # Save the video
    # vec_env.close()
