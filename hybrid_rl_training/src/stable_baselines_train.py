#!/usr/bin/env python

import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import gym
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
from openai_ros.task_envs.turtlebot2.turtlebot2_maze import TurtleBot2MazeEnv
import rospy
import os
from customPolicy import *
# Custom MLP policy of three layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[144, 144, 144],
                                                          vf=[144, 144, 144])],
                                           feature_extraction="mlp")

if __name__ == '__main__':
	world_file = sys.argv[1]
	number_of_robots = sys.argv[2]
	rospy.init_node('stable_training', anonymous=True, log_level=rospy.WARN)
	env_temp = TurtleBot2MazeEnv
	env = SubprocVecEnv([lambda k=k:env_temp(world_file, k) for k in range(int(number_of_robots))])
	model = PPO2(CustomTinyDeepCNNPolicy, env, n_steps=900, ent_coef=0.01, learning_rate=0.0001, nminibatches=5, tensorboard_log="../PPO2_turtlebot_tensorboard/", verbose=1)
	model.learn(total_timesteps=1200000)
	model.save("ppo2_turtlebot")
																																																																																																																																																											