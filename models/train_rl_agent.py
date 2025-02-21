# models/train_rl_agent.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.trajectories import trajectory
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy

# Import your custom trading environment
from models.rl_env import GoldTradingEnv

# ----------------------------
# Load Historical Data
# ----------------------------
# Load your gold data from CSV (make sure the 'data' folder exists and contains gold_data.csv)
data = pd.read_csv("data/gold_data.csv")
# For this example, we use the following columns: Open, High, Low, Close, Volume.
data_np = data[['Open', 'High', 'Low', 'Close', 'Volume']].values

# ----------------------------
# Create the Custom Environment
# ----------------------------
# Instantiate your custom environment with the loaded data.
env = GoldTradingEnv(data_np)
train_env = tf_py_environment.TFPyEnvironment(env)
eval_env = tf_py_environment.TFPyEnvironment(env)

# ----------------------------
# Build the Q-Network for the Agent
# ----------------------------
fc_layer_params = (64, 64)
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params
)

# ----------------------------
# Create the DQN Agent
# ----------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_huber_loss,
    train_step_counter=train_step_counter
)
agent.initialize()

# ----------------------------
# Set Up the Replay Buffer
# ----------------------------
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=10000
)

# ----------------------------
# Data Collection Function
# ----------------------------
def collect_step(environment, policy, buffer):
    """Collects a single step of experience from the environment."""
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    # Add trajectory to the replay buffer
    buffer.add_batch(traj)

# ----------------------------
# Initial Data Collection
# ----------------------------
# Use a random policy to populate the replay buffer initially.
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())
for _ in range(100):
    collect_step(train_env, random_policy, replay_buffer)

# Create a dataset from the replay buffer.
dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    single_deterministic_pass=False
)
dataset = iter(dataset)

# ----------------------------
# Training Loop
# ----------------------------
num_iterations = 1000  # Adjust as needed
for _ in range(num_iterations):
    # Collect a step of data using the agent's current policy.
    collect_step(train_env, agent.policy, replay_buffer)
    
    # Sample a batch of data from the replay buffer and update the agent.
    experience, _ = next(dataset)
    train_loss = agent.train(experience).loss

    if train_step_counter.numpy() % 100 == 0:
        print(f"Step: {train_step_counter.numpy()}, Loss: {train_loss:.4f}")

# ----------------------------
# Save the Trained Policy
# ----------------------------
policy_dir = "models/rl_policy"
if not os.path.exists(policy_dir):
    os.makedirs(policy_dir)
agent.policy.save(policy_dir)
print(f"Policy saved to: {policy_dir}")
