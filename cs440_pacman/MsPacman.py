import gym
import tensorflow as tf
import numpy as np
import collections
import random
from gym import wrappers
import matplotlib.pyplot as plt

class AGENT:
    def __init__(self, env):

        # Parameters to test
        self.env = env
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995
        self.learning_rate = 0.0025
        self.memory = collections.deque(maxlen=1000000)
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential()
        state_shape = self.env.observation_space.shape

        # Trial with # of hidden layers
        model.add(tf.keras.layers.Dense(256, input_dim=state_shape[0], activation="relu"))
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(self.env.action_space.n, activation="linear"))
        model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        # model = tf.keras.models.load_model("/usr4/ugrad/kevry/episode-401_model_failure.h5") # Continue from previous model
        return model

    def remember(self, state, action, reward, new_state, done): # Saving past states, rewards, etc..
        self.memory.append([state, action, reward, new_state, done])

    def replay(self, batch_size=512): # Once batch is 512, we start modifying the weights
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)

        # Getting all states
        states = np.reshape([np.squeeze(x[0]) for x in samples], (batch_size, 128)) # input size
        actions = np.reshape([x[1] for x in samples], (batch_size,))
        rewards = np.reshape([x[2] for x in samples], (batch_size,))
        new_states = np.reshape([np.squeeze(x[3]) for x in samples], (batch_size, 128)) # input size
        dones = np.reshape([x[4] for x in samples], (batch_size,))

        future_discounted_rewards = np.array(self.model.predict_on_batch(new_states))

        # Selecting max of all future discounted rewards
        future_max_reward = np.max(future_discounted_rewards, axis=1)

        # Modified equation for weights
        updated_future_discounted_rewards = rewards + self.gamma * future_max_reward * (~dones)

        # Predicting action based on state instance
        targets = np.array(self.model.predict_on_batch(states))

        targets[np.arange(len(targets)), np.array(actions)] = updated_future_discounted_rewards

        # Updating the weights in the model. Training
        self.model.train_on_batch(states, targets)

    def get_action(self, state):
        # Decay of the epsilon
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon) # Decay until 0.01

        if np.random.random() > self.epsilon: # Random actions occur in the beginning episodes
            action = np.argmax(self.model.predict(state)[0])
        else:
            action = env.action_space.sample()

        return action

    def save_model(self, fn): # Saving model for future use.
        self.model.save(fn)


########################################################################################
EPISODES = 1000

env = gym.make("MsPacman-ram-v0")
my_agent = AGENT(env=env)

totalreward = []
totalstep = []
for episode in range(0, EPISODES):

    # Saves the episode video
    if episode == 249 or episode == 499 or episode == 749 or episode == 999:
        env = wrappers.Monitor(env, "./gym-episode_newNet{}".format(episode+1), force=True)
        my_agent.save_model("./episode-{}_model_failure_newNet.h5".format(episode+1))

    cur_observation = env.reset().reshape(1, 128)
    done = False

    episode_reward = 0
    steps = 0

    while not done:
        action = my_agent.get_action(state=cur_observation)
        new_observation, reward, done, info = env.step(action)
        new_observation = new_observation.reshape(1, 128)

        my_agent.remember(cur_observation, action, reward, new_observation, done)
        my_agent.replay()

        cur_observation = new_observation

        episode_reward += reward
        steps += 1
    totalreward.append(episode_reward)
    totalstep.append(steps)
    print("Episode {} finished with reward: {}, in {} steps. Epsilon = {}".format(episode + 1, episode_reward, steps, my_agent.epsilon))


# Plotting reward vs. episode
plt.plot(totalreward)
plt.ylabel('Total Reward')
plt.xlabel('Episodes')
plt.savefig('rewardProgress_newNet.pdf')

env.close()
