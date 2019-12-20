# Sangeeth - 12/14/2019 -  Training Double Deep Q Learning Keras model for Acrobot-v1 environment.

import keras
import gym, datetime, time
import random
import numpy as np
from keras.layers import Dense, Input
from keras import optimizers
from collections import deque
from keras.models import load_model

# Setting the parameters
num_episode = 2
replay_memory_length = 10000
minimum_epsilon = 0.01
minimum_rep_mem_length = 1000
epsilon_rate = 0.0001
gamma = 0.99
batch_size = 256
run_steps = 10
trials = 100
step_to_reset = 20

# Setting up the environment.
env = gym.make('Acrobot-v1')
input_shape = env.reset().size
output_shape = env.action_space.n

# Defining the neural network
x = Input(shape=(input_shape,), name='input')
fc1 = (Dense(64, activation='relu', kernel_initializer='normal'))(x)
keras.layers.Dropout(.25)
fc2 = (Dense(64, activation='relu', kernel_initializer='normal'))(fc1)
keras.layers.Dropout(.25)
fc3 = (Dense(output_shape, activation='softmax', kernel_initializer='normal'))(fc2)

neural_network = keras.models.Model(inputs=x, outputs=fc3)
sgd = optimizers.Adam(lr=0.001)
neural_network.compile(loss='mse', optimizer=sgd)

# Pretrained model for 2 episodes
# neural_network = load_model('model_s.h5')
class agent():
# Initializing the model.
    def __init__(self):
        """
        Args:
             main_model : A neural network to find an approximation of optimal action value function. This is calculated
                          for the current state.
             target_model : This is needed to calculate the action value function determined using the Bellman equation.
                            Now we can train the main model using the predicted action value function and the target
                            action value function determined using the Bellman equation. This is calculated for the next state.
             epsilon : Exploration probability
             replay_memory : Choose an action and perform a step to achieve reward. This step is added as a transition
                             in the replay memory. It draws a minibatch from replay memory to perform a gradient
                             descent step. replay memory stores replay_memory_length transitions. A transition is
                             state, action, reward, next state, done.
             steps_update_targetmodel : After every C steps the target model is updated with the same parameters as the main model.
        """
        self.main_model = neural_network
        self.target_model = self.main_model
        self.epsilon = 1.0
        self.replay_memory = deque(maxlen=replay_memory_length)
        self.steps_update_targetmodel = 0

# Part of the learning process. Find the best action and evaluate the action value function at that action point.
    def evaluate(self, terminated_episode):
        """
        done : whether the episode terminated
        action : integer between  0 and env.action_space.n - 1
        reward : determine reward agent received for performing an action.
        gamma :  discount factor for the Bellman equation
        """
        if len(self.replay_memory) < minimum_rep_mem_length:
            return
        # Select a random minibatch of transitions from the replay memory
        replay_batch = random.sample(population=self.replay_memory, k=batch_size)
        # For every transition in the minibatch, estimate the action value function.
        for sarsa in replay_batch:
            try:
                states_array
            except NameError:
                states_array = sarsa[0].reshape(1, sarsa[0].shape[0])
                next_states_array = sarsa[3].reshape(1, sarsa[3].shape[0])
            else:
                states_array = np.append(states_array, sarsa[0].reshape(1, sarsa[0].shape[0]), axis=0)
                next_states_array = np.append(next_states_array, sarsa[3].reshape(1, sarsa[3].shape[0]), axis=0)

        state_q_values = self.main_model.predict(states_array)
        next_q_values = self.target_model.predict(next_states_array)

        for index, (state, action, reward, next_state, done) in enumerate(replay_batch):
            # If the game is terminated, action value function = rewards as there is no next state.
            if done:
                q_y = reward
            else:
                # Using double deep q learning,
                # The main network estimates which action is best, corresponding to the next state.
                best_action_loc = np.argmax(self.main_model.predict(next_state.reshape(1, 6)))
                # The target network estimates the q values following Bellman Equation.
                q_y = reward + gamma * next_q_values[index][best_action_loc]

            state_q_values[index][action] = q_y
            if index == 0:
                x = state.reshape(1, state.shape[0])
                y = state_q_values[index].reshape(1, state_q_values[index].shape[0])
            else:
                x = np.append(x, state.reshape(1, state.shape[0]), axis=0)
                y = np.append(y, state_q_values[index].reshape(1, state_q_values[index].shape[0]), axis=0)

        # parameter update on the main model using gradient descent
        self.main_model.fit(x, y, batch_size=batch_size, verbose=0)

        # For every step_to_reset steps, reset the target model to the main model
        if terminated_episode:
            self.steps_update_targetmodel = self.steps_update_targetmodel + 1
        if self.steps_update_targetmodel == step_to_reset:
            self.target_model = self.main_model
            self.steps_update_targetmodel = 0

# Learning the double deep q learning model.
    def ddqn(self, num_episode):
        start = time.time()
        for i in range(num_episode):
            state = env.reset()
            done = False
            while not done:
                # Determine action according to epsilon greedy strategy
                if np.random.random() < self.epsilon:
                    action = np.random.randint(0, output_shape)
                else:
                    q_values = self.main_model.predict(state.reshape(1, state.shape[0]))
                    action = np.argmax(q_values)
                # Execute action and observe reward
                new_state, reward, done, info = env.step(action)
                # Store the transition in the replay memory
                self.replay_memory.append((state, action, reward, new_state, done))
                self.evaluate(done)
                state = new_state
            # Epsilon greedy strategy with annealing action
            if self.epsilon > minimum_epsilon:
                self.epsilon = self.epsilon * (1 - epsilon_rate)
                self.epsilon = max(self.epsilon, minimum_epsilon)

        end = time.time()
        diff = end - start
        # diff = datetime.timedelta(seconds=diff)
        print("Training Time: " + np.str(diff))

# Run the game by rendering the environment.
    def play(self):
        for i in range(run_steps):
            # resets the environment
            state = env.reset()
            terminal_steps = 0
            done = False
            flag = 0
            while not done:
                q_values = self.main_model.predict(state.reshape(1, state.shape[0]))
                best_action_loc = np.argmax(q_values)
                new_state, reward, done, info = env.step(best_action_loc)
                env.render()
                terminal_steps = terminal_steps + 1
                if done and terminal_steps < 500:
                    flag = 1
                state = new_state

            if flag == 1:
                print("Goal achieved")
            else:
                print("Goal not achieved")

# Testing to see how many times goal is achieved out of trial times.
    def test_play(self):
        num_success = 0
        for i in range(trials):
            state = env.reset()
            terminal_steps = 0
            done = False
            while not done:
                q_values = self.main_model.predict(state.reshape(1, state.shape[0]))
                best_action = np.argmax(q_values)
                new_state, reward, done, info = env.step(best_action)
                terminal_steps = terminal_steps + 1
                if done and terminal_steps < 500:
                    num_success = num_success + 1
                state = new_state

        return num_success
# Save the model that can be used later for playing or testing the model
    def save_model(self, trials):
        self.main_model.save("model_s_new.h5")

if __name__ == "__main__":
    acr_agent = agent()
    acr_agent.ddqn(num_episode=num_episode)
    # num_success = acr_agent.test_play()
    # print("Test Efficiency : " + np.str(num_success/trials))
    acr_agent.save_model(trials=100)
    env.close()
    # acr_agent.play()
