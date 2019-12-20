# Sangeeth - 12/14/2019 -  Testing Double Deep Q Learning Keras model for Acrobot-v1 environment.

# Code for testing the data

import keras
import gym
import random
import numpy as np
from keras.layers import Dense, Input
from keras import optimizers
from collections import deque
from keras.models import load_model

# Setting the parameters
run_steps = 10
trials = 100

# Setting up the environment.
env = gym.make('Acrobot-v1')
input_shape = env.reset().size
output_shape = env.action_space.n

#Load the pretrained model
neural_network = load_model('model_s_pretrained.h5')

class agent():
    def __init__(self):
        self.main_model = neural_network
        self.target_model = self.main_model
        self.epsilon = 1.0
        self.steps_update_targetmodel = 0

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
                    score = terminal_steps
                    if i == 0:
                        print("Score per episode: " + np.str(score))
                state = new_state

        return num_success

if __name__ == "__main__":
    acr_agent = agent()
    num_success = acr_agent.test_play()
    print("Test Efficiency : " + np.str(num_success / trials))
    env.close()
    acr_agent.play()
