import asyncio
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from typing import Tuple
from puppeteer_actions import *
from pyppeteer import launch


# Assuming state is a string
state = "new_tab"

# Define the state space
states = [
    "new_tab",
    "open_google_docs",
    "name_document",
    "search_for_article",
    "click_on_article",
    "copy_paragraph",
    "paste_document",
    "change_font",
    "change_font_size",
    "finish_writing"
]

# Define the action space
actions = {
    "new_tab": "open_new_tab",
    "open_google_docs": "open_google_docs",
    "name_document": "name_document",
    "search_for_article": "search_google",
    "click_on_article": "click_result",
    "copy_paragraph": "copy_paragraph",
    "paste_document": "paste_document",
    "change_font": "change_font",
    "change_font_size": "change_font_size",
    "finish_writing": "finish_writing"
}

# Define the rewards
rewards = {
    "new_tab": 0.1,
    "open_google_docs": 0.1,
    "name_document": 0.1,
    "search_for_article": 0.1,
    "click_on_article": 0.1,
    "copy_paragraph": 0.1,
    "paste_document": 0.1,
    "change_font": 0.1,
    "change_font_size": 0.1,
    "finish_writing": 1.0,
    "invalid_action": -0.1
}



# Convert the state to a one-hot encoded vector
state_vector = np.zeros(len(states))
state_index = states.index(state)
state_vector[state_index] = 1


# Define the state size and action size
state_size = len(states)
action_size = len(actions)

# Define the RL agent
class RLAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.memory = []

    def _build_model(self):
        # Neural network with 2 hidden layers of 24 neurons each
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(keras.layers.Dense(24, activation="relu"))
        model.add(keras.layers.Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # Sample a batch of experiences from the memory
        batch = random.sample(self.memory, batch_size)

        # Exclude experiences with None next_state
        batch = [experience for experience in batch if experience[3] is not None]

        if len(batch) == 0:
            return

        # Extract components from the batch
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])

        # Compute Q-values for current and next states
        current_q_values = self.model.predict(states)
        next_q_values = self.model.predict(next_states)

        # Compute target Q-values
        target_q_values = current_q_values.copy()

        for i in range(len(batch)):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.amax(next_q_values[i])

            target_q_values[i][actions[i]] = target

        # Train the model on the updated target Q-values
        self.model.fit(states, target_q_values, epochs=1, verbose=0)


    def act(self, state):
        if state is None:
            return None  # Return None if state is None
        print("state shape:",state)

        # Check the type and shape of the state
        print("state type:", type(state))
        print("state shape:", state.shape)

        state = np.array(state)  
        # Check the type and shape of the input data
        print("input data type:", type(state))
        print("input data shape:", state.shape)

        act_values = self.model.predict(state)
        action = np.argmax(act_values[0])
        return action

    

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)



async def main():
    # Create an instance of the RL agent
    agent = RLAgent(state_size, action_size)
    browser = await launch(headless=False)
    page  = await open_new_tab(browser)
    # Open a new tab
    # browser, page = await open_new_tab()

    # Initialize the current state
    current_state = np.array([state_vector])

    for step in range(100):
        # Choose an action based on the current state
        if current_state is not None:  # Add a check to handle None state
            action = agent.act(current_state)
        else:
            action = None
        # Choose an action based on the current state
        # action = agent.act(current_state)

        # Perform the chosen action and observe the next state and reward
        next_state, reward = await perform_action(page, action)

        # Remember the current state, action, reward, next state, and done flag
        agent.remember(current_state, action, reward, next_state, done=False)

        # Update the current state
        current_state = np.array([next_state]) if next_state is not None else None  # Convert next_state to a NumPy array


        # Update the epsilon value
        agent.decay_epsilon()

        # Perform the replay and train the model
        agent.replay(batch_size=32)

    # Save the trained model
    agent.model.save("rl_model.h5")

    # Close the browser
    await browser.close()

# Run the main function
async def run_main():
    browser = await launch(headless=False)
    try:
        await main()
    finally:
        await browser.close()

# Run the main function using a new event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
loop.run_until_complete(run_main())



