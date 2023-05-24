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

        # Extract components from the batch
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        # Predict the Q-values of the current states
        current_q_values = self.model.predict(states)

        # Predict the Q-values of the next states
        next_q_values = self.model.predict(next_states)

        # Update the Q-values
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            target = reward
            if not done:
                target += self.gamma * np.amax(next_q_values[i])

            current_q_values[i][action] = target

        # Fit the model to the updated Q-values
        self.model.fit(states, current_q_values, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Explore by choosing a random action
            return np.random.choice(self.action_size)
        else:
            # Exploit by choosing the best action based on current Q-values
            state = np.reshape(state, [1, self.state_size])
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])
        

async def perform_action(action_name, action):
    next_state = "new_tab"  # Replace with the actual next state
    reward = 0.1  # Replace with the actual reward
    done = False  # Replace with the actual done flag

    if action == "new_tab":
        page = await actions[action](page)
        next_state = get_state("open_google_docs", None)
    elif action == "open_google_docs":
        browser, page = await actions[action]()
        next_state = get_state("name_document", None)
    elif action == "name_document":
        await actions[action](page)
        next_state = get_state("search_for_article", None)
    elif action == "search_for_article":
        await actions[action](page)
        next_state = get_state("click_on_article", None)
    elif action == "click_on_article":
        await actions[action](page)
        next_state = get_state("copy_paragraph", None)
    elif action == "copy_paragraph":
        paragraph = await actions[action](page)
        next_state = get_state("paste_document", paragraph)
    elif action == "paste_document":
        await actions[action](page, action.split(":")[1])
        next_state = get_state("change_font", None)
    elif action == "change_font":
        await actions[action](page)
        next_state = get_state("change_font_size", None)
    elif action == "change_font_size":
        await actions[action](page)
        next_state = get_state("finish_writing", None)
    elif action == "finish_writing":
        await actions[action](page)
        next_state = get_state("finish_writing", None)

    reward = get_reward(action)

    return next_state, reward


async def main():
    agent = RLAgent(state_size, action_size)
    episodes = 1000
    batch_size = 32

    for episode in range(episodes):
        current_state = state_vector
        done = False
        total_reward = 0

        while not done:
            action = agent.act(current_state)
            action_name = actions[states[np.argmax(current_state)]]
            next_state, reward = await perform_action(action_name, action)  # Pass the 'action' argument

            if next_state not in states:
                next_state = "new_tab"  # Replace with a valid default state

            next_state_vector = np.zeros(len(states))
            next_state_index = states.index(next_state)
            next_state_vector[next_state_index] = 1

            agent.remember(current_state, action, reward, next_state_vector, done)
            total_reward += reward
            current_state = next_state_vector

        agent.replay(batch_size)
        print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}")

    # Save the trained model
    agent.model.save("rl_agent_model.h5")

async def run_main():
    browser = await launch()
    await main()
    await browser.close()

loop = asyncio.get_event_loop()
loop.run_until_complete(run_main())


# async def perform_action(action: str) -> Tuple[str, float, bool]:
#     next_state, reward, done = "", 0.0, False
#     if action == "open_new_tab":
        
#         next_state = await open_new_tab(browser)
#         reward = rewards[action]
#     elif action == "open_google_docs":
#         next_state = await open_google_docs()
#         reward = rewards[action]
#     elif action == "name_document":
#         next_state = await name_document()
#         reward = rewards[action]
#     elif action == "search_google":
#         next_state = await search_google()
#         reward = rewards[action]
#     elif action == "click_result":
#         next_state = await click_result()
#         reward = rewards[action]
#     elif action == "copy_paragraph":
#         next_state = await copy_paragraph()
#         reward = rewards[action]
#     elif action == "paste_document":
#         next_state = await paste_document()
#         reward = rewards[action]
#     elif action == "change_font":
#         next_state = await change_font()
#         reward = rewards[action]
#     elif action == "change_font_size":
#         next_state = await change_font_size()
#         reward = rewards[action]
#     elif action == "finish_writing":
#         next_state = await finish_writing()
#         reward = rewards[action]
#         done = True
#     else:
#         reward = rewards["invalid_action"]

#     return next_state, reward, done


# async def main():
#     # Create an instance of the RL agent
#     agent = RLAgent(state_size, action_size)
#     browser = await launch(headless=False)
#     page  = await open_new_tab(browser)
#     episodes = 1000
#     batch_size = 32
#     for episode in range(episodes):
#         current_state = state_vector
#         done = False
#         total_reward = 0

#         while not done:
#             action = agent.act(current_state)
#             action_name = actions[states[np.argmax(current_state)]]
#             next_state, reward, done = await perform_action(action_name)

#             next_state_vector = np.zeros(len(states))
#             next_state_index = states.index(next_state)
#             next_state_vector[next_state_index] = 1

#             agent.remember(current_state, action, reward, next_state_vector, done)
#             total_reward += reward
#             current_state = next_state_vector

#         agent.replay(batch_size)

#         print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}")
        
#     agent.model.save("rl_model.h5")

# # Run the main function
# async def run_main():
#     browser = await launch(headless=False)
#     try:
#         await main()
#     finally:
#         await browser.close()

# Run the main function using a new event loop
# loop = asyncio.get_event_loop()
# loop.run_until_complete(run_main())

