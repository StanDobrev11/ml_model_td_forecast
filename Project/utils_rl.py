# import chardet
#
# file_path = 'data/raw/d_2024_07_01.dat'
# # Open the file and detect encoding
# with open(file_path, 'rb') as file:
#     raw_data = []
#     for i in range(20):
#         data = file.readline()
#         print(chardet.detect(data))
#     # result = chardet.detect(raw_data)
#     # encoding = result['encoding']
#     # print(f"Detected encoding: {encoding}")
#
# with open(file_path, 'r', encoding='Windows-1252', errors='replace') as f:
#     for i in range(10):
#         lines = f.readline()
#         print(lines)
#     # print(lines[9140:9150])  # Print around the problematic line

import numpy as np

# Maze dimensions
maze = np.array([
    [0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0],
])

# Q-learning parameters
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1  # exploration rate

# Q-table (state-action table)
q_table = np.zeros((maze.shape[0], maze.shape[1], 4))  # 4 actions (up, down, left, right)

# Action space
actions = ['up', 'down', 'left', 'right']
action_dict = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}


def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(actions)  # explore
    else:
        return actions[np.argmax(q_table[state[0], state[1]])]  # exploit


def update_q_table(state, action, reward, next_state):
    action_index = actions.index(action)
    best_next_action = np.argmax(q_table[next_state[0], next_state[1]])
    q_table[state[0], state[1], action_index] += alpha * (
            reward + gamma * q_table[next_state[0], next_state[1], best_next_action] - q_table[
        state[0], state[1], action_index]
    )


# Simulate Q-learning for 1000 episodes
for episode in range(1):
    state = [0, 0]  # Starting position
    done = False
    while not done:
        action = choose_action(state)
        next_state = [state[0] + action_dict[action][0], state[1] + action_dict[action][1]]

        if next_state == [4, 4]:  # Reached the goal
            reward = 10
            done = True
        elif next_state[0] < 0 or next_state[1] < 0: # hit a wall which is outside of the maze
            reward = -1
            next_state = state  # stay in place
        elif next_state[0] >= maze.shape[0] or next_state[1] >= maze.shape[1]:
            reward = -1
            next_state = state
        elif maze[next_state[0], next_state[1]] == 1:  # Hit a wall
            reward = -1
            next_state = state  # stay in place
        else:
            reward = -0.1  # Small penalty for each step

        update_q_table(state, action, reward, next_state)
        state = next_state
print(q_table)