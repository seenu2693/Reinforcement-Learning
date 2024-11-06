import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

GRID_SIZE = 50         
OBSTACLE_DENSITY = 0.1 
ALPHA = 0.1            
GAMMA = 0.9            
EPSILON = 0.1        
EPISODES = 500         

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def create_grid():
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    num_obstacles = int(OBSTACLE_DENSITY * GRID_SIZE * GRID_SIZE)
    obstacles = random.sample([(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)], num_obstacles)
    for (i, j) in obstacles:
        grid[i, j] = -1  
    return grid


def epsilon_greedy(state, q_table):
    if random.uniform(0, 1) < EPSILON or state not in q_table:
        return random.choice(range(len(actions)))
    return np.argmax(q_table[state])


def q_learning(grid, start, end):
    q_table = defaultdict(lambda: np.zeros(len(actions)))
    
    for episode in range(EPISODES):
        state = start
        done = False
        
        while not done:

            action_index = epsilon_greedy(state, q_table)
            action = actions[action_index]
            
            next_state = (state[0] + action[0], state[1] + action[1])
            
            if (0 <= next_state[0] < GRID_SIZE and 0 <= next_state[1] < GRID_SIZE and grid[next_state] != -1):
                reward = 1 if next_state == end else -0.1
                done = next_state == end
            else:
                next_state = state  
                reward = -1  
            
            best_next_action = np.max(q_table[next_state]) if next_state in q_table else 0
            q_table[state][action_index] += ALPHA * (reward + GAMMA * best_next_action - q_table[state][action_index])
            
            state = next_state
    
    return q_table

def sarsa(grid, start, end):
    q_table = defaultdict(lambda: np.zeros(len(actions)))
    
    for episode in range(EPISODES):
        state = start
        action_index = epsilon_greedy(state, q_table)  
        done = False
        
        while not done:
            action = actions[action_index]
            next_state = (state[0] + action[0], state[1] + action[1])

            if (0 <= next_state[0] < GRID_SIZE and 0 <= next_state[1] < GRID_SIZE and grid[next_state] != -1):
                reward = 1 if next_state == end else -0.1
                done = next_state == end
            else:
                next_state = state  
                reward = -1  

            next_action_index = epsilon_greedy(next_state, q_table)
            
            q_table[state][action_index] += ALPHA * (reward + GAMMA * q_table[next_state][next_action_index] - q_table[state][action_index])

            state, action_index = next_state, next_action_index
    
    return q_table

def extract_path(q_table, start, end):
    path = []
    state = start
    while state != end:
        path.append(state)
        if state not in q_table:
            break
        action_index = np.argmax(q_table[state])
        action = actions[action_index]
        next_state = (state[0] + action[0], state[1] + action[1])
        
        if next_state == state or next_state[0] < 0 or next_state[0] >= GRID_SIZE or next_state[1] < 0 or next_state[1] >= GRID_SIZE:
            break  
        state = next_state
    
    path.append(end)  
    return path

def visualize_grid(grid, path=[]):
    grid_copy = np.copy(grid)
    for (x, y) in path:
        grid_copy[x, y] = 0.5  
    plt.imshow(grid_copy, cmap="gray_r")
    plt.colorbar()
    plt.show()

grid = create_grid()
start = (0, 0)
end = (GRID_SIZE - 1, GRID_SIZE - 1)

while grid[start] == -1 or grid[end] == -1:
    grid = create_grid()
    grid[end] = 1


q_table_q_learning = q_learning(grid, start, end)
path_q_learning = extract_path(q_table_q_learning, start, end)
print("Q-Learning Path Length:", len(path_q_learning))
visualize_grid(grid, path_q_learning)

q_table_sarsa = sarsa(grid, start, end)
path_sarsa = extract_path(q_table_sarsa, start, end)
print("SARSA Path Length:", len(path_sarsa))
visualize_grid(grid, path_sarsa)