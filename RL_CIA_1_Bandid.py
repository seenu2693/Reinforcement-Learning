import numpy as np
import scipy.stats as stats

K = 10  
trials = 1000  

counts = np.zeros(K)  
rewards = np.zeros(K) 


def ucb_select_arm(counts, rewards, t):
    ucb_values = np.zeros(K)
    for i in range(K):
        if counts[i] == 0:
            ucb_values[i] = float('inf')  
        else:
            avg_reward = rewards[i] / counts[i]
            confidence_bound = np.sqrt(2 * np.log(t + 1) / counts[i])
            ucb_values[i] = avg_reward + confidence_bound
    return np.argmax(ucb_values)


def simulate_bandit(arm):
    return np.random.normal(loc=0.5 + arm * 0.1, scale=0.1)


for t in range(trials):
    arm = ucb_select_arm(counts, rewards, t)
    reward = simulate_bandit(arm)
    counts[arm] += 1
    rewards[arm] += reward

print("Final counts per arm:", counts)
print("Final rewards per arm:", rewards / counts)


alpha = np.ones(K)  
beta = np.ones(K)   

for t in range(trials):
    samples = [np.random.beta(alpha[i], beta[i]) for i in range(K)]
    arm = np.argmax(samples)  
    reward = simulate_bandit(arm)
    alpha[arm] += reward
    beta[arm] += 1 - reward

print("Final alpha per arm:", alpha)
print("Final beta per arm:", beta)
