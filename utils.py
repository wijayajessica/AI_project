import numpy as np
import matplotlib.pyplot as plt
import time
from gridworld import GridWorld

def epsilon_greedy_action(state, Q, epsilon=0.1):
    """
    Select a random action with probability epsilon or the action suggested
    by Q with probability 1-epsilon.
    Inputs:
    -state: current state.
    -Q: 2D numpy array of dimensions (num_states, num_actions).
    -epsilon: probability of randomizing an action.
    
    Retuns: action.
    """

    if np.random.random() < epsilon:
        return np.random.choice(Q.shape[1])
    else:
        return np.argmax(Q[state])

def run_simulation(
        # Common parameters
        env,
        method,
        min_num_episodes=100,
        min_num_iters=5000,
        epsilon=0.1,
        discount=0.95,
        # SARSA/Q-learning parameters
        step_size=0.5,
        Q_initial=0.0,
        MAX_STEPS_PER_EPISODE = 100
    ):
    # Ensure valid parameters
    if method not in ('SARSA', 'Expected-SARSA', 'Q-learning', 'Double-Q-learning', ):
        raise ValueError("method not in {SARSA, Expected-SARSA, Q-learning, Double-Q-learning}")

    # Initialize arrays for our estimate of Q and observations about T and R,
    # and our list of rewards by episode
    num_states, num_actions = env.num_states, env.num_actions
    Q = np.zeros((num_states, num_actions)) + Q_initial
    if method == 'Doubly-Q-learning':
        Q2 = np.zeros((num_states, num_actions)) + Q_initial #second Q table for double Q learning
    observed_T_counts = np.zeros((num_states, num_actions, num_states))
    observed_R_values = np.zeros((num_states, num_actions, num_states))
    episode_rewards = []
    num_cliff_falls = 0
    global_iter = 0

    # Loop through episodes
    while len(episode_rewards) < min_num_episodes or global_iter < min_num_iters:
        # Reset environment and episode-specific counters
        env.reset()
        episode_step = 0
        episode_reward = 0

        # Get our starting state
        s1 = env.observe()

        # Loop until the episode completes
        while not env.is_terminal(s1) and episode_step < MAX_STEPS_PER_EPISODE:
            # Take eps-best action & receive reward
            a = epsilon_greedy_action(s1, Q, epsilon)
            s2, r = env.perform_action(a)

            # Update counters
            episode_step += 1
            episode_reward += r
            observed_T_counts[s1][a][s2] += 1
            observed_R_values[s1][a][s2] = r
            num_cliff_falls += env.is_cliff(s2)

            # Use one of the RL methods to update Q
            if method == 'SARSA':
                # Compute the next action, use it for next state value estimate
                a2 = epsilon_greedy_action(s2, Q, epsilon)
                next_state_val = Q[s2,a2]
                Q[s1,a] += step_size * (r + discount * next_state_val - Q[s1,a]) # Update Q
            
            elif method == 'Expected-SARSA':
                state_action_values = Q[s2,:]
                value_sum = np.sum(state_action_values)
                max_value = np.max(state_action_values)
                max_count = len(state_action_values[state_action_values == max_value])
                k = Q.shape[1] # total number of actions
                
                expected_value_for_max = max_value * ((1 - epsilon) / max_count + epsilon / k) * max_count
                expected_value_for_non_max = (value_sum - max_value * max_count) * (epsilon / k)
                expected_value = expected_value_for_max + expected_value_for_non_max

                Q[s1,a] += step_size * (r + discount * expected_value - Q[s1,a])  # Update Q
                                
            elif method == 'Q-learning':
                next_state_val = Q[s2].max() # Treat the next state value as the best possible
                Q[s1,a] += step_size * (r + discount * next_state_val - Q[s1,a])# Update Q
            
            elif method == 'Doubly-Q-learning':
                if np.random.random() < 0.5: 
                    #use Q2 to update Q1
                    next_state_val = Q2[s2].max()
                    Q[s1,a] += step_size * (r + discount * next_state_val - Q[s1,a])
                else:
                    #use Q1 to update Q2
                    next_state_val = Q[s2].max()
                    Q2[s1,a] += step_size * (r + discount * next_state_val - Q2[s1,a])
                
            s1 = s2
            global_iter += 1

        episode_rewards.append(episode_reward)

    return { 'Q': Q,
            'num_cliff_falls': num_cliff_falls,
            'episode_rewards': np.array(episode_rewards) }

def plot_policy(env, Q):
    # credit: code inspired by Harvard's AC209b materials

    row_count, col_count = env.maze_dimensions
    maze_dims = (row_count, col_count)
    value_function = np.reshape(np.max(Q, 1), maze_dims)
    policy_function = np.reshape(np.argmax(Q, 1), maze_dims)
    wall_info = .5 + np.zeros(maze_dims)
    wall_mask = np.zeros(maze_dims)
    for row in range(row_count):
        for col in range(col_count):
            if env.maze.topology[row][col] == '#':
                wall_mask[row,col] = 1
    wall_info = np.ma.masked_where(wall_mask==0, wall_info)
    value_function *= (1-wall_mask)**2
    plt.imshow(value_function, interpolation='none', cmap='jet')
    plt.colorbar(label='Value Function')
    plt.imshow(wall_info, interpolation='none' , cmap='gray')
    y,x = env.maze.start_coords
    plt.text(x,y,'start', color='gray', fontsize=14, va='center', ha='center', fontweight='bold')
    y,x = env.maze.goal_coords
    plt.text(x,y,'goal', color='yellow', fontsize=14, va='center', ha='center', fontweight='bold')
    for row in range( row_count ):
        for col in range( col_count ):
            if wall_mask[row][col] == 1:
                continue
            if policy_function[row,col] == 0:
                dx = 0; dy = -.5
            if policy_function[row,col] == 1:
                dx = 0; dy = .5
            if policy_function[row,col] == 2:
                dx = .5; dy = 0
            if policy_function[row,col] == 3:
                dx = -.5; dy = 0
            plt.arrow(col, row, dx, dy,
                shape='full', fc='w' , ec='w' , lw=3, length_includes_head=True, head_width=.2)
    plt.xlabel("X-Coordinate")
    plt.ylabel("Y-Coordinate")
