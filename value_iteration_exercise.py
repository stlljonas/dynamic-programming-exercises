import numpy as np

q = np.array([[4, 6], [-5, -3]]) # use: q[state, u]

P = np.array([[[.8, .2], [.5, .5]], [[.7, .3], [.4, .6]]]) # use: P[state_i, u, state_j]

alpha = 0.99

def main():
    n_states = 2
    value_function = np.ones(n_states)
    old_value_function = value_function
    n_iterations = 1000 
    for i in range(n_iterations):
        for state in range(n_states):
            possible_us = np.empty(2)
            possible_us[0] = q[state, 0] + alpha * (P[state, 0, 0] * old_value_function[0] + P[state, 0, 1] * old_value_function[1])
            possible_us[1] = q[state, 1] + alpha * (P[state, 1, 0] * old_value_function[0] + P[state, 1, 1] * old_value_function[1])
            
            value_function[state] = np.max(possible_us)
        old_value_function = value_function
        print(f"value_function: {value_function}")
    optimal_policy = np.zeros(n_states)
    for state in range(n_states):
        possible_us = np.empty(2)
        possible_us[0] = q[state, 0] + alpha * (P[state, 0, 0] * value_function[0] + P[state, 0, 1] * value_function[1])
        possible_us[1] = q[state, 1] + alpha * (P[state, 1, 0] * value_function[0] + P[state, 1, 1] * value_function[1])
        optimal_policy[state] = np.argmax(possible_us)
    print(f"optimal_policy: {optimal_policy}")
    return


if __name__ == '__main__':
    main()
