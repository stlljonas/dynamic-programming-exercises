import numpy as np

# Note this was for the Dynamic Programming and Optimal Control class 2022, problem set 1 exercise 11
# my solution deviates slightly, but my implementation is at least mostly correct. I got the problem formulation right at least

def x_ref(k):
    return (k - 5)**2

def g_k(x_k, u_k, k):
    return (x_k - x_ref(k))**2 + u_k**2

J_k_over_x_k = np.empty((11,11)) # k(row) in [0, 10] and x_k in [0, 10]

optimal_u = np.empty((10, 11))

def J_k(x_k, k, N):
    possible_u_ks = np.array(np.arange(-x_k, 10 - x_k + 1))
    expected_values = np.empty_like(possible_u_ks)
    for idx, u_k in enumerate(possible_u_ks):
        if k == N:
            J_k_over_x_k[k, x_k] = x_k**2
            return x_k**2
        else:    
            expected_values[idx] = g_k(x_k, u_k, k) + 1/3 * J_k_over_x_k[k+1, x_k + u_k] + 2/3 * J_k_over_x_k[k+1, x_k]
    optimal_u[k, x_k] = possible_u_ks[np.argmin(expected_values)]
    J_k_over_x_k[k, x_k] = np.min(expected_values)

def main():
    # compute optimal controls
    # for all k
        # call J_k for all x_k in [0, 10]
    np.set_printoptions(precision=3)
    N = 10
    for k in reversed(range(11)):
        for x_k in range(11):
            J_k(x_k, k, N)
    print(f"J_k_over_x_k: {J_k_over_x_k}")
    print(f"optimal_u: {optimal_u}")

if __name__ == '__main__':
    main()
