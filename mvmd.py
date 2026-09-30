import numpy as np

def MVMD(signal, alpha, tau, K, DC, init, tol, omega0 = None, Niter = 500):
    # Multivariate Variational Mode Decomposition
    # Check for Input Signal dimensions
    x, y = signal.shape
    if x > y:
        C = y  # number of channels
        T = x  # length of the signal
        signal = signal.T
    else:
        C = x  # number of channels
        T = y  # length of the signal

    # Preparations
    fs = 1 / T  # Sampling Frequency

    # Mirroring
    f = np.zeros((C, 2 * T))
    f[:, :T//2] = signal[:, T//2-1::-1]
    f[:, T//2:3*T//2] = signal
    f[:, 3*T//2:] = signal[:, T-1:T//2-1:-1]

    # Time Domain 0 to T (of mirrored signal)
    T = f.shape[1]
    t = np.arange(1, T+1) / T

    # frequencies
    freqs = t - 0.5 - 1 / T

    # Construct and center f_hat
    f_hat = np.fft.fftshift(np.fft.fft(f, axis=1), axes=1)
    f_hat_plus = np.copy(f_hat)
    f_hat_plus[:, :T//2] = 0

    # Initialization
    N = Niter  # Maximum number of iterations
    Alpha = alpha * np.ones(K)  # For future generalizations

    u_hat_plus_00 = np.zeros((len(freqs), C, K), dtype=complex)
    u_hat_plus = np.zeros((len(freqs), C, K), dtype=complex)
    omega_plus = np.zeros((N, K))

    # initialize omegas
    if omega0 is None:
        if init == 1:
            omega_plus[0, :] = (0.5 / K) * (np.arange(1, K+1) - 1)
        elif init == 2:
            omega_plus[0, :] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(K)))
        else:
            omega_plus[0, :] = 0

        # if DC mode imposed, set its omega to 0
        if DC:
            omega_plus[0, 0] = 0
    elif init == 4: # By Paolo
        omega_plus[0,K//2:K] = np.sort(np.exp(np.log(0.25) + (np.log(0.5)-np.log(0.25))*np.random.rand(1,K//2)))
        for i in range(K//2):
            omega_plus[0,i] = (0.5/K)*(i)
    else:
        omega_plus[0, :] = omega0[:]

        # if DC mode imposed, set its omega to 0
        if DC:
            assert(omega0 is not None)
            omega_plus[0, 0] = 0

    # start with empty dual variables
    lambda_hat = np.zeros((len(freqs), C, N), dtype=complex)

    # Algorithm variables
    uDiff = tol + np.finfo(float).eps  # update step
    n = 0  # loop counter
    sum_uk = np.zeros((len(freqs), C), dtype=complex)  # accumulator

    # Algorithm of MVMD
    while uDiff > tol and n < N-1:
        for k in range(K):
            if k > 0:
                sum_uk = u_hat_plus[:, :, k-1] + sum_uk - u_hat_plus_00[:, :, k]
            else:
                sum_uk = u_hat_plus_00[:, :, K-1] + sum_uk - u_hat_plus_00[:, :, k]

            for c in range(C):
                u_hat_plus[:, c, k] = (f_hat_plus[c, :].T - sum_uk[:, c] - lambda_hat[:, c, n] / 2) / (1 + Alpha[k] * (freqs.T - omega_plus[n, k])**2)

            if DC or (k > 0):
                numerator = np.dot(freqs[T//2:, np.newaxis].T, (np.abs(u_hat_plus[T//2:, :, k])**2))
                denominator = np.sum(np.abs(u_hat_plus[T//2:, :, k])**2, 0)
                temp1 = np.sum(numerator)
                temp2 = np.sum(denominator)
                if omega0 is None:
                    omega_plus[n+1, k] = temp1 / temp2
                else:
                    omega_plus[n+1, :] = omega_plus[n, :]

        # # Debug    
        # if n == 33:    
        #     import matplotlib.pyplot as plt
        #     _, axs = plt.subplots(3, figsize=(10, 8))  # Create a 3x2 grid of subplots
        #     for i in range(3):
        #         axs[i].plot(omega_plus[:, i])  # Plot the data (:, i)
        #         axs[i].set_title(f"omega_plus[:, {i}]")  # Set title for each subplot
        #     plt.show()
        # # Debug   
        # if n == 33:     
        #     import matplotlib.pyplot as plt
        #     _, axs = plt.subplots(3, 2, figsize=(10, 8))  # Create a 3x2 grid of subplots
        #     for i in range(3):
        #         for j in range(2):
        #             axs[i, j].plot(np.abs(u_hat_plus[:, j, i]))  # Plot the data (:, j, i)
        #             axs[i, j].set_title(f"u_hat_plus[{j}, :, {i}]")  # Set title for each subplot

        # Dual ascent
        lambda_hat[:, :, n+1] = lambda_hat[:, :, n] + tau * (np.sum(u_hat_plus, axis=2) - f_hat_plus.T)

        # Update iteration counter and check for convergence
        n += 1
        u_hat_plus_m1 = np.copy(u_hat_plus_00)
        u_hat_plus_00 = np.copy(u_hat_plus)
        # converged yet?
        uDiff = u_hat_plus_00 - u_hat_plus_m1
        uDiff = (1 / T) * (uDiff * np.conj(uDiff))
        uDiff = np.finfo(float).eps + np.abs(np.sum(uDiff))

    print(f'MVMD: The number of iterations is: {n}')

    # Post-processing and cleanup
    N = min(N, n+1)
    omega = omega_plus[:N, :]

    # # Debug   
    # import matplotlib.pyplot as plt
    # _, axs = plt.subplots(3, 2, figsize=(10, 8))  # Create a 3x2 grid of subplots
    # for i in range(3):
    #     for j in range(2):
    #         axs[i, j].plot(np.imag(u_hat_plus[:, j, i]))  # Plot the data (:, j, i)
    #         axs[i, j].set_title(f"u_hat_plus[{j}, :, {i}]")  # Set title for each subplot


    # Signal reconstruction
    idxs = np.flip(np.arange(1,T//2+1),axis = 0)
    u_hat = np.zeros((T, K, C), dtype=complex)
    for c in range(C):
        u_hat[T//2:, :, c] = np.squeeze(u_hat_plus[T//2:, c, :])
        u_hat[T//2-1::-1, :, c] = np.conj(np.squeeze(u_hat_plus[T//2:, c, :]))
        u_hat[idxs,:, c] = np.conj(u_hat_plus[T//2:T, c, :])
        u_hat[0, :, c] = np.conj(u_hat[-1, :, c])

    # # Debug   
    # import matplotlib.pyplot as plt
    # _, axs = plt.subplots(3, 2, figsize=(10, 8))  # Create a 3x2 grid of subplots
    # for i in range(3):
    #     for j in range(2):
    #         axs[i, j].plot(np.real(u_hat[:, i, j]))  # Plot the data (:, j, i)
    #         axs[i, j].set_title(f"u_hat[{i}, :, {j}]")  # Set title for each subplot


    u = np.zeros([K, len(t), C])
    for k in range(K):
        for c in range(C):
            u[k, :, c] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k, c])))

    #u = np.zeros([K,len(t)])
    #for k in range(K):
    #    u[k,:] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:,k])))


    # # Debug        
    # import matplotlib.pyplot as plt
    # _, axs = plt.subplots(3, 2, figsize=(10, 8))  # Create a 3x2 grid of subplots
    # for i in range(3):
    #     for j in range(2):
    #         axs[i, j].plot(u[i, :, j])  # Plot the data (i, :, j)
    #         axs[i, j].set_title(f"u[{i}, :, {j}]")  # Set title for each subplot

    # plt.tight_layout()  # Adjust layout to prevent overlap
    # plt.show()

    # remove mirror part
    u = u[:, T//4:3*T//4, :]

    # recompute spectrum
    u_hat = np.zeros((T//2, K, C), dtype=complex)
    for k in range(K):
        for c in range(C):
            u_hat[:, k, c] = np.fft.fftshift(np.fft.fft(u[k, :, c]))

    u_hat = np.transpose(u_hat, (1, 0, 2))
    return u, u_hat, omega
