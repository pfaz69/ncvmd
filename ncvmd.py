import numpy as np
from VMD_extended import VMD_extended

def NCVMD(f, K, id_main_signal, tau, betas=None, Niter=500):
    # Put main signal first
    if id_main_signal > 0:
        tmp = f[0].copy()
        f[0] = f[id_main_signal].copy()
        f[id_main_signal] = tmp.copy()
    
    
    # Decompose each component of the signal separately
    decomposed_signals = []
    omegas_vmd = []
    u_hats_vmd = []


    if betas is None:
        length = f.shape[0]
        betas = (0,) + (0,) * (length - 1)

    main_omega = None
    for i in range(f.shape[0]):
        omega_ = main_omega
        if i == 0:
            init_vmd = 1
        else:
            init_vmd = 3
    
        # In this case alpha = 1000 as I had to multiply time 2 the original alpha in order toi inser the new term
        u_vmd, u_hat_vmd, omega_vmd = VMD_extended(f[i, :], 1000, tau, K, 0, init_vmd, 1e-7, omega_, Niter, betas[i])
        decomposed_signals.append(u_vmd)
        omegas_vmd.append(omega_vmd)
        u_hats_vmd.append(u_hat_vmd)
        if i == 0:
            main_omega = omega_vmd[-1, :]

    # Normally I should put things back where they belong
    # However for this specific forecasting application
    # which requires that the mode to forecast should 
    # always be the first one for the way multistep
    # forecasting is implemented I'm going to 
    # leave the results where they are (just f needs 
    # be rearranged back in its original position)
    if id_main_signal > 0:
        tmp = [f[0].copy()]#(f[0].copy(), decomposed_signals[0].copy(), omegas_vmd[0].copy(), u_hats_vmd[0].copy())
        f[0] = f[id_main_signal].copy()
        # decomposed_signals[0] = decomposed_signals[id_main_signal].copy()
        # omegas_vmd[0] = omegas_vmd[id_main_signal].copy()
        # u_hats_vmd[0] = u_hats_vmd[id_main_signal].copy()
        f[id_main_signal] = tmp[0].copy()
        # decomposed_signals[id_main_signal] = tmp[1].copy()
        # omegas_vmd[id_main_signal] = tmp[2].copy()
        # u_hats_vmd[id_main_signal] = tmp[3].copy()
    
    # Recompose internally the signal to check the error
    recomposed_signal_ncvmd = np.sum(decomposed_signals, axis=1)
    for i, (f_, rs_) in enumerate(zip(f, recomposed_signal_ncvmd)):
        print(f'NCVMD Error_{i+1}: {np.sqrt(np.mean((f_ - rs_) ** 2))}')

    # # Plot original signals and reconstructions
    # import matplotlib.pyplot as plt
    # _, axes = plt.subplots(f.shape[0], 1, figsize=(12, 4*f.shape[0]))
    # if f.shape[0] == 1:
    #     axes = [axes]
    
    # len = 100# f.shape[1]
    # time = np.arange(len)
    # for i, (ax, f_, rs_) in enumerate(zip(axes, f, recomposed_signal_ncvmd)):
    #     ax.plot(time, f_[-len:], 'b-', label=f'Original Signal {i+1}', alpha=0.7)
    #     ax.plot(time, rs_[-len:], 'r--', label=f'Reconstructed Signal {i+1}', alpha=0.7)
    #     ax.set_title(f'Signal {i+1} Comparison')
    #     ax.set_xlabel('Time')
    #     ax.set_ylabel('Amplitude')
    #     ax.legend()
    #     ax.grid(True)
    
    # plt.tight_layout()
    # plt.show()

    return decomposed_signals, u_hats_vmd, omegas_vmd


