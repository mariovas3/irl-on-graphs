"""
This module implements Linear Quadratic Regulator - based control.

Given dynamics and cost function parameters, find a plan / sequence 
of actions that minimise the cost along the path.

The dynamics are linear:
x_{t+1} = F_t @ (x_t, u_t) + f_t
with known time-variant matrix F_t, and time-variant vector f_t.

The cost function is quadratic:
c(x_t, u_t) = 0.5 * (x_t, u_t) @ C_t @ (x_t, u_t) + (x_t, u_t) @ c_t
with known time-variant PD matrix C_t, and known time-variant vector c_t.
"""


import numpy as np


def get_Kt_kt(x_dim, Cs, cs, Fs, fs):
    V, v = np.zeros((x_dim, x_dim)), np.zeros((x_dim,))
    Ks, ks = [], []
    T = len(Fs) + 1
    for t in range(T, 0, -1):
        Q = Cs[t] + Fs[t].T @ V @ Fs[t]
        q = cs[t] + Fs[t].T @ (V @ fs[t] + v)

        # get Ks;
        Q_inv = np.linalg.inv(Q[x_dim:, x_dim:])
        Ks.append(-Q_inv @ Q[x_dim:, :x_dim])
        ks.append(-Q_inv @ q[x_dim:])

        # update V and v;
        KT = Ks[-1].T
        V = (
            Q[:x_dim, :x_dim] 
            + Q[:x_dim, x_dim:] @ Ks[-1] 
            + KT @ Q[x_dim:, :x_dim] 
            + KT @ Q[x_dim:, x_dim:] @ Ks[-1]
        )
        v = (
            Q[:x_dim, x_dim:] @ ks[-1] 
            + KT @ Q[x_dim:, x_dim:] @ ks[-1]
            + q[:x_dim]
            + KT @ q[x_dim:]
        )
    return Ks, ks


def get_plan(x, Ks, ks, Fs, fs):
    us = []
    T = len(Fs) + 1
    for t in range(T):
        us.append(Ks[t] @ x + ks[t])
        x = Fs[t] @ np.concatenate((x, us[-1])) + fs[t]
    return us
