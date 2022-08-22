import numpy as np
import scipy as sp
import scipy.linalg
import scipy.optimize
from scipy.io import savemat
from itertools import combinations
import numba as nb
from numba import jit

def save_EM_model(EM_model, savePath, Y_data=None, U_data=None):
    """ This function allows us to store the data matrices within the EM_model.

        Parameters
        ----------
        EM_model: the surrogate model of type 'ActuatedKoopmanEM'
        savePath: where to store the matrices as an npz file
        Y_data:   aray of data points used for the training
        U_data:   aray of corresponding control inputs

        Returns
        -------
        Nothing
    """
    if savePath[-4:] != '.npz':
        savePath = savePath + '.npz'

    if Y_data is not None:
        np.savez(savePath, C=EM_model.C, EM_iter=EM_model.EM_iter, L=EM_model.L, Sig_0=EM_model.Sig_0,
                 Sig_v=EM_model.Sig_v,
                 Sig_w=EM_model.Sig_w, Sighats_kk=EM_model.Sighats_kk, Sighats_kkp1=EM_model.Sighats_kkp1,
                 U_matrices=EM_model.U_matrices, V_matrices=EM_model.V_matrices, h=EM_model.h, mu_0=EM_model.mu_0,
                 muhats=EM_model.muhats, n_t=EM_model.n_t, u_data=EM_model.u_data, y_data=EM_model.y_data,
                 u_dim=EM_model.u_dim, y_dim=EM_model.y_dim, z_dim=EM_model.z_dim, U_data=U_data, Y_data=Y_data)
    else:
        np.savez(savePath, C=EM_model.C, EM_iter=EM_model.EM_iter, L=EM_model.L, Sig_0=EM_model.Sig_0, Sig_v=EM_model.Sig_v,
                 Sig_w=EM_model.Sig_w, Sighats_kk=EM_model.Sighats_kk, Sighats_kkp1=EM_model.Sighats_kkp1,
                 U_matrices=EM_model.U_matrices, V_matrices=EM_model.V_matrices, h=EM_model.h, mu_0=EM_model.mu_0,
                 muhats=EM_model.muhats, n_t=EM_model.n_t, u_data=EM_model.u_data, y_data=EM_model.y_data,
                 u_dim=EM_model.u_dim, y_dim=EM_model.y_dim, z_dim=EM_model.z_dim)
    # L1=EM_model.L1, L2=EM_model.L2, L3=EM_model.L3, H_Q=EM_model.H_Q,


def load_EM_model(EM_model, loadPath):
    """ This function allows us to load the data matrices into the EM_model.

        Parameters
        ----------
        EM_model: the surrogate model of type 'ActuatedKoopmanEM'
        loadPath: where to store the matrices as an npz file

        Returns
        -------
        EM_model: the updated EM_model
    """
    if loadPath[-4:] != '.npz':
        loadPath = loadPath + '.npz'
    dataIn = np.load(loadPath)

    EM_model.C = dataIn['C']
    EM_model.EM_iter = dataIn['EM_iter']
    EM_model.L = dataIn['L']
    # EM_model.H_Q = dataIn['H_Q']
    # EM_model.L1 = dataIn['L1']
    # EM_model.L2 = dataIn['L2']
    # EM_model.L3 = dataIn['L3']
    EM_model.Sig_0 = dataIn['Sig_0']
    EM_model.Sig_v = dataIn['Sig_v']
    EM_model.Sig_w = dataIn['Sig_w']
    EM_model.Sighats_kk = dataIn['Sighats_kk']
    EM_model.Sighats_kkp1 = dataIn['Sighats_kkp1']
    EM_model.U_matrices = dataIn['U_matrices']
    EM_model.V_matrices = dataIn['V_matrices']
    EM_model.h = dataIn['h']
    EM_model.mu_0 = dataIn['mu_0']
    EM_model.muhats = dataIn['muhats']
    EM_model.n_t = dataIn['n_t']
    EM_model.u_data = dataIn['u_data']
    EM_model.y_data = dataIn['y_data']
    EM_model.u_dim = dataIn['u_dim']
    EM_model.y_dim = dataIn['y_dim']
    EM_model.z_dim = dataIn['z_dim']
    if 'U_data' in dataIn:
        U_data = dataIn['U_data']
        Y_data = dataIn['Y_data']
    else:
        U_data = []
        Y_data = []
    return EM_model, Y_data, U_data


def reduce_EM_model(EM_model, indices):
    Sighats_kk = np.copy(EM_model.Sighats_kk)
    Sighats_kkp1 = np.copy(EM_model.Sighats_kkp1)
    U_matrices = np.copy(EM_model.U_matrices)
    muhats = np.copy(EM_model.muhats)
    u_data = np.copy(EM_model.u_data)
    y_data = np.copy(EM_model.y_data)

    # indices = np.random.permutation(np.arange(0, EM_model.y_data.shape[0]))[:n_data]

    EM_model.u_data = EM_model.u_data[indices, :, :]
    EM_model.y_data = EM_model.y_data[indices, :, :]
    EM_model.Sighats_kk = EM_model.Sighats_kk[indices, :, :, :]
    EM_model.Sighats_kkp1 = EM_model.Sighats_kkp1[indices, :, :, :]
    EM_model.U_matrices = EM_model.U_matrices[indices, :, :, :]
    EM_model.muhats = EM_model.muhats[indices, :, :]

    return EM_model, u_data, y_data, Sighats_kk, Sighats_kkp1, U_matrices, muhats


def restore_EM_model(indices, EM_model, u_data, y_data, Sighats_kk, Sighats_kkp1, U_matrices, muhats):
    u_data[indices, :, :] = EM_model.u_data
    y_data[indices, :, :] = EM_model.y_data
    Sighats_kk[indices, :, :] = EM_model.Sighats_kk
    Sighats_kkp1[indices, :, :] = EM_model.Sighats_kkp1
    U_matrices[indices, :, :] = EM_model.U_matrices
    muhats[indices, :, :] = EM_model.muhats

    EM_model.u_data = np.copy(u_data)
    EM_model.y_data = np.copy(y_data)
    EM_model.Sighats_kk = np.copy(Sighats_kk)
    EM_model.Sighats_kkp1 = np.copy(Sighats_kkp1)
    EM_model.U_matrices = np.copy(U_matrices)
    EM_model.muhats = np.copy(muhats)
    return EM_model

@jit(nopython=True, parallel=True)
def _compute_implicit_evolution_matrices(u_data: nb.float32[:,:,:], \
    V_matrices: nb.float32[:,:,:]):
    """Helper function for computing evolution matrices by implicit Euler
    approximation from linear combinations of generators. In particular, the 
    function computes 
    U_k = (I - u0(t_k)*V0 - ... - uq(t_k)*Vq)^-1
    for sequence of inputs u(t_k).

    Parameters
    ----------
    u_data: array of shape (num_traj, len_traj-1, num_inputs)
        Inputs to the system along each trajectory.
    V_matrices: array of shape (num_inputs, num_states, num_states)
        Matrix approximations of Koopman generators associated with each
        component vector field.
        
    Returns
    -------
    U_matrices: array of shape (num_traj, len_traj-1, num_states, num_states)
        Evolution matrices over each time interval along trajectories. In
        particular, U_k = (I - u0(t_k)*V0 - ... - uq(t_k)*Vq)^-1.
    """
    
    # computes U_k = (I-u0(t_k)*V0 - ... - uq(t_k)*Vq)^-1 
    # matrices assuming Delta_t = 1
    U_matrices = np.zeros((u_data.shape[0], u_data.shape[1], \
        V_matrices.shape[2], V_matrices.shape[2]))
    
    for j in range(u_data.shape[0]):
        for k in range(u_data.shape[1]):
            # compute V_k = u0(t_k)*V0 + ... + uq(t_k)*Vq
            V = np.zeros((V_matrices.shape[2],V_matrices.shape[2]))
            for i in range(u_data.shape[2]):
                V = V + u_data[j,k,i]*V_matrices[i,:,:]
            
            # compute U_k
            U_matrices[j,k,:,:] = np.linalg.inv(np.eye(V_matrices.shape[2]) - V)
    
    return U_matrices

@jit(nopython=True, parallel=True)
def _compute_explicit_evolution_matrices(u_data: nb.float32[:,:,:], \
    V_matrices: nb.float32[:,:,:]):
    """Helper function for computing evolution matrices from linear
    combinations of generators using first-order explicit Euler approximation. 
    In particular, the function computes
    U_k = I + u0(t_k)*V0 + ... + uq(t_k)*Vq for sequence of inputs u(t_k).

    Parameters
    ----------
    u_data: array of shape (num_traj, len_traj-1, num_inputs)
        Inputs to the system along each trajectory.
    V_matrices: array of shape (num_inputs, num_states, num_states)
        Matrix approximations of Koopman generators associated with each
        component vector field.
        
    Returns
    -------
    U_matrices: array of shape (num_traj, len_traj-1, num_states, num_states)
        Evolution matrices over each time interval along trajectories. In
        particular, U_k = I + u0(t_k)*V0 + ... + uq(t_k)*Vq.
    """
    U_matrices = np.zeros((u_data.shape[0], u_data.shape[1], \
        V_matrices.shape[2], V_matrices.shape[2]))
    for j in range(u_data.shape[0]):
        for k in range(u_data.shape[1]):
            U_matrices[j,k,:,:] = np.eye(V_matrices.shape[2])
            for i in range(u_data.shape[2]):
                U_matrices[j,k,:,:] = U_matrices[j,k,:,:] + \
                    u_data[j,k,i]*V_matrices[i,:,:]
    
    return U_matrices

@jit(nopython=True, parallel=True)
def _predict_dynamics(z_0: nb.float32[:,:], Sig_0: nb.float32[:,:,:], \
        U_matrices: nb.float32[:,:,:,:], Sig_v: nb.float32[:,:], \
        C: nb.float32[:,:], h: nb.float32[:], Sig_w: nb.float32[:,:]):
    """Helper function for predict_dynamics that computes the latent state 
    trajectories and observations along with the uncertainty from given initial 
    conditions and inputs. This function assumes the evolution matrices have
    already been computed.

    The initial condition for the state is modeled as a probability distribution 
    accounting for our uncertainty about the initial state.

    Parameters
    ----------
    z_0: array of shape (num_traj, num_states)
        Means of the initial condition distributions.
    Sig_0: array of shape (num_traj, num_states, num_states)
        Covariances of the initial condition distributions.
    U_matrices: array of shape (num_traj, len_traj-1, num_states, num_states)
        Evolution matrices over each time interval along trajectories.
    Sig_v: array of shape (num_states, num_states)
        Process noise covariance.
    C: array of shape (num_observations, num_states)
        Observation matrix that maps latent states to observations
    h: array of shape (num_observations)
        Constant shift for observations
    Sig_w: array of shape (num_observations, num_observations)
        Observation noise covariance.
    
    Returns
    -------
    Y_pred: array of shape (num_traj, len_traj, num_obs)
        Means of observed quantities along predicted trajectory.
    Sig_y_pred: array of shape (num_traj, len_traj, num_obs, num_obs)
        Covariances of observed quantities along predicted trajectory.
    Z_pred: array of shape (num_traj, len_traj, num_states, num_states)
        Means of latent states along predicted trajectory.
    Sig_z_pred: array of shape (num_traj, len_traj, num_states, num_states)
        Covariances of latent states along predicted trajectory.
    """
    
    n_traj = z_0.shape[0]
    n_t = U_matrices.shape[1]
    z_dim = z_0.shape[1]
    y_dim = C.shape[0]

    # predicted latent state variables
    Z_pred = np.zeros((n_traj, n_t, z_dim))
    Sig_z_pred = np.zeros((n_traj, n_t, z_dim, z_dim))

    # predicted outputs and uncertainty covariance
    Y_pred = np.zeros((n_traj, n_t, y_dim))
    Sig_y_pred = np.zeros((n_traj, n_t, y_dim, y_dim))

    z = z_0
    Sig_z = Sig_0
    for j in range(n_traj):
        for k in range(n_t):
            A = U_matrices[j,k,1:,1:]
            b = U_matrices[j,k,1:,0]
            z[j,:] = np.dot(A, z[j,:]) + b
            Sig_z[j,:,:] = Sig_v + np.dot(A, np.dot(Sig_z[j,:,:], A.T))
            
            Z_pred[j,k,:] = z[j,:]
            Sig_z_pred[j,k,:,:] = Sig_z[j,:,:]

            Y_pred[j,k,:] = np.dot(C, z[j,:]) + h
            Sig_y_pred[j,k,:,:] = Sig_w + np.dot(C, np.dot(Sig_z[j,:,:], C.T))
    
    return Y_pred, Sig_y_pred, Z_pred, Sig_z_pred

@jit(nopython=True, parallel=True)
def _propagate_implicit_Euler_gradients(U_matrices: nb.float32[:,:,:,:], \
    grad_matrices: nb.float32[:,:,:,:]):
    """ Helper function for _dynamical_loss_and_gradient that back-propagates 
    the gradients of functions through the matrix exponential. In particular, 
    we have the gradients of the dynamical loss during the M-step with respect
    to each evolution matrix. This function back-propagates these gradients
    through the matrix exponential in order to compute the gradients of the
    dynamical loss with respect to the linear combinations of generator 
    matrices.

    To be precise, the dynamical loss can be expressed as
    f(V_1, ..., V_t) = g((I-V_1)^-1, ..., (I-V_t)^-1)
    where V_k = u0(t_k)*V0 + ... + uq(t_k)*Vq. Given the gradients of g with
    respect to each U_k = (I-V_k)^-1, we compute the gradients of f with respect 
    to each V_k.

    Parameters
    ----------
    U_matrices: array of shape (num_traj, len_traj-1, num_states, num_states)
        Evolution matrices governing dynamics along trajectories computed by
        implicit Euler approximation.
    grad_matrices: array of shape (num_traj, len_traj-1, num_states, num_states)
        Gradients of the dynamical loss along each trajectory with respect to
        evolution matrices. In particular, these are the gradients of g with
        respect to each U_k = (I-V_k)^-1.
    
    Returns
    -------
    grad_prop_matrices: array of shape (num_traj, len_traj-1, num_states, 
        num_states)
        Gradients of dynamical loss with respect to linear combinations of
        generators. In particular, the gradients of f with respect to each V_k.
    """
    grad_prop_matrices = np.zeros(U_matrices.shape)
    for j in range(U_matrices.shape[0]):
        for k in range(U_matrices.shape[1]):
            grad_prop_matrices[j,k,:,:] = np.dot(U_matrices[j,k,:,:].T, \
                np.dot(grad_matrices[j,k,:,:], U_matrices[j,k,:,:].T))
    
    return grad_prop_matrices

@jit(nopython=True, parallel=True)
def _Kalman_init(y_0: nb.float32[:], mu_0: nb.float32[:], \
        Sig_0: nb.float32[:,:], C: nb.float32[:,:], Sig_w: nb.float32[:,:]):
    """Helper function for _Kalman_Rauch that computes the posterior mean and
    covariance of the initial latent state given the first observation. These
    are used to initialize the forward Kalman filtering pass.

    Parameters
    ----------
    y_0: array of shape (num_observations)
        Observations at the first time along a trajectory. Here we assume the
        constant shift, h, has already been subtracted.
    mu_0: array of shape (num_states)
        Mean of the prior distribution over initial conditions for latent state
        trajectories.
    Sig_0: array of shape (num_states, num_states)
        Covariances of the prior distribution over initial conditions for latent 
        state trajectories.
    C: array of shape (num_observations, num_states)
        Observation matrix mapping latent states to observations.
    Sig_w: array of shape (num_observations, num_observations)
        Observation noise covariance.
    
    Returns
    -------
    muhat_0_0: array of shape (num_states)
        Mean of posterior distribution for the initial latent state given only
        the first observation along the trajectory.
    Sighat_0_0: array of shape (num_states, num_states)
        Covariance of posterior distribution for the initial latent state given 
        only the first observation along the trajectory.
    """
    
    S = Sig_w + C.dot(Sig_0).dot(C.T)

    # compute Kalman gain
    K_0 = Sig_0.dot(np.linalg.solve(S, C).T)

    # mean update
    muhat_0_0 = mu_0 + K_0.dot(y_0 - C.dot(mu_0))

    # covariance update
    Sighat_0_0 = Sig_0 - K_0.dot(C).dot(Sig_0)
    
    return muhat_0_0, Sighat_0_0

@jit(nopython=True, parallel=True)
def _Kalman_update(y_k: nb.float32[:], muhat_km1_km1: nb.float32[:], \
        Sighat_km1_km1: nb.float32[:,:], U_km1: nb.float32[:,:], \
        C: nb.float32[:,:], Sig_v: nb.float32[:,:], Sig_w: nb.float32[:,:]):
    """Helper function for _Kalman_Rauch that computes the optimal estimate of
    a latent state given the observation up to that time from the optimal
    estimate of the previous state given the previous observations. This 
    function implements one step of the classical Kalman filter and is called 
    recursively in order to perform the forward-pass of the Kalman-Rauch 
    smoother.

    Parameters
    ----------
    y_k: array of shape (num_observations)
        Observations at the current time along trajectory. Here we assume the
        constant shift, h, has already been subtracted.
    muhat_km1_km1: array of shape (num_states)
        Posterior mean of the previous latent state given the previous 
        observations.
    Sighat_km1_km1: array of shape (num_states, num_states)
        Posterior covariance of the previous latent state given the previous
        observations.
    U_km1: array of shape (num_states+1, num_states+1)
        Evolution matrix between the previous latent state and the current one.
        This matrix includes the evolution of the constant state and has the
        structure U = [[0, --0--], [b, --A--]]
    C: array of shape (num_observations, num_states)
        Observation matrix mapping latent states to observations.
    Sig_v: array of shape (num_states, num_states)
        Process noise covariance for latent state model.
    Sig_w: array of shape (num_observations, num_observations)
        Observation noise covariance for model.
    
    Returns
    -------
    muhat_k_k: array of shape (num_states)
        Posterior mean of the current latent state given the observations up to
        (and including) the current time.
    Sighat_k_k: array of shape (num_states, num_states)
        Posterior covariance of the current latent state given the observations
        up to (and including) the current time.
    Sighat_k_km1: array of shape (num_states, num_states)
        Posterior covariance of the current latent state given the previous
        observations not including the one made at the current time.
    """
    # extract evolution matrix and constant drift
    A_km1 = U_km1[1:, 1:]
    b_km1 = U_km1[1:, 0]
    
    Sighat_k_km1 = Sig_v + np.dot(A_km1, np.dot(Sighat_km1_km1, A_km1.T))
    S = Sig_w + np.dot(C, np.dot(Sighat_k_km1, C.T))

    # compute Kalman gain
    K_k = Sighat_k_km1.dot(np.linalg.solve(S, C).T)

    # mean update
    mu_k_km1 = A_km1.dot(muhat_km1_km1) + b_km1
    muhat_k_k = mu_k_km1 + K_k.dot(y_k - C.dot(mu_k_km1))

    # covariance update
    Sighat_k_k = Sighat_k_km1 - np.dot(K_k, np.dot(C, Sighat_k_km1))

    # ensure no loss of symmetry due to round-off
    Sighat_k_km1 = 0.5*(Sighat_k_km1 + Sighat_k_km1.T)
    Sighat_k_k = 0.5*(Sighat_k_k + Sighat_k_k.T)
    
    return muhat_k_k, Sighat_k_k, Sighat_k_km1

@jit(nopython=True, parallel=True)
def _Rauch_update(muhat_kp1: nb.float32[:], Sighat_kp1: nb.float32[:,:], \
        muhat_k_k: nb.float32[:], Sighat_k_k: nb.float32[:,:], \
        Sighat_kp1_k: nb.float32[:,:], U_k: nb.float32[:,:]):
    """Helper function for _Kalman_Rauch that computes the optimal esimate of a
    latent state given all of the observations by making use of the optimal
    estimate of the state at the next time step. This function implements the
    classical Rauch smoother and is called recursively in order to perform the 
    backward-pass of the Kalman-Rauch smoother.

    Parameters
    ----------
    muhat_kp1: array of shape (num_states)
        Posterior mean of latent state at time k+1 given all observations along
        the trajectory.
    Sighat_kp1: array of shape (num_states, num_states)
        Posterior covariance of latent states at time k+1 given all observations
        along the trajectory.
    muhat_k_k: array of shape (num_states)
        Posterior mean of the current latent state at time k given all
        observations at times up to and including the current time. This is
        computed by the Kalman filter during the forward pass.
    Sighat_k_k: array of shape (num_states, num_states)
        Posterior covariance of the current state at time k given all
        observations at times up to and including the current time. This is
        computed by the Kalman filter during the forward pass.
    Sighat_kp1_k: array of shape (num_states, num_states)
        Posterior covariance of the next state at time k+1 given all 
        observations at times up to and including the current time k. This is
        computed by the Kalman filter during the forward pass.
    U_k: array of shape (num_states+1, num_states+1)
        Evolution matrix between the latent state at the current time k and 
        time k+1. This matrix includes the evolution of the constant state and 
        has the structure U = [[0, --0--], [b, --A--]].
    
    Returns
    -------
    muhat_k: array of shape (num_states)
        Posterior mean of the latent state at time k given all observations
        along the trajectory.
    Sighat_k: array of shape (num_states, num_states)
        Posterior covariance of the latent state at time k given all 
        observations along the trajectory.
    Sighat_kkp1: array of shape (num_states, num_states)
        Posterior covariance of the latent state at time k and the latent state
        at time k+1 given all observations along the trajectory.
    """
    # extract evolution matrix and constant drift
    A_k = U_k[1:, 1:]
    b_k = U_k[1:, 0]
    
    # compute smoothing gain
    J_k = np.dot(Sighat_k_k, np.linalg.solve(Sighat_kp1_k, A_k).T)

    # mean update
    muhat_k = muhat_k_k + J_k.dot(muhat_kp1 - A_k.dot(muhat_k_k) - b_k)

    # covariance update
    Sighat_kkp1 = np.dot(J_k, Sighat_kp1)
    Sighat_k = Sighat_k_k + np.dot(J_k, \
        np.dot(Sighat_kp1 - Sighat_kp1_k, J_k.T))

    # ensure no loss of symmetry due to round-off
    Sighat_k = 0.5*(Sighat_k + Sighat_k.T)
    
    return muhat_k, Sighat_k, Sighat_kkp1

@jit(nopython=True, parallel=True)
def _Kalman_Rauch(y_data: nb.float32[:,:,:], U_matrices: nb.float32[:,:,:,:], \
        C: nb.float32[:,:], h: nb.float32[:], Sig_v: nb.float32[:,:], \
        Sig_w: nb.float32[:,:], mu_0: nb.float32[:], Sig_0: nb.float32[:,:]):
    """ This function performs the Expectation or E-step of the EM algorithm by
    computing the required parameters of the posterior latent state distribution
    given all of the observations along a collection of trajectories and the
    current model. This posterior distribution is the optimal inference 
    distribution for the EM algorithm. We use the efficient forward-backward
    algorithm of Kalman and Rauch to compute the parameters of the inference
    distribution required during the M-step.

    Parameters
    ----------
    y_data: array of shape (num_traj, len_traj, num_observations)
        Observations recorded along multiple trajectories.
    U_matrices: array of shape (num_traj, len_traj-1, num_states, num_states)
        Evolution matrices between latent states along each trajectory.
    C: array of shape (num_observations, num_states)
        Observation matrix mapping latent states to observations.
    h: array of shape (num_observations)
        Constant shift for observations
    Sig_v: array of shape (num_states, num_states)
        Process noise covariance for the latent state model.
    Sig_w: array of shape (num_observations, num_observations)
        Observation noise covariance for the model.
    mu_0: array of shape (num_states)
        Prior distribution mean for the latent state initial condition.
    Sig_0: array of shape (num_states, num_states)
        Prior distribution covariance for the latent state initial condition.


    Returns
    -------
    muhats: array of shape (num_traj, len_traj, num_states)
        Posterior means of the latent states given all observations along 
        trajectories.
    Sighats_kk: array of shape (num_traj, len_traj, num_states, num_states)
        Posterior covariances of the latent states given all observations along
        trajectories.
    Sighats_kkp1: array of shape (num_traj, len_traj-1, num_states, num_states)
        Posterior covariances of latent states at successive times given all
        observations along trajectories.
    """
    # number of independent trajectories
    n_traj = y_data.shape[0]
    # number of observations along each trajectory
    n_t = y_data.shape[1]
    # dimension of the latent state in the dynamical model
    dim_z = mu_0.shape[0]

    # optimal state estimates
    muhats = np.zeros((n_traj, n_t, dim_z))

    # covariance estimates
    Sighats_kk = np.zeros((n_traj, n_t, dim_z, dim_z))
    Sighats_kkp1 = np.zeros((n_traj, n_t-1, dim_z, dim_z))

    for j in range(n_traj):
        ## forward pass (Kalman filter)
        # temporary variables
        muhats_k_k = np.zeros((n_t, dim_z))
        Sighats_k_k = np.zeros((n_t, dim_z, dim_z))
        Sighats_kp1_k = np.zeros((n_t-1, dim_z, dim_z))

        # initialize recursion with initial condition estimates
        muhat_0_0, Sighat_0_0 = _Kalman_init(y_0 = y_data[j,0,:] - h, \
            mu_0 = mu_0, Sig_0 = Sig_0, C = C, Sig_w = Sig_w)
        muhats_k_k[0, :] = muhat_0_0
        Sighats_k_k[0, :, :] = Sighat_0_0

        # Kalman filter recursion
        for k in range(1, n_t):
            muhat_k_k, Sighat_k_k, Sighat_k_km1 = _Kalman_update(\
                y_k = y_data[j,k,:] - h, muhat_km1_km1 = muhats_k_k[k-1,:], \
                Sighat_km1_km1 = Sighats_k_k[k-1,:,:], \
                U_km1 = U_matrices[j,k-1,:,:], C = C, Sig_v = Sig_v, \
                Sig_w = Sig_w)
            muhats_k_k[k, :] = muhat_k_k
            Sighats_k_k[k, :, :] = Sighat_k_k
            Sighats_kp1_k[k-1, :, :] = Sighat_k_km1
        
        ## backward pass (Rauch smoother)
        # initialize recursion with Kalman filter estimates at final time
        muhats[j, n_t-1, :] = muhats_k_k[n_t-1, :]
        Sighats_kk[j, n_t-1, :, :] = Sighats_k_k[n_t-1, :, :]

        for kback in range(1, n_t):
            k = n_t - kback - 1
            muhat_k, Sighat_k, Sighat_kkp1 = _Rauch_update(\
                muhat_kp1 = muhats[j,k+1,:], \
                Sighat_kp1 = Sighats_kk[j,k+1,:,:], \
                muhat_k_k = muhats_k_k[k,:], \
                Sighat_k_k = Sighats_k_k[k, :, :], \
                Sighat_kp1_k = Sighats_kp1_k[k, :, :], \
                U_k = U_matrices[j,k,:,:])
            muhats[j, k, :] = muhat_k
            Sighats_kk[j, k, :, :] = Sighat_k
            Sighats_kkp1[j, k, :, :] = Sighat_kkp1
    
    return muhats, Sighats_kk, Sighats_kkp1

@jit(nopython=True, parallel=True)
def _compute_y_estimates(muhats: nb.float32[:,:,:], \
    Sighats_kk: nb.float32[:,:,:,:], C: nb.float32[:,:], h: nb.float32[:], \
    Sig_w: nb.float32[:,:]):
    """Helper function for get_y_estimates that Compute means and covariances of 
    observation variables from means and covariances of the latent states. This 
    is useful for predicting the observations from a predicted latent state 
    trajectory.

    Parameters
    ----------
    muhats: array of shape (num_traj, len_traj, num_states)
        Means of the distribution over latent states.
    Sighats_kk: array of shape (num_traj, len_traj, num_states, num_states)
        Covariances of the distribution over latent states.
    C: array of shape (num_observations, num_states)
        Observation matrix mapping latent states to observations.
    h: array of shape (num_observations)
        Constant shift for observations
    Sig_w: array of shape (num_observations, num_observations)
        Observation noise covariance for the model.
    
    Returns
    -------
    yhats: array of shape (num_traj, len_traj, num_obs)
        Means of the observations.
    Sighats_y: array of shape (num_traj, len_traj, num_obs, num_obs)
        Covariances of the observations.
    """
    
    n_traj = muhats.shape[0]
    n_t = muhats.shape[1]
    y_dim = C.shape[0]

    yhats = np.zeros((n_traj, n_t, y_dim))
    Sighats_y = np.zeros((n_traj, n_t, y_dim, y_dim))

    for j in range(n_traj):
        for k in range(n_t):
            yhats[j,k,:] = C.dot(muhats[j,k,:]) + h
            Sighats_y[j,k,:,:] = Sig_w + np.dot(C, \
                np.dot(Sighats_kk[j,k,:,:], C.T))
    
    return yhats, Sighats_y

@jit(nopython=True, parallel=True)
def _optimize_initial_condition(muhats: nb.float32[:,:,:], \
    Sighats_kk: nb.float32[:,:,:,:]):
    """Optimize the prior distribution for the latent state's initial condition
    during the maximization or M-step of the EM algorithm.

    Parameters
    ----------
    muhats: array of shape (num_traj, len_traj, num_states)
        Posterior means of the latent states given all observations along 
        trajectories. This is computed during the E-step by Kalman_Rauch.
    Sighats_kk: array of shape (num_traj, len_traj, num_states, num_states)
        Posterior covariances of the latent states given all observations along
        trajectories. This is computed during the E-step by Kalman_Rauch

    Returns
    -------
    mu_0: array of shape (num_states)
        Updated prior distribution mean for the latent state initial condition.
    Sig_0: array of shape (num_states, num_states)
        Updated prior distribution covariance for the latent state initial 
        condition.
    """
    n_traj = muhats.shape[0]

    # compute the average initial condition
    mu_0 = np.sum(muhats[:,0,:], axis=0) / n_traj

    # compute the covariance of the initial condtion
    Sig_0 = np.zeros((Sighats_kk.shape[2], Sighats_kk.shape[3]))
    for j in range(n_traj):
        Sig_0 = Sig_0 + (Sighats_kk[j,0,:,:] + \
            np.outer(muhats[j,0,:] - mu_0, muhats[j,0,:] - mu_0)) / n_traj
    
    return mu_0, Sig_0

@jit(nopython=True, parallel=True)
def _optimize_observation_shift(y_data: nb.float32[:,:,:], \
    muhats: nb.float32[:,:,:], C: nb.float32[:,:]):
    """Optimize the constant shift applied to the observations during the
    maximization or M-step or the EM algorithm.

    Parameters
    ----------
    y_data: array of shape (num_traj, len_traj, num_observations)
        Observations recorded along trajectories.
    muhats: array of shape (num_traj, len_traj, num_states)
        Posterior means of the latent states given all observations along 
        trajectories. This is computed during the E-step by Kalman_Rauch.
    C: array of shape (num_observations, num_states)
        Matrix mapping states to observations.
    
    Returns
    -------
    h: array of shape (num_observations)
        Constant shift applied to observations
    """
    n_traj = y_data.shape[0]
    n_t = y_data.shape[1]

    h = np.zeros(y_data.shape[2])
    for j in range(n_traj):
        for k in range(n_t):
            h = h + (y_data[j,k,:] - C.dot(muhats[j,k,:]))
    
    h = h / (y_data.shape[0]*y_data.shape[1])

    return h

@jit(nopython=True, parallel=True)
def _optimize_observation_map(y_data: nb.float32[:,:,:], \
    muhats: nb.float32[:,:,:], Sighats_kk: nb.float32[:,:,:,:]):
    """Optimize the observation map and observation noise covariance during the
    maximization or M-step of the EM algorithm.

    Parameters
    ----------
    y_data: array of shape (num_traj, len_traj, num_observations)
        Observations recorded along trajectories.
    muhats: array of shape (num_traj, len_traj, num_states)
        Posterior means of the latent states given all observations along 
        trajectories. This is computed during the E-step by Kalman_Rauch.
    Sighats_kk: array of shape (num_traj, len_traj, num_states, num_states)
        Posterior covariances of the latent states given all observations along
        trajectories. This is computed during the E-step by Kalman_Rauch

    Returns
    -------
    C: array of shape (num_observations, num_states)
        Updated observation matrix mapping latent states to observations.
    h: array of shape (num_observations)
        Constant shift applied to observations
    Sig_w: array of shape (num_observations, num_observations)
        Updated observation noise covariance for the model.
    """
    n_traj = y_data.shape[0]
    n_t = y_data.shape[1]

    # compute the C matrix by solving a least-squares problem
    G = np.zeros((Sighats_kk.shape[2]+1, Sighats_kk.shape[3]+1))
    Ymu = np.zeros((y_data.shape[2], muhats.shape[2]+1))
    for j in range(n_traj):
        for k in range(n_t):
            G[0,0] = G[0,0] + 1
            G[0,1:] = G[0,1:] + muhats[j,k,:]
            G[1:,0] = G[1:,0] + muhats[j,k,:]
            G[1:,1:] = G[1:,1:] + Sighats_kk[j,k,:,:] + \
                np.outer(muhats[j,k,:], muhats[j,k,:])
            Ymu[:,0] = Ymu[:,0] + y_data[j,k,:]
            Ymu[:,1:] = Ymu[:,1:] + np.outer(y_data[j,k,:], muhats[j,k,:])
    
    hC = np.linalg.solve(G, Ymu.T).T
    h = hC[:,0]
    C = hC[:,1:]

    # compute the noise covariance
    Sig_w = np.zeros((y_data.shape[2], y_data.shape[2]))
    for j in range(n_traj):
        for k in range(n_t):
            err = y_data[j,k,:] - h - C.dot(muhats[j,k,:])
            Sig_w = Sig_w + np.dot(C, np.dot(Sighats_kk[j,k,:,:], C.T)) + \
                np.outer(err, err)
    
    Sig_w = Sig_w / (y_data.shape[0]*y_data.shape[1])

    return C, h, Sig_w

@jit(nopython=True, parallel=True)
def _optimize_observation_noise(C: nb.float32[:,:,:], h: nb.float32[:], \
    y_data: nb.float32[:,:,:], muhats: nb.float32[:,:,:], \
    Sighats_kk: nb.float32[:,:,:,:]):
    """Optimize the observation noise covariance while keeping the observation
    matrix fixed during the maximization or M-step of the EM algorithm. In many
    cases we will want to fix the observation matix due to non-uniquness under
    linear transformations of the latent state. On the other hand, we will
    usually still want to determine the appropriate level of observation noise.

    Parameters
    ----------
    C: array of shape (num_observations, num_states)
        Observation matrix mapping latent states to observations. Not to be
        updated during the M-step.
    h: array of shape (num_observations)
        Constant shift applied to observation map.
    y_data: array of shape (num_traj, len_traj, num_observations)
        Observations recorded along trajectories.
    muhats: array of shape (num_traj, len_traj, num_states)
        Posterior means of the latent states given all observations along 
        trajectories. This is computed during the E-step by Kalman_Rauch.
    Sighats_kk: array of shape (num_traj, len_traj, num_states, num_states)
        Posterior covariances of the latent states given all observations along
        trajectories. This is computed during the E-step by Kalman_Rauch

    Returns
    -------
    Sig_w: array of shape (num_observations, num_observations)
        Updated observation noise covariance for the model.
    """
    n_traj = y_data.shape[0]
    n_t = y_data.shape[1]

    # compute the noise covariance
    Sig_w = np.zeros((y_data.shape[2], y_data.shape[2]))
    for j in range(n_traj):
        for k in range(n_t):
            err = y_data[j,k,:] - h - C.dot(muhats[j,k,:])
            Sig_w = Sig_w + np.dot(C, np.dot(Sighats_kk[j,k,:,:], C.T)) + \
                np.outer(err, err)
    
    Sig_w = Sig_w / (y_data.shape[0]*y_data.shape[1])

    return Sig_w

@jit(nopython=True, parallel=True)
def _optimize_explicit_Euler_generator_matrices(u_data: nb.float32[:,:,:], \
    muhats: nb.float32[:,:,:], Sighats_kk: nb.float32[:,:,:,:], \
    Sighats_kkp1: nb.float32[:,:,:,:]):
    """Optimize the matrix approximations of the Koopman generators for each
    component vector field in the control-affine model during the maximization 
    or M-step of the EM algorithm. This function uses the short-time explicit
    Euler approximation for the evolution matrices in order to compute the
    generator approximations analytically and with higher computational 
    efficiency.

    Parameters
    ----------
    u_data: array of shape (num_traj, len_traj-1, num_inputs)
        Inputs to the system along each trajectory.
    muhats: array of shape (num_traj, len_traj, num_states)
        Posterior means of the latent states given all observations along 
        trajectories. This is computed during the E-step by Kalman_Rauch.
    Sighats_kk: array of shape (num_traj, len_traj, num_states, num_states)
        Posterior covariances of the latent states given all observations along
        trajectories. This is computed during the E-step by Kalman_Rauch.
    Sighats_kkp1: array of shape (num_traj, len_traj-1, num_states, num_states)
        Posterior covariances of latent states at successive times given all
        observations along trajectories. This is computed during the E-step by 
        Kalman_Rauch.

    Returns
    -------
    V_matrices: array of shape (num_inputs, num_states+1, num_states+1)
        Matrix approximations of Koopman generators associated with each 
        component vector field giving rise to short-time evolution matrices
        U_k = I + u0(t_k)*V0 + ... + uq(t_k)*Vq.
    """

    n_traj = u_data.shape[0]
    n_t = u_data.shape[1]
    u_dim = u_data.shape[2]
    z_dim = muhats.shape[2]

    G = np.zeros((u_dim*(z_dim+1), u_dim*(z_dim+1)))
    T = np.zeros((z_dim, u_dim*(z_dim+1)))

    for j in range(n_traj):
        for k in range(n_t):
            G_k = np.zeros((z_dim+1, z_dim+1))
            G_k[0,0] = 1.0
            G_k[0,1:] = muhats[j,k,:]
            G_k[1:,0] = muhats[j,k,:]
            G_k[1:,1:] = Sighats_kk[j,k,:,:] + \
                np.outer(muhats[j,k,:], muhats[j,k,:])
            
            H_k = np.zeros((z_dim, z_dim+1))
            H_k[:,0] = muhats[j,k+1,:] - muhats[j,k,:]
            H_k[:,1:] = Sighats_kkp1[j,k,:,:] - Sighats_kk[j,k,:,:] \
                + np.outer(muhats[j,k+1,:] - muhats[j,k,:], muhats[j,k,:])

            G = G + np.kron(np.outer(u_data[j,k,:], u_data[j,k,:]), G_k)
            T = T + np.kron(u_data[j,k,:], H_k)
    
    # compute V matrices
    bA_hstacked = np.transpose(np.linalg.solve(G, T.T))
    # lam_min = np.min(np.linalg.eigvalsh(G))
    # if lam_min > 0.0:
    #     bA_hstacked = np.transpose(np.linalg.solve(G, T.T))
    # else:
    #     bA_hstacked = np.transpose(np.linalg.lstsq(G, T.T)[0])
    V_matrices = np.zeros((u_dim, z_dim+1, z_dim+1))
    for i in range(u_dim):
        V_matrices[i,1:,:] = bA_hstacked[:,(z_dim+1)*i:(z_dim+1)*(i+1)]
    
    return V_matrices

@jit(nopython=True, parallel=True)
def _assemble_dynamical_loss_matrix(U_matrices: nb.float32[:,:,:,:], \
    muhats: nb.float32[:,:,:], Sighats_kk: nb.float32[:,:,:,:], \
    Sighats_kkp1: nb.float32[:,:,:,:]):
    """Helper function for _dynamical_loss_and_gradient that computes the 
    component of the loss function during the M-step assocated with the latent 
    state dynamics.

    When the optimal Sig_v is used, the dynamical loss can be written as
    L = log det(G). The optimal Sig_v is proportional to G. This function 
    assembles G and computes L. L is used as the optimization objective for the 
    generator matrices.

    Parameters
    ----------
    U_matrices: array of shape (num_traj, len_traj-1, num_states, num_states)
        Evolution matrices between latent states along each trajectory computed
        using a given set of generator matrices.
    muhats: array of shape (num_traj, len_traj, num_states)
        Posterior means of the latent states given all observations along 
        trajectories. This is computed during the E-step by Kalman_Rauch.
    Sighats_kk: array of shape (num_traj, len_traj, num_states, num_states)
        Posterior covariances of the latent states given all observations along
        trajectories. This is computed during the E-step by Kalman_Rauch.
    Sighats_kkp1: array of shape (num_traj, len_traj-1, num_states, num_states)
        Posterior covariances of latent states at successive times given all
        observations along trajectories. This is computed during the E-step by 
        Kalman_Rauch.

    Returns
    -------
    L: float
        Minimization objective for the optimizing the generator matrices during
        the M-step.
    G: array of shape (num_states, num_states)
        Dynamical loss matrix, proportional to the optimal process noise
        covariance.
    """
    
    n_traj = muhats.shape[0]
    n_t = muhats.shape[1]
    z_dim = muhats.shape[2]

    G = np.zeros((z_dim, z_dim))
    for j in range(n_traj):
        for k in range(n_t-1):
            A = U_matrices[j,k,1:,1:]
            b = U_matrices[j,k,1:,0]
            AS = np.dot(A, Sighats_kkp1[j,k,:,:])
            ASAT = np.dot(A, np.dot(Sighats_kk[j,k,:,:], A.T))
            err = muhats[j,k+1,:] - A.dot(muhats[j,k,:]) - b
            G = G + Sighats_kk[j,k+1,:,:] - AS - AS.T + ASAT + \
                np.outer(err,err)
    
    # value of dynamical loss function
    _, L = np.linalg.slogdet(G)
    
    return L, G

@jit(nopython=True, parallel=True)
def _dynamical_loss_and_gradient(V_matrices: nb.float32[:,:,:], \
    u_data: nb.float32[:,:,:], muhats: nb.float32[:,:,:], \
    Sighats_kk: nb.float32[:,:,:,:], Sighats_kkp1: nb.float32[:,:,:,:]):
    """Compute the dynamical loss objective function for the generator matrices
    during the M-step along with its gradient. This function uses the implicit
    Euler approximation of the matrix exponential to compute evolution matrices 
    that are still stable over longer time-steps.

    Parameters
    ----------
    V_matrices: array of shape (num_inputs, num_states, num_states)
        Matrix approximations of Koopman generators associated with each
        component vector field.
    u_data: array of shape (num_traj, len_traj-1, num_inputs)
        Inputs to the system along each trajectory.
    muhats: array of shape (num_traj, len_traj, num_states)
        Posterior means of the latent states given all observations along 
        trajectories. This is computed during the E-step by Kalman_Rauch.
    Sighats_kk: array of shape (num_traj, len_traj, num_states, num_states)
        Posterior covariances of the latent states given all observations along
        trajectories. This is computed during the E-step by Kalman_Rauch.
    Sighats_kkp1: array of shape (num_traj, len_traj-1, num_states, num_states)
        Posterior covariances of latent states at successive times given all
        observations along trajectories. This is computed during the E-step by 
        Kalman_Rauch.

    Returns
    -------
    L: float
        Minimization objective for the generator matrices during the M-step.
    grad_V: array of shape (num_inputs, num_states, num_states)
        Gradient of the objective with respect to generator matrices.
    """
    n_traj = muhats.shape[0]
    n_t = muhats.shape[1]
    u_dim = u_data.shape[2]
    z_dim = V_matrices.shape[2]-1

    # compute evolution matrices
    U_matrices = _compute_implicit_evolution_matrices(u_data, V_matrices)

    # compute the dynamical loss function
    L, G = _assemble_dynamical_loss_matrix(U_matrices, muhats, Sighats_kk, \
        Sighats_kkp1)

    # compute gradient with respect to evolution matrices
    grad_A = np.zeros(U_matrices.shape)
    for j in range(n_traj):
        for k in range(n_t-1):
            A = U_matrices[j,k,1:,1:]
            b = U_matrices[j,k,1:,0]
            err = muhats[j,k+1,:] - A.dot(muhats[j,k,:]) - b
            F = np.zeros((z_dim, z_dim+1))
            F[:,0] = err
            F[:,1:] = np.dot(A, Sighats_kk[j,k,:,:] + \
                np.outer(muhats[j,k,:], muhats[j,k,:])) - \
                Sighats_kkp1[j,k,:,:].T - \
                np.outer(muhats[j,k+1,:], muhats[j,k,:])
            grad_A[j,k,1:,:] = 2.0*np.linalg.solve(G, F)
    
    # compute gradient with respect to generator sums
    grad_S = _propagate_implicit_Euler_gradients(U_matrices, grad_A)

    # compute gradients with respect to generator matrices
    grad_V = np.zeros(V_matrices.shape)
    for i in range(u_dim):
        for j in range(n_traj):
            for k in range(n_t-1):
                grad_V[i,1:,:] = grad_V[i,1:,:] + \
                    u_data[j,k,i]*grad_S[j,k,1:,:]
    
    return L, grad_V

@jit(nopython=True, parallel=True)
def _dynamical_loss_and_gradient_vect(V_matrices_vect: nb.float32[:], \
    u_data: nb.float32[:,:,:], muhats: nb.float32[:,:,:], \
    Sighats_kk: nb.float32[:,:,:,:], Sighats_kkp1: nb.float32[:,:,:,:]):
    """Helper function for _optimize_implicit_Euler_generator_matrices that
    reshapes gradient computed by _dynamical_loss_and_gradient into a vector
    that can be used by optimization algorithms in Scipy.

    Parameters
    ----------
    V_matrices_vect: array of shape (num_inputs*num_states*num_states)
        Matrix approximations of Koopman generators associated with each
        component vector field flattened into a vector.
    u_data: array of shape (num_traj, len_traj-1, num_inputs)
        Inputs to the system along each trajectory.
    muhats: array of shape (num_traj, len_traj, num_states)
        Posterior means of the latent states given all observations along 
        trajectories. This is computed during the E-step by Kalman_Rauch.
    Sighats_kk: array of shape (num_traj, len_traj, num_states, num_states)
        Posterior covariances of the latent states given all observations along
        trajectories. This is computed during the E-step by Kalman_Rauch.
    Sighats_kkp1: array of shape (num_traj, len_traj-1, num_states, num_states)
        Posterior covariances of latent states at successive times given all
        observations along trajectories. This is computed during the E-step by 
        Kalman_Rauch.

    Returns
    -------
    L: float
        Minimization objective for the generator matrices during the M-step.
    grad_V: array of shape (num_inputs*num_states*num_states)
        Gradient of the objective with respect to generator matrices flattened
        into a vector.
    """
    z_dim = muhats.shape[2]
    u_dim = u_data.shape[2]
    L, grad_V = _dynamical_loss_and_gradient(np.reshape(V_matrices_vect, \
        (u_dim, z_dim+1, z_dim+1)), u_data, muhats, Sighats_kk, Sighats_kkp1)
    
    grad_V_vect = np.reshape(grad_V, (-1))
    return L, grad_V_vect

def _optimize_implicit_Euler_generator_matrices(V_matrices_guess, u_data, \
    muhats, Sighats_kk, Sighats_kkp1, maxiter=10):
    """Optimize the matrix approximations of the Koopman generators for each
    component vector field in the control-affine model during the maximization 
    or M-step of the EM algorithm. This function approximates the short-time
    evolution matrices via implicit Euler discretization using linear 
    combinations of generators.

    We use the quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno
    (BFGS) to optimize the generator matrices.

    Parameters
    ----------
    V_matrices_guess: array of shape (num_inputs, num_states, num_states)
        Initial guess for the matrix approximations of Koopman generators 
        associated with each component vector field.
    u_data: array of shape (num_traj, len_traj-1, num_inputs)
        Inputs to the system along each trajectory.
    muhats: array of shape (num_traj, len_traj, num_states)
        Posterior means of the latent states given all observations along 
        trajectories. This is computed during the E-step by Kalman_Rauch.
    Sighats_kk: array of shape (num_traj, len_traj, num_states, num_states)
        Posterior covariances of the latent states given all observations along
        trajectories. This is computed during the E-step by Kalman_Rauch.
    Sighats_kkp1: array of shape (num_traj, len_traj-1, num_states, num_states)
        Posterior covariances of latent states at successive times given all
        observations along trajectories. This is computed during the E-step by 
        Kalman_Rauch.
    maxiter: integer, optional
        Maximum number of interations of the optimization algorithm to perform.
        Default value is 10.

    Returns
    -------
    V_matrices: array of shape (num_inputs, num_states, num_states)
        Matrix approximations of Koopman generators associated with each 
        component vector field.
    """
    z_dim = muhats.shape[2]
    u_dim = u_data.shape[2]

    # BFGS optimization
    sol = sp.optimize.minimize(_dynamical_loss_and_gradient_vect, \
        np.reshape(V_matrices_guess, (-1)), method='bfgs', jac=True, \
        options={'maxiter': maxiter}, \
        args=(u_data, muhats, Sighats_kk, Sighats_kkp1))
    
    V_matrices = np.reshape(sol.x, (u_dim, z_dim+1, z_dim+1))

    for i in range(u_dim):
        V_matrices[i,0,:] = 0.0
    
    return V_matrices

@jit(nopython=True, parallel=True)
def _optimize_process_noise_from_A(U_matrices: nb.float32[:,:,:,:], \
    muhats: nb.float32[:,:,:], Sighats_kk: nb.float32[:,:,:,:], \
    Sighats_kkp1: nb.float32[:,:,:,:]):
    """Optimize the process noise covariance of the model during the
    Maximization or M-step of the EM algorithm.

    This function relies on the evolution matrices computed using the optimal
    generators found during the current M-step.

    Parameters
    ----------
    U_matrices: array of shape (num_traj, len_traj-1, num_states, num_states)
        Evolution matrices between latent states along each trajectory computed
        using the optimal generator matrices during the current M-step.
    muhats: array of shape (num_traj, len_traj, num_states)
        Posterior means of the latent states given all observations along 
        trajectories. This is computed during the E-step by Kalman_Rauch.
    Sighats_kk: array of shape (num_traj, len_traj, num_states, num_states)
        Posterior covariances of the latent states given all observations along
        trajectories. This is computed during the E-step by Kalman_Rauch.
    Sighats_kkp1: array of shape (num_traj, len_traj-1, num_states, num_states)
        Posterior covariances of latent states at successive times given all
        observations along trajectories. This is computed during the E-step by 
        Kalman_Rauch.
    
    Returns
    -------
    Sig_v: array of shape (num_states, num_states)
        Updated process noise covariance for latent state model.
    """
    
    n_traj = muhats.shape[0]
    n_t = muhats.shape[1]
    z_dim = muhats.shape[2]

    Sig_v = np.zeros((z_dim, z_dim))
    for j in range(n_traj):
        for k in range(n_t-1):
            A = U_matrices[j,k,1:,1:]
            b = U_matrices[j,k,1:,0]
            AS = np.dot(A, Sighats_kkp1[j,k,:,:])
            ASAT = np.dot(A, np.dot(Sighats_kk[j,k,:,:], A.T))
            err = muhats[j,k+1,:] - A.dot(muhats[j,k,:]) - b
            Sig_v = Sig_v + Sighats_kk[j,k+1,:,:] - AS - AS.T + ASAT + \
                np.outer(err,err)
    
    Sig_v = Sig_v / (n_traj*(n_t-1.0))

    return Sig_v


@jit(nopython=True, parallel=True)
def _compute_log_likelihood(y_data: nb.float32[:,:,:], \
    U_matrices: nb.float32[:,:,:,:], Sig_v: nb.float32[:,:], \
    C: nb.float32[:,:], h: nb.float32[:], Sig_w: nb.float32[:,:], \
    mu_0: nb.float32[:], Sig_0: nb.float32[:,:], num_off_diag_blocks):
    """Compute total log likelihood of the data using the current model 
    parameters. The parameters of the inference distribution
    are computed using the current model parameters, that is, after the E-step
    and before the next M-step.

    Parameters
    ----------
    y_data: array of shape (num_traj, len_traj, num_observations)
        Observations recorded along trajectories.
    U_matrices: array of shape (num_traj, len_traj-1, num_states, num_states)
        Evolution matrices between latent states along each trajectory.
    Sig_v: array of shape (num_states, num_states)
        Process noise covariance for latent state model.
    C: array of shape (num_observations, num_states)
        Observation matrix mapping latent states to observations.
    h: array of shape (num_observations)
        Constant shift for observations
    Sig_w: array of shape (num_observations, num_observations)
        Observation noise covariance for the model.
    mu_0: array of shape (num_states)
        Prior distribution mean for the latent state initial condition.
    Sig_0: array of shape (num_states, num_states)
        Prior distribution covariance for the latent state initial 
        condition.
    num_off_diag_blocks: integer
        Number of off-diagonal blocks to use when assembling the observation
        covariance matrix. Using num_off_diag_blocks = 0 gives a block-diagonal
        approximation for the covariance while using num_off_diag_blocks = n_t-1
        gives the full covariance matrix.
    
    Returns
    -------
    L: float
        Log likelihood for the entire model.
    """
    
    n_traj = y_data.shape[0]
    n_t = y_data.shape[1]
    z_dim = mu_0.shape[0]
    y_dim = y_data.shape[2]

    # initialize log likelihood
    L = 0.0

    for j in range(n_traj):
        # construct expected observed signal
        y_expected = np.zeros((n_t, y_dim))
        x = mu_0
        y_expected[0,:] = C.dot(x) + h
        for k in range(1,n_t):
            A = U_matrices[j,k-1,1:,1:]
            b = U_matrices[j,k-1,1:,0]
            x = A.dot(x) + b
            y_expected[k,:] = C.dot(x) + h
        
        # construct observation covariance matrix
        Sig_YY = np.zeros((n_t*y_dim, n_t*y_dim))

        # assemble covariance matrices
        S = Sig_0 # block diagonal state covariance entries
        for k in range(n_t):
            # block diagonal observation covariance entries
            Sig_YY[k*y_dim:(k+1)*y_dim, k*y_dim:(k+1)*y_dim] = \
                np.dot(C, np.dot(S, C.T)) + Sig_w
            
            # off-diagonal entries
            P = S
            for l in range(k+1, min(n_t, k+1+num_off_diag_blocks)):
                # extract evolution matrix
                A = U_matrices[j,l-1,1:,1:]
                
                # update latent state covariance going across row block-wise
                P = np.dot(P, A.T)

                # observation covariance
                Sig_YY[k*y_dim:(k+1)*y_dim, l*y_dim:(l+1)*y_dim] = \
                    np.dot(C, np.dot(P, C.T))
                Sig_YY[l*y_dim:(l+1)*y_dim, k*y_dim:(k+1)*y_dim] = \
                    Sig_YY[k*y_dim:(k+1)*y_dim, l*y_dim:(l+1)*y_dim].T

            if k < n_t-1:
                # extract evolution matrix
                A = U_matrices[j,k,1:,1:]

                # update block diagonal state covariance entries
                S = np.dot(A, np.dot(S, A.T)) + Sig_v
        
        # compute log likelihood for the trajectory
        y_err = np.reshape(y_data[j,:,:] - y_expected, (-1))

        L_Sig_YY = np.linalg.cholesky(Sig_YY)
        v_err = np.linalg.solve(L_Sig_YY, y_err)

        log_det = np.sum(np.log(np.sqrt(2.0*np.pi) \
            * np.absolute(np.diag(L_Sig_YY))))

        L = L - log_det - 0.5*np.dot(v_err, v_err)
    
    return L

@jit(nopython=True, parallel=True)
def _forward_trajectory_explicit(z_0: nb.float32[:], u_traj: nb.float32[:,:], \
    V_matrices: nb.float32[:,:,:], C: nb.float32[:,:], h: nb.float32[:]):
    """Predict a single trajectory with given input and initial condition using
    explicit Euler time stepping.

    Parameters
    ----------
    z_0: array of shape (num_states)
        Initial condition for model's latent states.
    u_traj: array of shape (len_traj-1, num_inputs)
        Inputs along the trajectory (assumed to be constant over each interval).
    V_matrices: array of shape (num_inputs, num_states+1, num_states+1)
        Matrix approximation of Koopman generators for component vector fields.
    C: array of shape (num_observations, num_states)
        Linear mapping between non-trivial latent states and observations.
    h: array of shape (num_observations):
        Constant offset for observations.
    
    Returns
    -------
    z_traj: array of shape (len_traj, num_states)
        Predicted state trajectory
    y_traj: array of shape (len_traj, num_observations)
        Predicted observations along trajectory
    """
    n_t = u_traj.shape[0]+1
    z_dim = z_0.shape[0]
    u_dim = V_matrices.shape[0]
    y_dim = C.shape[0]

    z_traj = np.zeros((n_t, z_dim))
    y_traj = np.zeros((n_t, y_dim))

    # initial condition
    x = np.zeros((z_dim+1))
    x[0] = 1.0
    x[1:] = z_0[:]
    z_traj[0,:] = x[1:]
    y_traj[0,:] = C.dot(x[1:]) + h

    # predict the dynamics
    for k in range(n_t-1):
        # construct the generator
        V = np.zeros((z_dim+1, z_dim+1))
        for i in range(u_dim):
            V = V + u_traj[k,i]*V_matrices[i,:,:]

        # update the model state
        x = np.dot(np.eye(z_dim+1)+V, x)

        # observations and non-trivial model states
        z_traj[k+1,:] = x[1:]
        y_traj[k+1,:] = C.dot(x[1:]) + h
    
    return z_traj, y_traj

@jit(nopython=True, parallel=True)
def _forward_trajectory_implicit(z_0: nb.float32[:], u_traj: nb.float32[:,:], \
    V_matrices: nb.float32[:,:,:], C: nb.float32[:,:], h: nb.float32[:]):
    """Predict a single trajectory with given input and initial condition using
    implicit Euler time stepping.

    Parameters
    ----------
    z_0: array of shape (num_states)
        Initial condition for model's latent states.
    u_traj: array of shape (len_traj-1, num_inputs)
        Inputs along the trajectory (assumed to be constant over each interval).
    V_matrices: array of shape (num_inputs, num_states+1, num_states+1)
        Matrix approximation of Koopman generators for component vector fields.
    C: array of shape (num_observations, num_states)
        Linear mapping between non-trivial latent states and observations.
    h: array of shape (num_observations):
        Constant offset for observations.
    
    Returns
    -------
    z_traj: array of shape (len_traj, num_states)
        Predicted state trajectory
    y_traj: array of shape (len_traj, num_observations)
        Predicted observations along trajectory
    """
    n_t = u_traj.shape[0]+1
    z_dim = z_0.shape[0]
    y_dim = C.shape[0]
    u_dim = V_matrices.shape[0]

    z_traj = np.zeros((n_t, z_dim))
    y_traj = np.zeros((n_t, y_dim))

    # initial condition
    x = np.zeros((z_dim+1))
    x[0] = 1.0
    x[1:] = z_0[:]
    z_traj[0,:] = x[1:]
    y_traj[0,:] = C.dot(x[1:]) + h

    # predict the dynamics
    for k in range(n_t-1):
        # construct the generator
        V = np.zeros((z_dim+1, z_dim+1))
        for i in range(u_dim):
            V = V + u_traj[k,i]*V_matrices[i,:,:]

        # update the model state
        x = np.linalg.solve(np.eye(z_dim+1)-V, x)

        # observations and non-trivial model states
        z_traj[k+1,:] = x[1:]
        y_traj[k+1,:] = C.dot(x[1:]) + h
    
    return z_traj, y_traj

@jit(nopython=True, parallel=True)
def _adjoint_trajectory_explicit(u_traj: nb.float32[:,:], \
    z_traj: nb.float32[:,:], V_matrices: nb.float32[:,:,:], C: nb.float32[:,:], \
    h: nb.float32[:], grad_L_y: nb.float32[:,:], grad_L_u: nb.float32[:,:]):
    """Predict a single trajectory with given input and initial condition using
    explicit Euler time stepping.

    Parameters
    ----------
    u_traj: array of shape (len_traj-1, num_inputs)
        Inputs along the trajectory (assumed to be constant over each interval).
    z_traj: array of shape (len_traj, num_states)
        State trajectory of model.
    V_matrices: array of shape (num_inputs, num_states+1, num_states+1)
        Matrix approximation of Koopman generators for component vector fields.
    C: array of shape (num_observations, num_states)
        Linear mapping between non-trivial latent states and observations.
    h: array of shape (num_observations):
        Constant offset for observations.
    grad_L_y: array of shape (len_traj-1, num_observations)
        Gradients of the cost function integrand with respect to the 
        observations evaluated at each point along the trajectory.
        This is to be scaled by Delta t.
    grad_L_u: array of shape (len_traj-1, num_inputs)
        Gradients of the cost function integrand with respect to the 
        inputs evaluated at each point along the trajectory.
        This is to be scaled by Delta t.
    
    Returns
    -------
    adj_traj: array of shape (len_traj, num_states+1)
        Trajectory of adjoint variables.
    grad_u_traj: array of shape (len_traj-1, num_inputs)
        Gradient of cost function with respect to the inputs along the
        trajectory. This gradient includes the factor Delta t.
    """
    n_t = u_traj.shape[0]+1
    u_dim = u_traj.shape[1]
    z_dim = z_traj.shape[1]
    y_dim = C.shape[0]

    hC = np.zeros((y_dim, z_dim+1))
    hC[:,0] = h
    hC[:,1:] = C

    adj_traj = np.zeros((n_t, z_dim+1))
    grad_u_traj = np.zeros((n_t-1, u_dim))

    # final condition
    lam = np.zeros((z_dim+1))
    adj_traj[n_t-1,:] = lam

    # adjoint dynamics
    for k_back in range(n_t-1):
        # time index
        k = n_t-2-k_back

        # construct the generator
        V = np.zeros((z_dim+1, z_dim+1))
        for i in range(u_dim):
            V = V + u_traj[k,i]*V_matrices[i,:,:]

        # update the adjoint variable
        lam = np.dot(np.eye(z_dim+1)+V.T, lam) + np.dot(hC.T, grad_L_y[k,:])
        adj_traj[k,:] = lam

        # compute gradient with respect to inputs
        x = np.zeros((z_dim+1))
        x[0] = 1.0
        x[1:] = z_traj[k,:]
        for i in range(u_dim):
            grad_u_traj[k,i] = np.dot(np.dot(V_matrices[i,:,:], x), lam) + \
                grad_L_u[k,i]
    
    return adj_traj, grad_u_traj

@jit(nopython=True, parallel=True)
def _adjoint_trajectory_implicit(u_traj: nb.float32[:,:], \
    z_traj: nb.float32[:,:], V_matrices: nb.float32[:,:,:], C: nb.float32[:,:], \
    h: nb.float32[:], grad_L_y: nb.float32[:,:], grad_L_u: nb.float32[:,:]):
    """Predict a single trajectory with given input and initial condition using
    implicit Euler time stepping.

    Parameters
    ----------
    u_traj: array of shape (len_traj-1, num_inputs)
        Inputs along the trajectory (assumed to be constant over each interval).
    z_traj: array of shape (len_traj, num_states)
        State trajectory of model.
    V_matrices: array of shape (num_inputs, num_states+1, num_states+1)
        Matrix approximation of Koopman generators for component vector fields.
    C: array of shape (num_observations, num_states)
        Linear mapping between non-trivial latent states and observations.
    h: array of shape (num_observations):
        Constant offset for observations.
    grad_L_y: array of shape (len_traj-1, num_observations)
        Gradients of the cost function integrand with respect to the 
        observations evaluated at each point along the trajectory.
        This is to be scaled by Delta t.
    grad_L_u: array of shape (len_traj-1, num_inputs)
        Gradients of the cost function integrand with respect to the 
        inputs evaluated at each point along the trajectory.
        This is to be scaled by Delta t.
    
    Returns
    -------
    adj_traj: array of shape (len_traj, num_states+1)
        Trajectory of adjoint variables.
    grad_u_traj: array of shape (len_traj-1, num_inputs)
        Gradient of cost function with respect to the inputs along the
        trajectory. This gradient includes the factor Delta t.
    """
    n_t = u_traj.shape[0]+1
    u_dim = u_traj.shape[1]
    z_dim = z_traj.shape[1]
    y_dim = C.shape[0]

    hC = np.zeros((y_dim, z_dim+1))
    hC[:,0] = h
    hC[:,1:] = C

    adj_traj = np.zeros((n_t, z_dim+1))
    grad_u_traj = np.zeros((n_t-1, u_dim))

    # final condition
    lam = np.zeros((z_dim+1))
    adj_traj[n_t-1,:] = lam

    # adjoint dynamics
    for k_back in range(n_t-1):
        # time index
        k = n_t-2-k_back

        # construct the generator
        V = np.zeros((z_dim+1, z_dim+1))
        for i in range(u_dim):
            V = V + u_traj[k,i]*V_matrices[i,:,:]

        # update the adjoint variable
        lam = np.linalg.solve(np.eye(z_dim+1)-V.T, \
            lam + np.dot(hC.T, grad_L_y[k,:]))
        adj_traj[k,:] = lam

        # compute gradient with respect to inputs
        x = np.zeros((z_dim+1))
        x[0] = 1.0
        x[1:] = z_traj[k,:]
        for i in range(u_dim):
            grad_u_traj[k,i] = np.dot(np.dot(V_matrices[i,:,:], x), lam) + \
                grad_L_u[k,i]
    
    return adj_traj, grad_u_traj

@jit(nopython=True, parallel=True)
def _quadratic_objective_and_gradient(u_traj: nb.float32[:,:], \
    z_0: nb.float32[:], y_ref: nb.float32[:,:], Q: nb.float32[:,:], \
    R: nb.float32[:,:], V_matrices: nb.float32[:,:,:], C: nb.float32[:,:], \
    h: nb.float32[:], explicit_time_step: nb.boolean):
    """ Computes the objective
    integral (y-y_ref)^T Q (y-y_ref) + u^T R u dt
    and its gradient with respect to the input subject to the dynamics.
    The time-step is assumed to be unity via an appropriate scaling Q, R, and
    the input.

    Parameters
    ----------
    u_traj: array of shape (len_traj-1, num_inputs)
        Inputs along the trajectory (assumed to be constant over each interval).
    z_0: array of shape (num_states)
        Initial condition for model state.
    y_ref: array of shape (len_traj, num_observations)
        Reference observations.
    Q: array of shape (num_observations, num_observations)
        Weight matrix for observations. Assumed to be symmetric, positive 
        semi-definite.
    R: array of shape (num_inputs, num_inputs)
        Weight matrix for inputs. Assumed to be symmetric, positive 
        semi-definite.
    V_matrices: array of shape (num_inputs, num_states+1, num_states+1)
        Matrix approximation of Koopman generators for component vector fields.
    C: array of shape (num_observations, num_states)
        Linear mapping between non-trivial latent states and observations.
    h: array of shape (num_observations):
        Constant offset for observations.
    explicit_time_step: Boolean
        Whether to use explicit time step or implicit time step for the dynamics
        and adjoint equation.
    
    Returns
    -------
    cost_traj: float
        Value of the cost function using the provided inputs.
    grad_u_traj: array of shape (len_traj-1, num_inputs)
        Gradient of cost function with respect to the inputs along the
        trajectory. This gradient includes the factor Delta t.
    z_traj: array of shape (len_traj, num_states)
        Predicted model trajectory.
    y_traj: array of shape (len_traj, num_observations)
        Predicted observations along the trajectory.
    adj_traj: array of shape (len_traj, num_states+1)
        Adjoint variables along the trajectory.
    """

    n_t = u_traj.shape[0]+1
    u_dim = u_traj.shape[1]
    z_dim = z_0.shape[0]
    y_dim = C.shape[0]

    # predict the trajectory
    if explicit_time_step:
        z_traj, y_traj = _forward_trajectory_explicit(z_0, u_traj, V_matrices, \
            C, h)
    else:
        z_traj, y_traj = _forward_trajectory_implicit(z_0, u_traj, V_matrices, \
            C, h)
    
    # compute cost function and gradients of the integrand
    cost_traj = 0.0
    grad_L_u = np.zeros((n_t-1, u_dim))
    grad_L_y = np.zeros((n_t-1, y_dim))

    y_err = y_traj[0,:]-y_ref[0,:]
    L_y = np.dot(np.dot(Q, y_err), y_err)
    for k in range(n_t-1):
        # gradient of integrand with respect to input
        grad_L_u[k,:] = 2.0*R.dot(u_traj[k,:])

        # update the cost function using trapezoidal method
        y_err = y_traj[k+1,:]-y_ref[k+1,:]
        L_y_next = np.dot(np.dot(Q, y_err), y_err)
        cost_traj = cost_traj + 0.5*(L_y + L_y_next)
        L_y = L_y_next

        # gradient of integrand with respect to observations
        grad_L_y[k,:] = 2.0*Q.dot(y_err)
    
    # compute the gradient with respect to the input signal by solving the
    # adjoint equation

    if explicit_time_step:
        adj_traj, grad_u_traj = _adjoint_trajectory_explicit(u_traj, z_traj, \
            V_matrices, C, h, grad_L_y, grad_L_u)
    else:
        adj_traj, grad_u_traj = _adjoint_trajectory_implicit(u_traj, z_traj, \
            V_matrices, C, h, grad_L_y, grad_L_u)
    
    return cost_traj, grad_u_traj, z_traj, y_traj, adj_traj

@jit(nopython=True, parallel=True)
def _quadratic_objective_and_gradient_vect(u_free_traj_vect: nb.float32[:], \
    u_fixed_traj: nb.float32[:,:], u_free_inds: nb.intp[:], \
    u_fixed_inds: nb.intp[:], z_0: nb.float32[:], y_ref: nb.float32[:,:], \
    Q: nb.float32[:,:], R: nb.float32[:,:], V_matrices: nb.float32[:,:,:], \
    C: nb.float32[:,:], h: nb.float32[:], explicit_time_step: nb.boolean):
    """ Vectorized version of _quadratic_objective_and_gradient that can be used
    for optimization. We allow some of the inputs to be optimized and the others
    to remain fixed.

    Parameters
    ----------
    u_free_traj_vect: array of shape (len_traj-1)*num_free_inputs
        Inputs along the trajectory (assumed to be constant over each interval)
        that are to be optimized. Their values correspond to the indices in
        u_free_inds. This is a vectorized version obtained by 
        np.reshape(u_free_traj, (-1)).
    u_fixed_traj: array of shape (len_traj-1, num_fixed_inputs)
        Inputs along the trajectory (assumed to be constant over each interval)
        that are not to be changed during optimization. Their values correspond 
        to the indices in u_fixed_inds.
    u_free_inds: integer array of shape num_free_inputs
        Indices corresponding to the inputs that are to be optimized.
    u_fixed_inds: integer array of shape num_fixed_inputs
        Indices corresponding to the inputs that are not to be optimized.
    z_0: array of shape (num_states)
        Initial condition for model state.
    y_ref: array of shape (len_traj, num_observations)
        Reference observations.
    Q: array of shape (num_observations, num_observations)
        Weight matrix for observations. Assumed to be symmetric, positive 
        semi-definite.
    R: array of shape (num_inputs, num_inputs)
        Weight matrix for inputs. Assumed to be symmetric, positive 
        semi-definite.
    V_matrices: array of shape (num_inputs, num_states+1, num_states+1)
        Matrix approximation of Koopman generators for component vector fields.
    C: array of shape (num_observations, num_states)
        Linear mapping between non-trivial latent states and observations.
    h: array of shape (num_observations):
        Constant offset for observations.
    explicit_time_step: Boolean
        Whether to use explicit time step or implicit time step for the dynamics
        and adjoint equation.
    
    Returns
    -------
    cost_traj: float
        Value of the cost function using the provided inputs.
    grad_u_free_traj: array of shape (len_traj-1, num_inputs)
        Gradient of cost function with respect to the inputs along the
        trajectory that are allowed to be optimized. This gradient includes the 
        factor Delta t.
    adj_traj: array of shape (len_traj, num_states+1)
        Adjoint variables along the trajectory.
    """

    n_t = y_ref.shape[0]
    u_dim = V_matrices.shape[0]
    z_dim = z_0.shape[0]
    y_dim = C.shape[0]

    # assemble the inputs
    u_traj = np.zeros((n_t-1, u_dim))
    u_traj[:,u_free_inds] = np.reshape(u_free_traj_vect, \
        (n_t-1, len(u_free_inds)))
    if len(u_fixed_inds) > 0:
        u_traj[:,u_fixed_inds] = u_fixed_traj

    cost_traj, grad_u_traj, _, _, _ = _quadratic_objective_and_gradient( \
        u_traj, z_0, y_ref, Q, R, V_matrices, C, h, explicit_time_step)
    
    grad_u_free_traj_vect = np.reshape(grad_u_traj[:,u_free_inds], (-1))

    return cost_traj, grad_u_free_traj_vect

class ActuatedKoopmanEM(object):
    """Expectation Maximization (EM) algorithm for identifying actuated bilinear 
    latent space models from partial observations.

    Latent space dynamics are assumed to be modeled as
    (1, z_{k+1}) = U(u_k)*(1, z_k) + (0, v_k)
    y(t_k) = h + C*z_k + w_k
    where
    U(u) = I + u0*V0 + u1*V1 + ... + uq*Vq     (explicit Euler) or
    U(u) = (I - u0*V0 + u1*V1 + ... + uq*Vq)^-1     (implicit Euler)
    are matrix approximations of the Koopman operators over each time interval.

    Note that we assume Delta t = 1 by absorbing it into u.

    Note also that the constant state is explicitly included in the model and is
    not subjected to any process noise. This means that the first row of each 
    generator matrix V_i must contain only zeros. We will often refer to the
    remaining rows of U as [b, A], i.e.,
    U = [[1, 0, ..., 0], 
         [b, ----A----]].

    The process noise, measurement noise, and modeling errors are modeled using
    independent mean-zero random variables v_k and w_k. We model the initial 
    conditions and noise according to z_0 ~ N(mu_0, Sig_0), v_k ~ N(0, Sig_v), 
    and w_k ~ N(0, Sig_w).

    Parameters
    ----------
    y_data: array of shape (num_traj, len_traj, num_observations)
        Observations recorded along each trajectory of the system.
    u_data: array of shape (num_traj, len_traj-1, num_inputs)
        Inputs to the system along each trajectory.
    V_matrices: array of shape (num_inputs, num_states+1, num_states+1)
        Initial guess for matrix approximations of Koopman generators governing
        latent state dynamics. First row of each matrix must contain only zeros.
    C: array of shape (num_observations, num_states)
        Initial guess for observation matrix mapping latent states to 
        observations.
    h: array of shape (num_observations)
        Constant shift for observations.
    Sig_v: array of shape (num_states, num_states)
        Initial guess for process noise covariance.
    Sig_w: array of shape (num_observations, num_observations)
        Initial guess for measurement noise covariance
    mu_0: array of shape (num_states)
        Initial guess for mean of the prior distribution over the initial
        condition for latent states.
    Sig_0: array of shape (num_states, num_states)
        Initial guess for covariance of the prior distribution over the initial
        condition for latent states.
    compute_log_likelihood: Boolean
        Whether to compute the initial log likelihood of the model. Default is
        False (because it's expensive).

    Attributes
    ----------
    V_matrices: array of shape (num_inputs, num_states+1, num_states+1)
        Matrix approximations of Koopman generators governing latent state 
        dynamics. The first row of each matrix is always zero since we assume
        that the model includes an additional constant state taking value 1 for 
        all time.
    U_matrices: array of shape (num_traj, len_traj-1, num_states+1, num_states+1)
        Evolution matrices governing latent state transitions.
    C: array of shape (num_observations, num_states)
        Observation matrix mapping latent states to observations.
    h: array of shape (num_observations)
        Constant shift for observations.
    Sig_v: array of shape (num_states, num_states)
        Process noise covariance.
    Sig_w: array of shape (num_observations, num_observations)
        Measurement noise covariance.
    mu_0: array of shape (num_states)
        Mean of the prior distribution over the initial condition for latent 
        states.
    Sig_0: array of shape (num_states, num_states)
        Covariance of the prior distribution over the initial condition for 
        latent states.
    muhats: array of shape (num_traj, len_traj, num_states)
        Posterior means of latent states given the observations and current 
        model.
    Sighats_kk: array of shape (num_traj, len_traj, num_states, num_states)
        Posterior covariances of latent states given the observations and 
        current model.
    Sighats_kkp1: array of shape (num_traj, len_traj-1, num_states, num_states)
        Posterior covariances of latent states at sequential time steps k and 
        k+1 given the observations and current model.
    L: float
        Negative log likelihood used as objective for maximization.
    EM_iter: integer
        Number of iterations of EM algorithm performed so far.
    """

    def __init__(self, y_data=np.zeros((1,2,1)), u_data=np.zeros((1,1,1)), \
        V_matrices=np.zeros((1,2,2)), C=np.zeros((1,1)), h=np.zeros((1)), \
        Sig_v=np.eye(1), Sig_w=np.eye(1), mu_0=np.zeros((1)), \
        Sig_0=np.eye(1), compute_log_likelihood=False):

        self.y_data = y_data
        self.u_data = u_data

        # initial model parameters
        self.V_matrices = V_matrices
        self.C = C
        self.h = h
        self.Sig_v = Sig_v
        self.Sig_w = Sig_w
        self.mu_0 = mu_0
        self.Sig_0 = Sig_0

        # model dimensions
        self.z_dim = self.V_matrices.shape[2]
        self.u_dim = self.u_data.shape[2]
        self.y_dim = self.y_data.shape[2]
        self.n_t = self.y_data.shape[1]

        print('\t inferring latent states...\n')

        # compute evolution matrices
        self.U_matrices = _compute_implicit_evolution_matrices(self.u_data, \
            self.V_matrices)

        # infer the latent model state given the parameters
        self.muhats, self.Sighats_kk, self.Sighats_kkp1 = _Kalman_Rauch(\
            self.y_data, self.U_matrices, self.C, self.h, self.Sig_v, \
            self.Sig_w, self.mu_0, self.Sig_0)
        
        # compute the initial log likelihood of model
        if compute_log_likelihood:
            self.L = _compute_log_likelihood(self.y_data, self.U_matrices, \
                self.Sig_v, self.C, self.h, self.Sig_w, self.mu_0, self.Sig_0, \
                num_off_diag_blocks = self.n_t-1)
        
        # number of EM steps
        self.EM_iter = 0

        if compute_log_likelihood:
            print('\n iter {:d} log likelihood = {:5e}\n'.format(self.EM_iter, \
                self.L))

    def run_EM_step(self, explicit_time_step=True, optimize_IC=True, \
        optimize_observation_map=False, \
        optimize_process_noise=True, compute_log_likelihood=False, \
        bfgs_iter=10):
        """Run one step of the EM algorithm. First optimize the model 
        parameters, then construct optimal latent state estimates and compute
        the log likelihood objective.

        We provide the option to optimize the different model parameters. In
        most cases we will want to keep the observation matrix C fixed, as can
        always be done by linearly transforming the latent state.

        We also provide an option to use the explicit Euler approximation for 
        the evolution matrices
        U(u) ~ I + u0*V0 + u1*V1 + ... + uq*Vq.
        This allows for a computationally efficient analytical solution during
        the M-step of the EM algorithm.
        Otherwise, gradient-based quasi-Newton optimization is performed using 
        the implicit Euler approximation
        U(u) = (I - u0*V0 + u1*V1 - ... - uq*Vq)^-1.

        Parameters
        ----------
        explicit_time_step: bool, optional
            Whether to use the computationally efficient explicit Euler
            approximation for the evolution matrices. If false, implicit Euler
            is used. Default is True.
        optimize_IC: bool, optional
            Whether to optimize the mean and covariance of the prior
            distribution for the initial condition during the M-step. Default is
            True.
        optimize_observation_map: bool, optional
            Whether to optimize the observation matrix C during the M-step.
            Default is False.
        optimize_process_noise: bool, optional
            Whether to optimize the process noise covariance during the M-step.
            Default is True.
        compute_log_likelihood: bool
            Whether to compute the log likelihood of the model
        bfgs_iter: integer, optional
            Maximum number of quasi-Newton steps to take during the M-step if 
            explicit_time_step = False.
        """
        ## M-step: update model parameters
        print('\t optimizing parameters...\n')

        # minimize dynamical loss
        print('\t\t optimizing dynamical parameters\n')

        if explicit_time_step:
            self.V_matrices = _optimize_explicit_Euler_generator_matrices(\
                self.u_data, self.muhats, self.Sighats_kk, self.Sighats_kkp1)
            self.U_matrices = _compute_explicit_evolution_matrices(\
                self.u_data, self.V_matrices)
        else:
            self.V_matrices = _optimize_implicit_Euler_generator_matrices(\
                self.V_matrices, self.u_data, self.muhats, self.Sighats_kk, \
                self.Sighats_kkp1, bfgs_iter)
            self.U_matrices = _compute_implicit_evolution_matrices(self.u_data, \
                self.V_matrices)
        
        # compute process noise
        if optimize_process_noise:
            self.Sig_v = _optimize_process_noise_from_A(self.U_matrices, \
                self.muhats, self.Sighats_kk, self.Sighats_kkp1)

        # minimize observation map loss
        if optimize_observation_map:
            print('\t\t optimizing observation map\n')
            self.C, self.h, self.Sig_w = _optimize_observation_map( \
                self.y_data, self.muhats, self.Sighats_kk)
        else:
            self.h = _optimize_observation_shift(self.y_data, self.muhats, \
                self.C)
            self.Sig_w = _optimize_observation_noise(self.C, self.h, \
                self.y_data, self.muhats, self.Sighats_kk)
        
        # minimize initial condition loss
        if optimize_IC:
            print('\t\t optimizing initial condition\n')
            self.mu_0, self.Sig_0 = _optimize_initial_condition(self.muhats, \
                self.Sighats_kk)
        
        self.EM_iter = self.EM_iter + 1

        ## E-step: latent variable inference
        print('\t inferring latent states...\n')

        # infer the latent model state given the parameters
        self.muhats, self.Sighats_kk, self.Sighats_kkp1 = _Kalman_Rauch(\
            self.y_data, self.U_matrices, self.C, self.h, self.Sig_v, \
            self.Sig_w, self.mu_0, self.Sig_0)
        
        # compute the log likelihood of the current model
        if compute_log_likelihood:
            self.L = _compute_log_likelihood(self.y_data, self.U_matrices, \
                self.Sig_v, self.C, self.h, self.Sig_w, self.mu_0, self.Sig_0, \
                num_off_diag_blocks = self.n_t - 1)

            print('\n iter {:d} log likelihood = {:5e}\n'.format(self.EM_iter, \
                self.L))

    def infer_latent_states(self, y_data, u_data, explicit_time_step = True):
        """Compute parameters of the inference distribution for a new set of 
        observations and inputs. To do this we use optimal state estimation and
        smoothing.

        We also provide an option to use the short-time approximation for the
        evolution matrices
        U(u) ~ I + u0*V0 + u1*V1 + ... + uq*Vq.
        This allows for a computationally efficient construction of evolution
        matrices.
        Otherwise, we use the implicit Euler approximation
        U(u) = (I - u0*V0 - u1*V1 - ... - uq*Vq)^-1.

        Parameters
        ----------
        y_data: array of shape (num_traj, len_traj, num_observations)
            New observations recorded along each trajectory of the system.
        u_data: array of shape (num_traj, len_traj-1, num_inputs)
            Inputs to the system along each new trajectory.
        explicit_time_step: bool, optional
            Whether to use the computationally efficient explicit Euler
            approximation for the evolution matrices. If False, implicit Euler
            is used. Default is True.
        
        Returns
        -------
        muhats: array of shape (num_traj, len_traj, num_states)
            Posterior means for latent states given the new observations and 
            current model.
        Sighats: array of shape (num_traj, len_traj, num_states, num_states)
            Posterior covariances for latent states given the new observations 
            and current model.
        """

        # compute evolution matrices
        if explicit_time_step:
            U_matrices = _compute_explicit_evolution_matrices(u_data, \
                self.V_matrices)
        else:
            U_matrices = _compute_implicit_evolution_matrices(u_data, self.V_matrices)
        
        # infer the latent states using Kalman-Rauch smoothing
        muhats, Sighats, _ = _Kalman_Rauch(y_data, U_matrices, \
            self.C, self.h, self.Sig_v, self.Sig_w, self.mu_0, self.Sig_0)
        
        return muhats, Sighats
    
    def get_y_estimates(self, muhats, Sighats):
        """Compute means and covariances of observation variables from means
        and covariances of the latent states. This is useful for predicting 
        the observations from a predicted latent state trajectory.

        Parameters
        ----------
        muhats: array of shape (num_traj, len_traj, num_states)
            Means of the distribution over latent states.
        Sighats: array of shape (num_traj, len_traj, num_states, num_states)
            Covariances of the distribution over latent states.
        
        Returns
        -------
        yhats: array of shape (num_traj, len_traj, num_obs)
            Means of the observations
        Sighats_y: array of shape (num_traj, len_traj, num_obs, num_obs)
            Covariances of the observations
        """
        yhats, Sighats_y = _compute_y_estimates(muhats, Sighats, self.C, \
            self.h, self.Sig_w)
        return yhats, Sighats_y
    
    def predict_dynamics(self, z_0, Sig_0, u_data, explicit_time_step = True):
        """Predict the latent state trajectory and observations along with the 
        uncertainty from given initial conditions and inputs using the current
        model.

        We model the initial condition for the state as a probability
        distribution that accounts for our uncertainty about the initial state.
        One might obtain this information using an optimal state estimator.

        Parameters
        ----------
        z_0: array of shape (num_traj, num_states)
            Means of the initial condition distributions.
        Sig_0: array of shape (num_traj, num_states, num_states)
            Covariances of the initial condition distributions.
        u_data: array of shape (num_traj, len_traj, num_inputs)
            Inputs to the system along each new trajectory.
        explicit_time_step: bool, optional
            Whether to use the computationally efficient explicit Euler
            approximation for the evolution matrices. If False, implicit Euler
            is used. Default is True.
        
        Returns
        -------
        Y_pred: array of shape (num_traj, len_traj, num_obs)
            Means of observed quantities along predicted trajectory.
        Sig_y_pred: array of shape (num_traj, len_traj, num_obs, num_obs)
            Covariances of observed quantities along predicted trajectory.
        Z_pred: array of shape (num_traj, len_traj, num_states, num_states)
            Means of latent states along predicted trajectory.
        Sig_z_pred: array of shape (num_traj, len_traj, num_states, num_states)
            Covariances of latent states along predicted trajectory.
        """

        # compute evolution matrices
        if explicit_time_step:
            U_matrices = _compute_explicit_evolution_matrices(u_data, \
                self.V_matrices)
        else:
            U_matrices = _compute_implicit_evolution_matrices(u_data, self.V_matrices)
        
        # predict the trajectory and uncertainty
        Y_pred, Sig_y_pred, Z_pred, Sig_z_pred = _predict_dynamics(z_0, Sig_0, \
            U_matrices, self.Sig_v, self.C, self.h, self.Sig_w)
        
        return Y_pred, Sig_y_pred, Z_pred, Sig_z_pred

    def quadratic_objective_and_gradient(self, u_traj, z_0, y_ref, Q, R, \
        explicit_time_step: nb.boolean):
        """Computes the quadratic objective
        integral (y-y_ref)^T Q (y-y_ref) + u^T R u dt
        and its gradient with respect to the input subject to the dynamics.
        The time-step is assumed to be unity via an appropriate scaling Q, R, and
        the input.

        Parameters
        ----------
        u_traj: array of shape (len_traj-1, num_inputs)
            Inputs along the trajectory (assumed to be constant over each 
            interval).
        z_0: array of shape (num_states)
            Initial condition for model state.
        y_ref: array of shape (len_traj, num_observations)
            Reference observations.
        Q: array of shape (num_observations, num_observations)
            Weight matrix for observations. Assumed to be symmetric, positive 
            semi-definite.
        R: array of shape (num_inputs, num_inputs)
            Weight matrix for inputs. Assumed to be symmetric, positive 
            semi-definite.
        explicit_time_step: Boolean
            Whether to use explicit time step or implicit time step for the 
            dynamics and adjoint equation.

        Returns
        -------
        cost_traj: float
            Value of the cost function using the provided inputs.
        grad_u_traj: array of shape (len_traj-1, num_inputs)
            Gradient of cost function with respect to the inputs along the
            trajectory. This gradient includes the factor Delta t.
        z_traj: array of shape (len_traj, num_states)
            Predicted model trajectory.
        y_traj: array of shape (len_traj, num_observations)
            Predicted observations along the trajectory.
        adj_traj: array of shape (len_traj, num_states+1)
            Adjoint variables along the trajectory.
        """

        cost_traj, grad_u_traj, z_traj, y_traj, adj_traj = \
            _quadratic_objective_and_gradient( u_traj, z_0, y_ref, Q, R, \
                self.V_matrices, self.C, self.h, explicit_time_step)
        
        return cost_traj, grad_u_traj, z_traj, y_traj, adj_traj

    def quadratic_objective_optimal_control(self, u_traj_guess, u_free_inds, \
        u_fixed_inds, z_0, y_ref, Q, R, explicit_time_step=True, maxiter=10):
        """Computes the optimal control input signal to minimize the quadratic 
        objective
        integral (y-y_ref)^T Q (y-y_ref) + u^T R u dt
        over a finite time horizon subject to the dynamics.
        The time-step is assumed to be unity via an appropriate scaling Q, R, 
        the input.

        Parameters
        ----------
        u_traj_guess: array of shape (len_traj-1, num_inputs)
            Initial guess for optimal inputs along the trajectory (assumed to be
            constant over each interval).
        u_free_inds: integer array of shape num_free_inputs
            The indices of inputs to be optimized.
        u_fixed_inds: integer array of shape num_fixed_inputs
            The indices of inputs that are not to be changed during 
            optimization. Pass an empty array if there are none.
        z_0: array of shape (num_states)
            Initial condition for model state.
        y_ref: array of shape (len_traj, num_observations)
            Reference observations.
        Q: array of shape (num_observations, num_observations)
            Weight matrix for observations. Assumed to be symmetric, positive 
            semi-definite.
        R: array of shape (num_inputs, num_inputs)
            Weight matrix for inputs. Assumed to be symmetric, positive 
            semi-definite.
        explicit_time_step: Boolean
            Whether to use explicit time step or implicit time step for the 
            dynamics and adjoint equation. Default is True.
        maxiter: integer
            Number of iterations for optimization algorithm. Default is 10.

        Returns
        -------
        u_traj_opt: array of shape (len_traj-1, num_inputs)
            Optimal inputs to use along the trajectory
        cost_traj_opt: float
            Value of the cost function using the optimal inputs.
        z_traj_opt: array of shape (len_traj, num_states)
            Predicted optimal model trajectory.
        y_traj_opt: array of shape (len_traj, num_observations)
            Predicted observations along the optimal trajectory.
        """

        n_t = u_traj_guess.shape[0] + 1
        u_dim = u_traj_guess.shape[1]

        # extract the inputs that are to remain fixed during optimization
        u_fixed_traj = u_traj_guess[:,u_fixed_inds]

        # extract the part of the initial guess that is to change during
        # optimization
        u_free_traj_guess_vect = np.reshape(u_traj_guess[:,u_free_inds], (-1))

        # BFGS optimization
        sol = sp.optimize.minimize(_quadratic_objective_and_gradient_vect, \
            u_free_traj_guess_vect, method='bfgs', jac=True, \
            options={'maxiter': maxiter}, \
            args=(u_fixed_traj, u_free_inds, u_fixed_inds, z_0, y_ref, Q, \
                R, self.V_matrices, self.C, self.h, explicit_time_step))

        # reassemble the input
        u_traj_opt = np.zeros((n_t-1, u_dim))
        u_traj_opt[:,u_free_inds] = np.reshape(sol.x, (n_t-1, len(u_free_inds)))
        u_traj_opt[:,u_fixed_inds] = u_fixed_traj
        cost_traj_opt = sol.fun

        # predicted trajectory
        if explicit_time_step:
            z_traj_opt, y_traj_opt = _forward_trajectory_explicit(z_0, \
                u_traj_opt, self.V_matrices, self.C, self.h)
        else:
            z_traj_opt, y_traj_opt = _forward_trajectory_implicit(z_0, \
                u_traj_opt, self.V_matrices, self.C, self.h)
        
        return u_traj_opt, cost_traj_opt, z_traj_opt, y_traj_opt