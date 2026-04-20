import os
import time
import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal
from scipy.optimize import minimize
from scipy.linalg import block_diag
from scipy.special import factorial, gammaln, logsumexp
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from kneed import KneeLocator, find_shape


class SpikeOsc(BaseEstimator):
    def __init__(self, Fs, max_oscillations, max_iter=100, use_PF = False, 
                 N_particles = 1000, seed=None, use_reg = True, early_stopping=True, patience=None, 
                 change_beta=False, assume_phase_lock = True, verbose=True, log_dir=None, track_params=True):

        self.Fs = Fs
        self.max_oscillations = max_oscillations
        self.max_iter = max_iter
        self.use_PF = use_PF
        self.N_particles = N_particles
        self.early_stopping = early_stopping
        if self.early_stopping:
            if patience is not None:
                self.patience = patience  
            else:
                self.patience = 20 if use_PF else 10
        self.seed = seed
        self.use_reg = use_reg
        self.change_beta = change_beta
        self.assume_phase_lock = assume_phase_lock
        
        self.verbose = verbose
        self.log_dir = log_dir
        self.track_params = track_params
        self.method = 'trust-krylov'
        self.tol = 1e-4, 
        


        if os.path.exists(log_dir) and os.path.isdir(log_dir):
            print(f"Log directory '{log_dir}' already exists. Logs will be saved here.")
        else:
            os.makedirs(log_dir, exist_ok=True)
            print(f"Log directory '{log_dir}' created.")
    

    def _get_oscillation_frequencies(self, y, n_oscillations=1, plot=False, max_frequency=None, nperseg=None):
        """Estimates initial oscillation frequencies using Welch's PSD and peak detection.
        
        Parameters
        ----------
        y : ndarray of shape (n_neurons, K)
            The binary spike train data.
        n_oscillations : int
            The number of top frequencies to return.
        plot : bool, default False
            If True, plots the PSD and identified peaks.
        """
        # Compute Power Spectral Density using Welch's method along the time axis
        # Using a window size of 2 seconds (or data length) for decent resolution
        
        if nperseg is None:
            nperseg = min(int(self.Fs * 2), y.shape[-1])
        f_raw, pxx_raw = signal.welch(y, fs=self.Fs, nperseg=nperseg)

        # Filter by max_frequency if provided
        if max_frequency is not None:
            freq_mask = f_raw <= max_frequency
            f = f_raw[freq_mask]
            pxx = pxx_raw[:, freq_mask] if pxx_raw.ndim > 1 else pxx_raw[freq_mask]
        else:
            f = f_raw
            pxx = pxx_raw

        # Average PSD across neurons to identify population-wide oscillations
        mean_pxx = np.mean(pxx, axis=0) if pxx.ndim > 1 else pxx
        
        # Detect peaks in the summary PSD
        peaks, _ = signal.find_peaks(mean_pxx, prominence=np.max(mean_pxx) * 0.05)
        
        if len(peaks) == 0:
            return [10.0] * n_oscillations # Default fallback if no peaks found
            
        # Sort detected peaks by power in descending order
        top_peaks = peaks[np.argsort(mean_pxx[peaks])[::-1]]
        
        # Return the frequencies corresponding to the top peaks
        est_freqs = f[top_peaks[:n_oscillations]].tolist()
        
        if plot:
            plt.figure(figsize=(10, 4))
            plt.semilogy(f, mean_pxx, label=f'Mean PSD, \n(nperseg={nperseg})', color='black', alpha=0.7)
            if len(est_freqs) > 0:
                selected_indices = top_peaks[:n_oscillations]
                colors = [f'C{i}' for i in range(len(est_freqs))]
                
                for ixf, (col, freq, freq_idx) in enumerate(zip(colors, est_freqs, selected_indices)):
                    plt.semilogy([f[freq_idx]], [mean_pxx[freq_idx]], color=col, marker='o', label='__nolegend__')
                    plt.axvline(freq, color=col, linestyle='--', alpha=0.3, label=f'Osc {ixf}: {freq:.2f} Hz')
            plt.title(f'Initial Frequency Selection (Top {n_oscillations})')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power/Frequency (dB/Hz)')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.show()
            
        # Ensure we return exactly n_oscillations
        while len(est_freqs) < n_oscillations:
            est_freqs.append(est_freqs[-1] if est_freqs else 10.0)
            
        return est_freqs


    def _initialize_parameters(self, y, x0=None, mu0=None, freq0=None, sigma0=None, alpha0=None, beta0=None, am=None, bm=None, gamma=10,
                               plot_init=False, max_freq_init=50, nperseg=None):
        self.y = y
        self.K = y.shape[-1]
        self.n_neurons = y.shape[0]
        self.x0 = np.zeros((2*self.max_oscillations, 1)) if x0 is None else x0

        self.mu0 = np.log(y.sum(axis=1)/self.K) if mu0 is None else mu0
        self.beta0 = [np.ones((1, 2*self.max_oscillations))]*self.n_neurons if beta0 is None else beta0
        self.freq0 = self._get_oscillation_frequencies(y, n_oscillations=self.max_oscillations, plot=plot_init, max_frequency=max_freq_init, nperseg=nperseg) if freq0 is None else freq0
        self.sigma0 = [1e-4]*self.max_oscillations if sigma0 is None else sigma0
        self.alpha0 = [0.999]*self.max_oscillations if alpha0 is None else alpha0
        
        am, bm = y.sum(axis=1), self.K
        self.am = am
        self.bm = bm
        self.gamma = gamma
        if self.verbose:
            print(f'--> Initialized parameters for {self.max_oscillations} oscillations:')
            print(f"\t\tFrequencies: {self.freq0}")
            print(f"\t\tAlphas: {np.round(self.alpha0, 4)}")
            print(f"\t\tSigmas: {np.round(self.sigma0, 4)}")

            print(f'--> Detected {self.n_neurons} neurons')
            print(f"\t\tInitialized mus: {np.round(self.mu0, 4)}")

            if self.use_reg:
                print(f"--> Using regularization with \n\t\tgamma = {self.gamma}, am = {self.am}, bm = {self.bm}")
            else:
                print(f"--> Not using regularization")
        

    def plot_raster(self, y=None, title='Spike Raster'):
        if y is None:
            y = self.y
        plt.figure(figsize=(10, 6))
        for iy, yc in enumerate(y):
            plt.eventplot(np.where(yc)[0]/self.Fs, colors=plt.cm.tab10(iy), lineoffsets=iy, linelengths=0.8)
        plt.xlabel('Time (s)')
        plt.ylabel('Neuron Index')
        plt.title(title)
        plt.show()

    def plot_psd(self, y=None, Fs=None, title='Mean Power Spectral Density', max_frequency=None, nperseg=None):    
        if y is None:
            y = self.y
        if Fs is None:
            Fs = self.Fs
        if nperseg is None:
            nperseg = min(int(Fs * 2), y.shape[-1])
        f_raw, pxx_raw = signal.welch(y, fs=self.Fs, nperseg=nperseg)

        # Filter by max_frequency if provided
        if max_frequency is not None:
            freq_mask = f_raw <= max_frequency
            f = f_raw[freq_mask]
            pxx = pxx_raw[:, freq_mask] if pxx_raw.ndim > 1 else pxx_raw[freq_mask]
        else:
            f = f_raw
            pxx = pxx_raw
        # Average PSD across neurons to identify population-wide oscillations
        mean_pxx = np.mean(pxx, axis=0) if pxx.ndim > 1 else pxx
        
        plt.figure(figsize=(10, 4))
        plt.semilogy(f, mean_pxx, label='Population Mean PSD', color='black', alpha=0.7)
        plt.title(title)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power/Frequency (dB/Hz)')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.show()

    def iterate(self):
        '''
        Iterate through the number of oscillations to fit the model.
        '''
        for osc_iter in range(1, self.max_oscillations+1):
            print('Fitting model with {} oscillation(s)'.format(osc_iter), end= ' ')
            alpha0_osc = self.alpha0[:osc_iter]
            freq0_osc = self.freq0[:osc_iter]
            sigma0_osc = self.sigma0[:osc_iter]
            beta0_osc = [b[0, :2*osc_iter] for b in self.beta0]
            assert len(beta0_osc) == self.n_neurons, f"Length of beta0_osc {len(beta0_osc)} != No. of neurons {self.n_neurons}"

            if self.log_dir is not None:
                log_file = os.path.join(self.log_dir, f"EM_log_osc{osc_iter}_{'PF' if self.use_PF else 'EKF'}.csv")
            else:
                log_file = None
            outputs = self._run_em_algorithm(self.x0[:2*osc_iter], self.y, self.mu0, alpha0_osc, freq0_osc, sigma0_osc, beta0_osc, max_iter = self.max_iter, log_file = log_file, track_params=self.track_params)
            keys = ['best_params', 'best_hidden_states', 'latest_params', 'latest_hidden_states', 'logL_history']
            if self.track_params:
                keys.append('param_history')
            setattr(self, f'model_osc{osc_iter}', dict(zip(keys, outputs)))
            print()

    def get_knee(self, plot=True, ret=False):
        '''
        Get the knee point in the log-likelihood curve to determine optimal number of oscillations.
        '''
        logL_maxs = []
        model_ixs = np.arange(1, self.max_oscillations+1)
        for osc_iter in model_ixs:
            model = getattr(self, f'model_osc{osc_iter}', None)
            if model is not None:
                logL_maxs.append(model['best_params']['logL'])
            else:
                logL_maxs.append(0.0)
        
        direction, curve = find_shape(model_ixs, logL_maxs)
        kl = KneeLocator(model_ixs, logL_maxs, curve=curve, direction=direction)
        print(f"Optimal number of oscillations: {kl.knee}")
        print(f"Log-likelihood at knee:{kl.knee_y}")
        kl.plot_knee()
        
        if ret:
            return kl.knee
        

        
    def _rt_matrix(self, array):
        """Returns rt(array) = array[1,0] - array[0,1]
    
        Inputs
        ------
        array : ndarray, shape (2, 2)

        Returns
        -------
        rt : float, rt of matrix. 
        """
        assert array.shape == (2,2), f"Expected shape (2, 2), but got {array.shape} for rt()"
        rt = array[1,0] - array[0,1]
        return rt
    
    def _rotation_matrix_omega(self, w):
        """Rotation matrix for a given angle.
        
        Inputs
        ------
        w : float, Angle in radians (scalar).
        
        Returns
        -------
        Rw : ndarray of shape (2, 2), 2x2 rotation matrix.
        """
        Rw = np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]], dtype=np.float64)
        return Rw
    
    def _rotation_matrix(self, freqs, Fs=None):
        """Rotation matrix for a given oscillation frequency(/-ies).
        
        Inputs
        ------
        f : float or array-like, frequency in Hz (scalar).
        
        Returns
        -------
        block diagonal of rotational matrices
        """
        if Fs is None:
            Fs = self.Fs
        freqs = np.asarray(freqs)
        if freqs.size == 1:
            w = 2*np.pi*freqs.item()/Fs
            return self._rotation_matrix_omega(w)
        elif freqs.squeeze().ndim == 1:
            blocks = [self._rotation_matrix(freq, Fs) for freq in freqs.squeeze()]
            return block_diag(*blocks)
        else:
            raise AssertionError(f"Input `f` to _rotation_matrix() must be a scalar or a 1D array-like, but got shape {np.asarray(freqs).shape}")
    

    def _decompose_block_diag(self, M):
        D = M.shape[-1]//2
        assert D == M.shape[-1]/2, f"Input {M} is not a block diagonal"
        if M.shape == (2, 2):
            return M.reshape(1,2,2)
        return np.array([M[d*2:(d+1)*2, d*2:(d+1)*2] for d in range(D)])

    def _k_c_likelihood(self, xp_k, Pp_k, dN_k, mu_c, beta_c, order = 3):
        """Computes the approximate log-likelihood contribution for a single time point.
        
        Parameters
        ----------
        xp_k : ndarray of shape (2*D, 1)
            Predicted state vector at time k.
        Pp_k : ndarray of shape (2*D, 2*D)
            Predicted covariance matrix at time k.
        dN_k : float
            Observed count or event at time k.
        mu_c : float
            Baseline Firing rate parameter of single neuron.
        beta_c : float
            Scaling parameter of single neuron.
        order : int, optional
            Order of the Poisson expansion for log-likelihood approximation. Default is 3.
        
        Returns
        -------
        logL_k : float
            Approximate log-likelihood for the time step k.
        """
        ### DO LSE as kxn array after subtracting max(kxn) !!!  FIXX
        ### FIX : calculate for multiple mus and multiple betas
        
        exponents = []
        signs = []
        for n in range(0, order+1):
            exp_term = (dN_k+n)*(mu_c + np.dot(beta_c, xp_k) + (dN_k+n)*np.linalg.multi_dot([beta_c, Pp_k, beta_c.T])/2) - gammaln(n + 1)
            exponents.append(exp_term)
            signs.append((-1)**n)
        log_k = logsumexp(np.array(exponents).squeeze(), b = np.array(signs))
        # log_k = np.abs(logval)*logsign
        return log_k
        
    def _log_c_likelihood(self, x_p, P_p, dN, mus, betas, order = 3):
        """Computes the total log-likelihood over all time steps.
    
        Parameters
        ----------
        x_p : ndarray of shape (2, K)
            Predicted states from the Kalman filter.
        P_p : ndarray of shape (2, 2, K)
            Predicted covariance matrices from the Kalman filter.
        dN : ndarray of shape (K,)
            Observed spike counts or event data.
        mus : float or array-like
            Baseline Firing rate parameter.
        betas : float
            Scaling parameter.
        order : int, optional
            Order of the Poisson expansion for log-likelihood approximation. Default is 3.
    
        Returns
        -------
        logL : float
            Total log-likelihood across time steps.
        """
        
        K = self.K
        logL = 0
        for k in range(K):
            for dN_c, mu_c, beta_c in zip(dN, mus, betas):
                beta_c = beta_c.reshape(1, -1)
                logL += self._k_c_likelihood(x_p[:, k:k+1], P_p[..., k], dN_c[:, k], mu_c, beta_c, order = order)
        return logL
    
    def _likelihood(self, x, xm, Pinv, dN, mus, betas):
        """Computes the negative log-likelihood of p(x_k|H(k))
        
        Inputs
        ------
        x : ndarray of shape (2D,1), Current state estimate.
        xm : ndarray of shape (2D,1), Mean of the prior distribution.
        Pinv : ndarray of shape (2D, 2D), Inverse of the prior covariance matrix.
        dN : float, Observed data.
        mus : array-like, 
        betas : float, Scaling parameter.
        
        Returns
        -------
        L : float, Negative log-likelihood value.
        """
        assert len(mus) == self.n_neurons, f"Length of mus {len(mus)} != No. of neurons {self.n_neurons}"
        x = x.reshape(-1, 1)
        dot2 = (-1/2)*np.linalg.multi_dot([(x - xm).T, Pinv, x - xm])
        L = 0
        for dN_c, mu_c, beta_c in zip(dN,  mus, betas):
            beta_c = beta_c.reshape(1, -1)
            L += dN_c*(mu_c + np.dot(beta_c,x)) - np.exp(mu_c + np.dot(beta_c,x))
        L +=  dot2
        out = -L.item() # minimize needs a scalar L value
        return out 
    
    def _jacobian(self, x, xm, Pinv, dN, mus, betas):
        """Computes the Jacobian (gradient) of the likelihood function of p(x_k|H(k)).
        
        Inputs
        ------
        x : ndarray of shape (2D,1), Current state estimate.
        xm : ndarray of shape (2D,1), Mean of the prior distribution.
        Pinv : ndarray of shape (2D, 2D), Inverse of the prior covariance matrix.
        dN : float, Observed data.
        mus : array-like, Mean parameter.
        betas : float, Scaling parameter.
        
        Returns
        -------
        jac : ndarray of shape (2D,), Jacobian vector.
        """
        assert len(mus) == self.n_neurons, f"Length of mus {len(mus)} != No. of neurons {self.n_neurons}"
        x = x.reshape(-1, 1)
        jac = -np.dot(Pinv, x - xm)
        for dN_c, mu_c, beta_c in zip(dN,  mus, betas):
            beta_c = beta_c.reshape(1, -1)
            jac += (dN_c - np.exp(mu_c + np.dot(beta_c,x)))*beta_c.T
        out = -jac.squeeze() #Minimize needs 1D jacobian
        return  out

    
    def _hessian(self, x, xm, Pinv, dN, mus, betas):
        """Computes the Hessian (second derivative) of the likelihood function of p(x_k|H(k)).
        
        Inputs
        ------
        x : ndarray of shape (2D,1), Current state estimate.
        xm : ndarray of shape (2D,1), Mean of the prior distribution.
        Pinv : ndarray of shape (2D, 2D), Inverse of the prior covariance matrix.
        dN : float, Observed data.
        mus : array-like, Mean parameter.
        betas : float, Scaling parameter.
        
        Returns
        -------
        hess : ndarray of shape (2D,2D), 2Dx2D Hessian matrix.
        """
        assert len(mus) == self.n_neurons, f"Length of mus {len(mus)} != No. of neurons {self.n_neurons}"
        x = x.reshape(-1, 1)
        hess = -Pinv.copy()
        for mu_c, beta_c in zip(mus, betas):
            beta_c = beta_c.reshape(1, -1)
            hess -= np.exp(mu_c + np.dot(beta_c,x))*np.dot(beta_c.T, beta_c)
        out = -hess
        return out

    def _max_L(self, x0, xp_k, Pinv_k, y_k, mus, betas):
        """Maximizes the likelihood function to estimate the state.
        
        Inputs
        ------
        x0 : ndarray of shape (2D,1), Initial estimate of state.
        xp_k : ndarray of shape (2D,1), Predicted state.
        Pinv_k : ndarray of shape (2D, 2D), Inverse of predicted covariance matrix.
        y_k : float, Observation datapoint.
        mus : array-like, Mean parameter.
        betas : float, Scaling parameter.
        
        Returns
        -------
        res : OptimizeResult, Result of the optimization. (From scipy.minimze)
        """
        ## Maybe FIXX? beta shape/size assertion for multiple betas
        # assert np.asarray(beta).ndim == 2, f"Expected beta.ndim = 2, but got beta.ndim = {beta.ndim}"
        if x0.ndim == 2:
            # Required because minimize() only take 1-D x0. It is converted back to 2-D within the derivative fxs
            x0 = x0[:, 0]
        res = minimize(self._likelihood, x0, args=(xp_k, Pinv_k, y_k, mus, betas), 
                           jac=self._jacobian, hess=self._hessian, method = self.method)
        return res
    
    def _KF_step(self, x_m_k1, P_m_k1, y_k, mus, phi, Q, betas):
        """Performs a single recursive step of the Kalman filter.
        
        Inputs
        ------
        x_m_k1 : ndarray of shape (2D,1), Previous posterior/measurement state estimate.
        P_m_k1 : ndarray of shape (2D, 2D), Previous posterior/measurement covariance matrix.
        y_k : float, Current observation.
        mus : array-like, Mean parameter.
        phi : ndarray of shape (2D, 2D), Transformation matrix (Latent equation).
        Q : ndarray of shape (2D, 2D), Process noise matrix 
        betas : float, Scaling parameter.
        
        
        Returns
        -------
        tuple
            xp_k : ndarray of shape (2,), Predicted state.
            xm_k : ndarray of shape (2,), Updated state estimate.
            Pp_k : ndarray of shape (2, 2), Predicted covariance.
            Pm_k : ndarray of shape (2, 2), Updated covariance estimate.
        """

        # print(np.asarray(betas).shape)
        #Prediction Step
        xp_k = np.dot(phi, x_m_k1)
        Pp_k = np.linalg.multi_dot([phi, P_m_k1, phi.T]) + Q
        
        Pinv_k = np.linalg.pinv(Pp_k)
    
        # Measurement step : maximizing likelihood function
        # Using x_{k|k-1} as initial estimate for minimize function
        res = self._max_L(x_m_k1, xp_k, Pinv_k, y_k, mus, betas)
        
        if not res.success:
            print("Estimation of $x_{m, k}$  failed to converge : " + res.message)
            raise AssertionError("Estimation of $x_{m, k}$  failed to converge : " + res.message)
        xm_k = res.x.reshape(-1, 1)
        Pm_k = np.linalg.pinv(res.hess) # Negative not required because hessian function minimizes the "negative likelihood".
        logL_k = 0
        for yk_c, mu_c, beta_c in zip(y_k, mus, betas):
            beta_c = beta_c.reshape(1, -1)
            logL_k += self._k_c_likelihood(xp_k, Pp_k, yk_c, mu_c, beta_c)
        
        ### FIX ^ : I removed it for multiple mu testing Edit: I thought I did fix this?
        return xp_k, xm_k, Pp_k, Pm_k, logL_k
        
    def _PF_step(self, x_m_k1, P_m_k1, y_k, mus, phi, Q, betas):
        """Performs a single recursive step of the Particle filter.
        
        Inputs
        ------
        x_m_k1 : ndarray of shape (2D,1), Previous posterior/measurement state estimate.
        P_m_k1 : ndarray of shape (2D, 2D), Previous posterior/measurement covariance matrix.
        y_k : float, Current observation.
        mu : float, Mean parameter.
        phi : ndarray of shape (2D, 2D), Transformation matrix (Latent equation).
        Q : ndarray of shape (2D, 2D), Process noise matrix 
        betas : float, Scaling parameter.
        N_particles : int, Number of particles to be sampled in the filter
        
        
        Returns
        -------
        tuple
            xp_k : ndarray of shape (2,), Predicted state.
            xm_k : ndarray of shape (2,), Updated state estimate.
            Pp_k : ndarray of shape (2, 2), Predicted covariance.
            Pm_k : ndarray of shape (2, 2), Updated covariance estimate.
        """
        
        if self.seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(self.seed)
    
        #Prediction Step
        xp_k = np.dot(phi, x_m_k1)
        Pp_k = np.linalg.multi_dot([phi, P_m_k1, phi.T]) + Q

        # Sample particles from predicted distribution
        particles = rng.multivariate_normal(mean = xp_k.flatten(), cov = Pp_k, size = self.N_particles).T
        inst = []
        for mu_c, beta_c in zip(mus, betas):
            inst.append(np.exp(mu_c + np.dot(beta_c, particles)))
        inst = np.vstack(inst)

        #Get weights and Normalize
        log_weights = (y_k * np.log(inst) - inst).sum(axis=0).reshape(1,-1)
        logw_max = log_weights.max()
        weights = np.exp(log_weights - logw_max)
        # log_marg_like = logw_max + np.log(np.sum(weights))
        log_marg_like = logw_max + np.log(np.mean(weights))
        weights /= weights.sum()
        
        # Measurement Update (empirical)
        xm_k = np.average(particles, axis=1, weights=weights[0]).reshape(-1,1)
        diffs = particles - xm_k
        Pm_k = (diffs*weights) @ diffs.T
    
        return xp_k, xm_k, Pp_k, Pm_k, log_marg_like

    def _KFilter(self, x0, P0, y, mus, phi, Q, betas, nD = None, K=None):
        """Runs the Kalman or Particle filter for the forward pass of the E-step.
    
        Inputs
        ------
        x0 : ndarray of shape (2D,1), Initial state estimate.
        P0 : ndarray of shape (2D,2D), Initial state covariance.
        y : ndarray of shape (1, K), Observed data.
        mus : array-like, Mean parameter.
        phi : ndarray of shape (2D, 2D), Transformation matrix (Latent equation).
        Q : ndarray of shape (2D, 2D), Process noise matrix 
        betas : float, Scaling parameter.
        nD : int, number of oscillations
        K : int, optional, Number of time steps, inferred from y if not provided.
        
        Returns
        -------
        tuple
            x_p : ndarray of shape (2, K), Predicted states.
            x_m : ndarray of shape (2, K), Filtered (posterior) states.
            P_p : ndarray of shape (2, 2, K), Predicted covariance matrices.
            P_m : ndarray of shape (2, 2, K), Filtered (posterior) covariance matrices.
        """
        
        if K is None:
            K = self.K
        if nD is None:
            nD = phi.shape[0]
            # nD = self.n_oscillations
            
        x_p = np.zeros((nD,K))
        x_m = np.zeros((nD,K))
        
        P_p = np.zeros((nD,nD,K))
        P_m = np.zeros((nD,nD,K))
    
        P_m[:, :, 0] = P0.copy()
        x_m[:, 0:1] = x0.copy()

        logL = 0
    
        #Forward Pass : Filter
        for k in range(1, K):
            if self.use_PF:
                x_p[:, k:k+1], x_m[:, k:k+1], P_p[..., k], P_m[..., k], logL_k = self._PF_step(x_m[:, k-1:k], P_m[...,k-1], y[:,k:k+1], mus, phi, Q, betas)
            else:
                x_p[:, k:k+1], x_m[:, k:k+1], P_p[..., k], P_m[..., k], logL_k = self._KF_step(x_m[:, k-1:k], P_m[...,k-1], y[:,k:k+1], mus, phi, Q, betas)
            logL += logL_k

        return x_p, x_m, P_p, P_m, logL
    
    def _backward_pass(self, x_p, x_m, P_p, P_m, phi, K = None):
        """Performs the backward smoothing pass for state estimation.
        
        Inputs
        ------
        phi : ndarray of shape (2D, 2D), State transition matrix.
        x_p : ndarray of shape (2D, K), Predicted states.
        x_m : ndarray of shape (2D, K), Filtered (posterior) states.
        P_p : ndarray of shape (2D, 2D, K), Predicted covariance matrices.
        P_m : ndarray of shape (2D, 2D, K), Filtered (posterior) covariance matrices.
        K : int, Number of time steps.
        
        Returns
        -------
        tuple
            x_b : ndarray of shape (2D, K), Smoothed states.
            P_b : ndarray of shape (2D, 2D, K), Smoothed covariance matrices.
            back_gains : ndarray of shape (2D, 2D, K), Smoothing gain matrices.
        """
        if K is None:
            K = self.K
        
        back_gains = np.zeros_like(P_m)
        x_b = np.zeros_like(x_m)
        P_b = np.zeros_like(P_m)
        x_b[:, -1] = x_m[:, -1].copy()
        P_b[:, :, -1] = P_m[:, :, -1].copy()
        for k in range(K-1, 0, -1):
            A_k1 = np.linalg.multi_dot([P_m[:, :, k-1], phi.T, np.linalg.pinv(P_p[:, :, k])])
            back_gains[:, :, k-1] = A_k1
    
            x_b[:, k-1:k] = x_m[:, k-1:k] + np.dot(A_k1, x_b[:, k:k+1] - x_p[:, k:k+1])
            P_b[:, :, k-1] = P_m[:, :, k-1] + np.linalg.multi_dot([A_k1, P_b[:, :, k] - P_p[:, :, k], A_k1.T])
            
        return x_b, P_b, back_gains

    def _get_beta_param(self, beta):
        return [beta[0, i] for i in range(beta.size) if np.mod(i,2)==0]
    
    def _make_beta(self, params):
        beta = np.repeat(params, 2)
        beta[[i for i in range(beta.size) if np.mod(i,2)!=0]] = 0
        return beta.reshape(1, -1)
        
    def _beta_objective(self, beta, x_b, P_b, y, mu, K = None):

        ### FIX : for multiple mus - REMEMEBER TO REMOVE THE BELOW SNIPPET
        # if np.asarray(mus).size>1:
        #     mu = mus[0]
        # else:
        #     mu = mus
        # #### REMOVE ^ WHEN FIXED

        ## FIX ?? seriously whats going on here

        
        if K is None:
            K = x_b.shape[-1]
        # beta = self._make_beta(params)
        beta = beta.reshape(1, -1)
        sum1 = mu + np.dot(beta, x_b)
        term1 = (y * sum1).sum()
        sum2 = mu + np.dot(beta, x_b) + np.einsum('ij,jkl,ki->il', beta, P_b, beta.T)/2
        np.testing.assert_array_less(sum2, 709)
        term2 = np.exp(sum2).sum()
    
        G = term2 - term1 #negative G because we're using minimize function.
    
        return G
    
    def _beta_jacobian(self, beta, x_b, P_b, y, mu, K = None):
        ### FIX : for multiple mus - REMEMEBER TO REMOVE THE BELOW SNIPPET
        # if np.asarray(mus).size>1:
        #     mu = mus[0]
        # else:
        #     mu = mus
        # #### REMOVE ^ WHEN FIXED

        
        if K is None:
            K = x_b.shape[-1]
        # beta = self._make_beta(params)

        beta = beta.reshape(1, -1)
        term1 = (y*x_b).sum(axis=1)
    
        exp2 = np.dot(beta, x_b) + np.einsum('ij,jkl,ki->il', beta, P_b, beta.T)/2
        exp2 = np.exp(exp2)
        sum2 = x_b + np.einsum('jkl,ki->jl', P_b, beta.T)
        
        term2 = np.exp(mu)*(sum2*exp2).sum(axis=1)
        
        J = term2 - term1 #negative J because we're using minimize function.
        # J_reduced = self._get_beta_param(np.asarray(J).reshape(1,-1))
        return J
    
    def _update_beta(self, betas, x_b, P_b, y, mus, get_mu_sum = False, K = None):
        if K is None:
            K = self.K

        self.update_beta_flag = [None]*self.n_neurons
        bounds = [(1e-6, None)]*P_b.shape[0]
        beta_out = []
        for c, mu_c, beta_c in zip(np.arange(self.n_neurons), mus, betas):
            if self.assume_phase_lock:
                res = minimize(fun = self._beta_objective, x0 = beta_c.squeeze(), jac = self._beta_jacobian, args = (x_b, P_b, y[c:c+1], mu_c, K),
                               method='L-BFGS-B', bounds = bounds)
            else:
                res = minimize(fun = self._beta_objective, 
                           x0 = beta_c.squeeze(), jac = self._beta_jacobian, args = (x_b, P_b, y[c:c+1], mu_c, K), method='Newton-CG')
    
            self.update_beta_flag[c] = res.success
            beta_r = res.x
            # beta_r = np.where(beta_r<1, 1 + 1e-3, beta_r)
            beta_out.append(beta_r.reshape(1, -1))
        
        if get_mu_sum:
            mu_sum = []
            for beta_c in beta_out:
                mu_sum.append(np.dot(beta_c, x_b) + np.einsum('ij,jkl,ki->il', beta_c, P_b, beta_c.T)/2)
            mu_sum = np.array(mu_sum).squeeze()
            return beta_out, mu_sum
        
        return beta_out
    
    def _E_step(self, x0, y, mus, alpha, freq, sigma, betas, K=None, forward_flag = False):
        """Expectation step of the EM algorithm. Runs the Kalman Filter and Backward pass. Computes intermediate statistics A, B, C, and expected spike count that are used in the M-step for maximizing the parameter estimates.
    
        Inputs
        ------
        x0 : ndarray of shape (2D,1), Initial state estimate.
        y : ndarray of shape (K,), Observed data.
        mus : array-like, Mean parameter.
        sigma : float, Process noise variance.
        alpha : float, State transition scaling factor.
        freq : float, Frequency parameter.
        betas : float, Scaling parameter.
        K : int, optional, Number of time steps, inferred from y if not provided.
        forward_flag : boolean, optional, returns only predictions if set to True
        
        Returns
        -------
        tuple
            x_p : ndarray of shape (2D, K), Predicted states.
            x_m : ndarray of shape (2D, K), Filtered (posterior) states.
            x_b : ndarray of shape (2D, K), Smoothed states.
            P_p : ndarray of shape (2D, 2D, K), Predicted covariance matrices.
            P_m : ndarray of shape (2D, 2D, K), Filtered (posterior) covariance matrices.
            P_b : ndarray of shape (2D, 2D, K), Smoothed covariance matrices.
            back_gains : ndarray of shape (2D, 2D, K), Smoothing gain matrices.
            A : ndarray, shape (2D, 2D), Summation of smoothed covariances and state estimates.
            B : ndarray, shape (2D, 2D), Cross-covariance matrix for consecutive time steps.
            C : ndarray, shape (2D, 2D), Summation of smoothed covariance estimates for all time steps.
            mu_sum : float, Value of the intensity function used to maximize mu parameter.
        """
        if K is None:
            K = self.K
        
        nD = 2*len(freq)
        
        phi = np.repeat(alpha, 2)*self._rotation_matrix(freq)
        Q = block_diag(*[sig*np.eye(2) for sig in sigma])
        P0 = (np.repeat(sigma, 2) / (1 - np.repeat(alpha, 2)**2)) * np.eye(nD)

        assert phi.shape == Q.shape == P0.shape == (nD, nD), f"Expected shape of phi, Q, P0 : ({nD}, {nD}), but got {phi.shape, Q.shape, P0.shape}"
        assert x0.shape == (nD, 1), f"Expected shape of x0 : ({nD}, 1), but got {x0.shape}"
        assert len(betas) == self.n_neurons, f"Length of betas {len(betas)} != No. of neurons {self.n_neurons}"
        
        # print('x0:', x0.shape, 'mus:', np.asarray(mus).shape, 'betas:', np.asarray(betas).shape, 'freq:', np.asarray(freq).shape)
        #Forward Pass
        x_p, x_m, P_p, P_m, logL = self._KFilter(x0, P0, y, mus, phi, Q, betas, nD, K)
        
        if forward_flag:
            return x_p, P_p
        #Backward Pass
        x_b, P_b, back_gains = self._backward_pass(x_p, x_m, P_p, P_m, phi, K) 

        # Second-order moment calculations
        A = P_b[:, :, :-1].sum(axis=-1) + np.dot(x_b[:, :-1], x_b[:, :-1].T)
        
        B = np.matmul(np.transpose(P_b[:, :, 1:], axes=(2,0,1)), np.transpose(back_gains[:, :, :-1], axes=[2,1,0])).sum(axis=0)
        B += np.dot(x_b[:, 1:], x_b[:, :-1].T)
        
        C = P_b[:, :, 1:].sum(axis=-1) + np.dot(x_b[:, 1:], x_b[:, 1:].T)
        
        #Mu update term -> estimate of expinential term from x_b and P_b
        mu_sum = []
        for beta_c in betas:
            beta_c = beta_c.reshape(1, -1)
            mu_sum.append(np.dot(beta_c, x_b) + np.einsum('ij,jkl,ki->il', beta_c, P_b, beta_c.T)/2)
        mu_sum = np.array(mu_sum).squeeze()
        mu_sum = mu_sum.reshape(y.shape)

        return x_p, x_m, x_b, P_p, P_m, P_b, back_gains, A, B, C, mu_sum, logL
    
    def _M_step(self, dN, A, B, C, mu_sum, K=None):
        
        """Maximization step of the EM algorithm. Updates model parameters.

        Inputs
        ------
        A : ndarray, shape (2D, 2D), Summation of smoothed covariances and state estimates.
        B : ndarray, shape (2D, 2D), Cross-covariance matrix for consecutive time steps.
        C : ndarray, shape (2D, 2D), Summation of smoothed covariance estimates for all time steps.
        mu_sum : float, Expected value of the intensity function.
        dN : ndarray, shape (1,K), Observed spike counts or event occurrences.
        K : int, (optional) Number of time steps.

        Returns
        -------
        mu_r : float, Updated intensity function parameter.
        alpha_r : float, Updated state transition scaling factor.
        omega_r : float, Updated rotation angle parameter.
        sigma_r : float, Updated process noise variance.
        """
        if K is None:
            K = self.K
        
        freq_r = []
        alpha_r = []
        sigma_r = []

        D = A.shape[0]//2
        am = self.am
        bm = self.bm
        
        for Ad, Bd, Cd, d in zip(self._decompose_block_diag(A), self._decompose_block_diag(B), self._decompose_block_diag(C), range(D)):
            assert Ad.shape == Bd.shape == Cd.shape == (2, 2), f"Expected shape of Ad,Bd,Cd - (2, 2), but got {Ad.shape, Bd.shape, Cd.shape}"
            omega_r = np.arctan2(self._rt_matrix(Bd), np.trace(Bd))
            freq_r.append(self.Fs*omega_r/(2*np.pi))
            
            a_r = (np.trace(Bd)*np.cos(omega_r) + self._rt_matrix(Bd)*np.sin(omega_r)) / np.trace(Ad)
            a_r = min(1 - 1e-6, a_r)
            alpha_r.append(a_r)
            
            Tr = np.trace(Cd) - np.power(a_r, 2)*np.trace(Ad)
            if self.use_reg:
                sigma_r.append((-K + np.sqrt(K**2 + 2*self.gamma*Tr))/(2*self.gamma))
            else:
                sigma_r.append(Tr/(2*K))
            
        if self.use_reg:
            mu_r = np.log(np.sum(dN, axis=1) + am - 1) - mu_sum.max(axis=1)
            mu_r -= np.log(np.exp(mu_sum - mu_sum.max(axis=1, keepdims=True)).sum(axis=1) + np.exp(np.log(bm) - mu_sum.max(axis=1))) 
        else:
            mu_r = np.log(np.sum(dN, axis=1)) - mu_sum.max(axis=1) - np.log(np.exp(mu_sum - mu_sum.max(axis=1, keepdims=True)).sum(axis=1))
        return list(mu_r), alpha_r, freq_r, sigma_r
            
    def _verify_shapes(self, x0, y, mu0, alpha0, freq0, sigma0, beta0):
        '''
        
        '''
        if np.asarray(x0).ndim == 1:
            x0 = x0.reshape(-1, 1)
        # if np.asarray(beta0).ndim == 1:
        #     beta0 = np.asarray(beta0).reshape(1, -1)
        if np.asarray(y).ndim == 1:
            y = y.reshape(1, -1)
        # assert beta0.shape == x0.T.shape, f"Expected shape beta0 and x0.T to be same shape, but got beta0 = {beta0.shape}, x0 = {x0.T.shape}"
        
        n_osc = x0.shape[0]//2

        freq0 = list(freq0) if isinstance(freq0, (np.ndarray)) else freq0
        alpha0 = list(alpha0) if isinstance(alpha0, (np.ndarray)) else alpha0
        sigma0 = list(sigma0) if isinstance(sigma0, (np.ndarray)) else sigma0
        mu0 = list(mu0) if isinstance(mu0, (np.ndarray)) else mu0

        freq0 = [freq0] if isinstance(freq0, (int, float)) else freq0
        alpha0 = [alpha0] if isinstance(alpha0, (np.ndarray)) else alpha0
        sigma0 = [sigma0] if isinstance(sigma0, (np.ndarray)) else sigma0
        mu0 = [mu0] if isinstance(mu0, (np.ndarray)) else mu0

        assert len(alpha0) == len(freq0) == len(sigma0) == n_osc, f"Expected length of alpha0, freq0 and sigma0 to match x0.shape (2*D, 1)={x0.shape}, but got len(alpha0) = {len(alpha0)}, len(freq0) = {len(freq0)}, len(sigma0) = {len(sigma0)}"
        return x0, y, mu0, alpha0, freq0, sigma0, beta0


    def _run_em_algorithm(self, x0, y, mu0, alpha0, freq0, sigma0, beta0, max_iter = 100, log_file = None, track_params=True):

        # FIX beta assertion
        x0, y, mu0, alpha0, freq0, sigma0, beta0 = self._verify_shapes(x0, y, mu0, alpha0, freq0, sigma0, beta0)
        
        K = self.K
        
        # Early stopping setup
        logL_history = []
        best_logL = -np.inf
        best_params = None
        best_hidden_states = None
        latest_params = None
        latest_hidden_states = None
        
        if track_params:
            param_names = [f'{prm}{c}_' for prm in ['mu', 'beta'] for c in range(1, self.n_neurons+1)] + [f'{prm}{d}_' for prm in ['alpha', 'freq', 'sigma'] for d in range(1, self.max_oscillations+1)] + ['time_taken (s)']
            param_history = pd.DataFrame(columns=param_names)
            param_history.loc[0, :] = list(mu0) + list(beta0) + list(alpha0) + list(freq0) + list(sigma0) + [0] #$$

        
        for itr in range(1, max_iter+1):
            if self.verbose:
                print(itr, end=' ')
            start_time = time.time()

            try:
                # E_step
                x_p, x_m, x_b, P_p, P_m, P_b, BG, A, B, C, mu_sum, logL = self._E_step(x0, y, mu0, alpha0, freq0, sigma0, beta0, K, forward_flag=False)

                # M-step
                mu_r, alpha_r, freq_r, sigma_r = self._M_step(y, A, B, C, mu_sum, K)

                #Update beta
                if self.change_beta:
                    beta_r = self._update_beta(beta0, x_b, P_b, y, mu_r, K=K) #$$
                else:
                    beta_r = beta0
                
                
                latest_params = {'mus_': mu_r, 'alphas_': alpha_r, 'freqs_': freq_r, 'sigmas_': sigma_r, 'beta_': beta_r, 'logL': logL, 'itr': itr}
                latest_hidden_states = {'x_p': x_p, 'x_m': x_m, 'x_b': x_b, 'P_p': P_p, 'P_m': P_m, 'P_b': P_b, 'BG': BG, 'A': A, 'B': B, 'C': C}
                
                # Track the best iteration
                if logL > best_logL:
                    best_logL = logL
                    best_params = latest_params.copy()
                    best_hidden_states = latest_hidden_states.copy()
                
                #Save M-step Updated parameter values
                if track_params:
                    param_history.loc[itr, :] = mu_r + beta_r + alpha_r + freq_r + sigma_r + [time.time()-start_time] #$$
                    if log_file:
                        param_history.to_csv(log_file)
                    
                # Stop if the max logL in the last 'patience' steps hasn't improved over the global best
                logL_history.append(logL)
                if len(logL_history) > self.patience:
                    if max(logL_history[-self.patience:]) <= best_logL:
                        break

                #Update for next iter
                mu0, beta0 = mu_r, beta_r
                alpha0, freq0, sigma0 = alpha_r, freq_r, sigma_r
                
            except AssertionError as err:
                print(f"Iteration {itr} failed: {err}")
                break
        
        best_params = best_params
        best_hidden_states = best_hidden_states

        if track_params:
            return best_params, best_hidden_states, latest_params, latest_hidden_states, logL_history, param_history
        return best_params, best_hidden_states, latest_params, latest_hidden_states, logL_history
        
        

