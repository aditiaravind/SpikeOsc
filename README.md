# SpikeOsc
Runs an Expectation-Maximization (EM) Algorithm to estimate the latent oscillatory dynamics of a spike train modeled as a Poisson point process.


## Observation Model
We are using a point process model as the observation model with the following Conditional Intensity Function to represent the likelihood of obsrving a spike at time $k\Delta$.
```math
\begin{aligned}
    \lambda_c (k\Delta) &= \exp(\mu_c + x_k) \\
\end{aligned}
```
Here $x_k$ is the real part of the latent vector.

## Latent Model
We define the latent model as an autoregressive model AR(1) where the variable is transformed by a rotation matrix $R(\omega)$.
 ```math
\begin{align*}
\mathbf{x_t} = \alpha R(\omega)\mathbf{x_{t-1}} + \mathbf{u_t} \qquad \forall t = 1...T
\end{align*}
```
Where, $x_t$ is a 2-d vector representing the hidden states that affect a neuron's activity. 

## Expectation Step
- Estimates the oscillatory signal with a current set of model parameters
- Implemented using an Extended Kalman Filter (EKF) or Gaussian Particle Filter (GPF)

## Maximization Step
- Uses estimated oscillations to maximize the estimate of model parameters
- Includes options for regularization  - especially important for EKF
