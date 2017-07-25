import os

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.special import erfc, erfcinv
import scipy.stats
from scipy.stats import invgamma, expon
from multiprocessing import Pool

import signal

# Todo: option to use Raftery-Lewis burn-in (and thinning?)

"""
Based on Schmidt et al. 2009, "Bayesian non-negative matrix factorization"
fit() method uses their notation.

The rectified_normal_sample() method is a minor modification of the bound _randr() method of the Bd class in Nimfa
Some implementation details also follow Nimfa.
https://github.com/marinkaz/nimfa/blob/master/nimfa/methods/factorization/bd.py

Here is the Nimfa license:

New BSD License

Copyright (c) 2016 The Nimfa developers.
All rights reserved.


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the Nimfa Developers  nor the names of
     its contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""


def rectified_normal_sample(m, s, l):
    """
    Return random number from distribution with density
    (x)=K*exp(-(x-m)^2/s-l'x), x>=0.
    m and l are vectors and s is scalar
    """
    m = np.array(m)
    l = np.array(l)
    A = (l * s - m) / np.sqrt(2 * s)
    a = A > 26.
    x = np.zeros(m.shape)
    y = np.random.random_sample(m.shape)
    x[a] = - np.log(y[a]) / ((l[a] * s - m[a]) / s)
    a = np.array(1 - a, dtype=bool)
    R = erfc(abs(A[a]))
    x[a] = erfcinv(y[a] * R - (A[a] < 0) * (2 * y[a] + R - 2)) * \
        np.sqrt(2 * s) + m[a] - l[a] * s
    x[np.isnan(x)] = 0
    x[x < 0] = 0
    x[np.isinf(x)] = 0
    return x.real


def rectified_normal_pdf(x, m, s, l):
    #different formulation?
    # return 0 if x < 0 else norm.pdf(x, loc=m, scale=s) * 2 / erfc(-m / np.sqrt(2 * s))
    # as in Schmidt and Mohamed 2009 "PROBABILISTIC NON-NEGATIVE TENSOR FACTORIZATION USING MARKOV CHAIN MONTE CARLO"
    # same author (and year) as Bayesian NMF paper
    #numerator = np.sqrt(2 / (np.pi * s)) * np.exp(-1 * ((x - m)** 2) / (2 * s))
    #denominator = erfc(-1 * m / np.sqrt(2 * s))
    #return numerator / denominator
    #normalizing_constant = 1 / np.sqrt(2 * np.pi * s)
    #exp = np.exp(-1 * (((x - (m - s * l))**2)/ (2 * s)))
    #other = (-0.5 * s * l**2) +
    #return exp * normalizing_constant
    #pass
    # accepting CrossValidated post saying it is a truncated normal
    # pdf = normal pdf over
    pdf = np.array(x)
    pdf[x < 0] = 0
    newm = m - s * l
    std = np.sqrt(s)
    normpdf = scipy.stats.norm(loc=newm, scale=std).pdf(x[x >= 0])
    denom = std * (1 - scipy.stats.norm(loc=newm, scale=std).cdf(0.0)).clip(min=np.finfo(float).eps)
    pdf[x >= 0] = np.divide(normpdf, denom)
    return pdf


class BayesianNMF:

    def __init__(self, n_components=None,
                  mode='map',
                  tol=1E-6,
                  max_iter=10000,
                  burnin_fraction=0.5,
                  mean_only=True,
                  random_state=None,
                  verbose=0):
        """
        Bayesian non-negative matrix factorization
        Based on Schmidt et al. 2009, "Bayesian non-negative matrix factorization"
        :param n_components: number of latent variables
        :param mode: 'map' (default) or 'gibbs'
        :param tol: convergence criterion in iterated conditional modes (ICM) algorithm ('map' mode)
        :param max_iter: number of Gibbs samples ('gibbs' mode) or maximum iterations of ICM ('map' mode)
        :param burnin_fraction: fraction of initial Gibbs samples to discard ('gibbs' mode)
        :param mean_only: store only mean of parameters, rather than samples ('gibbs' mode). To save memory.
        :param random_state: not yet supported
        :param verbose: whether to print output
        """
        if mode not in ['map', 'gibbs']:
            raise ValueError("{} is not a supported mode. Try 'map' or 'gibbs'".format(mode))

        self.n_components = n_components
        self.mode=mode
        self.max_iter = max_iter
        self.tol = tol
        self.burnin_fraction = burnin_fraction
        self.mean_only = mean_only
        self.random_state = random_state
        self.verbose = verbose
        self.bases_prior_ = None
        self.components_prior_ = None
        self.variance_prior_shape_ = None
        self.variance_prior_scale_ = None
        self.burnin_bases_samples_ = None
        self.burnin_components_samples_ = None
        self.burnin_variance_samples_ = None
        self.bases_samples_ = None
        self.components_samples_ = None
        self.variance_samples_ = None
        self.bases_ = None
        self.components_ = None
        self.variance_ = None
        self.reconstruction_err_ = None

    def fit(self, X):
        N = self.n_components
        M = self.max_iter
        burnin_index = int(M * self.burnin_fraction)
        n_after_burnin = M - burnin_index
        I, J = X.shape
        alpha_prior_scalar = 1
        # in Schmidt et al. they have all alpha_i,n = 1
        # and choose the beta prior to match the amplitude of the data
        beta_prior_scalar = 1 / N * X.mean()  # since the mean of the exponential is 1 / its param
        # alpha_prior_scalar, beta_prior_scalar = 0, 0 # flat prior used in other part of Schmidt et al.
        k = 0  # uninformative prior used in Schmidt et al.
        theta = 0  # uninformative prior used in Schmidt et al.
        self.variance_prior_shape_ = k
        self.variance_prior_scale_ = theta
        alpha = np.ones((I, N)) * alpha_prior_scalar
        self.bases_prior_ = alpha
        beta = np.ones((N, J)) * beta_prior_scalar
        self.components_prior_ = beta
        A = np.random.exponential(scale=alpha_prior_scalar, size=(I, N))
        B = np.random.exponential(scale=beta_prior_scalar, size=(N, J))
        A_mean = np.zeros(A.shape)
        B_mean = np.zeros(B.shape)
        mu2_mean = 0
        As = []
        Bs = []
        mu2s = []
        chi = 0.5 * np.square(X).sum()
        mu2 = np.var(X - np.dot(A, B))
        m = 0
        diff = self.tol + 1
        obj = mean_squared_error(X, np.dot(A, B))
        while m < M and diff > self.tol:
            C = np.dot(B, B.T)
            D = np.dot(X, B.T)
            for n in range(N):
                notn = list(range(n)) + list(range(n + 1, N))
                an = (D[:, n] - np.dot(A[:, notn], C[notn, n]) - alpha[:, n] * mu2) / (C[n, n] + np.finfo(float).eps)
                if self.mode == 'gibbs':
                    rnorm_variance = mu2 / (C[n, n] + np.finfo(C.dtype).eps)
                    A[:, n] = rectified_normal_sample(an, rnorm_variance, alpha[:, n])
                else:
                    A[:, n] = an.clip(min=0)
            ac_2d_diff = np.dot(A, C) - (2 * D)
            xi = 0.5 * np.multiply(A, ac_2d_diff).sum()
            if self.mode == 'gibbs':
                # As in Schmidt et al. 2009:
                mu2 = invgamma.rvs(a=(I*J / 2) + k + 1, scale=chi + theta + xi)
                # As in Nimfa, inverted (for speed? could compare):
                #gamma_scale = 1. / max(np.finfo(float).eps, theta + chi + xi)
                #mu2 = 1 / np.random.gamma(shape=(I * J / 2) + 1 + k, scale=gamma_scale)
            else:
                mu2 = (theta + chi + xi) / ((I * J / 2) + k + 1)
            #print(mu2)
            E = np.dot(A.T, A)
            F = np.dot(A.T, X)
            for n in range(N):
                notn = list(range(n)) + list(range(n + 1, N))
                bn = (F[n] - np.dot(E[n, notn], B[notn]) - beta[n] * mu2) / (E[n, n] + np.finfo(float).eps)
                if self.mode == 'gibbs':
                    rnorm_variance = mu2 / (E[n, n] + np.finfo(float).eps)
                    B[n] = rectified_normal_sample(bn, rnorm_variance, beta[n])
                else:
                    B[n] = bn.clip(min=0)
            if self.mode == 'gibbs':
                if self.mean_only:
                    if m >= burnin_index:
                        A_mean += A / n_after_burnin
                        B_mean += B / n_after_burnin
                        mu2_mean += mu2 / n_after_burnin
                else:
                    As.append(A.copy())
                    Bs.append(B.copy())
                    mu2s.append(mu2)
            else:
                new_obj = mean_squared_error(X, np.dot(A, B))
                diff = obj - new_obj
                obj = new_obj
                if self.verbose:
                    print("MSE: ", obj)
            m += 1
        if self.mode == 'gibbs':
            As, Bs, mu2s = [np.array(arr) for arr in [As, Bs, mu2s]]
            self.A_mean = A_mean
            self.B_mean = B_mean
            self.mu2_mean = mu2_mean
            if self.mean_only:
                A = A_mean
                B = B_mean
                mu2 = mu2_mean
            else:
                self.burnin_bases_samples_ = As[:burnin_index]
                self.burnin_components_samples_ = Bs[:burnin_index]
                self.burnin_variance_samples_ = mu2s[:burnin_index]
                self.bases_samples_ = As[burnin_index:]
                self.components_samples_ = Bs[burnin_index:]
                self.variance_samples_ = mu2s[burnin_index:]
                A = self.bases_samples_.mean(axis=0)
                B = self.components_samples_.mean(axis=0)
                mu2 = self.variance_samples_.mean(axis=0)
        obj = mean_squared_error(X, np.dot(A, B))
        self.bases_ = A
        self.components_ = B
        self.variance_ = mu2
        self.reconstruction_err_ = obj
        return self


def chibs_approx_marginal_likelihood(X, nmf, max_iter=1000, burnin_fraction=0.5):
    M = max_iter
    A, B, mu2, alpha, beta, k, theta = [nmf.bases_,
                                        nmf.components_,
                                        nmf.variance_,
                                        nmf.bases_prior_,
                                        nmf.components_prior_,
                                        nmf.variance_prior_shape_,
                                        nmf.variance_prior_scale_]
    N = nmf.n_components
    X_model = np.dot(A, B)
    log_p_x_g_theta = scipy.stats.norm.logpdf(X, loc=X_model, scale=mu2).sum()
    log_p_a = expon.logpdf(A, scale=1 / alpha).sum()
    log_p_b = expon.logpdf(B, scale=1 / beta).sum()
    minval = np.finfo(float).eps
    log_p_mu2 = scipy.stats.invgamma.logpdf(mu2, a=max(minval, k), scale=max(minval, theta))
    log_numerator = sum([log_p_x_g_theta, log_p_a, log_p_b, log_p_mu2])

    # Now to do Gibbs sampling for each parameter block (columns of A, rows of B, so 2N runs total)
    # What about the variance parameter? Should it be Gibbs sampled? Maybe sampled along with each block?
    """
    p(t | X) =
    p(t1 | X) * p(t2 | t1, X) ... p(tK| tK-1, ... t1, X)
    p(tk | tk-1...t1, x) =
    mean over gibbs samples of
    p(tk | t1, t2, ... tk-1, tk+1(sampled), ...tK(sampled), X)
    """
    chi = 0.5 * np.square(X).sum()
    gibbs_param_inputs = [(param_block_idx, X, A, B, alpha, beta, mu2, chi, k, theta, M, burnin_fraction) for param_block_idx in range(2 * N)]
    nprocs = os.cpu_count()
    pool = Pool(processes=nprocs)

    def sigint_handler(signum, frame):
        pool.close()
        pool.join()
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        param_prob_means = pool.map(gibbs_sample_param, gibbs_param_inputs)
        pool.close()
        pool.join()
    except Exception as e:
        pool.close()
        pool.join()
        raise e
    log_denom = np.log(param_prob_means).sum()
    return log_numerator - log_denom


def gibbs_sample_param(inputs):
    # apparently in Python3.x multiprocessing can pickle instance methods, so I could include this in the NMF object?
    # so I don't have to pass around so many parameters
    pbidx, X, A, B, alpha, beta, mu2, chi, k, theta, M, burnin_fraction = inputs
    burnin_index = int(M * burnin_fraction)
    n_after_burnin = M - burnin_index
    I, N = A.shape
    N, J = B.shape
    nB = pbidx - N
    An = A.copy()
    Bn = B.copy()
    prob_mean = 0
    for m in range(M):
        C = np.dot(Bn, Bn.T)
        D = np.dot(X, Bn.T)
        """
        If I sample single columns in turn, conditioned on all other columns, and update after each column
        it is not quite right
        It should sample from p(tk+1...tK| tk-1...t1).
        So I use a not-updated matrix until I've sampled all those columns, then update
        What about tk?
        I excluded its index so it doesn't contribute to an and bn.
        I believe the current code is correctly not conditioning on it
        """
        An_temp = An.copy()
        for n in range(min(pbidx + 1, N), N):
            notn = list(range(n - 1)) + list(range(n + 1, N))
            an = (D[:, n] - np.dot(An[:, notn], C[notn, n]) - alpha[:, n] * mu2) / (C[n, n] + np.finfo(float).eps)
            rnorm_variance_a = mu2 / (C[n, n] + np.finfo(float).eps)
            An_temp[:, n] = rectified_normal_sample(an, rnorm_variance_a, alpha[:, n])
        An = An_temp
        ac_2d_diff = np.dot(An, C) - (2 * D)
        xi = 0.5 * np.multiply(An, ac_2d_diff).sum()
        # Should I be sampling mu2 or just using the fitted value?
        mu2 = invgamma.rvs(a=(I * J / 2) + k + 1, scale=chi + theta + xi)
        E = np.dot(An.T, An)
        F = np.dot(An.T, X)
        Bn_temp = Bn.copy()
        for n in range(nB + 1, N):
            notn = list(range(n - 1)) + list(range(n + 1, N))
            bn = (F[n] - np.dot(E[n, notn], Bn[notn]) - beta[n] * mu2) / (E[n, n] + np.finfo(float).eps)
            rnorm_variance_b = mu2 / (E[n, n] + np.finfo(float).eps)
            Bn_temp[n] = rectified_normal_sample(bn, rnorm_variance_b, beta[n])
        Bn = Bn.copy()
        if pbidx < N:
            C = np.dot(Bn, Bn.T)
            D = np.dot(X, Bn.T)
            x = A[:, pbidx]
            notn = list(range(pbidx)) + list(range(pbidx + 1, N))
            param_mean = (D[:, pbidx] -
                          np.dot(An[:, notn], C[notn, pbidx]) -
                          alpha[:, pbidx] * mu2) / (C[pbidx, pbidx] + np.finfo(C.dtype).eps)
            param_variance = mu2 / (C[pbidx, pbidx] + np.finfo(float).eps)
            param_scale = alpha[:, pbidx]
        else:
            E = np.dot(An.T, An)
            F = np.dot(An.T, X)
            x = B[nB]
            notn = list(range(nB)) + list(range(nB + 1, N))
            param_mean = (F[nB] - np.dot(E[nB, notn], Bn[notn]) - beta[nB] * mu2) / (
                E[nB, nB] + np.finfo(float).eps)
            param_variance = mu2 / (E[nB, nB] + np.finfo(float).eps)
            param_scale = beta[nB]
        prob_sample = rectified_normal_pdf(x, param_mean, param_variance, param_scale)
        if m >= burnin_index:
            prob_mean += prob_sample / n_after_burnin  # I hope this doesn't ever give me underflow issues
    return prob_mean
