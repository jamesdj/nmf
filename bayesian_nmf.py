import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.special import erfc, erfcinv

"""
Based on Schmidt et al. 2009, "Bayesian non-negative matrix factorization"
fit() method uses their notation.

The rectified_normal() method is a minor modification of the bound _randr() method of the Bd class in Nimfa
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

def rectified_normal(m, s, l):
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


class BayesianNMF:

    def __init__(self, n_components=None,
                  mode='map',
                  tol=1E-6,
                  max_iter=10000,
                  burnin_fraction=0.5,
                  random_state=None,
                  verbose=0):
        if mode not in ['map', 'gibbs']:
            raise ValueError("{} is not a supported mode. Try 'map' or 'gibbs'".format(mode))

        self.n_components = n_components
        self.mode=mode
        self.max_iter = max_iter
        self.tol = tol
        self.burnin_fraction = burnin_fraction
        self.random_state = random_state
        self.verbose = verbose
        self.bases_samples_ = None
        self.components_samples_ = None
        self.bases_ = None
        self.components_ = None
        self.reconstruction_err_ = None

    def fit(self, X):
        N = self.n_components
        M = self.max_iter
        I, J = X.shape
        alpha_prior = 1
        # in Schmidt et al. they have all alpha_i,n = 1
        # and choose the beta prior to match the amplitude of the data
        beta_prior = 1 / N * X.mean()  # since the mean of the exponential is 1 / its param
        # alpha_prior, beta_prior = 0, 0 # flat prior used in other part of Schmidt et al.
        k = 0  # uninformative prior used in Schmidt et al.
        theta = 0  # uninformative prior used in Schmidt et al.
        alpha = np.ones((I, N)) * alpha_prior
        beta = np.ones((N, J)) * beta_prior

        A = np.random.exponential(scale=alpha_prior, size=(I, N))
        B = np.random.exponential(scale=beta_prior, size=(N, J))
        As = []
        Bs = []
        chi = 0.5 * np.square(X).sum()
        mu2 = np.var(X - np.dot(A, B))
        m = 0
        diff = self.tol + 1
        obj = mean_squared_error(X, np.dot(A, B))
        while m < M and diff > self.tol:
            m += 1
            C = np.dot(B, B.T)
            D = np.dot(X, B.T)
            for n in range(N):
                notn = list(range(n)) + list(range(n + 1, N))
                an = (D[:, n] - np.dot(A[:, notn], C[notn, n]) - alpha[:, n] * mu2) / (C[n, n] + np.finfo(C.dtype).eps)
                if self.mode == 'gibbs':
                    rnorm_variance = mu2 / (C[n, n] + np.finfo(C.dtype).eps)
                    A[:, n] = rectified_normal(an, rnorm_variance, alpha[:, n])
                else:
                    A[:, n] = an.clip(min=0)
            ac_2d_diff = np.dot(A, C) - (2 * D)
            xi = 0.5 * np.multiply(A, ac_2d_diff).sum()
            if self.mode == 'gibbs':
                # mu2 = invgamma.rvs((I*J / 2) + k + 1, chi + theta + xi) # not sure why incorrect
                gamma_scale = 1. / max(np.finfo(float).eps, theta + chi + xi)
                mu2 = 1 / np.random.gamma(shape=(I * J / 2) + 1 + k, scale=gamma_scale)
            else:
                mu2 = (theta + chi + xi) / ((I * J / 2) + k + 1)
            E = np.dot(A.T, A)
            F = np.dot(A.T, X)
            for n in range(N):
                notn = list(range(n)) + list(range(n + 1, N))
                bn = (F[n] - np.dot(E[n, notn], B[notn]) - beta[n] * mu2) / (E[n, n] + np.finfo(E.dtype).eps)
                if self.mode == 'gibbs':
                    rnorm_variance = mu2 / (E[n, n] + np.finfo(E.dtype).eps)
                    B[n] = rectified_normal(bn, rnorm_variance, beta[n])
                else:
                    B[n] = bn.clip(min=0)
            if self.mode == 'gibbs':
                As.append(A)
                Bs.append(B)
            else:
                new_obj = mean_squared_error(X, np.dot(A, B))
                diff = obj - new_obj
                obj = new_obj
                if self.verbose:
                    print("MSE: ", obj)
        if self.mode == 'gibbs':
            self.bases_samples_ = As
            self.components_samples_ = Bs
            A = np.array(As[int(M * self.burnin_fraction):]).mean(axis=0)
            B = np.array(Bs[int(M * self.burnin_fraction):]).mean(axis=0)
            obj = mean_squared_error(X, np.dot(A, B))
            # note: could instead do the mean MSE across the gibbs samples.
        self.bases_ = A
        self.components_ = B
        self.reconstruction_err_ = obj
        return self
