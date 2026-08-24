"""The single probe implementation: StandardScaler -> PCA -> RidgeCV.

Previously copy-pasted into four scripts. The fit must stay byte-identical
across experiments or their numbers stop being comparable, which is exactly
what a probe study cannot afford.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-1, 6, 22)


def feature_variants(M, kind):
    """[N, T, D] pooled features -> flat design matrix.

    full         appearance + motion, the default
    diff         frame-to-frame differences only; the real-footage experiments
                 showed this channel stays in-distribution across sim-to-real
    diff-nomean  diff with the per-clip mean drift removed
    """
    if kind == "full":
        return M.reshape(len(M), -1)
    D = np.diff(M, axis=1)
    if kind == "diff":
        return D.reshape(len(M), -1)
    if kind == "diff-nomean":
        D = D - D.mean(axis=1, keepdims=True)
        return D.reshape(len(M), -1)
    raise ValueError(kind)


class GaussianCloud:
    """Scaler -> PCA -> Mahalanobis distance to the training distribution.

    Used both inside LinearProbe and on its own for out-of-distribution
    checks (e.g. how far a real DROID clip sits from the sim feature cloud).
    """

    def __init__(self, npca=96):
        self.npca = npca

    def fit(self, X):
        X = np.asarray(X, np.float64)
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.pca = None
        if Z.shape[1] > self.npca and Z.shape[0] > 8:
            self.pca = PCA(n_components=min(self.npca, Z.shape[0] - 1), random_state=0).fit(Z)
            Z = self.pca.transform(Z)
        self.mu = Z.mean(0)
        self.icov = np.linalg.inv(np.cov(Z.T) + 1e-6 * np.eye(Z.shape[1]))
        ref = np.sqrt(np.einsum("ij,jk,ik->i", Z - self.mu, self.icov, Z - self.mu))
        self.ref_p95 = float(np.percentile(ref, 95))
        return self

    def transform(self, X):
        Z = self.scaler.transform(np.asarray(X, np.float64))
        return self.pca.transform(Z) if self.pca is not None else Z

    def maha(self, X):
        Z = self.transform(X) - self.mu
        return np.sqrt(np.einsum("ij,jk,ik->i", Z, self.icov, Z))


class LinearProbe:
    """Ridge read-out on top of a GaussianCloud projection."""

    def __init__(self, npca=96, alphas=ALPHAS):
        self.cloud = GaussianCloud(npca)
        self.alphas = alphas

    def fit(self, X, y):
        self.cloud.fit(X)
        self.model = RidgeCV(alphas=self.alphas).fit(self.cloud.transform(X),
                                                     np.asarray(y, np.float64))
        return self

    def predict(self, X):
        return self.model.predict(self.cloud.transform(X))

    def maha(self, X):
        return self.cloud.maha(X)

    @property
    def ref_p95(self):
        return self.cloud.ref_p95
