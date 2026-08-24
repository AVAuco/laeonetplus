"""
NumPy implementation of a constant-velocity Kalman filter for bounding-box tracking,
parameterized as (center_x, center_y, aspect_ratio, height) + their velocities, following
the standard DeepSORT formulation.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

# Table of the 0.95 quantile of the chi-square distribution with N degrees of freedom
# (used for the Mahalanobis gating distance).
CHI2_INV_95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class KalmanFilter:
    """8-dimensional Kalman filter: state = [cx, cy, a, h, vx, vy, va, vh]."""

    def __init__(self) -> None:
        ndim, dt = 4, 1.0

        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion/observation uncertainty weights (relative to object size), as in the
        # original DeepSORT paper/implementation.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T
        return mean, covariance + innovation_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), (covariance @ self._update_mat.T).T, check_finite=False
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance

    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray) -> np.ndarray:
        mean, covariance = self.project(mean, covariance)
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        return np.sum(z * z, axis=0)
