"""
H4wkQuant - Kalman Filter for Half-Life Smoothing
Prevents sudden half-life jumps that generate false signals.
"""
from dataclasses import dataclass
from loguru import logger


@dataclass
class KalmanResult:
    raw_value: float
    filtered_value: float
    gain: float
    estimate_variance: float


class KalmanFilter:
    """
    1D Kalman filter for smoothing noisy scalar observations (e.g. half-life).

    State model:  x_t = x_{t-1} + w,  w ~ N(0, Q)
    Obs model:    z_t = x_t + v,       v ~ N(0, R)
    """

    def __init__(self, process_variance: float = 0.5, measurement_variance: float = 5.0):
        self.Q = process_variance     # process noise
        self.R = measurement_variance  # measurement noise
        self.x_hat: float | None = None  # state estimate
        self.P: float = 1.0              # estimate covariance

    def update(self, measurement: float) -> KalmanResult:
        if self.x_hat is None:
            self.x_hat = measurement
            self.P = self.R
            return KalmanResult(
                raw_value=measurement,
                filtered_value=measurement,
                gain=1.0,
                estimate_variance=self.P,
            )

        # Predict
        x_pred = self.x_hat
        P_pred = self.P + self.Q

        # Update
        K = P_pred / (P_pred + self.R)  # Kalman gain
        self.x_hat = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred

        return KalmanResult(
            raw_value=measurement,
            filtered_value=self.x_hat,
            gain=K,
            estimate_variance=self.P,
        )

    def reset(self):
        self.x_hat = None
        self.P = 1.0
