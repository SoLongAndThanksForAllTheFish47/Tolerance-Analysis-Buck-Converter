import numpy as np
import os

def calculate_v_out(v_in, duty_cycle, efficiency=1.0):
    """Calculates the ideal output voltage of a buck converter."""
    # Simple linear model formula of a step-down converter
    if not (0 <= duty_cycle <= 1):
        raise ValueError("Duty cycle must be between 0 and 1")
    if isinstance(v_in, (int, float)) and v_in < 0:
        raise ValueError("Input voltage cannot be negative")
    return v_in * duty_cycle * efficiency

def run_monte_carlo(v_in, duty_cycle, tolerance, iterations=1000):
    """Simulates V_out distribution based on input voltage tolerances."""
    # e.g -> 3-sigma approach: tolerance represents the 99.7% confidence interval
    v_in_samples = np.random.normal(v_in, (v_in * tolerance) / 3, iterations)
    return calculate_v_out(v_in_samples, duty_cycle)

def get_statistics(samples):
    """Returns the mean and standard deviation of the simulation samples."""
    return float(np.mean(samples)), float(np.std(samples))

def calculate_yield(samples, target_v, allowed_error=0.05):
    """Calculates the percentage of units within the allowed voltage range."""
    if len(samples) == 0:
        raise ZeroDivisionError("Samples array is empty")
    lower_bound = target_v * (1 - allowed_error)
    upper_bound = target_v * (1 + allowed_error)
    within_limits = np.logical_and(samples >= lower_bound, samples <= upper_bound)
    return float(np.sum(within_limits) / len(samples)) * 100

