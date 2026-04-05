import numpy as np
import pytest
import Worst_Case_Buck_Converter as wcbc


def test_calculate_v_out_nominal():
    """Verify standard calculation with ideal parameters."""
    assert wcbc.calculate_v_out(12, 0.5) == 7.0
    assert wcbc.calculate_v_out(10, 0.2, efficiency=0.9) == 1.8


def test_calculate_v_out_invalid_input():
    """Ensure the function raises ValueErrors for unphysical inputs."""
    with pytest.raises(ValueError):
        wcbc.calculate_v_out(12, 1.5)
    with pytest.raises(ValueError):
        wcbc.calculate_v_out(-5, 0.5)


def test_run_monte_carlo_stability():
    """Check if simulation returns correct number of positive samples."""
    iterations = 500
    samples = wcbc.run_monte_carlo(12, 0.41, 0.1, iterations=iterations)
    assert len(samples) == iterations
    assert np.all(samples >= 0)


def test_get_statistics_logic():
    """Verify mean and standard deviation logic."""
    test_data = np.array([9.0, 10.0, 11.0])
    mu, sigma = wcbc.get_statistics(test_data)
    assert mu == 10.0
    assert sigma > 0


def test_calculate_yield_edge_cases():
    """Test yield calculation for 100% and 0% success rates."""
    # 100% Yield Case
    perfect_samples = np.array([5.0, 5.05, 4.95])
    assert wcbc.calculate_yield(perfect_samples, 5.0, allowed_error=0.02) == 100.0
    # 0% Yield Case
    bad_samples = np.array([10.0, 10.0])
    assert wcbc.calculate_yield(bad_samples, 5.0, allowed_error=0.1) == 0.0


def test_calculate_yield_empty():
    """Ensure ZeroDivisionError is handled when no samples are provided."""
    with pytest.raises(ZeroDivisionError):
        wcbc.calculate_yield(np.array([]), 5.0)
