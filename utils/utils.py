def dict_has_at_least_two_nonzero_values(probabilities: dict) -> bool:
    """Return True if there are at least two non-zero entries in the dictionary."""
    non_zero_count = sum(1 for value in probabilities.values() if value > 0)
    return non_zero_count >= 2
