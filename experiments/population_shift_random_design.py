"""Label-blind partition and feedback sampling for the random-feedback rerun."""

from __future__ import annotations

import random

try:  # Works both as an ``experiments`` package and as a script sibling.
    from .population_shift_design import (
        SEED, TARGET_STATES, make_partitions, normalize_state,
    )
except ImportError:
    from population_shift_design import (
        SEED, TARGET_STATES, make_partitions, normalize_state,
    )


def make_random_partitions(states, labels, target_states=TARGET_STATES, seed=SEED):
    """Reuse the established split so only feedback sampling changes.

    In particular, target-test row indices remain identical to the initial
    run for a paired comparison. Outcome stratification is used only when
    defining the fixed target pool/test partitions; sample_feedback_random
    below never receives or inspects labels when choosing feedback rows.
    """
    return make_partitions(states, labels, target_states=target_states, seed=seed)


def sample_feedback_random(pool_indices, size, seed):
    """Draw a uniform sample without replacement; no label argument by design."""
    pool = list(pool_indices)
    if size < 0 or size > len(pool):
        raise ValueError("Feedback size must be between zero and pool size")
    return random.Random(seed).sample(pool, size)
