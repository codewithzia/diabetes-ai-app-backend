"""Label-blind partition and feedback sampling for the random-feedback rerun."""

from __future__ import annotations

import random

try:  # Works both as an ``experiments`` package and as a script sibling.
    from .population_shift_design import (
        SEED, TARGET_STATES, _stratified_chunks, normalize_state,
    )
except ImportError:
    from population_shift_design import (
        SEED, TARGET_STATES, _stratified_chunks, normalize_state,
    )


def _random_chunks(indices, fractions, seed):
    if abs(sum(fractions) - 1.0) > 1e-9 or any(f <= 0 for f in fractions):
        raise ValueError("Fractions must be positive and sum to 1")
    values = list(indices)
    random.Random(seed).shuffle(values)
    counts = [int(len(values) * fraction) for fraction in fractions]
    remainder = len(values) - sum(counts)
    for i in range(remainder):
        counts[i % len(counts)] += 1
    chunks = []
    offset = 0
    for count in counts:
        chunks.append(values[offset:offset + count])
        offset += count
    return chunks


def make_random_partitions(states, labels, target_states=TARGET_STATES, seed=SEED):
    """Split whole target states and make target pool/test assignment label-blind.

    Source train/validation/test are outcome-stratified. Target feedback-pool
    and test membership are a random 50/50 row split that does not inspect
    target labels, so evaluation prevalence is allowed to vary naturally.
    """
    states = list(states)
    labels = list(labels)
    if len(states) != len(labels) or len(states) == 0:
        raise ValueError("States and labels must have equal, non-zero length")
    target_codes = {normalize_state(value) for value in target_states}
    state_codes = [normalize_state(value) for value in states]
    missing = target_codes - set(state_codes)
    if missing:
        raise ValueError(f"Target states absent from data: {sorted(missing)}")
    if any(int(label) not in (0, 1) for label in labels):
        raise ValueError("Labels must be binary 0/1")

    source = [i for i, code in enumerate(state_codes) if code not in target_codes]
    target = [i for i, code in enumerate(state_codes) if code in target_codes]
    source_parts = _stratified_chunks(source, labels, (0.70, 0.15, 0.15), seed)
    target_parts = _random_chunks(target, (0.50, 0.50), seed + 1)
    result = {
        "source_train": source_parts[0],
        "source_validation": source_parts[1],
        "source_test": source_parts[2],
        "target_feedback_pool": target_parts[0],
        "target_test": target_parts[1],
    }
    flattened = [index for part in result.values() for index in part]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(range(len(states))):
        raise AssertionError("Partitions must cover every row exactly once")
    source_state_set = {state_codes[index] for index in source}
    target_state_set = {state_codes[index] for index in target}
    if source_state_set & target_state_set:
        raise AssertionError("Source and target state groups overlap")
    return result


def sample_feedback_random(pool_indices, size, seed):
    """Draw a uniform sample without replacement; no label argument by design."""
    pool = list(pool_indices)
    if size < 0 or size > len(pool):
        raise ValueError("Feedback size must be between zero and pool size")
    return random.Random(seed).sample(pool, size)
