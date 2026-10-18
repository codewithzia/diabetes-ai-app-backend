"""Dependency-free split and feedback-sampling rules for population shift.

All identifiers are zero-based row numbers in the input CSV. State is used
only to define source/target membership and is never a model feature.
"""

from __future__ import annotations

import random
from collections import defaultdict


TARGET_STATES = ("24", "36", "48", "53")
SEED = 42


def normalize_state(value: object) -> str:
    """Normalize CSV state codes such as ``'24.0'`` to ``'24'``."""
    raw = str(value).strip()
    if not raw:
        raise ValueError("State code is empty")
    try:
        number = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid state code: {value!r}") from exc
    if not number.is_integer():
        raise ValueError(f"State code must be an integer: {value!r}")
    return str(int(number))


def _stratified_chunks(indices, labels, fractions, seed):
    if abs(sum(fractions) - 1.0) > 1e-9 or any(f <= 0 for f in fractions):
        raise ValueError("Fractions must be positive and sum to 1")
    by_class = defaultdict(list)
    for index in indices:
        by_class[int(labels[index])].append(index)
    chunks = [[] for _ in fractions]
    rng = random.Random(seed)
    for label in sorted(by_class):
        values = by_class[label]
        rng.shuffle(values)
        counts = [int(len(values) * f) for f in fractions]
        remainder = len(values) - sum(counts)
        for i in range(remainder):
            counts[i % len(counts)] += 1
        offset = 0
        for chunk, count in zip(chunks, counts):
            chunk.extend(values[offset:offset + count])
            offset += count
    for chunk in chunks:
        rng.shuffle(chunk)
    return chunks


def make_partitions(states, labels, target_states=TARGET_STATES, seed=SEED):
    """Return disjoint source train/validation/test and target feedback/test.

    Whole states are held out as target. Within each domain, rows are split
    by outcome to keep the pools usable while leaving outcome values out of
    model fitting except where labels are assigned to their partition.
    """
    # Accept any sequence/array-like (numpy arrays are ambiguous in
    # boolean context).
    states = list(states)
    labels = list(labels)
    if len(states) != len(labels) or not states:
        raise ValueError("States and labels must have equal, non-zero length")
    target_codes = {normalize_state(x) for x in target_states}
    state_codes = [normalize_state(x) for x in states]
    missing = target_codes - set(state_codes)
    if missing:
        raise ValueError(f"Target states absent from data: {sorted(missing)}")
    if any(int(y) not in (0, 1) for y in labels):
        raise ValueError("Labels must be binary 0/1")

    source = [i for i, code in enumerate(state_codes) if code not in target_codes]
    target = [i for i, code in enumerate(state_codes) if code in target_codes]
    source_parts = _stratified_chunks(source, labels, (0.70, 0.15, 0.15), seed)
    target_parts = _stratified_chunks(target, labels, (0.50, 0.50), seed + 1)
    result = {
        "source_train": source_parts[0],
        "source_validation": source_parts[1],
        "source_test": source_parts[2],
        "target_feedback_pool": target_parts[0],
        "target_test": target_parts[1],
    }
    flattened = [i for part in result.values() for i in part]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(range(len(states))):
        raise AssertionError("Partitions must cover every row exactly once")
    if ({state_codes[i] for i in source} & {state_codes[i] for i in target}):
        raise AssertionError("Source and target state groups overlap")
    return result


def sample_feedback(pool_indices, labels, size, seed):
    """Deterministically draw an approximately class-stratified feedback set."""
    pool_indices = list(pool_indices)
    labels = list(labels)
    if size < 0 or size > len(pool_indices):
        raise ValueError("Feedback size must be between zero and pool size")
    if size == 0:
        return []
    by_class = defaultdict(list)
    for index in pool_indices:
        by_class[int(labels[index])].append(index)
    if len(by_class) != 2 or any(not by_class[c] for c in (0, 1)):
        raise ValueError("Feedback pool must contain both classes")

    total = len(pool_indices)
    exact_pos = size * len(by_class[1]) / total
    n_pos = round(exact_pos)
    n_neg = size - n_pos
    if n_pos > len(by_class[1]) or n_neg > len(by_class[0]):
        raise ValueError("Requested size exceeds an available class")
    rng = random.Random(seed)
    selected = []
    for label, count in ((0, n_neg), (1, n_pos)):
        rng.shuffle(by_class[label])
        selected.extend(by_class[label][:count])
    rng.shuffle(selected)
    return selected
