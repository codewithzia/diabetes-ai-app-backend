import inspect
import unittest

from experiments.population_shift_random_design import (
    make_random_partitions,
    sample_feedback_random,
)


class RandomFeedbackDesignTests(unittest.TestCase):
    def setUp(self):
        self.states = []
        self.labels = []
        for state in range(1, 9):
            for i in range(100):
                self.states.append(f"{state}.0")
                self.labels.append(i % 5 == 0)

    def test_target_pool_test_split_does_not_depend_on_target_labels(self):
        targets = ("7", "8")
        original = make_random_partitions(self.states, self.labels, targets, seed=91)
        changed = self.labels.copy()
        for i, state in enumerate(self.states):
            if state.startswith(("7.", "8.")):
                changed[i] = not changed[i]
        relabeled = make_random_partitions(self.states, changed, targets, seed=91)
        self.assertEqual(original["target_feedback_pool"], relabeled["target_feedback_pool"])
        self.assertEqual(original["target_test"], relabeled["target_test"])

    def test_random_feedback_sample_is_reproducible_and_label_blind_by_api(self):
        pool = list(range(200))
        first = sample_feedback_random(pool, 40, seed=12)
        second = sample_feedback_random(pool, 40, seed=12)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 40)
        self.assertEqual(len(set(first)), 40)
        self.assertTrue(set(first).issubset(pool))
        self.assertEqual(
            set(inspect.signature(sample_feedback_random).parameters),
            {"pool_indices", "size", "seed"},
        )

    def test_target_states_are_disjoint_and_partitions_cover_rows(self):
        parts = make_random_partitions(self.states, self.labels, ("7", "8"), seed=91)
        all_rows = [i for chunk in parts.values() for i in chunk]
        self.assertEqual(sorted(all_rows), list(range(len(self.states))))
        self.assertEqual(len(all_rows), len(set(all_rows)))
        source_states = {
            self.states[i].split(".")[0]
            for name in ("source_train", "source_validation", "source_test")
            for i in parts[name]
        }
        target_states = {
            self.states[i].split(".")[0]
            for name in ("target_feedback_pool", "target_test")
            for i in parts[name]
        }
        self.assertTrue(source_states.isdisjoint(target_states))
        self.assertEqual(target_states, {"7", "8"})


if __name__ == "__main__":
    unittest.main()
