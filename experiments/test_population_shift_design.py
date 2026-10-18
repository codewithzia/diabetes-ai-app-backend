import unittest

from experiments.population_shift_design import (
    TARGET_STATES,
    make_partitions,
    normalize_state,
    sample_feedback,
)


class PopulationShiftDesignTests(unittest.TestCase):
    def setUp(self):
        # 8 states, two held out, both outcomes within every state.
        self.states = []
        self.labels = []
        for state in range(1, 9):
            for i in range(100):
                self.states.append(f"{state}.0")
                self.labels.append(i % 5 == 0)

    def test_state_normalization(self):
        self.assertEqual(normalize_state("24.0"), "24")
        with self.assertRaises(ValueError):
            normalize_state("24.5")

    def test_partitions_are_deterministic_complete_and_disjoint(self):
        targets = ("7", "8")
        first = make_partitions(self.states, self.labels, targets, seed=17)
        second = make_partitions(self.states, self.labels, targets, seed=17)
        self.assertEqual(first, second)
        all_rows = [i for rows in first.values() for i in rows]
        self.assertEqual(sorted(all_rows), list(range(len(self.states))))
        self.assertEqual(len(all_rows), len(set(all_rows)))
        source_states = {
            self.states[i].split(".")[0]
            for name in ("source_train", "source_validation", "source_test")
            for i in first[name]
        }
        target_states = {
            self.states[i].split(".")[0]
            for name in ("target_feedback_pool", "target_test")
            for i in first[name]
        }
        self.assertEqual(target_states, set(targets))
        self.assertTrue(source_states.isdisjoint(target_states))

    def test_feedback_is_reproducible_stratified_and_pool_bounded(self):
        pool = list(range(200))
        first = sample_feedback(pool, self.labels, 40, seed=123)
        second = sample_feedback(pool, self.labels, 40, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 40)
        self.assertTrue(set(first).issubset(pool))
        self.assertLessEqual(abs(sum(self.labels[i] for i in first) - 8), 1)
        self.assertEqual(sample_feedback(pool, self.labels, 0, seed=1), [])

    def test_unknown_target_state_is_rejected(self):
        with self.assertRaises(ValueError):
            make_partitions(self.states, self.labels, ("99",))


if __name__ == "__main__":
    unittest.main()
