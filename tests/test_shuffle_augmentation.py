import unittest
from unittest.mock import patch

import numpy as np

from gpm.data_utils.iterators import round_robin_iter, shuffle_aligned_batch


class _FakeSeqDataset:
    def __init__(self, batches):
        self.dataset = batches

    def make_dataset(self):
        return None


class ShuffleAugmentationTests(unittest.TestCase):
    def test_shuffle_aligned_batch_is_noop_when_probability_zero(self):
        x = np.arange(1 * 16 * 1, dtype=np.float32).reshape(1, 16, 1)
        y = np.arange(1 * 6 * 1, dtype=np.float32).reshape(1, 6, 1)

        x_aug, y_aug = shuffle_aligned_batch(
            x,
            y,
            pool_width=2,
            p_shuffle=0.0,
            min_chunks=3,
            max_chunks=3,
            rng=np.random.default_rng(0),
        )

        np.testing.assert_array_equal(x_aug, x)
        np.testing.assert_array_equal(y_aug, y)

    def test_shuffle_aligned_batch_preserves_flanks_and_alignment(self):
        pool_width = 2
        target_length = 6
        seq_length = 16
        left_flank = 2
        right_flank = 2

        y = np.arange(target_length, dtype=np.float32).reshape(1, target_length, 1)
        x = np.full((1, seq_length, 1), fill_value=-1.0, dtype=np.float32)
        x[:, :left_flank, :] = -7.0
        x[:, -right_flank:, :] = -9.0
        for i in range(target_length):
            start = left_flank + i * pool_width
            end = start + pool_width
            x[:, start:end, :] = float(i)

        x_aug, y_aug = shuffle_aligned_batch(
            x,
            y,
            pool_width=pool_width,
            p_shuffle=1.0,
            min_chunks=3,
            max_chunks=3,
            rng=np.random.default_rng(7),
        )

        np.testing.assert_array_equal(x_aug[:, :left_flank, :], x[:, :left_flank, :])
        np.testing.assert_array_equal(x_aug[:, -right_flank:, :], x[:, -right_flank:, :])

        for i in range(target_length):
            start = left_flank + i * pool_width
            end = start + pool_width
            block_values = x_aug[0, start:end, 0]
            expected = y_aug[0, i, 0]
            self.assertTrue(np.all(block_values == expected))

    def test_round_robin_iter_validates_shuffle_arguments(self):
        with self.assertRaises(ValueError):
            round_robin_iter([], batch_size=1, shuffle=True, p_shuffle=1.2, min_chunks=3, max_chunks=6, pool_width=2)
        with self.assertRaises(ValueError):
            round_robin_iter([], batch_size=1, shuffle=True, p_shuffle=0.5, min_chunks=7, max_chunks=3, pool_width=2)
        with self.assertRaises(ValueError):
            round_robin_iter([], batch_size=1, shuffle=True, p_shuffle=0.5, min_chunks=0, max_chunks=3, pool_width=2)

    def test_round_robin_iter_applies_shuffle_when_enabled(self):
        x = np.full((1, 16, 1), fill_value=-1.0, dtype=np.float32)
        y = np.arange(6, dtype=np.float32).reshape(1, 6, 1)
        x[:, :2, :] = -7.0
        x[:, -2:, :] = -9.0
        for i in range(6):
            x[:, 2 + i * 2 : 2 + (i + 1) * 2, :] = float(i)

        sd = _FakeSeqDataset(batches=[(x.copy(), y.copy())])

        with patch("gpm.data_utils.iterators.tfds.as_numpy", side_effect=lambda ds: iter(ds)):
            outer = round_robin_iter(
                [sd],
                batch_size=1,
                shuffle=True,
                p_shuffle=1.0,
                min_chunks=3,
                max_chunks=3,
                pool_width=2,
                rng=np.random.default_rng(11),
            )
            inner = next(outer)
            x_aug, y_aug = next(inner)

        np.testing.assert_array_equal(x_aug[:, :2, :], x[:, :2, :])
        np.testing.assert_array_equal(x_aug[:, -2:, :], x[:, -2:, :])
        for i in range(6):
            block = x_aug[0, 2 + i * 2 : 2 + (i + 1) * 2, 0]
            self.assertTrue(np.all(block == y_aug[0, i, 0]))

    def test_shuffle_aligned_batch_returns_per_sample_decisions(self):
        x = np.arange(2 * 16 * 1, dtype=np.float32).reshape(2, 16, 1)
        y = np.arange(2 * 6 * 1, dtype=np.float32).reshape(2, 6, 1)

        x_aug, y_aug, decisions = shuffle_aligned_batch(
            x,
            y,
            pool_width=2,
            p_shuffle=0.0,
            min_chunks=3,
            max_chunks=3,
            rng=np.random.default_rng(0),
            return_decisions=True,
        )

        np.testing.assert_array_equal(x_aug, x)
        np.testing.assert_array_equal(y_aug, y)
        self.assertEqual(len(decisions), 2)
        self.assertEqual(decisions[0]["applied"], False)
        self.assertEqual(decisions[0]["chunks"], 0)
        self.assertEqual(decisions[1]["applied"], False)
        self.assertEqual(decisions[1]["chunks"], 0)

    def test_round_robin_iter_logs_per_sample_shuffle_decisions_when_enabled(self):
        x = np.full((1, 16, 1), fill_value=-1.0, dtype=np.float32)
        y = np.arange(6, dtype=np.float32).reshape(1, 6, 1)
        x[:, :2, :] = -7.0
        x[:, -2:, :] = -9.0
        for i in range(6):
            x[:, 2 + i * 2 : 2 + (i + 1) * 2, :] = float(i)

        sd = _FakeSeqDataset(batches=[(x.copy(), y.copy())])

        with patch("gpm.data_utils.iterators.tfds.as_numpy", side_effect=lambda ds: iter(ds)):
            with patch("gpm.data_utils.iterators.logging.warning") as log_warning:
                outer = round_robin_iter(
                    [sd],
                    batch_size=1,
                    shuffle=True,
                    p_shuffle=1.0,
                    min_chunks=3,
                    max_chunks=3,
                    pool_width=2,
                    shuffle_log_per_example=True,
                    rng=np.random.default_rng(11),
                )
                inner = next(outer)
                next(inner)

        messages = [str(call.args[0]) for call in log_warning.call_args_list if call.args]
        self.assertTrue(any("shuffle decision" in m and "applied=1" in m for m in messages))


if __name__ == "__main__":
    unittest.main()
