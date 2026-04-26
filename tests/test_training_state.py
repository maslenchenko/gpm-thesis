import unittest
import os
import pickle
import tempfile

import numpy as np
import torch

from gpm.training.state import TrainConfig, _compute_learning_rate, compute_metrics, run_training_loop


class CountingIdentityModel(torch.nn.Module):
    def __init__(self, max_forward_calls=None):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0
        self.max_forward_calls = max_forward_calls

    def forward(self, x):
        self.forward_calls += 1
        if self.max_forward_calls is not None and self.forward_calls > self.max_forward_calls:
            raise RuntimeError(f"forward called too many times: {self.forward_calls}")
        return x * self.scale


class DummyWandbRun:
    def __init__(self, step=0):
        self.step = step
        self.logged = []
        self.summary = {}

    def log(self, metrics, step=None):
        self.logged.append((metrics, step))


def _infinite_outer_iterator(x, y):
    def _infinite_epoch_iter():
        while True:
            yield x, y

    while True:
        yield _infinite_epoch_iter()


class TrainingStateBoundedIterationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = rng.random((1, 8, 4), dtype=np.float32)
        self.y = rng.random((1, 8, 4), dtype=np.float32)
        self.loss_fn = lambda y_pred, y_true: torch.mean((y_pred - y_true) ** 2)

    def test_compute_metrics_respects_n_batches_with_infinite_iterator(self):
        n_batches = 3
        model = CountingIdentityModel(max_forward_calls=n_batches)
        iterator = _infinite_outer_iterator(self.x, self.y)

        metrics = compute_metrics(
            model=model,
            loss_fn=self.loss_fn,
            iterator=iterator,
            n_batches=n_batches,
            device="cpu",
        )

        self.assertEqual(model.forward_calls, n_batches)
        self.assertIn("loss", metrics)
        self.assertIn("pearson_r", metrics)
        self.assertIn("r_squared", metrics)

    def test_run_training_loop_respects_n_train_batches_with_infinite_iterator(self):
        n_train_batches = 4
        n_valid_batches = 2
        max_calls = n_train_batches + n_valid_batches
        model = CountingIdentityModel(max_forward_calls=max_calls)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

        config = TrainConfig(
            device="cpu",
            max_shift=0,
            max_epochs=1,
            patience=None,
            prevalidate=False,
            amp_enabled=False,
        )
        strand_pair = torch.arange(self.x.shape[-1], dtype=torch.long)

        run_training_loop(
            model=model,
            loss_fn=self.loss_fn,
            optimizer=optimizer,
            train_iter=_infinite_outer_iterator(self.x, self.y),
            valid_iter=_infinite_outer_iterator(self.x, self.y),
            n_train_batches=n_train_batches,
            n_valid_batches=n_valid_batches,
            strand_pair=strand_pair,
            config=config,
            save_filename=None,
        )

        self.assertEqual(model.forward_calls, max_calls)

    def test_run_training_loop_resumes_from_initial_epoch_and_step(self):
        model = CountingIdentityModel(max_forward_calls=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        config = TrainConfig(
            device="cpu",
            max_shift=0,
            max_epochs=2,
            patience=None,
            prevalidate=False,
            amp_enabled=False,
        )
        strand_pair = torch.arange(self.x.shape[-1], dtype=torch.long)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "resume_state.pt")
            run_training_loop(
                model=model,
                loss_fn=self.loss_fn,
                optimizer=optimizer,
                train_iter=_infinite_outer_iterator(self.x, self.y),
                valid_iter=_infinite_outer_iterator(self.x, self.y),
                n_train_batches=1,
                n_valid_batches=1,
                strand_pair=strand_pair,
                config=config,
                save_filename=save_path,
                initial_epoch=1,
                initial_step=5,
            )

            with open(save_path + ".best", "rb") as f:
                payload = pickle.load(f)

        self.assertEqual(payload["epoch"], 2)
        self.assertEqual(payload["step"], 6)
        self.assertEqual(model.forward_calls, 2)

    def test_run_training_loop_writes_step_suffixed_checkpoints(self):
        model = CountingIdentityModel(max_forward_calls=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        config = TrainConfig(
            device="cpu",
            max_shift=0,
            max_epochs=1,
            patience=None,
            prevalidate=False,
            amp_enabled=False,
        )
        strand_pair = torch.arange(self.x.shape[-1], dtype=torch.long)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "step_named.pt")
            run_training_loop(
                model=model,
                loss_fn=self.loss_fn,
                optimizer=optimizer,
                train_iter=_infinite_outer_iterator(self.x, self.y),
                valid_iter=_infinite_outer_iterator(self.x, self.y),
                n_train_batches=1,
                n_valid_batches=1,
                strand_pair=strand_pair,
                config=config,
                save_filename=save_path,
                initial_epoch=0,
                initial_step=0,
            )

            self.assertTrue(os.path.exists(save_path))
            self.assertTrue(os.path.exists(save_path + ".step1"))
            self.assertTrue(os.path.exists(save_path + ".best"))
            self.assertTrue(os.path.exists(save_path + ".best.step1"))

    def test_run_training_loop_replaces_previous_latest_step_checkpoint(self):
        model = CountingIdentityModel(max_forward_calls=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        config = TrainConfig(
            device="cpu",
            max_shift=0,
            max_epochs=2,
            patience=None,
            prevalidate=False,
            amp_enabled=False,
        )
        strand_pair = torch.arange(self.x.shape[-1], dtype=torch.long)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "rolling_latest.pt")
            run_training_loop(
                model=model,
                loss_fn=self.loss_fn,
                optimizer=optimizer,
                train_iter=_infinite_outer_iterator(self.x, self.y),
                valid_iter=_infinite_outer_iterator(self.x, self.y),
                n_train_batches=1,
                n_valid_batches=1,
                strand_pair=strand_pair,
                config=config,
                save_filename=save_path,
                initial_epoch=0,
                initial_step=0,
            )

            self.assertTrue(os.path.exists(save_path))
            self.assertFalse(os.path.exists(save_path + ".step1"))
            self.assertTrue(os.path.exists(save_path + ".step2"))

    def test_run_training_loop_offsets_wandb_step_when_resumed_step_is_behind(self):
        model = CountingIdentityModel(max_forward_calls=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        config = TrainConfig(
            device="cpu",
            max_shift=0,
            max_epochs=1,
            patience=None,
            prevalidate=False,
            amp_enabled=False,
        )
        strand_pair = torch.arange(self.x.shape[-1], dtype=torch.long)
        wandb_run = DummyWandbRun(step=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "wandb_offset.pt")
            run_training_loop(
                model=model,
                loss_fn=self.loss_fn,
                optimizer=optimizer,
                train_iter=_infinite_outer_iterator(self.x, self.y),
                valid_iter=_infinite_outer_iterator(self.x, self.y),
                n_train_batches=1,
                n_valid_batches=1,
                strand_pair=strand_pair,
                config=config,
                save_filename=save_path,
                wandb_run=wandb_run,
                initial_epoch=0,
                initial_step=5,
            )

            with open(save_path + ".best", "rb") as f:
                payload = pickle.load(f)

        self.assertGreaterEqual(len(wandb_run.logged), 3)
        self.assertEqual(wandb_run.logged[0][1], 101)
        self.assertEqual(payload["step"], 6)

    def test_run_training_loop_offsets_wandb_step_when_resumed_step_equals_current(self):
        model = CountingIdentityModel(max_forward_calls=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        config = TrainConfig(
            device="cpu",
            max_shift=0,
            max_epochs=1,
            patience=None,
            prevalidate=False,
            amp_enabled=False,
        )
        strand_pair = torch.arange(self.x.shape[-1], dtype=torch.long)
        wandb_run = DummyWandbRun(step=5)

        run_training_loop(
            model=model,
            loss_fn=self.loss_fn,
            optimizer=optimizer,
            train_iter=_infinite_outer_iterator(self.x, self.y),
            valid_iter=_infinite_outer_iterator(self.x, self.y),
            n_train_batches=1,
            n_valid_batches=1,
            strand_pair=strand_pair,
            config=config,
            save_filename=None,
            wandb_run=wandb_run,
            initial_epoch=0,
            initial_step=5,
        )

        self.assertGreaterEqual(len(wandb_run.logged), 3)
        self.assertEqual(wandb_run.logged[0][1], 6)

    def test_compute_learning_rate_constant_schedule_with_warmup(self):
        config = TrainConfig(learn_rate=1e-3, warmup_steps=2, lr_schedule="constant", cosine_min_lr_ratio=0.0)
        self.assertAlmostEqual(_compute_learning_rate(step=0, config=config), 5e-4, places=12)
        self.assertAlmostEqual(_compute_learning_rate(step=1, config=config), 1e-3, places=12)
        self.assertAlmostEqual(_compute_learning_rate(step=2, config=config), 1e-3, places=12)

    def test_compute_learning_rate_cosine_schedule_after_warmup(self):
        config = TrainConfig(learn_rate=1e-2, warmup_steps=2, lr_schedule="cosine", cosine_min_lr_ratio=0.1)
        total_train_steps = 12

        lr_at_start_of_decay = _compute_learning_rate(step=2, config=config, total_train_steps=total_train_steps)
        lr_mid_decay = _compute_learning_rate(step=7, config=config, total_train_steps=total_train_steps)
        lr_at_end = _compute_learning_rate(step=12, config=config, total_train_steps=total_train_steps)

        self.assertAlmostEqual(lr_at_start_of_decay, 1e-2, places=12)
        self.assertAlmostEqual(lr_mid_decay, 5.5e-3, places=12)
        self.assertAlmostEqual(lr_at_end, 1e-3, places=12)

    def test_compute_learning_rate_validates_arguments(self):
        config = TrainConfig(learn_rate=1e-3, warmup_steps=2, lr_schedule="cosine", cosine_min_lr_ratio=1.1)
        with self.assertRaises(ValueError):
            _compute_learning_rate(step=0, config=config, total_train_steps=10)

        config = TrainConfig(learn_rate=1e-3, warmup_steps=2, lr_schedule="cosine", cosine_min_lr_ratio=0.0)
        with self.assertRaises(ValueError):
            _compute_learning_rate(step=-1, config=config, total_train_steps=10)


if __name__ == "__main__":
    unittest.main()
