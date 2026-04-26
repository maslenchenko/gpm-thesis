import logging
import math
import numpy as np

try:
    import tensorflow_datasets as tfds
except Exception:  # pragma: no cover - optional dependency for borzoi backend only
    tfds = None


def _require_tfds():
    if tfds is None:
        raise ImportError(
            "tensorflow_datasets is required for borzoi data iterators. "
            "Install it (e.g. `pip install tensorflow-datasets`) or use `--data_backend ecoli`."
        )


def _validate_shuffle_args(shuffle, p_shuffle, min_chunks, max_chunks, pool_width):
    if not shuffle:
        return
    if not (0.0 <= p_shuffle <= 1.0):
        raise ValueError(f"p_shuffle must be in [0, 1], got {p_shuffle}")
    if min_chunks < 1:
        raise ValueError(f"min_chunks must be >= 1, got {min_chunks}")
    if max_chunks < 1:
        raise ValueError(f"max_chunks must be >= 1, got {max_chunks}")
    if min_chunks > max_chunks:
        raise ValueError(f"min_chunks ({min_chunks}) must be <= max_chunks ({max_chunks})")
    if pool_width is None or pool_width <= 0:
        raise ValueError(f"pool_width must be a positive integer when shuffle is enabled, got {pool_width}")


def shuffle_aligned_batch(xs, ys, pool_width, p_shuffle, min_chunks, max_chunks, rng, return_decisions=False):
    if xs.ndim != 3 or ys.ndim != 3:
        raise ValueError(
            "Expected xs shape [batch, seq_length, seq_depth] and ys shape [batch, target_length, num_targets], "
            f"got xs.ndim={xs.ndim}, ys.ndim={ys.ndim}"
        )

    batch_size, seq_length, _ = xs.shape
    target_batch_size, target_length, _ = ys.shape
    if batch_size != target_batch_size:
        raise ValueError(f"Mismatched batch size between xs ({batch_size}) and ys ({target_batch_size})")

    decisions = None
    if return_decisions:
        decisions = [{"applied": False, "chunks": 0} for _ in range(batch_size)]

    if batch_size == 0 or target_length == 0 or p_shuffle <= 0.0:
        if return_decisions:
            return xs, ys, decisions
        return xs, ys

    target_bp = target_length * pool_width
    if target_bp > seq_length:
        raise ValueError(
            f"Target-aligned sequence span ({target_bp}) exceeds sequence length ({seq_length}); "
            "cannot apply aligned shuffle."
        )

    flank_total = seq_length - target_bp
    if flank_total % 2 != 0:
        raise ValueError(
            f"Expected centered target alignment with even flank span, got seq_length={seq_length}, "
            f"target_length={target_length}, pool_width={pool_width}."
        )

    if min_chunks > target_length:
        raise ValueError(
            f"min_chunks ({min_chunks}) cannot exceed target_length ({target_length}) for this batch."
        )

    left_flank = flank_total // 2
    max_chunks_eff = min(max_chunks, target_length)
    if max_chunks_eff <= 1:
        if return_decisions:
            return xs, ys, decisions
        return xs, ys

    xs_aug = xs
    ys_aug = ys
    for b in range(batch_size):
        if rng.random() >= p_shuffle:
            continue

        k = int(rng.integers(min_chunks, max_chunks_eff + 1))
        if k <= 1:
            continue
        if return_decisions:
            decisions[b] = {"applied": True, "chunks": k}

        cuts = np.sort(rng.choice(np.arange(1, target_length), size=k - 1, replace=False))
        boundaries = np.concatenate(([0], cuts, [target_length]))
        perm = rng.permutation(k)

        if xs_aug is xs:
            xs_aug = xs.copy()
            ys_aug = ys.copy()

        y_chunks = [ys[b, boundaries[i] : boundaries[i + 1], :] for i in perm]
        ys_aug[b, :, :] = np.concatenate(y_chunks, axis=0)

        seq_center = xs[b, left_flank : left_flank + target_bp, :]
        x_chunks = [
            seq_center[boundaries[i] * pool_width : boundaries[i + 1] * pool_width, :]
            for i in perm
        ]
        xs_aug[b, left_flank : left_flank + target_bp, :] = np.concatenate(x_chunks, axis=0)

    if return_decisions:
        return xs_aug, ys_aug, decisions
    return xs_aug, ys_aug


def round_robin_iter(
    seq_datasets,
    batch_size,
    shuffle=False,
    p_shuffle=0.0,
    min_chunks=3,
    max_chunks=6,
    pool_width=None,
    rng=None,
    shuffle_log_per_example=False,
):
    _require_tfds()
    if shuffle and pool_width is None and seq_datasets:
        pool_width = getattr(seq_datasets[0], "pool_width", None)
    _validate_shuffle_args(
        shuffle=shuffle,
        p_shuffle=p_shuffle,
        min_chunks=min_chunks,
        max_chunks=max_chunks,
        pool_width=pool_width,
    )
    if shuffle and seq_datasets:
        target_length = getattr(seq_datasets[0], "target_length", None)
        if target_length is not None:
            if min_chunks > target_length:
                raise ValueError(
                    f"min_chunks ({min_chunks}) cannot exceed target_length ({target_length}) for this dataset."
                )
            if max_chunks > target_length:
                raise ValueError(
                    f"max_chunks ({max_chunks}) cannot exceed target_length ({target_length}) for this dataset."
                )
    rng = rng or np.random.default_rng()

    def make_iter(sd):
        for batch in tfds.as_numpy(sd.dataset):
            yield batch
        sd.make_dataset()

    def round_robin():
        iterators = [make_iter(sd) for sd in seq_datasets]
        while iterators:
            it = iterators.pop(0)
            try:
                yield next(it)
                iterators.append(it)
            except StopIteration:
                pass

    def combine_batch_iter(it):
        batch_idx = 0
        while True:
            try:
                batch = [next(it) for _ in range(batch_size)]
                xs = np.concatenate([b[0] for b in batch], axis=0)
                ys = np.concatenate([b[1] for b in batch], axis=0)
                batch_idx += 1
                if shuffle:
                    if shuffle_log_per_example:
                        xs, ys, decisions = shuffle_aligned_batch(
                            xs=xs,
                            ys=ys,
                            pool_width=pool_width,
                            p_shuffle=p_shuffle,
                            min_chunks=min_chunks,
                            max_chunks=max_chunks,
                            rng=rng,
                            return_decisions=True,
                        )
                        for sample_idx, decision in enumerate(decisions):
                            logging.warning(
                                "shuffle decision: "
                                f"batch={batch_idx} sample={sample_idx} "
                                f"applied={1 if decision['applied'] else 0} chunks={int(decision['chunks'])}"
                            )
                    else:
                        xs, ys = shuffle_aligned_batch(
                            xs=xs,
                            ys=ys,
                            pool_width=pool_width,
                            p_shuffle=p_shuffle,
                            min_chunks=min_chunks,
                            max_chunks=max_chunks,
                            rng=rng,
                        )
                yield xs, ys
            except StopIteration:
                break

    def make_outer_iter():
        while True:
            yield combine_batch_iter(round_robin())

    return make_outer_iter()


def real_data_iter(seq_datasets):
    _require_tfds()

    def make_iter():
        for sd in seq_datasets:
            for batch in tfds.as_numpy(sd.dataset):
                yield batch
            sd.make_dataset()
    while True:
        yield make_iter()


def fake_data_iter(seq_datasets, seq_length, seq_depth, target_length, num_targets, lam=1.0, seed=0):
    batch_size = seq_datasets[0].batch_size
    rng = np.random.default_rng(seed)

    def make_iter():
        for sd in seq_datasets:
            for i in range(0, sd.num_seqs, sd.batch_size):
                batch_size_local = min(sd.batch_size, sd.num_seqs - i)
                x = rng.integers(0, seq_depth, size=(batch_size_local, seq_length))
                x = np.eye(seq_depth, dtype=np.float32)[x]
                y = rng.poisson(lam, size=(batch_size_local, target_length, num_targets)).astype(np.float32)
                yield x, y

    while True:
        yield make_iter()


def batch_limiter(it, limit, first):
    def make_iter(batch_iter):
        for i, batch in enumerate(batch_iter):
            if i + 1 >= (first or 0):
                yield batch
            if limit and i + 1 >= limit + (first or 0):
                break
    while True:
        yield make_iter(next(it))


def count_batches(seq_datasets, batch_size, limit=None, first=None):
    n = sum(math.ceil(sd.num_seqs / batch_size) for sd in seq_datasets)
    if first is not None:
        n = max(n - first, 0)
    if limit is not None:
        n = min(n, limit)
    return n
