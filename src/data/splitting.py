"""Stratified CV outer split + inner TRAIN/CAL-A/CAL-B split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit


@dataclass
class SplitIndices:
    train: np.ndarray
    cal_a: np.ndarray
    cal_b: np.ndarray
    test: np.ndarray
    repeat: int
    fold: int
    seed: int

    def assert_no_overlap(self) -> None:
        parts = [self.train, self.cal_a, self.cal_b, self.test]
        all_idx = np.concatenate(parts)
        if len(all_idx) != len(np.unique(all_idx)):
            raise AssertionError("row leakage: overlapping split indices")
        for name, a in [("train", self.train), ("cal_a", self.cal_a), ("cal_b", self.cal_b), ("test", self.test)]:
            if len(a) != len(np.unique(a)):
                raise AssertionError(f"duplicate indices within {name}")


def make_outer_splits(y: np.ndarray, n_splits: int = 5, n_repeats: int = 2, seed: int = 42):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    splits = []
    for i, (rest_idx, test_idx) in enumerate(rskf.split(np.zeros(len(y)), y)):
        repeat = i // n_splits
        fold = i % n_splits
        splits.append((repeat, fold, rest_idx, test_idx))
    return splits


def split_rest_into_train_cala_calb(
    y_rest: np.ndarray,
    rest_idx: np.ndarray,
    seed: int,
    train_frac: float = 0.625,
    cal_a_frac: float = 0.1875,
    cal_b_frac: float = 0.1875,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split remainder into TRAIN / CAL-A / CAL-B with approximate protocol fractions."""
    if not np.isclose(train_frac + cal_a_frac + cal_b_frac, 1.0):
        raise ValueError("inner fractions must sum to 1")

    # First peel TRAIN vs (CAL-A+CAL-B)
    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    tr_loc, cal_loc = next(sss1.split(np.zeros(len(y_rest)), y_rest))
    train_idx = rest_idx[tr_loc]
    cal_rest_idx = rest_idx[cal_loc]
    y_cal_rest = y_rest[cal_loc]

    # Split cal rest into CAL-A / CAL-B equally (each ~18.75% of remainder)
    cal_a_share = cal_a_frac / (cal_a_frac + cal_b_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=cal_a_share, random_state=seed + 1)
    a_loc, b_loc = next(sss2.split(np.zeros(len(y_cal_rest)), y_cal_rest))
    cal_a_idx = cal_rest_idx[a_loc]
    cal_b_idx = cal_rest_idx[b_loc]
    return train_idx, cal_a_idx, cal_b_idx


def build_split_for_fold(
    y: np.ndarray,
    *,
    n_splits: int,
    n_repeats: int,
    seed: int,
    repeat_index: int,
    fold_index: int,
    train_frac: float = 0.625,
    cal_a_frac: float = 0.1875,
    cal_b_frac: float = 0.1875,
) -> SplitIndices:
    splits = make_outer_splits(y, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    match = [s for s in splits if s[0] == repeat_index and s[1] == fold_index]
    if not match:
        raise ValueError(f"no split for repeat={repeat_index} fold={fold_index}")
    repeat, fold, rest_idx, test_idx = match[0]
    fold_seed = seed + 1000 * repeat + fold
    y_rest = y[rest_idx]
    train_idx, cal_a_idx, cal_b_idx = split_rest_into_train_cala_calb(
        y_rest,
        rest_idx,
        seed=fold_seed,
        train_frac=train_frac,
        cal_a_frac=cal_a_frac,
        cal_b_frac=cal_b_frac,
    )
    out = SplitIndices(
        train=np.sort(train_idx),
        cal_a=np.sort(cal_a_idx),
        cal_b=np.sort(cal_b_idx),
        test=np.sort(test_idx),
        repeat=repeat,
        fold=fold,
        seed=fold_seed,
    )
    out.assert_no_overlap()
    return out
