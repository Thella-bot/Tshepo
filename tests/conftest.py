"""Shared pytest fixtures."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tshepo.data.generator import build  # noqa: E402


@pytest.fixture(scope="session")
def small_dataset():
    """A tiny synthetic dataset: 6 users, enough for splitting and training."""
    records, users = build(n_users=6, seed=99, events=(10, 18))
    return records, users


@pytest.fixture(scope="session")
def by_user(small_dataset):
    records, _ = small_dataset
    out = {}
    for r in records:
        out.setdefault(r["user_id"], []).append(r)
    return out
