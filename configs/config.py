from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class Config:
    class local:
        # Repo-internal prior-analysis dir (notebooks, scripts).
        brain_tree_root = REPO_ROOT / "brain_tree"
        # Actual BrainTreebank data on the M5.
        braintree_data_root = Path("/Users/wojemann/local_data/BrainTree")
        # BaRISTA reference repo (read-only).
        barista_repo_root = Path("/Users/wojemann/local_data/BaRISTA")

    class server:
        # Update these roots if your server checkout lives elsewhere.
        brain_tree_root = Path("/mnt/sauce/littlab/users/wojemann/BrainTree/")
        braintree_data_root = Path("/mnt/sauce/littlab/users/wojemann/BrainTree/")
        barista_repo_root = Path("/mnt/sauce/littlab/users/wojemann/BaRISTA/")


def get_config(path_profile: str = "local"):
    profile = path_profile.strip().lower()
    if profile not in ("local", "server"):
        raise ValueError("path_profile must be 'local' or 'server'")
    return getattr(Config, profile)
