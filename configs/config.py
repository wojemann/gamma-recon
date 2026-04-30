from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class Config:
    class local:
        brain_tree_root = REPO_ROOT / "brain_tree"

    class server:
        # Update this root if your server checkout lives elsewhere.
        brain_tree_root = Path("/mnt/sauce/littlab/users/wojemann/BrainTree/")


def get_config(path_profile: str = "local"):
    profile = path_profile.strip().lower()
    if profile not in ("local", "server"):
        raise ValueError("path_profile must be 'local' or 'server'")
    return getattr(Config, profile)
