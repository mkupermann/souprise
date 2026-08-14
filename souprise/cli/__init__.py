"""Command-line interface for Souprise."""

from .chat import app as chat_app
from .index import app as index_app
from .train import app as train_app

__all__ = ["chat_app", "train_app", "index_app"]
