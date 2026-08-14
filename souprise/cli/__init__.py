"""Command-line interface for Souprise."""

from .chat import app as chat_app
from .index import app as index_app
from .main import app
from .train import app as train_app

__all__ = ["app", "chat_app", "train_app", "index_app"]
