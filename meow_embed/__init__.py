"""meow-embed package."""

from importlib.metadata import metadata, version

from meow_embed.cache import EmbedCache, EmbedCacheProgress
from meow_embed.client import MeowEmbedClient

__version__ = version("meow-embed")
__description__ = metadata("meow-embed")["Summary"]

__all__ = [
    "EmbedCache",
    "EmbedCacheProgress",
    "MeowEmbedClient",
    "__description__",
    "__version__",
]
