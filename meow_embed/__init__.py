"""meow-embed package."""

from importlib.metadata import metadata, version

__version__ = version("meow-embed")
__description__ = metadata("meow-embed")["Summary"]
