"""
Command-line entrypoints for SEAWRD.

Each module here exposes a ``main(argv)`` function and an ``if __name__ == "__main__"`` guard, and is registered as a
console script in pyproject.toml (``seawrd-train``, ``seawrd-predict``). They may also be run directly without
installation via ``python -m seawrd.cli.train`` / ``python -m seawrd.cli.predict``.
"""
