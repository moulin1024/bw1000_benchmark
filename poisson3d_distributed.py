#!/usr/bin/env python3
"""Compatibility entry point for :mod:`spectral_fd.cli`."""

from __future__ import annotations


def build_parser():
    """Return the package-owned benchmark argument parser."""
    from spectral_fd.cli import build_parser as package_parser

    return package_parser()


def main(argv=None) -> int:
    """Forward the historical script interface to the package CLI."""
    from spectral_fd.cli import main as package_main

    return package_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
