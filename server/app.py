"""Compatibility app entrypoint for validators expecting server/app.py at repo root."""

from clinical_trial_env.server.app import app, main as _package_main


def main():
    """Forward to the canonical package server main function."""
    _package_main()


if __name__ == "__main__":
    main()
