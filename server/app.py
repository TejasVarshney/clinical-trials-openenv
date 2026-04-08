"""Compatibility app entrypoint for validators expecting server/app.py at repo root."""

from clinical_trial_env.server.app import app, main as _package_main


def main(host: str = "0.0.0.0", port: int = 8000):
    """Forward to the canonical package server main function."""
    _package_main(host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
