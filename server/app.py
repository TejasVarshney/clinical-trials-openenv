"""Compatibility app entrypoint for validators expecting server/app.py at repo root."""

from clinical_trial_env.server.app import app, main


if __name__ == "__main__":
    main()
