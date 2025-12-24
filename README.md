# Tetris AI Engine (Reinforcement Learning Proof of Concept)

A small proof-of-concept project exploring reinforcement learning for playing Tetris.

This repository contains a minimal environment, training/playing scripts, and a small pre-trained model checkpoint. It's intended as a learning / experimentation project rather than a production-ready library.

## Contents

- `main.py` — training / runner entrypoint (project-specific; inspect to see exact behavior).
- `interactive_play.py` — interactive script for human or visualization-driven play.
- `test_tetris.py` — example unit tests / sanity checks.
- `tetris_m1_model.pth` — a saved PyTorch model checkpoint (toy/pretrained).
- `requirements.txt` — Python dependencies.

## Goals

- Provide a compact Tetris RL environment and training loop for experimentation.
- Offer utilities to debug common Gym/Gymnasium observation/action space issues.
- Serve as an educational starting point for RL research or hobby projects.

## Requirements

This project uses Python and the packages listed in `requirements.txt`. To create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate   # on macOS / Linux (zsh, bash)
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you don't have a `requirements.txt` (or you want to install manually), typical dependencies used are:

- gymnasium (or gym)
- numpy
- torch (if using the `.pth` checkpoint)
- pytest (for tests)

Adjust versions to match your system (M1/M2 users may prefer `torch` builds for macOS).

## Quick Start

1. Activate your virtualenv (see Requirements).
2. Run a quick smoke test or training run. The repository contains `main.py` and `interactive_play.py` as entry scripts. Example:

```bash
python main.py
# or for interactive play with rendering (if supported):
python interactive_play.py
```

Inspect the top of those files to see available CLI flags or configuration.

## Project Layout and Notes

If you're adding features, keep the repository small and focused. Typical files you'll edit:

- `tetris` or environment implementation (check for a `tetris` package/dir in repo)
- `main.py` (train/run harness)
- `interactive_play.py` (visualization / manual testing)
- `test_tetris.py` (tests)

Search for where the environment is registered (if using `gym.make`) and make sure it points to the class that sets `observation_space`.

