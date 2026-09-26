# Contributing Guide

Thank you for your interest in contributing to this project.

## Getting Started

1. Fork the repository.
2. Create a feature branch from `main`:
   - `git checkout -b feature/short-description`
3. Create and activate a virtual environment.
4. Install dependencies:
   - `pip install -r requirements.txt`

## Development Guidelines

- Keep changes focused and small.
- Do not break the existing pipeline:
  - Segmentation: `train_segmentation.py`
  - Classification: `train_classification.py`
  - Evaluation plots: `generate_evaluation_plots.py`
- Reuse existing config and utilities where possible (`config.py`, `utils.py`).
- Use clear variable names and add comments for non-obvious logic.

## Code Quality Checklist

Before opening a pull request:

1. Ensure Python files compile:
   - `python -m compileall .`
2. Confirm scripts run with your environment and paths.
3. Update documentation (`README.md`) if behavior changes.
4. Do not commit datasets, large artifacts, or model weights.

## Pull Request Process

1. Open a PR with a clear title and summary:
   - Problem
   - What changed
   - Why it helps
2. Include sample output paths (figures/reports) when relevant.
3. Mention any assumptions and limitations.

## Reporting Issues

When opening an issue, please include:

- OS and Python version
- Error message/traceback
- Script name and command used
- Steps to reproduce

This helps us diagnose quickly and keep the project stable.
