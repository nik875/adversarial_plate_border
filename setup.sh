#!/bin/bash
set -e
python -m venv env && source env/bin/activate && pip install -r requirements.txt && pip uninstall -y opencv-python && pip install --force-reinstall opencv-python-headless && python init_alpr.py
