#!/bin/sh
set -e

if [ "$HACIENDA_MODE" = "demo" ]; then
  echo "Starting Hacienda demo web app on port 8080..."
  exec python -m uvicorn demo.web_app:app --host 0.0.0.0 --port 8080
else
  echo "Running Hacienda batch pipeline..."
  exec python main.py
fi
