#!/bin/bash

bash clean.sh

# Parse command line arguments
LOCAL=false
for arg in "$@"; do
  if [ "$arg" == "--local" ]; then
    LOCAL=true
  fi
done

# Set host based on local flag
if [ "$LOCAL" = true ]; then
  HOST="127.0.0.1"  # localhost
else
  HOST="0.0.0.0"    # all interfaces
fi

# load config.yaml and define all worker ports     
PORTS=($(python3 -c 'import yaml; import sys; print(" ".join(str(p) for p in yaml.safe_load(open("config.yaml"))["ports"]))'))

# Start each worker
for port in "${PORTS[@]}"; do
  echo "Starting Uvicorn on port $port with host $HOST..."
  uvicorn main:app --host "$HOST" --port "$port" &
done

# wait for all background processes
wait
