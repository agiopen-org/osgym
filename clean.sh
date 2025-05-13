#!/bin/bash

echo "Before starting OSGym, killing previous OSGym processes..."

# Stop and remove all Docker containers
echo "Stopping Docker containers..."
docker stop $(docker ps -aq) > /dev/null 2>&1
docker rm $(docker ps -aq) > /dev/null 2>&1

# Kill processes more thoroughly
echo "Killing uvicorn processes..."
# First try with sudo
sudo pkill -f "uvicorn" || true
# Then try without sudo in case some are running as current user
pkill -f "uvicorn" || true

# Force kill any remaining uvicorn processes
echo "Force killing any remaining uvicorn processes..."
for pid in $(ps aux | grep uvicorn | grep -v grep | awk '{print $2}'); do
  echo "Killing process $pid..."
  sudo kill -9 $pid 2>/dev/null || kill -9 $pid 2>/dev/null || true
done

echo "Cleanup completed."