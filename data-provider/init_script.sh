#!/bin/bash

echo "Waiting for Ollama (Host) to start..."
# Attempt to connect to Ollama on the host multiple times
for i in {1..10}; do
  if curl -s http://host.docker.internal:11434/api/tags > /dev/null; then
    echo "Ollama on the host is ready!"
    break
  fi
  echo "Retrying in 5 seconds..."
  sleep 5
done

echo "Running FlashRAG data initialization script..."
python init_data.py

echo "Initialization completed. The container will stop."
