#!/bin/bash

echo "Esperando a que Ollama (Host) inicie..."
# Intentar conectar a Ollama en el host varias veces
for i in {1..10}; do
  if curl -s http://host.docker.internal:11434/api/tags > /dev/null; then
    echo "Ollama en el host está listo!"
    break
  fi
  echo "Reintentando en 5 segundos..."
  sleep 5
done

echo "Ejecutando script de inicialización de datos FlashRAG..."
python init_data.py

echo "Inicialización completada. El contenedor se detendrá."
