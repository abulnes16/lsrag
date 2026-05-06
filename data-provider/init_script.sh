#!/bin/bash

echo "Esperando a que Ollama inicie..."
# Intentar conectar a Ollama varias veces
for i in {1..10}; do
  if curl -s http://ollama:11434/api/tags > /dev/null; then
    echo "Ollama está listo!"
    break
  fi
  echo "Reintentando en 5 segundos..."
  sleep 5
done

echo "Descargando modelo phi3:mini en Ollama..."
curl -X POST http://ollama:11434/api/pull -d '{"name": "phi3:mini"}'

echo "Ejecutando script de inicialización de datos FlashRAG..."
python init_data.py

echo "Inicialización completada. El contenedor se detendrá."
