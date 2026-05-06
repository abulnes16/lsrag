FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir flashrag datasets requests

COPY init_script.sh .
COPY init_data.py .

RUN chmod +x init_script.sh

CMD ["./init_script.sh"]
