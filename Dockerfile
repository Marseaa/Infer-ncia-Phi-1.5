FROM python:3.9-slim

COPY . /app
WORKDIR /app

RUN pip install torch transformers

CMD ["python", "app.py"]
