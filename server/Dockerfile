FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# RUN apt update && apt-get install -y 

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./models/ ./models/

COPY static/ ./static/
COPY main.py .
COPY model_utils.py .

RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
