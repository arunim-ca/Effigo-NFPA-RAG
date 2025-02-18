FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/docs

COPY docs/ /app/docs/

ENV PORT=8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
