FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir prefect scikit-learn pandas

COPY ml_workflow.py .
COPY results.txt .

CMD ["python", "ml_workflow.py"]

