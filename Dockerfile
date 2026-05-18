FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY indra_cloud_proof_app.py .

EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "indra_cloud_proof_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
