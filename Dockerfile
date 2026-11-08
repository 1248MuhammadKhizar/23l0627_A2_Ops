FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=2
WORKDIR /app

# dependencies first: this layer is cached until requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# source afterwards: editing code does not trigger a reinstall
COPY app.py .
COPY src/ src/
COPY templates/ templates/
COPY static/ static/

RUN useradd --create-home --uid 1001 appuser && chown -R appuser /app
USER appuser

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health').status==200 else 1)"

CMD ["python", "app.py"]
