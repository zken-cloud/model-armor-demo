# Use the official Python image
FROM python:3.11-alpine

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run as an unprivileged user
RUN adduser -D -u 10001 appuser && chown -R appuser /app
USER appuser

# Run the application
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 120 app:app