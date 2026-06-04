FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 7860

# Run Flask app on HuggingFace's required port
ENV FLASK_ENV=production
CMD ["python", "app.py"]
