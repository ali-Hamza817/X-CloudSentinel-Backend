FROM python:3.9

# Create a non-root user but stay as root for installation
RUN useradd -m -u 1000 user
WORKDIR /app

# Install dependencies globally (more reliable in containers)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Binary Verification: Fail build if models are just Git LFS pointers (< 1MB)
RUN ls -lh models/x-cloudsentinel-distilbert/model.safetensors && \
    [ $(stat -c%s "models/x-cloudsentinel-distilbert/model.safetensors") -gt 1048576 ] || \
    (echo "CRITICAL: model.safetensors is too small. Git LFS binaries were not pulled!" && exit 1)

# Ensure the data directory and the whole app is owned by the user
# Hugging Face needs specific permissions for the persistent data
RUN mkdir -p /app/data && chown -R user:user /app && chmod -R 777 /app/data

# Switch to non-root user for security and HF compliance
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    X_CLOUDSENTINEL_API_KEY=Alpha2SecuredX-CloudSentinel-Research-2026

# Railway dynamic port
EXPOSE 8080

# Use 'python -m gunicorn' and support dynamic PORT
CMD ["sh", "-c", "python -m gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --timeout 600 app:app"]

