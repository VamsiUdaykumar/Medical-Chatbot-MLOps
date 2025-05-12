# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

USER root

# Optionally create log file with open permissions
RUN mkdir -p /mnt/object/data/production/retraining_data_raw && \
    touch /mnt/object/data/production/retraining_data_raw/online_log.json && \
    chmod 666 /mnt/object/data/production/retraining_data_raw/online_log.json

# Ensure model file is available inside the container (if needed)
#COPY food11.pth /app/

# Expose Flask port
EXPOSE 5000

# Command to run Flask (assumes app.py sets FLASK_APP internally or runs directly)
CMD ["python", "-u", "app.py"]
