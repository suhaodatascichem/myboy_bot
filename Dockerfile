FROM python:3.10-slim

WORKDIR /app

# Ensure tzdata is installed for any calendar logic
RUN apt-get update && apt-get install -y tzdata && rm -rf /var/lib/apt/lists/*

# Install python dependencies first to cache them highly in the Docker layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all abstract skills, scripts, and master engine into the container
COPY . .

# Launch the Master Agent Engine
CMD ["python", "main.py"]
