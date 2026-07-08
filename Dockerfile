FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

ARG HACIENDA_GEMMA_BASE_URL=""
ARG HACIENDA_GEMMA_TOKEN=""
ARG HACIENDA_GEMMA_MODEL="gemma"
ENV HACIENDA_GEMMA_BASE_URL=${HACIENDA_GEMMA_BASE_URL}
ENV HACIENDA_GEMMA_TOKEN=${HACIENDA_GEMMA_TOKEN}
ENV HACIENDA_GEMMA_MODEL=${HACIENDA_GEMMA_MODEL}
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

RUN mkdir -p /input /output

# Run your app
CMD ["python", "main.py"]
