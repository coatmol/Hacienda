FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

ARG HACIENDA_GEMMA_BASE_URL=""
ARG HACIENDA_GEMMA_TOKEN=""
ARG HACIENDA_GEMMA_MODEL="gemma"
ENV HACIENDA_GEMMA_BASE_URL=${HACIENDA_GEMMA_BASE_URL}
ENV HACIENDA_GEMMA_TOKEN=${HACIENDA_GEMMA_TOKEN}
ENV HACIENDA_GEMMA_MODEL=${HACIENDA_GEMMA_MODEL}
ENV PYTHONUNBUFFERED=1

# "pipeline" = batch mode (default), "demo" = web UI
ENV HACIENDA_MODE=pipeline

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

RUN mkdir -p /input /output

EXPOSE 8080

# Entrypoint script picks between batch pipeline and demo web app
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]
