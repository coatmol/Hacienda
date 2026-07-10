FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Credentials come from the baked .env file (COPY . . below), loaded at
# runtime by GemmaClient. Do NOT preset them as ENV here: empty-string ENV
# defaults shadow the .env file and silently disable every model call.
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
