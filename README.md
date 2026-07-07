# Hacienda

## How to run (for judges)

This project is fully containerized using Docker (including FFmpeg and all Python dependencies).

To build and run the application locally:

```bash
# 1. Build the Docker image
docker build -t hacienda .

# 2. Run the application
docker run --rm -it hacienda
```
