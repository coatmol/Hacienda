# Hacienda

Hacienda is a Track 2 video-captioning agent for the AMD Developer Hackathon.
It reads video tasks, samples compact frame batches, asks a hosted Gemma proxy
for visual evidence, generates four caption styles, repairs weak captions, and
writes the required results JSON.

## Runtime contract

In judging, the container reads:

```text
/input/tasks.json
```

and writes:

```text
/output/results.json
```

Local runs fall back to `inputs/tasks.json` and `output/results.json`.

## Gemma proxy configuration

Track 2 does not inject credentials. For the final public image, use a
revocable low-quota token that talks to your AMD Developer Cloud proxy.

Required settings:

```text
HACIENDA_GEMMA_BASE_URL=https://your-proxy.example/v1
HACIENDA_GEMMA_TOKEN=revocable-token
HACIENDA_GEMMA_MODEL=your-hosted-gemma-model
```

To build and run the application locally:

```bash
docker build -t hacienda .
docker run --rm -it hacienda
```

To bake final judging settings into the image:

```bash
docker buildx build --platform linux/amd64 \
  --build-arg HACIENDA_GEMMA_BASE_URL="https://your-proxy.example/v1" \
  --build-arg HACIENDA_GEMMA_TOKEN="revocable-token" \
  --build-arg HACIENDA_GEMMA_MODEL="your-hosted-gemma-model" \
  --tag your-registry/hacienda:latest \
  --push .
```

For a local contract test:

```bash
docker run --rm \
  -v ${PWD}/inputs:/input \
  -v ${PWD}/output:/output \
  hacienda
```
