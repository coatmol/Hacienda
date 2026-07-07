## How to run

Firstly, Clone the repository

```bash
git clone https://github.com/coatmol/Hacienda.git
```

Assuming docker is installed. [Docker windows install](https://docs.docker.com/desktop/setup/install/windows-install/)

And then onwards you can run the rest in powershell
You must first build the docker container

```bash
docker build -t hacienda .
```

Then as long as you don't change the contents of Dockerfile then you can just run:

```bash
docker run --rm -it -v ${PWD}:/app hacienda
```

Note that any changes in Dockerfile or requirements.txt requires a docker rebuild

## ToDo

- [x] Setup Dockerfile
- [x] Read inputs/tasks.json
- [x] Download clips from url in task json
- [x] Extract audio and some frames from each clip
- [ ] Analyze audio and frames of each clip using a video and audio capable model from Fireworks AI to generate a detailed description of the clip
- [ ] Use the detailed description to generate the 4 styles for each clip
- [ ] Write results to /output/results.json

Result must be in this exact JSON form:

```json
[
  {
    "task_id": "v1",
    "captions": {
      "formal": "...",
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    }
  }
]
```
