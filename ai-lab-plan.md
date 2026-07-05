# AI Lab — Project Plan & Setup Log

**Host:** emo-server (Ubuntu 24.04)
**GPU:** NVIDIA GeForce RTX 2070 (8 GB VRAM, Turing / Compute 7.5)
**Goal:** Run AI models locally for three planned projects:

1. **Image search engine** — text-to-image search over personal photo library, with optional facial recognition
2. **Blog post generator** — drafts new posts using past dtraleigh.com content as context
3. **Local chatbot** — named assistant with scoped capabilities (starting with weather), future voice I/O

**Operating principles:**

- Open source where possible
- Run as much as possible offline; live data integrations (weather, etc.) are acceptable but AI inference stays local
- Use existing infrastructure (Postgres, Docker, nginx) rather than adding parallel stacks

---

## Architecture Overview

Six layers, built bottom-up. Each layer has a clear verification step before moving on.

| Layer | Component | Status |
|-------|-----------|--------|
| 1 | NVIDIA driver on host | ✅ Complete |
| 2 | NVIDIA Container Toolkit (GPU in Docker) | ✅ Complete |
| 3 | Ollama (model serving) | ✅ Complete |
| 4 | pgvector (vector storage in Postgres) | ✅ Complete |
| 5 | Django integration (deps, multi-DB, router) | ✅ Complete |
| 6 | ~~nginx reverse proxy~~ (folded into Layer 5 — casa's existing nginx serves everything) | ✅ N/A |

---

## Layer 1: NVIDIA Driver on Host ✅

**Purpose:** Give the kernel and userspace programs the ability to talk to the GPU. Replaces the default `nouveau` driver with NVIDIA's proprietary driver.

**Note on CUDA toolkit:** The full `nvidia-cuda-toolkit` (compiler, headers) is *not* installed. Modern ML libraries bundle their own CUDA runtime; only the host driver is needed. Avoids version-pinning headaches.

### Steps

```bash
# Check current state
nvidia-smi                        # likely fails before install
uname -r                          # note kernel version

# See what Ubuntu recommends
ubuntu-drivers devices

# Install the recommended driver
sudo ubuntu-drivers install
# (or: sudo apt install nvidia-driver-XXX for a specific version)

# Reboot
sudo reboot
```

### Verification

```bash
nvidia-smi
```

Should show:
- Driver version
- CUDA Version (max supported by driver, *not* installed toolkit)
- RTX 2070 with 8192 MiB VRAM
- Temperature, power draw, processes

**Current state on emo-server:** Driver 595.71.05, CUDA 13.2 max supported.

### Troubleshooting

- **Secure Boot:** If `nvidia-smi` fails after install, check `mokutil --sb-state`. If enabled, either disable in BIOS or complete MOK enrollment at boot.
- **Wrong kernel:** If you upgrade kernels, DKMS should auto-rebuild the module. If not, `sudo dpkg-reconfigure nvidia-dkms-XXX`.

---

## Layer 2: NVIDIA Container Toolkit ✅

**Purpose:** Let Docker containers access the GPU via `--gpus all`. Bridge between Docker daemon and NVIDIA driver.

### Steps

```bash
# Add NVIDIA's GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add the repo
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure Docker to use it
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Verification

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

Should show the same GPU table as the host. Confirms the container can see the GPU.

### Prerequisite: Docker setup

A clean Docker install on 24.04 (if not already present):

```bash
# Remove any old packages
sudo apt remove docker docker-engine docker.io containerd runc

# Add Docker's official repo
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Allow non-root use
sudo usermod -aG docker $USER

# Enable on boot
sudo systemctl enable docker.service
sudo systemctl enable containerd.service
```

**Important gotcha:** Group membership is read at login time. If using tmux, kill the tmux server (`tmux kill-server`) after adding yourself to the `docker` group — existing tmux sessions hold the old group list. SSH back in, restart tmux.

### Verification of Docker

```bash
groups                            # should include 'docker'
docker run --rm hello-world       # should succeed without sudo
systemctl status docker           # should be active and enabled
```

---

## Layer 3: Ollama (Model Serving) ✅

**Purpose:** Run a daemon that loads LLMs into VRAM and exposes them over HTTP. Multiple projects share one Ollama instance.

**Mental model:** The model is a file on disk. Ollama is the engine. Your apps are clients hitting `http://emo-server:11434/api/...`.

### Steps

```bash
# Create persistent storage for models (chose larger volume on emo-server)
sudo mkdir -p /mnt/server_media2/ollama
sudo chown $USER:$USER /mnt/server_media2/ollama

# Run the container
docker run -d \
  --name ollama \
  --gpus all \
  --restart unless-stopped \
  -v /mnt/server_media2/ollama:/root/.ollama \
  -p 11434:11434 \
  ollama/ollama
```

**Flag reference:**
- `-d` — detached
- `--gpus all` — GPU passthrough (requires Layer 2)
- `--restart unless-stopped` — auto-start on boot, respects manual stops
- `-v` — bind mount; models persist across container rebuilds
- `-p 11434:11434` — exposes API on all interfaces; for localhost-only use `-p 127.0.0.1:11434:11434`

### Verification

```bash
docker ps                         # ollama container should be running
docker logs ollama                # look for: library=CUDA ... description="NVIDIA GeForce RTX 2070"
```

Pull a model and test:

```bash
docker exec -it ollama ollama pull llama3.1:8b
docker exec -it ollama ollama run llama3.1:8b "What is the capital of North Carolina?"

# HTTP API test
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Why is the sky blue? Answer in one sentence.",
  "stream": false
}'
```

**Current state on emo-server:**
- Ollama 0.30.8
- `llama3.1:8b` pulled (~4.9 GB)
- Inference verified at ~63 tokens/second
- GPU detected: CUDA0, 7.5 GiB available

### Operational notes

- **Keep-alive:** Default `OLLAMA_KEEP_ALIVE=5m` — model unloads from VRAM after 5 min idle. Override with `-e OLLAMA_KEEP_ALIVE=24h` for always-warm services.
- **VRAM ceiling:** Llama 3.1 8B Q4 uses ~5-6 GB. Comfortable for one model at a time + a smaller secondary model (e.g., CLIP).
- **Updates:** `docker pull ollama/ollama && docker stop ollama && docker rm ollama` then re-run. Models in the bind-mount survive.
- **Token speed measurement:** From API response, `eval_count / (eval_duration / 1e9)` = tokens/sec.

---

## Layer 4: pgvector ✅

**Purpose:** Add vector storage to existing PostgreSQL 17. All three projects need it — image embeddings, blog chunks, chatbot knowledge — because LLMs can't answer questions about your private data unless you retrieve the relevant pieces at query time and feed them into the model's context. This is the retrieval half of RAG (Retrieval-Augmented Generation).

**Mental model:** Ollama gives you the models; pgvector gives you the memory those models operate on. Your apps orchestrate them.

**Why pgvector and not a dedicated vector DB:** you already run Postgres, your data is small (thousands of vectors, not millions), and adding another daemon is complexity you'd forget the reason for in three years. Dedicated vector DBs (Pinecone, Qdrant, etc.) earn their keep at massive scale with strict SLAs.

### Steps

```bash
# Confirm Postgres version, then install matching extension package
psql --version
sudo apt install postgresql-17-pgvector
```

Enter psql as superuser and create the database, extension, and app user:

```bash
sudo -u postgres psql
```

```sql
CREATE DATABASE ai_lab;
\c ai_lab
CREATE EXTENSION vector;
\dx

-- Dedicated app user (don't use postgres superuser for app connections)
CREATE USER ai_lab_user WITH PASSWORD 'pick-a-real-password';
GRANT ALL PRIVILEGES ON DATABASE ai_lab TO ai_lab_user;
\c ai_lab
GRANT ALL ON SCHEMA public TO ai_lab_user;
\q
```

**PG15+ gotcha:** the second `GRANT ON SCHEMA public` is required. Recent Postgres versions tightened default schema permissions — without it your user can connect but can't create tables. Symptom: "permission denied for schema public" the first time you `CREATE TABLE`.

### Verification

```bash
psql -U ai_lab_user -d ai_lab -h localhost
```

The `-h localhost` forces TCP + password auth. Without it, peer authentication fails because the Linux user doesn't match the Postgres user. Always use `-h localhost` for app-style connections.

Test the extension actually works:

```sql
CREATE TABLE test_vectors (
    id serial PRIMARY KEY,
    description text,
    embedding vector(3)
);

INSERT INTO test_vectors (description, embedding) VALUES
    ('apple',  '[1, 0, 0]'),
    ('orange', '[0.9, 0.1, 0]'),
    ('car',    '[0, 0, 1]');

SELECT description, embedding <-> '[1, 0, 0]' AS distance
FROM test_vectors
ORDER BY embedding <-> '[1, 0, 0]'
LIMIT 3;

DROP TABLE test_vectors;
```

Expected output: apple at distance 0, orange at ~0.14, car at ~1.41. Three rows sorted by similarity.

**Current state on emo-server:**
- pgvector 0.8.4 installed in `ai_lab` database
- `ai_lab_user` created with table-creation privileges on `public` schema
- Test query returned correct L2 distances

### Distance operators reference

pgvector provides three distance operators. Which to use depends on the embedding model:

| Operator | Distance type | Common use |
|----------|--------------|------------|
| `<->` | L2 (Euclidean) | General purpose, most text embedders |
| `<=>` | Cosine | CLIP, models trained with cosine loss |
| `<#>` | Negative inner product | When magnitudes carry meaning |

Nomic-embed-text and most modern text embedders work fine with either L2 or cosine. CLIP specifically wants cosine.

### Indexing (deferred)

At current scale (thousands of rows), pgvector with no index does a full scan and computes distance to every row — sub-second, fine. At 100K+ vectors, add an approximate nearest neighbor index:

- **`ivfflat`** — divides vectors into clusters, searches nearest clusters only
- **`hnsw`** — hierarchical navigable small world graph, generally faster queries but slower builds and more memory

Both trade small accuracy for large speedups. Decide per-project when actually needed. Building indexes prematurely just adds complexity.

### Query pattern used by all three projects

Every semantic search boils down to this shape:

```sql
SELECT id, metadata_columns, embedding <=> $query_vector AS distance
FROM some_table
ORDER BY embedding <=> $query_vector
LIMIT N;
```

- Image search: photos table with CLIP embeddings, query is a text embedding of the search terms
- Blog RAG: post_chunks table with text embeddings, query is the user's question
- Chatbot: knowledge table with text embeddings, query is the user's message

---

## Layer 5: Django Integration ✅

**Purpose:** Extend the existing `casa` Django project to talk to Ollama and pgvector. All AI Lab apps live inside casa alongside existing apps (flat structure), sharing its venv, gunicorn, and nginx.

**Architectural decisions:**

- **Option A (unified project):** AI Lab apps live in casa rather than a separate Django project. Chosen for simplicity — casa isn't critical, and if experiments break something we fix it.
- **Separate databases with routers:** `default` points to casa's existing `django_project2`; `ai_lab` is a new connection to the pgvector-enabled `ai_lab` database. A router partitions models by app label prefix.
- **App naming convention:** any AI Lab Django app must start with `ai_lab_` (e.g., `ai_lab_core`, `ai_lab_chatbot`). The router uses this prefix to route models. There is no folder boundary enforcing this — the prefix is the contract.
- **pgvector via Django ORM:** using the `pgvector` Python package's `VectorField` and distance expressions rather than raw SQL.

### Steps executed

**1. Dependencies installed into casa venv:**

```bash
source /home/leo/workspace/casa/.venv/bin/activate
pip install uv
uv pip install ollama pgvector
```

Python 3.12.3; venv at `/home/leo/workspace/casa/.venv/`.

**2. Added `ai_lab` database to `settings.py`:**

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'django_project2',
        'USER': 'django_user',
        'PASSWORD': env("CASA_DB_PASS"),
        'HOST': '127.0.0.1',
        'PORT': '5432',
    },
    'ai_lab': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'ai_lab',
        'USER': 'ai_lab_user',
        'PASSWORD': env("AI_LAB_DB_PASS"),
        'HOST': '127.0.0.1',
        'PORT': '5432',
    },
}

DATABASE_ROUTERS = ['ai_lab_core.routers.AiLabRouter']
```

`AI_LAB_DB_PASS` added to the env source that `env()` reads from.

**3. Created `ai_lab_core` app** (holds shared router and utilities):

```bash
cd /home/leo/workspace/casa
python manage.py startapp ai_lab_core
```

Added `'ai_lab_core'` to `INSTALLED_APPS`.

**4. Wrote the database router** at `ai_lab_core/routers.py`:

```python
class AiLabRouter:
    """Routes models in ai_lab_* apps to the ai_lab database."""

    ai_lab_prefix = 'ai_lab'

    def _is_ai_lab(self, model):
        return model._meta.app_label.startswith(self.ai_lab_prefix)

    def db_for_read(self, model, **hints):
        return 'ai_lab' if self._is_ai_lab(model) else None

    def db_for_write(self, model, **hints):
        return 'ai_lab' if self._is_ai_lab(model) else None

    def allow_relation(self, obj1, obj2, **hints):
        db1 = 'ai_lab' if self._is_ai_lab(type(obj1)) else 'default'
        db2 = 'ai_lab' if self._is_ai_lab(type(obj2)) else 'default'
        return db1 == db2

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        if app_label.startswith(self.ai_lab_prefix):
            return db == 'ai_lab'
        return db == 'default'
```

**5. Ran migrations:**

```bash
python manage.py migrate --database=ai_lab   # sets up ai_lab DB
python manage.py migrate                     # no-op, confirms default untouched
```

### Verification

Django shell test — all three passed:

```python
from ollama import Client
client = Client(host='http://localhost:11434')

# 1. Text generation via Ollama
resp = client.generate(model='llama3.1:8b', prompt='Say hello in five words.')
# → "Hello, how are you today?"

# 2. Embedding via Ollama
emb = client.embed(model='nomic-embed-text', input='Raleigh urban planning')
# → 768-dimensional vector

# 3. pgvector extension check via ORM
from django.db import connections
cursor = connections['ai_lab'].cursor()
cursor.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
# → ('vector', '0.8.4')
```

### Known open item: gunicorn service

`restart_services.sh` fails to find `gunicorn.service` — likely a leftover from the 24.04 reinstall. Not blocking foundation work, but needed before serving casa over HTTP again. To diagnose:

```bash
systemctl list-units --type=service | grep -i gunicorn
systemctl --user list-units --type=service | grep -i gunicorn
```

If nothing turns up, the systemd unit file needs to be recreated. `python manage.py runserver` works for dev/testing in the meantime.

Bonus fix later: the restart script prints "Services restarted successfully!" on failure — should `exit 1` on any systemctl error.

---

## Foundation Complete

All six planned layers are done. From here, project work begins.

## Project Sequencing

Chatbot first (reordered from original plan). Reasons: fastest path to a working demo, exercises the full RAG loop end-to-end, iterative from day one (V1 → voice), and by the time we build image search we'll have all the plumbing patterns established.

1. **Chatbot** — named local assistant with scoped capabilities. V1 is basic chat + system prompt; adds conversation memory, tool calling (starting with weather), pgvector semantic recall, and eventually voice I/O.
2. **Image search** — CLIP-based text-to-image search over personal photos, with optional face recognition. Reuses the Django-Ollama-pgvector pattern from the chatbot.
3. **Blog post generator** — RAG over dtraleigh.com archive to draft new posts.

Each project gets its own plan doc when started.

---

## Open Questions / Notes

*Use this section to capture questions and ideas while exploring the current state before moving on.*

- _(add as they come up)_

---

## Reference: Quick Status Check Commands

```bash
# Host GPU
nvidia-smi

# Containerized GPU
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Ollama health
docker ps | grep ollama
curl http://localhost:11434/api/tags    # lists installed models

# Quick inference test
curl http://localhost:11434/api/generate -d '{"model":"llama3.1:8b","prompt":"hi","stream":false}'

# pgvector check (from casa venv)
source /home/leo/workspace/casa/.venv/bin/activate
python /home/leo/workspace/casa/manage.py dbshell --database=ai_lab
# then: \dx

# Django-to-Ollama sanity check
python /home/leo/workspace/casa/manage.py shell -c "
from ollama import Client
print(Client(host='http://localhost:11434').generate(model='llama3.1:8b', prompt='hi')['response'])
"
```
