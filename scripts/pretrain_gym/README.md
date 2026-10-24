# pretrain-gym images

Images for [`pretrain-gym`](../../environments/pretrain_gym/README.md). Neither holds environment code or evaluation data.

- `Dockerfile.agent` → `pretrain-gym-agent:v1`: the agent's workspace, with Python data tools, CPU `torch` and `transformers`, and fastText `lid.176` in `/opt/models/`. Built with `TORCH_INDEX` set to a CUDA index → `pretrain-gym-agent:v1-gpu`, for a workspace with a GPU (`workspace.gpu`).
- `Dockerfile.scorer` → `pretrain-gym-scorer:v1`: Prime scorer VMs only (PyTorch 2.14, CUDA 12.6). Docker runs score on the host.
- `push_prime_images.sh <tag>`: builds both images in the Prime registry, which Prime VMs require.
- `rescore.py`: final scoring of any corpus directory on this host's GPUs, with a chosen model, token budget, and seed, reporting validation and test loss and the loss on extra evaluation sets; `--eval-subsets` and `--eval-splits` evaluate fewer sets while still screening against all of them. Final corpora are kept for it with `scoring.keep_final_corpus`.
- `build_corpus.py`: a corpus directory from a manifest of Hub datasets and weights in the format the pilot agents submitted, each source's stream prefix filling its share of a token budget, documents shuffled across sources.
- `hellaswag.py`: HellaSwag validation accuracy of weights exported by final scoring (`scoring.export_weights`).
- `redteam_decontamination.py`: on CPU, how much held-out text gets through scoring's decontamination under transformations (case, Unicode, templates, markup, homoglyphs, dropped or inserted words), and what it removes from FineWeb and from public copies of Paloma sources (WikiText-103, a C4 validation shard). Run it with `PYTHONHASHSEED=0`.

```bash
docker build -f scripts/pretrain_gym/Dockerfile.agent -t pretrain-gym-agent:v1 scripts/pretrain_gym
docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 \
    -f scripts/pretrain_gym/Dockerfile.agent -t pretrain-gym-agent:v1-gpu scripts/pretrain_gym

uv pip install -e "./environments/pretrain_gym[train]"
uv run python scripts/pretrain_gym/rescore.py ~/.cache/pretrain_gym/rollouts/<rollout>/final/corpus \
    --out rescores/<rollout>-seed1 --seed 1
```
