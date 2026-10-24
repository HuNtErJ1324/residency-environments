# pretrain-gym

An agent builds a pretraining corpus for a fixed language model. The model is trained on the corpus and scored by its cross-entropy on held-out text (Paloma by default).

- **ID:** `pretrain-gym`
- **Tags:** data-curation, pretraining, huggingface, datasets, tool-use, multi-turn, train, eval

## How it works

- **Workspace:** a Docker container or Prime VM with no network, no GPU unless `workspace.gpu` gives it one, and data tools (`datasets`, `polars`, `duckdb`, `datasketch`, fastText `lid.176`, `torch`, `transformers`). `trainer/` holds the training and corpus-preparation code; `python turns.py` prints the turns left.
- **Deliverable:** `corpus/`, made of `.jsonl`, `.jsonl.gz`, `.jsonl.zst`, or `.parquet` files with a `text` field. Training reads the files in sorted order up to `scoring.train_tokens` and repeats a shorter corpus.
- **`hf` tools** (`search`, `repo_info`, `download`, `search_papers`, `read_paper`): Hugging Face datasets, models, and papers as of `hub.cutoff_date`. They run on the host, and its token never enters the workspace.
- **`scorer` tools** (`score`, `score_status`, `cancel_score`): train on a snapshot of `corpus/` and evaluate on the validation split, which the prompt calls the public part of the held-out set. Results give the overall validation loss (never per subset), training and validation curves, and throughput, and are also written to `scores/<job_id>/`. With `scoring.probe_budget`, runs share a token budget, which the prompt states; done and cancelled runs count the tokens they trained on, and failed runs count none.
- **Final score:** after the agent stops, the same training runs on `corpus/` and is evaluated on the test split (the private part).
- **Decontamination:** before any training, every 13-word passage shared with held-out text is cut from the corpus, and documents under 13 words are dropped. A document cut into more than 10 pieces is mostly a copy whose edits broke some runs, and is dropped whole, as in GPT-3.
  - Against Paloma, copies that are verbatim, re-cased, Unicode-altered, or templated are removed, as are 97% of web documents with a word dropped, inserted, or replaced every 12 words and 97% of the public WikiText-2 test split, whose `<unk>` tokens break some runs. Honest text barely changes: FineWeb loses 0.090% of its characters and the Kimi K3 pilot's corpus 0.064%.
  - Copies edited so that no 13-word run survives get through. Trained on an RTX 4070 SUPER at 100M tokens, the public WikiText-103 and Penn Treebank validation and test splits with a word inserted every 12 lowered mean Paloma test loss by 0.14 nats under cuts alone (seed noise is 0.014); the document rule drops the Penn Treebank copies but keeps 88% of the WikiText ones. Chaining shorter matches across small gaps removed those copies too, but copies rewritten a little more (a 4-word gap every 24 words) kept the full gain, so it is not used: no n-gram filter stops a deliberate rewrite. It needs the held-out text, which the agent's Hub token cannot read for Paloma itself; public copies of its sources are the remaining route.

## Reward

`reward = p²`, where `p = (ln V − L) / (ln V − target_loss)`, or `p` itself when negative. `L` is the test loss, averaged over subsets if the set has them, and `ln V` is the loss of a uniform prediction. With the default `target_loss = 0`, a uniform model scores 0 and zero loss scores 1. A missing or unreadable corpus, or a diverged run, scores 0.

## Swapping the model or evaluation set

- **Model:** `scoring.model` names a directory in `pretrain_gym/models/`, or a path to one, holding `model.json` (tokenizer and training settings) and `train.py`. The script runs as `train.py --config model.json --data <dir> --out <dir> --train-tokens <n>`, reads token files in which every document starts with BOS, and writes `metrics.jsonl` and `result.json`. A `metrics.jsonl` line with `"evaluating": true` makes the scorer report the final evaluation as its own stage. With `scoring.export_weights`, final scoring adds `--save-weights <path>` and keeps the weights at `<host_dir>/rollouts/<rollout>/final/out/model.pt`. The default, `nanogpt_speedrun`, is a GPT-2-small-sized modded-nanogpt-style model with the SmolLM2 tokenizer. It runs on A100 or H100 GPUs (FlexAttention needs compute capability 8.0 or newer).
- **Evaluation set:** `scoring.eval_set` names a Hugging Face dataset, a file glob with `{split}`, and the validation and test split names. If the set has subsets, list them in `subsets` and put `{subset}` in the glob; their losses are averaged. With `subsets = []`, the set is scored as a whole. `local_dir` points at a host directory with the same layout instead.

## Running

- **Docker on a GPU host:** the agent runs in a container, and scoring runs on the host's GPUs, which rollouts lease one at a time.
- **Prime from a CPU host:** the agent runs in a Prime CPU VM, and scoring runs on a Prime GPU VM that exists only while there is scoring to do. Prime GPU VMs need an account grant.

```bash
# Docker
docker build -f scripts/pretrain_gym/Dockerfile.agent -t pretrain-gym-agent:v1 scripts/pretrain_gym
uv pip install -e "./environments/pretrain_gym[train]"
uv run vf-eval @ configs/pretrain_gym/smoke.toml

# Prime (after `prime login`)
scripts/pretrain_gym/push_prime_images.sh v1
uv pip install -e ./environments/pretrain_gym
uv run vf-eval @ configs/pretrain_gym/smoke-prime.toml
```

The host reads two Hub tokens: `hub.token_env` for the agent's tools and `scoring.eval_set.token_env` for the evaluation set. Give the tools a token without gated-repo access, so they cannot fetch a gated evaluation set.

## Knobs

Under `env.taskset.task.`; every option is documented in `config.py`.

| Key                                       | Default                 | Meaning                                    |
| ----------------------------------------- | ----------------------- | ------------------------------------------ |
| `scoring.train_tokens`                    | `400000000`             | Training tokens                            |
| `scoring.probe_budget`                    | unlimited               | Tokens all probe runs may train on         |
| `scoring.model`                           | `nanogpt_speedrun`      | Model directory                            |
| `scoring.seed`                            | the model's             | Replaces the model's `train.seed`          |
| `scoring.ngram`                           | `13`                    | Words in a decontamination match           |
| `scoring.eval_set`                        | Paloma, 12 subsets      | Held-out evaluation set                    |
| `scoring.eval_set.disclose`               | `false`                 | Name the set and its subsets in the prompt |
| `scoring.target_loss` / `reward_exponent` | `0` / `2`               | Reward scale                               |
| `scoring.export_weights`                  | `false`                 | Keep the final model's weights             |
| `scoring.keep_final_corpus`               | `false`                 | Keep the final corpus snapshot to re-score |
| `scoring.scorer.backend`                  | `host`                  | `host` or `prime`                          |
| `scoring.scorer.gpus`                     | every GPU               | Host GPUs that scoring leases              |
| `scoring.scorer.prime`                    | `H200_141GB:1`          | Prime scorer VM                            |
| `hub.cutoff_date`                         | `2025-12-31`            | Hub cutoff                                 |
| `workspace.image`                         | `pretrain-gym-agent:v1` | Agent image                                |
| `workspace.gpu`                           | none                    | The workspace's GPUs, stated in the prompt |
| `host_dir`                                | `~/.cache/pretrain_gym` | Host data, caches, and job files           |

The turn limit is `env.agent.max_turns`, which the prompt states. `env.agent.timeout.finalize` must cover a full training run.

**A GPU in the workspace:** set `workspace.gpu` (not `env.agent.runtime.gpu`, which the prompt would not see) and use the agent image built with CUDA PyTorch (see `scripts/pretrain_gym/`). Docker gives the workspace the host's first GPUs, so host scoring must lease others through `scoring.scorer.gpus`; setup refuses an overlap. Concurrent Docker rollouts share those first GPUs. [`smoke-agent-gpu.toml`](../../configs/pretrain_gym/smoke-agent-gpu.toml) is a working profile.

**Re-scoring:** `scripts/pretrain_gym/rescore.py` runs final scoring on any corpus directory, with another model, budget, or seed, and reports both validation and test loss plus any extra evaluation sets.

## Metrics

| Metric                                      | Meaning                                                        |
| ------------------------------------------- | -------------------------------------------------------------- |
| `performance`                               | The reward                                                     |
| `loss`, `loss/<subset>`, `bits_per_byte`    | Final test results (`loss/all` for a set without subsets)      |
| `corpus_error`, `diverged`                  | Why a run scored 0                                             |
| `heldout_characters_removed`                | Held-out text cut at final scoring                             |
| `probes`, `probe_tokens`, `best_probe_loss` | Scorer tool use (`probe_tokens` is what `probe_budget` counts) |
| `downloads`, `download_bytes`               | Hub tool use                                                   |
