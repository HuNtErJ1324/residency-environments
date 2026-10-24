# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets", "huggingface-hub", "tokenizers", "zstandard"]
# ///
"""Materialize a source manifest into a corpus directory that scoring reads.

A manifest lists Hugging Face datasets with weights, in the format the pilot agents submitted:

    {"sources": [{"id": "...", "revision": "...", "config": "...", "split": "train",
                  "text_field": "text", "weight": 0.82}, ...]}

Each source contributes the prefix of its stream until it holds its share of `--tokens`, counted
with the scoring model's tokenizer (one BOS per document included). Documents from every source are
then shuffled together and written as `.jsonl.zst` shards, so training sees the mixture throughout
instead of one source after another.

usage: uv run scripts/pretrain_gym/build_corpus.py MANIFEST --out CORPUS_DIR --tokens 100000000
"""

import argparse
import json
import random
from pathlib import Path

import zstandard
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

# The tokenizer of the `nanogpt_speedrun` model (its model.json).
TOKENIZER = ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2")
BATCH = 512
SHARD_DOCUMENTS = 50_000


def stream(source: dict, quota: int, tokenizer: Tokenizer) -> tuple[list[str], int]:
    """The prefix of a source's stream holding `quota` tokens, and its token count."""
    rows = load_dataset(
        source["id"], name=source.get("config"), split=source["split"], revision=source["revision"], streaming=True
    )
    texts, tokens, batch = [], 0, []
    for row in rows:
        batch.append(row[source["text_field"]])
        if len(batch) < BATCH:
            continue
        for text, encoding in zip(batch, tokenizer.encode_batch(batch, add_special_tokens=False)):
            texts.append(text)
            tokens += len(encoding.ids) + 1
            if tokens >= quota:
                return texts, tokens
        batch = []
    raise ValueError(f"{source['id']} ran out at {tokens:,} of {quota:,} tokens")


def write_shards(texts: list[str], out: Path) -> None:
    out.mkdir(parents=True)
    for index in range(0, len(texts), SHARD_DOCUMENTS):
        lines = "".join(json.dumps({"text": text}) + "\n" for text in texts[index : index + SHARD_DOCUMENTS])
        with (out / f"part-{index // SHARD_DOCUMENTS:05d}.jsonl.zst").open("wb") as file:
            with zstandard.ZstdCompressor(level=3).stream_writer(file) as writer:
                writer.write(lines.encode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="Corpus directory, which must not exist yet.")
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sources = json.loads(args.manifest.read_text())["sources"]
    total_weight = sum(source["weight"] for source in sources)
    tokenizer = Tokenizer.from_file(hf_hub_download(TOKENIZER[0], "tokenizer.json", revision=TOKENIZER[1]))
    tokenizer.encode_special_tokens = True

    texts, report = [], []
    for source in sources:
        quota = round(args.tokens * source["weight"] / total_weight)
        source_texts, tokens = stream(source, quota, tokenizer)
        texts += source_texts
        report.append(dict(id=source["id"], config=source.get("config"), documents=len(source_texts), tokens=tokens))
        print(json.dumps(report[-1]), flush=True)
    random.Random(args.seed).shuffle(texts)
    write_shards(texts, args.out)
    summary = dict(manifest=str(args.manifest), tokens=args.tokens, seed=args.seed, sources=report)
    (args.out.parent / f"{args.out.name}.build.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
