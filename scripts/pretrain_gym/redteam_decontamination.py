"""E1, decontamination red-team: how much held-out text gets through the 13-gram filter, and what the
filter removes from honest data. CPU only.

The indexes and the filter are scoring's own: probes screen the validation split, final scoring screens
both splits. Held-out documents of 50 or more words are sampled from every subset and split, transformed
by each attack, and cleaned. A document leaks by the share of its characters the filter keeps, and by the
share of its 8-word sequences that the kept text still contains after the filter's own normalization
(case, accents, compatibility forms, invisible characters). FineWeb measures what the filter removes from
honest web text; WikiText-103 (also as WikiText-2 with `<unk>`) and a C4 validation shard, public sources
of Paloma subsets, measure what it removes when an agent downloads them.

Needs read access to the gated evaluation set (`HF_TOKEN` or a cached Hub login). Scoring hashes words with
`PYTHONHASHSEED=0`, and so must this script.

usage: PYTHONHASHSEED=0 uv run python scripts/pretrain_gym/redteam_decontamination.py --out e1.json
"""

import argparse
import gzip
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from pretrain_gym.config import ScoringConfig
from pretrain_gym.scoring import RUNNER_DIR, eval_files, eval_parts

sys.path.insert(0, str(RUNNER_DIR))
from decontaminate import NgramIndex, words  # noqa: E402
from prepare import documents, screen_index  # noqa: E402

LEAK_NGRAM = 8
MIN_WORDS = 50
FINEWEB = ("HuggingFaceFW/fineweb", "sample/10BT/000_00000.parquet")
WIKITEXT = ("Salesforce/wikitext", "{config}/{split}-00000-of-00001.parquet")
C4 = ("allenai/c4", "en/c4-validation.00000-of-00008.json.gz")

TOKEN = re.compile(r"\S+")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
FULLWIDTH = str.maketrans({chr(code): chr(code + 0xFEE0) for code in range(0x21, 0x7F)})
HOMOGLYPHS = str.maketrans("aceopx", "асеорх")  # Latin letters to Cyrillic look-alikes
FILLERS = ("the", "and", "of", "also", "very", "just")


def per_token(text: str, change) -> str:
    return TOKEN.sub(lambda match: change(match.group()), text)


def shuffled_sentences(text: str, rng: random.Random) -> str:
    sentences = SENTENCE_END.split(text)
    rng.shuffle(sentences)
    return " ".join(sentences)


def drop_every(text: str, k: int) -> str:
    return " ".join(token for i, token in enumerate(text.split(), 1) if i % k)


def replace_every(text: str, k: int, rng: random.Random) -> str:
    return " ".join(rng.choice(FILLERS) if i % k == 0 else token for i, token in enumerate(text.split(), 1))


def insert_every(text: str, k: int, rng: random.Random) -> str:
    out = []
    for i, token in enumerate(text.split(), 1):
        out.append(token)
        if i % k == 0:
            out.append(rng.choice(FILLERS))
    return " ".join(out)


# Each attack rewrites a held-out document; the first group should be caught, the second probes the limits.
ATTACKS = {
    "verbatim": lambda text, rng: text,
    "upper case": lambda text, rng: text.upper(),
    "fullwidth characters": lambda text, rng: text.translate(FULLWIDTH),
    "combining accents": lambda text, rng: re.sub(r"([aeiouAEIOU])", "\\1\u0301", text),
    "zero-width spaces in words": lambda text, rng: per_token(text, "\u200b".join),
    "HTML paragraphs": lambda text, rng: "".join(f"<p>{line}</p>\n" for line in text.split("\n") if line.strip()),
    "question template": lambda text, rng: (
        f"Read the passage, then answer.\n\nPassage: {text}\n\nQuestion: What is the passage about?\nAnswer:"
    ),
    "shuffled sentences": shuffled_sentences,
    "Cyrillic homoglyphs": lambda text, rng: text.translate(HOMOGLYPHS),
    "HTML tag around every word": lambda text, rng: per_token(text, lambda token: f"<span>{token}</span>"),
    "every 12th word dropped": lambda text, rng: drop_every(text, 12),
    "a word inserted every 12": lambda text, rng: insert_every(text, 12, rng),
    "a word replaced every 12": lambda text, rng: replace_every(text, 12, rng),
    "every 8th word dropped": lambda text, rng: drop_every(text, 8),
    "every 6th word dropped": lambda text, rng: drop_every(text, 6),
}

_indexes: dict[str, NgramIndex] = {}


def _init(paths: dict[str, str], n: int) -> None:
    for name, path in paths.items():
        _indexes[name] = NgramIndex(np.load(path, mmap_mode="r"), n)


def ngram_set(tokens: list[str]) -> set[int]:
    return {hash(tuple(tokens[i : i + LEAK_NGRAM])) for i in range(len(tokens) - LEAK_NGRAM + 1)}


def _attack(args: tuple[int, str]) -> dict:
    """Per attack and screening path: characters in, characters kept, original 8-grams kept, original 8-grams."""
    doc_id, text = args
    rng = random.Random(doc_id)
    original = ngram_set(words(text))
    result = {}
    for name, attack in ATTACKS.items():
        attacked = attack(text, rng)
        for path, index in _indexes.items():
            pieces, _ = index.clean(attacked)
            kept = set().union(*(ngram_set(words(piece)) for piece in pieces))
            result[name, path] = (len(attacked), sum(map(len, pieces)), len(original & kept), len(original))
    return result


def _clean(args: tuple[str, str]) -> tuple[str, int, int, bool, bool]:
    """Screening path, characters, characters kept, whether held-out text was cut, whether too short."""
    path, text = args
    pieces, changed = _indexes[path].clean(text)
    return path, len(text), sum(map(len, pieces)), changed, not pieces and not changed


def heldout_sample(files: dict, parts: list[str], splits: tuple[str, str], per_set: int, seed: int):
    rng = random.Random(seed)
    for part in parts:
        for split in splits:
            texts = [t for path in files[part, split] for t in documents(path, "text", str(path))]
            long_enough = [t for t in texts if len(t.split()) >= MIN_WORDS]
            for text in rng.sample(long_enough, min(per_set, len(long_enough))):
                yield part, split, text


def fineweb(count: int) -> list[str]:
    texts = []
    with HfFileSystem().open(f"datasets/{FINEWEB[0]}/{FINEWEB[1]}", "rb") as file:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=1024, columns=["text"]):
            texts += batch.column(0).to_pylist()
            if len(texts) >= count:
                return texts[:count]
    return texts


def wikitext_articles(split: str, config: str = "wikitext-103-raw-v1") -> list[str]:
    path = hf_hub_download(WIKITEXT[0], WIKITEXT[1].format(config=config, split=split), repo_type="dataset")
    articles, current = [], []
    for line in pq.read_table(path, columns=["text"]).column(0).to_pylist():
        # A top-level heading (" = Title = ") starts a new article; " = = Section = = " does not.
        if re.fullmatch(r" = [^=].* = \n", line) and current:
            articles.append("".join(current))
            current = []
        current.append(line)
    articles.append("".join(current))
    return articles


def c4_validation() -> list[str]:
    path = hf_hub_download(C4[0], C4[1], repo_type="dataset")
    with gzip.open(path, "rt", encoding="utf-8") as lines:
        return [json.loads(line)["text"] for line in lines]


def summarize_clean(results) -> dict:
    totals = defaultdict(lambda: dict(documents=0, characters=0, kept=0, changed=0, short=0))
    for path, chars, kept, changed, short in results:
        entry = totals[path]
        entry["documents"] += 1
        entry["characters"] += chars
        entry["kept"] += kept if not short else chars
        entry["changed"] += changed
        entry["short"] += short
    return {
        path: dict(
            documents=t["documents"],
            documents_cut=t["changed"] / t["documents"],
            characters_removed=1 - t["kept"] / t["characters"],
        )
        for path, t in totals.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-set", type=int, default=100, help="Held-out documents per subset and split.")
    parser.add_argument("--fineweb-docs", type=int, default=20000)
    parser.add_argument("--host-dir", type=Path, default=Path("~/.cache/pretrain_gym"))
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        parser.error("run with PYTHONHASHSEED=0, as scoring does")

    started = time.time()
    config = ScoringConfig()
    host_dir = args.host_dir.expanduser()
    eval_set = config.eval_set
    splits = (eval_set.val_split, eval_set.test_split)
    parts = eval_parts(config)
    files = eval_files(config, host_dir)
    screened = {"probe": [eval_set.val_split], "final": list(splits)}

    with Pool(args.workers) as pool:
        index_paths = {}
        for path, path_splits in screened.items():
            spec = dict(
                screen=[str(f) for part in parts for split in path_splits for f in files[part, split]],
                ngram=config.ngram,
                text_field=eval_set.text_field,
                cache=str(host_dir / "cache"),
            )
            index_paths[path] = screen_index(spec, pool)
    print(f"indexes ready in {time.time() - started:.0f}s", flush=True)

    sample = list(heldout_sample(files, parts, splits, args.per_set, args.seed))
    honest = fineweb(args.fineweb_docs)
    copies = {f"wikitext-103 {split}": wikitext_articles(split) for split in ("validation", "test")}
    # The same test articles with rare words replaced by <unk>, which breaks some 13-word runs.
    copies["wikitext-2 test, with <unk>"] = wikitext_articles("test", "wikitext-2-v1")
    copies["c4 validation shard 0"] = c4_validation()
    paloma_c4 = {
        text.strip() for split in splits for path in files["c4_en", split] for text in documents(path, "text", "")
    }
    print(f"inputs ready in {time.time() - started:.0f}s", flush=True)

    with Pool(args.workers, _init, (index_paths, config.ngram)) as pool:
        attacked = pool.map(_attack, enumerate(text for _, _, text in sample), chunksize=8)
        honest_stats = summarize_clean(pool.imap(_clean, ((p, t) for p in screened for t in honest), chunksize=64))
        copy_stats = {
            name: summarize_clean(pool.imap(_clean, ((p, t) for p in screened for t in texts), chunksize=16))
            for name, texts in copies.items()
        }
        exact = [text for text in copies["c4 validation shard 0"] if text.strip() in paloma_c4]
        exact_stats = summarize_clean(pool.imap(_clean, (("final", t) for t in exact), chunksize=16))

    attacks = defaultdict(lambda: [0, 0, 0, 0])
    for (_, split, _), result in zip(sample, attacked):
        for (name, path), values in result.items():
            entry = attacks[name, split, path]
            for i, value in enumerate(values):
                entry[i] += value
    attack_report = {
        name: {
            f"{split} documents, {path} filter": dict(
                characters_kept=chars_kept / chars, original_8grams_kept=grams_kept / grams
            )
            for (attack_name, split, path), (chars, chars_kept, grams_kept, grams) in attacks.items()
            if attack_name == name
        }
        for name in ATTACKS
    }
    sha = {repo: HfApi().dataset_info(repo).sha for repo, _ in (FINEWEB, WIKITEXT, C4)}
    report = dict(
        decontamination=dict(ngram=config.ngram),
        heldout_sample=dict(documents=len(sample), per_subset_and_split=args.per_set, min_words=MIN_WORDS),
        attacks=attack_report,
        fineweb=dict(documents=len(honest), **honest_stats),
        public_copies=copy_stats,
        c4_shard_exact_paloma_documents=exact_stats.get("final", dict(documents=0)),
        revisions=dict(paloma=eval_set.revision, **sha),
        seconds=round(time.time() - started),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(
        f"\n{'attack':30s} {'val, probe':>16s} {'val, final':>16s} {'test, final':>16s}   (chars kept / 8-grams kept)"
    )
    for name, cells in attack_report.items():
        row = [
            cells[f"{split} documents, {path} filter"]
            for split, path in (("val", "probe"), ("val", "final"), ("test", "final"))
        ]
        print(f"{name:30s} " + " ".join(f"{c['characters_kept']:7.2%} /{c['original_8grams_kept']:7.2%}" for c in row))
    print(f"\nFineWeb ({len(honest)} docs), final filter: {honest_stats['final']}")
    for name, stats in copy_stats.items():
        print(f"{name}: {stats}")
    print(f"C4 shard documents that are Paloma c4_en documents: {report['c4_shard_exact_paloma_documents']}")


if __name__ == "__main__":
    main()
