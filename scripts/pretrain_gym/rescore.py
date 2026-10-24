"""Score a corpus outside an episode: final scoring, evaluated on both halves of the held-out set.

Training is final scoring's, on this host's GPUs: the corpus is read in sorted order up to
`--train-tokens`, after every passage it shares with the validation or test split is cut. `--model`
and `--seed` swap the model or its initialization, for rank-stability and noise-floor runs. Each
`--extra NAME=GLOB` adds an evaluation set from local `.jsonl`, `.jsonl.gz`, `.jsonl.zst`, or
`.parquet` files with a `text` field; it is cut from the training text too. `--eval-set` replaces
Paloma; Paloma is gated, so a first download needs `HF_TOKEN` with access to it. `--eval-subsets` and
`--eval-splits` evaluate fewer sets, for speed; training text is still screened against every subset
and both splits, and the reported losses are means over the sets evaluated.

A final corpus survives its episode with `scoring.keep_final_corpus`, at
`<host_dir>/rollouts/<rollout>/final/corpus`. Needs `pretrain-gym[train]` installed.

usage: uv run python scripts/pretrain_gym/rescore.py <corpus> --out <dir> [--seed 1] [--extra recent=<glob>]
Writes `<out>/rescore.json`; `--export-weights` also keeps `<out>/out/model.pt`.
"""

import argparse
import asyncio
import glob
import json
import statistics
import tomllib
from pathlib import Path

from pretrain_gym.config import EvalSetConfig, ScorerConfig, ScoringConfig
from pretrain_gym.scoring import Job, eval_parts


class RescoreJob(Job):
    """A final-scoring job on a corpus directory, evaluated on `val/<part>`, `test/<part>`, and `extra/<name>`."""

    def __init__(
        self,
        config: ScoringConfig,
        host_dir: Path,
        directory: Path,
        corpus: Path,
        extra: dict[str, list[Path]],
        subsets: list[str],
        splits: list[str],
    ) -> None:
        super().__init__(config, host_dir, directory, "final", config.train_tokens)
        self.corpus, self.extra, self.subsets, self.splits = corpus, extra, subsets, splits

    def _spec(self, files: dict[tuple[str, str], list[Path]], root: Path) -> dict:
        spec = super()._spec(files, root)
        spec["corpus"] = str(self.corpus)
        spec["eval"] = {
            f"{split}/{part}": [str(path) for path in files[part, split]]
            for split in self.splits
            for part in self.subsets
        }
        spec["eval"] |= {f"extra/{name}": [str(path) for path in paths] for name, paths in self.extra.items()}
        spec["screen"] += [str(path) for paths in self.extra.values() for path in paths]
        return spec


def extra_sets(items: list[str], parser: argparse.ArgumentParser) -> dict[str, list[Path]]:
    sets = {}
    for item in items:
        name, _, pattern = item.partition("=")
        if not name or "/" in name or not pattern:
            parser.error(f"--extra takes NAME=GLOB with a NAME free of '/', not {item!r}")
        paths = sorted(Path(path) for path in glob.glob(str(Path(pattern).expanduser()), recursive=True))
        if not paths:
            parser.error(f"--extra {name}: no files match {pattern!r}")
        sets[name] = paths
    return sets


def main() -> None:
    defaults = ScoringConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="Job directory, which must not exist yet.")
    parser.add_argument("--model", default=defaults.model, help="A bundled model name, or an absolute path to one.")
    parser.add_argument("--train-tokens", type=int, default=defaults.train_tokens)
    parser.add_argument("--seed", type=int, help="Replaces the model's train.seed.")
    parser.add_argument("--extra", action="append", default=[], metavar="NAME=GLOB")
    parser.add_argument("--eval-set", type=Path, help="TOML of `scoring.eval_set` options (default: Paloma).")
    parser.add_argument("--host-dir", type=Path, default=Path("~/.cache/pretrain_gym"))
    parser.add_argument("--gpus", type=int, nargs="+", help="GPUs to lease from (default: every GPU).")
    parser.add_argument("--workers", type=int, help="Corpus preparation processes (default: every CPU).")
    parser.add_argument("--export-weights", action="store_true")
    parser.add_argument("--eval-subsets", nargs="+", help="Subsets to evaluate (default: every one).")
    parser.add_argument("--eval-splits", nargs="+", help="Splits to evaluate (default: validation and test).")
    args = parser.parse_args()
    if not args.corpus.is_dir():
        parser.error(f"{args.corpus} is not a directory")
    extra = extra_sets(args.extra, parser)

    scoring = ScoringConfig(
        model=args.model,
        train_tokens=args.train_tokens,
        seed=args.seed,
        export_weights=args.export_weights,
        eval_set=EvalSetConfig(**tomllib.loads(args.eval_set.read_text())) if args.eval_set else EvalSetConfig(),
        scorer=ScorerConfig(gpus=args.gpus, workers=args.workers),
    )
    eval_set = scoring.eval_set
    splits = args.eval_splits or [eval_set.val_split, eval_set.test_split]
    subsets = args.eval_subsets or eval_parts(scoring)
    unknown = (set(splits) - {eval_set.val_split, eval_set.test_split}) | (set(subsets) - set(eval_parts(scoring)))
    if unknown:
        parser.error(f"unknown subsets or splits: {sorted(unknown)}")
    args.out.mkdir(parents=True)
    job = RescoreJob(scoring, args.host_dir.expanduser(), args.out, args.corpus.resolve(), extra, subsets, splits)
    try:
        result = asyncio.run(job.run())
    finally:
        # The corpus is read in place, so only the token files are deleted.
        job.cleanup(keep_corpus=True)

    def by_prefix(values: dict[str, float], prefix: str) -> dict[str, float]:
        return {name.removeprefix(prefix): value for name, value in values.items() if name.startswith(prefix)}

    def mean(values: dict[str, float], split: str) -> float | None:
        return statistics.fmean(by_prefix(values, f"{split}/").values()) if split in splits else None

    test_loss = mean(result.losses, eval_set.test_split)
    summary = dict(
        corpus=str(args.corpus.resolve()),
        model=scoring.model,
        seed=json.loads((args.out / "code" / "model.json").read_text())["train"]["seed"],
        decontamination=dict(ngram=scoring.ngram),
        evaluated=dict(subsets=subsets, splits=splits),
        train_tokens=result.train["train_tokens"],
        diverged=result.train["diverged"],
        val_loss=mean(result.losses, eval_set.val_split),
        test_loss=test_loss,
        reward=None if test_loss is None else scoring.performance(test_loss, result.vocab_size),
        val_bits_per_byte=mean(result.bits_per_byte, eval_set.val_split),
        test_bits_per_byte=mean(result.bits_per_byte, eval_set.test_split),
        extra_losses=by_prefix(result.losses, "extra/"),
        losses=result.losses,
        bits_per_byte=result.bits_per_byte,
        train=result.train,
        prepare=result.prepare,
    )
    (args.out / "rescore.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary[key] for key in ("val_loss", "test_loss", "reward", "extra_losses")}))


if __name__ == "__main__":
    main()
