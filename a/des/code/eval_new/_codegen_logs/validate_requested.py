#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import py_compile
import shutil
import signal
import sys
import traceback
from pathlib import Path

import numpy as np


SEEDS = ("558458950", "1967842053", "1572225882", "202524619")


class FileTimeout:
    def __init__(self, seconds: int):
        self.seconds = seconds
        self._old_handler = None

    def _handle_timeout(self, _signum, _frame):
        raise TimeoutError(f"validation exceeded {self.seconds}s")

    def __enter__(self):
        if self.seconds > 0:
            self._old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, self._handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, _exc_type, _exc, _tb):
        if self.seconds > 0:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._old_handler)


def iter_targets(root: Path):
    for seed in SEEDS:
        seed_dir = root / f"seed{seed}"
        for prompt in sorted(seed_dir.glob("*/codegen_prompts/*.md")):
            yield prompt, prompt.parent.parent / f"{prompt.stem}.py"


def load_module(path: Path):
    name = "arc_eval_" + hashlib.sha1(str(path).encode()).hexdigest()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_file(path: Path, samples: int):
    py_compile.compile(str(path), doraise=True)
    module = load_module(path)
    if not callable(getattr(module, "generate_input", None)):
        raise AssertionError("missing callable generate_input")
    if not callable(getattr(module, "main", None)):
        raise AssertionError("missing callable main")

    variants = set()
    max_input_shape = (0, 0)
    max_output_shape = (0, 0)
    for _ in range(samples):
        inp = module.generate_input()
        out = module.main(inp)
        if not isinstance(inp, np.ndarray):
            raise AssertionError(f"generate_input returned {type(inp).__name__}, not ndarray")
        if not isinstance(out, np.ndarray):
            raise AssertionError(f"main returned {type(out).__name__}, not ndarray")
        if inp.ndim != 2:
            raise AssertionError(f"input ndim is {inp.ndim}, expected 2")
        if out.ndim != 2:
            raise AssertionError(f"output ndim is {out.ndim}, expected 2")
        if max(inp.shape) > 30:
            raise AssertionError(f"input grid exceeds 30x30: {inp.shape}")
        if max(out.shape) > 30:
            raise AssertionError(f"output grid exceeds 30x30: {out.shape}")
        max_input_shape = max(max_input_shape, tuple(inp.shape))
        max_output_shape = max(max_output_shape, tuple(out.shape))
        variants.add(hashlib.md5(inp.tobytes() + str(inp.shape).encode()).hexdigest())

    if len(variants) < 20:
        raise AssertionError(f"only {len(variants)} input variants, expected at least 20")
    return {
        "variants": len(variants),
        "max_input_shape": max_input_shape,
        "max_output_shape": max_output_shape,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/lixinyu/snap/a/des/code/eval_new")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--file-timeout", type=int, default=30)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    root = Path(args.root)
    results = []
    ok = 0
    missing = 0
    failed = 0

    for prompt, target in iter_targets(root):
        row = {
            "prompt": str(prompt),
            "target": str(target),
            "status": "OK",
            "error": "",
        }
        if not target.exists():
            row["status"] = "MISSING"
            row["error"] = "target file does not exist"
            missing += 1
        else:
            try:
                with FileTimeout(args.file_timeout):
                    row.update(validate_file(target, args.samples))
                ok += 1
            except Exception as exc:  # noqa: BLE001 - report every validation failure.
                row["status"] = "FAIL"
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc(limit=4)
                failed += 1
        print(json.dumps(row, ensure_ascii=True), flush=True)
        results.append(row)

    for seed in SEEDS:
        for cache_dir in (root / f"seed{seed}").glob("**/__pycache__"):
            shutil.rmtree(cache_dir, ignore_errors=True)

    summary = {
        "total": len(results),
        "ok": ok,
        "missing": missing,
        "failed": failed,
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=True), flush=True)
    if args.out:
        Path(args.out).write_text(
            "\n".join(json.dumps(row, ensure_ascii=True) for row in results)
            + "\nSUMMARY "
            + json.dumps(summary, ensure_ascii=True)
            + "\n",
            encoding="utf-8",
        )
    return 0 if missing == 0 and failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
