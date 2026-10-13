from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import py_compile
import shutil
import signal
import sys
import traceback

import numpy as np


class FileTimeout:
    def __init__(self, seconds: int):
        self.seconds = int(seconds)
        self.old_handler = None

    def __enter__(self):
        if self.seconds > 0:
            self.old_handler = signal.signal(signal.SIGALRM, self._handle)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.seconds > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.old_handler)
        return False

    @staticmethod
    def _handle(_signum, _frame):
        raise TimeoutError("file validation timed out")


def _sort_key(path: Path) -> tuple[int, int, str]:
    folder = path.parent.parent.name
    stem = path.stem
    return (int(folder) if folder.isdigit() else 10**9, int(stem) if stem.isdigit() else 10**9, str(path))


def _target_from_prompt(prompt: Path) -> Path:
    return prompt.parent.parent / f"{prompt.stem}.py"


def _load_module(path: Path):
    module_name = f"arc_validate_227494705_{path.parent.name}_{path.stem}_{abs(hash(str(path))) & 0xFFFFFFFF:x}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop(module_name, None)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def validate_file(path: Path, samples: int, file_timeout: int) -> dict:
    record = {"path": str(path), "status": "OK"}
    try:
        with FileTimeout(file_timeout):
            py_compile.compile(str(path), doraise=True)
            module = _load_module(path)
            if not callable(getattr(module, "generate_input", None)):
                raise AttributeError("missing callable generate_input")
            if not callable(getattr(module, "main", None)):
                raise AttributeError("missing callable main")

            unique_inputs: set[tuple[tuple[int, ...], bytes]] = set()
            for _ in range(samples):
                inp = module.generate_input()
                if not isinstance(inp, np.ndarray):
                    raise TypeError("generate_input did not return numpy.ndarray")
                if inp.ndim != 2:
                    raise ValueError(f"generate_input returned ndim={inp.ndim}")
                if max(inp.shape) > 30:
                    raise ValueError(f"input shape too large: {inp.shape}")
                unique_inputs.add((tuple(int(x) for x in inp.shape), inp.tobytes()))

                out = module.main(inp.copy())
                if not isinstance(out, np.ndarray):
                    raise TypeError("main did not return numpy.ndarray")
                if out.ndim != 2:
                    raise ValueError(f"main returned ndim={out.ndim}")
                if max(out.shape) > 30:
                    raise ValueError(f"output shape too large: {out.shape}")

            record["unique_inputs"] = len(unique_inputs)
            if len(unique_inputs) < 20:
                raise ValueError(f"only {len(unique_inputs)} unique inputs over {samples} samples")
    except Exception as exc:  # noqa: BLE001
        record["status"] = "FAIL"
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc(limit=8)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--file-timeout", type=int, default=60)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    prompts = sorted(args.root.glob("*/codegen_prompts/*.md"), key=_sort_key)
    ok = failed = missing = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for prompt in prompts:
            target = _target_from_prompt(prompt)
            if not target.exists():
                record = {"path": str(target), "prompt": str(prompt), "status": "MISSING"}
                missing += 1
            else:
                record = validate_file(target, args.samples, args.file_timeout)
                record["prompt"] = str(prompt)
                if record["status"] == "OK":
                    ok += 1
                else:
                    failed += 1
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()

    for cache in args.root.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)

    summary = {
        "total_prompts": len(prompts),
        "ok": ok,
        "missing": missing,
        "failed": failed,
        "out": str(args.out),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if missing == 0 and failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
