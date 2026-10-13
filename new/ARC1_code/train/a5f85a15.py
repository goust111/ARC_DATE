from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every second pixel in each down-right foreground diagonal run to yellow.

    Input structure:
    - The grid has a dominant background and exactly one non-background foreground color.
    - The foreground pixels form one-cell-wide runs using down-right diagonal adjacency.

    Rule:
    - For each maximal down-right diagonal run, keep the first pixel in its original color.
    - Walking down-right, recolor the odd-indexed pixels of that run to yellow.
    - Leave the background and even-indexed foreground pixels unchanged.

    Key insight:
    - Alternation restarts at each top/left endpoint of a diagonal run; rows with
      multiple foreground pixels belong to different runs, not one global sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = int(get_bg(g))
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg_color = int(fg_colors[0])
    assert fg_color != int(Color.YELLOW), "foreground is recolored to yellow"

    foreground = g == fg_color
    starts: list[tuple[int, int]] = []
    for r, c in np.argwhere(foreground):
        r = int(r)
        c = int(c)
        if r == 0 or c == 0 or not foreground[r - 1, c - 1]:
            starts.append((r, c))
        assert not (
            (r > 0 and foreground[r - 1, c])
            or (r + 1 < height and foreground[r + 1, c])
            or (c > 0 and foreground[r, c - 1])
            or (c + 1 < width and foreground[r, c + 1])
        ), "foreground must be one-cell-wide diagonal runs"
    assert starts, "expected at least one diagonal run"

    out = g.copy()
    seen = np.zeros_like(foreground, dtype=bool)
    for start_r, start_c in starts:
        index = 0
        r, c = start_r, start_c
        while r < height and c < width and foreground[r, c]:
            assert not seen[r, c], "diagonal runs must not overlap"
            seen[r, c] = True
            if index % 2 == 1:
                out[r, c] = int(Color.YELLOW)
            index += 1
            r += 1
            c += 1

    assert np.array_equal(seen, foreground), "all foreground pixels must lie in diagonal runs"
    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
_GEN_SIZE_MIN = 3
_GEN_SIZE_MAX = 12
_GEN_BACKGROUND_COLOR = int(Color.BLACK)
_GEN_OUTPUT_COLOR = int(Color.YELLOW)
_GEN_FG_COLORS = np.array([c for c in Color.NOT_BLACK if c != _GEN_OUTPUT_COLOR], dtype=int)

# Diagonal segment and non-wrapping run constraints.
_GEN_DIAG_COUNT_OPTIONS = (1, 2)
_GEN_MIN_COVERAGE_DENOMINATOR = 2
_GEN_MIN_RUN_COUNT = 1
_GEN_MAX_RUN_COUNT = 3
_GEN_MIN_RUN_LEN = 3
_GEN_MAX_RUN_LEN = 11

# Retry limit for rare rejected structural samples.
_GEN_MAX_ATTEMPTS = 300

_GEN_CONFIG_CACHE: dict[int, list[dict]] = {}


def _gen_diag_cells(n: int, diag: int, start: int, length: int) -> tuple[tuple[int, int], ...]:
    return tuple((int((start + i) % n), int((diag + start + i) % n)) for i in range(length))


def _gen_run_cells(n: int, cells: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    mask = np.zeros((n, n), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    runs: list[tuple[tuple[int, int], ...]] = []
    for r, c in sorted(cells):
        if r == 0 or c == 0 or not mask[r - 1, c - 1]:
            run: list[tuple[int, int]] = []
            rr, cc = r, c
            while rr < n and cc < n and mask[rr, cc]:
                run.append((rr, cc))
                rr += 1
                cc += 1
            runs.append(tuple(run))
    return runs


def _gen_has_orthogonal_contact(cells: tuple[tuple[int, int], ...]) -> bool:
    occupied = set(cells)
    for r, c in occupied:
        if (r + 1, c) in occupied or (r, c + 1) in occupied:
            return True
    return False


def _gen_segment_options(n: int, diag: int) -> list[dict]:
    options: list[dict] = []
    min_len = (n + _GEN_MIN_COVERAGE_DENOMINATOR - 1) // _GEN_MIN_COVERAGE_DENOMINATOR
    for length in range(min_len, n + 1):
        for start in range(n):
            cells = _gen_diag_cells(n, diag, start, length)
            runs = _gen_run_cells(n, cells)
            run_lengths = tuple(len(run) for run in runs)
            if run_lengths and all(_GEN_MIN_RUN_LEN <= run_len <= _GEN_MAX_RUN_LEN for run_len in run_lengths):
                options.append({
                    "diag": diag, "start": start, "length": length,
                    "cells": cells, "runs": runs, "run_count": len(runs),
                })
    return options


def _gen_config_from_segments(segments: tuple[dict, ...]) -> dict | None:
    """Build one structural config, rejecting overlaps, contacts, and run-count mismatches."""
    run_count = sum(int(segment["run_count"]) for segment in segments)
    if not _GEN_MIN_RUN_COUNT <= run_count <= _GEN_MAX_RUN_COUNT:
        return None

    cells = tuple(cell for segment in segments for cell in segment["cells"])
    if (len(segments) > 1 and len(set(cells)) != len(cells)) or _gen_has_orthogonal_contact(cells):
        return None

    runs: list[tuple[tuple[int, int], ...]] = []
    for segment in segments:
        runs.extend(segment["runs"])
    return {"diag_count": len(segments), "run_count": run_count, "segments": segments, "cells": cells, "runs": runs}


def _gen_configs_for_size(n: int) -> list[dict]:
    cached = _GEN_CONFIG_CACHE.get(n)
    if cached is not None:
        return cached

    by_diag = [_gen_segment_options(n, diag) for diag in range(n)]
    configs: list[dict] = []

    if 1 in _GEN_DIAG_COUNT_OPTIONS:
        for diag_options in by_diag:
            for opt in diag_options:
                cfg = _gen_config_from_segments((opt,))
                if cfg is not None:
                    configs.append(cfg)

    if 2 in _GEN_DIAG_COUNT_OPTIONS:
        for diag_a in range(n):
            for diag_b in range(diag_a + 1, n):
                for opt_a in by_diag[diag_a]:
                    for opt_b in by_diag[diag_b]:
                        cfg = _gen_config_from_segments((opt_a, opt_b))
                        if cfg is not None:
                            configs.append(cfg)

    _GEN_CONFIG_CACHE[n] = configs
    return configs


def _gen_expected_output(grid: np.ndarray, runs: tuple[tuple[tuple[int, int], ...], ...]) -> np.ndarray:
    expected = grid.copy()
    for run in runs:
        for idx, (r, c) in enumerate(run):
            if idx % 2 == 1:
                expected[r, c] = _GEN_OUTPUT_COLOR
    return expected


def _gen_choose_config(configs: list[dict]) -> dict:
    """Sample in size -> diagonal-count -> run-count -> candidate order."""
    diag_count = int(np.random.choice(sorted({int(cfg["diag_count"]) for cfg in configs})))
    count_configs = [cfg for cfg in configs if cfg["diag_count"] == diag_count]

    run_count = int(np.random.choice(sorted({int(cfg["run_count"]) for cfg in count_configs})))
    candidates = [cfg for cfg in count_configs if cfg["run_count"] == run_count]
    return candidates[int(np.random.randint(len(candidates)))]


def _gen_matches_solver(grid: np.ndarray, runs: list[tuple[tuple[int, int], ...]]) -> bool:
    expected = _gen_expected_output(grid, tuple(runs))
    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.randint(_GEN_SIZE_MIN, _GEN_SIZE_MAX + 1))
    configs = _gen_configs_for_size(n)
    if not configs:
        return None

    cfg = _gen_choose_config(configs)
    fg = int(np.random.choice(_GEN_FG_COLORS))
    grid = np.full((n, n), _GEN_BACKGROUND_COLOR, dtype=int)
    for r, c in cfg["cells"]:
        grid[r, c] = fg

    if not _gen_matches_solver(grid, cfg["runs"]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a5f85a15 input")
