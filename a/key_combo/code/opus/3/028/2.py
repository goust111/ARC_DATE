from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


def _ensure_arc_utils_on_path() -> None:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "new" / "solvers"
        if candidate.is_dir():
            path = str(candidate)
            if path not in sys.path:
                sys.path.append(path)
            return
    raise ImportError("could not locate new/solvers for arc_utils import")


_ensure_arc_utils_on_path()

from arc_utils import Color, get_bg, in_bounds  # noqa: E402


GRID_SIDE_RANGE = (13, 18)
SEGMENT_COUNT_CHOICES = (3, 4)
LENGTH_RANGE = (2, 5)
COLOR_POOL = (Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
GEN_RETRIES = 400
SCENARIOS = ("none", "unequal_collision")
SCENARIO_WEIGHTS = (0.45, 0.55)


def _runs(cols: list[int]) -> list[tuple[int, int]]:
    assert cols
    runs: list[tuple[int, int]] = []
    start = prev = int(cols[0])
    for c in cols[1:]:
        c = int(c)
        if c == prev + 1:
            prev = c
            continue
        runs.append((start, prev))
        start = prev = c
    runs.append((start, prev))
    return runs


def _segment_info(shape: tuple[int, int], s: int, color: int, c0: int, c1: int) -> dict:
    length = int(c1 - c0 + 1)
    bottom_row = int(s - c0)
    top_row = int(s - c1)
    cells = [(int(s - c), int(c)) for c in range(c0, c1 + 1)]

    # The moved segment keeps the original bottom row and starts immediately to the
    # right of the old segment's right end. This keeps the slide/growth rule visible
    # while allowing different source diagonals to compete for the same target cells.
    target_full = [(bottom_row - k, c1 + 1 + k) for k in range(length + 1)]
    target = [cell for cell in target_full if in_bounds(shape, cell[0], cell[1])]
    return {
        "s": int(s),
        "color": int(color),
        "c0": int(c0),
        "c1": int(c1),
        "length": int(length),
        "top_row": int(top_row),
        "bottom_row": int(bottom_row),
        "cells": cells,
        "target": target,
        "target_sum": int(bottom_row + c1 + 1),
    }


def _priority_key(seg: dict) -> tuple[int, int, int, int]:
    return (-int(seg["length"]), int(seg["top_row"]), int(seg["c0"]), int(seg["s"]))


def _extract_segments(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    fg_coords = np.argwhere(grid != bg)
    assert fg_coords.size > 0, "expected foreground pixels"

    by_sum: dict[int, list[tuple[int, int]]] = {}
    for r, c in fg_coords:
        by_sum.setdefault(int(r + c), []).append((int(r), int(c)))

    cleaned = np.full_like(grid, bg)
    segments: list[dict] = []
    for s in sorted(by_sum):
        coords = by_sum[s]
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "each occupied anti-diagonal must use exactly one color"
        color = colors.pop()

        cols = sorted(int(c) for _r, c in coords)
        runs = _runs(cols)
        lengths = [c1 - c0 + 1 for c0, c1 in runs]
        max_len = max(lengths)
        keep = [run for run, run_len in zip(runs, lengths) if run_len == max_len]
        assert len(keep) == 1, "expected a unique longest run on each occupied anti-diagonal"
        c0, c1 = keep[0]

        for c in range(c0, c1 + 1):
            cleaned[s - c, c] = color
        segments.append(_segment_info(grid.shape, s, color, c0, c1))

    assert segments, "expected at least one recovered segment"
    return segments, cleaned


def _resolve_movers(segments: list[dict]) -> set[int]:
    claims: dict[tuple[int, int], list[int]] = {}
    for idx, seg in enumerate(segments):
        for cell in seg["target"]:
            claims.setdefault(cell, []).append(idx)

    blocked: set[int] = set()
    for idxs in claims.values():
        if len(idxs) <= 1:
            continue
        winner = min(idxs, key=lambda idx: _priority_key(segments[idx]))
        for idx in idxs:
            if idx != winner:
                blocked.add(idx)
    return {idx for idx in range(len(segments)) if idx not in blocked}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the longest run on each occupied anti-diagonal, then move the surviving segments into their next longer slots.

    Input structure:
    - The background is black.
    - Each foreground color appears on one occupied anti-diagonal as one true anti-diagonal segment plus 1-2 same-color singleton noises on that same anti-diagonal.
    - The true segment is the unique longest consecutive run of occupied columns on its anti-diagonal.

    Rule:
    - For every occupied anti-diagonal, keep only the unique longest consecutive same-color run and erase the shorter same-color runs on that anti-diagonal.
    - For a kept segment of length L occupying columns c0..c1, erase it and draw a length-(L+1) anti-diagonal segment whose leftmost column is c1+1 while keeping the original bottom row fixed.
    - Resolve all target overlaps before drawing: the shorter source segment stays in place, and if two colliding segments have the same length, the one with the smaller top-row index wins.
    - Clip any moved pixels that would leave the grid.

    Key insight:
    - Denoising must happen first because the recovered run length controls both the destination slot and the collision priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    segments, cleaned = _extract_segments(g, bg)
    movers = _resolve_movers(segments)

    out = cleaned.copy()
    for idx, seg in enumerate(segments):
        if idx in movers:
            for r, c in seg["cells"]:
                out[r, c] = bg

    for idx in sorted(movers, key=lambda i: _priority_key(segments[i])):
        seg = segments[idx]
        for r, c in seg["target"]:
            out[r, c] = int(seg["color"])

    return out


def _cells_are_clear(cells_a: list[tuple[int, int]], cells_b: list[tuple[int, int]]) -> bool:
    for ra, ca in cells_a:
        for rb, cb in cells_b:
            if abs(ra - rb) + abs(ca - cb) <= 1:
                return False
    return True


def _candidate_segments(h: int, w: int) -> list[dict]:
    candidates: list[dict] = []
    for length in range(LENGTH_RANGE[0], LENGTH_RANGE[1] + 1):
        for c0 in range(w):
            c1 = c0 + length - 1
            if c1 >= w:
                continue
            for s in range(h + w - 1):
                top_row = s - c1
                bottom_row = s - c0
                if top_row < 0 or bottom_row >= h:
                    continue
                seg = _segment_info((h, w), s, Color.RED, c0, c1)
                if len(seg["target"]) != length + 1:
                    continue
                diag_cols = [c for c in range(w) if 0 <= s - c < h]
                noise_cols = [c for c in diag_cols if c < c0 - 1 or c > c1 + 1]
                if not noise_cols:
                    continue
                seg["noise_cols"] = noise_cols
                candidates.append(seg)
    return candidates


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _target_overlap(a: dict, b: dict) -> bool:
    return bool(set(a["target"]) & set(b["target"]))


def _source_target_clear(a: dict, b: dict) -> bool:
    a_cells = set(a["cells"])
    b_cells = set(b["cells"])
    a_target = set(a["target"])
    b_target = set(b["target"])
    return not (a_target & b_cells or b_target & a_cells)


def _pick_pair(candidates: list[dict], scenario: str) -> tuple[dict, dict] | None:
    if scenario == "none":
        return None

    by_target_sum: dict[int, list[dict]] = {}
    for seg in candidates:
        by_target_sum.setdefault(int(seg["target_sum"]), []).append(seg)

    pairs: list[tuple[dict, dict]] = []
    for group in by_target_sum.values():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a = group[i]
                b = group[j]
                if a["s"] == b["s"]:
                    continue
                if a["length"] == b["length"]:
                    continue
                if not _interval_overlap(a["target"][0][1], a["target"][-1][1], b["target"][0][1], b["target"][-1][1]):
                    continue
                if not _target_overlap(a, b):
                    continue
                if not _source_target_clear(a, b):
                    continue
                if not _cells_are_clear(a["cells"], b["cells"]):
                    continue
                pairs.append((a, b))

    if not pairs:
        return None
    return random.choice(pairs)


def _pick_segments(candidates: list[dict], count: int, scenario: str) -> list[dict] | None:
    chosen: list[dict] = []
    pair = _pick_pair(candidates, scenario)
    if pair is None and scenario != "none":
        return None
    if pair is not None:
        chosen.extend(pair)

    pool = candidates[:]
    random.shuffle(pool)
    for seg in pool:
        if len(chosen) == count:
            break
        if any(seg["s"] == other["s"] for other in chosen):
            continue
        if any(not _cells_are_clear(seg["cells"], other["cells"]) for other in chosen):
            continue
        if any(not _source_target_clear(seg, other) for other in chosen):
            continue

        if any(_target_overlap(seg, other) for other in chosen):
            continue
        chosen.append(seg)

    if len(chosen) != count:
        return None

    overlap_pairs = 0
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if _target_overlap(chosen[i], chosen[j]):
                overlap_pairs += 1
    if scenario == "none" and overlap_pairs != 0:
        return None
    if scenario != "none" and overlap_pairs != 1:
        return None
    return chosen


def _can_place_noise(
    cell: tuple[int, int],
    occupied: set[tuple[int, int]],
    foreign_cells: set[tuple[int, int]],
) -> bool:
    if cell in occupied:
        return False
    r, c = cell
    for rr, cc in occupied | foreign_cells:
        if abs(rr - r) + abs(cc - c) <= 1:
            return False
    return True


def _add_noise(grid: np.ndarray, segments: list[dict]) -> bool:
    occupied = {tuple(cell) for seg in segments for cell in seg["cells"]}
    for seg in segments:
        s = int(seg["s"])
        color = int(seg["color"])
        want = 2 if random.random() < 0.45 else 1
        placed = 0
        picked_cols: list[int] = []
        cols = list(seg["noise_cols"])
        random.shuffle(cols)
        for c in cols:
            if any(abs(c - prev) <= 1 for prev in picked_cols):
                continue
            cell = (s - c, c)
            foreign_cells = occupied - set(seg["cells"])
            if not _can_place_noise(cell, occupied | set((s - pc, pc) for pc in picked_cols), foreign_cells):
                continue
            grid[cell] = color
            picked_cols.append(c)
            occupied.add(cell)
            placed += 1
            if placed == want:
                break
        if placed == 0:
            return False
    return True


def _segment_summary(seg: dict) -> tuple[int, int, int]:
    return int(seg["s"]), int(seg["c0"]), int(seg["length"])


def _validate_generated(grid: np.ndarray, scenario: str, source_segments: list[dict]) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    bg = int(get_bg(grid))
    recovered, cleaned = _extract_segments(grid, bg)
    if sorted(_segment_summary(seg) for seg in recovered) != sorted(_segment_summary(seg) for seg in source_segments):
        return False
    if np.count_nonzero(cleaned != grid) < len(recovered):
        return False

    movers = _resolve_movers(recovered)
    overlap_pairs: list[tuple[int, int]] = []
    for i in range(len(recovered)):
        for j in range(i + 1, len(recovered)):
            if _target_overlap(recovered[i], recovered[j]):
                overlap_pairs.append((i, j))

    if scenario == "none":
        if overlap_pairs or len(movers) != len(recovered):
            return False
    else:
        if len(overlap_pairs) != 1:
            return False
        if len(movers) != len(recovered) - 1:
            return False
        i, j = overlap_pairs[0]
        winner = min((i, j), key=lambda idx: _priority_key(recovered[idx]))
        loser = j if winner == i else i
        if winner not in movers or loser in movers:
            return False
        if recovered[winner]["length"] <= recovered[loser]["length"]:
            return False

    if out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    count = random.choice(SEGMENT_COUNT_CHOICES)
    scenario = random.choices(SCENARIOS, weights=SCENARIO_WEIGHTS, k=1)[0]

    candidates = _candidate_segments(h, w)
    if len(candidates) < count:
        return None

    for _ in range(80):
        chosen = _pick_segments(candidates, count, scenario)
        if chosen is None:
            continue

        colors = random.sample(COLOR_POOL, count)
        segments = []
        grid = np.full((h, w), Color.BLACK, dtype=int)
        for base, color in zip(chosen, colors):
            seg = _segment_info((h, w), int(base["s"]), int(color), int(base["c0"]), int(base["c1"]))
            seg["noise_cols"] = list(base["noise_cols"])
            segments.append(seg)
            for r, c in seg["cells"]:
                grid[r, c] = color

        if not _add_noise(grid, segments):
            continue
        if not _validate_generated(grid, scenario, segments):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
