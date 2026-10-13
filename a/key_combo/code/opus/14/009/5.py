from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

GRID_SIDE_RANGE = (12, 16)
SOURCE_COUNT_RANGE = (2, 4)
DEFLECTOR_COUNT_RANGE = (4, 8)
GENERATION_RETRIES = 500


def _trace_path(
    start_row: int, top_row: np.ndarray, shape: tuple[int, int]
) -> tuple[list[tuple[int, int]], dict]:
    """Trace one flow path and collect lightweight branch-coverage stats."""
    height, width = shape
    r = int(start_row)
    c = 0
    cells: list[tuple[int, int]] = []
    matched_colors: set[int] = set()
    deflections = 0
    mismatches = 0

    while True:
        if r == height - 1 or c == width - 1:
            end = "bottom" if r == height - 1 else "right"
            break

        c += 1
        cells.append((r, c))

        if r == height - 1 or c == width - 1:
            end = "bottom" if r == height - 1 else "right"
            break

        top_color = int(top_row[c])
        if top_color not in (BLUE, RED):
            continue

        active_color = BLUE if (r + c) % 2 == 0 else RED
        if top_color == active_color:
            deflections += 1
            matched_colors.add(active_color)
            r += 1
            cells.append((r, c))
            if r == height - 1:
                end = "bottom"
                break
        else:
            mismatches += 1

    stats = {
        "end": end,
        "deflections": deflections,
        "mismatches": mismatches,
        "matched_colors": matched_colors,
        "path_len": len(cells),
    }
    return cells, stats


def main(grid: np.ndarray) -> np.ndarray:
    """Flow green streams rightward while top-edge deflectors only activate on matching checkerboard parity.

    Input structure:
    - A mostly black grid contains green source pixels on the left edge and blue/red deflectors on the top edge.
    - Blue deflectors matter only on even-parity cells, while red deflectors matter only on odd-parity cells.
    - The rest of the canvas is empty background.

    Rule:
    - Start a green path from every green pixel on the left edge and move it one cell right at a time.
    - After each horizontal step, stop if the path has reached the right or bottom edge.
    - Otherwise, inspect the current cell parity `(row + col) % 2`: on even cells, only a blue deflector in that column pushes the path down one cell; on odd cells, only a red deflector does.
    - Color every traversed path cell green while keeping the original sources and top-edge deflectors unchanged.

    Key insight:
    - Every downward deflection flips the path parity for all later columns, so one earlier response changes which later deflectors are active or ignored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    top_row = g[0, :]

    for start_row in np.flatnonzero(g[:, 0] == GREEN):
        cells, _ = _trace_path(int(start_row), top_row, g.shape)
        for r, c in cells:
            if g[r, c] in (BLUE, RED):
                continue
            out[r, c] = GREEN

    return out


def _sample_deflectors(width: int) -> dict[int, int]:
    count = random.randint(DEFLECTOR_COUNT_RANGE[0], min(DEFLECTOR_COUNT_RANGE[1], width - 2))

    must_have = {random.randint(1, min(3, width - 2))}
    late_lo = max(1, width - 4)
    must_have.add(random.randint(late_lo, width - 2))

    available = [c for c in range(1, width - 1) if c not in must_have]
    need = count - len(must_have)
    if need > 0:
        must_have.update(random.sample(available, need))
    cols = sorted(must_have)

    deflectors: dict[int, int] = {}
    for c in cols:
        preferred = RED if c % 2 == 1 else BLUE
        other = BLUE if preferred == RED else RED
        deflectors[c] = preferred if random.random() < 0.65 else other

    first_col = cols[0]
    deflectors[first_col] = RED if first_col % 2 == 1 else BLUE

    colors_used = set(deflectors.values())
    if len(colors_used) == 1:
        flip_col = random.choice(cols[1:] or cols)
        deflectors[flip_col] = BLUE if deflectors[flip_col] == RED else RED

    return deflectors


def _build_top_row(width: int, deflectors: dict[int, int]) -> np.ndarray:
    top_row = np.zeros(width, dtype=int)
    for c, color in deflectors.items():
        top_row[int(c)] = int(color)
    return top_row


def _row_candidates(height: int, top_row: np.ndarray) -> dict[int, dict]:
    candidates: dict[int, dict] = {}
    for r in range(2, height - 1, 2):
        cells, stats = _trace_path(r, top_row, (height, top_row.size))
        if stats["path_len"] < 6:
            continue
        candidates[r] = {"cells": cells, **stats}
    return candidates


def _pick_sources(candidates: dict[int, dict]) -> list[int] | None:
    rows = sorted(candidates)
    if len(rows) < 2:
        return None

    rights = [r for r in rows if candidates[r]["end"] == "right"]
    bottoms = [r for r in rows if candidates[r]["end"] == "bottom"]
    if not rights or not bottoms:
        return None

    chosen = [rights[0]]
    if bottoms[-1] not in chosen:
        chosen.append(bottoms[-1])

    target = random.randint(max(SOURCE_COUNT_RANGE[0], len(chosen)), min(SOURCE_COUNT_RANGE[1], len(rows)))
    remaining = [r for r in rows if r not in chosen]

    while len(chosen) < target and remaining:
        best = max(
            remaining,
            key=lambda r: (
                min(abs(r - s) for s in chosen),
                candidates[r]["deflections"],
                candidates[r]["mismatches"],
            ),
        )
        chosen.append(best)
        remaining.remove(best)

    return sorted(chosen)


def _branchy_enough(candidates: dict[int, dict], sources: list[int]) -> bool:
    selected = [candidates[r] for r in sources]
    matched_colors = set()
    total_deflections = 0
    total_mismatches = 0
    ends = set()

    for item in selected:
        matched_colors.update(item["matched_colors"])
        total_deflections += int(item["deflections"])
        total_mismatches += int(item["mismatches"])
        ends.add(str(item["end"]))

    return (
        matched_colors == {BLUE, RED}
        and total_deflections >= 3
        and max(int(item["deflections"]) for item in selected) >= 2
        and total_mismatches >= 2
        and ends == {"right", "bottom"}
    )


def _attempt_input() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)

    deflectors = _sample_deflectors(width)
    top_row = _build_top_row(width, deflectors)
    candidates = _row_candidates(height, top_row)
    sources = _pick_sources(candidates)
    if sources is None or not _branchy_enough(candidates, sources):
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[0, :] = top_row
    for r in sources:
        grid[r, 0] = GREEN

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    added_green = int(np.count_nonzero(out == GREEN) - np.count_nonzero(grid == GREEN))
    if added_green < max(10, width // 2):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid parity-deflection flow input")
