from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREY = 5
TEAL = 8


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _plus_cells(shape: tuple[int, int], r: int, c: int, radius: int) -> tuple[tuple[int, int], ...]:
    h, w = shape
    if not (radius <= r < h - radius and radius <= c < w - radius):
        return ()
    cells = [(r, cc) for cc in range(c - radius, c + radius + 1)]
    cells.extend((rr, c) for rr in range(r - radius, r + radius + 1) if rr != r)
    return tuple(cells)


def _red_spans_in_non_black_runs(line: np.ndarray) -> list[int]:
    spans: list[int] = []
    i = 0
    n = len(line)
    while i < n:
        if int(line[i]) == BLACK:
            i += 1
            continue

        red_positions: list[int] = []
        j = i
        while j < n and int(line[j]) != BLACK:
            if int(line[j]) == RED:
                red_positions.append(j)
            j += 1

        if len(red_positions) >= 2:
            spans.append(red_positions[-1] - red_positions[0] + 1)
        i = j
    return spans


def _infer_radius(grid: np.ndarray) -> int:
    spans: list[int] = []
    for r in range(grid.shape[0]):
        spans.extend(_red_spans_in_non_black_runs(grid[r, :]))
    for c in range(grid.shape[1]):
        spans.extend(_red_spans_in_non_black_runs(grid[:, c]))

    odd_spans = [span for span in spans if span >= 3 and span % 2 == 1]
    if not odd_spans:
        raise AssertionError("could not infer a shared cross radius")
    return (max(odd_spans) - 1) // 2


def _candidate_for_center(grid: np.ndarray, r: int, c: int, radius: int):
    if int(grid[r, c]) != RED:
        return None

    cells = _plus_cells(grid.shape, r, c, radius)
    if not cells:
        return None
    if any(int(grid[rr, cc]) == BLACK for rr, cc in cells):
        return None

    row_red = sum(int(grid[r, cc]) == RED for cc in range(c - radius, c + radius + 1))
    col_red = sum(int(grid[rr, c]) == RED for rr in range(r - radius, r + radius + 1))
    if row_red < 2 or col_red < 2:
        return None

    red_cells = frozenset((rr, cc) for rr, cc in cells if int(grid[rr, cc]) == RED)
    if len(red_cells) < 3:
        return None

    return {
        "center": (r, c),
        "cells": cells,
        "cell_set": frozenset(cells),
        "red_cells": red_cells,
        "red_count": len(red_cells),
    }


def _find_candidates(grid: np.ndarray, radius: int) -> list[dict]:
    candidates: list[dict] = []
    for r, c in np.argwhere(grid == RED):
        cand = _candidate_for_center(grid, int(r), int(c), radius)
        if cand is not None:
            candidates.append(cand)
    candidates.sort(key=lambda item: (-item["red_count"], item["center"][0], item["center"][1]))
    return candidates


def _select_crosses(candidates: list[dict]) -> list[dict]:
    selected: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    for cand in candidates:
        if cand["cell_set"] & occupied:
            continue
        selected.append(cand)
        occupied.update(cand["cell_set"])
    if not selected:
        raise AssertionError("no valid crosses found")
    return selected


def _complete_crosses(grid: np.ndarray, selected: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for cand in selected:
        for r, c in cand["cells"]:
            if int(out[r, c]) == GREY:
                out[r, c] = TEAL
    return out


def _mirror_by_count(grid: np.ndarray, count: int) -> np.ndarray:
    return np.fliplr(grid) if count % 2 == 1 else np.flipud(grid)


def _analyze(grid: np.ndarray) -> tuple[int, list[dict], np.ndarray]:
    radius = _infer_radius(grid)
    candidates = _find_candidates(grid, radius)
    selected = _select_crosses(candidates)
    completed = _complete_crosses(grid, selected)
    return radius, selected, completed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete partial red equal-arm crosses in the grey base, count them, then mirror the whole grid by that parity.

    Input structure:
    - The grid contains black background, a grey non-black base region, red marker pixels, and no other required input colors.
    - Valid objects are axis-aligned equal-arm crosses embedded entirely inside non-black cells; all valid crosses share one radius.
    - Each valid cross has a red center plus some red arm evidence, while unrelated red distractor pixels may also appear in the grey base.

    Rule:
    - Infer the shared radius from the largest odd red span that lies inside a contiguous non-black horizontal or vertical run.
    - For each red pixel, test whether it can be the center of a full radius-sized cross whose cells are all non-black and whose row and column each contain red evidence.
    - Sort candidates by how many red pixels they explain, greedily keep non-overlapping crosses, recolor only their grey cells to teal, then count the kept crosses.
    - If the count is odd, mirror the completed grid left-to-right; if the count is even, mirror it top-to-bottom.

    Key insight:
    - The puzzle is compositional: local cross completion produces a count, and only that count determines the global mirror axis.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        raise AssertionError("expected a 2D ARC grid no larger than 30x30")
    if not set(map(int, np.unique(g))) <= {BLACK, RED, GREY}:
        raise AssertionError("input must use only black, red, and grey")

    _radius, selected, completed = _analyze(g)
    return _mirror_by_count(completed, len(selected))


# --------------- generator ---------------

GRID_SIDE_RANGE = (15, 21)
RADIUS_CHOICES = (2, 3)
EVEN_COUNTS = (2, 4)
ODD_COUNTS = (3,)
TARGET_FILL_RANGE = (0.42, 0.58)
PLACEMENT_RETRIES = 120
GENERATION_RETRIES = 400
DISTRACTOR_PROB = 0.7
MAX_DISTRACTORS = 3


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _corridor(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r1, c1 = a
    r2, c2 = b
    if random.random() < 0.5:
        mid = [(r1, c) for c in range(min(c1, c2), max(c1, c2) + 1)]
        tail = [(r, c2) for r in range(min(r1, r2), max(r1, r2) + 1)]
    else:
        mid = [(r, c1) for r in range(min(r1, r2), max(r1, r2) + 1)]
        tail = [(r2, c) for c in range(min(c1, c2), max(c1, c2) + 1)]
    return list(dict.fromkeys(mid + tail))


def _place_centers(h: int, w: int, radius: int, count: int) -> list[tuple[int, int]] | None:
    cells_by_center: dict[tuple[int, int], set[tuple[int, int]]] = {}
    candidates = [(r, c) for r in range(radius, h - radius) for c in range(radius, w - radius)]
    random.shuffle(candidates)
    for center in candidates:
        cells_by_center[center] = set(_plus_cells((h, w), center[0], center[1], radius))

    for _ in range(PLACEMENT_RETRIES):
        random.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        occupied: set[tuple[int, int]] = set()
        for center in candidates:
            cells = cells_by_center[center]
            if cells & occupied:
                continue
            if any(abs(center[0] - rr) + abs(center[1] - cc) <= radius for rr, cc in chosen):
                continue
            chosen.append(center)
            occupied.update(cells)
            if len(chosen) == count:
                return chosen
    return None


def _pattern_options(center: tuple[int, int], radius: int, signal_axis: str | None) -> list[dict]:
    r, c = center
    opts: list[dict] = []
    if signal_axis == "row":
        for dr in (-radius, radius):
            red_cells = {(r, cc) for cc in range(c - radius, c + radius + 1)}
            red_cells.add((r + dr, c))
            opts.append(
                {
                    "red_cells": red_cells,
                    "rows": {rr for rr, _ in red_cells},
                    "cols": {cc for _, cc in red_cells},
                }
            )
    elif signal_axis == "col":
        for dc in (-radius, radius):
            red_cells = {(rr, c) for rr in range(r - radius, r + radius + 1)}
            red_cells.add((r, c + dc))
            opts.append(
                {
                    "red_cells": red_cells,
                    "rows": {rr for rr, _ in red_cells},
                    "cols": {cc for _, cc in red_cells},
                }
            )
    else:
        for dr in (-radius, radius):
            for dc in (-radius, radius):
                red_cells = {(r, c), (r + dr, c), (r, c + dc)}
                opts.append(
                    {
                        "red_cells": red_cells,
                        "rows": {r, r + dr},
                        "cols": {c, c + dc},
                    }
                )
    random.shuffle(opts)
    return opts


def _assign_true_reds(
    centers: list[tuple[int, int]],
    radius: int,
) -> tuple[dict[tuple[int, int], set[tuple[int, int]]], int, str] | None:
    signal_idx = random.randrange(len(centers))
    signal_axis = random.choice(("row", "col"))
    ordered_indices = [signal_idx] + [i for i in range(len(centers)) if i != signal_idx]
    options = {
        i: _pattern_options(centers[i], radius, signal_axis if i == signal_idx else None)
        for i in range(len(centers))
    }
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    chosen: dict[tuple[int, int], set[tuple[int, int]]] = {}

    def backtrack(pos: int) -> bool:
        if pos == len(ordered_indices):
            return True
        idx = ordered_indices[pos]
        center = centers[idx]
        for opt in options[idx]:
            if opt["rows"] & used_rows or opt["cols"] & used_cols:
                continue
            used_rows.update(opt["rows"])
            used_cols.update(opt["cols"])
            chosen[center] = set(opt["red_cells"])
            if backtrack(pos + 1):
                return True
            del chosen[center]
            used_rows.difference_update(opt["rows"])
            used_cols.difference_update(opt["cols"])
        return False

    if not backtrack(0):
        return None
    return chosen, signal_idx, signal_axis


def _seed_mask(h: int, w: int, centers: list[tuple[int, int]], radius: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in centers:
        for rr, cc in _plus_cells((h, w), r, c, radius):
            mask[rr, cc] = True
    chain = centers[:]
    random.shuffle(chain)
    for a, b in zip(chain, chain[1:]):
        for rr, cc in _corridor(a, b):
            mask[rr, cc] = True
    return mask


def _grow_mask(seed: np.ndarray, target_area: int) -> np.ndarray:
    mask = seed.copy()
    h, w = mask.shape
    frontier = {
        (nr, nc)
        for r, c in np.argwhere(mask)
        for nr, nc in _neighbors4(int(r), int(c))
        if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]
    }
    while int(mask.sum()) < target_area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        r, c = cell
        mask[r, c] = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _reserved_rows_cols(red_map: dict[tuple[int, int], set[tuple[int, int]]]) -> tuple[set[int], set[int]]:
    rows = {r for cells in red_map.values() for r, _ in cells}
    cols = {c for cells in red_map.values() for _, c in cells}
    return rows, cols


def _place_distractors(
    grid: np.ndarray,
    true_plus_cells: set[tuple[int, int]],
    reserved_rows: set[int],
    reserved_cols: set[int],
) -> set[tuple[int, int]]:
    candidates = [
        (r, c)
        for r, c in np.argwhere(grid == GREY)
        if (int(r), int(c)) not in true_plus_cells
        and int(r) not in reserved_rows
        and int(c) not in reserved_cols
    ]
    random.shuffle(candidates)
    chosen: set[tuple[int, int]] = set()
    want = random.randint(1, MAX_DISTRACTORS) if random.random() < DISTRACTOR_PROB else 0
    used_rows = set(reserved_rows)
    used_cols = set(reserved_cols)
    for r, c in candidates:
        r = int(r)
        c = int(c)
        if len(chosen) == want:
            break
        if r in used_rows or c in used_cols:
            continue
        chosen.add((r, c))
        used_rows.add(r)
        used_cols.add(c)
    for r, c in chosen:
        grid[r, c] = RED
    return chosen


def _expected_from_truth(grid: np.ndarray, centers: list[tuple[int, int]], radius: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r, c in centers:
        for rr, cc in _plus_cells(grid.shape, r, c, radius):
            if int(out[rr, cc]) == GREY:
                out[rr, cc] = TEAL
    return out


def _semantic_ok(
    inp: np.ndarray,
    centers: list[tuple[int, int]],
    radius: int,
    distractors: set[tuple[int, int]],
) -> bool:
    try:
        inferred_radius, selected, completed = _analyze(inp)
    except Exception:
        return False

    selected_centers = [tuple(map(int, cand["center"])) for cand in selected]
    if inferred_radius != radius:
        return False
    if set(selected_centers) != set(centers) or len(selected_centers) != len(centers):
        return False
    if any(tuple(map(int, d)) in set(selected_centers) for d in distractors):
        return False

    expected_completed = _expected_from_truth(inp, centers, radius)
    if not np.array_equal(completed, expected_completed):
        return False
    if any(int(completed[r, c]) != RED for r, c in distractors):
        return False

    expected_output = _mirror_by_count(expected_completed, len(centers))
    try:
        observed_output = main(inp)
    except Exception:
        return False
    return np.array_equal(observed_output, expected_output)


def _gen_attempt() -> np.ndarray | None:
    odd_branch = random.random() < 0.5
    count = random.choice(ODD_COUNTS if odd_branch else EVEN_COUNTS)
    radius = random.choice(RADIUS_CHOICES)
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)

    centers = _place_centers(h, w, radius, count)
    if centers is None:
        return None

    red_assignment = _assign_true_reds(centers, radius)
    if red_assignment is None:
        return None
    red_map, _signal_idx, _signal_axis = red_assignment

    seed = _seed_mask(h, w, centers, radius)
    total = h * w
    target_low = max(int(seed.sum()), math.ceil(TARGET_FILL_RANGE[0] * total))
    target_high = math.floor(TARGET_FILL_RANGE[1] * total)
    if target_low > target_high:
        return None
    target_area = random.randint(target_low, target_high)
    mask = _grow_mask(seed, target_area)
    if int(mask.sum()) < target_low or not _connected(mask):
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    grid[mask] = GREY

    true_plus_cells: set[tuple[int, int]] = set()
    for center in centers:
        true_plus_cells.update(_plus_cells((h, w), center[0], center[1], radius))

    if not all(grid[r, c] == GREY for r, c in true_plus_cells):
        return None

    for cells in red_map.values():
        for r, c in cells:
            if grid[r, c] != GREY:
                return None
            grid[r, c] = RED

    reserved_rows, reserved_cols = _reserved_rows_cols(red_map)
    distractors = _place_distractors(grid, true_plus_cells, reserved_rows, reserved_cols)

    if not _semantic_ok(grid, centers, radius, distractors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
