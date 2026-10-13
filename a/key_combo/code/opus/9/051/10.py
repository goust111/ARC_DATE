from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
TEAL = 8

ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_MIN = 17
GRID_MAX = 19
CONTROL_MARGIN = 4
MAX_ATTEMPTS = 400

MANDATORY_NEAR_SQUARES = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)
OPTIONAL_DISTRACTORS = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for dr, dc in ORTHO_DIRS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "cells": cells,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(cells),
                }
            )
    return comps


def _count_full_red_squares(grid: np.ndarray) -> int:
    count = 0
    for comp in _components_of_color(grid, RED):
        r0, c0, r1, c1 = comp["bbox"]
        if comp["area"] == 4 and r1 - r0 == 1 and c1 - c0 == 1:
            count += 1
    return count


def _baseline_counts(width: int, centers: list[int], extent: int) -> np.ndarray:
    counts = np.zeros(width, dtype=int)
    for c in centers:
        left = max(0, c - extent)
        right = min(width, c + extent + 1)
        counts[left:right] += 1
    return counts


def _render_from_spec(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    n_squares = _count_full_red_squares(out)
    controls = [(int(r), int(c)) for r, c in np.argwhere(out == GREEN)]
    by_row: dict[int, list[int]] = defaultdict(list)

    for r, c in controls:
        out[r - 2, c - 2 : c + 3] = GREY
        out[r - 1, c] = GREY
        for rr in (r - 1, r, r + 1):
            out[rr, c - 2] = RED
            out[rr, c + 2] = RED
        by_row[r + 2].append(c)

    width = out.shape[1]
    for row, centers in by_row.items():
        counts = _baseline_counts(width, centers, n_squares)
        out[row, counts > 0] = RED
        for c in centers:
            for cc in range(c - 2, c + 3):
                if 0 <= cc < width and counts[cc] == 1:
                    out[row, cc] = TEAL

    for r, c in controls:
        out[r, c] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the complete red 2x2 squares and use that count as the scaffold baseline width for each green control pixel.

    Input structure:
    - The grid is a black background containing 1 or 2 green control pixels.
    - Red foreground objects include some complete 2x2 squares plus smaller red distractor components such as single pixels, dominoes, or broken 2x2 fragments.
    - Control pixels have enough surrounding space for the fixed 5-column scaffold and for a baseline that can extend sideways by the counted square total.

    Rule:
    - Identify red 4-connected components and count only those whose bounding box is exactly 2x2 and whose area is exactly 4.
    - Let that count be N. Around each green control pixel, draw the fixed scaffold: a grey 5-cell cap at row r-2, red side posts at columns c-2 and c+2 on rows r-1 through r+1, and a grey connector at (r-1, c).
    - On row r+2, draw a red baseline from c-N to c+N, then recolor the centered 5 cells c-2..c+2 to teal unless that cell belongs to an overlap between two baselines on the same row; overlapping baseline cells stay red. Keep the green control pixels and all original red objects in place.

    Key insight:
    - The red square objects do not directly appear as an encoded answer; instead, their count becomes a global geometric parameter that controls how far every scaffold baseline extends.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.all((0 <= g) & (g <= 9))
    return _render_from_spec(g)


def _expanded_scaffold_forbidden(shape: tuple[int, int], controls: list[tuple[int, int]], extent: int) -> np.ndarray:
    h, w = shape
    forbidden = np.zeros((h, w), dtype=bool)
    for r, c in controls:
        left = max(0, min(c - 2, c - extent) - 1)
        right = min(w - 1, max(c + 2, c + extent) + 1)
        top = max(0, r - 3)
        bottom = min(h - 1, r + 3)
        forbidden[top : bottom + 1, left : right + 1] = True
    return forbidden


def _shape_bbox(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _can_place_shape(
    grid: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    anchor: tuple[int, int],
    forbidden: np.ndarray,
) -> bool:
    h, w = grid.shape
    cells = [(anchor[0] + dr, anchor[1] + dc) for dr, dc in shape]
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if forbidden[r, c] or int(grid[r, c]) != BLACK:
            return False
    cell_set = set(cells)
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in cell_set:
                    continue
                if int(grid[nr, nc]) == RED:
                    return False
    return True


def _place_shape(
    grid: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    forbidden: np.ndarray,
) -> bool:
    h, w = grid.shape
    sh, sw = _shape_bbox(shape)
    anchors = [
        (r, c)
        for r in range(h - sh + 1)
        for c in range(w - sw + 1)
        if _can_place_shape(grid, shape, (r, c), forbidden)
    ]
    if not anchors:
        return False
    r0, c0 = random.choice(anchors)
    for dr, dc in shape:
        grid[r0 + dr, c0 + dc] = RED
    return True


def _pick_controls(height: int, width: int, n_squares: int, control_count: int) -> tuple[list[tuple[int, int]], str] | None:
    rows = list(range(CONTROL_MARGIN, height - CONTROL_MARGIN))
    cols = list(range(max(CONTROL_MARGIN, n_squares), width - max(CONTROL_MARGIN, n_squares)))
    if control_count == 1:
        if not rows or not cols:
            return None
        return [(random.choice(rows), random.choice(cols))], "single"

    branches = ["distinct_rows", "same_row_disjoint"]
    if n_squares >= 3:
        branches.extend(["same_row_overlap", "same_row_overlap", "same_row_overlap"])
    branch = random.choice(branches)

    if branch == "distinct_rows":
        candidates = []
        for r1 in rows:
            for c1 in cols:
                for r2 in rows:
                    for c2 in cols:
                        if (r1, c1) >= (r2, c2):
                            continue
                        if abs(r1 - r2) >= 5 or abs(c1 - c2) >= 5:
                            candidates.append([(r1, c1), (r2, c2)])
        if not candidates:
            return None
        return random.choice(candidates), branch

    if branch == "same_row_disjoint":
        candidates = []
        min_gap = max(5, 2 * n_squares + 1)
        for r in rows:
            valid_cols = [c for c in cols]
            for i, c1 in enumerate(valid_cols):
                for c2 in valid_cols[i + 1 :]:
                    if c2 - c1 >= min_gap:
                        candidates.append([(r, c1), (r, c2)])
        if not candidates:
            return None
        return random.choice(candidates), branch

    candidates = []
    for r in rows:
        valid_cols = [c for c in cols]
        for i, c1 in enumerate(valid_cols):
            for c2 in valid_cols[i + 1 :]:
                gap = c2 - c1
                if 5 <= gap <= 2 * n_squares:
                    candidates.append([(r, c1), (r, c2)])
    if not candidates:
        return None
    return random.choice(candidates), branch


def _semantic_validate(inp: np.ndarray, out: np.ndarray, branch: str) -> bool:
    if inp.ndim != 2 or out.ndim != 2 or inp.shape != out.shape:
        return False

    controls = [(int(r), int(c)) for r, c in np.argwhere(inp == GREEN)]
    if len(controls) not in (1, 2):
        return False

    n_squares = _count_full_red_squares(inp)
    if not (1 <= n_squares <= 4):
        return False

    red_components = _components_of_color(inp, RED)
    full_squares = 0
    near_square_distractors = 0
    for comp in red_components:
        r0, c0, r1, c1 = comp["bbox"]
        if comp["area"] == 4 and r1 - r0 == 1 and c1 - c0 == 1:
            full_squares += 1
        else:
            if comp["area"] == 3 and r1 - r0 == 1 and c1 - c0 == 1:
                near_square_distractors += 1
    if full_squares != n_squares or near_square_distractors < 1:
        return False

    if not np.array_equal(out, _render_from_spec(inp)):
        return False

    for r, c in np.argwhere(inp == RED):
        if int(out[r, c]) != RED:
            return False

    by_row: dict[int, list[int]] = defaultdict(list)
    width = inp.shape[1]
    for r, c in controls:
        if min(r, c, inp.shape[0] - 1 - r, inp.shape[1] - 1 - c) < CONTROL_MARGIN:
            return False
        if tuple(out[r - 2, c - 2 : c + 3]) != (GREY, GREY, GREY, GREY, GREY):
            return False
        if int(out[r - 1, c]) != GREY or int(out[r, c]) != GREEN:
            return False
        for rr in (r - 1, r, r + 1):
            if int(out[rr, c - 2]) != RED or int(out[rr, c + 2]) != RED:
                return False
        by_row[r + 2].append(c)

    for row, centers in by_row.items():
        counts = _baseline_counts(width, centers, n_squares)
        for cc in range(width):
            expected = int(out[row, cc])
            in_teal_zone = any(c - 2 <= cc <= c + 2 for c in centers)
            if counts[cc] == 0:
                continue
            if counts[cc] >= 2:
                if expected != RED:
                    return False
            elif in_teal_zone:
                if expected != TEAL:
                    return False
            elif expected != RED:
                return False

    if branch == "same_row_overlap" and len(controls) == 2:
        row = controls[0][0] + 2
        counts = _baseline_counts(width, [c for _, c in controls], n_squares)
        overlap_cols = np.where(counts >= 2)[0]
        if overlap_cols.size == 0:
            return False
        if any(int(out[row, cc]) != RED for cc in overlap_cols):
            return False

    return True


def _attempt_generate() -> tuple[np.ndarray, str] | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    n_squares = random.randint(1, 4)
    control_count = random.choice((1, 2, 2, 2))

    picked = _pick_controls(height, width, n_squares, control_count)
    if picked is None:
        return None
    controls, branch = picked

    grid = np.zeros((height, width), dtype=int)
    for r, c in controls:
        grid[r, c] = GREEN

    forbidden = _expanded_scaffold_forbidden(grid.shape, controls, n_squares)

    mandatory_shapes = [((0, 0), (0, 1), (1, 0))]
    mandatory_shapes[0] = random.choice(MANDATORY_NEAR_SQUARES)
    extra_distractors = random.randint(1, 4)
    distractor_shapes = mandatory_shapes + [random.choice(OPTIONAL_DISTRACTORS) for _ in range(extra_distractors)]

    object_shapes = [((0, 0), (0, 1), (1, 0), (1, 1))] * n_squares + distractor_shapes
    random.shuffle(object_shapes)

    for shape in object_shapes:
        if not _place_shape(grid, shape, forbidden):
            return None

    out = main(grid)
    if not _semantic_validate(grid, out, branch):
        return None
    if int(np.count_nonzero(out == TEAL)) < 5:
        return None
    return grid, branch


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        candidate = _attempt_generate()
        if candidate is not None:
            return candidate[0]
    raise RuntimeError("failed to generate a valid puzzle input")
