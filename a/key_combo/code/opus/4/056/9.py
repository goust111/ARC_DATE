from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


try:
    from arc_utils import Color, bbox
except ModuleNotFoundError:
    _ARC_UTILS = SourceFileLoader(
        "arc_utils_fallback", "/home/lixinyu/snap/new/solvers/arc_utils.py"
    ).load_module()
    Color = _ARC_UTILS.Color
    bbox = _ARC_UTILS.bbox


GEN_ATTEMPTS = 400
LOCAL_HEIGHT_RANGE = (8, 13)
LOCAL_WIDTH_RANGE = (8, 13)
BROKEN_EXTRA_PROB = 0.55
MISSING_COUNT_OPTIONS = (1, 2)


def _classify_lines_local(blue: np.ndarray) -> tuple[set[int], set[int], set[int], set[int]]:
    row_pairs = blue[:, :-1] & blue[:, 1:]
    col_pairs = blue[:-1, :] & blue[1:, :]
    broken_rows = {int(r) for r in np.flatnonzero(np.any(row_pairs, axis=1))}
    broken_cols = {int(c) for c in np.flatnonzero(np.any(col_pairs, axis=0))}
    missing_rows = {int(r) for r in np.flatnonzero(~np.any(blue, axis=1))}
    missing_cols = {int(c) for c in np.flatnonzero(~np.any(blue, axis=0))}
    return broken_rows, missing_rows, broken_cols, missing_cols


def _line_sets(grid: np.ndarray) -> tuple[tuple[int, int, int, int], set[int], set[int], set[int], set[int]]:
    blue_mask = np.asarray(grid == Color.BLUE, dtype=bool)
    box = bbox(blue_mask)
    assert box is not None, "expected at least one blue pixel"
    r0, c0, r1, c1 = box
    local_blue = blue_mask[r0 : r1 + 1, c0 : c1 + 1]
    broken_rows, missing_rows, broken_cols, missing_cols = _classify_lines_local(local_blue)
    return (int(r0), int(c0), int(r1), int(c1)), broken_rows, missing_rows, broken_cols, missing_cols


def _expected_output(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    broken_rows: set[int],
    missing_rows: set[int],
    broken_cols: set[int],
    missing_cols: set[int],
) -> np.ndarray:
    r0, c0, r1, c1 = box
    out = np.asarray(grid, dtype=int).copy()
    local = out[r0 : r1 + 1, c0 : c1 + 1]
    orig_blue = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1] == Color.BLUE, dtype=bool)
    h, w = local.shape

    broken_row_mask = np.zeros(h, dtype=bool)
    missing_row_mask = np.zeros(h, dtype=bool)
    broken_col_mask = np.zeros(w, dtype=bool)
    missing_col_mask = np.zeros(w, dtype=bool)
    if broken_rows:
        broken_row_mask[list(broken_rows)] = True
    if missing_rows:
        missing_row_mask[list(missing_rows)] = True
    if broken_cols:
        broken_col_mask[list(broken_cols)] = True
    if missing_cols:
        missing_col_mask[list(missing_cols)] = True

    for r in broken_rows:
        row = local[r]
        row[row == Color.BLACK] = Color.RED
    for c in broken_cols:
        col = local[:, c]
        col[col == Color.BLACK] = Color.RED

    for r in missing_rows:
        local[r, :] = Color.GREEN
    for c in missing_cols:
        local[:, c] = Color.GREEN

    yellow = np.zeros((h, w), dtype=bool)
    yellow |= broken_row_mask[:, None] & missing_col_mask[None, :]
    yellow |= missing_row_mask[:, None] & broken_col_mask[None, :]
    local[yellow & ~orig_blue] = Color.YELLOW
    local[orig_blue] = Color.BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repair incomplete blue grid lines inside the blue bbox and color fully missing lines differently.

    Input structure:
    - The grid has a black background and blue pixels only inside one tight blue bounding box.
    - Inside that box, some rows and columns contain blue runs with at least one
      adjacent blue pair but also leave black gaps; these are incomplete line fragments.
    - Other rows and columns inside the same box contain no blue at all; these are fully missing lines.

    Rule:
    - Compute the bounding box of all blue pixels.
    - Within that box, mark each row with a horizontal adjacent blue pair as a broken row line,
      and each column with a vertical adjacent blue pair as a broken column line.
    - Mark each row and column with no blue anywhere in the box as missing.
    - Fill black gaps on broken rows and broken columns red, paint missing rows and missing columns green,
      and recolor every broken/missing row-column crossing yellow.
    - Preserve all original blue pixels and leave everything outside the blue bounding box unchanged.

    Key insight:
    - Black cells inside the same box have two different meanings: gaps on an already evidenced line
      versus cells on a completely absent line, and the adjacency test is what separates those cases.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({Color.BLACK, Color.BLUE}), "input must be black/blue only"

    box, broken_rows, missing_rows, broken_cols, missing_cols = _line_sets(g)
    return _expected_output(g, box, broken_rows, missing_rows, broken_cols, missing_cols)


def _sample_with_gap(length: int, count: int, required: list[int]) -> list[int] | None:
    selected = list(required)
    if len(set(selected)) != len(selected):
        return None
    if any(not (0 <= idx < length) for idx in selected):
        return None
    if any(abs(a - b) < 2 for i, a in enumerate(selected) for b in selected[i + 1 :]):
        return None

    candidates = list(range(length))
    random.shuffle(candidates)
    for idx in candidates:
        if idx in selected:
            continue
        if all(abs(idx - other) >= 2 for other in selected):
            selected.append(idx)
            if len(selected) == count:
                break
    if len(selected) != count:
        return None
    return sorted(selected)


def _runs(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    runs = [[indices[0]]]
    for idx in indices[1:]:
        if idx == runs[-1][-1] + 1:
            runs[-1].append(idx)
        else:
            runs.append([idx])
    return runs


def _choose_segment(indices: list[int]) -> list[int] | None:
    runs = [run for run in _runs(indices) if len(run) >= 2]
    if not runs:
        return None
    run = random.choice(runs)
    seg_len = random.randint(2, min(4, len(run)))
    start = random.randint(0, len(run) - seg_len)
    return run[start : start + seg_len]


def _choose_structure(length: int, axis_name: str) -> tuple[list[int], list[int]] | None:
    for _ in range(80):
        broken_count = 2 + int(random.random() < BROKEN_EXTRA_PROB)
        missing_count = random.choice(MISSING_COUNT_OPTIONS)
        if broken_count + missing_count + 1 > length:
            continue

        broken = _sample_with_gap(length, broken_count, [0, length - 1])
        if broken is None:
            continue

        missing_candidates = [idx for idx in range(1, length - 1) if idx not in broken]
        random.shuffle(missing_candidates)
        missing = []
        for idx in missing_candidates:
            if all(abs(idx - other) >= 2 for other in missing):
                missing.append(idx)
                if len(missing) == missing_count:
                    break
        if len(missing) != missing_count:
            continue

        neutral = [idx for idx in range(length) if idx not in broken and idx not in missing]
        if not neutral:
            continue

        allowed = [idx for idx in range(length) if idx not in missing]
        if _choose_segment(allowed) is None:
            continue
        return broken, sorted(missing)

    raise RuntimeError(f"failed to choose {axis_name} structure")


def _fill_row_segments(blue: np.ndarray, broken_rows: list[int], missing_cols: set[int]) -> None:
    allowed_cols = [c for c in range(blue.shape[1]) if c not in missing_cols]
    for r in broken_rows:
        segment = _choose_segment(allowed_cols)
        assert segment is not None
        blue[r, segment] = True


def _fill_col_segments(blue: np.ndarray, broken_cols: list[int], missing_rows: set[int]) -> None:
    allowed_rows = [r for r in range(blue.shape[0]) if r not in missing_rows]
    for c in broken_cols:
        segment = _choose_segment(allowed_rows)
        assert segment is not None
        blue[segment, c] = True


def _ensure_nonmissing_coverage(
    blue: np.ndarray,
    broken_rows: list[int],
    missing_rows: set[int],
    broken_cols: list[int],
    missing_cols: set[int],
) -> None:
    for r in range(blue.shape[0]):
        if r in missing_rows or blue[r].any():
            continue
        blue[r, random.choice(broken_cols)] = True

    for c in range(blue.shape[1]):
        if c in missing_cols or blue[:, c].any():
            continue
        blue[random.choice(broken_rows), c] = True

    neutral_rows = [r for r in range(blue.shape[0]) if r not in missing_rows and r not in broken_rows]
    neutral_cols = [c for c in range(blue.shape[1]) if c not in missing_cols and c not in broken_cols]

    if neutral_rows:
        r = random.choice(neutral_rows)
        choices = [c for c in broken_cols if not blue[r, c]]
        if choices:
            blue[r, random.choice(choices)] = True
    if neutral_cols:
        c = random.choice(neutral_cols)
        choices = [r for r in broken_rows if not blue[r, c]]
        if choices:
            blue[random.choice(choices), c] = True


def _validate_case(
    grid: np.ndarray,
    local_box: tuple[int, int, int, int],
    intended_broken_rows: set[int],
    intended_missing_rows: set[int],
    intended_broken_cols: set[int],
    intended_missing_cols: set[int],
) -> bool:
    box, broken_rows, missing_rows, broken_cols, missing_cols = _line_sets(grid)
    if box != local_box:
        return False
    if broken_rows != intended_broken_rows or missing_rows != intended_missing_rows:
        return False
    if broken_cols != intended_broken_cols or missing_cols != intended_missing_cols:
        return False

    r0, c0, r1, c1 = box
    local_blue = grid[r0 : r1 + 1, c0 : c1 + 1] == Color.BLUE
    row_nonempty = {int(r) for r in np.flatnonzero(np.any(local_blue, axis=1))}
    col_nonempty = {int(c) for c in np.flatnonzero(np.any(local_blue, axis=0))}
    neutral_rows = row_nonempty - broken_rows
    neutral_cols = col_nonempty - broken_cols
    if not neutral_rows or not neutral_cols:
        return False

    out = main(grid)
    expected = _expected_output(grid, box, broken_rows, missing_rows, broken_cols, missing_cols)
    if not np.array_equal(out, expected):
        return False
    if not np.any(out == Color.RED):
        return False
    if not np.any(out == Color.GREEN):
        return False
    if not np.any(out == Color.YELLOW):
        return False

    outside = np.ones(grid.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    if not np.array_equal(out[outside], grid[outside]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    local_h = random.randint(*LOCAL_HEIGHT_RANGE)
    local_w = random.randint(*LOCAL_WIDTH_RANGE)

    broken_rows, missing_rows = _choose_structure(local_h, "row")
    broken_cols, missing_cols = _choose_structure(local_w, "col")
    broken_row_set = set(broken_rows)
    missing_row_set = set(missing_rows)
    broken_col_set = set(broken_cols)
    missing_col_set = set(missing_cols)

    blue = np.zeros((local_h, local_w), dtype=bool)
    _fill_row_segments(blue, broken_rows, missing_col_set)
    _fill_col_segments(blue, broken_cols, missing_row_set)
    _ensure_nonmissing_coverage(
        blue,
        broken_rows,
        missing_row_set,
        broken_cols,
        missing_col_set,
    )

    if not np.all(blue[sorted(missing_row_set), :] == 0):
        return None
    if not np.all(blue[:, sorted(missing_col_set)] == 0):
        return None

    canvas_h = local_h + random.randint(0, 5)
    canvas_w = local_w + random.randint(0, 5)
    top = random.randint(0, canvas_h - local_h)
    left = random.randint(0, canvas_w - local_w)

    grid = np.zeros((canvas_h, canvas_w), dtype=int)
    grid[top : top + local_h, left : left + local_w][blue] = Color.BLUE
    local_box = (top, left, top + local_h - 1, left + local_w - 1)

    if not _validate_case(
        grid,
        local_box,
        broken_row_set,
        missing_row_set,
        broken_col_set,
        missing_col_set,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
