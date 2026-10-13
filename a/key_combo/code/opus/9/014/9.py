from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

MAX_GENERATION_ATTEMPTS = 500


def _touches_border(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _expand_seed_lines(a: int, b: int, size: int) -> list[int]:
    a, b = sorted((int(a), int(b)))
    d = b - a
    assert d > 0, "seed lines must be distinct"
    period = 2 * d
    lines: set[int] = set()
    for start in (a, b):
        x = start
        while x < size:
            lines.add(int(x))
            x += period
        x = start - period
        while x >= 0:
            lines.add(int(x))
            x -= period
    return sorted(lines)


def _pair_mask(points: list[tuple[int, int]], shape: tuple[int, int]) -> tuple[np.ndarray, str, list[int]]:
    h, w = shape
    assert len(points) == 2
    horizontal = any(_touches_border(pt, shape) for pt in points)
    mask = np.zeros(shape, dtype=bool)

    if horizontal:
        rows = _expand_seed_lines(points[0][0], points[1][0], h)
        mask[rows, :] = True
        return mask, "horizontal", rows

    cols = _expand_seed_lines(points[0][1], points[1][1], w)
    mask[:, cols] = True
    return mask, "vertical", cols


def _infer_pairs(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    reds = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]
    blues = [tuple(map(int, rc)) for rc in np.argwhere(grid == BLUE)]
    assert len(reds) == 2 and len(blues) == 2, "expected exactly two red and two blue seeds"
    assert np.count_nonzero((grid != BLACK) & (grid != RED) & (grid != BLUE)) == 0
    return reds, blues


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    reds, blues = _infer_pairs(g)
    red_mask, red_dir, _red_lines = _pair_mask(reds, g.shape)
    blue_mask, blue_dir, _blue_lines = _pair_mask(blues, g.shape)
    assert red_dir != blue_dir, "the two seed pairs must define different stripe orientations"

    out = np.zeros_like(g)
    out[red_mask & blue_mask] = GREEN
    for r, c in reds + blues:
        out[r, c] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand two seed-defined stripe families and keep only their intersections.

    Input structure:
    - A black grid contains exactly four foreground pixels: two red seeds and two blue seeds.
    - Each same-color pair defines one stripe family.
    - One pair has at least one border-touching seed, while the other pair is fully interior.

    Rule:
    - For each color pair, if either seed touches the grid border, treat the pair as horizontal and use the
      two seed rows as the base rows; otherwise treat it as vertical and use the two seed columns as the
      base columns.
    - Let d be the absolute gap between the two base rows or columns, and repeat full rows or columns from
      each base line every 2d cells until the grid ends.
    - Output a same-sized black grid whose only foreground cells are the intersections of the red and blue
      stripe families, colored green; clear the original seed cells even if they lie on an intersection.

    Key insight:
    - The visible seeds are only phase markers for two hidden periodic stripe masks; the answer is the
      boolean intersection of those fully expanded masks.
    """
    return _solve(grid)


def _progression_options(size: int, *, interior_pair: bool) -> list[tuple[list[int], tuple[int, int], int]]:
    options: list[tuple[list[int], tuple[int, int], int]] = []
    max_d = min(6, size - 2)
    for d in range(2, max_d + 1):
        for offset in range(d):
            lines = list(range(offset, size, d))
            if len(lines) < 3:
                continue
            for i in range(len(lines) - 1):
                pair = (lines[i], lines[i + 1])
                if interior_pair and not all(0 < x < size - 1 for x in pair):
                    continue
                options.append((lines, pair, d))
    return options


def _is_well_separated(points: list[tuple[int, int]]) -> bool:
    for i, (r1, c1) in enumerate(points):
        for r2, c2 in points[i + 1 :]:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _build_expected(shape: tuple[int, int], row_lines: list[int], col_lines: list[int], seeds: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[np.ix_(row_lines, col_lines)] = GREEN
    for r, c in seeds:
        out[r, c] = BLACK
    return out


def _candidate_ok(grid: np.ndarray, row_lines: list[int], col_lines: list[int], h_pair: list[tuple[int, int]], v_pair: list[tuple[int, int]]) -> bool:
    reds, blues = _infer_pairs(grid)
    red_mask, red_dir, red_lines = _pair_mask(reds, grid.shape)
    blue_mask, blue_dir, blue_lines = _pair_mask(blues, grid.shape)

    if {red_dir, blue_dir} != {"horizontal", "vertical"}:
        return False

    inferred_rows = red_lines if red_dir == "horizontal" else blue_lines
    inferred_cols = blue_lines if blue_dir == "vertical" else red_lines
    if inferred_rows != row_lines or inferred_cols != col_lines:
        return False

    if len(row_lines) < 3 or len(col_lines) < 3:
        return False

    seed_rows = {r for r, _ in h_pair}
    seed_cols = {c for _, c in v_pair}
    if not any(r not in seed_rows for r in row_lines):
        return False
    if not any(c not in seed_cols for c in col_lines):
        return False

    expected = _build_expected(grid.shape, row_lines, col_lines, reds + blues)
    actual = _solve(grid)
    if not np.array_equal(actual, expected):
        return False

    green = np.argwhere(actual == GREEN)
    if len(green) < 6 or len(green) >= (grid.shape[0] * grid.shape[1]) // 2:
        return False
    if len({int(r) for r, _ in green}) < 3 or len({int(c) for _, c in green}) < 3:
        return False

    intersection = red_mask & blue_mask
    if np.array_equal(actual == GREEN, intersection):
        return False

    return True


def _pick_with_overlap(candidates: list[int], preferred: set[int], fallback: set[int]) -> int:
    overlap = [x for x in candidates if x in preferred and x not in fallback]
    if overlap and random.random() < 0.65:
        return random.choice(overlap)
    regular = [x for x in candidates if x not in fallback]
    return random.choice(regular)


def _attempt() -> np.ndarray | None:
    h = random.randint(12, 18)
    w = random.randint(12, 18)

    row_options = _progression_options(h, interior_pair=False)
    col_options = _progression_options(w, interior_pair=True)
    if not row_options or not col_options:
        return None

    row_lines, base_rows, _row_d = random.choice(row_options)
    col_lines, base_cols, _col_d = random.choice(col_options)

    h_rows = [int(base_rows[0]), int(base_rows[1])]
    v_cols = [int(base_cols[0]), int(base_cols[1])]

    interior_rows = list(range(1, h - 1))
    interior_cols = list(range(1, w - 1))

    overlap_mode = random.choices(
        ["none", "horizontal", "vertical", "both"],
        weights=[2, 3, 3, 2],
        k=1,
    )[0]

    border_col = random.choice([0, w - 1])
    second_h_candidates = interior_cols
    second_h_col = _pick_with_overlap(
        second_h_candidates,
        set(col_lines) if overlap_mode in {"horizontal", "both"} else set(),
        set(),
    )
    h_pair = [(h_rows[0], border_col), (h_rows[1], second_h_col)]

    v_row_candidates = interior_rows
    used_rows_for_overlap = set()
    v_row1 = _pick_with_overlap(
        v_row_candidates,
        set(row_lines) if overlap_mode in {"vertical", "both"} else set(),
        used_rows_for_overlap,
    )
    used_rows_for_overlap.add(v_row1)
    v_row2 = _pick_with_overlap(
        v_row_candidates,
        set(row_lines) if overlap_mode in {"vertical", "both"} else set(),
        used_rows_for_overlap,
    )
    v_pair = [(v_row1, v_cols[0]), (v_row2, v_cols[1])]

    all_points = h_pair + v_pair
    if len(set(all_points)) != 4:
        return None
    if not _is_well_separated(all_points):
        return None
    if any(_touches_border(pt, (h, w)) for pt in v_pair):
        return None
    if not any(_touches_border(pt, (h, w)) for pt in h_pair):
        return None

    horizontal_color = random.choice([RED, BLUE])
    vertical_color = BLUE if horizontal_color == RED else RED

    grid = np.zeros((h, w), dtype=int)
    for r, c in h_pair:
        grid[r, c] = horizontal_color
    for r, c in v_pair:
        grid[r, c] = vertical_color

    if not _candidate_ok(grid, row_lines, col_lines, h_pair, v_pair):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stripe-intersection puzzle")
