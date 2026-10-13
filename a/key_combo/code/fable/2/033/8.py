from __future__ import annotations

from importlib.machinery import SourceFileLoader
import random

import numpy as np


# Keep the file SourceFileLoader-safe by loading the shared toolbox from its
# absolute path instead of relying on PYTHONPATH.
_ARC_UTILS = SourceFileLoader(
    "_arc_utils_fable_2_033_8",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()
get_bg = _ARC_UTILS.get_bg


GRID_MIN = 13
GRID_MAX = 25
MAX_RUN_LEN = 3
MAX_GENERATION_ATTEMPTS = 500


def _center(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    return h // 2, w // 2


def _shared_colors(row: np.ndarray, col: np.ndarray, bg: int) -> list[int]:
    row_colors = set(map(int, row)) - {int(bg)}
    col_colors = set(map(int, col)) - {int(bg)}
    return sorted(row_colors & col_colors)


def _run_count(axis: np.ndarray, color: int) -> int:
    runs = 0
    inside = False
    for value in map(int, axis):
        if value == int(color):
            if not inside:
                runs += 1
                inside = True
        else:
            inside = False
    return runs


def _compose_from_axes(row: np.ndarray, col: np.ndarray, bg: int) -> np.ndarray:
    h = int(col.shape[0])
    w = int(row.shape[0])
    cr, cc = _center((h, w))

    out = np.full((h, w), int(bg), dtype=int)
    out[cr, :] = row
    out[:, cc] = col

    matches = (col[:, None] == row[None, :]) & (row[None, :] != int(bg))
    matches[cr, :] = False
    matches[:, cc] = False
    if np.any(matches):
        out[matches] = np.broadcast_to(row, (h, w))[matches]
    return out


def _naive_all_non_bg(row: np.ndarray, col: np.ndarray, bg: int) -> np.ndarray:
    h = int(col.shape[0])
    w = int(row.shape[0])
    cr, cc = _center((h, w))
    out = np.full((h, w), int(bg), dtype=int)
    out[cr, :] = row
    out[:, cc] = col

    mask = (row[None, :] != int(bg)) & (col[:, None] != int(bg))
    mask[cr, :] = False
    mask[:, cc] = False
    if np.any(mask):
        out[mask] = np.broadcast_to(row, (h, w))[mask]
    return out


def _naive_bbox_fill(row: np.ndarray, col: np.ndarray, bg: int) -> np.ndarray:
    h = int(col.shape[0])
    w = int(row.shape[0])
    cr, cc = _center((h, w))
    out = np.full((h, w), int(bg), dtype=int)
    out[cr, :] = row
    out[:, cc] = col

    for color in _shared_colors(row, col, bg):
        row_pos = np.flatnonzero(row == int(color))
        col_pos = np.flatnonzero(col == int(color))
        r0, r1 = int(col_pos[0]), int(col_pos[-1])
        c0, c1 = int(row_pos[0]), int(row_pos[-1])
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)

    out[cr, :] = row
    out[:, cc] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden multicolor pattern by matching equal colors on the center row and center column.

    Input structure:
    - The grid has odd height and odd width.
    - Only the center row and the center column may contain non-background cells; together they form a multicolor cross.
    - Background cells may also appear on either axis.

    Rule:
    - Keep the original cross unchanged.
    - For each off-axis cell (r, c), compare the center-row color at column c with the center-column color at row r.
    - If those two axis values are the same non-background color, paint (r, c) with that color; otherwise leave it as background.

    Key insight:
    - The input shows only two one-dimensional generators of the final picture, so the hidden shape is recovered by taking a color-wise outer product of the row and column axes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    assert h % 2 == 1 and w % 2 == 1
    bg = int(get_bg(g))
    cr, cc = _center(g.shape)

    off_axis = np.ones(g.shape, dtype=bool)
    off_axis[cr, :] = False
    off_axis[:, cc] = False
    assert np.all(g[off_axis] == bg), "only the center row/column may be visible"

    row = g[cr, :].copy()
    col = g[:, cc].copy()
    return _compose_from_axes(row, col, bg)


def _odd_between(lo: int, hi: int) -> int:
    options = [n for n in range(int(lo), int(hi) + 1) if n % 2 == 1]
    return int(random.choice(options))


def _sample_specs(line_len: int, shared: list[int], side_only: int | None, split_color: int) -> list[tuple[int, int]]:
    max_run_len = min(MAX_RUN_LEN, max(1, (line_len - 5) // 4))
    specs: list[tuple[int, int]] = []

    for color in shared:
        runs = 1
        if color == split_color:
            runs = 2
            if line_len >= 19 and random.random() < 0.25:
                runs = 3
        elif line_len >= 21 and random.random() < 0.2:
            runs = 2

        for _ in range(runs):
            run_len = random.randint(1, max_run_len)
            if color == split_color and random.random() < 0.5:
                run_len = min(max_run_len, run_len + 1)
            specs.append((int(color), int(run_len)))

    if side_only is not None:
        specs.append((int(side_only), random.randint(1, max(1, max_run_len - 1))))

    return specs


def _place_runs(line_len: int, center_idx: int, specs: list[tuple[int, int]]) -> list[tuple[int, int, int]] | None:
    order = list(specs)
    random.shuffle(order)
    order.sort(key=lambda item: item[1], reverse=True)

    occupied = np.zeros(line_len, dtype=bool)
    placements: list[tuple[int, int, int]] = []
    for color, run_len in order:
        candidates: list[int] = []
        for start in range(line_len - run_len + 1):
            end = start + run_len - 1
            if start <= center_idx <= end:
                continue
            lo = max(0, start - 1)
            hi = min(line_len, end + 2)
            if np.any(occupied[lo:hi]):
                continue
            candidates.append(int(start))
        if not candidates:
            return None

        left = [start for start in candidates if start + run_len - 1 < center_idx]
        right = [start for start in candidates if start > center_idx]
        if left and right:
            side_choices = left if random.random() < 0.5 else right
            start = random.choice(side_choices)
        else:
            start = random.choice(candidates)

        occupied[start : start + run_len] = True
        placements.append((int(color), int(start), int(run_len)))
    return placements


def _build_axis(line_len: int, bg: int, center_color: int, specs: list[tuple[int, int]]) -> np.ndarray | None:
    axis = np.full(line_len, int(bg), dtype=int)
    center_idx = line_len // 2
    axis[center_idx] = int(center_color)

    placements = _place_runs(line_len, center_idx, specs)
    if placements is None:
        return None
    for color, start, run_len in placements:
        axis[start : start + run_len] = int(color)
    return axis


def _semantic_checks(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    if g.ndim != 2 or h > 30 or w > 30 or h % 2 == 0 or w % 2 == 0:
        return False

    bg = int(get_bg(g))
    cr, cc = _center(g.shape)
    off_axis = np.ones(g.shape, dtype=bool)
    off_axis[cr, :] = False
    off_axis[:, cc] = False
    if not np.all(g[off_axis] == bg):
        return False

    row = g[cr, :]
    col = g[:, cc]
    row_colors = set(map(int, row)) - {bg}
    col_colors = set(map(int, col)) - {bg}
    shared = sorted(row_colors & col_colors)

    if len(shared) < 2:
        return False
    if len(row_colors) < 2 or len(col_colors) < 2:
        return False

    expected = _compose_from_axes(row, col, bg)
    try:
        actual = main(g)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    off_axis_colors = set(map(int, expected[off_axis])) - {bg}
    if len(off_axis_colors) < 2:
        return False

    filled_off_axis = int(np.sum((expected != bg) & off_axis))
    if filled_off_axis < 8 or filled_off_axis > int(off_axis.sum() * 0.35):
        return False

    mismatch_pair = False
    bg_gap_pair = False
    for r in range(h):
        if r == cr:
            continue
        for c in range(w):
            if c == cc:
                continue
            rv = int(row[c])
            cv = int(col[r])
            if rv != bg and cv != bg and rv != cv:
                mismatch_pair = True
            if (rv == bg) ^ (cv == bg):
                bg_gap_pair = True
            if mismatch_pair and bg_gap_pair:
                break
        if mismatch_pair and bg_gap_pair:
            break
    if not (mismatch_pair and bg_gap_pair):
        return False

    split_match = False
    for color in shared:
        row_hits = int(np.count_nonzero(row == int(color)))
        col_hits = int(np.count_nonzero(col == int(color)))
        if row_hits >= 2 and col_hits >= 2 and (_run_count(row, color) >= 2 or _run_count(col, color) >= 2):
            split_match = True
            break
    if not split_match:
        return False

    if np.array_equal(_naive_all_non_bg(row, col, bg), expected):
        return False
    if np.array_equal(_naive_bbox_fill(row, col, bg), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = _odd_between(GRID_MIN, GRID_MAX)
    w = _odd_between(GRID_MIN, GRID_MAX)
    bg = random.randint(0, 9)

    non_bg = [color for color in range(10) if color != bg]
    random.shuffle(non_bg)

    shared_count = 2 if min(h, w) < 19 else random.choice((2, 3))
    palette_size = min(len(non_bg), shared_count + random.randint(1, 2))
    palette = non_bg[:palette_size]
    shared = palette[:shared_count]
    extras = palette[shared_count:]
    split_color = random.choice(shared)

    row_only = extras[0] if extras and random.random() < 0.8 else None
    col_only = None
    if len(extras) >= 2 and random.random() < 0.8:
        col_only = extras[1]
    elif len(extras) == 1 and row_only is None:
        col_only = extras[0]

    center_candidates = [bg, bg] + [color for color in shared if color != split_color]
    if not center_candidates:
        center_candidates = [bg, split_color]
    center_color = int(random.choice(center_candidates))

    row_specs = _sample_specs(w, shared, row_only, split_color)
    col_specs = _sample_specs(h, shared, col_only, split_color)
    row = _build_axis(w, bg, center_color, row_specs)
    col = _build_axis(h, bg, center_color, col_specs)
    if row is None or col is None:
        return None

    grid = np.full((h, w), int(bg), dtype=int)
    cr, cc = _center(grid.shape)
    grid[cr, :] = row
    grid[:, cc] = col
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid center-axis cross sample")
