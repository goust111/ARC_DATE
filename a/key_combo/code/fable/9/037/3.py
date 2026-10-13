from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


def _solid_rect_candidate(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != 0):
        coords = np.argwhere(grid == color)
        if coords.shape[0] <= 1:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if area == coords.shape[0] and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
            candidates.append((color, (int(r0), int(c0), int(r1), int(c1))))
    assert len(candidates) == 1, "expected one unique solid rectangular occluder"
    return candidates[0]


def _collect_top_seeds(grid: np.ndarray, occ_bbox: tuple[int, int, int, int]) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = occ_bbox
    h, _w = grid.shape
    mid = h // 2

    visible = grid.copy()
    visible[r0 : r1 + 1, c0 : c1 + 1] = 0

    topmost: dict[tuple[int, int], tuple[int, int, int]] = {}
    for col in range(visible.shape[1]):
        for row in range(mid):
            color = int(visible[row, col])
            if color == 0:
                continue
            key = (color, col)
            if key not in topmost:
                topmost[key] = (row, col, color)
    seeds = sorted(topmost.values())
    assert seeds, "expected at least one visible top-half seed"
    return seeds


def _draw_full_trails(shape: tuple[int, int], seeds: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    h, _w = shape
    for row, col, color in seeds:
        rr = int(row)
        while rr < h:
            out[rr, col] = color
            rr += 2
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Remove the occluder, recover mirrored hidden seeds, and regrow every downward dotted trail.

    Input structure:
    - The grid has a black background, several colored seed pixels, and one unique solid rectangular occluder.
    - Seed positions are mirrored across the grid's horizontal midline, while each seed's same-color trail extends downward in the same column every 2 rows.
    - The occluder may hide bottom-half seeds and arbitrary trail fragments, but the mirrored top-half seed for every hidden seed remains visible.

    Rule:
    - Erase the solid occluder back to black.
    - Read one visible seed per `(color, column)` from the top half and reflect each one across the horizontal midline to recover its partner seed.
    - Redraw complete same-color dotted trails for all recovered seeds, placing pixels in the seed column at rows `seed_row, seed_row+2, seed_row+4, ...` until the bottom edge.

    Key insight:
    - Only seed positions obey mirror symmetry; trail pixels do not. The correct repair is "recover seeds by symmetry, then regrow trails by gravity" rather than mirroring the occluded patch itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h % 2 == 0 and h <= 30 and w <= 30

    occ_color, occ_bbox = _solid_rect_candidate(g)
    del occ_color

    top_seeds = _collect_top_seeds(g, occ_bbox)
    all_seeds = top_seeds[:]
    for row, col, color in top_seeds:
        all_seeds.append((h - 1 - row, col, color))

    return _draw_full_trails(g.shape, all_seeds)


_COLORS = tuple(range(1, 10))
_MAX_ATTEMPTS = 500


def _choose_spaced_columns(width: int, count: int) -> list[int] | None:
    candidates = list(range(1, width - 1))
    random.shuffle(candidates)
    chosen: list[int] = []
    for col in candidates:
        if all(abs(col - other) >= 2 for other in chosen):
            chosen.append(col)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _build_output(
    height: int,
    width: int,
    top_rows: list[int],
    cols: list[int],
    colors: list[int],
) -> np.ndarray:
    seeds: list[tuple[int, int, int]] = []
    for row, col, color in zip(top_rows, cols, colors):
        seeds.append((row, col, color))
        seeds.append((height - 1 - row, col, color))
    return _draw_full_trails((height, width), seeds)


def _pick_occluder(
    height: int,
    width: int,
    top_rows: list[int],
    cols: list[int],
) -> tuple[int, int, int, int] | None:
    mid = height // 2
    idx0 = random.randint(0, len(cols) - 1)
    idx1 = random.randint(idx0, len(cols) - 1)
    hidden_cols = cols[idx0 : idx1 + 1]
    hidden_bottom_rows = [height - 1 - top_rows[cols.index(col)] for col in hidden_cols]

    c0 = max(0, hidden_cols[0] - random.randint(0, 1))
    c1 = min(width - 1, hidden_cols[-1] + random.randint(0, 1))
    if c1 - c0 < 1:
        c1 = min(width - 1, c0 + 1)

    low_r0 = max(mid - 1, min(top_rows) + 3)
    high_r0 = min(min(hidden_bottom_rows), height - 3)
    if low_r0 > high_r0:
        return None
    r0 = random.randint(low_r0, high_r0)

    low_r1 = max(max(hidden_bottom_rows), r0 + 2)
    high_r1 = height - 1
    if low_r1 > high_r1:
        return None
    r1 = random.randint(low_r1, high_r1)
    return r0, c0, r1, c1


def _mirror_fill_attempt(inp: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    h, _w = inp.shape
    out = inp.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = 0
    for r in range(r0, r1 + 1):
        mr = h - 1 - r
        out[r, c0 : c1 + 1] = out[mr, c0 : c1 + 1]
    return out


def _restore_hidden_seeds_only(
    inp: np.ndarray,
    bbox: tuple[int, int, int, int],
    top_rows: list[int],
    cols: list[int],
    colors: list[int],
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = inp.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = 0
    for row, col, color in zip(top_rows, cols, colors):
        mirror_row = inp.shape[0] - 1 - row
        if r0 <= mirror_row <= r1 and c0 <= col <= c1:
            out[mirror_row, col] = color
    return out


def _validate_instance(
    inp: np.ndarray,
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    top_rows: list[int],
    cols: list[int],
    colors: list[int],
) -> bool:
    try:
        occ_color, found_bbox = _solid_rect_candidate(inp)
        if occ_color not in colors and occ_color != 0:
            pass
        if found_bbox != bbox:
            return False
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, out):
        return False

    mid = inp.shape[0] // 2
    visible = inp.copy()
    r0, c0, r1, c1 = bbox
    visible[r0 : r1 + 1, c0 : c1 + 1] = 0

    hidden_seed_count = 0
    for row, col, color in zip(top_rows, cols, colors):
        if visible[row, col] != color:
            return False
        mirror_row = inp.shape[0] - 1 - row
        if r0 <= mirror_row <= r1 and c0 <= col <= c1:
            hidden_seed_count += 1
        top_half_rows = [rr for rr in range(mid) if visible[rr, col] == color]
        if not top_half_rows or min(top_half_rows) != row:
            return False
    if hidden_seed_count == 0:
        return False

    mirrored = _mirror_fill_attempt(inp, bbox)
    if np.array_equal(mirrored, out):
        return False

    seeds_only = _restore_hidden_seeds_only(inp, bbox, top_rows, cols, colors)
    if np.array_equal(seeds_only, out):
        return False

    if np.count_nonzero(inp != out) < 4:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.choice([12, 14, 16, 18, 20, 22, 24])
    width = random.randint(10, 20)
    pair_count = random.randint(2, min(5, width // 2, len(_COLORS) - 1))

    cols = _choose_spaced_columns(width, pair_count)
    if cols is None:
        return None

    occluder_color = random.choice(_COLORS)
    colors = random.sample([c for c in _COLORS if c != occluder_color], pair_count)

    mid = height // 2
    top_rows = [random.randint(1, mid - 3) for _ in range(pair_count)]
    if len(set((r % 2) for r in top_rows)) == 1 and random.random() < 0.5:
        idx = random.randrange(pair_count)
        if top_rows[idx] < mid - 3:
            top_rows[idx] += 1
        elif top_rows[idx] > 1:
            top_rows[idx] -= 1

    out = _build_output(height, width, top_rows, cols, colors)
    bbox = _pick_occluder(height, width, top_rows, cols)
    if bbox is None:
        return None
    r0, c0, r1, c1 = bbox

    inp = out.copy()
    inp[r0 : r1 + 1, c0 : c1 + 1] = occluder_color

    if not _validate_instance(inp, out, bbox, top_rows, cols, colors):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid semi-symmetric occlusion input")
