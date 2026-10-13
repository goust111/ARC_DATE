from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_grid_lines(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    by_color: dict[int, dict[str, list[int]]] = {}

    for r in range(h):
        if bool(np.all(grid[r, :] == grid[r, 0])):
            color = int(grid[r, 0])
            by_color.setdefault(color, {"rows": [], "cols": []})["rows"].append(int(r))

    for c in range(w):
        if bool(np.all(grid[:, c] == grid[0, c])):
            color = int(grid[0, c])
            by_color.setdefault(color, {"rows": [], "cols": []})["cols"].append(int(c))

    candidates = [
        (color, data["rows"], data["cols"])
        for color, data in by_color.items()
        if data["rows"] and data["cols"]
    ]
    assert candidates, "expected full separator rows and columns"
    color, rows, cols = max(candidates, key=lambda item: (len(item[1]) + len(item[2]), item[0]))
    return int(color), sorted(rows), sorted(cols)


def _content_spans(line_indices: list[int], limit: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    line_set = set(int(idx) for idx in line_indices)
    start = 0
    for idx in range(limit):
        if idx in line_set:
            if start < idx:
                spans.append((int(start), int(idx)))
            start = idx + 1
    if start < limit:
        spans.append((int(start), int(limit)))
    return spans


def _grow_cell(cell: np.ndarray) -> np.ndarray:
    out = np.array(cell, dtype=int, copy=True)
    bg = _most_common_color(out)

    fg_coords = np.argwhere(out != bg)
    assert fg_coords.shape == (3, 2), "each partition should contain one 3-pixel seed"
    colors = {int(out[int(r), int(c)]) for r, c in fg_coords}
    assert len(colors) == 1, "seed must be monochrome"
    seed_color = next(iter(colors))

    r0, c0 = map(int, fg_coords.min(axis=0))
    r1, c1 = map(int, fg_coords.max(axis=0))
    assert r1 - r0 == 1 and c1 - c0 == 1, "seed must occupy a 2x2 box with one missing corner"

    footprint = np.zeros((2, 2), dtype=bool)
    for r, c in fg_coords:
        footprint[int(r) - r0, int(c) - c0] = True

    missing = np.argwhere(~footprint)
    assert missing.shape == (1, 2), "expected exactly one missing corner in the 2x2 seed box"
    miss_r, miss_c = map(int, missing[0])

    fill_r0 = r0 if miss_r == 1 else 0
    fill_r1 = out.shape[0] - 1 if miss_r == 1 else r1
    fill_c0 = c0 if miss_c == 1 else 0
    fill_c1 = out.shape[1] - 1 if miss_c == 1 else c1

    out[fill_r0 : fill_r1 + 1, fill_c0 : fill_c1 + 1] = seed_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each 3-pixel L seed into an inward-facing solid corner rectangle inside its partition.

    Input structure:
    - The board is split by one-cell-wide grey separator rows and columns into equal-sized rectangular partitions.
    - Each partition contains a dominant background and exactly one monochrome 3-pixel L seed.
    - The seed occupies three corners of a 2x2 box, so one corner of that box is visibly missing.

    Rule:
    - Identify the full grey separator rows and columns, then process each partition independently.
    - In each partition, locate the 2x2 box containing the L seed and read which corner of that box is missing; that missing corner is the opening direction.
    - Extend the seed's two arms away from the opening until they hit the relevant partition borders, filling the entire implied corner rectangle with the seed color.
    - Leave the grey separator lines unchanged.

    Key insight:
    - The transformation is not repeated translation of the L; the missing corner of the seed's 2x2 footprint selects which inward quadrant expands into a solid rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _grid_color, row_lines, col_lines = _find_grid_lines(g)
    row_spans = _content_spans(row_lines, g.shape[0])
    col_spans = _content_spans(col_lines, g.shape[1])
    assert row_spans and col_spans, "expected at least one partition"
    assert len({r1 - r0 for r0, r1 in row_spans}) == 1, "partition heights must match"
    assert len({c1 - c0 for c0, c1 in col_spans}) == 1, "partition widths must match"

    out = np.array(g, copy=True)
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            out[r0:r1, c0:c1] = _grow_cell(g[r0:r1, c0:c1])
    return out


_BG = 0
_GRID = 5
_SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# Keys are the missing corner inside the 2x2 seed footprint.
_SEED_SHAPES: dict[str, tuple[tuple[int, int], ...]] = {
    "tl": ((0, 1), (1, 0), (1, 1)),
    "tr": ((0, 0), (1, 0), (1, 1)),
    "bl": ((0, 0), (0, 1), (1, 1)),
    "br": ((0, 0), (0, 1), (1, 0)),
}


def _choose_corner_index(rng: random.Random, size: int, toward_start: bool) -> int:
    max_start = size - 2
    assert max_start >= 0
    midpoint = max_start // 2
    if toward_start:
        choices = list(range(0, midpoint + 1))
    else:
        choices = list(range(midpoint, max_start + 1))
    return int(rng.choice(choices))


def _seed_position(rng: random.Random, cell_h: int, cell_w: int, missing_corner: str) -> tuple[int, int]:
    open_down = missing_corner in {"bl", "br"}
    open_right = missing_corner in {"tr", "br"}

    r0 = _choose_corner_index(rng, cell_h, toward_start=open_down)
    c0 = _choose_corner_index(rng, cell_w, toward_start=open_right)
    return int(r0), int(c0)


def _pick_color(rng: random.Random, above: int | None, left: int | None) -> int:
    banned = {color for color in (above, left) if color is not None}
    candidates = [color for color in _SEED_COLORS if color not in banned]
    if not candidates:
        candidates = list(_SEED_COLORS)
    return int(rng.choice(candidates))


def _build_grid(
    rng: random.Random,
    n_part_rows: int,
    n_part_cols: int,
    cell_h: int,
    cell_w: int,
) -> np.ndarray:
    h = n_part_rows * cell_h + (n_part_rows + 1)
    w = n_part_cols * cell_w + (n_part_cols + 1)
    grid = np.full((h, w), _GRID, dtype=int)

    used_colors: list[list[int | None]] = [[None for _ in range(n_part_cols)] for _ in range(n_part_rows)]

    orientations: list[str] = []
    base = list(_SEED_SHAPES.keys())
    while len(orientations) < n_part_rows * n_part_cols:
        batch = base[:]
        rng.shuffle(batch)
        orientations.extend(batch)

    for idx, missing_corner in enumerate(orientations[: n_part_rows * n_part_cols]):
        pr = idx // n_part_cols
        pc = idx % n_part_cols

        rs = 1 + pr * (cell_h + 1)
        cs = 1 + pc * (cell_w + 1)
        grid[rs : rs + cell_h, cs : cs + cell_w] = _BG

        above = used_colors[pr - 1][pc] if pr > 0 else None
        left = used_colors[pr][pc - 1] if pc > 0 else None
        color = _pick_color(rng, above, left)
        used_colors[pr][pc] = color

        seed_r0, seed_c0 = _seed_position(rng, cell_h, cell_w, missing_corner)
        for dr, dc in _SEED_SHAPES[missing_corner]:
            grid[rs + seed_r0 + dr, cs + seed_c0 + dc] = color

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray:
    n_part_rows = rng.choice([2, 3])
    n_part_cols = rng.choice([2, 3])
    cell_h = rng.randint(4, 7)
    cell_w = rng.randint(4, 7)

    grid = _build_grid(rng, n_part_rows, n_part_cols, cell_h, cell_w)
    out = main(grid)
    assert isinstance(out, np.ndarray)
    assert out.shape == grid.shape
    assert np.any(out != grid), "output must differ from input"
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        try:
            return _gen_attempt(rng)
        except Exception:
            continue
    raise RuntimeError("generate_input: failed to produce a valid grid after 300 attempts")
