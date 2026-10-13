from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
GEN_ATTEMPTS = 600


def _most_common(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        pts: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            pts.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _open_runs(length: int, separators: np.ndarray) -> list[tuple[int, int]]:
    sep = set(map(int, separators))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(length):
        if idx in sep:
            if start is not None:
                runs.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        runs.append((start, length - 1))
    return runs


def _lattice_candidates(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for raw_color in np.unique(g):
        color = int(raw_color)
        if color == bg:
            continue
        for coords in _components(g == color):
            if coords.shape[0] < 8:
                continue
            r0, c0, r1, c1 = _bbox_coords(coords)
            bh = r1 - r0 + 1
            bw = c1 - c0 + 1
            if bh < 3 or bw < 3:
                continue

            sub = np.zeros((bh, bw), dtype=bool)
            sub[coords[:, 0] - r0, coords[:, 1] - c0] = True
            row_lines = np.flatnonzero(np.all(sub, axis=1))
            col_lines = np.flatnonzero(np.all(sub, axis=0))
            if len(row_lines) < 2 or len(col_lines) < 2:
                continue
            if int(row_lines[0]) != 0 or int(row_lines[-1]) != bh - 1:
                continue
            if int(col_lines[0]) != 0 or int(col_lines[-1]) != bw - 1:
                continue

            row_cells_local = _open_runs(bh, row_lines)
            col_cells_local = _open_runs(bw, col_lines)
            if not row_cells_local or not col_cells_local:
                continue

            row_cells = [(r0 + a, r0 + b) for a, b in row_cells_local]
            col_cells = [(c0 + a, c0 + b) for a, b in col_cells_local]
            candidates.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "row_cells": row_cells,
                    "col_cells": col_cells,
                    "rows": len(row_cells),
                    "cols": len(col_cells),
                    "area": int(coords.shape[0]),
                }
            )
    return candidates


def _select_scaffold_and_key(g: np.ndarray, bg: int) -> tuple[dict[str, object], dict[str, object]]:
    cands = _lattice_candidates(g, bg)
    pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    for i, a in enumerate(cands):
        for b in cands[i + 1 :]:
            if int(a["color"]) != int(b["color"]):
                continue
            ar, ac = int(a["rows"]), int(a["cols"])
            br, bc = int(b["rows"]), int(b["cols"])
            if ar > br and ac > bc:
                pairs.append((a, b))
            elif br > ar and bc > ac:
                pairs.append((b, a))
    assert len(pairs) == 1, "expected one large scaffold and one smaller same-color key lattice"
    return pairs[0]


def _cell_rects(lattice: dict[str, object]) -> list[list[tuple[int, int, int, int]]]:
    row_cells = lattice["row_cells"]
    col_cells = lattice["col_cells"]
    assert isinstance(row_cells, list) and isinstance(col_cells, list)
    return [[(r0, c0, r1, c1) for c0, c1 in col_cells] for r0, r1 in row_cells]


def _read_key(g: np.ndarray, key: dict[str, object], bg: int, line_color: int) -> np.ndarray:
    rects = _cell_rects(key)
    out = np.empty((len(rects), len(rects[0])), dtype=int)
    for i, row in enumerate(rects):
        for j, (r0, c0, r1, c1) in enumerate(row):
            vals = set(map(int, np.unique(g[r0 : r1 + 1, c0 : c1 + 1])))
            assert len(vals) == 1, "each key entry must be a solid color block"
            value = vals.pop()
            assert value == bg or value != line_color, "key entries cannot reuse the lattice color"
            out[i, j] = value
    return out


def _find_seed(
    g: np.ndarray, scaffold: dict[str, object], bg: int, line_color: int
) -> tuple[int, int, int, list[tuple[int, int]]]:
    hits: list[tuple[int, int, int, list[tuple[int, int]]]] = []
    for i, row in enumerate(_cell_rects(scaffold)):
        for j, (r0, c0, r1, c1) in enumerate(row):
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            mask = (sub != bg) & (sub != line_color)
            if not bool(mask.any()):
                continue
            colors = set(map(int, np.unique(sub[mask])))
            assert len(colors) == 1, "the seed motif must be monochrome"
            offsets = [(int(r), int(c)) for r, c in np.argwhere(mask)]
            hits.append((i, j, colors.pop(), offsets))
    assert len(hits) == 1, "expected exactly one scaffold cell to contain the seed motif"
    return hits[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp a seed motif through a scaffold cell array using a phase-locked periodic color key.

    Input structure:
    - One lattice color forms two separate complete rectangular lattices: a larger empty
      scaffold of R by C cells and a smaller r by c color-key lattice with r < R and c < C.
    - The key cells are solid color blocks, and exactly one key entry has the same color
      as the monochrome seed motif drawn inside one scaffold cell.
    - All other scaffold cells are empty background cells before the transformation.

    Rule:
    - Read the seed motif as offsets from its scaffold cell upper-left corner.
    - Tile the r by c key over the R by C scaffold, shifted so the seed scaffold cell
      lands on the unique key entry with the seed color.
    - For every scaffold cell whose shifted key entry is not background, stamp the seed
      offsets there recolored to that entry color; background key entries remain blank.

    Key insight:
    - The key is a periodic color map, not a picture to copy: the unique same-colored key
      entry anchors the tiling phase before the seed shape is recolored and stamped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common(g)
    scaffold, key_lattice = _select_scaffold_and_key(g, bg)
    line_color = int(scaffold["color"])

    source_row, source_col, seed_color, offsets = _find_seed(g, scaffold, bg, line_color)
    key = _read_key(g, key_lattice, bg, line_color)
    anchors = np.argwhere(key == seed_color)
    assert anchors.shape[0] == 1, "the key must contain exactly one same-color anchor"
    anchor_row, anchor_col = map(int, anchors[0])

    out = g.copy()
    scaffold_cells = _cell_rects(scaffold)
    kr, kc = key.shape
    for row_idx, row in enumerate(scaffold_cells):
        for col_idx, (r0, c0, r1, c1) in enumerate(row):
            key_row = (row_idx - source_row + anchor_row) % kr
            key_col = (col_idx - source_col + anchor_col) % kc
            mapped_color = int(key[key_row, key_col])
            if mapped_color == bg:
                continue
            assert mapped_color != line_color, "stamped color cannot be the lattice color"
            for dr, dc in offsets:
                r = r0 + dr
                c = c0 + dc
                if r <= r1 and c <= c1:
                    out[r, c] = mapped_color
    return out


# --------------- generator ---------------


def _lattice_shape(rows: int, cols: int, cell_h: int, cell_w: int) -> tuple[int, int]:
    return rows * cell_h + rows + 1, cols * cell_w + cols + 1


def _draw_lattice(
    grid: np.ndarray,
    top: int,
    left: int,
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    color: int,
) -> None:
    height, width = _lattice_shape(rows, cols, cell_h, cell_w)
    for i in range(rows + 1):
        rr = top + i * (cell_h + 1)
        grid[rr, left : left + width] = color
    for j in range(cols + 1):
        cc = left + j * (cell_w + 1)
        grid[top : top + height, cc] = color


def _gen_cell_rect(
    top: int, left: int, row: int, col: int, cell_h: int, cell_w: int
) -> tuple[int, int, int, int]:
    r0 = top + 1 + row * (cell_h + 1)
    c0 = left + 1 + col * (cell_w + 1)
    return r0, c0, r0 + cell_h - 1, c0 + cell_w - 1


def _fill_lattice_cells(
    grid: np.ndarray,
    top: int,
    left: int,
    values: np.ndarray,
    cell_h: int,
    cell_w: int,
) -> None:
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = int(values[i, j])
            if color == BACKGROUND:
                continue
            r0, c0, r1, c1 = _gen_cell_rect(top, left, i, j, cell_h, cell_w)
            grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _connected_motif(cell_h: int, cell_w: int) -> list[tuple[int, int]] | None:
    for _ in range(200):
        count = random.randint(3, min(6, cell_h * cell_w - 1))
        cells = {(random.randrange(cell_h), random.randrange(cell_w))}
        while len(cells) < count:
            r, c = random.choice(tuple(cells))
            nbrs = [(nr, nc) for nr, nc in _neighbors4(r, c) if 0 <= nr < cell_h and 0 <= nc < cell_w]
            cells.add(random.choice(nbrs))
        rows = {r for r, _c in cells}
        cols = {c for _r, c in cells}
        if len(rows) < 2 or len(cols) < 2:
            continue
        area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
        if area == len(cells):
            continue
        return sorted(cells)
    if cell_h >= 3 and cell_w >= 3:
        return [(0, 0), (1, 0), (1, 1), (2, 1)]
    return None


def _make_key(rows: int, cols: int, line_color: int) -> tuple[np.ndarray, tuple[int, int], int] | None:
    colors = [c for c in range(1, 10) if c != line_color]
    if len(colors) < 3:
        return None
    seed_color = random.choice(colors)
    others = [c for c in colors if c != seed_color]
    other_count = min(random.randint(2, 3), len(others), rows * cols - 2)
    if other_count < 2:
        return None
    other_colors = random.sample(others, other_count)

    all_positions = [(i, j) for i in range(rows) for j in range(cols)]
    anchor_choices = [p for p in all_positions if p != (0, 0)]
    anchor = random.choice(anchor_choices)
    rest = [p for p in all_positions if p != anchor]
    random.shuffle(rest)

    bg_count = random.randint(1, max(1, len(rest) - other_count))
    key = np.empty((rows, cols), dtype=int)
    key[:, :] = random.choice(other_colors)
    key[anchor] = seed_color

    pos_idx = 0
    for _ in range(bg_count):
        key[rest[pos_idx]] = BACKGROUND
        pos_idx += 1
    for color in other_colors:
        key[rest[pos_idx]] = color
        pos_idx += 1
    fill_pool = other_colors + [BACKGROUND]
    while pos_idx < len(rest):
        key[rest[pos_idx]] = random.choice(fill_pool)
        pos_idx += 1

    if np.count_nonzero(key == seed_color) != 1:
        return None
    if BACKGROUND not in set(map(int, key.ravel())):
        return None
    if len(set(map(int, key.ravel())) - {BACKGROUND, seed_color}) < 2:
        return None
    return key, anchor, seed_color


def _layout_regions(
    scaffold_h: int, scaffold_w: int, key_h: int, key_w: int
) -> tuple[int, int, tuple[int, int], tuple[int, int]] | None:
    gap = random.randint(1, 2)
    choices: list[tuple[str, int, int]] = []
    if scaffold_w + gap + key_w <= 30 and max(scaffold_h, key_h) <= 30:
        choices.append(("right", max(scaffold_h, key_h), scaffold_w + gap + key_w))
        choices.append(("left", max(scaffold_h, key_h), scaffold_w + gap + key_w))
    if scaffold_h + gap + key_h <= 30 and max(scaffold_w, key_w) <= 30:
        choices.append(("below", scaffold_h + gap + key_h, max(scaffold_w, key_w)))
        choices.append(("above", scaffold_h + gap + key_h, max(scaffold_w, key_w)))
    if not choices:
        return None

    orient, height, width = random.choice(choices)
    if orient == "right":
        scaffold_pos = (random.randrange(height - scaffold_h + 1), 0)
        key_pos = (random.randrange(height - key_h + 1), scaffold_w + gap)
    elif orient == "left":
        key_pos = (random.randrange(height - key_h + 1), 0)
        scaffold_pos = (random.randrange(height - scaffold_h + 1), key_w + gap)
    elif orient == "below":
        scaffold_pos = (0, random.randrange(width - scaffold_w + 1))
        key_pos = (scaffold_h + gap, random.randrange(width - key_w + 1))
    else:
        key_pos = (0, random.randrange(width - key_w + 1))
        scaffold_pos = (key_h + gap, random.randrange(width - scaffold_w + 1))
    return height, width, scaffold_pos, key_pos


def _stamp_with_key(
    grid: np.ndarray,
    scaffold_pos: tuple[int, int],
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    key: np.ndarray,
    anchor: tuple[int, int],
    source: tuple[int, int],
    offsets: list[tuple[int, int]],
    *,
    phase_locked: bool,
) -> np.ndarray:
    out = grid.copy()
    sr, sc = source
    ar, ac = anchor
    kr, kc = key.shape
    top, left = scaffold_pos
    for i in range(rows):
        for j in range(cols):
            if phase_locked:
                ki = (i - sr + ar) % kr
                kj = (j - sc + ac) % kc
            else:
                ki = i % kr
                kj = j % kc
            color = int(key[ki, kj])
            if color == BACKGROUND:
                continue
            r0, c0, r1, c1 = _gen_cell_rect(top, left, i, j, cell_h, cell_w)
            for dr, dc in offsets:
                r = r0 + dr
                c = c0 + dc
                if r <= r1 and c <= c1:
                    out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(4, 6)
    cols = random.randint(4, 6)
    cell_h = random.randint(3, 4)
    cell_w = random.randint(3, 4)
    scaffold_h, scaffold_w = _lattice_shape(rows, cols, cell_h, cell_w)
    if scaffold_h > 30 or scaffold_w > 30:
        return None

    key_rows = random.randint(2, min(4, rows - 1))
    key_cols = random.randint(2, min(4, cols - 1))
    key_cell = random.choice([1, 2])
    key_h, key_w = _lattice_shape(key_rows, key_cols, key_cell, key_cell)

    layout = _layout_regions(scaffold_h, scaffold_w, key_h, key_w)
    if layout is None:
        return None
    height, width, scaffold_pos, key_pos = layout

    line_color = random.randint(1, 9)
    key = _make_key(key_rows, key_cols, line_color)
    if key is None:
        return None
    key_values, anchor, seed_color = key

    source_options = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if (i % key_rows, j % key_cols) != anchor
    ]
    if not source_options:
        return None
    source = random.choice(source_options)
    offsets = _connected_motif(cell_h, cell_w)
    if offsets is None:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    _draw_lattice(grid, *scaffold_pos, rows, cols, cell_h, cell_w, line_color)
    _draw_lattice(grid, *key_pos, key_rows, key_cols, key_cell, key_cell, line_color)
    _fill_lattice_cells(grid, key_pos[0], key_pos[1], key_values, key_cell, key_cell)

    sr, sc = source
    source_r0, source_c0, _source_r1, _source_c1 = _gen_cell_rect(
        scaffold_pos[0], scaffold_pos[1], sr, sc, cell_h, cell_w
    )
    for dr, dc in offsets:
        grid[source_r0 + dr, source_c0 + dc] = seed_color

    expected = _stamp_with_key(
        grid,
        scaffold_pos,
        rows,
        cols,
        cell_h,
        cell_w,
        key_values,
        anchor,
        source,
        offsets,
        phase_locked=True,
    )
    wrong_default_phase = _stamp_with_key(
        grid,
        scaffold_pos,
        rows,
        cols,
        cell_h,
        cell_w,
        key_values,
        anchor,
        source,
        offsets,
        phase_locked=False,
    )
    if np.array_equal(expected, wrong_default_phase):
        return None

    tiled_colors = [
        int(key_values[(i - sr + anchor[0]) % key_rows, (j - sc + anchor[1]) % key_cols])
        for i in range(rows)
        for j in range(cols)
    ]
    if BACKGROUND not in tiled_colors:
        return None
    if len(set(tiled_colors) - {BACKGROUND}) < 3:
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
