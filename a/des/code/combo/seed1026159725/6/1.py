from __future__ import annotations

from collections import Counter
import random

import numpy as np


RED = 2
MAX_GENERATION_ATTEMPTS = 400
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    assert counts, "expected nonempty values"
    best = counts.most_common()
    assert len(best) == 1 or best[0][1] > best[1][1], "expected a unique dominant color"
    return int(best[0][0])


def _separator_lines(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    height, width = map(int, grid.shape)
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in map(int, np.unique(grid)):
        rows = [r for r in range(height) if np.all(grid[r, :] == color)]
        cols = [c for c in range(width) if np.all(grid[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected exactly one separator color"
    return candidates[0]


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        assert start < sep, "separator must leave a nonempty panel"
        spans.append((start, sep - 1))
        start = sep + 1
    assert start < length, "separator must leave a final nonempty panel"
    spans.append((start, length - 1))
    return spans


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected a nonempty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paint_frame(tile: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    tile[r0, c0 : c1 + 1] = color
    tile[r1, c0 : c1 + 1] = color
    tile[r0 : r1 + 1, c0] = color
    tile[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert the number of red marker dots in each separator-defined panel into
    concentric rectangular frames around the panel's central object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    separator, sep_rows, sep_cols = _separator_lines(g)
    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)

    panel_heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    panel_widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(panel_heights) == 1 and len(panel_widths) == 1, "expected equal-sized panels"

    bg = _unique_mode(g[g != separator])
    out = g.copy()

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            out_tile = out[r0 : r1 + 1, c0 : c1 + 1]

            marker_count = int(np.sum(tile == RED))
            obj_mask = (tile != bg) & (tile != separator) & (tile != RED)
            assert obj_mask.any(), "each panel must contain an object"

            colors = set(map(int, np.unique(tile[obj_mask])))
            assert len(colors) == 1, "object must be monochrome"
            obj_color = int(next(iter(colors)))

            out_tile[tile == RED] = bg
            br0, bc0, br1, bc1 = _bbox(obj_mask)
            for layer in range(1, marker_count + 1):
                fr0 = br0 - layer
                fc0 = bc0 - layer
                fr1 = br1 + layer
                fc1 = bc1 + layer
                assert fr0 >= 0 and fc0 >= 0 and fr1 < tile.shape[0] and fc1 < tile.shape[1]
                _paint_frame(out_tile, fr0, fc0, fr1, fc1, obj_color)

    return out


def _trim_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _shape_array(cells: set[tuple[int, int]], color: int) -> np.ndarray:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    arr = np.zeros((max_r + 1, max_c + 1), dtype=int)
    for r, c in cells:
        arr[r, c] = color
    return arr


def _make_object_shape(max_h: int, max_w: int, color: int) -> np.ndarray | None:
    candidates = [(h, w) for h in range(2, max_h + 1) for w in range(2, max_w + 1)]
    if not candidates:
        return None
    h, w = random.choice(candidates)
    area_cap = min(h * w, 7)
    area = random.randint(3, area_cap)

    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = [start]
    while len(cells) < area:
        anchor = random.choice(frontier)
        growth = []
        for dr, dc in DIR4:
            rr = anchor[0] + dr
            cc = anchor[1] + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                growth.append((rr, cc))
        if not growth:
            frontier = list(cells)
            if not frontier:
                return None
            continue
        new_cell = random.choice(growth)
        cells.add(new_cell)
        frontier.append(new_cell)

    return _shape_array(_trim_cells(cells), color)


def _expanded_bbox_cells(top: int, left: int, shape: np.ndarray, layers: int) -> set[tuple[int, int]]:
    sh, sw = map(int, shape.shape)
    r0 = top - layers
    c0 = left - layers
    r1 = top + sh - 1 + layers
    c1 = left + sw - 1 + layers
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _pick_markers(
    panel_h: int,
    panel_w: int,
    forbidden: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    candidates = [
        (r, c)
        for r in range(panel_h)
        for c in range(panel_w)
        if (r, c) not in forbidden
    ]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for cell in candidates:
        if all(abs(cell[0] - r) + abs(cell[1] - c) >= 2 for r, c in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return None


def _render_panel(
    panel_h: int,
    panel_w: int,
    bg: int,
    obj_color: int,
    marker_count: int,
) -> np.ndarray | None:
    tile = np.full((panel_h, panel_w), bg, dtype=int)

    max_h = panel_h - 2 * (marker_count + 1)
    max_w = panel_w - 2 * (marker_count + 1)
    shape = _make_object_shape(max_h, max_w, obj_color)
    if shape is None:
        return None
    sh, sw = map(int, shape.shape)

    top = (panel_h - sh) // 2
    left = (panel_w - sw) // 2
    if top < marker_count or left < marker_count:
        return None
    if top + sh - 1 + marker_count >= panel_h or left + sw - 1 + marker_count >= panel_w:
        return None

    shape_mask = shape != 0
    tile[top : top + sh, left : left + sw][shape_mask] = obj_color

    object_cells = {
        (top + r, left + c)
        for r in range(sh)
        for c in range(sw)
        if shape_mask[r, c]
    }
    halo_cells = _expanded_bbox_cells(top, left, shape, marker_count)
    marker_positions = _pick_markers(panel_h, panel_w, object_cells | halo_cells, marker_count)
    if marker_positions is None:
        return None
    for r, c in marker_positions:
        tile[r, c] = RED
    return tile


def _build_count_schedule(panel_count: int) -> list[int]:
    base = [1, 2, 3] * ((panel_count + 2) // 3)
    counts = base[:panel_count]
    random.shuffle(counts)
    return counts


def _gen_attempt() -> np.ndarray | None:
    panel_rows = random.randint(2, 3)
    panel_cols = random.randint(2, 3)
    panel_h = random.randint(9, 12)
    panel_w = random.randint(9, 12)

    height = panel_rows * panel_h + (panel_rows - 1)
    width = panel_cols * panel_w + (panel_cols - 1)

    palette = list(range(10))
    palette.remove(RED)
    bg, separator = random.sample(palette, 2)
    object_palette = [c for c in range(10) if c not in {bg, separator, RED}]
    if not object_palette:
        return None

    grid = np.full((height, width), bg, dtype=int)
    for r in range(1, panel_rows):
        grid[r * panel_h + (r - 1), :] = separator
    for c in range(1, panel_cols):
        grid[:, c * panel_w + (c - 1)] = separator

    counts = _build_count_schedule(panel_rows * panel_cols)
    random.shuffle(counts)

    idx = 0
    for pr in range(panel_rows):
        for pc in range(panel_cols):
            obj_color = random.choice(object_palette)
            marker_count = counts[idx]
            idx += 1
            tile = _render_panel(panel_h, panel_w, bg, obj_color, marker_count)
            if tile is None:
                return None
            r0 = pr * (panel_h + 1)
            c0 = pc * (panel_w + 1)
            grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = tile

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
