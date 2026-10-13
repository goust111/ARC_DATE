from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8)


def _groups(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []

    groups: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        groups.append((start, prev))
        start = prev = idx
    groups.append((start, prev))
    return groups


def _segments(length: int, groups: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not groups:
        return [(0, int(length))]

    segments: list[tuple[int, int]] = []
    start = 0
    for g0, g1 in groups:
        assert start < g0, "separator line cannot touch the grid border"
        segments.append((start, int(g0)))
        start = int(g1) + 1
    assert start < length, "separator line cannot touch the grid border"
    segments.append((start, int(length)))
    return segments


def _layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    row_candidates = [
        int(r)
        for r in range(h)
        if int(grid[r, 0]) != BLACK and np.all(grid[r, :] == grid[r, 0])
    ]
    col_candidates = [
        int(c)
        for c in range(w)
        if int(grid[0, c]) != BLACK and np.all(grid[:, c] == grid[0, c])
    ]
    assert row_candidates or col_candidates, "expected full separator rows or columns"

    sep_colors = {int(grid[r, 0]) for r in row_candidates} | {int(grid[0, c]) for c in col_candidates}
    assert len(sep_colors) == 1, "separator lines must all have the same color"
    sep = sep_colors.pop()

    row_groups = _groups(row_candidates)
    col_groups = _groups(col_candidates)
    row_segments = _segments(h, row_groups)
    col_segments = _segments(w, col_groups)

    heights = {r1 - r0 for r0, r1 in row_segments}
    widths = {c1 - c0 for c0, c1 in col_segments}
    assert len(heights) == 1 and next(iter(heights)) > 0
    assert len(widths) == 1 and next(iter(widths)) > 0
    assert len(row_segments) * len(col_segments) > 1, "need multiple layer panels"
    return int(sep), row_segments, col_segments


def _marker_run(tile: np.ndarray, marker: int) -> int:
    width = tile.shape[1]
    count = 0
    while count < width and int(tile[0, count]) == int(marker):
        count += 1
    assert count > 0, "each layer must have at least one corner marker pixel"
    return int(count)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite equal-sized layer panels by the count of top-left marker pixels.

    Input structure:
    - Black background is transparent inside each panel.
    - Equal-sized panels are arranged in a grid and separated by full grey lines.
    - Each panel has a contiguous marker run from its top-left corner.

    Rule:
    - Marker-run length 1..N encodes bottom-to-top layer order.
    - Remove those marker pixels, then overlay the panels in ascending order.
    - Black pixels are transparent; every other color is opaque and overwrites.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _sep, row_segments, col_segments = _layout(g)
    tile_h = row_segments[0][1] - row_segments[0][0]
    tile_w = col_segments[0][1] - col_segments[0][0]

    first_tile = g[row_segments[0][0] : row_segments[0][1], col_segments[0][0] : col_segments[0][1]]
    marker = int(first_tile[0, 0])
    assert marker != BLACK, "marker color must be non-black"

    n_layers = len(row_segments) * len(col_segments)
    by_order: dict[int, np.ndarray] = {}

    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            tile = g[r0:r1, c0:c1]
            assert tile.shape == (tile_h, tile_w)
            assert int(tile[0, 0]) == marker, "all panels must share the same marker color"

            order = _marker_run(tile, marker)
            assert 1 <= order <= n_layers, "marker count must fall within 1..N"
            assert order not in by_order, "marker counts must be unique"

            cleaned = tile.copy()
            cleaned[0, :order] = BLACK
            by_order[order] = cleaned

    assert set(by_order) == set(range(1, n_layers + 1)), "marker counts must cover the full order range"

    out = np.full((tile_h, tile_w), BLACK, dtype=int)
    for order in range(1, n_layers + 1):
        layer = by_order[order]
        mask = layer != BLACK
        out[mask] = layer[mask]
    return out


# --------------- generator ---------------

LAYOUTS = ((2, 2), (2, 3), (3, 2))
MAX_GENERATION_ATTEMPTS = 300


def _sample_layout() -> tuple[int, int, int, int]:
    rows, cols = random.choice(LAYOUTS)
    n_layers = rows * cols

    min_h = 5 if n_layers == 4 else 6
    max_h = min(9, (30 - (rows - 1)) // rows)
    min_w = n_layers + 2
    max_w = min(10 if cols == 3 else 12, (30 - (cols - 1)) // cols)
    assert min_h <= max_h and min_w <= max_w

    tile_h = random.randint(min_h, max_h)
    tile_w = random.randint(min_w, max_w)
    return int(rows), int(cols), int(tile_h), int(tile_w)


def _shape_relative(max_h: int, max_w: int) -> list[tuple[int, int]] | None:
    families = ("rect", "hline", "vline", "ell", "diag_down", "diag_up", "plus", "frame")
    weights = (5, 3, 3, 4, 2, 2, 2, 2)

    for _ in range(20):
        family = random.choices(families, weights=weights, k=1)[0]

        if family == "rect":
            sh = random.randint(1, min(3, max_h))
            sw = random.randint(1, min(4, max_w))
            if sh * sw < 2:
                continue
            return [(r, c) for r in range(sh) for c in range(sw)]

        if family == "hline" and max_w >= 2:
            ln = random.randint(2, min(5, max_w))
            return [(0, c) for c in range(ln)]

        if family == "vline" and max_h >= 2:
            ln = random.randint(2, min(5, max_h))
            return [(r, 0) for r in range(ln)]

        if family == "ell" and max_h >= 2 and max_w >= 2:
            arm_h = random.randint(2, min(4, max_h))
            arm_w = random.randint(2, min(4, max_w))
            return list({(r, 0) for r in range(arm_h)} | {(0, c) for c in range(arm_w)})

        if family == "diag_down" and min(max_h, max_w) >= 2:
            ln = random.randint(2, min(4, max_h, max_w))
            return [(k, k) for k in range(ln)]

        if family == "diag_up" and min(max_h, max_w) >= 2:
            ln = random.randint(2, min(4, max_h, max_w))
            return [(ln - 1 - k, k) for k in range(ln)]

        if family == "plus" and max_h >= 3 and max_w >= 3:
            return [(1, 0), (1, 1), (1, 2), (0, 1), (2, 1)]

        if family == "frame" and max_h >= 3 and max_w >= 3:
            sh = random.randint(3, min(4, max_h))
            sw = random.randint(3, min(4, max_w))
            coords = []
            for r in range(sh):
                for c in range(sw):
                    if r in (0, sh - 1) or c in (0, sw - 1):
                        coords.append((r, c))
            return coords

    return None


def _sample_shape(
    tile_h: int,
    tile_w: int,
    allowed: np.ndarray,
    occupied: np.ndarray,
    *,
    must_intersect: np.ndarray | None = None,
) -> list[tuple[int, int]] | None:
    free = allowed & ~occupied
    if must_intersect is not None and not bool(np.any(free & must_intersect)):
        return None

    for _ in range(80):
        rel = _shape_relative(tile_h, tile_w)
        if rel is None:
            continue
        sh = max(r for r, _ in rel) + 1
        sw = max(c for _, c in rel) + 1
        top = random.randint(0, tile_h - sh)
        left = random.randint(0, tile_w - sw)
        coords = [(top + r, left + c) for r, c in rel]
        if any(not free[r, c] for r, c in coords):
            continue
        if must_intersect is not None and not any(bool(must_intersect[r, c]) for r, c in coords):
            continue
        return coords
    return None


def _paint(tile: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        tile[r, c] = int(color)


def _composite(layers: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = np.full_like(layers[0], BLACK)
    stack_depth = np.zeros_like(layers[0], dtype=int)
    owner = np.full_like(layers[0], -1, dtype=int)

    for idx, layer in enumerate(layers):
        mask = layer != BLACK
        stack_depth += mask.astype(int)
        out[mask] = layer[mask]
        owner[mask] = int(idx)
    return out, stack_depth, owner


def _build_layers(tile_h: int, tile_w: int, n_layers: int) -> list[np.ndarray] | None:
    reserved = np.zeros((tile_h, tile_w), dtype=bool)
    reserved[0, :n_layers] = True

    layers = [np.full((tile_h, tile_w), BLACK, dtype=int) for _ in range(n_layers)]
    visible_claim = np.zeros((tile_h, tile_w), dtype=bool)

    for idx in range(n_layers - 1, -1, -1):
        layer = layers[idx]
        occupied = np.zeros((tile_h, tile_w), dtype=bool)
        above_visible = visible_claim.copy()

        guaranteed = _sample_shape(tile_h, tile_w, ~reserved & ~visible_claim, occupied)
        if guaranteed is None:
            return None
        base_color = random.choice(PALETTE)
        _paint(layer, guaranteed, base_color)
        for r, c in guaranteed:
            occupied[r, c] = True

        if above_visible.any() and (random.random() < 0.85 or idx == 0):
            hidden = _sample_shape(
                tile_h,
                tile_w,
                ~reserved,
                occupied,
                must_intersect=above_visible,
            )
            if hidden is not None:
                hidden_color_choices = [color for color in PALETTE if color != base_color]
                _paint(layer, hidden, random.choice(hidden_color_choices or list(PALETTE)))
                for r, c in hidden:
                    occupied[r, c] = True

        if random.random() < 0.60:
            extra = _sample_shape(tile_h, tile_w, ~reserved, occupied)
            if extra is not None:
                _paint(layer, extra, random.choice(PALETTE))
                for r, c in extra:
                    occupied[r, c] = True

        layer_mask = layer != BLACK
        visible_claim |= layer_mask & ~above_visible

        if not np.any(layer_mask):
            return None
        if np.count_nonzero(layer_mask) >= tile_h * tile_w - n_layers:
            return None

    return layers


def _assemble_input(
    layers: list[np.ndarray],
    rows: int,
    cols: int,
    tile_h: int,
    tile_w: int,
) -> np.ndarray | None:
    n_layers = len(layers)
    order_by_position = list(range(1, n_layers + 1))
    random.shuffle(order_by_position)
    if order_by_position == list(range(1, n_layers + 1)):
        return None

    grid_h = rows * tile_h + (rows - 1)
    grid_w = cols * tile_w + (cols - 1)
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    for r in range(1, rows):
        grid[r * tile_h + (r - 1), :] = GREY
    for c in range(1, cols):
        grid[:, c * tile_w + (c - 1)] = GREY

    pos = 0
    for rr in range(rows):
        for cc in range(cols):
            order = order_by_position[pos]
            tile = layers[order - 1].copy()
            tile[0, :order] = MAROON
            r0 = rr * (tile_h + 1)
            c0 = cc * (tile_w + 1)
            grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile
            pos += 1

    return grid


def _gen_attempt() -> np.ndarray | None:
    rows, cols, tile_h, tile_w = _sample_layout()
    n_layers = rows * cols

    layers = _build_layers(tile_h, tile_w, n_layers)
    if layers is None:
        return None

    output, stack_depth, owner = _composite(layers)
    if len(np.unique(output[output != BLACK])) < 2:
        return None
    if np.count_nonzero(output) < n_layers + 4:
        return None
    if np.count_nonzero(stack_depth >= 2) < 2:
        return None
    if any(np.count_nonzero(owner == idx) == 0 for idx in range(n_layers)):
        return None

    grid = _assemble_input(layers, rows, cols, tile_h, tile_w)
    if grid is None:
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid layering puzzle")
