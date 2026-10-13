from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MODES = ("direct", "mixed", "exhaust")
MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _distance_map(height: int, width: int) -> np.ndarray:
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    return np.minimum(np.minimum(rows, height - 1 - rows), np.minimum(cols, width - 1 - cols))


def _analyze_strip(grid: np.ndarray, coords: np.ndarray) -> dict | None:
    h, w = grid.shape
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])

    if len(rows) == 1:
        row = int(rows[0])
        if row not in (0, h - 1):
            return None
        ordered_cols = np.sort(coords[:, 1])
        expected = np.arange(int(ordered_cols[0]), int(ordered_cols[-1]) + 1)
        if not np.array_equal(ordered_cols, expected):
            return None
        ordered_coords = [(row, int(c)) for c in ordered_cols]
    elif len(cols) == 1:
        col = int(cols[0])
        if col not in (0, w - 1):
            return None
        ordered_rows = np.sort(coords[:, 0])
        expected = np.arange(int(ordered_rows[0]), int(ordered_rows[-1]) + 1)
        if not np.array_equal(ordered_rows, expected):
            return None
        ordered_coords = [(int(r), col) for r in ordered_rows]
    else:
        return None

    return {
        "coords": ordered_coords,
        "colors": [int(grid[r, c]) for r, c in ordered_coords],
    }


def _analyze_rings(grid: np.ndarray, coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    height, width = sub.shape
    dist = _distance_map(height, width)

    colors: list[int] = []
    hole_started = False
    for layer in range(int(dist.max()) + 1):
        vals = sub[dist == layer]
        if np.all(vals == BG):
            hole_started = True
            break
        if np.any(vals == BG):
            return None
        color = int(vals[0])
        if not np.all(vals == color):
            return None
        colors.append(color)

    if not hole_started or not colors:
        return None

    layer_count = len(colors)
    non_bg = sub != BG
    if not np.array_equal(non_bg, dist < layer_count):
        return None
    if not np.any(~non_bg):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "dist": dist,
        "colors": colors,
    }


def _assign_ring_colors(ring_colors: list[int], sequence: list[int]) -> list[int]:
    assigned: list[int] = []
    ptr = 0
    for original in ring_colors:
        while ptr < len(sequence) and sequence[ptr] == original:
            ptr += 1
        if ptr < len(sequence):
            assigned.append(sequence[ptr])
            ptr += 1
        else:
            assigned.append(original)
    return assigned


def _paint_rings(grid: np.ndarray, bbox: tuple[int, int, int, int], colors: list[int]) -> None:
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    dist = _distance_map(*sub.shape)
    for layer, color in enumerate(colors):
        sub[dist == layer] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor concentric rectangular rings from an external strip while skipping same-color matches for each layer.

    Input structure:
    - The background is black.
    - One non-black object is a stack of tightly nested one-pixel-thick rectangular rings around a black inner hole.
    - A separate non-black component lies on exactly one grid edge as a one-cell-thick color sequence strip, read left-to-right on a row or top-to-bottom on a column.

    Rule:
    - Identify the ring colors from the outer layer inward and read the edge strip colors in strip order.
    - For each ring layer from outside to inside, advance through the strip until finding a color different from that layer's original color; use that different color to repaint the layer.
    - If a strip color equals the current layer's original color, discard it and keep advancing; if the strip ends before a usable color is found, that layer keeps its original color.
    - Clear the sequence strip back to black in the output.

    Key insight:
    - The strip is not aligned one-to-one with ring depth: per-layer same-color collisions are skipped, so earlier skips shift which later strip colors reach the inner rings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _components(g != BG)
    assert len(comps) == 2, "expected exactly one ring object and one edge strip"

    strip_candidates = []
    ring_candidates = []
    for comp in comps:
        strip = _analyze_strip(g, comp)
        if strip is not None:
            strip_candidates.append(strip)
        rings = _analyze_rings(g, comp)
        if rings is not None:
            ring_candidates.append(rings)

    assert len(strip_candidates) == 1, "expected exactly one edge strip"
    assert len(ring_candidates) == 1, "expected exactly one concentric-ring object"

    strip = strip_candidates[0]
    rings = ring_candidates[0]
    new_colors = _assign_ring_colors(rings["colors"], strip["colors"])

    out = g.copy()
    for r, c in strip["coords"]:
        out[r, c] = BG

    r0, c0, r1, c1 = rings["bbox"]
    sub = out[r0 : r1 + 1, c0 : c1 + 1]
    for layer, color in enumerate(new_colors):
        sub[rings["dist"] == layer] = color

    return out


def _rand_other_color(color: int) -> int:
    return random.choice([candidate for candidate in PALETTE if candidate != color])


def _build_sequence(ring_colors: list[int], mode: str) -> list[int]:
    layer_count = len(ring_colors)

    if mode == "direct":
        return [_rand_other_color(color) for color in ring_colors]

    if mode == "mixed":
        skip_count = random.randint(1, layer_count - 1)
        skip_layers = set(random.sample(range(layer_count), skip_count))
        sequence: list[int] = []
        for layer, color in enumerate(ring_colors):
            if layer in skip_layers:
                sequence.append(color)
            sequence.append(_rand_other_color(color))
        return sequence

    assert mode == "exhaust"
    active_layers = random.randint((layer_count + 1) // 2, layer_count - 1)
    sequence = []
    for color in ring_colors[:active_layers]:
        sequence.extend((color, _rand_other_color(color)))
    return sequence


def _materialize_strip_cells(side: str, start: int, length: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    if side == "top":
        return [(0, c) for c in range(start, start + length)]
    if side == "bottom":
        return [(h - 1, c) for c in range(start, start + length)]
    if side == "left":
        return [(r, 0) for r in range(start, start + length)]
    return [(r, w - 1) for r in range(start, start + length)]


def _expected_output(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    ring_colors: list[int],
    strip_cells: list[tuple[int, int]],
    sequence: list[int],
) -> np.ndarray:
    out = grid.copy()
    for r, c in strip_cells:
        out[r, c] = BG
    _paint_rings(out, bbox, _assign_ring_colors(ring_colors, sequence))
    return out


def _sample_layout() -> tuple[tuple[int, int], tuple[int, int, int, int], str, int]:
    layers = random.randint(3, 5)
    inner_h = random.randint(1, 8)
    inner_w = random.randint(1, 8)
    ring_h = inner_h + 2 * layers
    ring_w = inner_w + 2 * layers
    side = random.choice(("top", "bottom", "left", "right"))

    top_low = 2 if side == "top" else 1
    bottom_low = 2 if side == "bottom" else 1
    left_low = 2 if side == "left" else 1
    right_low = 2 if side == "right" else 1

    min_h_margin = top_low + bottom_low
    min_w_margin = left_low + right_low
    max_h_margin = min(min_h_margin + 10, 30 - ring_h)
    max_w_margin = min(min_w_margin + 10, 30 - ring_w)
    assert max_h_margin >= min_h_margin and max_w_margin >= min_w_margin

    total_h_margin = random.randint(min_h_margin, max_h_margin)
    total_w_margin = random.randint(min_w_margin, max_w_margin)

    top = random.randint(top_low, total_h_margin - bottom_low)
    left = random.randint(left_low, total_w_margin - right_low)
    bottom = total_h_margin - top
    right = total_w_margin - left

    shape = (ring_h + top + bottom, ring_w + left + right)
    bbox = (top, left, top + ring_h - 1, left + ring_w - 1)
    assert max(shape) <= 30
    return shape, bbox, side, layers


def _gen_attempt() -> np.ndarray | None:
    shape, bbox, side, layers = _sample_layout()
    h, w = shape
    ring_colors = random.sample(PALETTE, layers)

    mode = random.choice(MODES)
    sequence = _build_sequence(ring_colors, mode)
    edge_len = w if side in ("top", "bottom") else h
    if len(sequence) > edge_len:
        return None

    if mode != "exhaust":
        tail = random.randint(0, min(3, edge_len - len(sequence)))
        sequence.extend(random.choice(PALETTE) for _ in range(tail))

    start = random.randint(0, edge_len - len(sequence))
    strip_cells = _materialize_strip_cells(side, start, len(sequence), shape)

    grid = np.zeros(shape, dtype=int)
    _paint_rings(grid, bbox, ring_colors)
    for (r, c), color in zip(strip_cells, sequence):
        grid[r, c] = color

    expected = _expected_output(grid, bbox, ring_colors, strip_cells, sequence)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-ring recoloring input")
