from __future__ import annotations

import random

import numpy as np


BLACK = 0
SEPARATOR = 9
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8]
QUADRANTS = ("ul", "ur", "ll", "lr")
ROTATION_STEPS_CW = {"ul": 0, "ur": 1, "lr": 2, "ll": 3}
GENERATION_ATTEMPTS = 300

TEMPLATES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
]


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one foreground pixel"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1], (int(r0), int(c0), int(r1), int(c1))


def _infer_separator_color(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    candidates = []
    for color in sorted(int(v) for v in np.unique(g)):
        row_hits = [r for r in range(g.shape[0]) if np.all(g[r, :] == color)]
        col_hits = [c for c in range(g.shape[1]) if np.all(g[:, c] == color)]
        if row_hits and col_hits:
            candidates.append((len(row_hits) + len(col_hits), color))
    assert candidates, "expected a separator color that fills complete rows and columns"
    candidates.sort(reverse=True)
    return int(candidates[0][1])


def _split_intervals(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + separators + [length]
    intervals = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a > 1:
            intervals.append((int(a + 1), int(b)))
    assert intervals, "expected at least one panel interval"
    return intervals


def _infer_panel_layout(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], int]:
    g = np.asarray(grid, dtype=int)
    separator = _infer_separator_color(g)
    row_separators = [r for r in range(g.shape[0]) if np.all(g[r, :] == separator)]
    col_separators = [c for c in range(g.shape[1]) if np.all(g[:, c] == separator)]
    row_intervals = _split_intervals(g.shape[0], row_separators)
    col_intervals = _split_intervals(g.shape[1], col_separators)
    row_sizes = {r1 - r0 for r0, r1 in row_intervals}
    col_sizes = {c1 - c0 for c0, c1 in col_intervals}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "panels must all be the same size"
    panel_size = row_sizes.pop()
    assert panel_size == col_sizes.pop(), "panels must be square"
    assert panel_size % 2 == 1, "panel side length must be odd"
    return int(separator), int(panel_size), row_intervals, col_intervals, int(panel_size // 2)


def _infer_background_color(grid: np.ndarray, separator: int) -> int:
    vals, counts = np.unique(grid[grid != separator], return_counts=True)
    assert len(vals) > 0
    return int(vals[np.argmax(counts)])


def _quadrant_from_bbox(bbox: tuple[int, int, int, int], center: int) -> str:
    r0, c0, r1, c1 = bbox
    if r1 < center and c1 < center:
        return "ul"
    if r1 < center and c0 > center:
        return "ur"
    if r0 > center and c1 < center:
        return "ll"
    if r0 > center and c0 > center:
        return "lr"
    raise AssertionError("shape bbox must lie strictly inside one quadrant")


def _place_centered(shape: np.ndarray, color: int, panel_size: int, bg: int) -> np.ndarray:
    out = np.full((panel_size, panel_size), bg, dtype=int)
    h, w = shape.shape
    center = panel_size // 2
    top = center - h // 2
    left = center - w // 2
    assert 0 <= top <= panel_size - h
    assert 0 <= left <= panel_size - w
    out[top : top + h, left : left + w][shape] = color
    return out


def _extract_panel_shape(panel: np.ndarray, bg: int, separator: int) -> tuple[np.ndarray, int, str]:
    colors = [int(v) for v in np.unique(panel) if int(v) not in (bg, separator)]
    assert len(colors) == 1, "each panel must contain exactly one foreground color"
    color = colors[0]
    mask = panel == color
    cropped, bbox = _crop_mask(mask)
    quadrant = _quadrant_from_bbox(bbox, panel.shape[0] // 2)
    return cropped, int(color), quadrant


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each panel's corner shape according to its quadrant, then place it at the panel center.

    Input structure:
    - Complete separator-colored rows and columns partition the grid into equal odd-sized square panels.
    - Inside every panel, the background is black except for one connected monochrome shape.
    - That shape lies wholly inside exactly one quadrant of the panel, leaving the center pixel empty.

    Rule:
    - Determine which quadrant contains the shape: upper-left, upper-right, lower-left, or lower-right.
    - Use the quadrant to choose a clockwise rotation: upper-left 0 degrees, upper-right 90 degrees, lower-right 180 degrees, lower-left 270 degrees.
    - Rotate the shape by that amount, erase its original corner placement, and redraw the rotated shape so its bounding-box center is aligned with the panel center.
    - Keep the separator rows and columns unchanged.

    Key insight:
    - The corner location is not just a placement cue; it encodes how much the shape must rotate before being moved into the center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator, panel_size, row_intervals, col_intervals, _center = _infer_panel_layout(g)
    bg = _infer_background_color(g, separator)

    out = np.full_like(g, bg)
    out[[r for r in range(g.shape[0]) if np.all(g[r, :] == separator)], :] = separator
    out[:, [c for c in range(g.shape[1]) if np.all(g[:, c] == separator)]] = separator

    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            panel = g[r0:r1, c0:c1]
            shape, color, quadrant = _extract_panel_shape(panel, bg, separator)
            k_cw = ROTATION_STEPS_CW[quadrant]
            rotated = np.rot90(shape, -k_cw)
            rotated, _bbox = _crop_mask(rotated)
            out[r0:r1, c0:c1] = _place_centered(rotated, color, panel_size, bg)

    return out


def _transform_mask(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    transformed = mask.copy()
    if rng.random() < 0.5:
        transformed = np.fliplr(transformed)
    if rng.random() < 0.5:
        transformed = np.flipud(transformed)
    transformed = np.rot90(transformed, rng.randrange(4))
    transformed, _bbox = _crop_mask(transformed)
    return transformed


def _random_shape(panel_size: int, rng: random.Random) -> np.ndarray:
    max_dim = panel_size // 2
    for _ in range(80):
        mask = _transform_mask(rng.choice(TEMPLATES), rng)
        h, w = mask.shape
        if h <= max_dim and w <= max_dim:
            return mask
    raise RuntimeError("failed to sample a shape that fits inside a quadrant")


def _quadrant_origin(quadrant: str, panel_size: int, shape: np.ndarray, rng: random.Random) -> tuple[int, int]:
    center = panel_size // 2
    h, w = shape.shape
    row_min = 0 if quadrant in ("ul", "ur") else center + 1
    row_max = center - h if quadrant in ("ul", "ur") else panel_size - h
    col_min = 0 if quadrant in ("ul", "ll") else center + 1
    col_max = center - w if quadrant in ("ul", "ll") else panel_size - w
    assert row_min <= row_max and col_min <= col_max
    return rng.randint(row_min, row_max), rng.randint(col_min, col_max)


def _render_panel(panel_size: int, quadrant: str, color: int, rng: random.Random) -> np.ndarray:
    panel = np.full((panel_size, panel_size), BLACK, dtype=int)
    shape = _random_shape(panel_size, rng)
    r0, c0 = _quadrant_origin(quadrant, panel_size, shape, rng)
    panel[r0 : r0 + shape.shape[0], c0 : c0 + shape.shape[1]][shape] = color
    assert panel[panel_size // 2, panel_size // 2] == BLACK
    return panel


def _valid_layouts() -> list[tuple[int, int, int]]:
    layouts = []
    for panel_size in (7, 9):
        for panel_rows in (2, 3):
            for panel_cols in (2, 3):
                height = panel_rows * panel_size + (panel_rows - 1)
                width = panel_cols * panel_size + (panel_cols - 1)
                if height <= 30 and width <= 30 and panel_rows * panel_cols >= 4:
                    layouts.append((panel_size, panel_rows, panel_cols))
    return layouts


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    panel_size, panel_rows, panel_cols = rng.choice(_valid_layouts())
    height = panel_rows * panel_size + (panel_rows - 1)
    width = panel_cols * panel_size + (panel_cols - 1)
    grid = np.full((height, width), BLACK, dtype=int)

    for r in range(1, panel_rows):
        grid[r * panel_size + (r - 1), :] = SEPARATOR
    for c in range(1, panel_cols):
        grid[:, c * panel_size + (c - 1)] = SEPARATOR

    panel_count = panel_rows * panel_cols
    quadrants = list(QUADRANTS)
    while len(quadrants) < panel_count:
        quadrants.append(rng.choice(QUADRANTS))
    rng.shuffle(quadrants)

    colors = [rng.choice(PALETTE) for _ in range(panel_count)]
    if len(set(colors)) == 1:
        colors[rng.randrange(panel_count)] = rng.choice([c for c in PALETTE if c != colors[0]])

    idx = 0
    for pr in range(panel_rows):
        for pc in range(panel_cols):
            r0 = pr * (panel_size + 1)
            c0 = pc * (panel_size + 1)
            grid[r0 : r0 + panel_size, c0 : c0 + panel_size] = _render_panel(
                panel_size,
                quadrants[idx],
                colors[idx],
                rng,
            )
            idx += 1

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
