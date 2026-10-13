from __future__ import annotations

import random

import numpy as np


BLACK = 0
COLOR_POOL = tuple(range(1, 10))
GRID_SIZE_RANGE = (15, 30)
PALETTE_LENGTH_RANGE = (4, 7)
MAX_ATTEMPTS = 400

_CONTROL_ABOVE_BAG: list[bool] = []


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_marker(coords: np.ndarray) -> dict[str, int] | None:
    if int(coords.shape[0]) != 4:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 3):
        return None

    rel = {(int(r - r0), int(c - c0)) for r, c in coords}
    control_above = {(0, 1), (1, 0), (1, 1), (1, 2)}
    control_below = {(0, 0), (0, 1), (0, 2), (1, 1)}
    mid_col = c0 + 1
    if rel == control_above:
        return {"bar_row": r1, "control_row": r0, "mid_col": mid_col}
    if rel == control_below:
        return {"bar_row": r0, "control_row": r1, "mid_col": mid_col}
    return None


def _parse_palette(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r0 != r1:
        return None
    if int(coords.shape[0]) != c1 - c0 + 1:
        return None
    colors = np.asarray(grid[r0, c0 : c1 + 1], dtype=int)
    if np.any(colors == bg):
        return None
    return {"row": r0, "c0": c0, "c1": c1, "colors": colors.copy()}


def main(grid: np.ndarray) -> np.ndarray:
    """Use the palette row chosen by a hidden ray to paint Chebyshev-distance rings.

    Input structure:
    - The background is black.
    - One foreground object is a pointing marker: a horizontal three-cell bar with a
      single control pixel touching the bar midpoint from above or below.
    - Two separate continuous multicolor palette rows lie on opposite sides of the
      marker, and each row crosses the marker midpoint column.

    Rule:
    - Read the bar midpoint and follow the vertical ray that goes away from the
      control pixel.
    - The palette row hit by that ray is the active palette; the ray intersection
      cell is the ring center.
    - Recolor every background cell by the active palette color at index
      `(chebyshev_distance_to_center mod palette_length)`, while leaving the marker
      and both palette rows unchanged.

    Key insight:
    - The output colors are controlled by an unseen ray-target selection step: the
      two visible palette rows look similar, but only the one on the side opposite
      the control pixel provides the cyclic ring code.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    components = _connected_components(g != bg)
    assert len(components) == 3, "expected one marker and two palette rows"

    marker = None
    palettes: list[dict[str, object]] = []
    for coords in components:
        parsed_marker = _parse_marker(coords)
        if parsed_marker is not None:
            assert marker is None, "expected a unique marker"
            marker = parsed_marker
            continue
        parsed_palette = _parse_palette(g, coords, bg)
        assert parsed_palette is not None, "non-marker components must be palette rows"
        palettes.append(parsed_palette)

    assert marker is not None, "marker not found"
    assert len(palettes) == 2, "expected exactly two palette rows"

    bar_row = int(marker["bar_row"])
    control_row = int(marker["control_row"])
    mid_col = int(marker["mid_col"])
    direction = 1 if control_row < bar_row else -1

    candidates = [
        palette
        for palette in palettes
        if (int(palette["row"]) - bar_row) * direction > 0
        and int(palette["c0"]) <= mid_col <= int(palette["c1"])
    ]
    assert len(candidates) == 1, "ray must hit exactly one palette row"
    active = candidates[0]

    center_row = int(active["row"])
    colors = np.asarray(active["colors"], dtype=int)
    palette_len = int(colors.size)
    assert palette_len > 0

    rows = np.arange(g.shape[0], dtype=int)[:, None]
    cols = np.arange(g.shape[1], dtype=int)[None, :]
    dist = np.maximum(np.abs(rows - center_row), np.abs(cols - mid_col))

    out = g.copy()
    bg_mask = g == bg
    out[bg_mask] = colors[dist[bg_mask] % palette_len]
    return out


def _next_control_above() -> bool:
    global _CONTROL_ABOVE_BAG
    if not _CONTROL_ABOVE_BAG:
        _CONTROL_ABOVE_BAG = [True, False, True, False]
        random.shuffle(_CONTROL_ABOVE_BAG)
    return _CONTROL_ABOVE_BAG.pop()


def _sample_palette_row(width: int, mid_col: int, length: int, forbidden: set[int]) -> tuple[int, np.ndarray]:
    allowed = [color for color in COLOR_POOL if color not in forbidden]
    colors = np.array(random.sample(allowed, length), dtype=int)
    hit_lo = max(0, length - width + mid_col)
    hit_hi = min(length - 1, mid_col)
    hit_index = random.randint(hit_lo, hit_hi)
    start = mid_col - hit_index
    return start, colors


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    if min(h, w) < 8:
        return None

    control_above = _next_control_above()
    bar_row = random.randint(4, h - 5)
    control_row = bar_row - 1 if control_above else bar_row + 1
    top_hi = bar_row - 3 if control_above else bar_row - 2
    bottom_lo = bar_row + 2 if control_above else bar_row + 3
    top_row = random.randint(1, top_hi)
    bottom_row = random.randint(bottom_lo, h - 2)
    mid_col = random.randint(1, w - 2)

    bar_color, control_color = random.sample(COLOR_POOL, 2)
    forbidden = {bar_color, control_color}
    max_len = min(PALETTE_LENGTH_RANGE[1], w)
    min_len = PALETTE_LENGTH_RANGE[0]
    if max_len < min_len:
        return None

    top_len = random.randint(min_len, max_len)
    bottom_len = random.randint(min_len, max_len)
    top_c0, top_colors = _sample_palette_row(w, mid_col, top_len, forbidden)
    bottom_c0, bottom_colors = _sample_palette_row(w, mid_col, bottom_len, forbidden)

    if top_len == bottom_len and np.array_equal(top_colors, bottom_colors):
        return None
    if len(set(top_colors.tolist())) < 2 or len(set(bottom_colors.tolist())) < 2:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[top_row, top_c0 : top_c0 + top_len] = top_colors
    grid[bottom_row, bottom_c0 : bottom_c0 + bottom_len] = bottom_colors
    grid[bar_row, mid_col - 1 : mid_col + 2] = bar_color
    grid[control_row, mid_col] = control_color

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-selected palette ring input")
