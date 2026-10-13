import random

import numpy as np


BG = 8
PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)
HALF_HEIGHT_RANGE = (4, 15)
HALF_WIDTH_RANGE = (4, 15)
ACTIVE_PALETTE_SIZE_RANGE = (3, 6)
MAX_MARKERS_PER_QUADRANT = 16
MAX_GENERATION_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).reshape(-1)
    counts = np.bincount(values, minlength=10)
    if counts[BG] > 0 and counts[BG] == counts.max():
        return BG
    return int(counts.argmax())


def _quadrant_index(r: int, c: int, half_h: int, half_w: int) -> int:
    return (2 if r >= half_h else 0) + (1 if c >= half_w else 0)


def _quadrant_bounds(shape: tuple[int, int], quadrant: int) -> tuple[int, int, int, int]:
    h, w = shape
    half_h = h // 2
    half_w = w // 2
    top = quadrant < 2
    left = quadrant % 2 == 0
    r0 = 0 if top else half_h
    r1 = half_h if top else h
    c0 = 0 if left else half_w
    c1 = half_w if left else w
    return r0, r1, c0, c1


def _pack_positions(shape: tuple[int, int], quadrant: int) -> list[tuple[int, int]]:
    h, w = shape
    half_h = h // 2
    half_w = w // 2
    if quadrant < 2:
        rows = range(0, half_h)
    else:
        rows = range(h - 1, half_h - 1, -1)
    if quadrant % 2 == 0:
        cols = range(0, half_w)
    else:
        cols = range(w - 1, half_w - 1, -1)
    return [(r, c) for r in rows for c in cols]


def _choose_count(half_h: int, half_w: int) -> int:
    capacity = half_h * half_w
    limit = min(MAX_MARKERS_PER_QUADRANT, max(1, (capacity - 1) // 2))
    assert limit >= 1, "quadrant must have at least one usable cell"
    if limit > half_w and random.random() < 0.5:
        return random.randint(half_w + 1, limit)
    return random.randint(1, min(limit, half_w))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move scattered quadrant markers into compact corner stacks based on their quadrant membership.

    Input structure:
    - The background is Azure, and the grid has even height and width so it splits into four equal quadrants.
    - Every non-background cell is a colored single-pixel marker somewhere inside one quadrant.
    - There is no visible divider; quadrant membership is determined only by the marker's row and column halves.

    Rule:
    - Read the markers in each quadrant in the original row-major order.
    - Move upper-left markers into the upper-left corner, filling left-to-right across rows from the top.
    - Move upper-right markers into the upper-right corner, filling right-to-left across rows from the top.
    - Move lower-left markers into the lower-left corner, filling left-to-right across rows from the bottom upward.
    - Move lower-right markers into the lower-right corner, filling right-to-left across rows from the bottom upward, and clear every original marker location to background.

    Key insight:
    - Each marker's quadrant chooses the destination corner, while the original scan order only determines the stack order inside that corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h % 2 == 0 and w % 2 == 0, "grid must split into equal quadrants"

    bg = _background_color(g)
    half_h = h // 2
    half_w = w // 2
    groups: list[list[int]] = [[], [], [], []]

    for r in range(h):
        for c in range(w):
            value = int(g[r, c])
            if value == bg:
                continue
            groups[_quadrant_index(r, c, half_h, half_w)].append(value)

    out = np.full((h, w), bg, dtype=int)
    quadrant_capacity = half_h * half_w
    for quadrant, colors in enumerate(groups):
        assert len(colors) <= quadrant_capacity, "quadrant stack exceeds its capacity"
        for color, (r, c) in zip(colors, _pack_positions((h, w), quadrant)):
            out[r, c] = int(color)

    return out


def _gen_attempt() -> np.ndarray | None:
    half_h = random.randint(*HALF_HEIGHT_RANGE)
    half_w = random.randint(*HALF_WIDTH_RANGE)
    h = 2 * half_h
    w = 2 * half_w
    grid = np.full((h, w), BG, dtype=int)

    counts = [_choose_count(half_h, half_w) for _ in range(4)]
    total_markers = sum(counts)
    palette_size = min(random.randint(*ACTIVE_PALETTE_SIZE_RANGE), total_markers, len(PALETTE))
    active_palette = random.sample(PALETTE, palette_size)

    colors = active_palette.copy()
    while len(colors) < total_markers:
        colors.append(random.choice(active_palette))
    random.shuffle(colors)
    color_index = 0

    for quadrant, count in enumerate(counts):
        reserved = set(_pack_positions((h, w), quadrant)[:count])
        r0, r1, c0, c1 = _quadrant_bounds((h, w), quadrant)
        candidates = [
            (r, c)
            for r in range(r0, r1)
            for c in range(c0, c1)
            if (r, c) not in reserved
        ]
        if len(candidates) < count:
            return None

        chosen = random.sample(candidates, count)
        for r, c in chosen:
            grid[r, c] = colors[color_index]
            color_index += 1

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
