from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import get_bg, paste
import random


VALID_DIRECTIONS = {(-1, 0), (1, 0), (0, -1), (0, 1)}


def _marker_direction(g: np.ndarray, body_color: int, r: int, c: int) -> tuple[int, int]:
    H, W = g.shape
    rel: list[tuple[int, int]] = []
    for rr in range(max(0, r - 2), min(H, r + 3)):
        for cc in range(max(0, c - 2), min(W, c + 3)):
            if int(g[rr, cc]) == int(body_color):
                rel.append((int(rr - r), int(cc - c)))

    assert len(rel) == 7
    row_bias = sum(dr for dr, _dc in rel)
    col_bias = sum(dc for _dr, dc in rel)
    assert (row_bias == 0) != (col_bias == 0)

    if row_bias != 0:
        assert abs(row_bias) == 3 and col_bias == 0
        return (-1 if row_bias > 0 else 1, 0)

    assert abs(col_bias) == 3 and row_bias == 0
    return (0, -1 if col_bias > 0 else 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each colored marker from its directional sprite to a grid edge, painting a dotted ray and the whole edge side.

    Input structure:
    - The background is the dominant color.
    - One repeated body color forms small 8-connected directional sprites.
    - Each marker is a singleton non-background, non-body color embedded in one sprite.

    Rule:
    - The body pixels around a marker lie mostly behind it; the opposite direction is the marker's ray direction.
    - Paint every second pixel from the marker along that ray, then paint the entire boundary side reached by the ray.
    - If two painted boundary sides meet at a corner, make that shared corner black.

    Key insight:
    - The same seven-pixel body sprite rotates to encode the direction; its local center-of-mass bias points away from the ray.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    counts = Counter(map(int, g.ravel()))

    body_candidates = [color for color, count in counts.items() if int(color) != bg and int(count) > 1]
    assert len(body_candidates) == 1
    body_color = int(body_candidates[0])

    markers: list[tuple[int, int, int]] = []
    for color, count in counts.items():
        color = int(color)
        if color == bg or color == body_color:
            continue
        assert int(count) == 1
        coords = np.argwhere(g == color)
        assert coords.shape == (1, 2)
        r, c = map(int, coords[0])
        markers.append((color, r, c))

    assert markers
    out = g.copy()
    directions: dict[tuple[int, int], int] = {}

    for color, r, c in markers:
        direction = _marker_direction(g, body_color, r, c)
        assert direction in VALID_DIRECTIONS
        assert direction not in directions
        directions[direction] = int(color)

        dr, dc = direction
        rr, cc = int(r + 2 * dr), int(c + 2 * dc)
        while 0 <= rr < H and 0 <= cc < W:
            assert int(g[rr, cc]) == bg
            out[rr, cc] = int(color)
            rr += int(2 * dr)
            cc += int(2 * dc)

        if direction == (-1, 0):
            assert np.all(g[0, :] == bg)
            out[0, :] = int(color)
        elif direction == (1, 0):
            assert np.all(g[H - 1, :] == bg)
            out[H - 1, :] = int(color)
        elif direction == (0, -1):
            assert np.all(g[:, 0] == bg)
            out[:, 0] = int(color)
        elif direction == (0, 1):
            assert np.all(g[:, W - 1] == bg)
            out[:, W - 1] = int(color)

    if (-1, 0) in directions and (0, -1) in directions:
        out[0, 0] = 0
    if (-1, 0) in directions and (0, 1) in directions:
        out[0, W - 1] = 0
    if (1, 0) in directions and (0, -1) in directions:
        out[H - 1, 0] = 0
    if (1, 0) in directions and (0, 1) in directions:
        out[H - 1, W - 1] = 0

    return out

# --------------- generator (o5x) ---------------

# ── CFG ──────────────────────────────────────────────────────────────────
H_RANGE = (11, 15)         # grid height bounds (inclusive)
W_RANGE = (12, 17)         # grid width bounds (inclusive)
MARKER_COUNT_RANGE = (1, 3)  # number of marker sprites per grid
EDGE_GAP = 1               # min cells between sprite bbox and grid edge
GEN_RETRIES = 300           # max attempts before raising RuntimeError

# ── Sprite geometry ──────────────────────────────────────────────────────
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 7 body-pixel offsets relative to the marker center, keyed by ray direction.
BODY_REL = {
    (-1, 0): [(0, -2), (0, -1), (0, 1), (0, 2), (1, -1), (1, 0), (1, 1)],
    (1, 0): [(-1, -1), (-1, 0), (-1, 1), (0, -2), (0, -1), (0, 1), (0, 2)],
    (0, -1): [(-2, 0), (-1, 0), (-1, 1), (0, 1), (1, 0), (1, 1), (2, 0)],
    (0, 1): [(-2, 0), (-1, -1), (-1, 0), (0, -1), (1, -1), (1, 0), (2, 0)],
}

# Pre-computed sprite bbox extents (min_r, max_r, min_c, max_c) per direction.
_SPRITE_EXTENT: dict[tuple[int, int], tuple[int, int, int, int]] = {}
for _d, _rels in BODY_REL.items():
    _all = _rels + [(0, 0)]
    _SPRITE_EXTENT[_d] = (
        min(dr for dr, _ in _all), max(dr for dr, _ in _all),
        min(dc for _, dc in _all), max(dc for _, dc in _all),
    )


def _sprite_cells(direction: tuple[int, int], r: int, c: int) -> set[tuple[int, int]]:
    """Return the set of (row, col) occupied by the full sprite (body + marker)."""
    return {(r, c)} | {(r + dr, c + dc) for dr, dc in BODY_REL[direction]}


def _ray_cells(H: int, W: int, r: int, c: int, direction: tuple[int, int]) -> set[tuple[int, int]]:
    """Return every cell from the marker outward along *direction* to the grid edge."""
    dr, dc = direction
    cells = set()
    rr, cc = r + dr, c + dc
    while 0 <= rr < H and 0 <= cc < W:
        cells.add((rr, cc))
        rr += dr
        cc += dc
    return cells


def _center_ranges(H: int, W: int, direction: tuple[int, int]) -> tuple[range, range]:
    """Valid marker-center row/col ranges ensuring EDGE_GAP from all grid edges."""
    min_r, max_r, min_c, max_c = _SPRITE_EXTENT[direction]
    return (range(EDGE_GAP - min_r, H - EDGE_GAP - max_r),
            range(EDGE_GAP - min_c, W - EDGE_GAP - max_c))


def _choose_center(
    H: int, W: int, direction: tuple[int, int],
    occupied: set[tuple[int, int]], reserved_rays: set[tuple[int, int]],
) -> tuple[int, int] | None:
    """Pick a random valid marker center that avoids *occupied* and *reserved_rays*."""
    candidates: list[tuple[int, int]] = []
    rows, cols = _center_ranges(H, W, direction)
    for r in rows:
        for c in cols:
            sprite = _sprite_cells(direction, r, c)
            if sprite & occupied or sprite & reserved_rays:
                continue
            ray = _ray_cells(H, W, r, c, direction)
            if ray & occupied or ray & reserved_rays:
                continue
            candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _sprite_patch(
    direction: tuple[int, int], body_color: int, marker_color: int,
) -> tuple[np.ndarray, int, int]:
    """Build a small patch array for the sprite; returns (patch, row_offset, col_offset)."""
    min_r, max_r, min_c, max_c = _SPRITE_EXTENT[direction]
    patch = np.full((max_r - min_r + 1, max_c - min_c + 1), -1, dtype=int)
    for dr, dc in BODY_REL[direction]:
        patch[dr - min_r, dc - min_c] = body_color
    patch[-min_r, -min_c] = marker_color
    return patch, min_r, min_c


def _stamp_sprite(
    grid: np.ndarray, keep: np.ndarray, direction: tuple[int, int],
    r: int, c: int, body_color: int, marker_color: int,
) -> None:
    """Render sprite onto *grid* and mark its cells in *keep*."""
    patch, off_r, off_c = _sprite_patch(direction, body_color, marker_color)
    paste(grid, patch, r + off_r, c + off_c, transparent=-1)
    for rr, cc in _sprite_cells(direction, r, c):
        keep[rr, cc] = True


def _paint_output(
    out: np.ndarray, marker_color: int, r: int, c: int, direction: tuple[int, int],
) -> None:
    """Paint the dotted ray and fill the target boundary row/col."""
    H, W = out.shape
    dr, dc = direction
    rr, cc = r + 2 * dr, c + 2 * dc
    while 0 <= rr < H and 0 <= cc < W:
        out[rr, cc] = marker_color
        rr += 2 * dr
        cc += 2 * dc

    if direction == (-1, 0):
        out[0, :] = marker_color
    elif direction == (1, 0):
        out[H - 1, :] = marker_color
    elif direction == (0, -1):
        out[:, 0] = marker_color
    else:
        out[:, W - 1] = marker_color


def _black_shared_corners(out: np.ndarray, directions: set[tuple[int, int]]) -> None:
    """Set corner to black where two painted boundary sides meet."""
    H, W = out.shape
    if (-1, 0) in directions and (0, -1) in directions:
        out[0, 0] = 0
    if (-1, 0) in directions and (0, 1) in directions:
        out[0, W - 1] = 0
    if (1, 0) in directions and (0, -1) in directions:
        out[H - 1, 0] = 0
    if (1, 0) in directions and (0, 1) in directions:
        out[H - 1, W - 1] = 0


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)

    colors = list(range(1, 10))
    bg = random.choice(colors)
    remaining = [c for c in colors if c != bg]
    body_color = random.choice(remaining)
    remaining = [c for c in remaining if c != body_color]

    n_markers = random.randint(*MARKER_COUNT_RANGE)
    marker_colors = random.sample(remaining, n_markers)
    directions = random.sample(DIRS, n_markers)

    occupied: set[tuple[int, int]] = set()
    reserved_rays: set[tuple[int, int]] = set()
    placements: list[tuple[int, int, int, tuple[int, int]]] = []

    for marker_color, direction in zip(marker_colors, directions):
        center = _choose_center(H, W, direction, occupied, reserved_rays)
        if center is None:
            return None
        r, c = center
        placements.append((marker_color, r, c, direction))
        occupied.update(_sprite_cells(direction, r, c))
        reserved_rays.update(_ray_cells(H, W, r, c, direction))

    out = np.full((H, W), bg, dtype=int)
    keep = np.zeros((H, W), dtype=bool)
    for marker_color, r, c, direction in placements:
        _stamp_sprite(out, keep, direction, r, c, body_color, marker_color)
    for marker_color, r, c, direction in placements:
        _paint_output(out, marker_color, r, c, direction)
    _black_shared_corners(out, {d for _, _, _, d in placements})

    inp = out.copy()
    inp[~keep] = bg

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 13f06aa5")
