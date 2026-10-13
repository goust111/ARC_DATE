from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
import random


def _singleton_markers(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    markers = []
    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == color)
        if coords.shape[0] == 1:
            r, c = map(int, coords[0])
            markers.append((r, c, int(color)))

    assert len(markers) == 2, "expected exactly two singleton marker colors"
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored ray endpoint inward to touch the singleton marker it points at.

    Input structure
    - The background contains exactly two singleton marker colors.
    - Every other foreground pixel is on the same row or column as exactly one marker.
    - The landing cells adjacent to those markers are empty and do not conflict.

    Rule
    - Keep the two singleton markers fixed.
    - For each non-marker foreground pixel, find the unique marker sharing its row or column.
    - Clear the original pixel and repaint it in the cell immediately next to that marker along the same ray.

    Key insight
    - The scattered pixels are already directional endpoints of cardinal rays; the output compresses each ray to its nearest cell around the ray's marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    markers = _singleton_markers(g, bg)
    marker_colors = {color for _, _, color in markers}

    moves: list[tuple[int, int, int, int, int]] = []
    planned: dict[tuple[int, int], int] = {}
    for r_raw, c_raw in np.argwhere(g != bg):
        r = int(r_raw)
        c = int(c_raw)
        color = int(g[r, c])
        if color in marker_colors:
            continue

        aligned = [(mr, mc) for mr, mc, _ in markers if (r == mr) ^ (c == mc)]
        assert len(aligned) == 1, "each moving pixel must align with exactly one marker"
        mr, mc = aligned[0]

        dr = (r > mr) - (r < mr)
        dc = (c > mc) - (c < mc)
        nr = mr + dr
        nc = mc + dc
        assert 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1], "landing cell must be in bounds"
        assert (nr, nc) == (r, c) or int(g[nr, nc]) == bg, "landing cell must be empty"
        assert (nr, nc) not in planned, "two rays cannot land on the same cell"

        moves.append((r, c, nr, nc, color))
        planned[(nr, nc)] = color

    assert moves, "expected at least one moving pixel"

    out = g.copy()
    for r, c, _, _, _ in moves:
        out[r, c] = bg
    for _, _, nr, nc, color in moves:
        out[nr, nc] = color
    return out

# --------------- generator (o5x) ---------------

# Canvas and marker bounds.
GRID_SIZE, BG_COLOR, MARKER_MARGIN = 15, 0, 2
MARKER_RANGE = range(MARKER_MARGIN, GRID_SIZE - MARKER_MARGIN)
MARKER_CELLS = [(r, c) for r in MARKER_RANGE for c in MARKER_RANGE]
# Colors and ray sampling.
MARKER_COLORS, RAY_COLORS = (1, 2), (7, 3)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TOTAL_RAY_RANGE, RAYS_PER_MARKER_RANGE = (5, 8), (2, 4)
MIN_ENDPOINT_DIST, MAX_GENERATION_ATTEMPTS = 2, 300


def _gen_marker_pair():
    r1, c1 = random.choice(MARKER_CELLS)
    partners = [
        (r, c) for r, c in MARKER_CELLS
        if r != r1 and c != c1 and not (abs(r - r1) == 1 and abs(c - c1) == 1)
    ]
    return (r1, c1), random.choice(partners)


def _gen_endpoint(marker, other_marker, direction, used):
    mr, mc = marker
    orow, ocol = other_marker
    dr, dc = direction
    edge = GRID_SIZE - 1
    max_dist = (
        mr if dr < 0 else edge - mr if dr > 0 else mc if dc < 0 else edge - mc
    )
    candidates = []
    for dist in range(MIN_ENDPOINT_DIST, max_dist + 1):
        r = mr + dr * dist
        c = mc + dc * dist
        if r == orow or c == ocol or (r, c) in used:
            continue
        candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _sample_ray_directions():
    lo, hi = RAYS_PER_MARKER_RANGE
    total_rays = random.randint(*TOTAL_RAY_RANGE)
    valid_counts = [n for n in range(lo, hi + 1) if lo <= total_rays - n <= hi]
    ray1_count = random.choice(valid_counts)
    return random.sample(DIRECTIONS, ray1_count), random.sample(DIRECTIONS, total_rays - ray1_count)


def _build_output_and_rays(marker1, marker2, dirs1, dirs2):
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    out[marker1] = MARKER_COLORS[0]
    out[marker2] = MARKER_COLORS[1]
    rays, landings = [], set()
    groups = (
        (marker1, marker2, dirs1, RAY_COLORS[0]),
        (marker2, marker1, dirs2, RAY_COLORS[1]),
    )
    for marker, other_marker, dirs, color in groups:
        mr, mc = marker
        for dr, dc in dirs:
            landing = (mr + dr, mc + dc)
            if landing in landings or out[landing] != BG_COLOR:
                return None
            out[landing] = color
            landings.add(landing)
            rays.append((marker, other_marker, (dr, dc), landing, color))
    return out, rays


def _derive_input_from_output(out, rays, marker1, marker2):
    inp = out.copy()
    used = {marker1, marker2}
    random.shuffle(rays)
    for marker, other_marker, direction, landing, color in rays:
        inp[landing] = BG_COLOR
        endpoint = _gen_endpoint(marker, other_marker, direction, used)
        if endpoint is None:
            return None
        inp[endpoint] = color
        used.add(endpoint)
    return inp


def _gen_attempt() -> np.ndarray | None:
    marker1, marker2 = _gen_marker_pair()
    dirs1, dirs2 = _sample_ray_directions()

    built = _build_output_and_rays(marker1, marker2, dirs1, dirs2)
    if built is None:
        return None
    out, rays = built

    inp = _derive_input_from_output(out, rays, marker1, marker2)
    if inp is None:
        return None
    for mr, mc in (marker1, marker2):
        for dr, dc in DIRECTIONS:
            if inp[mr + dr, mc + dc] != BG_COLOR:
                return None
    try:
        if not np.array_equal(main(inp), out):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid ae3edfdc input")
