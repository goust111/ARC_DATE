from __future__ import annotations

import numpy as np

from arc_utils import bbox, colors_present, get_bg, paste


def _ring_projection(
    r: int,
    c: int,
    block_bbox: tuple[int, int, int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = block_bbox

    if r < r0:
        nr = r0 - 1
    elif r > r1:
        nr = r1 + 1
    else:
        nr = r

    if c < c0:
        nc = c0 - 1
    elif c > c1:
        nc = c1 + 1
    else:
        nc = c

    return int(nr), int(nc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each marker pixel straight inward until it touches the one-cell ring around the solid 2x2 block.

    Input structure
    - The background contains exactly two foreground colors.
    - One foreground color forms a unique filled 2x2 block.
    - The other foreground color appears as marker pixels aligned with the block by row, column, or 45-degree diagonal.

    Rule
    - Keep the 2x2 block fixed.
    - Clear every marker from its original position.
    - Project each marker along its row, column, or diagonal toward the block, stopping in the adjacent ring cell on the same side/corner of the 2x2 block.

    Key insight
    - The 2x2 block is the anchor, and every marker is already on one of the eight rays extending from the block's surrounding ring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) == 2, "expected two foreground colors"

    block_candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in foreground_colors:
        mask = g == int(color)
        box = bbox(mask)
        assert box is not None, "foreground color must be present"
        r0, c0, r1, c1 = box
        if (
            r1 - r0 + 1 == 2
            and c1 - c0 + 1 == 2
            and int(np.count_nonzero(mask)) == 4
            and np.all(g[r0 : r1 + 1, c0 : c1 + 1] == int(color))
        ):
            block_candidates.append((int(color), box))

    assert len(block_candidates) == 1, "expected one filled 2x2 foreground block"
    block_color, block_bbox = block_candidates[0]
    r0, c0, r1, c1 = block_bbox

    marker_colors = [int(color) for color in foreground_colors if int(color) != block_color]
    assert len(marker_colors) == 1, "expected one marker color"
    marker_color = marker_colors[0]
    marker_coords = np.argwhere(g == marker_color)
    assert marker_coords.size > 0, "expected at least one marker"

    out = g.copy()
    out[g == marker_color] = bg

    height, width = map(int, g.shape)
    destinations: set[tuple[int, int]] = set()
    for rr, cc in marker_coords:
        r = int(rr)
        c = int(cc)
        assert not (r0 <= r <= r1 and c0 <= c <= c1), "marker cannot lie inside the block"

        nr, nc = _ring_projection(r, c, block_bbox)
        row_aligned = r0 <= r <= r1
        col_aligned = c0 <= c <= c1
        if not row_aligned and not col_aligned:
            assert abs(r - nr) == abs(c - nc), "diagonal marker must lie on a 45-degree ray"

        assert 0 <= nr < height and 0 <= nc < width, "projected marker must stay inside the grid"
        assert int(g[nr, nc]) != block_color, "projected marker cannot overwrite the block"
        assert (nr, nc) not in destinations, "marker destinations must be unique"
        destinations.add((nr, nc))
        out[nr, nc] = marker_color

    return out

# --------------- generator (o5x) ---------------

_GEN_RNG = np.random.default_rng()

# Canvas, palette, and block placement.
GRID_SHAPE, BACKGROUND_COLOR = (10, 10), 0
NON_BACKGROUND_COLORS = np.arange(1, 10)
BLOCK_SHAPE, BLOCK_ANCHOR_RANGE = (2, 2), (2, 7)

# Marker count, ray distance, and retry policy.
MARKER_COUNT_RANGE, RAY_DISTANCE_RANGE = (3, 5), (1, 6)
GENERATION_RETRIES = 300

# Ring destinations as offsets from the 2x2 block top-left: (r, c, dr, dc).
RING_OFFSETS = [
    (-1, -1, -1, -1), (-1, 0, -1, 0), (-1, 1, -1, 0), (-1, 2, -1, 1),
    (0, -1, 0, -1), (1, -1, 0, -1), (0, 2, 0, 1), (1, 2, 0, 1),
    (2, -1, 1, -1), (2, 0, 1, 0), (2, 1, 1, 0), (2, 2, 1, 1),
]


def _gen_max_ray_distance(r: int, c: int, dr: int, dc: int) -> int:
    height, width = GRID_SHAPE
    row_room = height if dr == 0 else (r if dr < 0 else height - 1 - r)
    col_room = width if dc == 0 else (c if dc < 0 else width - 1 - c)
    return int(min(RAY_DISTANCE_RANGE[1], row_room, col_room))


def _gen_marker_paths(ring_specs: list[tuple[int, int, int, int]]) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    """Sample unique marker destinations and their distant source cells."""
    min_dist, _max_dist = RAY_DISTANCE_RANGE
    marker_count = int(_GEN_RNG.integers(*MARKER_COUNT_RANGE))
    chosen = _GEN_RNG.choice(len(ring_specs), size=marker_count, replace=False)

    source_positions: set[tuple[int, int]] = set()
    destinations: list[tuple[int, int]] = []
    height, width = GRID_SHAPE
    for idx in chosen:
        rr, cc, dr, dc = ring_specs[int(idx)]
        max_dist = _gen_max_ray_distance(rr, cc, dr, dc)
        if max_dist < min_dist:
            return None
        dist = int(_GEN_RNG.integers(min_dist, max_dist + 1))
        mr = rr + dr * dist
        mc = cc + dc * dist
        if not (0 <= mr < height and 0 <= mc < width):
            return None
        if (mr, mc) in source_positions or (mr, mc) in destinations:
            return None
        destinations.append((rr, cc))
        source_positions.add((mr, mc))
    return destinations, source_positions


def _gen_input_from_output(
    out: np.ndarray,
    destinations: list[tuple[int, int]],
    source_positions: set[tuple[int, int]],
    marker_color: int,
) -> np.ndarray | None:
    """Move generated markers from ring destinations back to source cells."""
    inp = out.copy()
    for rr, cc in destinations:
        inp[rr, cc] = BACKGROUND_COLOR
    for rr, cc in source_positions:
        if int(inp[rr, cc]) != BACKGROUND_COLOR:
            return None
        inp[rr, cc] = marker_color
    return inp


def _gen_attempt() -> np.ndarray | None:
    colors = _GEN_RNG.choice(NON_BACKGROUND_COLORS, size=2, replace=False)
    block_color = int(colors[0])
    marker_color = int(colors[1])

    r0 = int(_GEN_RNG.integers(*BLOCK_ANCHOR_RANGE))
    c0 = int(_GEN_RNG.integers(*BLOCK_ANCHOR_RANGE))
    ring_specs = [(r0 + ro, c0 + co, dr, dc) for ro, co, dr, dc in RING_OFFSETS]
    marker_paths = _gen_marker_paths(ring_specs)
    if marker_paths is None:
        return None
    destinations, source_positions = marker_paths

    out = np.full(GRID_SHAPE, BACKGROUND_COLOR, dtype=int)
    block = np.full(BLOCK_SHAPE, block_color, dtype=int)
    paste(out, block, r0, c0)
    for rr, cc in destinations:
        out[rr, cc] = marker_color

    inp = _gen_input_from_output(out, destinations, source_positions, marker_color)
    if inp is None:
        return None

    try:
        return inp if np.array_equal(main(inp), out) else None
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a48eeaf7 input")
