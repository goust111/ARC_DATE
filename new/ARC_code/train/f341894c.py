from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg
import random


def _is_border(r: int, c: int, height: int, width: int) -> bool:
    return r == 0 or r == height - 1 or c == 0 or c == width - 1


def _border_marker_color(g: np.ndarray, bg: int) -> int:
    height, width = map(int, g.shape)
    candidates = []
    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == color)
        if all(_is_border(int(r), int(c), height, width) for r, c in coords):
            candidates.append(int(color))

    assert len(candidates) == 1, "expected exactly one foreground color used only as border markers"
    return candidates[0]


def _inward_dirs(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    dirs = []
    if r == 0:
        dirs.append((1, 0))
    if r == height - 1:
        dirs.append((-1, 0))
    if c == 0:
        dirs.append((0, 1))
    if c == width - 1:
        dirs.append((0, -1))

    assert dirs, "marker must be on the grid border"
    return dirs


def _colored_cells_on_ray(
    g: np.ndarray,
    r: int,
    c: int,
    dr: int,
    dc: int,
    colors: set[int],
) -> list[tuple[int, int, int]]:
    height, width = map(int, g.shape)
    cells = []
    step = 1
    rr = r + dr
    cc = c + dc
    while 0 <= rr < height and 0 <= cc < width:
        if int(g[rr, cc]) in colors:
            cells.append((step, rr, cc))
        rr += dr
        cc += dc
        step += 1
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorient each two-pixel blue/pink segment so it reads pink then blue when viewed from its border marker.

    Input structure
    - The background is dominant, with one border-only marker color and blue/pink foreground segments.
    - Every marker lies on the outer border and points inward along its row or column.
    - Along exactly one inward ray from each marker is a single adjacent two-pixel blue/pink segment.

    Rule
    - For each marker, look inward perpendicular to the border side containing the marker.
    - Rewrite the two blue/pink pixels on that ray so the nearer one is pink and the farther one is blue.
    - Keep marker pixels, background pixels, and all other pixels unchanged.

    Key insight
    - The marker gives a local viewing direction; the segment's required order is relative to that marker, not to global row or column order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 1 and width > 1, "expected a nontrivial grid"

    bg = get_bg(g)
    marker = _border_marker_color(g, bg)
    pair_colors = set(colors_present(g, exclude={bg, marker}))
    assert pair_colors == {Color.BLUE, Color.PINK}, "expected blue and pink segment colors"

    planned: dict[tuple[int, int], int] = {}
    marker_coords = np.argwhere(g == marker)
    assert marker_coords.size > 0, "expected at least one border marker"

    for r_raw, c_raw in marker_coords:
        r = int(r_raw)
        c = int(c_raw)
        valid_rays = []
        for dr, dc in _inward_dirs(r, c, height, width):
            cells = _colored_cells_on_ray(g, r, c, dr, dc, pair_colors)
            if cells:
                valid_rays.append(cells)

        assert len(valid_rays) == 1, "expected each marker to govern exactly one row or column segment"
        cells = valid_rays[0]
        assert len(cells) == 2, "expected exactly two blue/pink pixels on each marker ray"
        assert cells[1][0] == cells[0][0] + 1, "expected marker ray pixels to form an adjacent segment"

        for (_step, rr, cc), color in zip(cells, (Color.PINK, Color.BLUE)):
            previous = planned.get((rr, cc))
            assert previous is None or previous == color, "conflicting marker segment orientations"
            planned[(rr, cc)] = color

    out = g.copy()
    for (r, c), color in planned.items():
        out[r, c] = color
    return out

# --------------- generator (o5x) ---------------
# Canvas, marker, and side-selection bounds.
GRID_SIZES = [8, 10, 12, 14, 16, 18, 9, 11, 13, 15, 17]
MIN_MARKERS, MAX_MARKERS, SIZE_MARKER_LIMIT_OFFSET = 4, 14, 4
SIDE_COUNT_MIN, SIDE_COUNT_MAX = 1, 2
HORIZONTAL_SIDES, VERTICAL_SIDES = ("L", "R"), ("T", "B")
# Segment placement, input perturbation, and retry controls.
SEGMENT_MIN_DISTANCE, SEGMENT_STOP_BORDER_MARGIN = 2, 2
MIN_CORRECT_RATE, MAX_CORRECT_RATE = 0.3, 0.7
MIN_CORRECT_SEGMENTS, GEN_RETRIES = 1, 300
# Task colors.
BG_COLOR, MARKER_COLOR = Color.TEAL, Color.ORANGE
NEAR_COLOR, FAR_COLOR = Color.PINK, Color.BLUE

def _gen_allocate(total: int, bins: int) -> list[int]:
    """Randomly split total marker positions across the chosen sides."""
    cuts = sorted(random.sample(range(1, total), bins - 1))
    parts = [a - b for a, b in zip(cuts + [total], [0] + cuts)]
    random.shuffle(parts)
    return parts


def _gen_segment_options(n: int, blocked: set[int], reverse: bool = False) -> list[int]:
    """Return valid near-cell distances for a two-pixel segment on one ray."""
    starts = []
    for dist in range(SEGMENT_MIN_DISTANCE, n - SEGMENT_STOP_BORDER_MARGIN):
        a = n - 1 - dist if reverse else dist
        b = n - 1 - (dist + 1) if reverse else dist + 1
        if a not in blocked and b not in blocked:
            starts.append(dist)
    return starts


def _gen_side_rays(n: int, sides: list[str], quotas: list[int], positions: list[int], horizontal: bool):
    """Build marker rays for one axis; positions are already shuffled."""
    rays = []
    pos = 0
    for side, count in zip(sides, quotas):
        chunk = positions[pos : pos + count]
        if horizontal:
            rays.extend((value, 0, 0, 1) if side == "L" else (value, n - 1, 0, -1) for value in chunk)
        else:
            rays.extend((0, value, 1, 0) if side == "T" else (n - 1, value, -1, 0) for value in chunk)
        pos += count
    return rays


def _gen_layout() -> tuple | None:
    """Choose grid size, marker sides, border positions, and inward rays."""
    n = random.choice(GRID_SIZES)
    k = random.randint(MIN_MARKERS, min(MAX_MARKERS, 2 * n - SIZE_MARKER_LIMIT_OFFSET))
    h_side_count = random.randint(SIDE_COUNT_MIN, SIDE_COUNT_MAX)
    v_side_count = random.randint(SIDE_COUNT_MIN, SIDE_COUNT_MAX)
    h_sides = random.sample(HORIZONTAL_SIDES, h_side_count)
    v_sides = random.sample(VERTICAL_SIDES, v_side_count)
    h_min, v_min = h_side_count, v_side_count
    h_max = min(n - 2, k - v_min)
    h_choices = [h for h in range(h_min, h_max + 1) if v_min <= k - h <= n - 2]
    if not h_choices:
        return None
    h_count = random.choice(h_choices)
    v_count = k - h_count
    h_rows = random.sample(range(1, n - 1), h_count)
    v_cols = random.sample(range(1, n - 1), v_count)
    h_quotas = _gen_allocate(h_count, h_side_count)
    v_quotas = _gen_allocate(v_count, v_side_count)
    random.shuffle(h_rows)
    rays = _gen_side_rays(n, h_sides, h_quotas, h_rows, horizontal=True)
    random.shuffle(v_cols)
    rays.extend(_gen_side_rays(n, v_sides, v_quotas, v_cols, horizontal=False))
    return n, rays, h_rows, v_cols


def _gen_input_from_output(out: np.ndarray, segments: list) -> np.ndarray:
    """Flip a sampled subset of solved segments to create the input grid."""
    inp = out.copy()
    k = len(segments)
    min_correct = max(MIN_CORRECT_SEGMENTS, int(np.ceil(MIN_CORRECT_RATE * k)))
    max_correct = max(min_correct, int(np.floor(MAX_CORRECT_RATE * k)))
    n_correct = random.randint(min_correct, max_correct)
    correct = set(random.sample(range(k), n_correct))
    for idx, (near, far) in enumerate(segments):
        if idx not in correct:
            inp[near] = FAR_COLOR
            inp[far] = NEAR_COLOR
    return inp


def _gen_attempt() -> np.ndarray | None:
    layout = _gen_layout()
    if layout is None:
        return None
    n, rays, h_rows, v_cols = layout
    out = np.full((n, n), BG_COLOR, dtype=int)
    for r, c, _dr, _dc in rays:
        out[r, c] = MARKER_COLOR
    segments = []
    blocked_rows = set(h_rows)
    blocked_cols = set(v_cols)
    for r, c, dr, dc in rays:
        if dr == 0:
            options = _gen_segment_options(n, blocked_cols, reverse=dc < 0)
        else:
            options = _gen_segment_options(n, blocked_rows, reverse=dr < 0)
        if not options:
            return None
        dist = random.choice(options)
        near = (r + dr * dist, c + dc * dist)
        far = (r + dr * (dist + 1), c + dc * (dist + 1))
        if out[near] != BG_COLOR or out[far] != BG_COLOR:
            return None
        out[near] = NEAR_COLOR
        out[far] = FAR_COLOR
        segments.append((near, far))
    inp = _gen_input_from_output(out, segments)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid f341894c input")
