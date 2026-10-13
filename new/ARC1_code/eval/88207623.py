from __future__ import annotations

import numpy as np

from arc_utils import Color, Color as ArcColor, components_mask, find_components, get_bg, paste as gen_paste, random_free_location as gen_random_free_location
import math
import random


def _vertical_axis(component: dict) -> tuple[int, set[int]]:
    coords = component["coords"]
    cols = set(map(int, coords[:, 1]))
    assert len(cols) == 1, "red component must be a vertical segment"
    axis_col = cols.pop()

    rows = sorted(map(int, coords[:, 0]))
    assert rows == list(range(rows[0], rows[-1] + 1)), "red segment must be contiguous"
    return axis_col, set(rows)


def _yellow_source_for_axis(
    yellow_objects: list[np.ndarray],
    axis_col: int,
    axis_rows: set[int],
    used_yellow: set[int],
) -> tuple[int, np.ndarray]:
    candidates = []
    for idx, coords in enumerate(yellow_objects):
        if idx in used_yellow:
            continue

        cols = coords[:, 1]
        if np.all(cols < axis_col):
            side = -1
        elif np.all(cols > axis_col):
            side = 1
        else:
            continue

        touches_axis = any(
            int(r) in axis_rows and int(c) - side == axis_col
            for r, c in coords
        )
        if touches_axis:
            candidates.append((idx, coords))

    assert len(candidates) == 1, "each red segment must touch one yellow source object"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each yellow object across its red vertical segment and paint the reflected copy with the marker color already inside the mirror footprint.

    Input structure
    - The background is the dominant color.
    - Red pixels form one or more contiguous vertical mirror-axis segments.
    - Each red segment has exactly one 8-connected yellow object touching it on one side.
    - The opposite-side reflected footprint contains exactly one non-background marker pixel.

    Rule
    - For each red segment, reflect every pixel of its yellow object horizontally across the red column.
    - Use the marker pixel in that reflected footprint as the paint color for the whole reflected object.
    - Preserve the original grid and add the colored reflected copies.

    Key insight
    - The lone marker is not a destination by distance; it is a color instruction embedded in the mirror image of the adjacent yellow mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = get_bg(g)
    red_segments = find_components(g, bg=bg, colors=[Color.RED], connectivity=4)
    yellow_objects = components_mask(g == Color.YELLOW, connectivity=8)
    assert red_segments, "expected at least one red mirror segment"

    out = g.copy()
    used_yellow: set[int] = set()
    for segment in red_segments:
        axis_col, axis_rows = _vertical_axis(segment)
        source_idx, source_coords = _yellow_source_for_axis(
            yellow_objects, axis_col, axis_rows, used_yellow
        )
        used_yellow.add(source_idx)
        assert all(int(r) in axis_rows for r, _c in source_coords), (
            "yellow source rows must align with its red segment"
        )

        reflected = [(int(r), 2 * axis_col - int(c)) for r, c in source_coords]
        assert all(
            0 <= r < g.shape[0] and 0 <= c < g.shape[1]
            for r, c in reflected
        ), "reflected source must stay inside the grid"

        target_values = [int(g[r, c]) for r, c in reflected]
        marker_values = [
            value
            for value in target_values
            if value not in {bg, Color.RED, Color.YELLOW}
        ]
        assert len(marker_values) == 1, "reflected footprint must contain one marker pixel"
        marker_color = marker_values[0]

        for r, c in reflected:
            assert int(g[r, c]) in {bg, marker_color}, (
                "reflected footprint must contain only background and its marker"
            )
            out[r, c] = marker_color

    assert len(used_yellow) == len(red_segments), "each red segment must use one yellow object"
    return out

# --------------- generator (o5x) ---------------

# Canvas/object counts.
_GEN_GRID_ROWS = (14, 16)
_GEN_GRID_COLS = (16, 18)
_GEN_SEGMENT_COUNT = (2, 4)

# Yellow shape sampling.
_GEN_AXIS_HEIGHT = (3, 6)
_GEN_YELLOW_WIDTH = (2, 5)
_GEN_YELLOW_DENSITY = (0.40, 0.75)
_GEN_LADDER_PROBABILITY = 0.45
_GEN_WALK_STEPS = (-1, 0, 1)

# Marker, placement, and retry sampling.
_GEN_MARKER_COLORS = [
    ArcColor.BLUE,
    ArcColor.GREEN,
    ArcColor.GRAY,
    ArcColor.PINK,
    ArcColor.ORANGE,
    ArcColor.TEAL,
    ArcColor.MAROON,
]
_GEN_MARKER_DISTANCE = (2, 4)
_GEN_PLACEMENT_PADDING = 1
_GEN_PLACEMENT_CONNECTIVITY = 8
_GEN_MAX_ATTEMPTS = 300


def _gen_bridge_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    bridge_r = random.randrange(h)
    mask[bridge_r, :] = True

    c = random.randrange(w)
    for r in range(bridge_r - 1, -1, -1):
        c = min(w - 1, max(0, c + random.choice(_GEN_WALK_STEPS)))
        mask[r, c] = True

    c = random.randrange(w)
    for r in range(bridge_r + 1, h):
        c = min(w - 1, max(0, c + random.choice(_GEN_WALK_STEPS)))
        mask[r, c] = True

    return mask


def _gen_ladder_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start_c = random.choice([0, w - 1])
    step = 1 if start_c == 0 else -1
    moves = [step] * (w - 1) + [0] * (h - w)
    random.shuffle(moves)

    c = start_c
    mask[0, c] = True
    for r, move in enumerate(moves, start=1):
        c += move
        mask[r, c] = True

    return mask


def _gen_growth_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    adjacent = np.zeros_like(mask, dtype=bool)
    adjacent[1:, :] |= mask[:-1, :]
    adjacent[:-1, :] |= mask[1:, :]
    adjacent[:, 1:] |= mask[:, :-1]
    adjacent[:, :-1] |= mask[:, 1:]
    adjacent[1:, 1:] |= mask[:-1, :-1]
    adjacent[1:, :-1] |= mask[:-1, 1:]
    adjacent[:-1, 1:] |= mask[1:, :-1]
    adjacent[:-1, :-1] |= mask[1:, 1:]

    return [(int(r), int(c)) for r, c in np.argwhere(adjacent & ~mask)]


def _gen_yellow_mask(h: int, w: int) -> np.ndarray:
    if h >= w and random.random() < _GEN_LADDER_PROBABILITY:
        mask = _gen_ladder_mask(h, w)
    else:
        mask = _gen_bridge_mask(h, w)

    area = h * w
    lo = max(int(math.ceil(_GEN_YELLOW_DENSITY[0] * area)), int(mask.sum()))
    hi = int(math.floor(_GEN_YELLOW_DENSITY[1] * area))
    if lo > hi:
        return mask

    target = random.randint(lo, hi)
    while int(mask.sum()) < target:
        candidates = _gen_growth_candidates(mask)
        r, c = random.choice(candidates)
        mask[r, c] = True

    return mask


def _gen_unit(marker_color: int) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int, int]]:
    h = random.randint(*_GEN_AXIS_HEIGHT)
    w = random.randint(*_GEN_YELLOW_WIDTH)
    axis_c = w
    yellow = _gen_yellow_mask(h, w)
    unit = np.zeros((h, 2 * w + 1), dtype=int)
    unit[:, axis_c] = ArcColor.RED

    coords = np.argwhere(yellow)
    rows = coords[:, 0]
    src_cols = coords[:, 1]
    dst_cols = 2 * axis_c - src_cols
    unit[rows, src_cols] = ArcColor.YELLOW
    unit[rows, dst_cols] = marker_color
    reflected = [(int(r), int(c)) for r, c in zip(rows, dst_cols)]

    marker_choices = [
        (r, c)
        for r, c in reflected
        if _GEN_MARKER_DISTANCE[0] <= abs(c - axis_c) <= _GEN_MARKER_DISTANCE[1]
    ]
    if not marker_choices:
        marker_choices = reflected
    marker_r, marker_c = random.choice(marker_choices)

    if random.choice([False, True]):
        unit = np.fliplr(unit)
        reflected = [(r, unit.shape[1] - 1 - c) for r, c in reflected]
        marker_c = unit.shape[1] - 1 - marker_c

    return unit, reflected, (marker_r, marker_c, marker_color)


def _gen_place_units(
    out: np.ndarray,
    units: list[tuple[np.ndarray, list[tuple[int, int]], tuple[int, int, int]]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]] | None:
    occupied = np.zeros_like(out)
    reflected_global: list[tuple[int, int]] = []
    markers_global: list[tuple[int, int, int]] = []

    for unit, reflected, marker in units:
        footprint = np.ones_like(unit, dtype=int)
        try:
            r0, c0 = gen_random_free_location(
                occupied,
                footprint,
                transparent=ArcColor.BLACK,
                padding=_GEN_PLACEMENT_PADDING,
                padding_connectivity=_GEN_PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None

        gen_paste(out, unit, r0, c0, transparent=ArcColor.BLACK)
        gen_paste(occupied, footprint, r0, c0, transparent=ArcColor.BLACK)
        reflected_global.extend((r0 + r, c0 + c) for r, c in reflected)
        marker_r, marker_c, marker_color = marker
        markers_global.append((r0 + marker_r, c0 + marker_c, marker_color))

    return reflected_global, markers_global


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*_GEN_GRID_ROWS)
    cols = random.randint(*_GEN_GRID_COLS)
    n_segments = random.randint(*_GEN_SEGMENT_COUNT)
    out = np.zeros((rows, cols), dtype=int)
    marker_colors = random.sample(_GEN_MARKER_COLORS, n_segments)

    units = [_gen_unit(color) for color in marker_colors]
    units.sort(key=lambda item: item[0].size, reverse=True)
    placed = _gen_place_units(out, units)
    if placed is None:
        return None

    reflected_global, markers_global = placed
    inp = out.copy()
    for r, c in reflected_global:
        inp[r, c] = ArcColor.BLACK
    for r, c, color in markers_global:
        inp[r, c] = color

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 88207623 input")
