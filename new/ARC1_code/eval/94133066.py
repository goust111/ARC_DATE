from collections import Counter

import numpy as np

from arc_utils import bbox, colors_present, get_bg, paste
import math
import random


def _normalized(points: dict[int, tuple[int, int]]) -> dict[int, tuple[int, int]]:
    rows = [r for r, _ in points.values()]
    cols = [c for _, c in points.values()]
    r0 = min(rows)
    c0 = min(cols)
    return {color: (r - r0, c - c0) for color, (r, c) in points.items()}


def _orientations(crop: np.ndarray) -> list[np.ndarray]:
    return [
        crop.copy(),
        np.rot90(crop, 1),
        np.rot90(crop, 2),
        np.rot90(crop, 3),
        np.flipud(crop),
        np.fliplr(crop),
        crop.T,
        np.flipud(np.fliplr(crop)).T,
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the main rectangle and orient it so its internal markers match the external marker arrangement.

    Input structure
    A background canvas contains one filled rectangle of the dominant foreground color. Some non-base colors
    are inside that rectangle; orientation marker colors occur exactly once inside the rectangle and exactly
    once as an external pixel.

    Rule
    Remove the background and external marker pixels by cropping to the base rectangle. Return the unique
    rotation/reflection of that crop whose marker-color coordinates, normalized by their top-left marker
    position, equal the normalized coordinates of the external marker pixels with the same colors.

    Key insight
    The outside pixels are not copied into the output; their relative layout is an orientation legend for
    the matching singleton colors embedded in the rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    fg_counts = Counter(int(x) for x in g.ravel() if int(x) != bg)
    assert fg_counts
    largest = max(fg_counts.values())
    base_colors = [color for color, count in fg_counts.items() if count == largest]
    assert len(base_colors) == 1
    base = base_colors[0]

    rect = bbox(g == base)
    assert rect is not None
    r0, c0, r1, c1 = rect
    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert crop.size > 0
    assert not np.any(crop == bg)

    marker_inside: dict[int, tuple[int, int]] = {}
    marker_outside: dict[int, tuple[int, int]] = {}
    for color in colors_present(g, exclude={bg, base}):
        inside = []
        outside = []
        for rr, cc in np.argwhere(g == color):
            r = int(rr)
            c = int(cc)
            if r0 <= r <= r1 and c0 <= c <= c1:
                inside.append((r - r0, c - c0))
            else:
                outside.append((r, c))

        if len(inside) == 1 and len(outside) == 1:
            marker_inside[color] = inside[0]
            marker_outside[color] = outside[0]
        else:
            assert len(outside) == 0

    assert len(marker_inside) >= 3
    target_layout = _normalized(marker_outside)

    matches = []
    for oriented in _orientations(crop):
        oriented_markers = {}
        for color in marker_inside:
            locs = np.argwhere(oriented == color)
            assert len(locs) == 1
            oriented_markers[color] = (int(locs[0, 0]), int(locs[0, 1]))
        if _normalized(oriented_markers) == target_layout:
            matches.append(oriented)

    assert len(matches) == 1
    return matches[0].copy()

# --------------- generator (o5x) ---------------

# Canvas and crop bounds.
GEN_GRID_ROWS_RANGE, GEN_GRID_COLS_RANGE = (19, 23), (19, 30)
GEN_CROP_H_RANGE, GEN_CROP_W_RANGE = (9, 13), (9, 10)
GEN_RECT_EDGE_GAP, GEN_EXTERNAL_CLEARANCE = 1, 3

# Color roles.
GEN_BACKGROUND_COLOR, GEN_BASE_COLOR = 0, 1
GEN_AVAILABLE_COLORS = tuple(range(2, 10))
GEN_MARKER_COUNT = 3
GEN_COLOR_SAMPLE_COUNT = GEN_MARKER_COUNT + 1

# Marker slots and internal pattern growth.
GEN_MARKER_CORNER_OFFSET = 1
GEN_PATTERN_COUNT_RANGE, GEN_PATTERN_AREA_FRACTION = (9, 15), (0.11, 0.17)
GEN_PATTERN_INTERIOR_MARGIN = 2
GEN_PATTERN_MODES = ("walk", "frontier", "mixed")
GEN_MIXED_WALK_PROB = 0.6
GEN_NEIGHBOR_OFFSETS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Orientation and retry limits.
GEN_ORIENTATION_COUNT, GEN_MAX_ATTEMPTS = 8, 300


def _gen_orient(crop: np.ndarray, idx: int) -> np.ndarray:
    return _orientations(crop)[int(idx)].copy()


def _gen_is_asymmetric(arr: np.ndarray) -> bool:
    base = arr.copy()
    for idx in range(1, GEN_ORIENTATION_COUNT):
        other = _gen_orient(arr, idx)
        if base.shape == other.shape and np.array_equal(base, other):
            return False
    return True


def _gen_marker_points(crop: np.ndarray, colors: list[int]) -> dict[int, tuple[int, int]]:
    points = {}
    for color in colors:
        locs = np.argwhere(crop == int(color))
        if len(locs) != 1:
            return {}
        points[int(color)] = (int(locs[0, 0]), int(locs[0, 1]))
    return points


def _gen_neighbors4(pos: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pos
    return [(r + dr, c + dc) for dr, dc in GEN_NEIGHBOR_OFFSETS_4]


def _gen_pattern_coords(
    h: int, w: int, count: int, blocked: set[tuple[int, int]]
) -> set[tuple[int, int]] | None:
    """Grow one connected internal pattern away from borders and marker slots."""
    allowed = {
        (r, c)
        for r in range(GEN_PATTERN_INTERIOR_MARGIN, h - GEN_PATTERN_INTERIOR_MARGIN)
        for c in range(GEN_PATTERN_INTERIOR_MARGIN, w - GEN_PATTERN_INTERIOR_MARGIN)
        if (r, c) not in blocked
    }
    if len(allowed) < count:
        return None

    current = random.choice(tuple(allowed))
    coords = {current}
    mode = random.choice(GEN_PATTERN_MODES)

    while len(coords) < count:
        frontier = {
            n
            for pos in coords
            for n in _gen_neighbors4(pos)
            if n in allowed and n not in coords
        }
        if not frontier:
            return None

        walk_options = [
            n for n in _gen_neighbors4(current) if n in allowed and n not in coords
        ]
        if mode == "walk" and walk_options:
            nxt = random.choice(walk_options)
        elif mode == "mixed" and walk_options and random.random() < GEN_MIXED_WALK_PROB:
            nxt = random.choice(walk_options)
        else:
            nxt = random.choice(tuple(frontier))
        coords.add(nxt)
        current = nxt

    return coords


def _gen_pattern_count(h: int, w: int) -> int | None:
    area = h * w
    lo = max(GEN_PATTERN_COUNT_RANGE[0], int(math.ceil(area * GEN_PATTERN_AREA_FRACTION[0])))
    hi = min(GEN_PATTERN_COUNT_RANGE[1], int(math.floor(area * GEN_PATTERN_AREA_FRACTION[1])))
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _gen_marker_slots(h: int, w: int) -> list[tuple[int, int]]:
    offset = GEN_MARKER_CORNER_OFFSET
    slots = [
        (offset, offset),
        (offset, w - offset - 1),
        (h - offset - 1, offset),
        (h - offset - 1, w - offset - 1),
    ]
    random.shuffle(slots)
    return slots[:GEN_MARKER_COUNT]


def _gen_make_crop() -> tuple[np.ndarray, list[int]] | None:
    """Build the filled rectangle crop with singleton markers and an asymmetric pattern."""
    h = random.randint(*GEN_CROP_H_RANGE)
    w = random.randint(*GEN_CROP_W_RANGE)
    pattern_count = _gen_pattern_count(h, w)
    if pattern_count is None:
        return None

    colors = random.sample(GEN_AVAILABLE_COLORS, GEN_COLOR_SAMPLE_COUNT)
    marker_colors = colors[:GEN_MARKER_COUNT]
    pattern_color = colors[GEN_MARKER_COUNT]

    crop = np.full((h, w), GEN_BASE_COLOR, dtype=int)
    marker_slots = _gen_marker_slots(h, w)
    random.shuffle(marker_colors)
    for color, (r, c) in zip(marker_colors, marker_slots):
        crop[r, c] = color

    pattern_coords = _gen_pattern_coords(h, w, pattern_count, set(marker_slots))
    if pattern_coords is None:
        return None

    pattern_mask = np.zeros(crop.shape, dtype=bool)
    pattern_rows, pattern_cols = zip(*pattern_coords)
    crop[pattern_rows, pattern_cols] = pattern_color
    pattern_mask[pattern_rows, pattern_cols] = True
    if not _gen_is_asymmetric(pattern_mask) or not _gen_is_asymmetric(crop):
        return None
    return crop, marker_colors


def _gen_external_layout(output: np.ndarray, marker_colors: list[int]) -> tuple[dict[int, tuple[int, int]], int, int] | None:
    """Return normalized external marker offsets plus their bounding-box size."""
    oriented_markers = _gen_marker_points(output, marker_colors)
    if len(oriented_markers) != GEN_MARKER_COUNT:
        return None

    external_rel = _normalized(oriented_markers)
    ext_h = max(r for r, _ in external_rel.values()) + 1
    ext_w = max(c for _, c in external_rel.values()) + 1
    return external_rel, ext_h, ext_w


def _gen_choose_placement(rows: int, cols: int, h: int, w: int, ext_h: int, ext_w: int) -> tuple[int, int, int, int] | None:
    """Choose non-overlapping rectangle and external-marker origins on the canvas."""
    placements = []
    for rect_r in range(GEN_RECT_EDGE_GAP, rows - h - GEN_RECT_EDGE_GAP + 1):
        for rect_c in range(GEN_RECT_EDGE_GAP, cols - w - GEN_RECT_EDGE_GAP + 1):
            top = range(0, max(0, rect_r - GEN_EXTERNAL_CLEARANCE - ext_h))
            bottom = range(rect_r + h + GEN_EXTERNAL_CLEARANCE, rows - ext_h + 1)
            left = range(0, max(0, rect_c - GEN_EXTERNAL_CLEARANCE - ext_w))
            right = range(rect_c + w + GEN_EXTERNAL_CLEARANCE, cols - ext_w + 1)

            for ext_r in top:
                for ext_c in range(0, cols - ext_w + 1):
                    placements.append((rect_r, rect_c, ext_r, ext_c))
            for ext_r in bottom:
                for ext_c in range(0, cols - ext_w + 1):
                    placements.append((rect_r, rect_c, ext_r, ext_c))
            for ext_c in left:
                for ext_r in range(0, rows - ext_h + 1):
                    placements.append((rect_r, rect_c, ext_r, ext_c))
            for ext_c in right:
                for ext_r in range(0, rows - ext_h + 1):
                    placements.append((rect_r, rect_c, ext_r, ext_c))

    if not placements:
        return None
    return random.choice(placements)


def _gen_attempt() -> np.ndarray | None:
    crop_data = _gen_make_crop()
    if crop_data is None:
        return None
    crop, marker_colors = crop_data

    output = _gen_orient(crop, random.randrange(GEN_ORIENTATION_COUNT))
    layout = _gen_external_layout(output, marker_colors)
    if layout is None:
        return None
    external_rel, ext_h, ext_w = layout

    rows = random.randint(*GEN_GRID_ROWS_RANGE)
    cols = random.randint(*GEN_GRID_COLS_RANGE)
    h, w = crop.shape
    placement = _gen_choose_placement(rows, cols, h, w, ext_h, ext_w)
    if placement is None:
        return None

    rect_r, rect_c, ext_r, ext_c = placement
    grid = np.full((rows, cols), GEN_BACKGROUND_COLOR, dtype=int)
    paste(grid, crop, rect_r, rect_c)
    for color, (r, c) in external_rel.items():
        grid[ext_r + r, ext_c + c] = int(color)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 94133066")
