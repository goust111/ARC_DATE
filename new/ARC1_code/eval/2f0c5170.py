from collections import Counter

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste, random_free_location
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Copy the marked sprite window into the marked empty window, aligned by the shared marker.

    Input structure
    - The dominant color is the outer background.
    - There are exactly two solid rectangular non-background windows.
    - Each window has one pixel of a shared marker color; only one window also
      contains the sprite color(s).

    Rule
    - The marker-only window gives the output size and fill color.
    - Translate every non-fill pixel from the sprite window so the source marker
      lands on the destination marker.

    Key insight
    The two equal marker pixels define the offset; the output is cropped to the
    marker-only window, not to the sprite window.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    windows = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert len(windows) == 2

    regions = []
    for window in windows:
        r0, c0, r1, c1 = window["bbox"]
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        assert int(window["area"]) == int(region.size)
        regions.append(region)

    colors = colors_present(g, exclude={bg})
    counts = Counter(map(int, g.ravel()))

    marker_candidates = [
        color
        for color in colors
        if counts[color] == len(regions)
        and all(np.count_nonzero(region == color) == 1 for region in regions)
    ]
    assert len(marker_candidates) == 1
    marker = marker_candidates[0]

    fill_candidates = [
        color
        for color in colors
        if color != marker and all(np.any(region == color) for region in regions)
    ]
    assert len(fill_candidates) == 1
    fill = fill_candidates[0]

    sprite_colors = [color for color in colors if color not in {fill, marker}]
    assert sprite_colors

    classified = []
    for region in regions:
        marker_coords = np.argwhere(region == marker)
        assert marker_coords.shape == (1, 2)
        has_sprite = any(np.any(region == color) for color in sprite_colors)
        classified.append((has_sprite, region, tuple(map(int, marker_coords[0]))))

    source = [item for item in classified if item[0]]
    destination = [item for item in classified if not item[0]]
    assert len(source) == 1
    assert len(destination) == 1

    _, src_region, src_marker = source[0]
    _, dst_region, dst_marker = destination[0]

    out = np.full(dst_region.shape, fill, dtype=int)
    dr = dst_marker[0] - src_marker[0]
    dc = dst_marker[1] - src_marker[1]

    for r, c in np.argwhere(src_region != fill):
        rr = int(r) + dr
        cc = int(c) + dc
        assert 0 <= rr < out.shape[0]
        assert 0 <= cc < out.shape[1]
        out[rr, cc] = src_region[r, c]

    return out

# --------------- generator (o5x) ---------------
# Fixed colors and draw palette.
BG_COLOR, FILL_COLOR = 8, 0
MARKER_AND_SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# Sprite blob geometry.
SPRITE_PIXEL_RANGE = (7, 11)
BLOB_HEIGHT_OPTIONS, BLOB_WIDTH_OPTIONS = (3, 4), (3, 4, 5)
BLOB_MARKER_MARGIN = 1
FOUR_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Source and destination window constraints.
SOURCE_HEIGHT_RANGE, SOURCE_WIDTH_RANGE = (4, 7), (5, 7)
SOURCE_DENSITY_RANGE = (0.19, 0.36)
DEST_SIZE_RANGE = (5, 9)

# Canvas placement and retry budget.
GRID_SIZE_RANGE = (18, 23)
WINDOW_BORDER, WINDOW_PADDING = 1, 1
GENERATION_ATTEMPTS = 300

def _gen_monotone_path(start, end):
    r, c = start
    er, ec = end
    moves = []
    moves.extend([(1 if er > r else -1, 0)] * abs(er - r))
    moves.extend([(0, 1 if ec > c else -1)] * abs(ec - c))
    random.shuffle(moves)

    path = [(r, c)]
    for dr, dc in moves:
        r += dr
        c += dc
        path.append((r, c))
    return path

def _gen_spanning_component(height, width, total_cells):
    marker = (
        random.randrange(BLOB_MARKER_MARGIN, height - BLOB_MARKER_MARGIN),
        random.randrange(BLOB_MARKER_MARGIN, width - BLOB_MARKER_MARGIN),
    )
    corner = random.choice([(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)])
    opposite = (height - 1 - corner[0], width - 1 - corner[1])

    coords = set(_gen_monotone_path(corner, marker))
    coords.update(_gen_monotone_path(marker, opposite))

    while len(coords) < total_cells:
        frontier = []
        for r, c in coords:
            for dr, dc in FOUR_NEIGHBORS:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in coords:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        coords.add(random.choice(frontier))

    return coords, marker

def _gen_pick_bbox(sprite_pixels):
    total_cells = sprite_pixels + 1
    choices = [
        (height, width)
        for height in BLOB_HEIGHT_OPTIONS
        for width in BLOB_WIDTH_OPTIONS
        if height + width - 1 <= total_cells <= height * width
    ]
    return random.choice(choices)

def _gen_pick_source_size(blob_height, blob_width, sprite_pixels):
    min_density, max_density = SOURCE_DENSITY_RANGE
    choices = []
    for height in range(max(SOURCE_HEIGHT_RANGE[0], blob_height), SOURCE_HEIGHT_RANGE[1] + 1):
        for width in range(max(SOURCE_WIDTH_RANGE[0], blob_width), SOURCE_WIDTH_RANGE[1] + 1):
            density = sprite_pixels / (height * width)
            if min_density <= density <= max_density:
                choices.append((height, width))
    if not choices:
        return None
    return random.choice(choices)

def _paint_aligned_sprite(canvas, sprite_coords, src_marker, dst_marker, marker_color, sprite_color):
    dr = dst_marker[0] - src_marker[0]
    dc = dst_marker[1] - src_marker[1]
    for r, c in sprite_coords:
        canvas[dr + r, dc + c] = sprite_color
    canvas[dst_marker] = marker_color

def _gen_sprite_blob():
    sprite_pixels = random.randint(*SPRITE_PIXEL_RANGE)
    blob_height, blob_width = _gen_pick_bbox(sprite_pixels)
    component = _gen_spanning_component(blob_height, blob_width, sprite_pixels + 1)
    if component is None:
        return None
    blob_coords, src_marker = component
    sprite_coords = [coord for coord in blob_coords if coord != src_marker]
    if len(sprite_coords) != sprite_pixels:
        return None
    return sprite_pixels, blob_height, blob_width, src_marker, sprite_coords

def _place_windows(source, dest):
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    grid = np.full((height, width), BG_COLOR, dtype=int)
    windows = [source, dest]
    random.shuffle(windows)
    try:
        for window in windows:
            r0, c0 = random_free_location(grid, window, transparent=BG_COLOR, border=WINDOW_BORDER, padding=WINDOW_PADDING)
            paste(grid, window, r0, c0)
    except ValueError:
        return None
    return grid

def _gen_attempt():
    marker_color, sprite_color = random.sample(MARKER_AND_SPRITE_COLORS, 2)
    sprite = _gen_sprite_blob()
    if sprite is None:
        return None
    sprite_pixels, blob_height, blob_width, src_marker, sprite_coords = sprite

    source_size = _gen_pick_source_size(blob_height, blob_width, sprite_pixels)
    if source_size is None:
        return None
    source_height, source_width = source_size
    source_top = random.randint(0, source_height - blob_height)
    source_left = random.randint(0, source_width - blob_width)
    source = np.full((source_height, source_width), FILL_COLOR, dtype=int)
    source_marker = (source_top + src_marker[0], source_left + src_marker[1])
    _paint_aligned_sprite(source, sprite_coords, src_marker, source_marker, marker_color, sprite_color)

    top_extent = src_marker[0]
    bottom_extent = blob_height - 1 - src_marker[0]
    left_extent = src_marker[1]
    right_extent = blob_width - 1 - src_marker[1]
    dest_height = random.randint(max(DEST_SIZE_RANGE[0], blob_height), DEST_SIZE_RANGE[1])
    dest_width = random.randint(max(DEST_SIZE_RANGE[0], blob_width), DEST_SIZE_RANGE[1])
    dest_marker = (
        random.randint(top_extent, dest_height - 1 - bottom_extent),
        random.randint(left_extent, dest_width - 1 - right_extent),
    )

    expected = np.full((dest_height, dest_width), FILL_COLOR, dtype=int)
    _paint_aligned_sprite(expected, sprite_coords, src_marker, dest_marker, marker_color, sprite_color)
    dest = np.full((dest_height, dest_width), FILL_COLOR, dtype=int)
    dest[dest_marker] = marker_color

    grid = _place_windows(source, dest)
    if grid is None:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid

def generate_input():
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 2f0c5170")
