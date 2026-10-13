from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste
import random


_CORNER_OUTWARD = {
    "NW": (-1, -1),
    "NE": (-1, 1),
    "SW": (1, -1),
    "SE": (1, 1),
}

_ROTATED_CORNER = {
    0: {"NW": "NW", "NE": "NE", "SW": "SW", "SE": "SE"},
    1: {"NW": "SW", "NE": "NW", "SW": "SE", "SE": "NE"},
    2: {"NW": "SE", "NE": "SW", "SW": "NE", "SE": "NW"},
    3: {"NW": "NE", "NE": "SE", "SW": "NW", "SE": "SW"},
}


def _corner_coords(bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {
        "NW": (r0, c0),
        "NE": (r0, c1),
        "SW": (r1, c0),
        "SE": (r1, c1),
    }


def _outside_corner_coords(corners: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    return {
        name: (pos[0] + _CORNER_OUTWARD[name][0], pos[1] + _CORNER_OUTWARD[name][1])
        for name, pos in corners.items()
    }


def _rotation_steps(src_corner: str, dst_corner: str) -> int:
    steps = [k for k, mapping in _ROTATED_CORNER.items() if mapping[src_corner] == dst_corner]
    assert len(steps) == 1, "expected a unique rotation mapping between marked corners"
    return steps[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the 3x3 two-color sprite so the singleton marker corner maps to the missing-corner marker, then collapse that marker to its center.

    Input structure
    - The dominant color is the background.
    - The foreground has one filled 3x3 two-color sprite and four same-color singleton marker pixels.
    - Three marker pixels form the outer corners of a missing-corner square whose missing corner touches one sprite corner.
    - The fourth marker pixel sits diagonally outside another sprite corner.

    Rule
    - Identify the sprite corner touched by the missing-corner marker as the destination corner.
    - Identify the sprite corner diagonally adjacent to the singleton marker as the source corner.
    - Rotate the sprite in place by the unique quarter-turn count that maps the source corner to the destination corner.
    - Remove the four input marker pixels and paint one marker pixel at the missing-corner square's center.

    Key insight
    - The marker pixels are rotation controls: the L-shaped group names the target corner, while the lone marker names the corner that must rotate into it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    bg = get_bg(g)

    components = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    sprite_candidates = []
    for component in components:
        bbox = component["bbox"]
        if component["area"] == 9 and bbox_h(bbox) == 3 and bbox_w(bbox) == 3:
            r0, c0, r1, c1 = bbox
            if np.all(g[r0 : r1 + 1, c0 : c1 + 1] != bg):
                sprite_candidates.append(component)
    assert len(sprite_candidates) == 1, "expected one filled 3x3 foreground sprite"

    sprite = sprite_candidates[0]
    r0, c0, r1, c1 = sprite["bbox"]
    sprite_patch = g[r0 : r1 + 1, c0 : c1 + 1]
    sprite_colors = set(map(int, sprite_patch.flat))
    assert len(sprite_colors) == 2, "sprite must use exactly two colors"

    marker_components = [component for component in components if component is not sprite]
    assert len(marker_components) == 4, "expected four marker pixels outside the sprite"
    assert all(component["area"] == 1 for component in marker_components), "markers must be singleton pixels"

    marker_coords = [tuple(map(int, component["coords"][0])) for component in marker_components]
    marker_colors = {int(g[r, c]) for r, c in marker_coords}
    assert len(marker_colors) == 1, "all markers must have the same color"
    marker_color = next(iter(marker_colors))
    assert marker_color != bg and marker_color not in sprite_colors, "marker color must be distinct"
    marker_set = set(marker_coords)

    corners = _corner_coords(sprite["bbox"])
    outside = _outside_corner_coords(corners)
    matches = []
    for dst_corner, sprite_corner_coord in corners.items():
        center = outside[dst_corner]
        square_corners = {
            (center[0] + dr, center[1] + dc)
            for dr in (-1, 1)
            for dc in (-1, 1)
        }
        l_marker_coords = square_corners - {sprite_corner_coord}
        if not l_marker_coords.issubset(marker_set):
            continue

        remaining_markers = marker_set - l_marker_coords
        if len(remaining_markers) != 1:
            continue
        singleton = next(iter(remaining_markers))
        src_corners = [corner for corner, pos in outside.items() if pos == singleton]
        if len(src_corners) == 1:
            matches.append((src_corners[0], dst_corner, center))

    assert len(matches) == 1, "expected a unique marker-defined source and destination corner"
    src_corner, dst_corner, center = matches[0]
    cr, cc = center
    assert 0 <= cr < g.shape[0] and 0 <= cc < g.shape[1], "collapsed marker center must be in bounds"

    out = g.copy()
    for r, c in marker_set:
        out[r, c] = bg
    out[r0 : r1 + 1, c0 : c1 + 1] = np.rot90(sprite_patch, k=_rotation_steps(src_corner, dst_corner))
    out[cr, cc] = marker_color
    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
GEN_ROW_RANGE = (10, 12)
GEN_COL_RANGE = (10, 11)
GEN_COLOR_RANGE = range(1, 10)
GEN_PALETTE_SIZE = 3
GEN_BACKGROUND = 0

# Sprite structure.
GEN_SPRITE_SIZE = 3
GEN_SPRITE_C1_COUNT_RANGE = (3, 6)
GEN_ROTATION_STEPS = (1, 2, 3)
GEN_FULL_TURN_STEPS = 4

# Marker geometry, placement, and retry budget.
GEN_CORNERS = ["NW", "NE", "SW", "SE"]
GEN_SPRITE_MARGIN = 3
GEN_L_GROUP_OFFSETS = (-1, 1)
GEN_MARKER_COUNT = 4
GEN_MAX_ATTEMPTS = 300

GEN_CORNER_OUTWARD = {"NW": (-1, -1), "NE": (-1, 1), "SW": (1, -1), "SE": (1, 1)}
GEN_ROTATED_CORNER = {
    1: {"NW": "SW", "NE": "NW", "SW": "SE", "SE": "NE"},
    2: {"NW": "SE", "NE": "SW", "SW": "NE", "SE": "NW"},
    3: {"NW": "NE", "NE": "SE", "SW": "NW", "SE": "SW"},
}


def _gen_rotation_steps(src_corner: str, dst_corner: str) -> int:
    for steps, mapping in GEN_ROTATED_CORNER.items():
        if mapping[src_corner] == dst_corner:
            return steps
    raise ValueError("source and destination corners must be distinct")


def _gen_sprite_corners(r0: int, c0: int) -> dict[str, tuple[int, int]]:
    far = GEN_SPRITE_SIZE - 1
    return {
        "NW": (r0, c0),
        "NE": (r0, c0 + far),
        "SW": (r0 + far, c0),
        "SE": (r0 + far, c0 + far),
    }


def _gen_outside_corner(corner: str, pos: tuple[int, int]) -> tuple[int, int]:
    dr, dc = GEN_CORNER_OUTWARD[corner]
    return pos[0] + dr, pos[1] + dc


def _gen_l_group(dst_corner: str, corners: dict[str, tuple[int, int]]):
    sr, sc = corners[dst_corner]
    cr, cc = _gen_outside_corner(dst_corner, (sr, sc))
    square = [(cr + dr, cc + dc) for dr in GEN_L_GROUP_OFFSETS for dc in GEN_L_GROUP_OFFSETS]
    return (cr, cc), [pos for pos in square if pos != (sr, sc)]


def _gen_is_rot_asymmetric(patch: np.ndarray) -> bool:
    return all(not np.array_equal(patch, np.rot90(patch, k=k)) for k in GEN_ROTATION_STEPS)


def _gen_make_sprite(sprite_c1: int, sprite_c2: int) -> np.ndarray:
    """Sample a filled sprite and repair accidental rotational symmetry."""
    c1_count = random.randint(*GEN_SPRITE_C1_COUNT_RANGE)
    positions = [(r, c) for r in range(GEN_SPRITE_SIZE) for c in range(GEN_SPRITE_SIZE)]
    c1_positions = random.sample(positions, c1_count)
    patch = np.full((GEN_SPRITE_SIZE, GEN_SPRITE_SIZE), sprite_c2, dtype=int)
    rr, cc = zip(*c1_positions)
    patch[rr, cc] = sprite_c1
    if _gen_is_rot_asymmetric(patch):
        return patch

    swaps = [(a, b) for a in positions for b in positions if patch[a] != patch[b]]
    random.shuffle(swaps)
    for a, b in swaps:
        repaired = patch.copy()
        repaired[a], repaired[b] = repaired[b], repaired[a]
        if _gen_is_rot_asymmetric(repaired):
            return repaired
    raise ValueError("failed to repair sprite asymmetry")


def _gen_layout(rows: int, cols: int, src_corner: str, dst_corner: str):
    """Sample sprite placement and derive valid marker coordinates."""
    r0 = random.randint(GEN_SPRITE_MARGIN, rows - GEN_SPRITE_MARGIN - GEN_SPRITE_SIZE)
    c0 = random.randint(GEN_SPRITE_MARGIN, cols - GEN_SPRITE_MARGIN - GEN_SPRITE_SIZE)
    corners = _gen_sprite_corners(r0, c0)
    collapsed_marker, l_markers = _gen_l_group(dst_corner, corners)
    singleton_marker = _gen_outside_corner(src_corner, corners[src_corner])
    marker_positions = l_markers + [singleton_marker]
    if len(set(marker_positions)) != GEN_MARKER_COUNT:
        return None

    for r, c in marker_positions + [collapsed_marker]:
        if not (0 <= r < rows and 0 <= c < cols):
            return None
    return r0, c0, collapsed_marker, marker_positions


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GEN_ROW_RANGE)
    cols = random.randint(*GEN_COL_RANGE)
    sprite_c1, sprite_c2, marker_color = random.sample(GEN_COLOR_RANGE, GEN_PALETTE_SIZE)

    try:
        output_sprite = _gen_make_sprite(sprite_c1, sprite_c2)
    except ValueError:
        return None
    src_corner, dst_corner = random.sample(GEN_CORNERS, 2)
    steps = _gen_rotation_steps(src_corner, dst_corner)
    input_sprite = np.rot90(output_sprite, k=GEN_FULL_TURN_STEPS - steps)

    layout = _gen_layout(rows, cols, src_corner, dst_corner)
    if layout is None:
        return None
    r0, c0, collapsed_marker, marker_positions = layout

    inp = np.zeros((rows, cols), dtype=int)
    out = np.zeros((rows, cols), dtype=int)
    paste(inp, input_sprite, r0, c0, transparent=None)
    paste(out, output_sprite, r0, c0, transparent=None)

    marker_rows, marker_cols = zip(*marker_positions)
    if np.any(inp[marker_rows, marker_cols] != GEN_BACKGROUND):
        return None
    inp[marker_rows, marker_cols] = marker_color
    cr, cc = collapsed_marker
    if out[cr, cc] != GEN_BACKGROUND:
        return None
    out[cr, cc] = marker_color

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ecaa0ec1 input")
