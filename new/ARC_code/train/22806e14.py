import numpy as np

from arc_utils import find_components, get_bg, paste, random_free_location
import random
from scipy import ndimage


_PLUS_REL = {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}


def _is_plus_marker(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    rel = {(int(r) - r0, int(c) - c0) for r, c in comp["coords"]}
    return (
        int(comp["area"]) == 5
        and int(r1 - r0 + 1) == 3
        and int(c1 - c0 + 1) == 3
        and rel == _PLUS_REL
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark the center of each odd-sized solid square with the plus marker color, erasing the marker only when it is outside the marked-center field.

    Input structure:
    - The dominant color is background.
    - Exactly two foreground colors are present: one unique 3x3 plus marker and one color forming separated solid square objects.

    Rule:
    - For every non-marker 4-connected object, assert it is a solid square.
    - If the square side length is odd, recolor its center pixel to the plus marker color; even-sided squares stay unchanged.
    - Preserve the plus marker when its center lies inside the bounding box of the odd-square centers; otherwise erase it to background.

    Key insight:
    - Odd-sided solid squares have a single visible center pixel, and the plus supplies the color used to identify those centers.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)

    markers = [comp for comp in comps if _is_plus_marker(comp)]
    assert len(markers) == 1, "grid must contain exactly one 5-pixel plus marker"
    marker = markers[0]
    marker_color = int(marker["color"])

    fg_colors = set(map(int, g.flatten())) - {bg}
    assert marker_color in fg_colors
    object_colors = fg_colors - {marker_color}
    assert len(object_colors) == 1, "all solid square objects must share one non-marker color"
    object_color = object_colors.pop()

    out = g.copy()
    odd_centers: list[tuple[int, int]] = []

    for comp in comps:
        if comp is marker:
            continue
        assert int(comp["color"]) == object_color, "unexpected foreground object color"
        r0, c0, r1, c1 = comp["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert h == w, "foreground objects must be square"
        assert int(comp["area"]) == h * w, "foreground square objects must be solid"
        if h % 2 == 1:
            rr = int(r0 + h // 2)
            cc = int(c0 + w // 2)
            assert int(g[rr, cc]) == object_color
            odd_centers.append((rr, cc))
            out[rr, cc] = marker_color

    assert odd_centers, "at least one odd-sided square must be present"
    marker_r0, marker_c0, _marker_r1, _marker_c1 = marker["bbox"]
    marker_center = (int(marker_r0 + 1), int(marker_c0 + 1))
    rows = [r for r, _c in odd_centers]
    cols = [c for _r, c in odd_centers]
    marker_is_in_center_field = (
        min(rows) <= marker_center[0] <= max(rows)
        and min(cols) <= marker_center[1] <= max(cols)
    )

    if not marker_is_in_center_field:
        for r, c in marker["coords"]:
            out[int(r), int(c)] = bg

    return out

# --------------- generator (c) ---------------

# Canvas, palette, and retries.
GRID_H, GRID_W, BACKGROUND_COLOR, MAX_GENERATION_ATTEMPTS, COLOR_RANGE = 16, 16, 7, 300, range(10)

# Square counts and side lengths.
ODD_COUNT_RANGE, EVEN_COUNT_RANGE, TOTAL_SQUARE_RANGE = (2, 8), (1, 4), (5, 11)
ODD_SIDE_CHOICES, EVEN_SIDE_CHOICES = (1, 3, 5, 7), (2, 4, 6)

# Component separation and plus-marker geometry.
PLACEMENT_PADDING, PLACEMENT_CONNECTIVITY = 1, 8
PLUS_SIZE, PLUS_CENTER_OFFSET = 3, 1
PLUS_CELLS = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))

# Marker preservation branch weights.
MARKER_PRESERVED_CHOICES = [True, False]


def _sample_square_sides():
    while True:
        n_odd = random.randint(*ODD_COUNT_RANGE)
        n_even = random.randint(*EVEN_COUNT_RANGE)
        if TOTAL_SQUARE_RANGE[0] <= n_odd + n_even <= TOTAL_SQUARE_RANGE[1]:
            break

    odd_sides = [random.choice(ODD_SIDE_CHOICES) for _ in range(n_odd)]
    even_sides = [random.choice(EVEN_SIDE_CHOICES) for _ in range(n_even)]
    all_sides = odd_sides + even_sides
    random.shuffle(all_sides)
    all_sides.sort(reverse=True)
    return all_sides


def _make_plus(color, bg):
    p = np.full((PLUS_SIZE, PLUS_SIZE), bg, dtype=int)
    for dr, dc in PLUS_CELLS:
        p[dr, dc] = color
    return p


def _place_squares(grid, sides, object_color, bg):
    """Paste separated squares; return placements or None if placement fails."""
    square_placements = []
    for side in sides:
        patch = np.full((side, side), object_color, dtype=int)
        try:
            r, c = random_free_location(
                grid,
                patch,
                transparent=bg,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(grid, patch, r, c, transparent=bg)
        square_placements.append((r, c, side))
    return square_placements


def _valid_marker_positions(grid, plus_patch, bg, center_bbox, marker_preserved):
    """Return marker top-lefts with the requested center-in-bbox relation."""
    bbox_r_min, bbox_r_max, bbox_c_min, bbox_c_max = center_bbox
    plus_mask = plus_patch != bg
    occ = grid != bg
    struct8 = np.ones((3, 3), dtype=bool)
    dilated = ndimage.binary_dilation(occ, structure=struct8, iterations=1)

    valid_positions = []
    for r0 in range(GRID_H - PLUS_SIZE + 1):
        for c0 in range(GRID_W - PLUS_SIZE + 1):
            region = dilated[r0:r0 + PLUS_SIZE, c0:c0 + PLUS_SIZE]
            if np.any(region & plus_mask):
                continue

            mc_r = r0 + PLUS_CENTER_OFFSET
            mc_c = c0 + PLUS_CENTER_OFFSET
            inside = bbox_r_min <= mc_r <= bbox_r_max and bbox_c_min <= mc_c <= bbox_c_max
            if marker_preserved and inside:
                valid_positions.append((r0, c0))
            elif not marker_preserved and not inside:
                valid_positions.append((r0, c0))
    return valid_positions


def _build_output(grid, odd_centers, marker_color, marker_pos, marker_preserved, bg):
    """Apply the solver's target transform to the generated input grid."""
    output = grid.copy()
    for cr, cc in odd_centers:
        output[cr, cc] = marker_color

    if not marker_preserved:
        mr, mc = marker_pos
        for dr, dc in PLUS_CELLS:
            output[mr + dr, mc + dc] = bg
    return output


def _gen_attempt():
    """Try one complete sample, returning None when constraints cannot be met."""
    bg = BACKGROUND_COLOR
    palette = [c for c in COLOR_RANGE if c != bg]
    marker_color, object_color = random.sample(palette, 2)
    grid = np.full((GRID_H, GRID_W), bg, dtype=int)

    all_sides = _sample_square_sides()
    square_placements = _place_squares(grid, all_sides, object_color, bg)
    if square_placements is None:
        return None

    odd_centers = [
        (r0 + side // 2, c0 + side // 2)
        for r0, c0, side in square_placements
        if side % 2 == 1
    ]
    if not odd_centers:
        return None
    rows_oc = [r for r, c in odd_centers]
    cols_oc = [c for r, c in odd_centers]
    center_bbox = min(rows_oc), max(rows_oc), min(cols_oc), max(cols_oc)
    marker_preserved = random.choice(MARKER_PRESERVED_CHOICES)
    plus_patch = _make_plus(marker_color, bg)
    valid_positions = _valid_marker_positions(grid, plus_patch, bg, center_bbox, marker_preserved)
    if not valid_positions:
        return None
    marker_pos = random.choice(valid_positions)
    paste(grid, plus_patch, marker_pos[0], marker_pos[1], transparent=bg)

    output = _build_output(grid, odd_centers, marker_color, marker_pos, marker_preserved, bg)
    try:
        result = main(grid)
        if np.array_equal(result, output):
            return grid
    except Exception:
        pass
    return None


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
