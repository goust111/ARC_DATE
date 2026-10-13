from collections import Counter

import numpy as np

from arc_utils import find_components, get_bg
import random


def _corner_distances(shape: tuple[int, int], corner: str, r: int, c: int) -> tuple[int, int]:
    H, W = shape
    if corner == "TL":
        return int(r), int(c)
    if corner == "TR":
        return int(r), int(W - 1 - c)
    if corner == "BL":
        return int(H - 1 - r), int(c)
    assert corner == "BR"
    return int(H - 1 - r), int(W - 1 - c)


def _paint_corner_triangle(out: np.ndarray, corner: str, side: int, color: int) -> None:
    H, W = out.shape
    assert side > 0, "triangle side length must be positive"
    for r in range(H):
        for c in range(W):
            a, b = _corner_distances((H, W), corner, r, c)
            if a + b < side:
                out[r, c] = int(color)


def _diagonal_direction(coords: set[tuple[int, int]]) -> str | None:
    r0 = min(r for r, _ in coords)
    r1 = max(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    c1 = max(c for _, c in coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height != width:
        return None
    n = height
    local = {(r - r0, c - c0) for r, c in coords}
    main_ok = all((c, r) in local for r, c in local)
    anti_ok = all((n - 1 - c, n - 1 - r) in local for r, c in local)
    if main_ok:
        if (0, 0) in local and (n - 1, n - 1) not in local:
            return "TL"
        if (n - 1, n - 1) in local and (0, 0) not in local:
            return "BR"
    if anti_ok:
        if (0, n - 1) in local and (n - 1, 0) not in local:
            return "TR"
        if (n - 1, 0) in local and (0, n - 1) not in local:
            return "BL"
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the largest 8-connected motif and replace the corner/control pixels with a solid corner triangle.

    Input structure
    - The background is the dominant color.
    - Exactly one grid corner contains a foreground control pixel; it gives the
      corner from which the triangle is painted.
    - There is one unique largest 8-connected monochrome foreground object, the
      motif that remains in the output; all other foreground pixels are controls.
    - Every foreground component outside that motif is a singleton control pixel.
    - The fill color is the unique control color that appears in at least two
      non-corner control pixels.

    Rule
    - Start from a blank background canvas and redraw only the largest motif.
    - Count the non-corner control pixels; that count is the triangle side.
    - Paint the fill color on the right-triangle region where those two distances
      from the marked corner's edges sum to less than the side length.

    Key insight
    - The scattered singleton controls are counted into a compact triangular
      stack at the marked corner; only the largest motif is not part of the count.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    H, W = g.shape

    corners = {
        "TL": (0, 0),
        "TR": (0, W - 1),
        "BL": (H - 1, 0),
        "BR": (H - 1, W - 1),
    }
    marked_corners = [
        (name, rc) for name, rc in corners.items() if int(g[rc]) != int(bg)
    ]
    assert len(marked_corners) == 1, "expected exactly one foreground corner marker"
    corner, (corner_r, corner_c) = marked_corners[0]

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True)
    assert comps, "expected at least one foreground component"
    max_area = max(int(comp["area"]) for comp in comps)
    motif_candidates = [comp for comp in comps if int(comp["area"]) == max_area]
    assert len(motif_candidates) == 1, "expected a unique largest motif"
    motif = motif_candidates[0]
    motif_color = int(motif["color"])
    motif_coords = {tuple(map(int, coord)) for coord in motif["coords"]}
    motif_direction = _diagonal_direction(motif_coords)
    assert motif_direction is not None, "motif must be square and diagonal-symmetric"
    assert motif_direction == corner, "corner marker must match motif diagonal direction"

    control_coords = [
        (int(r), int(c))
        for r, c in np.argwhere(g != bg)
        if (int(r), int(c)) not in motif_coords
    ]
    assert (corner_r, corner_c) in control_coords, "corner marker must be a control pixel"
    assert len(control_coords) > 1, "expected non-corner control pixels"

    control_components = [comp for comp in comps if comp is not motif]
    assert all(int(comp["area"]) == 1 for comp in control_components), "controls must be singleton components"

    control_counts = Counter(
        int(g[r, c])
        for r, c in control_coords
        if (int(r), int(c)) != (corner_r, corner_c)
    )
    fill_candidates = [
        int(color) for color, count in control_counts.items() if int(count) >= 2
    ]
    assert len(fill_candidates) == 1, "expected one repeated non-corner fill color"
    fill_color = fill_candidates[0]
    side = len(control_coords) - 1

    out = np.full_like(g, bg)
    for r, c in motif_coords:
        out[r, c] = motif_color
    _paint_corner_triangle(out, corner, int(side), fill_color)
    return out

# --------------- generator (att2) ---------------

_GEN_ATTEMPTS = 300

# Canvas and triangle bounds.
_GEN_ROW_BOUNDS = (6, 15)
_GEN_COL_BOUNDS = (9, 14)
_GEN_TRIANGLE_SIDE_BOUNDS = (2, 5)

# Color and control-pixel sampling.
_GEN_COLORS = tuple(range(1, 10))
_GEN_MARKER_AVOID_FILL_PROB = 0.8
_GEN_FILL_COUNT_BOUNDS = (2, 3)

# Sparse square motifs observed in the task examples.
_GEN_MOTIF_TEMPLATES = [
    ("..#.", ".#..", "#.#.", "...#"),
    (".#...", "..#..", "..##.", ".#..#", "#...."),
    (
        "...###...",
        "..#......",
        ".##......",
        "#..#.....",
        "#...#....",
        "#....#...",
        "......#..",
        ".......#.",
        "........#",
    ),
    ("#...", ".#.#", "..#.", ".#.."),
    ("...#", "#.#.", "##..", ".##."),
]


def _gen_triangle_mask(shape, corner, side):
    H, W = shape
    rr, cc = np.indices((H, W))
    if corner == "TL":
        return rr + cc < side
    if corner == "TR":
        return rr + (W - 1 - cc) < side
    if corner == "BL":
        return (H - 1 - rr) + cc < side
    return (H - 1 - rr) + (W - 1 - cc) < side


def _gen_dilate8(mask):
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _gen_patch_from_lines(lines):
    return np.array([[ch == "#" for ch in row] for row in lines], dtype=bool)


def _gen_patch_variants(patch):
    variants = []
    seen = set()
    for k in range(4):
        rotated = np.rot90(patch, k)
        for variant in (rotated, np.fliplr(rotated)):
            key = tuple(map(tuple, variant.astype(int)))
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def _gen_reflect_anti_diagonal(patch):
    return np.flipud(np.fliplr(patch.T))


def _gen_is_diagonal_symmetric_patch(patch):
    if patch.shape[0] != patch.shape[1]:
        return False
    main_ok = np.array_equal(patch, patch.T)
    return main_ok or np.array_equal(patch, _gen_reflect_anti_diagonal(patch))


def _gen_patch_direction(patch):
    if patch.shape[0] != patch.shape[1]:
        return None
    n = patch.shape[0]
    main_ok = np.array_equal(patch, patch.T)
    anti_ok = np.array_equal(patch, _gen_reflect_anti_diagonal(patch))
    if main_ok:
        if patch[0, 0] and not patch[n - 1, n - 1]:
            return "TL"
        if patch[n - 1, n - 1] and not patch[0, 0]:
            return "BR"
    if anti_ok:
        if patch[0, n - 1] and not patch[n - 1, 0]:
            return "TR"
        if patch[n - 1, 0] and not patch[0, n - 1]:
            return "BL"
    return None


def _gen_symmetric_motif_patch(max_h, max_w):
    candidates = []
    for lines in _GEN_MOTIF_TEMPLATES:
        patch = _gen_patch_from_lines(lines)
        for variant in _gen_patch_variants(patch):
            h, w = variant.shape
            direction = _gen_patch_direction(variant)
            if (
                h <= max_h
                and w <= max_w
                and _gen_is_diagonal_symmetric_patch(variant)
                and direction is not None
            ):
                candidates.append((variant, direction))
    if not candidates:
        return None
    patch, direction = random.choice(candidates)
    return patch.copy(), direction


def _gen_place_patch(forbidden, patch):
    H, W = forbidden.shape
    m, n = patch.shape
    cells = np.argwhere(patch)
    placements = []
    for r0 in range(H - m + 1):
        for c0 in range(W - n + 1):
            if all(not forbidden[r0 + r, c0 + c] for r, c in cells):
                placements.append((r0, c0))
    if not placements:
        return None
    r0, c0 = random.choice(placements)
    return {(int(r0 + r), int(c0 + c)) for r, c in cells}


def _gen_corner_cells(shape):
    H, W = shape
    return {
        "TL": (0, 0),
        "TR": (0, W - 1),
        "BL": (H - 1, 0),
        "BR": (H - 1, W - 1),
    }


def _gen_control_pool(grid, motif_mask, blocked_mask):
    H, W = grid.shape
    near_motif = _gen_dilate8(motif_mask)
    pool = []
    corners = set(_gen_corner_cells((H, W)).values())
    for r in range(H):
        for c in range(W):
            if (r, c) in corners:
                continue
            if grid[r, c] != 0 or near_motif[r, c] or blocked_mask[r, c]:
                continue
            pool.append((r, c))
    return pool


def _gen_choose_scene():
    H = random.randint(*_GEN_ROW_BOUNDS)
    W = random.randint(*_GEN_COL_BOUNDS)
    side = random.randint(*_GEN_TRIANGLE_SIDE_BOUNDS)
    colors = list(_GEN_COLORS)
    fill_color = random.choice(colors)
    motif_color = random.choice([c for c in colors if c != fill_color])
    return H, W, side, colors, fill_color, motif_color


def _gen_place_motif(H, W, side, fill_color, motif_color, motif_patch, corner):
    triangle_mask = _gen_triangle_mask((H, W), corner, side)
    output = np.zeros((H, W), dtype=int)
    output[triangle_mask] = fill_color

    forbidden = _gen_dilate8(triangle_mask)
    for rc in _gen_corner_cells((H, W)).values():
        forbidden[rc] = True
    motif_coords = _gen_place_patch(forbidden, motif_patch)
    if motif_coords is None:
        return None

    motif_mask = np.zeros((H, W), dtype=bool)
    for r, c in motif_coords:
        motif_mask[r, c] = True
        output[r, c] = motif_color
    return output, triangle_mask, motif_mask


def _gen_marker_choices(colors, motif_color, fill_color):
    marker_choices = [c for c in colors if c != motif_color]
    if random.random() < _GEN_MARKER_AVOID_FILL_PROB and len(marker_choices) > 1:
        marker_choices = [c for c in marker_choices if c != fill_color]
    return marker_choices


def _gen_control_colors(colors, side, fill_color):
    fill_count = random.randint(_GEN_FILL_COUNT_BOUNDS[0], min(_GEN_FILL_COUNT_BOUNDS[1], side))
    distractor_count = side - fill_count
    distractor_choices = [c for c in colors if c != fill_color]
    random.shuffle(distractor_choices)
    control_colors = [fill_color] * fill_count + distractor_choices[:distractor_count]
    random.shuffle(control_colors)
    return control_colors


def _gen_add_controls(inp, triangle_mask, motif_mask, corner, side, colors, fill_color, motif_color):
    H, W = inp.shape
    corner_rc = _gen_corner_cells((H, W))[corner]
    inp[corner_rc] = random.choice(_gen_marker_choices(colors, motif_color, fill_color))
    control_colors = _gen_control_colors(colors, side, fill_color)

    occupied_controls = np.zeros((H, W), dtype=bool)
    occupied_controls[corner_rc] = True
    blocked_controls = triangle_mask.copy()
    for color in control_colors:
        blocked = blocked_controls | _gen_dilate8(occupied_controls)
        pool = _gen_control_pool(inp, motif_mask, blocked)
        if not pool:
            return False
        r, c = random.choice(pool)
        inp[r, c] = color
        occupied_controls[r, c] = True
    return True


def _gen_attempt():
    H, W, side, colors, fill_color, motif_color = _gen_choose_scene()
    motif_result = _gen_symmetric_motif_patch(H, W)
    if motif_result is None:
        return None
    motif_patch, corner = motif_result

    placed = _gen_place_motif(H, W, side, fill_color, motif_color, motif_patch, corner)
    if placed is None:
        return None
    output, triangle_mask, motif_mask = placed

    inp = output.copy()
    inp[triangle_mask] = 0

    if not _gen_add_controls(inp, triangle_mask, motif_mask, corner, side, colors, fill_color, motif_color):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, output):
        return None
    return inp


def generate_input():
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 902510d5 input")
