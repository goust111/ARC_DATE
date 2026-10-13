import numpy as np

from arc_utils import colors_present, find_components, get_bg
import random


def _row_span_interior(g: np.ndarray, coords: np.ndarray, bg: int) -> tuple[tuple[int, int], ...]:
    occupied = {(int(r), int(c)) for r, c in coords}
    interior: list[tuple[int, int]] = []

    for r in sorted({r for r, _ in occupied}):
        cols = sorted(c for rr, c in occupied if rr == r)
        if len(cols) < 2:
            continue
        for c in range(cols[0] + 1, cols[-1]):
            if (r, c) in occupied:
                continue
            assert int(g[r, c]) == int(bg), "outline interior cells must be background"
            interior.append((r, c))

    return tuple(interior)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the source-color pixels into the unique outline interior whose size matches the source count.

    Input structure
    - The grid has a dominant background and exactly two foreground colors.
    - One foreground color forms 8-connected outline objects with row-wise interior cells.
    - The other foreground color supplies a set of source pixels away from the chosen outline.
    Rule
    - For each possible outline color, compute each 8-connected outline object's
      row-span interior: background pixels strictly between the leftmost and
      rightmost outline pixels on each occupied row.
    - Select the unique outline interior whose cell count equals the total
      number of pixels of the other foreground color.
    - Erase the source pixels and paint that matched interior with the source color.
    Key insight
    - The colors are roles, not fixed IDs: the source color is identified by
      counting, and the destination is the outline with a matching interior area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected exactly two foreground colors"

    candidates: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for outline_color in colors:
        source_colors = [c for c in colors if c != outline_color]
        assert len(source_colors) == 1
        source_color = int(source_colors[0])
        source_count = int(np.count_nonzero(g == source_color))
        assert source_count > 0, "source color must be present"

        components = find_components(
            g,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[outline_color],
        )
        assert components, "outline color must have at least one object"

        for component in components:
            interior = _row_span_interior(g, component["coords"], bg)
            if len(interior) == source_count:
                candidates.append((source_color, interior))

    assert len(candidates) == 1, "expected exactly one outline interior matching the source count"
    source_color, interior = candidates[0]

    out = g.copy()
    out[g == source_color] = int(bg)
    for r, c in interior:
        out[r, c] = source_color
    return out

# --------------- generator (att1) ---------------

# Canvas and palette bounds.
GRID_SIZE_RANGE = (13, 16)
ARC_COLOR_IDS = range(10)

# Outline shape constraints.
CAVITY_SIZE_POOL = range(1, 7)
SHAPE_PIXEL_BOUNDS = (3, 8)
SHAPE_HEIGHT_BOUNDS = (2, 3)
SHAPE_WIDTH_BOUNDS = (2, 5)
SHAPE_STRATEGIES = ("funnel", "U", "bridge")
SHAPE_RETRY_LIMIT = 200
FLIP_PROB = 0.5

# Strategy-specific shape knobs.
FUNNEL_TOP_GAP_MAX = 3
U_MAX_FACTOR_GAP = 4
U_ARM_HEIGHT_BOUNDS = (1, 2)
BRIDGE_ARM_GAP_RANGE = (1, 3)
BRIDGE_ARM_ROWS_RANGE = (1, 2)
BRIDGE_EXTEND_PROB = 0.3

# Placement and retry limits.
MAX_FOREGROUND_DENSITY = 0.10
OUTLINE_BBOX_GAP = 1
SOURCE_OUTLINE_PADDING = 3
PLACEMENT_RETRY_LIMIT = 100
GEN_ATTEMPT_LIMIT = 300


def _cavity_rows(pixels):
    rows = {}
    for r, c in pixels:
        rows.setdefault(r, []).append(c)
    return rows


def _cavity_size(pixels):
    """Total row-span interior (cavity) cells."""
    return len(_cavity_cells(pixels))


def _cavity_cells(pixels):
    """Return cavity cell coordinates."""
    pixel_set = set(map(tuple, pixels))
    rows = _cavity_rows(pixels)
    cells = []
    for r, cols in rows.items():
        s = sorted(cols)
        if len(s) >= 2:
            cells.extend((r, c) for c in range(s[0] + 1, s[-1]) if (r, c) not in pixel_set)
    return cells


def _is_8_connected(pixels):
    """Check if pixels form a single 8-connected component."""
    pixel_set = set(pixels)
    if not pixel_set:
        return False
    visited = {(start := next(iter(pixel_set)))}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in pixel_set and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
    return len(visited) == len(pixel_set)


def _shape_dimensions(pixels):
    return max(r for r, _ in pixels) + 1, max(c for _, c in pixels) + 1


def _validate_shape(pixels, target_cavity):
    """Check all shape constraints. Returns True if valid."""
    n = len(pixels)
    if n < SHAPE_PIXEL_BOUNDS[0] or n > SHAPE_PIXEL_BOUNDS[1]:
        return False
    height, width = _shape_dimensions(pixels)
    if height < SHAPE_HEIGHT_BOUNDS[0] or height > SHAPE_HEIGHT_BOUNDS[1]:
        return False
    if width < SHAPE_WIDTH_BOUNDS[0] or width > SHAPE_WIDTH_BOUNDS[1]:
        return False
    if not _is_8_connected(pixels):
        return False
    if _cavity_size(pixels) != target_cavity:
        return False
    return True


def _maybe_flip(pixels):
    """Random vertical flip."""
    if random.random() < FLIP_PROB:
        max_r = max(r for r, _ in pixels)
        return {(max_r - r, c) for r, c in pixels}
    return pixels


def _normalize_columns(pixels):
    min_c = min(c for _, c in pixels)
    return {(r, c - min_c) for r, c in pixels}


def _build_funnel_shape(target_cavity):
    h = random.randint(SHAPE_HEIGHT_BOUNDS[0], SHAPE_HEIGHT_BOUNDS[1])
    top_gap = random.randint(1, min(FUNNEL_TOP_GAP_MAX, target_cavity + 1))
    left, right = 0, top_gap + 1

    rows_lr = []
    for ri in range(h):
        rows_lr.append((left, right))
        if ri < h - 1:
            left += random.choice([0, 1])
            right -= random.choice([0, 1])
            if right <= left:
                right = left
                break

    last_l, last_r = rows_lr[-1]
    if last_r - last_l > 1:
        if len(rows_lr) < SHAPE_HEIGHT_BOUNDS[1]:
            mid = (last_l + last_r) // 2
            rows_lr.append((mid, mid))
        else:
            return None

    pixels = set()
    for ri, (l, r) in enumerate(rows_lr):
        pixels.add((ri, l))
        if r != l:
            pixels.add((ri, r))

    pixels = _normalize_columns(pixels)
    if _validate_shape(pixels, target_cavity):
        return pixels
    return None


def _build_u_shape(target_cavity):
    min_arm_h, max_arm_h = U_ARM_HEIGHT_BOUNDS
    factors = [(g, target_cavity // g) for g in range(1, min(U_MAX_FACTOR_GAP, target_cavity) + 1)
               if target_cavity % g == 0 and min_arm_h <= target_cavity // g <= max_arm_h
               and g + 2 <= SHAPE_WIDTH_BOUNDS[1]]
    if not factors:
        return None
    arm_gap, arm_height = random.choice(factors)

    pixels = set()
    for ri in range(arm_height):
        pixels.add((ri, 0))
        pixels.add((ri, arm_gap + 1))
    for c in range(arm_gap + 2):
        pixels.add((arm_height, c))

    if _validate_shape(pixels, target_cavity):
        return pixels
    return None


def _build_bridge_shape(target_cavity):
    arm_gap = random.randint(BRIDGE_ARM_GAP_RANGE[0], BRIDGE_ARM_GAP_RANGE[1])
    n_arm_rows = random.randint(BRIDGE_ARM_ROWS_RANGE[0], BRIDGE_ARM_ROWS_RANGE[1])
    if arm_gap * n_arm_rows != target_cavity:
        return None

    left_arm, right_arm = 0, arm_gap + 1
    bridge_left = left_arm + 1
    bridge_right = right_arm - 1
    if random.random() < BRIDGE_EXTEND_PROB:
        bridge_left = left_arm
    if random.random() < BRIDGE_EXTEND_PROB:
        bridge_right = right_arm

    pixels = set()
    for c in range(bridge_left, bridge_right + 1):
        pixels.add((0, c))
    for ri in range(1, n_arm_rows + 1):
        pixels.add((ri, left_arm))
        pixels.add((ri, right_arm))

    pixels = _normalize_columns(pixels)
    if _validate_shape(pixels, target_cavity):
        return pixels
    return None


def _build_outline_shape(target_cavity):
    """Build one valid outline shape with the requested row-span cavity size."""
    builders = {
        "funnel": _build_funnel_shape,
        "U": _build_u_shape,
        "bridge": _build_bridge_shape,
    }
    for _ in range(SHAPE_RETRY_LIMIT):
        pixels = builders[random.choice(SHAPE_STRATEGIES)](target_cavity)
        if pixels is not None:
            return list(_maybe_flip(pixels))

    return None


def _choose_canvas_and_palette():
    H = random.randint(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1])
    W = random.randint(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1])
    bg = random.randint(0, 9)
    fg = random.sample([c for c in ARC_COLOR_IDS if c != bg], 2)
    outline_color, source_color = fg
    return H, W, bg, outline_color, source_color


def _choose_outline_shapes():
    cavities = random.sample(CAVITY_SIZE_POOL, 2)
    s1, s2 = cavities

    shape1 = _build_outline_shape(s1)
    if shape1 is None:
        return None
    shape2 = _build_outline_shape(s2)
    if shape2 is None:
        return None

    matched_idx = random.choice([0, 1])
    source_count = [s1, s2][matched_idx]
    return shape1, shape2, matched_idx, source_count


def _bboxes_overlap_with_gap(b1, b2, gap):
    return (b1[0] - gap <= b2[2] + gap and b1[2] + gap >= b2[0] - gap and
            b1[1] - gap <= b2[3] + gap and b1[3] + gap >= b2[1] - gap)


def _place_outline_pair(H, W, shape1, shape2):
    sh1_h, sh1_w = _shape_dimensions(shape1)
    sh2_h, sh2_w = _shape_dimensions(shape2)

    for _ in range(PLACEMENT_RETRY_LIMIT):
        r1 = random.randint(0, H - sh1_h)
        c1 = random.randint(0, W - sh1_w)
        r2 = random.randint(0, H - sh2_h)
        c2 = random.randint(0, W - sh2_w)
        b1 = (r1, c1, r1 + sh1_h - 1, c1 + sh1_w - 1)
        b2 = (r2, c2, r2 + sh2_h - 1, c2 + sh2_w - 1)
        if _bboxes_overlap_with_gap(b1, b2, OUTLINE_BBOX_GAP):
            continue
        return (r1, c1, b1), (r2, c2, b2)
    return None


def _paint_cells(grid, cells, color):
    if not cells:
        return
    rr, cc = np.array(cells, dtype=int).T
    grid[rr, cc] = color


def _expanded_bbox(bbox, H, W, pad):
    return (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
            min(H - 1, bbox[2] + pad), min(W - 1, bbox[3] + pad))


def _source_candidates(grid, bg, b1, b2):
    H, W = grid.shape
    ex1 = _expanded_bbox(b1, H, W, SOURCE_OUTLINE_PADDING)
    ex2 = _expanded_bbox(b2, H, W, SOURCE_OUTLINE_PADDING)
    return [(r, c) for r in range(H) for c in range(W)
            if grid[r, c] == bg
            and not (ex1[0] <= r <= ex1[2] and ex1[1] <= c <= ex1[3])
            and not (ex2[0] <= r <= ex2[2] and ex2[1] <= c <= ex2[3])]


def _oracle_checked_input(inp, output):
    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def _gen_attempt():
    H, W, bg, outline_color, source_color = _choose_canvas_and_palette()
    outline_shapes = _choose_outline_shapes()
    if outline_shapes is None:
        return None
    shape1, shape2, matched_idx, source_count = outline_shapes

    if len(shape1) + len(shape2) + source_count > int(H * W * MAX_FOREGROUND_DENSITY):
        return None

    placement = _place_outline_pair(H, W, shape1, shape2)
    if placement is None:
        return None
    (r1, c1, b1), (r2, c2, b2) = placement

    pos1 = [(r + r1, c + c1) for r, c in shape1]
    pos2 = [(r + r2, c + c2) for r, c in shape2]
    grid = np.full((H, W), bg, dtype=int)
    _paint_cells(grid, pos1 + pos2, outline_color)

    output = grid.copy()
    cavity = _cavity_cells(pos1 if matched_idx == 0 else pos2)
    assert len(cavity) == source_count
    _paint_cells(output, cavity, source_color)

    valid = _source_candidates(grid, bg, b1, b2)
    if len(valid) < source_count:
        return None

    inp = grid.copy()
    _paint_cells(inp, random.sample(valid, source_count), source_color)

    return _oracle_checked_input(inp, output)


def generate_input():
    for _ in range(GEN_ATTEMPT_LIMIT):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_ATTEMPT_LIMIT} attempts")
