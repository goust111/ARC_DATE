import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


def main(grid: np.ndarray) -> np.ndarray:
    """Complete blue guide lines from border marker pairs and recolor every red object touched by the completed lines.

    Input structure
    - The grid has one background color, one sparse singleton marker color, and one object color.
    - Marker pixels on matching left/right borders define horizontal guide rows; marker pixels on matching top/bottom borders define vertical guide columns.

    Rule
    - Draw each guide row or column across the whole grid in the marker color.
    - Any 4-connected object-color component that overlaps or 4-neighbor touches a completed guide line is recolored entirely to the marker color.

    Key insight
    - The border markers are not objects to move; they are controls for guide lines whose contact with object components selects the objects to recolor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = get_bg(g)
    non_bg = colors_present(g, exclude={bg})
    assert len(non_bg) == 2

    guide_specs = []
    for color in non_bg:
        comps = find_components(g, bg=bg, colors=[color], connectivity=4)
        horizontal_rows = [r for r in range(h) if g[r, 0] == color and g[r, w - 1] == color]
        vertical_cols = [c for c in range(w) if g[0, c] == color and g[h - 1, c] == color]
        if comps and all(comp["area"] == 1 for comp in comps) and (horizontal_rows or vertical_cols):
            guide_specs.append((color, horizontal_rows, vertical_cols))

    assert len(guide_specs) == 1
    guide_color, horizontal_rows, vertical_cols = guide_specs[0]
    object_colors = [color for color in non_bg if color != guide_color]
    assert len(object_colors) == 1
    object_color = object_colors[0]

    guide = np.zeros_like(g, dtype=bool)
    for r in horizontal_rows:
        guide[r, :] = True
    for c in vertical_cols:
        guide[:, c] = True
    assert guide.any()

    touched = guide.copy()
    touched[1:, :] |= guide[:-1, :]
    touched[:-1, :] |= guide[1:, :]
    touched[:, 1:] |= guide[:, :-1]
    touched[:, :-1] |= guide[:, 1:]

    out = g.copy()
    for comp in find_components(g, bg=bg, colors=[object_color], connectivity=4):
        coords = comp["coords"]
        if touched[coords[:, 0], coords[:, 1]].any():
            out[coords[:, 0], coords[:, 1]] = guide_color

    out[guide] = guide_color
    return out

# --------------- generator (o5x) ---------------

# Colors and canvas bounds.
GEN_BG_COLOR, GEN_GUIDE_COLOR, GEN_OBJECT_COLOR = 0, 1, 2
GRID_HEIGHT_RANGE, GRID_WIDTH_RANGE = (10, 24), (14, 26)  # rng highs are exclusive
# Guide-line sampling.
LINE_COUNT_CHOICES = ((1, 0), (2, 0), (0, 1), (1, 1), (2, 1))
LINE_POSITION_MIN, LINE_HIGH_OFFSET, MIN_LINE_SPACING = 1, 2, 3
# Rectangle/object sampling.
RECT_COUNT_RANGE = (5, 11)
RECT_SIDE_RANGE = (2, 10)
TOUCHED_COUNT_RANGE, UNTOUCHED_COUNT_RANGE = (1, 7), (2, 4)
MIN_RECT_AREA = RECT_SIDE_RANGE[0] ** 2
RECT_AREA_BUDGET_FRAC = 0.43
# Validation and retry limits.
GENERATION_RETRIES = 300


def _gen_spaced_positions(rng, lo, hi, count):
    vals = list(range(int(lo), int(hi) + 1))
    if count == 0:
        return []
    if count == 1:
        return [int(rng.choice(vals))]
    pairs = [(a, b) for i, a in enumerate(vals) for b in vals[i + 1 :] if b - a >= MIN_LINE_SPACING]
    if not pairs:
        return None
    return list(pairs[int(rng.integers(len(pairs)))])


def _gen_touch_mask(guide):
    touched = guide.copy()
    touched[1:, :] |= guide[:-1, :]
    touched[:-1, :] |= guide[1:, :]
    touched[:, 1:] |= guide[:, :-1]
    touched[:, :-1] |= guide[:, 1:]
    return touched


def _gen_mark_blocked(blocked, r0, c0, rh, cw):
    h, w = blocked.shape
    r1, c1 = r0 + rh, c0 + cw
    blocked[max(0, r0 - 1) : min(h, r1 + 1), c0:c1] = True
    blocked[r0:r1, max(0, c0 - 1) : min(w, c1 + 1)] = True


def _gen_rect_candidates(blocked, marker_mask, touch_mask, want_touched, max_area):
    h, w = blocked.shape
    out = []
    for rh in range(*RECT_SIDE_RANGE):
        for cw in range(*RECT_SIDE_RANGE):
            area = rh * cw
            if area > max_area:
                continue
            for r0 in range(0, h - rh + 1):
                r1 = r0 + rh
                for c0 in range(0, w - cw + 1):
                    c1 = c0 + cw
                    if blocked[r0:r1, c0:c1].any() or marker_mask[r0:r1, c0:c1].any():
                        continue
                    touched = bool(touch_mask[r0:r1, c0:c1].any())
                    if touched == want_touched:
                        out.append((r0, c0, rh, cw, area))
    return out


def _gen_guide_masks(rng, h, w):
    """Sample guide positions and build completed-guide and border-marker masks."""
    h_count, v_count = LINE_COUNT_CHOICES[int(rng.integers(len(LINE_COUNT_CHOICES)))]
    rows = _gen_spaced_positions(rng, LINE_POSITION_MIN, h - LINE_HIGH_OFFSET, h_count)
    cols = _gen_spaced_positions(rng, LINE_POSITION_MIN, w - LINE_HIGH_OFFSET, v_count)
    if rows is None or cols is None:
        return None
    guide = np.zeros((h, w), dtype=bool)
    marker_mask = np.zeros((h, w), dtype=bool)
    for r in rows:
        guide[r, :] = True
        marker_mask[r, 0] = True
        marker_mask[r, w - 1] = True
    for c in cols:
        guide[:, c] = True
        marker_mask[0, c] = True
        marker_mask[h - 1, c] = True
    return guide, marker_mask


def _gen_object_labels(rng):
    rect_count = int(rng.integers(*RECT_COUNT_RANGE))
    min_untouched = max(UNTOUCHED_COUNT_RANGE[0], rect_count - TOUCHED_COUNT_RANGE[1])
    max_untouched = min(UNTOUCHED_COUNT_RANGE[1], rect_count - TOUCHED_COUNT_RANGE[0])
    untouched_count = int(rng.integers(min_untouched, max_untouched + 1))
    touched_count = rect_count - untouched_count
    labels = [True] * touched_count + [False] * untouched_count
    rng.shuffle(labels)
    return labels, touched_count, untouched_count


def _gen_place_rects(rng, shape, marker_mask, touch_mask, labels, touched_count, untouched_count):
    """Place rectangles in sampled label order while preserving area for later ones."""
    blocked = marker_mask.copy()
    rects = []
    used_area = 0
    h, w = shape
    area_budget = int(h * w * RECT_AREA_BUDGET_FRAC)
    remaining_touched = touched_count
    remaining_untouched = untouched_count
    for want_touched in labels:
        remaining_total_after = remaining_touched + remaining_untouched - 1
        max_area = area_budget - used_area - MIN_RECT_AREA * remaining_total_after
        if max_area < MIN_RECT_AREA:
            return None
        candidates = _gen_rect_candidates(blocked, marker_mask, touch_mask, want_touched, max_area)
        if not candidates:
            return None
        by_shape = {}
        for cand in candidates:
            by_shape.setdefault((cand[2], cand[3]), []).append(cand)
        shapes = list(by_shape)
        shape = shapes[int(rng.integers(len(shapes)))]
        shape_candidates = by_shape[shape]
        r0, c0, rh, cw, area = shape_candidates[int(rng.integers(len(shape_candidates)))]
        rects.append((r0, c0, rh, cw, want_touched))
        _gen_mark_blocked(blocked, r0, c0, rh, cw)
        used_area += area
        if want_touched:
            remaining_touched -= 1
        else:
            remaining_untouched -= 1
    return rects


def _gen_render_pair(shape, guide, marker_mask, rects):
    out = np.full(shape, GEN_BG_COLOR, dtype=int)
    for r0, c0, rh, cw, is_touched in rects:
        color = GEN_GUIDE_COLOR if is_touched else GEN_OBJECT_COLOR
        paste(out, np.full((rh, cw), color, dtype=int), r0, c0)
    out[guide] = GEN_GUIDE_COLOR

    inp = out.copy()
    inp[guide] = GEN_BG_COLOR
    for r0, c0, rh, cw, _ in rects:
        paste(inp, np.full((rh, cw), GEN_OBJECT_COLOR, dtype=int), r0, c0)
    inp[marker_mask] = GEN_GUIDE_COLOR
    return inp, out


def _gen_attempt():
    rng = np.random.default_rng()
    h = int(rng.integers(*GRID_HEIGHT_RANGE))
    w = int(rng.integers(*GRID_WIDTH_RANGE))

    guide_pair = _gen_guide_masks(rng, h, w)
    if guide_pair is None:
        return None
    guide, marker_mask = guide_pair
    touch_mask = _gen_touch_mask(guide)
    labels, touched_count, untouched_count = _gen_object_labels(rng)
    rects = _gen_place_rects(rng, (h, w), marker_mask, touch_mask, labels, touched_count, untouched_count)
    if rects is None:
        return None

    inp, out = _gen_render_pair((h, w), guide, marker_mask, rects)
    bg_count = np.count_nonzero(inp == GEN_BG_COLOR)
    non_bg_max = max(np.count_nonzero(inp == GEN_GUIDE_COLOR), np.count_nonzero(inp == GEN_OBJECT_COLOR))
    if bg_count <= non_bg_max:
        return None
    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
