import numpy as np

from arc_utils import bbox, colors_present, get_bg, in_bounds, paste


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _arm_length(mask: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    length = 0
    rr = r + dr
    cc = c + dc
    while in_bounds(mask.shape, rr, cc) and mask[rr, cc]:
        length += 1
        rr += dr
        cc += dc
    return length


def _marker_stem_direction(g: np.ndarray, color: int) -> tuple[int, int] | None:
    mask = g == color
    coords = np.argwhere(mask)

    junctions = []
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        degree = sum(
            in_bounds(mask.shape, r + dr, c + dc) and mask[r + dr, c + dc]
            for dr, dc in _DIRS
        )
        if degree >= 3:
            junctions.append((r, c))
    if len(junctions) != 1:
        return None

    jr, jc = junctions[0]
    if any(int(r) != jr and int(c) != jc for r, c in coords):
        return None

    lengths = [_arm_length(mask, jr, jc, dr, dc) for dr, dc in _DIRS]
    longest = max(lengths)
    if longest == 0 or lengths.count(longest) != 1:
        return None
    return _DIRS[lengths.index(longest)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the cross marker and add a reflected adjacent copy of the other foreground sprite.

    Input structure
    - The background is the dominant color, and exactly two foreground colors are present.
    - One foreground color forms the sprite to preserve and duplicate.
    - The other foreground color forms a one-pixel-thick row/column marker with one junction and one uniquely longest arm.

    Rule
    - Interpret the marker's uniquely longest arm as a stem direction.
    - Rotate that stem direction 90 degrees in row/column coordinates to choose the side of the sprite's bounding box.
    - Mirror the sprite vertically for an above/below copy, or horizontally for a left/right copy, and paste it adjacent to that side.
    - Output only the original sprite and its reflected copy; the marker is erased to background.

    Key insight
    - The marker is not a target location: its asymmetric arm encodes which adjacent side receives the mirrored sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2

    marker_candidates = []
    for color in colors:
        stem = _marker_stem_direction(g, color)
        if stem is not None:
            marker_candidates.append((color, stem))
    assert len(marker_candidates) == 1, "expected exactly one asymmetric cross marker"

    marker_color, stem = marker_candidates[0]
    sprite_colors = [color for color in colors if color != marker_color]
    assert len(sprite_colors) == 1
    sprite_color = sprite_colors[0]

    sprite_mask = g == sprite_color
    sprite_bbox = bbox(sprite_mask)
    assert sprite_bbox is not None
    r0, c0, r1, c1 = sprite_bbox
    sprite = sprite_mask[r0 : r1 + 1, c0 : c1 + 1]
    h, w = sprite.shape

    dr, dc = stem
    copy_dr, copy_dc = -dc, dr
    if copy_dr:
        reflected = np.flipud(sprite)
        target_r = r1 + 1 if copy_dr > 0 else r0 - h
        target_c = c0
    else:
        assert copy_dc
        reflected = np.fliplr(sprite)
        target_r = r0
        target_c = c1 + 1 if copy_dc > 0 else c0 - w

    H, W = g.shape
    assert 0 <= target_r and 0 <= target_c
    assert target_r + h <= H and target_c + w <= W

    out = np.full_like(g, bg)
    out[sprite_mask] = sprite_color
    target = out[target_r : target_r + h, target_c : target_c + w]
    assert np.all(target[reflected] == bg)
    target[reflected] = sprite_color
    return out

# --------------- generator (o5x) ---------------

# Fixed palette.
_GEN_BG, _GEN_SPRITE_COLOR, _GEN_MARKER_COLOR = 0, 4, 6

# Canvas and sprite bounds; max values are exclusive for rng/range calls.
_GEN_GRID_H_RANGE, _GEN_GRID_W_RANGE = (6, 13), (10, 14)
_GEN_SPRITE_H_RANGE, _GEN_SPRITE_W_RANGE = (3, 6), (3, 8)

# Sprite structure constraints.
_GEN_SPRITE_MIN_DENSITY_DIVISOR = 2
_GEN_SPRITE_MAX_DENSITY = 0.61
_GEN_COMPONENT_COUNTS = (1, 2, 3)
_GEN_COMPONENT_GAP, _GEN_SPRITE_MASK_RETRIES = 1, 50

# Copy and marker geometry.
_GEN_COPY_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_GEN_MARKER_OPPOSITE_LENGTHS = (0, 1)
_GEN_MARKER_STEM_LEN, _GEN_MARKER_MIN_DISTANCE_SQ = 2, 4

_GEN_ATTEMPTS = 300  # Retry control.

def _gen_choice(rng, items):
    return items[int(rng.integers(len(items)))]

def _gen_sprite_dims(H, W, rng):
    dims = []
    for h in range(*_GEN_SPRITE_H_RANGE):
        for w in range(*_GEN_SPRITE_W_RANGE):
            if 2 * h <= H or 2 * w <= W:
                dims.append((h, w))
    if not dims:
        return None
    return _gen_choice(rng, dims)


def _gen_copy_direction(H, W, h, w, rng):
    copy_dirs = [d for d in _GEN_COPY_DIRS if (d[0] and 2 * h <= H) or (d[1] and 2 * w <= W)]
    if not copy_dirs:
        return None
    return _gen_choice(rng, copy_dirs)


def _gen_positive_parts(total, n):
    if n == 1:
        if total >= 1:
            return [(int(total),)]
        return []
    out = []
    for first in range(1, total - n + 2):
        for rest in _gen_positive_parts(total - first, n - 1):
            out.append((int(first),) + rest)
    return out


def _gen_size_splits(total, lows, highs):
    if len(lows) == 1:
        if lows[0] <= total <= highs[0]:
            return [(int(total),)]
        return []
    out = []
    lo = lows[0]
    hi = highs[0]
    rest_low = sum(lows[1:])
    rest_high = sum(highs[1:])
    for first in range(lo, hi + 1):
        if rest_low <= total - first <= rest_high:
            for rest in _gen_size_splits(total - first, lows[1:], highs[1:]):
                out.append((int(first),) + rest)
    return out


def _gen_connected_full_bbox(h, w, area, rng):
    mask = np.zeros((h, w), dtype=bool)
    if h == 1 or w == 1:
        mask[:, :] = True
        return mask

    corners = (
        ((0, 0), (h - 1, w - 1)),
        ((0, w - 1), (h - 1, 0)),
        ((h - 1, 0), (0, w - 1)),
        ((h - 1, w - 1), (0, 0)),
    )
    start, end = corners[int(rng.integers(len(corners)))]
    dr = 1 if end[0] > start[0] else -1
    dc = 1 if end[1] > start[1] else -1
    moves = [(dr, 0)] * (h - 1) + [(0, dc)] * (w - 1)
    rng.shuffle(moves)

    r, c = start
    mask[r, c] = True
    for rr, cc in moves:
        r += rr
        c += cc
        mask[r, c] = True

    while int(mask.sum()) < area:
        frontier = []
        for r, c in np.argwhere(mask):
            for dr, dc in _GEN_COPY_DIRS:
                rr = int(r) + dr
                cc = int(c) + dc
                if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                    frontier.append((rr, cc))
        rr, cc = frontier[int(rng.integers(len(frontier)))]
        mask[rr, cc] = True
    return mask


def _gen_component_layouts(h, w, ncomp, area):
    if ncomp == 1:
        low = h + w - 1
        if low <= area <= h * w:
            return [("single", (w,), (area,))]
        return []

    layouts = []
    layouts.extend(_gen_span_layouts("vertical", h, w, ncomp, area))
    layouts.extend(_gen_span_layouts("horizontal", h, w, ncomp, area))
    return layouts


def _gen_span_layouts(axis, h, w, ncomp, area):
    if axis == "vertical":
        extent, cross = w, h
    else:
        extent, cross = h, w

    if extent < ncomp + _GEN_COMPONENT_GAP * (ncomp - 1):
        return []

    layouts = []
    available = extent - _GEN_COMPONENT_GAP * (ncomp - 1)
    for spans in _gen_positive_parts(available, ncomp):
        lows = [cross + span - 1 for span in spans]
        highs = [cross * span for span in spans]
        for sizes in _gen_size_splits(area, lows, highs):
            layouts.append((axis, spans, sizes))
    return layouts


def _gen_paint_component_layout(h, w, axis, spans, sizes, rng):
    if axis == "single":
        return _gen_connected_full_bbox(h, w, sizes[0], rng)

    mask = np.zeros((h, w), dtype=bool)
    offset = 0
    for span, size in zip(spans, sizes):
        if axis == "vertical":
            mask[:, offset : offset + span] = _gen_connected_full_bbox(h, span, size, rng)
        else:
            mask[offset : offset + span, :] = _gen_connected_full_bbox(span, w, size, rng)
        offset += span + _GEN_COMPONENT_GAP
    return mask


def _gen_sprite_mask(h, w, rng):
    cells = h * w
    lo = (cells + _GEN_SPRITE_MIN_DENSITY_DIVISOR - 1) // _GEN_SPRITE_MIN_DENSITY_DIVISOR
    hi = max(lo, int(np.floor(_GEN_SPRITE_MAX_DENSITY * cells)))

    for _ in range(_GEN_SPRITE_MASK_RETRIES):
        area = int(rng.integers(lo, hi + 1))
        feasible = []
        for ncomp in _GEN_COMPONENT_COUNTS:
            layouts = _gen_component_layouts(h, w, ncomp, area)
            if layouts:
                feasible.append((ncomp, layouts))
        if not feasible:
            continue

        _, layouts = _gen_choice(rng, feasible)
        axis, spans, sizes = _gen_choice(rng, layouts)
        mask = _gen_paint_component_layout(h, w, axis, spans, sizes, rng)

        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    return None


def _gen_marker_patch(stem, opposite_len):
    dr, dc = stem
    pr, pc = -dc, dr
    cells = [(0, 0)]
    cells.extend((dr * i, dc * i) for i in range(1, _GEN_MARKER_STEM_LEN + 1))
    cells.append((pr, pc))
    cells.append((-pr, -pc))
    if opposite_len:
        cells.append((-dr, -dc))

    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = [(r - min_r, c - min_c) for r, c in cells]
    H = max(r for r, _ in norm) + 1
    W = max(c for _, c in norm) + 1
    patch = np.zeros((H, W), dtype=int)
    for r, c in norm:
        patch[r, c] = _GEN_MARKER_COLOR
    return patch


def _gen_sprite_origins(H, W, h, w, copy_dir):
    dr, dc = copy_dir
    origins = []
    if dr < 0:
        row_range = range(h, H - h + 1)
        col_range = range(0, W - w + 1)
    elif dr > 0:
        row_range = range(0, H - 2 * h + 1)
        col_range = range(0, W - w + 1)
    elif dc < 0:
        row_range = range(0, H - h + 1)
        col_range = range(w, W - w + 1)
    else:
        row_range = range(0, H - h + 1)
        col_range = range(0, W - 2 * w + 1)

    for r in row_range:
        for c in col_range:
            origins.append((int(r), int(c)))
    return origins


def _gen_bboxes_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _gen_marker_locations(H, W, marker, sprite_mask, sprite_r, sprite_c):
    mh, mw = marker.shape
    sprite_coords = np.argwhere(sprite_mask) + np.array([sprite_r, sprite_c])
    marker_local = np.argwhere(marker == _GEN_MARKER_COLOR)
    sr0 = sprite_r
    sc0 = sprite_c
    sr1 = sprite_r + sprite_mask.shape[0] - 1
    sc1 = sprite_c + sprite_mask.shape[1] - 1
    sprite_box = (sr0, sc0, sr1, sc1)

    locs = []
    for mr in range(0, H - mh + 1):
        for mc in range(0, W - mw + 1):
            marker_box = (mr, mc, mr + mh - 1, mc + mw - 1)
            if _gen_bboxes_overlap(sprite_box, marker_box):
                continue
            marker_coords = marker_local + np.array([mr, mc])
            diffs = sprite_coords[:, None, :] - marker_coords[None, :, :]
            if int(np.min(np.sum(diffs * diffs, axis=2))) < _GEN_MARKER_MIN_DISTANCE_SQ:
                continue
            locs.append((int(mr), int(mc)))
    return locs


def _gen_placement_choices(H, W, marker_patches, sprite_mask, origins):
    choices = []
    for sprite_r, sprite_c in origins:
        for marker in marker_patches:
            for marker_r, marker_c in _gen_marker_locations(H, W, marker, sprite_mask, sprite_r, sprite_c):
                choices.append((sprite_r, sprite_c, marker, marker_r, marker_c))
    return choices


def _gen_render_output(H, W, sprite, sprite_r, sprite_c, copy_dir):
    h, w = sprite.shape
    copy_dr, copy_dc = copy_dir
    out = np.full((H, W), _GEN_BG, dtype=int)
    paste(out, sprite, sprite_r, sprite_c, transparent=_GEN_BG)
    if copy_dr:
        reflected = np.flipud(sprite)
        copy_r = sprite_r + h if copy_dr > 0 else sprite_r - h
        copy_c = sprite_c
    else:
        reflected = np.fliplr(sprite)
        copy_r = sprite_r
        copy_c = sprite_c + w if copy_dc > 0 else sprite_c - w
    paste(out, reflected, copy_r, copy_c, transparent=_GEN_BG)
    return out, reflected, copy_r, copy_c


def _gen_verified_input(inp, out):
    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def _gen_attempt():
    rng = np.random.default_rng()
    H = int(rng.integers(*_GEN_GRID_H_RANGE))
    W = int(rng.integers(*_GEN_GRID_W_RANGE))

    dims = _gen_sprite_dims(H, W, rng)
    if dims is None:
        return None
    h, w = dims

    sprite_mask = _gen_sprite_mask(h, w, rng)
    if sprite_mask is None:
        return None

    copy_dir = _gen_copy_direction(H, W, h, w, rng)
    if copy_dir is None:
        return None
    copy_dr, copy_dc = copy_dir
    stem = (copy_dc, -copy_dr)

    origins = _gen_sprite_origins(H, W, h, w, copy_dir)
    rng.shuffle(origins)
    marker_patches = [_gen_marker_patch(stem, int(x)) for x in rng.permutation(_GEN_MARKER_OPPOSITE_LENGTHS)]

    choices = _gen_placement_choices(H, W, marker_patches, sprite_mask, origins)
    if not choices:
        return None

    sprite_r, sprite_c, marker, marker_r, marker_c = _gen_choice(rng, choices)
    sprite = np.where(sprite_mask, _GEN_SPRITE_COLOR, _GEN_BG)

    out, reflected, copy_r, copy_c = _gen_render_output(H, W, sprite, sprite_r, sprite_c, copy_dir)
    h, w = reflected.shape
    inp = out.copy()
    copy_region = inp[copy_r : copy_r + h, copy_c : copy_c + w]
    copy_region[reflected == _GEN_SPRITE_COLOR] = _GEN_BG
    paste(inp, marker, marker_r, marker_c, transparent=_GEN_BG)
    return _gen_verified_input(inp, out)


def generate_input():
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise ValueError("failed to generate a valid 9bebae7a input")
