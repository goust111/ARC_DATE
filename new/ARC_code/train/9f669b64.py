import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste
import random


def _center(obj: dict, axis: int) -> float:
    r0, c0, r1, c1 = obj["bbox"]
    return ((r0 + r1) / 2, (c0 + c1) / 2)[axis]


def _span(obj: dict, axis: int) -> int:
    return bbox_h(obj["bbox"]) if axis == 0 else bbox_w(obj["bbox"])


def _is_solid_rectangle(g: np.ndarray, obj: dict) -> bool:
    r0, c0, r1, c1 = obj["bbox"]
    color = int(obj["color"])
    return (
        int(obj["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1)
        and np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color)
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push the central solid rectangle to the grid edge, splitting the perpendicular target rectangle around its path.

    Input structure
    - The background is the dominant color.
    - There are exactly three monochrome foreground objects arranged on one
      horizontal or vertical axis.
    - The middle object on that axis is a solid rectangular moving object.
    - One end object is a solid rectangle or bar whose long side is perpendicular
      to the movement axis; this is the target to split.

    Rule
    - Move the middle rectangle along the axis toward the target until it is
      flush with the grid edge in that direction.
    - Clear the original middle rectangle and the original target.
    - Split the target through the moving rectangle's two-cell-wide path and
      shift the two halves one cell outward perpendicular to the movement.
    - Keep the other end object unchanged.

    Key insight
    - The non-target end object acts as a pusher; the central rectangle travels
      away from it through the perpendicular target, opening a matching gap.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    H, W = g.shape

    objects = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert len(objects) == 3, "expected exactly three foreground objects"

    row_centers = [_center(obj, 0) for obj in objects]
    col_centers = [_center(obj, 1) for obj in objects]
    row_spread = max(row_centers) - min(row_centers)
    col_spread = max(col_centers) - min(col_centers)
    assert min(row_spread, col_spread) <= 0.5, "objects must share a row or column axis"
    axis = 0 if col_spread <= 0.5 else 1
    perp = 1 - axis

    ordered = sorted(objects, key=lambda obj: _center(obj, axis))
    axis_centers = [_center(obj, axis) for obj in ordered]
    assert axis_centers[0] < axis_centers[1] < axis_centers[2], "expected a unique middle object"

    source = ordered[1]
    end_objects = [ordered[0], ordered[2]]
    assert _is_solid_rectangle(g, source), "moving object must be a solid rectangle"

    target_candidates = [
        obj
        for obj in end_objects
        if _is_solid_rectangle(g, obj) and _span(obj, perp) > _span(obj, axis)
    ]
    assert len(target_candidates) == 1, "expected one perpendicular solid target"
    target = target_candidates[0]

    direction = -1 if _center(target, axis) < _center(source, axis) else 1

    sr0, sc0, sr1, sc1 = source["bbox"]
    tr0, tc0, tr1, tc1 = target["bbox"]
    source_lo = [sr0, sc0]
    source_hi = [sr1, sc1]
    target_lo = [tr0, tc0]
    target_hi = [tr1, tc1]

    source_perp_len = source_hi[perp] - source_lo[perp] + 1
    assert source_perp_len == 2, "moving object path must be two cells wide"
    assert target_lo[perp] <= source_lo[perp] <= source_hi[perp] <= target_hi[perp]
    assert (
        source_lo[perp] - target_lo[perp] == target_hi[perp] - source_hi[perp]
    ), "moving object must be centered across the target"

    out = g.copy()
    for r, c in source["coords"]:
        out[int(r), int(c)] = bg
    for r, c in target["coords"]:
        out[int(r), int(c)] = bg

    shift = source_perp_len // 2
    for r, c in target["coords"]:
        new_rc = [int(r), int(c)]
        if new_rc[perp] <= source_lo[perp]:
            new_rc[perp] -= shift
        else:
            assert new_rc[perp] >= source_hi[perp]
            new_rc[perp] += shift
        assert 0 <= new_rc[0] < H and 0 <= new_rc[1] < W, "shifted target must stay in bounds"
        out[new_rc[0], new_rc[1]] = int(target["color"])

    source_patch = crop_bbox(g, source["bbox"])
    if axis == 0:
        new_r = 0 if direction < 0 else H - source_patch.shape[0]
        new_c = source_lo[1]
    else:
        new_r = source_lo[0]
        new_c = 0 if direction < 0 else W - source_patch.shape[1]
    dest = out[new_r : new_r + source_patch.shape[0], new_c : new_c + source_patch.shape[1]]
    assert np.all(dest == bg), "moving object destination must be clear"
    paste(out, source_patch, new_r, new_c)
    return out

# --------------- generator (att2) ---------------

GEN_GRID_SIZE = 10
GEN_BG = 7
GEN_FG_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]
GEN_SOURCE_PERP = 2
GEN_SOURCE_ALONG_CHOICES = [2, 3]
GEN_TARGET_ALONG_MIN, GEN_TARGET_ALONG_MAX = 1, 4
GEN_TARGET_OVERHANG_CHOICES = (1, 2)
GEN_SOURCE_TARGET_GAP_RANGE = range(5)
GEN_OTHER_WIDTH_CHOICES = (2, 4, 6)
GEN_OTHER_MIN_STAIR_WIDTH = 4
GEN_OTHER_MIN_STAIR_HEIGHT = 2
GEN_ORIENTATION_COUNT, GEN_RETRY_LIMIT = 4, 300


def _gen_transform(grid: np.ndarray, orientation: int) -> np.ndarray:
    if orientation == 0:
        return grid.copy()
    if orientation == 1:
        return np.flipud(grid).copy()
    if orientation == 2:
        return np.rot90(grid, 1).copy()
    return np.rot90(grid, -1).copy()


def _gen_even_centered_start(center_left: int, width: int) -> int:
    return center_left - (width // 2 - 1)


def _gen_stair_patch(h: int, w: int, color: int, bg: int) -> np.ndarray:
    patch = np.full((h, w), bg, dtype=int)
    max_extra = w // 2 - 1
    thresholds = sorted(random.randint(1, h - 1) for _ in range(max_extra))
    for r in range(h):
        extra = sum(r >= threshold for threshold in thresholds)
        c0 = w // 2 - 1 - extra
        c1 = w // 2 + extra
        patch[r, c0 : c1 + 1] = color
    return patch


def _gen_place_other(grid: np.ndarray, r0: int, center_left: int, color: int, bg: int) -> None:
    h = grid.shape[0] - r0
    in_bounds_widths = [
        w
        for w in GEN_OTHER_WIDTH_CHOICES
        if 0 <= _gen_even_centered_start(center_left, w)
        and _gen_even_centered_start(center_left, w) + w <= grid.shape[1]
    ]
    solid_widths = [w for w in in_bounds_widths if w <= h]
    stair_widths = [w for w in in_bounds_widths if w >= GEN_OTHER_MIN_STAIR_WIDTH and h >= GEN_OTHER_MIN_STAIR_HEIGHT]

    kinds = []
    if solid_widths:
        kinds.append("solid")
    if stair_widths:
        kinds.append("stair")
    kind = random.choice(kinds)

    if kind == "solid":
        w = random.choice(solid_widths)
        patch = np.full((h, w), color, dtype=int)
    else:
        w = random.choice(stair_widths)
        patch = _gen_stair_patch(h, w, color, bg)
    paste(grid, patch, r0, _gen_even_centered_start(center_left, w), transparent=bg)


def _gen_choose_layout() -> tuple[int, int, int, int, int, int] | None:
    s_h = random.choice(GEN_SOURCE_ALONG_CHOICES)
    t_h = random.randint(GEN_TARGET_ALONG_MIN, GEN_TARGET_ALONG_MAX)
    overhangs = [k for k in GEN_TARGET_OVERHANG_CHOICES if 2 * k + GEN_SOURCE_PERP > t_h]
    k = random.choice(overhangs)
    gap_options = [g for g in GEN_SOURCE_TARGET_GAP_RANGE if t_h + g + s_h <= GEN_GRID_SIZE - GEN_SOURCE_PERP]
    if not gap_options:
        return None
    gap = random.choice(gap_options)
    t0_max = GEN_GRID_SIZE - GEN_SOURCE_PERP - t_h - gap - s_h
    t0 = random.randint(0, t0_max)
    center_left = random.randint(k + 1, GEN_GRID_SIZE - k - 3)
    return s_h, t_h, k, gap, t0, center_left


def _gen_canonical_pair(s_h: int, t_h: int, k: int, gap: int, t0: int, center_left: int, colors: list[int]) -> tuple[np.ndarray, np.ndarray]:
    source_color, target_color, other_color = colors
    source_patch = np.full((s_h, GEN_SOURCE_PERP), source_color, dtype=int)
    target_patch = np.full((t_h, 2 * k + GEN_SOURCE_PERP), target_color, dtype=int)
    split_half = np.full((t_h, k + 1), target_color, dtype=int)

    out = np.full((GEN_GRID_SIZE, GEN_GRID_SIZE), GEN_BG, dtype=int)
    paste(out, source_patch, 0, center_left)
    paste(out, split_half, t0, center_left - k - 1)
    paste(out, split_half, t0, center_left + GEN_SOURCE_PERP)

    source_in_r = t0 + t_h + gap
    other_r = source_in_r + s_h
    _gen_place_other(out, other_r, center_left, other_color, GEN_BG)

    inp = out.copy()
    inp[0:s_h, center_left : center_left + GEN_SOURCE_PERP] = GEN_BG
    inp[t0 : t0 + t_h, center_left - k - 1 : center_left] = GEN_BG
    inp[t0 : t0 + t_h, center_left + GEN_SOURCE_PERP : center_left + k + 3] = GEN_BG
    paste(inp, target_patch, t0, center_left - k)
    paste(inp, source_patch, source_in_r, center_left)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    colors = random.sample(GEN_FG_COLORS, 3)
    layout = _gen_choose_layout()
    if layout is None:
        return None
    inp, out = _gen_canonical_pair(*layout, colors)

    orientation = random.randrange(GEN_ORIENTATION_COUNT)
    inp = _gen_transform(inp, orientation)
    out = _gen_transform(out, orientation)
    try:
        if np.array_equal(main(inp), out):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 9f669b64 input")
