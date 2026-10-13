import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 2
PATTERN = 9
MARKER_COLORS = (1, 3, 4, 7)
SIDE_ORDER = ("top", "right", "bottom", "left")
CASE_OPTIONS = ((4, 2), (3, 3))  # (mask_size, scale)
MAX_GENERATION_ATTEMPTS = 400
MASK_ATTEMPTS = 800
INNER_STRUCTURE_ATTEMPTS = 80


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _diamond_frame(grid: np.ndarray) -> tuple[tuple[int, int, int, int], int, int, int]:
    coords = np.argwhere(grid == FRAME)
    assert coords.size > 0, "expected a red diamond frame"
    r0, c0, r1, c1 = _bbox(grid == FRAME)
    assert r0 is not None
    side = r1 - r0 + 1
    assert side == c1 - c0 + 1 and side % 2 == 1, "diamond bbox must be an odd square"
    radius = side // 2
    center_r = r0 + radius
    center_c = c0 + radius
    dists = np.abs(coords[:, 0] - center_r) + np.abs(coords[:, 1] - center_c)
    assert coords.shape[0] == 4 * radius and np.all(dists == radius), "frame must be a complete 1px diamond"
    return (int(r0), int(c0), int(r1), int(c1)), int(center_r), int(center_c), int(radius)


def _block_homogeneous_mask(crop: np.ndarray, scale: int) -> np.ndarray | None:
    side = int(crop.shape[0])
    if crop.shape[0] != crop.shape[1] or side % scale != 0:
        return None
    small = side // scale
    mask = np.zeros((small, small), dtype=bool)
    for i in range(small):
        for j in range(small):
            block = crop[i * scale : (i + 1) * scale, j * scale : (j + 1) * scale]
            value = int(block[0, 0])
            if not np.all(block == value):
                return None
            if value not in (BG, PATTERN):
                return None
            mask[i, j] = value != BG
    return mask


def _scaled_mask(grid: np.ndarray) -> tuple[np.ndarray, int]:
    box = _bbox(grid == PATTERN)
    assert box is not None, "expected a maroon scaled pattern"
    r0, c0, r1, c1 = box
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert crop.shape[0] == crop.shape[1], "scaled pattern bbox must be square"

    candidates: list[tuple[np.ndarray, int]] = []
    for scale in (2, 3):
        mask = _block_homogeneous_mask(crop, scale)
        if mask is None:
            continue
        if mask.shape[0] not in (3, 4):
            continue
        if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
            continue
        candidates.append((mask, scale))

    assert len(candidates) == 1, "scaled pattern must imply a unique mask and scale"
    return candidates[0]


def _square_origin(center_r: int, center_c: int, side: int) -> tuple[int, int]:
    return int(center_r - side // 2), int(center_c - side // 2)


def _band_index(pos: int, start: int, scale: int, span: int) -> int:
    delta = int(pos - start)
    assert 0 <= delta < span, "marker must align to the gated square span"
    assert delta % scale == scale // 2, "marker must align to a mask band centerline"
    return int(delta // scale)


def _marker_side(r: int, c: int, frame_box: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = frame_box
    if r == r0:
        return "top"
    if c == c1:
        return "right"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    raise AssertionError("marker must lie on a frame-bbox side")


def _paint_vertical(
    out: np.ndarray,
    base: np.ndarray,
    mask: np.ndarray,
    color: int,
    column: int,
    band: int,
    scale: int,
    square_r0: int,
    square_c0: int,
    reverse: bool,
) -> None:
    n = int(mask.shape[0])
    rows = range(n - 1, -1, -1) if reverse else range(n)
    for i in rows:
        if not mask[i, band]:
            break
        rr0 = square_r0 + i * scale
        rr1 = rr0 + scale
        col = square_c0 + band * scale + scale // 2
        assert col == column
        paintable = base[rr0:rr1, col] == BG
        out[rr0:rr1, col][paintable] = int(color)


def _paint_horizontal(
    out: np.ndarray,
    base: np.ndarray,
    mask: np.ndarray,
    color: int,
    row: int,
    band: int,
    scale: int,
    square_r0: int,
    square_c0: int,
    reverse: bool,
) -> None:
    n = int(mask.shape[0])
    cols = range(n - 1, -1, -1) if reverse else range(n)
    for j in cols:
        if not mask[band, j]:
            break
        cc0 = square_c0 + j * scale
        cc1 = cc0 + scale
        rr = square_r0 + band * scale + scale // 2
        assert rr == row
        paintable = base[rr, cc0:cc1] == BG
        out[rr, cc0:cc1][paintable] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a downsampled monochrome mask to gate inward rays that fill square subregions inside a diamond frame.

    Input structure:
    - The grid has a black background, one red one-pixel hollow diamond frame, several colored side markers on the frame bbox sides, and one separate maroon scaled binary pattern.
    - Each side marker lies on the top, right, bottom, or left edge of the diamond's bounding box but not on a diamond vertex.
    - The maroon pattern is an integer 2x or 3x enlargement of a smaller square binary mask.

    Rule:
    - Crop the maroon pattern and downsample it blockwise to recover the small binary mask.
    - Center a same-size square block grid inside the diamond; each mask cell controls one square subregion.
    - Markers fire perpendicular rays inward, and a ray may paint its current subregion only while the corresponding mask cell is non-black; the first black mask cell stops that ray immediately.
    - Top-side rays are processed first, then right, then bottom, then left; later rays overwrite earlier ray paint where they cross, while the frame, markers, and maroon pattern stay unchanged.

    Key insight:
    - The maroon object is not painted directly: after integer downsampling it becomes a hidden pass/stop mask that gates how far each side ray can travel through the diamond's interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    frame_box, center_r, center_c, _radius = _diamond_frame(g)
    mask, scale = _scaled_mask(g)
    n = int(mask.shape[0])
    square_side = int(n * scale)
    square_r0, square_c0 = _square_origin(center_r, center_c, square_side)

    marker_cells = np.argwhere(
        (g != BG) & (g != FRAME) & (g != PATTERN)
    )
    assert marker_cells.size > 0, "expected at least one side marker"

    grouped: dict[str, list[tuple[int, int, int]]] = {side: [] for side in SIDE_ORDER}
    for r_raw, c_raw in marker_cells:
        r = int(r_raw)
        c = int(c_raw)
        color = int(g[r, c])
        assert color in MARKER_COLORS, "unexpected marker color"
        side = _marker_side(r, c, frame_box)
        grouped[side].append((r, c, color))

    out = g.copy()
    for side in SIDE_ORDER:
        items = sorted(grouped[side])
        for r, c, color in items:
            if side == "top":
                band = _band_index(c, square_c0, scale, square_side)
                _paint_vertical(out, g, mask, color, c, band, scale, square_r0, square_c0, False)
            elif side == "right":
                band = _band_index(r, square_r0, scale, square_side)
                _paint_horizontal(out, g, mask, color, r, band, scale, square_r0, square_c0, True)
            elif side == "bottom":
                band = _band_index(c, square_c0, scale, square_side)
                _paint_vertical(out, g, mask, color, c, band, scale, square_r0, square_c0, True)
            else:
                band = _band_index(r, square_r0, scale, square_side)
                _paint_horizontal(out, g, mask, color, r, band, scale, square_r0, square_c0, False)
    return out


def _draw_diamond_frame(grid: np.ndarray, top: int, left: int, radius: int) -> None:
    center_r = top + radius
    center_c = left + radius
    for r in range(top, top + 2 * radius + 1):
        dc = radius - abs(r - center_r)
        grid[r, center_c - dc] = FRAME
        grid[r, center_c + dc] = FRAME


def _candidate_bands(n: int, scale: int, radius: int) -> dict[str, list[int]]:
    square_r0, square_c0 = _square_origin(radius, radius, n * scale)
    top_cols = []
    left_rows = []
    for idx in range(n):
        col = square_c0 + idx * scale + scale // 2
        row = square_r0 + idx * scale + scale // 2
        if col != radius:
            top_cols.append(idx)
        if row != radius:
            left_rows.append(idx)
    return {
        "top": top_cols[:],
        "bottom": top_cols[:],
        "left": left_rows[:],
        "right": left_rows[:],
    }


def _sample_marker_sides(n: int, scale: int, radius: int) -> dict[str, list[int]]:
    bands = _candidate_bands(n, scale, radius)
    counts = {side: 0 for side in SIDE_ORDER}
    total = random.randint(2, 4)
    counts[random.choice(("top", "bottom"))] += 1
    counts[random.choice(("left", "right"))] += 1
    while sum(counts.values()) < total:
        choices = [side for side in SIDE_ORDER if counts[side] < min(2, len(bands[side]))]
        side = random.choice(choices)
        counts[side] += 1

    placement: dict[str, list[int]] = {}
    for side in SIDE_ORDER:
        placement[side] = sorted(random.sample(bands[side], counts[side])) if counts[side] else []
    return placement


def _ray_sequence(side: str, band: int, n: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(i, band) for i in range(n)]
    if side == "bottom":
        return [(i, band) for i in range(n - 1, -1, -1)]
    if side == "left":
        return [(band, j) for j in range(n)]
    return [(band, j) for j in range(n - 1, -1, -1)]


def _ray_run(mask: np.ndarray, seq: list[tuple[int, int]]) -> int:
    run = 0
    for i, j in seq:
        if not mask[i, j]:
            break
        run += 1
    return run


def _has_reachable_cross(mask: np.ndarray, placement: dict[str, list[int]], n: int) -> bool:
    vertical = [("top", band) for band in placement["top"]] + [("bottom", band) for band in placement["bottom"]]
    horizontal = [("left", band) for band in placement["left"]] + [("right", band) for band in placement["right"]]
    for v_side, col in vertical:
        v_seq = _ray_sequence(v_side, col, n)
        v_index = {cell: idx for idx, cell in enumerate(v_seq)}
        for h_side, row in horizontal:
            cell = (row, col)
            if cell not in v_index:
                continue
            h_seq = _ray_sequence(h_side, row, n)
            h_index = {step: idx for idx, step in enumerate(h_seq)}
            if cell not in h_index:
                continue
            if all(mask[i, j] for i, j in v_seq[: v_index[cell] + 1]) and all(
                mask[i, j] for i, j in h_seq[: h_index[cell] + 1]
            ):
                return True
    return False


def _mask_ok(mask: np.ndarray, placement: dict[str, list[int]]) -> bool:
    n = int(mask.shape[0])
    if mask.all() or (~mask).all():
        return False
    if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
        return False

    runs = []
    full = False
    stopped = False
    for side in SIDE_ORDER:
        for band in placement[side]:
            seq = _ray_sequence(side, band, n)
            run = _ray_run(mask, seq)
            if run == 0:
                return False
            runs.append(run)
            if run == n:
                full = True
            if 0 < run < n:
                stopped = True
    if not runs or not full or not stopped:
        return False
    return _has_reachable_cross(mask, placement, n)


def _sample_mask(n: int, placement: dict[str, list[int]]) -> np.ndarray:
    for _ in range(MASK_ATTEMPTS):
        mask = np.random.rand(n, n) < np.random.uniform(0.5, 0.72)
        if _mask_ok(mask, placement):
            return mask
    raise RuntimeError("failed to build a gated mask with full-pass and blocked rays")


def _scaled_pattern(mask: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(mask.astype(int) * PATTERN, np.ones((scale, scale), dtype=int))


def _layout(diamond_side: int, pattern_side: int) -> tuple[int, int, int, int, int, int]:
    for _ in range(100):
        gap = random.randint(1, 3)
        orientation = random.choice(("horizontal", "vertical"))
        if orientation == "horizontal":
            min_w = diamond_side + pattern_side + gap
            if min_w > 30:
                continue
            width = random.randint(min_w, 30)
            height = random.randint(diamond_side, 30)
            slack = width - min_w
            left_margin = random.randint(0, slack)
            order_diamond_first = random.choice((True, False))
            if order_diamond_first:
                diamond_c = left_margin
                pattern_c = diamond_c + diamond_side + gap
            else:
                pattern_c = left_margin
                diamond_c = pattern_c + pattern_side + gap
            diamond_r = random.randint(0, height - diamond_side)
            pattern_r = random.randint(0, height - pattern_side)
            return height, width, diamond_r, diamond_c, pattern_r, pattern_c
        min_h = diamond_side + pattern_side + gap
        if min_h > 30:
            continue
        height = random.randint(min_h, 30)
        width = random.randint(diamond_side, 30)
        slack = height - min_h
        top_margin = random.randint(0, slack)
        order_diamond_first = random.choice((True, False))
        if order_diamond_first:
            diamond_r = top_margin
            pattern_r = diamond_r + diamond_side + gap
        else:
            pattern_r = top_margin
            diamond_r = pattern_r + pattern_side + gap
        diamond_c = random.randint(0, width - diamond_side)
        pattern_c = random.randint(0, width - pattern_side)
        return height, width, diamond_r, diamond_c, pattern_r, pattern_c
    raise RuntimeError("failed to sample a valid non-overlapping layout")


def _place_markers(
    grid: np.ndarray,
    placement: dict[str, list[int]],
    colors: list[int],
    diamond_r0: int,
    diamond_c0: int,
    radius: int,
    scale: int,
    n: int,
) -> None:
    square_r0, square_c0 = _square_origin(radius, radius, n * scale)
    color_iter = iter(colors)
    for side in SIDE_ORDER:
        for band in placement[side]:
            color = next(color_iter)
            if side in ("top", "bottom"):
                r = diamond_r0 if side == "top" else diamond_r0 + 2 * radius
                c = diamond_c0 + square_c0 + band * scale + scale // 2
            else:
                r = diamond_r0 + square_r0 + band * scale + scale // 2
                c = diamond_c0 if side == "left" else diamond_c0 + 2 * radius
            grid[r, c] = int(color)


def _generated_grid() -> np.ndarray:
    for _ in range(INNER_STRUCTURE_ATTEMPTS):
        n, scale = random.choices(CASE_OPTIONS, weights=(3, 2))[0]
        pattern_side = n * scale
        radius = pattern_side if pattern_side % 2 == 1 else pattern_side + 1
        diamond_side = 2 * radius + 1

        placement = _sample_marker_sides(n, scale, radius)
        try:
            mask = _sample_mask(n, placement)
        except RuntimeError:
            continue

        pattern = _scaled_pattern(mask, scale)
        height, width, diamond_r0, diamond_c0, pattern_r0, pattern_c0 = _layout(diamond_side, pattern_side)
        grid = np.zeros((height, width), dtype=int)

        _draw_diamond_frame(grid, diamond_r0, diamond_c0, radius)
        grid[pattern_r0 : pattern_r0 + pattern_side, pattern_c0 : pattern_c0 + pattern_side] = pattern

        total_markers = sum(len(v) for v in placement.values())
        colors = random.sample(list(MARKER_COLORS), total_markers)
        _place_markers(grid, placement, colors, diamond_r0, diamond_c0, radius, scale, n)
        return grid
    raise RuntimeError("failed to assemble a valid puzzle structure")


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _generated_grid()
        except RuntimeError:
            continue
        try:
            out = main(grid)
        except Exception:
            continue

        if np.array_equal(grid, out):
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        if int(np.count_nonzero(out != grid)) < 6:
            continue
        changed_colors = Counter(map(int, out[out != grid]))
        if not changed_colors:
            continue
        return grid
    raise RuntimeError("failed to generate a valid new ARC input")
