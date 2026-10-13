import numpy as np

from arc_utils import Color, find_components, get_bg


def _gap_direction(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r, c = gap
    r0, c0, r1, c1 = bbox
    assert int(r in (r0, r1)) + int(c in (c0, c1)) == 1
    if r == r0:
        return -1, 0
    if r == r1:
        return 1, 0
    if c == c0:
        return 0, -1
    assert c == c1
    return 0, 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each blue rectangular frame with teal and extend teal outward from any one-cell border gap.

    Input structure
    - The grid has a black background and one foreground color forming 4-connected rectangular frame objects.
    - Each frame's bounding-box border is either complete or has exactly one missing non-corner border pixel.
    - Frame interiors and the outward line from each border gap are black in the input.
    Rule
    - For every frame, recolor all pixels strictly inside its bbox to teal.
    - If the frame has a missing border pixel, recolor that gap and every black pixel from it straight outward, perpendicular to that side, through the grid edge.
    - Keep the blue frame pixels and all other black background pixels unchanged.
    Key insight
    - A missing border pixel is not a defect to repair with blue; it is a doorway that defines the direction of the teal spill from the filled rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    assert bg == Color.BLACK
    frame_colors = set(map(int, g.ravel())) - {bg}
    assert len(frame_colors) == 1
    frame_color = next(iter(frame_colors))
    assert frame_color == Color.BLUE

    frames = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[frame_color],
        return_format="dict",
    )
    assert frames

    out = g.copy()
    h_grid, w_grid = g.shape
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3

        coords = frame["coords"]
        assert all(r in (r0, r1) or c in (c0, c1) for r, c in coords)

        missing: list[tuple[int, int]] = []
        border_area = 2 * h + 2 * w - 4
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if r in (r0, r1) or c in (c0, c1):
                    if g[r, c] != frame_color:
                        missing.append((r, c))
                else:
                    assert g[r, c] == bg

        assert len(missing) in (0, 1)
        assert frame["area"] == border_area - len(missing)

        out[r0 + 1 : r1, c0 + 1 : c1] = Color.TEAL

        if missing:
            dr, dc = _gap_direction(missing[0], frame["bbox"])
            r, c = missing[0]
            while 0 <= r < h_grid and 0 <= c < w_grid:
                assert g[r, c] == bg
                out[r, c] = Color.TEAL
                r += dr
                c += dc

    return out

# --------------- generator (o5x) ---------------

# Canvas bounds and frame geometry.
GRID_ROWS_RANGE, GRID_COLS_RANGE = (24, 27), (25, 29)
FRAME_COUNT_RANGE = (5, 7)
FRAME_HEIGHT_RANGE, FRAME_WIDTH_RANGE = (3, 8), (5, 9)
EDGE_MARGIN, MIN_BBOX_GAP = 1, 2

# Layout grouping and orientation.
LAYOUT_GROUP_COUNT_OPTIONS = (2, 3)
FRAMES_PER_GROUP_RANGE = (1, 3)
ROW_LAYOUT_PROBABILITY = 0.5

# Gap and ray sampling.
GAP_PROBABILITY, GAP_PREFERRED_PROBABILITY = 0.70, 0.75
RAY_LENGTH_RANGE = (1, 10)
GAP_PREFERRED_MIN_CHOICES = 4
GAP_PREFERRED_FRACTION_RANGE = (0.25, 0.60)
SIDE_ORDER = ("top", "bottom", "left", "right")

# Rendering and retry policy.
BACKGROUND_COLOR, FRAME_COLOR, FILL_COLOR = Color.BLACK, Color.BLUE, Color.TEAL
MAX_GENERATE_ATTEMPTS = 300


def _gen_choose_sizes(n, lo, hi, max_sum):
    min_sum = n * lo
    max_sum = min(max_sum, n * hi)
    if max_sum < min_sum:
        return None
    total = np.random.randint(min_sum, max_sum + 1)
    sizes = [lo] * n
    extra = total - min_sum
    while extra:
        choices = [i for i, size in enumerate(sizes) if size < hi]
        i = int(np.random.choice(choices))
        add = np.random.randint(1, min(hi - sizes[i], extra) + 1)
        sizes[i] += int(add)
        extra -= int(add)
    np.random.shuffle(sizes)
    return sizes


def _gen_split_extra(extra, n):
    if extra <= 0:
        return [0] * n
    weights = np.random.dirichlet(np.ones(n))
    return list(map(int, np.random.multinomial(int(extra), weights)))


def _gen_group_counts(total):
    options = []
    min_per_group, max_per_group = FRAMES_PER_GROUP_RANGE
    for n_groups in LAYOUT_GROUP_COUNT_OPTIONS:
        def rec(rem, parts):
            if len(parts) == n_groups:
                if rem == 0:
                    options.append(tuple(parts))
                return
            for x in range(min_per_group, min(max_per_group, rem) + 1):
                rec(rem - x, parts + [x])

        rec(total, [])
    return list(options[int(np.random.randint(len(options)))])


def _gen_bboxes_separated(bboxes):
    for i, a in enumerate(bboxes):
        ar0, ac0, ar1, ac1 = a
        for b in bboxes[i + 1 :]:
            br0, bc0, br1, bc1 = b
            row_separated = ar1 + MIN_BBOX_GAP < br0 or br1 + MIN_BBOX_GAP < ar0
            col_separated = ac1 + MIN_BBOX_GAP < bc0 or bc1 + MIN_BBOX_GAP < ac0
            if not (row_separated or col_separated):
                return False
    return True


def _gen_partition_axis(span, n_blocks, lo, hi):
    """Choose block sizes and edge/inter-block slack along one axis."""
    capacity = span - 2 * EDGE_MARGIN - MIN_BBOX_GAP * (n_blocks - 1)
    sizes = _gen_choose_sizes(n_blocks, lo, hi, capacity)
    if sizes is None:
        return None
    extra = capacity - sum(sizes)
    gaps = _gen_split_extra(extra, n_blocks + 1)
    return sizes, gaps


def _gen_row_layout(rows, cols, count):
    groups = _gen_group_counts(count)
    band_plan = _gen_partition_axis(rows, len(groups), *FRAME_HEIGHT_RANGE)
    if band_plan is None:
        return None
    band_heights, row_gaps = band_plan
    r = EDGE_MARGIN + row_gaps[0]
    bboxes = []
    for band_i, (k, band_h) in enumerate(zip(groups, band_heights)):
        col_plan = _gen_partition_axis(cols, k, *FRAME_WIDTH_RANGE)
        if col_plan is None:
            return None
        widths, col_gaps = col_plan
        c = EDGE_MARGIN + col_gaps[0]
        for j, w in enumerate(widths):
            h = int(np.random.randint(FRAME_HEIGHT_RANGE[0], band_h + 1))
            rr = r + int(np.random.randint(0, band_h - h + 1))
            bboxes.append((rr, c, rr + h - 1, c + w - 1))
            c += w + MIN_BBOX_GAP + col_gaps[j + 1]
        r += band_h
        if band_i + 1 < len(groups):
            r += MIN_BBOX_GAP + row_gaps[band_i + 1]
    return bboxes


def _gen_col_layout(rows, cols, count):
    groups = _gen_group_counts(count)
    col_plan = _gen_partition_axis(cols, len(groups), *FRAME_WIDTH_RANGE)
    if col_plan is None:
        return None
    col_widths, col_gaps = col_plan
    c = EDGE_MARGIN + col_gaps[0]
    bboxes = []
    for col_i, (k, col_w) in enumerate(zip(groups, col_widths)):
        row_plan = _gen_partition_axis(rows, k, *FRAME_HEIGHT_RANGE)
        if row_plan is None:
            return None
        heights, row_gaps = row_plan
        r = EDGE_MARGIN + row_gaps[0]
        for j, h in enumerate(heights):
            w = int(np.random.randint(FRAME_WIDTH_RANGE[0], col_w + 1))
            cc = c + int(np.random.randint(0, col_w - w + 1))
            bboxes.append((r, cc, r + h - 1, cc + w - 1))
            r += h + MIN_BBOX_GAP + row_gaps[j + 1]
        c += col_w
        if col_i + 1 < len(groups):
            c += MIN_BBOX_GAP + col_gaps[col_i + 1]
    return bboxes


def _gen_ray_length(shape, bbox, side):
    rows, cols = shape
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0
    if side == "bottom":
        return rows - r1 - 1
    if side == "left":
        return c0
    return cols - c1 - 1


def _gen_ray_step(side):
    if side == "top":
        return -1, 0
    if side == "bottom":
        return 1, 0
    if side == "left":
        return 0, -1
    return 0, 1


def _gen_side_cells(bbox, side):
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _gen_in_any_other_bbox(r, c, bboxes, own_i):
    for j, (r0, c0, r1, c1) in enumerate(bboxes):
        if j != own_i and r0 <= r <= r1 and c0 <= c <= c1:
            return True
    return False


def _gen_ray_cells(shape, gap, side):
    rows, cols = shape
    dr, dc = _gen_ray_step(side)
    r, c = gap
    cells = []
    while 0 <= r < rows and 0 <= c < cols:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _gen_gap_options(shape, bboxes, own_i):
    options = {}
    bbox = bboxes[own_i]
    min_ray_length, max_ray_length = RAY_LENGTH_RANGE
    for side in SIDE_ORDER:
        if not (min_ray_length <= _gen_ray_length(shape, bbox, side) <= max_ray_length):
            continue
        good = []
        for gap in _gen_side_cells(bbox, side):
            ray = _gen_ray_cells(shape, gap, side)
            if all(not _gen_in_any_other_bbox(r, c, bboxes, own_i) for r, c in ray[1:]):
                good.append((gap, ray))
        if good:
            options[side] = good
    return options


def _gen_choose_gap(shape, bboxes, own_i):
    options = _gen_gap_options(shape, bboxes, own_i)
    if not options:
        return None
    side = list(options)[int(np.random.randint(len(options)))]
    choices = options[side]
    if len(choices) >= GAP_PREFERRED_MIN_CHOICES and np.random.random() < GAP_PREFERRED_PROBABILITY:
        preferred = []
        low_frac, high_frac = GAP_PREFERRED_FRACTION_RANGE
        for i, item in enumerate(choices):
            frac = (i + 1) / (len(choices) + 1)
            if low_frac <= frac <= high_frac:
                preferred.append(item)
        if preferred:
            choices = preferred
    gap, ray = choices[int(np.random.randint(len(choices)))]
    return side, gap, ray


def _gen_paint_output(shape, bboxes, gaps):
    out = np.full(shape, BACKGROUND_COLOR, dtype=int)
    for bbox, gap_info in zip(bboxes, gaps):
        r0, c0, r1, c1 = bbox
        out[r0, c0 : c1 + 1] = FRAME_COLOR
        out[r1, c0 : c1 + 1] = FRAME_COLOR
        out[r0 : r1 + 1, c0] = FRAME_COLOR
        out[r0 : r1 + 1, c1] = FRAME_COLOR
        out[r0 + 1 : r1, c0 + 1 : c1] = FILL_COLOR
        if gap_info is not None:
            _, gap, ray = gap_info
            out[gap] = FILL_COLOR
            for r, c in ray:
                out[r, c] = FILL_COLOR
    return out


def _gen_select_layout(rows, cols, count):
    if np.random.random() < ROW_LAYOUT_PROBABILITY:
        return _gen_row_layout(rows, cols, count)
    return _gen_col_layout(rows, cols, count)


def _gen_choose_gaps(shape, bboxes, count):
    """Sample requested gaps and return gap info tuples, or None if any ray is invalid."""
    gap_wants = np.random.random(count) < GAP_PROBABILITY
    if not gap_wants.any():
        gap_wants[int(np.random.randint(count))] = True

    gaps = []
    for i, want_gap in enumerate(gap_wants):
        if want_gap:
            gap_info = _gen_choose_gap(shape, bboxes, i)
            if gap_info is None:
                return None
            gaps.append(gap_info)
        else:
            gaps.append(None)
    return gaps


def _gen_attempt():
    rows = int(np.random.randint(*GRID_ROWS_RANGE))
    cols = int(np.random.randint(*GRID_COLS_RANGE))
    count = int(np.random.randint(*FRAME_COUNT_RANGE))
    bboxes = _gen_select_layout(rows, cols, count)
    if bboxes is None or len(bboxes) != count or not _gen_bboxes_separated(bboxes):
        return None

    gaps = _gen_choose_gaps((rows, cols), bboxes, count)
    if gaps is None:
        return None

    out = _gen_paint_output((rows, cols), bboxes, gaps)
    inp = out.copy()
    inp[inp == FILL_COLOR] = BACKGROUND_COLOR
    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input():
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 551d5bf1 input")
