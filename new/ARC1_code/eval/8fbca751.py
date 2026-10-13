import numpy as np

from arc_utils import Color, bbox_coords, colors_present, find_components, get_bg


def _spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _bboxes_share_row_or_col(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return _spans_overlap(ar0, ar1, br0, br1) or _spans_overlap(ac0, ac1, bc0, bc1)


def _merged_fragment_groups(components: list[dict]) -> list[dict]:
    groups = [{"coords": comp["coords"], "bbox": comp["bbox"]} for comp in components]

    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            if changed:
                break
            for j in range(i + 1, len(groups)):
                if _bboxes_share_row_or_col(groups[i]["bbox"], groups[j]["bbox"]):
                    coords = np.vstack([groups[i]["coords"], groups[j]["coords"]])
                    groups[i] = {"coords": coords, "bbox": bbox_coords(coords)}
                    del groups[j]
                    changed = True
                    break
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each fragmented teal rectangle by filling its missing bbox cells red.

    Input structure
    - The grid has one background color and one foreground color.
    - Foreground cells form one or more incomplete rectangles; parts of the same rectangle may be split into several 4-connected fragments.
    - Distinct rectangles are separated diagonally, so their fragment bboxes do not share any row or column span.

    Rule
    - Find the 4-connected foreground fragments.
    - Merge fragments transitively when their bounding boxes overlap in row span or in column span.
    - For each merged group, recolor every background cell inside its inclusive bounding box to red.

    Key insight
    - The output rectangles are not ordinary connected components: diagonal corner contact can separate two rectangles, while disconnected fragments that align by row or column belong to the same rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert Color.RED not in set(map(int, g.flatten())), "red is the missing fill color, not an input color"

    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"

    fragments = find_components(g, bg=bg, colors=fg_colors, connectivity=4)
    assert fragments, "expected at least one foreground fragment"

    out = g.copy()
    for group in _merged_fragment_groups(fragments):
        r0, c0, r1, c1 = group["bbox"]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[region == bg] = Color.RED

    return out

# --------------- generator (o5x) ---------------

# Canvas and rectangle size bounds.
GRID_ROWS_RANGE = (7, 8)
GRID_COLS_RANGE = (7, 12)
TWO_GROUP_ROWS = 8
TWO_GROUP_COLS_RANGE = (8, 12)
RECT_HEIGHT = 4
RECT_WIDTHS = (4, 8)

# Structural counts and retry budget.
GROUP_COUNT_RANGE = (1, 2)
FRAGMENT_COUNT_RANGE = (1, 4)
GENERATION_ATTEMPTS = 300

# Fragment fill density.
FILL_PCT_RANGE = (44, 69)
PCT_DENOM = 100

# Seed layout modes and local growth geometry.
ONE_FRAGMENT_MODES = ("top", "bottom", "left", "right")
TWO_FRAGMENT_MODES = ("top_bottom", "bottom_top", "left_right", "right_left")
NEIGHBOR_OFFSETS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

def _gen_choice(seq):
    return seq[int(np.random.randint(len(seq)))]


def _gen_randint_inclusive(low, high):
    return int(np.random.randint(low, high + 1))


def _gen_neighbors4(r, c, h, w):
    for dr, dc in NEIGHBOR_OFFSETS_4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _gen_seed_labels(width, fragment_count):
    labels = np.zeros((RECT_HEIGHT, width), dtype=int)

    if fragment_count == 1:
        mode = _gen_choice(ONE_FRAGMENT_MODES)
        if mode == "top":
            c = int(np.random.randint(width))
            labels[0, :] = 1
            labels[:, c] = 1
        elif mode == "bottom":
            c = int(np.random.randint(width))
            labels[RECT_HEIGHT - 1, :] = 1
            labels[:, c] = 1
        elif mode == "left":
            r = int(np.random.randint(RECT_HEIGHT))
            labels[:, 0] = 1
            labels[r, :] = 1
        else:
            r = int(np.random.randint(RECT_HEIGHT))
            labels[:, width - 1] = 1
            labels[r, :] = 1
        return labels

    if fragment_count == 2:
        mode = _gen_choice(TWO_FRAGMENT_MODES)
        if mode == "top_bottom":
            labels[0, :] = 1
            labels[RECT_HEIGHT - 1, int(np.random.randint(width))] = 2
        elif mode == "bottom_top":
            labels[RECT_HEIGHT - 1, :] = 1
            labels[0, int(np.random.randint(width))] = 2
        elif mode == "left_right":
            labels[:, 0] = 1
            labels[int(np.random.randint(RECT_HEIGHT)), width - 1] = 2
        else:
            labels[:, width - 1] = 1
            labels[int(np.random.randint(RECT_HEIGHT)), 0] = 2
        return labels

    corners = [(0, 0), (0, width - 1), (RECT_HEIGHT - 1, 0), (RECT_HEIGHT - 1, width - 1)]
    np.random.shuffle(corners)
    corners = corners[:fragment_count]

    for label, (r, c) in enumerate(corners, start=1):
        labels[r, c] = label
    return labels


def _gen_growth_choices(labels, label):
    h, w = labels.shape
    choices = []
    for r in range(h):
        for c in range(w):
            if labels[r, c] != 0:
                continue
            touches_label = False
            touches_other = False
            for nr, nc in _gen_neighbors4(r, c, h, w):
                neighbor = int(labels[nr, nc])
                if neighbor == label:
                    touches_label = True
                elif neighbor != 0:
                    touches_other = True
            if touches_label and not touches_other:
                choices.append((r, c))
    return choices


def _gen_grow_labels(labels, target_area):
    labels = labels.copy()
    labels_present = list(range(1, int(labels.max()) + 1))

    while int(np.count_nonzero(labels)) < target_area:
        choices = []
        np.random.shuffle(labels_present)
        for label in labels_present:
            for r, c in _gen_growth_choices(labels, label):
                choices.append((label, r, c))
        if not choices:
            return None
        label, r, c = _gen_choice(choices)
        labels[r, c] = label

    return labels


def _gen_fragment_mask(width):
    area = RECT_HEIGHT * width
    min_area = (FILL_PCT_RANGE[0] * area + PCT_DENOM - 1) // PCT_DENOM
    max_area = (FILL_PCT_RANGE[1] * area) // PCT_DENOM
    fragment_count = _gen_randint_inclusive(*FRAGMENT_COUNT_RANGE)
    labels = _gen_seed_labels(width, fragment_count)
    target_area = _gen_randint_inclusive(max(min_area, int(np.count_nonzero(labels))), max_area)
    labels = _gen_grow_labels(labels, target_area)
    if labels is None:
        return None

    patch = np.where(labels > 0, Color.TEAL, Color.BLACK)
    components = find_components(patch, bg=Color.BLACK, colors=[Color.TEAL], connectivity=4)
    if len(components) != fragment_count:
        return None
    groups = _merged_fragment_groups(components)
    if len(groups) != 1 or tuple(groups[0]["bbox"]) != (0, 0, RECT_HEIGHT - 1, width - 1):
        return None
    return labels > 0


def _gen_bboxes():
    num_groups = _gen_randint_inclusive(*GROUP_COUNT_RANGE)

    if num_groups == 1:
        rows = _gen_randint_inclusive(*GRID_ROWS_RANGE)
        cols = _gen_randint_inclusive(*GRID_COLS_RANGE)
        available_widths = tuple(width for width in RECT_WIDTHS if width <= cols)
        rect_width = _gen_choice(available_widths)
        r0 = _gen_randint_inclusive(0, rows - RECT_HEIGHT)
        c0 = _gen_randint_inclusive(0, cols - rect_width)
        return rows, cols, [(r0, c0, r0 + RECT_HEIGHT - 1, c0 + rect_width - 1)]

    rows = TWO_GROUP_ROWS
    cols = _gen_randint_inclusive(*TWO_GROUP_COLS_RANGE)
    choices = []
    for w_top in RECT_WIDTHS:
        for w_bottom in RECT_WIDTHS:
            if w_top > cols or w_bottom > cols:
                continue
            for c_top in range(cols - w_top + 1):
                for c_bottom in range(cols - w_bottom + 1):
                    top = (c_top, c_top + w_top - 1)
                    bottom = (c_bottom, c_bottom + w_bottom - 1)
                    if top[1] < bottom[0] or bottom[1] < top[0]:
                        choices.append((w_top, c_top, w_bottom, c_bottom))

    if not choices:
        return None

    w_top, c_top, w_bottom, c_bottom = _gen_choice(choices)
    return rows, cols, [
        (0, c_top, RECT_HEIGHT - 1, c_top + w_top - 1),
        (RECT_HEIGHT, c_bottom, (2 * RECT_HEIGHT) - 1, c_bottom + w_bottom - 1),
    ]


def _gen_render_bboxes(rows, cols, bboxes):
    inp = np.zeros((rows, cols), dtype=int)
    out = np.zeros((rows, cols), dtype=int)

    for r0, c0, r1, c1 in bboxes:
        width = c1 - c0 + 1
        mask = _gen_fragment_mask(width)
        if mask is None:
            return None
        out[r0 : r1 + 1, c0 : c1 + 1] = Color.RED
        out[r0 : r1 + 1, c0 : c1 + 1][mask] = Color.TEAL
        inp[r0 : r1 + 1, c0 : c1 + 1][mask] = Color.TEAL

    return inp, out


def _gen_matches_solver(inp, out, bboxes):
    components = find_components(inp, bg=Color.BLACK, colors=[Color.TEAL], connectivity=4)
    merged = _merged_fragment_groups(components)
    expected_bboxes = sorted(tuple(bbox) for bbox in bboxes)
    actual_bboxes = sorted(tuple(group["bbox"]) for group in merged)
    return actual_bboxes == expected_bboxes and np.array_equal(main(inp), out)


def _gen_attempt():
    placed = _gen_bboxes()
    if placed is None:
        return None

    rows, cols, bboxes = placed
    rendered = _gen_render_bboxes(rows, cols, bboxes)
    if rendered is None:
        return None

    inp, out = rendered
    if not _gen_matches_solver(inp, out, bboxes):
        return None

    return inp


def generate_input():
    for _ in range(GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid 8fbca751 input")
