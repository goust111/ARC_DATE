import numpy as np


def _round_half_up(value):
    return int(np.floor(value + 0.5))


def _extract_object(grid, q_row, q_col):
    rows, cols = grid.shape
    half_r, half_c = rows // 2, cols // 2
    r0, r1 = q_row * half_r, (q_row + 1) * half_r
    c0, c1 = q_col * half_c, (q_col + 1) * half_c
    sub = grid[r0:r1, c0:c1]
    points = np.argwhere(sub != 0)
    if len(points) == 0:
        return None

    local_min_r, local_min_c = points.min(axis=0)
    local_max_r, local_max_c = points.max(axis=0)
    color = int(sub[tuple(points[0])])
    bbox = sub[local_min_r : local_max_r + 1, local_min_c : local_max_c + 1]
    return {
        "color": color,
        "mask": bbox == color,
        "min_r": r0 + int(local_min_r),
        "max_r": r0 + int(local_max_r),
        "min_c": c0 + int(local_min_c),
        "max_c": c0 + int(local_max_c),
    }


def _draw_at_partner_center(out, source, target):
    mask = source["mask"]
    height, width = mask.shape
    top = _round_half_up((target["min_r"] + target["max_r"] - (height - 1)) / 2.0)
    left = _round_half_up((target["min_c"] + target["max_c"] - (width - 1)) / 2.0)
    for rr, cc in np.argwhere(mask):
        r, c = top + int(rr), left + int(cc)
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            out[r, c] = source["color"]


def main(grid: np.ndarray) -> np.ndarray:
    """Move diagonal-quadrant objects to each other's centers with their own colors.

    Input structure:
    The grid has a black background and even height and width, so its invisible
    center row and column lines split it into four equal quadrants. Each quadrant
    contains exactly one nonzero 4-connected monochrome object, the four object
    colors are distinct, and no object crosses a quadrant boundary.

    Rule:
    Pair the top-left object with the bottom-right object, and pair the
    top-right object with the bottom-left object. For each pair, extract both
    object masks, colors, and bounding-box centers. Redraw each object's own
    mask at the partner object's original bounding-box center, using that
    object's original color. All remaining cells are black.

    Key insight:
    The invisible center cross is the only cue for the two diagonal pairings;
    after that, each color carries its own shape to the paired object's position.
    """
    out = np.zeros_like(grid)
    objects = {
        (0, 0): _extract_object(grid, 0, 0),
        (0, 1): _extract_object(grid, 0, 1),
        (1, 0): _extract_object(grid, 1, 0),
        (1, 1): _extract_object(grid, 1, 1),
    }

    pairs = [((0, 0), (1, 1)), ((0, 1), (1, 0))]
    for a_key, b_key in pairs:
        a_obj, b_obj = objects[a_key], objects[b_key]
        if a_obj is None or b_obj is None:
            continue
        _draw_at_partner_center(out, a_obj, b_obj)
        _draw_at_partner_center(out, b_obj, a_obj)
    return out


def _random_connected_mask(rng, max_h, max_w, max_bbox_area):
    for _ in range(200):
        height = int(rng.integers(1, max_h + 1))
        width = int(rng.integers(1, max_w + 1))
        if height * width > max_bbox_area:
            continue

        area = height * width
        min_cells = 1 if area == 1 else max(2, int(np.ceil(area * 0.35)))
        cell_count = int(rng.integers(min_cells, area + 1))
        mask = np.zeros((height, width), dtype=bool)
        start = (int(rng.integers(height)), int(rng.integers(width)))
        mask[start] = True
        frontier = set()

        def add_neighbors(cell):
            r, c = cell
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))

        add_neighbors(start)
        while int(mask.sum()) < cell_count and frontier:
            choices = tuple(frontier)
            cell = choices[int(rng.integers(len(choices)))]
            frontier.remove(cell)
            if mask[cell]:
                continue
            mask[cell] = True
            add_neighbors(cell)

        points = np.argwhere(mask)
        min_r, min_c = points.min(axis=0)
        max_r, max_c = points.max(axis=0)
        cropped = mask[min_r : max_r + 1, min_c : max_c + 1]
        if cropped.shape[0] * cropped.shape[1] <= max_bbox_area:
            return cropped

    return np.ones((1, 1), dtype=bool)


def _valid_starts(limit, own_len, incoming_len):
    starts = []
    for start in range(limit - own_len + 1):
        incoming_start = _round_half_up(
            (2 * start + own_len - 1 - (incoming_len - 1)) / 2.0
        )
        if 0 <= incoming_start and incoming_start + incoming_len <= limit:
            starts.append(start)
    return starts


def _paint_mask(grid, q_row, q_col, top, left, mask, color):
    half_r, half_c = grid.shape[0] // 2, grid.shape[1] // 2
    base_r, base_c = q_row * half_r, q_col * half_c
    for rr, cc in np.argwhere(mask):
        grid[base_r + top + int(rr), base_c + left + int(cc)] = color


def _color_stays_in_quadrant(grid, color, q_index):
    half_r, half_c = grid.shape[0] // 2, grid.shape[1] // 2
    q_row, q_col = divmod(q_index, 2)
    points = np.argwhere(grid == color)
    if len(points) == 0:
        return False
    r0, r1 = q_row * half_r, (q_row + 1) * half_r
    c0, c1 = q_col * half_c, (q_col + 1) * half_c
    return bool(
        np.all((r0 <= points[:, 0]) & (points[:, 0] < r1))
        and np.all((c0 <= points[:, 1]) & (points[:, 1] < c1))
    )


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    quadrant_pairs = {0: 3, 1: 2, 2: 1, 3: 0}

    for _ in range(1000):
        rows = int(rng.choice(np.arange(12, 31, 2)))
        cols = int(rng.choice(np.arange(12, 31, 2)))
        half_r, half_c = rows // 2, cols // 2
        max_bbox_area = (half_r * half_c) // 2
        max_h = max(1, min(6, half_r - 1))
        max_w = max(1, min(6, half_c - 1))

        masks = [
            _random_connected_mask(rng, max_h, max_w, max_bbox_area)
            for _ in range(4)
        ]
        starts = []
        feasible = True
        for q_index, mask in enumerate(masks):
            incoming = masks[quadrant_pairs[q_index]]
            row_starts = _valid_starts(half_r, mask.shape[0], incoming.shape[0])
            col_starts = _valid_starts(half_c, mask.shape[1], incoming.shape[1])
            if not row_starts or not col_starts:
                feasible = False
                break
            starts.append(
                (
                    int(row_starts[int(rng.integers(len(row_starts)))]),
                    int(col_starts[int(rng.integers(len(col_starts)))]),
                )
            )
        if not feasible:
            continue

        colors = rng.choice(np.arange(1, 10), size=4, replace=False)
        grid = np.zeros((rows, cols), dtype=np.int8)
        for q_index, mask in enumerate(masks):
            q_row, q_col = divmod(q_index, 2)
            top, left = starts[q_index]
            _paint_mask(grid, q_row, q_col, top, left, mask, int(colors[q_index]))

        out = main(grid)
        expected_counts = {
            int(colors[q_index]): int(masks[q_index].sum()) for q_index in range(4)
        }
        counts_ok = all(
            int(np.count_nonzero(out == color)) == count
            for color, count in expected_counts.items()
        )
        locations_ok = all(
            _color_stays_in_quadrant(out, int(colors[q_index]), quadrant_pairs[q_index])
            for q_index in range(4)
        )
        if counts_ok and locations_ok:
            return grid

    raise RuntimeError("failed to generate a valid ARC input")
