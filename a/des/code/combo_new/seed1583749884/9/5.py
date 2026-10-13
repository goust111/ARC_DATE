import numpy as np


BLACK = 0
YELLOW = 4
COLOR_POOL = [1, 2, 3, 5, 6, 7, 8, 9]

GRID_MIN_SIZE = 18
GRID_MAX_SIZE_EXCLUSIVE = 31
PAIR_COUNT_CHOICES = (1, 2, 3)
SINGLETON_COUNT_CHOICES = (0, 1, 2)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250
MASK_ATTEMPTS = 120

_RNG = np.random.default_rng()


def _find_components(grid: np.ndarray, bg: int = BLACK) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "anchor": np.array((int(r0), int(c0)), dtype=int),
                }
            )

    return components


def _template_targets(comp_a: dict, comp_b: dict) -> np.ndarray:
    def sort_key(comp: dict) -> tuple[int, int, int]:
        r0, c0, _r1, _c1 = comp["bbox"]
        return (int(comp["area"]), r0, c0)

    template, other = sorted((comp_a, comp_b), key=sort_key)

    a_r0, a_c0, a_r1, a_c1 = template["bbox"]
    b_r0, b_c0, b_r1, b_c1 = other["bbox"]
    center2_a = np.array((a_r0 + a_r1, a_c0 + a_c1), dtype=int)
    center2_b = np.array((b_r0 + b_r1, b_c0 + b_c1), dtype=int)

    same_row_center = int(center2_a[0]) == int(center2_b[0])
    same_col_center = int(center2_a[1]) == int(center2_b[1])
    assert same_row_center ^ same_col_center, "same-color objects must align horizontally or vertically"

    delta_center2 = center2_b - center2_a
    if np.all(delta_center2 % 4 == 0):
        shift = delta_center2 // 4
    else:
        delta_anchor = other["anchor"] - template["anchor"]
        assert np.all(delta_anchor % 2 == 0), "midpoint placement must land on integer cells"
        shift = delta_anchor // 2

    if same_row_center:
        assert int(shift[0]) == 0 and int(shift[1]) != 0
    else:
        assert int(shift[1]) == 0 and int(shift[0]) != 0

    return template["coords"] + shift


def main(grid: np.ndarray) -> np.ndarray:
    """Insert a yellow midpoint copy between each aligned pair of same-color objects.

    Input structure:
    - The background is black.
    - The grid contains several 4-connected monochrome foreground objects.
    - Some colors appear on exactly two objects that are aligned horizontally or
      vertically with an even background gap; other colors appear once.

    Rule:
    - Leave singleton-color objects unchanged.
    - For each color that appears twice, pair its two objects.
    - Select the smaller-area object as the template shape.
    - Translate that template halfway toward its same-color partner so the copy
      lands at the midpoint between the pair, and recolor the copy yellow.
    - Keep both original objects unchanged.

    Key insight:
    - Solve the task by grouping components by color, using alignment to place a
      midpoint translation, and taking the copied silhouette from the smaller of
      the two paired objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    components = _find_components(g, bg=BLACK)
    by_color: dict[int, list[dict]] = {}
    for comp in components:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    for color, comps in sorted(by_color.items()):
        assert color != YELLOW, "yellow is reserved for the midpoint copies"
        assert len(comps) in (1, 2), "each foreground color must appear once or twice"
        if len(comps) == 1:
            continue

        targets = _template_targets(comps[0], comps[1])
        assert np.all(targets[:, 0] >= 0) and np.all(targets[:, 0] < g.shape[0])
        assert np.all(targets[:, 1] >= 0) and np.all(targets[:, 1] < g.shape[1])
        assert np.all(out[targets[:, 0], targets[:, 1]] == BLACK), "midpoint copy must use empty cells"
        out[targets[:, 0], targets[:, 1]] = YELLOW

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if points.size == 0:
        return False

    start = tuple(int(v) for v in points[0])
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]):
                continue
            if not mask[nr, nc] or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))

    return len(seen) == int(mask.sum())


def _spans_full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _random_full_bbox_mask(height: int, width: int, target_area: int) -> np.ndarray | None:
    if target_area > height * width:
        return None
    if target_area == height * width:
        return np.ones((height, width), dtype=bool)

    for _ in range(MASK_ATTEMPTS):
        mask = np.ones((height, width), dtype=bool)
        area = height * width

        while area > target_area:
            removable = []
            cells = [(r, c) for r in range(height) for c in range(width) if mask[r, c]]
            _RNG.shuffle(cells)
            for r, c in cells:
                mask[r, c] = False
                if _spans_full_bbox(mask) and _mask_connected(mask):
                    removable.append((r, c))
                mask[r, c] = True
            if not removable:
                break
            rr, cc = removable[int(_RNG.integers(len(removable)))]
            mask[rr, cc] = False
            area -= 1

        if area == target_area and _spans_full_bbox(mask) and _mask_connected(mask):
            return mask

    return None


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    points = np.argwhere(mask)
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_connected_mask(height: int, width: int, target_area: int) -> np.ndarray | None:
    if not (1 <= target_area <= height * width):
        return None

    for _ in range(MASK_ATTEMPTS):
        start = (int(_RNG.integers(height)), int(_RNG.integers(width)))
        chosen = {start}
        frontier = set()

        def add_frontier(cell: tuple[int, int]) -> None:
            r, c = cell
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in chosen:
                    frontier.add((nr, nc))

        add_frontier(start)
        while len(chosen) < target_area and frontier:
            frontier_list = tuple(frontier)
            picked = frontier_list[int(_RNG.integers(len(frontier_list)))]
            frontier.remove(picked)
            chosen.add(picked)
            add_frontier(picked)

        if len(chosen) != target_area:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        return _trim_mask(mask)

    return None


def _sample_pair_masks(height: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    min_area = height + width - 1
    choices = list(range(min_area, height * width + 1))
    if len(choices) < 2:
        return None

    for _ in range(MASK_ATTEMPTS):
        area_a, area_b = sorted(_RNG.choice(choices, size=2, replace=False).tolist())
        mask_a = _random_full_bbox_mask(height, width, int(area_a))
        mask_b = _random_full_bbox_mask(height, width, int(area_b))
        if mask_a is None or mask_b is None:
            continue
        if int(mask_a.sum()) < int(mask_b.sum()):
            return mask_a, mask_b

    return None


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    rows, cols = np.argwhere(mask).T
    grid[top + rows, left + cols] = color


def _rect_fits(rect: tuple[int, int, int, int], reserved: list[tuple[int, int, int, int]], margin: int = 1) -> bool:
    r0, c0, r1, c1 = rect
    for or0, oc0, or1, oc1 in reserved:
        if not (r1 + margin < or0 or or1 + margin < r0 or c1 + margin < oc0 or oc1 + margin < c0):
            return False
    return True


def _place_rect(
    grid_h: int,
    grid_w: int,
    rect_h: int,
    rect_w: int,
    reserved: list[tuple[int, int, int, int]],
    margin: int = 1,
) -> tuple[int, int] | None:
    if rect_h > grid_h or rect_w > grid_w:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        top = int(_RNG.integers(0, grid_h - rect_h + 1))
        left = int(_RNG.integers(0, grid_w - rect_w + 1))
        rect = (top, left, top + rect_h - 1, left + rect_w - 1)
        if _rect_fits(rect, reserved, margin=margin):
            reserved.append(rect)
            return top, left

    return None


def _valid_generated_grid(grid: np.ndarray, expected_pairs: int) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if np.any(grid == YELLOW):
        return False

    components = _find_components(grid, bg=BLACK)
    if not components:
        return False

    by_color: dict[int, list[dict]] = {}
    for comp in components:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    if sum(len(comps) == 2 for comps in by_color.values()) != expected_pairs:
        return False
    if not all(len(comps) in (1, 2) for comps in by_color.values()):
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False

    return out.shape == grid.shape and np.any(out == YELLOW)


def _gen_attempt() -> np.ndarray | None:
    grid_h = int(_RNG.integers(GRID_MIN_SIZE, GRID_MAX_SIZE_EXCLUSIVE))
    grid_w = int(_RNG.integers(GRID_MIN_SIZE, GRID_MAX_SIZE_EXCLUSIVE))
    pair_count = int(_RNG.choice(PAIR_COUNT_CHOICES))

    if bool(_RNG.integers(0, 2)):
        singleton_count = int(_RNG.choice((1, 2)))
    else:
        singleton_count = 0

    if pair_count + singleton_count > len(COLOR_POOL):
        return None

    colors = _RNG.choice(COLOR_POOL, size=pair_count + singleton_count, replace=False).tolist()
    pair_colors = colors[:pair_count]
    singleton_colors = colors[pair_count:]

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    reserved: list[tuple[int, int, int, int]] = []

    for color in pair_colors:
        horizontal = bool(_RNG.integers(0, 2))
        if horizontal:
            box_h = int(_RNG.integers(2, 5))
            box_w = int(_RNG.choice((2, 4)))
            step = int(_RNG.integers(box_w + 1, box_w + 5))
            group_h, group_w = box_h, box_w + 2 * step
        else:
            box_h = int(_RNG.choice((2, 4)))
            box_w = int(_RNG.integers(2, 5))
            step = int(_RNG.integers(box_h + 1, box_h + 5))
            group_h, group_w = box_h + 2 * step, box_w

        masks = _sample_pair_masks(box_h, box_w)
        if masks is None:
            return None
        small_mask, large_mask = masks

        group_pos = _place_rect(grid_h, grid_w, group_h, group_w, reserved, margin=1)
        if group_pos is None:
            return None
        top, left = group_pos

        if horizontal:
            anchors = [(top, left), (top, left + 2 * step)]
        else:
            anchors = [(top, left), (top + 2 * step, left)]

        if bool(_RNG.integers(0, 2)):
            endpoint_masks = (small_mask, large_mask)
        else:
            endpoint_masks = (large_mask, small_mask)

        for (anchor_r, anchor_c), mask in zip(anchors, endpoint_masks):
            _paint_mask(grid, anchor_r, anchor_c, mask, int(color))

    for color in singleton_colors:
        box_h = int(_RNG.integers(1, 5))
        box_w = int(_RNG.integers(1, 5))
        target_area = int(_RNG.integers(1, box_h * box_w + 1))
        mask = _random_connected_mask(box_h, box_w, target_area)
        if mask is None:
            return None

        place = _place_rect(grid_h, grid_w, mask.shape[0], mask.shape[1], reserved, margin=1)
        if place is None:
            return None
        top, left = place
        _paint_mask(grid, top, left, mask, int(color))

    if not _valid_generated_grid(grid, expected_pairs=pair_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
