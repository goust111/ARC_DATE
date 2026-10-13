import numpy as np


GREY = 5
BG = 0
PALETTE = tuple(color for color in range(1, 10) if color != GREY)
MAX_GENERATION_ATTEMPTS = 400
PAIR_COUNT_RANGE = (1, 3)
GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
SHAPE_HEIGHT_RANGE = (2, 6)
SHAPE_WIDTH_RANGE = (2, 6)
PLACEMENT_ATTEMPTS = 250
SHAPE_ATTEMPTS = 80


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _shape_patch(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1] == color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored shapes, xor their aligned masks, and draw the grey difference
    pattern at the midpoint between the two source bounding boxes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    bg = _most_common_color(g)
    out = g.copy()

    colors = sorted(set(map(int, g.flat)) - {bg})
    assert colors, "expected at least one foreground color"

    for color in colors:
        comps = _components(g == color)
        assert len(comps) == 2, "each non-background color must form exactly two shapes"

        bbox_a = _bbox(comps[0])
        bbox_b = _bbox(comps[1])
        ha = bbox_a[2] - bbox_a[0] + 1
        wa = bbox_a[3] - bbox_a[1] + 1
        hb = bbox_b[2] - bbox_b[0] + 1
        wb = bbox_b[3] - bbox_b[1] + 1
        assert (ha, wa) == (hb, wb), "paired shapes must share bbox dimensions"

        patch_a = _shape_patch(g, color, bbox_a)
        patch_b = _shape_patch(g, color, bbox_b)
        diff = np.logical_xor(patch_a, patch_b)
        assert diff.any(), "paired shapes must differ"

        top_sum = bbox_a[0] + bbox_b[0]
        left_sum = bbox_a[1] + bbox_b[1]
        assert top_sum % 2 == 0 and left_sum % 2 == 0, "midpoint must align to the grid"
        top = top_sum // 2
        left = left_sum // 2
        assert 0 <= top and top + ha <= g.shape[0]
        assert 0 <= left and left + wa <= g.shape[1]

        region = out[top : top + ha, left : left + wa]
        region[diff] = GREY

    return out


def _manhattan_path(start: tuple[int, int], end: tuple[int, int], swap: bool) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    cells = [(r0, c0)]
    if swap:
        step = 1 if c1 >= c0 else -1
        for c in range(c0 + step, c1 + step, step):
            cells.append((r0, c))
        step = 1 if r1 >= r0 else -1
        for r in range(r0 + step, r1 + step, step):
            cells.append((r, c1))
    else:
        step = 1 if r1 >= r0 else -1
        for r in range(r0 + step, r1 + step, step):
            cells.append((r, c0))
        step = 1 if c1 >= c0 else -1
        for c in range(c0 + step, c1 + step, step):
            cells.append((r1, c))
    return cells


def _grow_shape(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    hub = (int(rng.integers(h)), int(rng.integers(w)))
    anchors = [
        (0, int(rng.integers(w))),
        (h - 1, int(rng.integers(w))),
        (int(rng.integers(h)), 0),
        (int(rng.integers(h)), w - 1),
    ]
    for anchor in anchors + [hub]:
        mask[anchor] = True
    for anchor in anchors:
        for r, c in _manhattan_path(anchor, hub, swap=bool(rng.integers(2))):
            mask[r, c] = True

    max_fill = min(h * w, max(int(np.ceil(0.75 * h * w)), int(mask.sum())))
    target = int(rng.integers(int(mask.sum()), max_fill + 1))
    while int(mask.sum()) < target:
        frontier = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if not mask[r, c] and any(mask[nr, nc] for nr, nc in _neighbors4(r, c, h, w))
        ]
        if not frontier:
            break
        r, c = frontier[int(rng.integers(len(frontier)))]
        mask[r, c] = True

    return mask


def _full_bbox(mask: np.ndarray) -> bool:
    comps = _components(mask)
    if len(comps) != 1:
        return False
    r0, c0, r1, c1 = _bbox(comps[0])
    return (r0, c0, r1, c1) == (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)


def _shape_pair(h: int, w: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(SHAPE_ATTEMPTS):
        first = _grow_shape(h, w, rng)
        second = _grow_shape(h, w, rng)
        if not (_full_bbox(first) and _full_bbox(second)):
            continue
        if np.array_equal(first, second):
            continue
        diff = np.logical_xor(first, second)
        if diff.sum() < 2:
            continue
        return first, second
    return None


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _placement_mode(h: int, w: int, rng: np.random.Generator) -> tuple[int, int]:
    mode = int(rng.integers(4))
    if mode == 0:
        return 0, int(rng.integers(w + 2, w + 8))
    if mode == 1:
        return int(rng.integers(h + 2, h + 8)), 0
    if mode == 2:
        return int(rng.integers(h + 2, h + 7)), int(rng.integers(w + 2, w + 7))
    return int(rng.integers(h + 2, h + 7)), -int(rng.integers(w + 2, w + 7))


def _all_clear(rects: list[tuple[int, int, int, int]], occupied: list[tuple[int, int, int, int]]) -> bool:
    for rect in rects:
        for other in occupied:
            if _rects_overlap(rect, other):
                return False
    return True


def _place_pair(
    grid_shape: tuple[int, int],
    occupied: list[tuple[int, int, int, int]],
    h: int,
    w: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    gh, gw = grid_shape
    for _ in range(PLACEMENT_ATTEMPTS):
        dr, dc = _placement_mode(h, w, rng)
        adr, adc = abs(dr), abs(dc)
        if gh - h - 2 * adr < 0 or gw - w - 2 * adc < 0:
            continue

        mid_r = int(rng.integers(adr, gh - h - adr + 1))
        mid_c = int(rng.integers(adc, gw - w - adc + 1))
        top_a = (mid_r - dr, mid_c - dc)
        top_b = (mid_r + dr, mid_c + dc)
        top_m = (mid_r, mid_c)

        rect_a = (top_a[0], top_a[1], top_a[0] + h - 1, top_a[1] + w - 1)
        rect_b = (top_b[0], top_b[1], top_b[0] + h - 1, top_b[1] + w - 1)
        rect_m = (top_m[0], top_m[1], top_m[0] + h - 1, top_m[1] + w - 1)
        rects = [rect_a, rect_b, rect_m]
        if not _all_clear(rects, occupied):
            continue
        return top_a, top_b, top_m
    return None


def _paint_mask(grid: np.ndarray, top: tuple[int, int], mask: np.ndarray, color: int) -> None:
    r0, c0 = top
    h, w = mask.shape
    grid[r0 : r0 + h, c0 : c0 + w][mask] = color


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    gh = int(rng.integers(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1] + 1))
    gw = int(rng.integers(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1] + 1))
    pair_count = int(rng.integers(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1] + 1))
    if pair_count > len(PALETTE):
        return None

    colors = list(rng.choice(np.array(PALETTE, dtype=int), size=pair_count, replace=False))
    grid = np.full((gh, gw), BG, dtype=int)
    expected = grid.copy()
    occupied: list[tuple[int, int, int, int]] = []

    for color in colors:
        h = int(rng.integers(SHAPE_HEIGHT_RANGE[0], SHAPE_HEIGHT_RANGE[1] + 1))
        w = int(rng.integers(SHAPE_WIDTH_RANGE[0], SHAPE_WIDTH_RANGE[1] + 1))
        pair = _shape_pair(h, w, rng)
        if pair is None:
            return None
        shape_a, shape_b = pair

        placement = _place_pair((gh, gw), occupied, h, w, rng)
        if placement is None:
            return None
        top_a, top_b, top_m = placement

        _paint_mask(grid, top_a, shape_a, int(color))
        _paint_mask(grid, top_b, shape_b, int(color))
        _paint_mask(expected, top_a, shape_a, int(color))
        _paint_mask(expected, top_b, shape_b, int(color))
        _paint_mask(expected, top_m, np.logical_xor(shape_a, shape_b), GREY)

        occupied.extend(
            [
                (top_a[0], top_a[1], top_a[0] + h - 1, top_a[1] + w - 1),
                (top_b[0], top_b[1], top_b[0] + h - 1, top_b[1] + w - 1),
                (top_m[0], top_m[1], top_m[0] + h - 1, top_m[1] + w - 1),
            ]
        )

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
