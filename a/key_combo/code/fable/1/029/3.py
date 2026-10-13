from __future__ import annotations

import itertools
import random

import numpy as np


BBox = tuple[int, int, int, int]


# Solver / generator config.
BG = 0
GRID_HEIGHT_RANGE = (12, 20)
MAIN_WIDTH_RANGE = (11, 16)
SCATTER_WIDTH_RANGE = (5, 7)
RECT_COUNT_CHOICES = (3, 4)
RECT_HEIGHT_RANGE = (4, 7)
RECT_WIDTH_RANGE = (4, 7)
COUNT_POOL = (1, 2, 3, 4, 5, 6)
MAX_PAIR_OVERLAP_FRACTION = 0.45
LAYOUT_ATTEMPTS = 300
GENERATION_ATTEMPTS = 500
MIRROR_PROB = 0.5


def get_bg(grid: np.ndarray) -> int:
    """Canonical ARC background = most common color."""
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def bbox_coords(coords: np.ndarray) -> BBox:
    """BBox of an (N,2) coords array, in canonical (r,c) convention."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_area(box: BBox) -> int:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _intersection_area(a: BBox, b: BBox) -> int:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return 0
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _same_color_adjacencies(region: np.ndarray, bg: int) -> int:
    """Dense rectangles create many same-color adjacencies; isolated scatter pixels do not."""
    if region.size == 0:
        return 0
    horiz = (region[:, 1:] == region[:, :-1]) & (region[:, :-1] != int(bg))
    vert = (region[1:, :] == region[:-1, :]) & (region[:-1, :] != int(bg))
    return int(np.count_nonzero(horiz) + np.count_nonzero(vert))


def _max_component_area(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            area = 0
            while stack:
                rr, cc = stack.pop()
                area += 1
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            best = max(best, area)
    return int(best)


def _find_separator_col(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates = []
    for c in range(1, w - 1):
        color = int(grid[0, c])
        if color == int(bg):
            continue
        if not np.all(grid[:, c] == color):
            continue
        if not np.any(grid[:, :c] != int(bg)):
            continue
        if not np.any(grid[:, c + 1 :] != int(bg)):
            continue
        candidates.append(c)
    assert len(candidates) == 1, "expected exactly one full-height separator column"
    return int(candidates[0])


def _split_regions(grid: np.ndarray, bg: int) -> tuple[int, slice, slice]:
    sep = _find_separator_col(grid, bg)
    left = grid[:, :sep]
    right = grid[:, sep + 1 :]
    left_score = (
        _same_color_adjacencies(left, bg),
        _max_component_area(left != int(bg)),
        int(np.count_nonzero(left != int(bg))),
    )
    right_score = (
        _same_color_adjacencies(right, bg),
        _max_component_area(right != int(bg)),
        int(np.count_nonzero(right != int(bg))),
    )
    assert left_score != right_score, "main region and scatter region must be distinguishable"
    if left_score > right_score:
        return sep, slice(0, sep), slice(sep + 1, grid.shape[1])
    return sep, slice(sep + 1, grid.shape[1]), slice(0, sep)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each overlapped solid rectangle from visible corners and redraw the stack using scatter-count order.

    Input structure:
    - One side of the grid is a main region containing several differently colored solid rectangles that overlap.
    - A single full-height separator column splits the main region from a scatter region of isolated color pixels.
    - Each rectangle color also appears in the scatter region, and each rectangle keeps at least one visible pair of opposite corners.

    Rule:
    - Detect the separator and identify which side is the dense rectangle region versus the sparse scatter region.
    - For each color in the main region, recover the full rectangle bbox from the min/max rows and columns of its still-visible cells.
    - Count that color's pixels in the scatter region, then redraw all recovered rectangles on a clean main region from largest count first (bottom) to smallest count last (top), leaving the separator column and scatter region unchanged.

    Key insight:
    - The input overlap order is a distraction: opposite visible corners are enough to recover each full rectangle, while the true stacking order is encoded only by color counts in the separated scatter strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    _sep, main_cols, scatter_cols = _split_regions(g, bg)
    main_region = g[:, main_cols]
    scatter_region = g[:, scatter_cols]

    colors = [int(color) for color in np.unique(main_region) if int(color) != int(bg)]
    assert colors, "expected at least one rectangle color in the main region"

    boxes: dict[int, BBox] = {}
    for color in colors:
        coords = np.argwhere(main_region == int(color))
        assert coords.size > 0, "every rectangle color must remain visible"
        boxes[int(color)] = bbox_coords(coords)

    counts = {int(color): int(np.count_nonzero(scatter_region == int(color))) for color in colors}
    assert all(count > 0 for count in counts.values()), "every rectangle color must appear in scatter region"
    assert len(set(counts.values())) == len(counts), "scatter counts must be unique"

    order = [
        color
        for color, _count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    ]

    out = g.copy()
    out[:, main_cols] = int(bg)
    out_main = out[:, main_cols]
    for color in order:
        r0, c0, r1, c1 = boxes[int(color)]
        out_main[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _render_main(shape: tuple[int, int], boxes: dict[int, BBox], order: list[int]) -> np.ndarray:
    canvas = np.full(shape, BG, dtype=int)
    for color in order:
        r0, c0, r1, c1 = boxes[int(color)]
        canvas[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return canvas


def _visible_opposite_corners(canvas: np.ndarray, color: int, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    diag_a = int(canvas[r0, c0]) == int(color) and int(canvas[r1, c1]) == int(color)
    diag_b = int(canvas[r0, c1]) == int(color) and int(canvas[r1, c0]) == int(color)
    return bool(diag_a or diag_b)


def _sample_boxes(height: int, width: int, colors: list[int]) -> dict[int, BBox] | None:
    max_h = min(RECT_HEIGHT_RANGE[1], height - 2)
    max_w = min(RECT_WIDTH_RANGE[1], width - 2)
    if max_h < RECT_HEIGHT_RANGE[0] or max_w < RECT_WIDTH_RANGE[0]:
        return None

    for _ in range(LAYOUT_ATTEMPTS):
        boxes: dict[int, BBox] = {}
        raw_boxes: list[BBox] = []
        for color in colors:
            h = random.randint(RECT_HEIGHT_RANGE[0], max_h)
            w = random.randint(RECT_WIDTH_RANGE[0], max_w)
            r0 = random.randint(1, height - h - 1)
            c0 = random.randint(1, width - w - 1)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            boxes[int(color)] = box
            raw_boxes.append(box)

        if len(set(raw_boxes)) != len(raw_boxes):
            continue

        areas = [_rect_area(box) for box in raw_boxes]
        if len(set(areas)) != len(areas):
            continue

        overlap_areas = []
        bad = False
        for a, b in itertools.combinations(raw_boxes, 2):
            overlap = _intersection_area(a, b)
            if overlap == 0:
                continue
            if overlap > int(min(_rect_area(a), _rect_area(b)) * MAX_PAIR_OVERLAP_FRACTION):
                bad = True
                break
            overlap_areas.append(overlap)
        if bad or not overlap_areas:
            continue
        return boxes
    return None


def _target_orders(colors: list[int], input_order: list[int], area_order: list[int], visible_order: list[int]) -> list[list[int]]:
    banned = {
        tuple(input_order),
        tuple(area_order),
        tuple(visible_order),
        tuple(sorted(colors)),
        tuple(sorted(colors, reverse=True)),
    }
    candidates = []
    for perm in itertools.permutations(colors):
        if tuple(perm) in banned:
            continue
        candidates.append(list(perm))
    return candidates


def _scatter_capacity(height: int, width: int) -> int:
    return ((int(height) + 1) // 2) * ((int(width) + 1) // 2)


def _place_scatter(height: int, width: int, count_by_color: dict[int, int]) -> np.ndarray | None:
    if sum(count_by_color.values()) > _scatter_capacity(height, width):
        return None

    scatter = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    color_order = sorted(count_by_color, key=lambda color: count_by_color[color], reverse=True)
    for color in color_order:
        for _ in range(int(count_by_color[color])):
            candidates = [(r, c) for r in range(height) for c in range(width) if not blocked[r, c]]
            if not candidates:
                return None
            r, c = random.choice(candidates)
            scatter[r, c] = int(color)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < height and 0 <= cc < width:
                        blocked[rr, cc] = True
    return scatter


def _validate_semantics(
    grid: np.ndarray,
    expected: np.ndarray,
    boxes: dict[int, BBox],
    input_order: list[int],
    target_order: list[int],
    count_by_color: dict[int, int],
) -> bool:
    if np.array_equal(grid, expected):
        return False
    if int(np.count_nonzero(grid != BG)) >= grid.size - int(np.count_nonzero(grid != BG)):
        return False

    bg = get_bg(grid)
    if int(bg) != int(BG):
        return False

    try:
        _sep, main_cols, scatter_cols = _split_regions(grid, bg)
    except AssertionError:
        return False

    main_region = grid[:, main_cols]
    scatter_region = grid[:, scatter_cols]

    visible_counts = {}
    for color, box in boxes.items():
        coords = np.argwhere(main_region == int(color))
        if coords.size == 0:
            return False
        if bbox_coords(coords) != box:
            return False
        if not _visible_opposite_corners(main_region, int(color), box):
            return False
        visible_counts[int(color)] = int(coords.shape[0])

    if len(set(visible_counts.values())) != len(visible_counts):
        return False

    if all(visible_counts[color] == _rect_area(boxes[color]) for color in boxes):
        return False

    if [color for color in input_order] == [color for color in target_order]:
        return False

    if any(int(np.count_nonzero(scatter_region == color)) != int(count_by_color[color]) for color in boxes):
        return False
    if any(int(np.count_nonzero(scatter_region == color)) == 0 for color in boxes):
        return False

    if any(_same_color_adjacencies((scatter_region == color).astype(int), 0) > 0 for color in boxes):
        return False

    actual = main(grid)
    return bool(np.array_equal(actual, expected))


def _attempt() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    main_width = random.randint(*MAIN_WIDTH_RANGE)
    scatter_width = random.randint(*SCATTER_WIDTH_RANGE)

    color_count = random.choice(RECT_COUNT_CHOICES)
    palette = list(range(1, 10))
    random.shuffle(palette)
    sep_color = int(palette.pop())
    colors = sorted(int(color) for color in palette[:color_count])

    boxes = _sample_boxes(height, main_width, colors)
    if boxes is None:
        return None

    input_order = random.sample(colors, len(colors))
    input_main = _render_main((height, main_width), boxes, input_order)

    if any(not _visible_opposite_corners(input_main, color, boxes[color]) for color in colors):
        return None

    area_order = [
        color
        for color, _area in sorted(
            ((color, _rect_area(boxes[color])) for color in colors),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    visible_order = [
        color
        for color, _visible in sorted(
            ((color, int(np.count_nonzero(input_main == color))) for color in colors),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    target_orders = _target_orders(colors, input_order, area_order, visible_order)
    if not target_orders:
        return None
    target_order = random.choice(target_orders)

    counts_desc = sorted(random.sample(COUNT_POOL, len(colors)), reverse=True)
    count_by_color = {int(target_order[i]): int(counts_desc[i]) for i in range(len(colors))}

    scatter = _place_scatter(height, scatter_width, count_by_color)
    if scatter is None:
        return None

    expected_main = _render_main((height, main_width), boxes, target_order)
    if np.array_equal(input_main, expected_main):
        return None

    grid = np.full((height, main_width + 1 + scatter_width), BG, dtype=int)
    expected = np.full_like(grid, BG)

    grid[:, :main_width] = input_main
    expected[:, :main_width] = expected_main
    grid[:, main_width] = int(sep_color)
    expected[:, main_width] = int(sep_color)
    grid[:, main_width + 1 :] = scatter
    expected[:, main_width + 1 :] = scatter

    boxes_for_validation = boxes
    if random.random() < MIRROR_PROB:
        grid = np.fliplr(grid)
        expected = np.fliplr(expected)
        boxes_for_validation = {
            color: (r0, main_width - 1 - c1, r1, main_width - 1 - c0)
            for color, (r0, c0, r1, c1) in boxes.items()
        }

    if not _validate_semantics(grid, expected, boxes_for_validation, input_order, target_order, count_by_color):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        attempt = _attempt()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("failed to generate a valid input for puzzle 3")
