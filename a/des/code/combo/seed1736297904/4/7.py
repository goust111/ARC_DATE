from __future__ import annotations

import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each same-color constellation's bounding box, then recolor by rank.

    Input structure:
    - Background is black.
    - Each non-black color appears once in a right-side vertical color bar and
      several more times as scattered 1x1 pixels elsewhere.
    - The scattered pixels of one color form a single constellation group.

    Rule:
    - For each color, ignore its bar pixel and take the bbox of the remaining
      pixels.
    - Order colors by constellation size from fewest to most pixels.
    - The bar read top-to-bottom gives the replacement colors for ranks
      1..k.
    - Draw the bboxes on a black canvas, with larger constellations painted
      later so they sit on top of smaller ones where bboxes overlap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bar_col, bar_rows, bar_colors = _find_bar(g)
    items = _constellations(g, bar_col, bar_rows, bar_colors)

    counts = [item["count"] for item in items]
    assert len(set(counts)) == len(counts), "constellation sizes must be distinct"

    ranked = sorted(items, key=lambda item: item["count"])
    recolor = {ranked[i]["color"]: int(bar_colors[i]) for i in range(len(ranked))}

    out = np.zeros_like(g)
    for item in ranked:
        r0, c0, r1, c1 = item["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = recolor[item["color"]]
    return out


BG = 0
COLOR_POOL = tuple(range(1, 10))
COUNT_POOL = (3, 4, 5, 6)

GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (18, 30)
N_COLORS_RANGE = (2, 4)
BAR_GAP_RANGE = (2, 4)
BAR_RIGHT_MARGIN_OPTIONS = (0, 1)
OVERLAP_PROB = 0.5
GEN_ATTEMPTS = 500
PLACEMENT_TRIES = 200

PATTERN_LIBRARY: dict[int, list[tuple[int, int, tuple[tuple[int, int], ...]]]] = {
    3: [
        (2, 3, ((0, 0), (0, 2), (1, 1))),
        (3, 2, ((0, 0), (1, 1), (2, 0))),
        (3, 3, ((0, 0), (0, 2), (2, 1))),
    ],
    4: [
        (2, 4, ((0, 0), (0, 2), (1, 1), (1, 3))),
        (4, 2, ((0, 0), (1, 1), (2, 0), (3, 1))),
        (3, 3, ((0, 0), (0, 2), (2, 0), (2, 2))),
    ],
    5: [
        (3, 4, ((0, 0), (0, 2), (1, 3), (2, 0), (2, 2))),
        (4, 3, ((0, 0), (0, 2), (1, 1), (2, 0), (3, 1))),
    ],
    6: [
        (3, 4, ((0, 0), (0, 2), (1, 1), (1, 3), (2, 0), (2, 2))),
        (4, 3, ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2), (3, 1))),
    ],
}


def _find_bar(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    colors = [int(c) for c in np.unique(grid) if int(c) != BG]
    assert colors, "expected at least one foreground color"
    all_colors = set(colors)

    candidates: list[tuple[int, list[int], list[int]]] = []
    for col in range(grid.shape[1] - 1, -1, -1):
        rows = np.flatnonzero(grid[:, col] != BG)
        if rows.size != len(colors):
            continue
        if int(rows[-1] - rows[0] + 1) != int(rows.size):
            continue
        bar_colors = [int(grid[int(r), int(col)]) for r in rows]
        if len(set(bar_colors)) != len(bar_colors):
            continue
        if set(bar_colors) != all_colors:
            continue
        candidates.append((int(col), [int(r) for r in rows], bar_colors))

    assert candidates, "expected a right-side color bar"
    return max(candidates, key=lambda item: item[0])


def _constellations(
    grid: np.ndarray, bar_col: int, bar_rows: list[int], bar_colors: list[int]
) -> list[dict[str, int | tuple[int, int, int, int]]]:
    items: list[dict[str, int | tuple[int, int, int, int]]] = []
    for row, color in zip(bar_rows, bar_colors):
        coords = np.argwhere(grid == int(color))
        points = [
            (int(r), int(c))
            for r, c in coords
            if not (int(r) == int(row) and int(c) == int(bar_col))
        ]
        assert points, "every bar color must also appear in a constellation"

        rows = [r for r, _c in points]
        cols = [c for _r, c in points]
        items.append(
            {
                "color": int(color),
                "count": int(len(points)),
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
            }
        )
    return items


def _transform_pattern(
    h: int, w: int, cells: tuple[tuple[int, int], ...], rng: random.Random
) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    pts = list(cells)
    if h != w and rng.random() < 0.5:
        pts = [(c, r) for r, c in pts]
        h, w = w, h
    if rng.random() < 0.5:
        pts = [(r, w - 1 - c) for r, c in pts]
    if rng.random() < 0.5:
        pts = [(h - 1 - r, c) for r, c in pts]
    pts = sorted(set(pts))
    return h, w, tuple(pts)


def _choose_pattern(count: int, rng: random.Random) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    base_h, base_w, base_cells = rng.choice(PATTERN_LIBRARY[count])
    return _transform_pattern(base_h, base_w, base_cells, rng)


def _bbox_from_top_left(top: int, left: int, h: int, w: int) -> tuple[int, int, int, int]:
    return (int(top), int(left), int(top + h - 1), int(left + w - 1))


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _bboxes_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1
) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _bbox_contains(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _shifted_points(
    top: int, left: int, cells: tuple[tuple[int, int], ...]
) -> set[tuple[int, int]]:
    return {(int(top + r), int(left + c)) for r, c in cells}


def _rect_area(box: tuple[int, int, int, int]) -> int:
    return int((box[2] - box[0] + 1) * (box[3] - box[1] + 1))


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    if not _bboxes_overlap(a, b):
        return 0
    row_overlap = min(a[2], b[2]) - max(a[0], b[0]) + 1
    col_overlap = min(a[3], b[3]) - max(a[1], b[1]) + 1
    return int(row_overlap * col_overlap)


def _place_nonoverlapping(
    height: int,
    max_col: int,
    h: int,
    w: int,
    existing_boxes: list[tuple[int, int, int, int]],
    occupied_points: set[tuple[int, int]],
    cells: tuple[tuple[int, int], ...],
    rng: random.Random,
) -> tuple[tuple[int, int, int, int], set[tuple[int, int]]] | None:
    if max_col - w + 1 < 0:
        return None

    for _ in range(PLACEMENT_TRIES):
        top = rng.randint(0, height - h)
        left = rng.randint(0, max_col - w + 1)
        box = _bbox_from_top_left(top, left, h, w)
        if any(_bboxes_too_close(box, other, pad=1) for other in existing_boxes):
            continue
        points = _shifted_points(top, left, cells)
        if points & occupied_points:
            continue
        return box, points
    return None


def _place_overlapping(
    anchor: tuple[int, int, int, int],
    height: int,
    max_col: int,
    h: int,
    w: int,
    existing_boxes: list[tuple[int, int, int, int]],
    occupied_points: set[tuple[int, int]],
    cells: tuple[tuple[int, int], ...],
    rng: random.Random,
) -> tuple[tuple[int, int, int, int], set[tuple[int, int]]] | None:
    if max_col - w + 1 < 0:
        return None

    min_top = max(0, anchor[0] - h + 1)
    max_top = min(height - h, anchor[2])
    min_left = max(0, anchor[1] - w + 1)
    max_left = min(max_col - w + 1, anchor[3])
    if min_top > max_top or min_left > max_left:
        return None

    for _ in range(PLACEMENT_TRIES):
        top = rng.randint(min_top, max_top)
        left = rng.randint(min_left, max_left)
        box = _bbox_from_top_left(top, left, h, w)
        if not _bboxes_overlap(box, anchor):
            continue
        if _bbox_contains(box, anchor) or _bbox_contains(anchor, box):
            continue
        inter = _intersection_area(box, anchor)
        if inter <= 0 or inter >= min(_rect_area(box), _rect_area(anchor)):
            continue
        blocked = False
        for other in existing_boxes:
            if other == anchor:
                continue
            if _bboxes_too_close(box, other, pad=1):
                blocked = True
                break
        if blocked:
            continue
        points = _shifted_points(top, left, cells)
        if points & occupied_points:
            continue
        return box, points
    return None


def _render_expected(
    shape: tuple[int, int],
    boxes_by_color: dict[int, tuple[int, int, int, int]],
    counts_by_color: dict[int, int],
    bar_colors: list[int],
) -> np.ndarray:
    ranked_colors = sorted(counts_by_color, key=lambda color: counts_by_color[color])
    recolor = {ranked_colors[i]: int(bar_colors[i]) for i in range(len(ranked_colors))}

    out = np.zeros(shape, dtype=int)
    for color in ranked_colors:
        r0, c0, r1, c1 = boxes_by_color[color]
        out[r0 : r1 + 1, c0 : c1 + 1] = recolor[color]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_colors = rng.randint(*N_COLORS_RANGE)
    colors = rng.sample(COLOR_POOL, n_colors)
    counts = rng.sample(COUNT_POOL, n_colors)
    rng.shuffle(counts)
    counts_by_color = {int(color): int(count) for color, count in zip(colors, counts)}

    ranked_colors = sorted(colors, key=lambda color: counts_by_color[int(color)])
    bar_colors = rng.sample(colors, len(colors))
    if len(colors) > 1:
        while bar_colors == ranked_colors:
            bar_colors = rng.sample(colors, len(colors))

    patterns = {int(color): _choose_pattern(counts_by_color[int(color)], rng) for color in colors}

    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    bar_col = width - 1 - rng.choice(BAR_RIGHT_MARGIN_OPTIONS)
    gap = rng.randint(*BAR_GAP_RANGE)
    max_workspace_col = bar_col - gap - 1
    if max_workspace_col < 4:
        return None

    bar_top_min = 1
    bar_top_max = height - n_colors - 1
    if bar_top_max < bar_top_min:
        return None
    bar_top = rng.randint(bar_top_min, bar_top_max)
    bar_rows = list(range(bar_top, bar_top + n_colors))

    overlap_pair: tuple[int, int] | None = None
    if n_colors >= 2 and rng.random() < OVERLAP_PROB:
        pair = rng.sample(colors, 2)
        overlap_pair = (int(pair[0]), int(pair[1]))

    boxes_by_color: dict[int, tuple[int, int, int, int]] = {}
    points_by_color: dict[int, set[tuple[int, int]]] = {}
    occupied_points: set[tuple[int, int]] = set()

    placement_order = sorted(
        colors,
        key=lambda color: (
            0 if overlap_pair and int(color) == overlap_pair[0] else 1,
            -(patterns[int(color)][0] * patterns[int(color)][1]),
        ),
    )

    for color in placement_order:
        color = int(color)
        h, w, cells = patterns[color]
        existing = list(boxes_by_color.values())

        if overlap_pair and color == overlap_pair[1]:
            anchor = boxes_by_color.get(overlap_pair[0])
            if anchor is None:
                return None
            placed = _place_overlapping(
                anchor, height, max_workspace_col, h, w, existing, occupied_points, cells, rng
            )
        else:
            placed = _place_nonoverlapping(
                height, max_workspace_col, h, w, existing, occupied_points, cells, rng
            )

        if placed is None:
            return None

        box, points = placed
        boxes_by_color[color] = box
        points_by_color[color] = points
        occupied_points |= points

    if overlap_pair and not _bboxes_overlap(
        boxes_by_color[overlap_pair[0]], boxes_by_color[overlap_pair[1]]
    ):
        return None
    if not overlap_pair:
        color_list = list(boxes_by_color)
        for i, color_a in enumerate(color_list):
            for color_b in color_list[i + 1 :]:
                if _bboxes_overlap(boxes_by_color[color_a], boxes_by_color[color_b]):
                    return None

    grid = np.zeros((height, width), dtype=int)
    for color, points in points_by_color.items():
        for r, c in points:
            grid[r, c] = int(color)
    for row, color in zip(bar_rows, bar_colors):
        grid[row, bar_col] = int(color)

    try:
        found_col, found_rows, found_colors = _find_bar(grid)
    except AssertionError:
        return None
    if found_col != bar_col or found_rows != bar_rows or found_colors != [int(c) for c in bar_colors]:
        return None

    expected = _render_expected(grid.shape, boxes_by_color, counts_by_color, [int(c) for c in bar_colors])
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
