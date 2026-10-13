import random

import numpy as np


BG = 0
MARKER = 2
FILL = 7
DIR_BY_COLOR = {
    3: "left",
    4: "down",
    6: "up",
    8: "right",
}
SQUARE_COLORS = tuple(DIR_BY_COLOR)

GRID_H_RANGE = (13, 18)
GRID_W_RANGE = (13, 18)
SQUARE_COUNT_CHOICES = [3, 4, 4]
EXTRA_MARKER_CHOICES = [0, 1, 2]
MAX_GENERATION_ATTEMPTS = 500
NEIGHBOR_OFFSETS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """BBox of an (N,2) coords array in the shared (r,c) convention."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _box_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {
        (r, c)
        for r in range(int(r0), int(r1) + 1)
        for c in range(int(c0), int(c1) + 1)
    }


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _box_hits_points(box: tuple[int, int, int, int], points: set[tuple[int, int]]) -> bool:
    return any(cell in points for cell in _box_cells(box))


def _square_border_coords(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    assert r1 - r0 == 2 and c1 - c0 == 2
    return [
        (r0, c0),
        (r0, c0 + 1),
        (r0, c1),
        (r0 + 1, c0),
        (r0 + 1, c1),
        (r1, c0),
        (r1, c0 + 1),
        (r1, c1),
    ]


def _render_square(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    for r, c in _square_border_coords(box):
        grid[r, c] = int(color)


def _render_scene(
    shape: tuple[int, int],
    boxes_by_color: dict[int, tuple[int, int, int, int]],
    markers: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for color in sorted(boxes_by_color):
        _render_square(out, boxes_by_color[color], color)
    for r, c in markers:
        out[int(r), int(c)] = MARKER
    return out


def _obstacle_mask(
    shape: tuple[int, int], boxes_by_color: dict[int, tuple[int, int, int, int]]
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for box in boxes_by_color.values():
        r0, c0, r1, c1 = map(int, box)
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _pair_specs_from_markers(
    markers: list[tuple[int, int]], shape: tuple[int, int]
) -> list[tuple[str, int, int, int]]:
    h, w = map(int, shape)
    by_row: dict[int, list[int]] = {}
    by_col: dict[int, list[int]] = {}
    for r, c in markers:
        by_row.setdefault(int(r), []).append(int(c))
        by_col.setdefault(int(c), []).append(int(r))

    pairs: list[tuple[str, int, int, int]] = []
    for r in range(h):
        cols = sorted(by_row.get(r, []))
        if len(cols) == 2:
            pairs.append(("row", int(r), int(cols[0]), int(cols[1])))
    for c in range(w):
        rows = sorted(by_col.get(c, []))
        if len(rows) == 2:
            pairs.append(("col", int(c), int(rows[0]), int(rows[1])))
    return pairs


def _segment_cells(pair: tuple[str, int, int, int]) -> list[tuple[int, int]]:
    kind, fixed, a, b = pair
    if kind == "row":
        return [(int(fixed), c) for c in range(int(a) + 1, int(b))]
    return [(r, int(fixed)) for r in range(int(a) + 1, int(b))]


def _pair_blocked(obstacle_mask: np.ndarray, pair: tuple[str, int, int, int]) -> bool:
    kind, fixed, a, b = pair
    if kind == "row":
        return bool(obstacle_mask[int(fixed), int(a) + 1 : int(b)].any())
    return bool(obstacle_mask[int(a) + 1 : int(b), int(fixed)].any())


def _apply_connectors(
    out: np.ndarray,
    markers: list[tuple[int, int]],
    obstacle_mask: np.ndarray,
) -> list[tuple[tuple[str, int, int, int], bool]]:
    outcomes: list[tuple[tuple[str, int, int, int], bool]] = []
    for pair in _pair_specs_from_markers(markers, out.shape):
        blocked = _pair_blocked(obstacle_mask, pair)
        outcomes.append((pair, blocked))
        if blocked:
            continue
        for r, c in _segment_cells(pair):
            if not obstacle_mask[r, c] and int(out[r, c]) == BG:
                out[r, c] = FILL
    return outcomes


def _find_squares(grid: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    boxes: dict[int, tuple[int, int, int, int]] = {}
    for color in SQUARE_COLORS:
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        assert len(coords) == 8, f"color {color} should form exactly one hollow 3x3 square"
        box = bbox_coords(coords)
        r0, c0, r1, c1 = box
        assert r1 - r0 == 2 and c1 - c0 == 2, f"color {color} must have a 3x3 bbox"
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        expected = np.full((3, 3), color, dtype=int)
        expected[1, 1] = BG
        assert np.array_equal(patch, expected), f"color {color} must be a hollow square"
        boxes[int(color)] = box
    return boxes


def _slide_box(
    box: tuple[int, int, int, int], direction: str, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = map(int, box)
    assert r1 - r0 == 2 and c1 - c0 == 2
    if direction == "left":
        return (r0, 0, r1, 2)
    if direction == "right":
        return (r0, w - 3, r1, w - 1)
    if direction == "up":
        return (0, c0, 2, c1)
    if direction == "down":
        return (h - 3, c0, h - 1, c1)
    raise AssertionError(direction)


def _transform(
    grid: np.ndarray, *, return_state: bool = False
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == MARKER)]
    boxes = _find_squares(g)
    assert 3 <= len(boxes) <= 4
    assert 4 <= len(markers) <= 6

    final_boxes = {
        color: _slide_box(box, DIR_BY_COLOR[color], g.shape) for color, box in boxes.items()
    }
    out = _render_scene(g.shape, final_boxes, markers)
    obstacle_mask = _obstacle_mask(g.shape, final_boxes)
    outcomes = _apply_connectors(out, markers, obstacle_mask)
    if return_state:
        return out, {"final_boxes": final_boxes, "pairs": outcomes}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide color-coded hollow squares to the edge, then let their landed bboxes gate orange connector fills between red markers.

    Input structure:
    - The background is black, with 3 or 4 hollow 3x3 squares whose colors are drawn from {green, yellow, magenta, teal}.
    - Each square is a monochrome 8-pixel border with a black center, and each color encodes one slide direction.
    - Four to six isolated red marker pixels are scattered so that some rows or columns contain exactly two markers.

    Rule:
    - Move every hollow square as a rigid 3x3 shape to its color-coded edge: green -> left, yellow -> down, magenta -> up, teal -> right.
    - After all slides, inspect every row with exactly two red markers and every column with exactly two red markers.
    - Treat each landed square's entire 3x3 bbox, including its hollow center, as an obstacle: if a marker pair's straight path crosses any obstacle cell, draw nothing; otherwise fill the background cells strictly between the markers with orange.
    - Keep all square pixels and red marker pixels unchanged.

    Key insight:
    - The connectors depend on the post-slide obstacle layout rather than the input layout, so the hollow squares must be moved first and only then used as 3x3 blocking gates.
    """
    return _transform(grid)


def _sample_primary_layout(
    h: int, w: int, color: int
) -> tuple[
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    list[tuple[int, int]],
    tuple[str, int, int, int],
    tuple[str, int, int, int],
]:
    direction = DIR_BY_COLOR[color]
    if direction == "up":
        c0 = random.randint(3, w - 6)
        start_r = random.randint(4, h - 6)
        final_box = (0, c0, 2, c0 + 2)
        initial_box = (start_r, c0, start_r + 2, c0 + 2)
        row_block = random.randint(0, 2)
        row_clear = random.randint(start_r, start_r + 2)
        lefts = random.sample(list(range(0, c0 - 1)), 2)
        rights = random.sample(list(range(c0 + 4, w)), 2)
        markers = [
            (row_block, lefts[0]),
            (row_block, rights[0]),
            (row_clear, lefts[1]),
            (row_clear, rights[1]),
        ]
        pair_block = ("row", row_block, lefts[0], rights[0])
        pair_clear = ("row", row_clear, lefts[1], rights[1])
        return initial_box, final_box, markers, pair_block, pair_clear

    if direction == "down":
        c0 = random.randint(3, w - 6)
        start_r = random.randint(1, h - 7)
        final_box = (h - 3, c0, h - 1, c0 + 2)
        initial_box = (start_r, c0, start_r + 2, c0 + 2)
        row_block = random.randint(h - 3, h - 1)
        row_clear = random.randint(start_r, start_r + 2)
        lefts = random.sample(list(range(0, c0 - 1)), 2)
        rights = random.sample(list(range(c0 + 4, w)), 2)
        markers = [
            (row_block, lefts[0]),
            (row_block, rights[0]),
            (row_clear, lefts[1]),
            (row_clear, rights[1]),
        ]
        pair_block = ("row", row_block, lefts[0], rights[0])
        pair_clear = ("row", row_clear, lefts[1], rights[1])
        return initial_box, final_box, markers, pair_block, pair_clear

    if direction == "left":
        r0 = random.randint(3, h - 6)
        start_c = random.randint(4, w - 6)
        final_box = (r0, 0, r0 + 2, 2)
        initial_box = (r0, start_c, r0 + 2, start_c + 2)
        col_block = random.randint(0, 2)
        col_clear = random.randint(start_c, start_c + 2)
        tops = random.sample(list(range(0, r0 - 1)), 2)
        bottoms = random.sample(list(range(r0 + 4, h)), 2)
        markers = [
            (tops[0], col_block),
            (bottoms[0], col_block),
            (tops[1], col_clear),
            (bottoms[1], col_clear),
        ]
        pair_block = ("col", col_block, tops[0], bottoms[0])
        pair_clear = ("col", col_clear, tops[1], bottoms[1])
        return initial_box, final_box, markers, pair_block, pair_clear

    assert direction == "right"
    r0 = random.randint(3, h - 6)
    start_c = random.randint(1, w - 7)
    final_box = (r0, w - 3, r0 + 2, w - 1)
    initial_box = (r0, start_c, r0 + 2, start_c + 2)
    col_block = random.randint(w - 3, w - 1)
    col_clear = random.randint(start_c, start_c + 2)
    tops = random.sample(list(range(0, r0 - 1)), 2)
    bottoms = random.sample(list(range(r0 + 4, h)), 2)
    markers = [
        (tops[0], col_block),
        (bottoms[0], col_block),
        (tops[1], col_clear),
        (bottoms[1], col_clear),
    ]
    pair_block = ("col", col_block, tops[0], bottoms[0])
    pair_clear = ("col", col_clear, tops[1], bottoms[1])
    return initial_box, final_box, markers, pair_block, pair_clear


def _extra_square_candidates(
    h: int, w: int, direction: str
) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    candidates: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    if direction == "up":
        for c0 in range(w - 2):
            final_box = (0, c0, 2, c0 + 2)
            for start_r in range(1, h - 2):
                initial_box = (start_r, c0, start_r + 2, c0 + 2)
                candidates.append((initial_box, final_box))
    elif direction == "down":
        for c0 in range(w - 2):
            final_box = (h - 3, c0, h - 1, c0 + 2)
            for start_r in range(0, h - 3):
                initial_box = (start_r, c0, start_r + 2, c0 + 2)
                candidates.append((initial_box, final_box))
    elif direction == "left":
        for r0 in range(h - 2):
            final_box = (r0, 0, r0 + 2, 2)
            for start_c in range(1, w - 2):
                initial_box = (r0, start_c, r0 + 2, start_c + 2)
                candidates.append((initial_box, final_box))
    else:
        assert direction == "right"
        for r0 in range(h - 2):
            final_box = (r0, w - 3, r0 + 2, w - 1)
            for start_c in range(0, w - 3):
                initial_box = (r0, start_c, r0 + 2, start_c + 2)
                candidates.append((initial_box, final_box))
    random.shuffle(candidates)
    return candidates


def _sample_extra_square(
    h: int,
    w: int,
    color: int,
    protected_points: set[tuple[int, int]],
    occupied_initial: list[tuple[int, int, int, int]],
    occupied_final: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    for initial_box, final_box in _extra_square_candidates(h, w, DIR_BY_COLOR[color]):
        if any(_boxes_overlap(initial_box, other) for other in occupied_initial):
            continue
        if any(_boxes_overlap(final_box, other) for other in occupied_final):
            continue
        if _box_hits_points(initial_box, protected_points):
            continue
        if _box_hits_points(final_box, protected_points):
            continue
        return initial_box, final_box
    return None


def _markers_are_isolated(markers: list[tuple[int, int]]) -> bool:
    for i, (r0, c0) in enumerate(markers):
        for r1, c1 in markers[i + 1 :]:
            if max(abs(int(r0) - int(r1)), abs(int(c0) - int(c1))) <= 1:
                return False
    return True


def _sample_extra_markers(
    h: int,
    w: int,
    extra_count: int,
    initial_boxes: dict[int, tuple[int, int, int, int]],
    final_boxes: dict[int, tuple[int, int, int, int]],
    markers: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    used_rows = {int(r) for r, _c in markers}
    used_cols = {int(c) for _r, c in markers}
    forbidden = set()
    for box in list(initial_boxes.values()) + list(final_boxes.values()):
        forbidden.update(_box_cells(box))

    all_markers = markers[:]
    for _ in range(extra_count):
        candidates = []
        for r in range(h):
            if r in used_rows:
                continue
            for c in range(w):
                if c in used_cols or (r, c) in forbidden:
                    continue
                if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in all_markers):
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        cell = random.choice(candidates)
        all_markers.append(cell)
        used_rows.add(int(cell[0]))
        used_cols.add(int(cell[1]))
    return all_markers


def _pair_status_map(
    markers: list[tuple[int, int]],
    shape: tuple[int, int],
    boxes_by_color: dict[int, tuple[int, int, int, int]],
) -> dict[tuple[str, int, int, int], bool]:
    obstacle_mask = _obstacle_mask(shape, boxes_by_color)
    return {
        pair: _pair_blocked(obstacle_mask, pair)
        for pair in _pair_specs_from_markers(markers, shape)
    }


def _expected_output(
    shape: tuple[int, int],
    final_boxes: dict[int, tuple[int, int, int, int]],
    markers: list[tuple[int, int]],
) -> tuple[np.ndarray, list[tuple[tuple[str, int, int, int], bool]]]:
    out = _render_scene(shape, final_boxes, markers)
    outcomes = _apply_connectors(out, markers, _obstacle_mask(shape, final_boxes))
    return out, outcomes


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    square_count = random.choice(SQUARE_COUNT_CHOICES)
    primary_color = random.choice(SQUARE_COLORS)
    extra_colors = random.sample(
        [color for color in SQUARE_COLORS if color != primary_color], square_count - 1
    )

    initial_primary, final_primary, markers, pair_block, pair_clear = _sample_primary_layout(
        h, w, primary_color
    )
    assert _markers_are_isolated(markers)

    initial_boxes = {primary_color: initial_primary}
    final_boxes = {primary_color: final_primary}
    protected_points = set(markers)
    protected_points.update(_segment_cells(pair_block))
    protected_points.update(_segment_cells(pair_clear))

    occupied_initial = [initial_primary]
    occupied_final = [final_primary]
    for color in extra_colors:
        result = _sample_extra_square(
            h,
            w,
            color,
            protected_points,
            occupied_initial,
            occupied_final,
        )
        if result is None:
            return None
        initial_box, final_box = result
        initial_boxes[color] = initial_box
        final_boxes[color] = final_box
        occupied_initial.append(initial_box)
        occupied_final.append(final_box)

    markers = _sample_extra_markers(
        h,
        w,
        random.choice(EXTRA_MARKER_CHOICES),
        initial_boxes,
        final_boxes,
        markers,
    )
    if markers is None or not _markers_are_isolated(markers):
        return None

    before = _pair_status_map(markers, (h, w), initial_boxes)
    after = _pair_status_map(markers, (h, w), final_boxes)
    if pair_block not in before or pair_clear not in before:
        return None
    if before[pair_block] or not after[pair_block]:
        return None
    if not before[pair_clear] or after[pair_clear]:
        return None
    if not any(after.values()) or not any(not blocked for blocked in after.values()):
        return None
    if not any(before[pair] != after[pair] for pair in after):
        return None

    inp = _render_scene((h, w), initial_boxes, markers)
    expected, outcomes = _expected_output((h, w), final_boxes, markers)
    pair_count = len(outcomes)
    if not (2 <= pair_count <= 4):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-6 input")
