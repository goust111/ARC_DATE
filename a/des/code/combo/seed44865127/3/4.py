import numpy as np
from collections import Counter


Rect = tuple[int, int, int, int]


RED = 2
BLUE = 1
GREEN = 3
LINE_COLORS = {RED, BLUE, GREEN}
RELATIONS = ("horizontal", "vertical", "diagonal")


# --------------- solver helpers ---------------


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _center(lo: int, hi: int) -> int:
    return (int(lo) + int(hi)) // 2


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == int(color)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            stack = [(int(r), int(c))]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> Rect:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _collect_pairs(grid: np.ndarray) -> tuple[int, list[tuple[int, Rect, Rect]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    pairs: list[tuple[int, Rect, Rect]] = []
    for color in (int(c) for c in np.unique(g) if int(c) != bg):
        components = _components_of_color(g, color)
        assert len(components) == 2, "each foreground color must appear in exactly two rectangles"

        rects: list[Rect] = []
        for coords in components:
            rect = _bbox(coords)
            r0, c0, r1, c1 = rect
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            assert len(coords) == height * width, "each object must be a solid rectangle"
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle footprint must be solid"
            rects.append(rect)
        pairs.append((color, rects[0], rects[1]))

    assert pairs, "expected at least one color pair"
    return bg, pairs


def _connection_plan(rect_a: Rect, rect_b: Rect) -> tuple[str, int, list[tuple[int, int]]]:
    ra0, ca0, ra1, ca1 = rect_a
    rb0, cb0, rb1, cb1 = rect_b

    row_overlap = _intervals_overlap(ra0, ra1, rb0, rb1)
    col_overlap = _intervals_overlap(ca0, ca1, cb0, cb1)
    assert not (row_overlap and col_overlap), "paired rectangles must be spatially separate"

    if row_overlap and not col_overlap:
        left, right = sorted((rect_a, rect_b), key=lambda rect: (rect[1], rect[0]))
        row = _center(max(left[0], right[0]), min(left[2], right[2]))
        cells = [(int(row), int(c)) for c in range(left[3] + 1, right[1])]
        assert cells, "horizontal pairs need a background gap"
        return "horizontal", RED, cells

    if col_overlap and not row_overlap:
        top, bottom = sorted((rect_a, rect_b), key=lambda rect: (rect[0], rect[1]))
        col = _center(max(top[1], bottom[1]), min(top[3], bottom[3]))
        cells = [(int(r), int(col)) for r in range(top[2] + 1, bottom[0])]
        assert cells, "vertical pairs need a background gap"
        return "vertical", BLUE, cells

    assert not row_overlap and not col_overlap
    left, right = sorted((rect_a, rect_b), key=lambda rect: (rect[1], rect[0]))
    start_row = _center(left[0], left[2])
    end_row = _center(right[0], right[2])
    bend_col = right[1] - 1
    horiz = [(int(start_row), int(c)) for c in range(left[3] + 1, bend_col + 1)]
    assert horiz, "diagonal pairs need horizontal space between rectangles"

    step = 1 if end_row > start_row else -1
    vert = [(int(r), int(bend_col)) for r in range(start_row + step, end_row + step, step)]
    assert vert, "diagonal pairs need vertical separation"
    return "diagonal", GREEN, horiz + vert


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color rectangle pair using a relation-dependent line color.

    Input structure
    - The dominant color is the background.
    - Each foreground color appears in exactly two disconnected solid rectangles.

    Rule
    - Horizontal alignment (overlapping rows only) gets a red horizontal connector.
    - Vertical alignment (overlapping columns only) gets a blue vertical connector.
    - Diagonal placement (no row overlap and no column overlap) gets a green L-shape.
    - Original rectangles remain unchanged; only background gap cells are recolored.

    Key insight
    The connector color depends on the spatial relationship of the rectangle pair,
    not on the rectangle color itself.
    """
    g = np.asarray(grid, dtype=int)
    bg, pairs = _collect_pairs(g)

    out = g.copy()
    used_cells: set[tuple[int, int]] = set()
    for _color, rect_a, rect_b in pairs:
        _relation, line_color, cells = _connection_plan(rect_a, rect_b)
        for cell in cells:
            r, c = cell
            assert int(g[r, c]) == bg, "connector paths must run through background cells"
            assert cell not in used_cells, "connector paths must not overlap each other"
            out[r, c] = int(line_color)
            used_cells.add(cell)
    return out


# --------------- generator ---------------


GRID_H_MIN = 24
GRID_H_MAX_EXCLUSIVE = 31
GRID_W_MIN = 24
GRID_W_MAX_EXCLUSIVE = 31

PAIR_COUNT_MIN = 3
PAIR_COUNT_MAX_EXCLUSIVE = 5

RECT_SIDE_MIN = 2
RECT_SIDE_MAX_EXCLUSIVE = 5
MIN_GAP = 2

OUTER_MARGIN = 1
ZONE_GUTTER = 2
MIN_ZONE_SIZE = 10

GEN_ATTEMPTS = 300
OBJECT_COLOR_POOL = tuple(c for c in range(10) if c not in LINE_COLORS)


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


def _sample_size(rng: np.random.Generator) -> tuple[int, int]:
    h = int(rng.integers(RECT_SIDE_MIN, RECT_SIDE_MAX_EXCLUSIVE))
    w = int(rng.integers(RECT_SIDE_MIN, RECT_SIDE_MAX_EXCLUSIVE))
    return h, w


def _zone_layout(height: int, width: int, rng: np.random.Generator) -> list[Rect]:
    usable_h = int(height) - 2 * OUTER_MARGIN - ZONE_GUTTER
    usable_w = int(width) - 2 * OUTER_MARGIN - ZONE_GUTTER
    assert usable_h >= 2 * MIN_ZONE_SIZE and usable_w >= 2 * MIN_ZONE_SIZE

    top_h = _randint(rng, MIN_ZONE_SIZE, usable_h - MIN_ZONE_SIZE)
    left_w = _randint(rng, MIN_ZONE_SIZE, usable_w - MIN_ZONE_SIZE)
    bottom_h = usable_h - top_h
    right_w = usable_w - left_w

    top_r0 = OUTER_MARGIN
    top_r1 = top_r0 + top_h - 1
    bottom_r0 = top_r1 + ZONE_GUTTER + 1
    bottom_r1 = bottom_r0 + bottom_h - 1

    left_c0 = OUTER_MARGIN
    left_c1 = left_c0 + left_w - 1
    right_c0 = left_c1 + ZONE_GUTTER + 1
    right_c1 = right_c0 + right_w - 1

    return [
        (top_r0, left_c0, top_r1, left_c1),
        (top_r0, right_c0, top_r1, right_c1),
        (bottom_r0, left_c0, bottom_r1, left_c1),
        (bottom_r0, right_c0, bottom_r1, right_c1),
    ]


def _sample_horizontal_pair(zone: Rect, rng: np.random.Generator) -> tuple[Rect, Rect] | None:
    r0, c0, r1, c1 = zone
    zone_h = r1 - r0 + 1
    zone_w = c1 - c0 + 1

    for _ in range(120):
        h1, w1 = _sample_size(rng)
        h2, w2 = _sample_size(rng)
        if max(h1, h2) > zone_h or w1 + w2 + MIN_GAP > zone_w:
            continue

        gap = _randint(rng, MIN_GAP, zone_w - w1 - w2)
        total_w = w1 + gap + w2
        left_pad = _randint(rng, 0, zone_w - total_w)
        ca0 = c0 + left_pad
        cb0 = ca0 + w1 + gap

        ra0 = _randint(rng, r0, r1 - h1 + 1)
        rb0 = _randint(rng, r0, r1 - h2 + 1)
        if not _intervals_overlap(ra0, ra0 + h1 - 1, rb0, rb0 + h2 - 1):
            continue

        rect_a = (ra0, ca0, ra0 + h1 - 1, ca0 + w1 - 1)
        rect_b = (rb0, cb0, rb0 + h2 - 1, cb0 + w2 - 1)
        return rect_a, rect_b
    return None


def _sample_vertical_pair(zone: Rect, rng: np.random.Generator) -> tuple[Rect, Rect] | None:
    r0, c0, r1, c1 = zone
    zone_h = r1 - r0 + 1
    zone_w = c1 - c0 + 1

    for _ in range(120):
        h1, w1 = _sample_size(rng)
        h2, w2 = _sample_size(rng)
        if h1 + h2 + MIN_GAP > zone_h or max(w1, w2) > zone_w:
            continue

        gap = _randint(rng, MIN_GAP, zone_h - h1 - h2)
        total_h = h1 + gap + h2
        top_pad = _randint(rng, 0, zone_h - total_h)
        ra0 = r0 + top_pad
        rb0 = ra0 + h1 + gap

        ca0 = _randint(rng, c0, c1 - w1 + 1)
        cb0 = _randint(rng, c0, c1 - w2 + 1)
        if not _intervals_overlap(ca0, ca0 + w1 - 1, cb0, cb0 + w2 - 1):
            continue

        rect_a = (ra0, ca0, ra0 + h1 - 1, ca0 + w1 - 1)
        rect_b = (rb0, cb0, rb0 + h2 - 1, cb0 + w2 - 1)
        return rect_a, rect_b
    return None


def _sample_diagonal_pair(zone: Rect, rng: np.random.Generator) -> tuple[Rect, Rect] | None:
    r0, c0, r1, c1 = zone
    zone_h = r1 - r0 + 1
    zone_w = c1 - c0 + 1

    for _ in range(120):
        h1, w1 = _sample_size(rng)
        h2, w2 = _sample_size(rng)
        if h1 + h2 + MIN_GAP > zone_h or w1 + w2 + MIN_GAP > zone_w:
            continue

        row_gap = _randint(rng, MIN_GAP, zone_h - h1 - h2)
        col_gap = _randint(rng, MIN_GAP, zone_w - w1 - w2)
        total_h = h1 + h2 + row_gap
        total_w = w1 + w2 + col_gap

        top_pad = _randint(rng, 0, zone_h - total_h)
        left_pad = _randint(rng, 0, zone_w - total_w)

        left_c0 = c0 + left_pad
        right_c0 = left_c0 + w1 + col_gap
        top_anchor = r0 + top_pad

        if bool(rng.integers(0, 2)):
            left_r0 = top_anchor
            right_r0 = left_r0 + h1 + row_gap
        else:
            right_r0 = top_anchor
            left_r0 = right_r0 + h2 + row_gap

        rect_a = (left_r0, left_c0, left_r0 + h1 - 1, left_c0 + w1 - 1)
        rect_b = (right_r0, right_c0, right_r0 + h2 - 1, right_c0 + w2 - 1)
        return rect_a, rect_b
    return None


def _sample_pair(zone: Rect, relation: str, rng: np.random.Generator) -> tuple[Rect, Rect] | None:
    if relation == "horizontal":
        return _sample_horizontal_pair(zone, rng)
    if relation == "vertical":
        return _sample_vertical_pair(zone, rng)
    assert relation == "diagonal"
    return _sample_diagonal_pair(zone, rng)


def _paint_rectangle(grid: np.ndarray, rect: Rect, color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _rect_cells(rect: Rect) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {(int(r), int(c)) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _validate_generation(grid: np.ndarray, expected_relations: list[str]) -> bool:
    try:
        bg, pairs = _collect_pairs(grid)
    except AssertionError:
        return False

    if len(pairs) != len(expected_relations):
        return False

    found_relations = Counter()
    occupied = set()
    for _color, rect_a, rect_b in pairs:
        occupied |= _rect_cells(rect_a)
        occupied |= _rect_cells(rect_b)

    used_lines: set[tuple[int, int]] = set()
    for _color, rect_a, rect_b in pairs:
        relation, _line_color, cells = _connection_plan(rect_a, rect_b)
        found_relations[relation] += 1
        for cell in cells:
            if cell in occupied or cell in used_lines:
                return False
            r, c = cell
            if int(grid[r, c]) != bg:
                return False
            used_lines.add(cell)

    if found_relations != Counter(expected_relations):
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False

    return bool(np.any(out != grid))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_H_MIN, GRID_H_MAX_EXCLUSIVE))
    width = int(rng.integers(GRID_W_MIN, GRID_W_MAX_EXCLUSIVE))
    bg = int(rng.choice(OBJECT_COLOR_POOL))

    object_colors = [color for color in OBJECT_COLOR_POOL if color != bg]
    pair_count = int(rng.integers(PAIR_COUNT_MIN, PAIR_COUNT_MAX_EXCLUSIVE))
    colors = [int(c) for c in rng.choice(np.array(object_colors), size=pair_count, replace=False)]

    relations = list(RELATIONS)
    for _ in range(pair_count - len(RELATIONS)):
        relations.append(str(rng.choice(RELATIONS)))
    rng.shuffle(relations)

    zones = _zone_layout(height, width, rng)
    zone_indices = [int(i) for i in rng.permutation(len(zones))[:pair_count]]

    grid = np.full((height, width), bg, dtype=int)
    for color, relation, zone_idx in zip(colors, relations, zone_indices):
        sample = _sample_pair(zones[zone_idx], relation, rng)
        if sample is None:
            return None
        rect_a, rect_b = sample
        _paint_rectangle(grid, rect_a, color)
        _paint_rectangle(grid, rect_b, color)

    if not _validate_generation(grid, relations):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
