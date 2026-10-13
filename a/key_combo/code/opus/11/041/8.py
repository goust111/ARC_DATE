import random
from itertools import combinations, permutations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

Coord = tuple[int, int]

GRID_H_RANGE = (18, 25)
GRID_W_RANGE = (18, 25)
RECT_COUNT_CHOICES = (2, 3)
RECT_SIZE_RANGE = (4, 6)
SEGMENT_LEN_RANGE = (8, 12)
SPRITE_SHAPES = (
    ((0, 0), (-1, 0), (1, 0), (0, -1)),
    ((0, 0), (-1, 0), (0, -1), (0, 1)),
    ((0, 0), (1, 0), (0, -1), (0, 1)),
    ((0, 0), (-1, 0), (1, 0), (0, 1), (1, 1)),
)
MAX_GENERATE_ATTEMPTS = 800


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _segment_endpoints(coords: np.ndarray) -> tuple[Coord, Coord]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if np.all(rows == rows[0]):
        row = int(rows[0])
        c0 = int(cols.min())
        c1 = int(cols.max())
        assert len(coords) == c1 - c0 + 1
        return (row, c0), (row, c1)
    if np.all(cols == cols[0]):
        col = int(cols[0])
        r0 = int(rows.min())
        r1 = int(rows.max())
        assert len(coords) == r1 - r0 + 1
        return (r0, col), (r1, col)
    raise AssertionError("red object must be one straight horizontal or vertical segment")


def _bresenham(a: Coord, b: Coord) -> list[Coord]:
    r0, c0 = map(int, a)
    r1, c1 = map(int, b)
    x0, y0, x1, y1 = c0, r0, c1, r1
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    pts: list[Coord] = []
    while True:
        pts.append((y0, x0))
        if x0 == x1 and y0 == y1:
            return pts
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _sprite_components(grid: np.ndarray) -> tuple[list[np.ndarray], list[Coord], tuple[Coord, ...]]:
    comps = _components_from_mask(np.isin(grid, (BLUE, YELLOW)))
    assert len(comps) == 3
    anchors: list[Coord] = []
    normalized_shapes: list[tuple[Coord, ...]] = []
    for comp in comps:
        colors = [int(grid[r, c]) for r, c in comp]
        assert colors.count(YELLOW) == 1
        anchor_arr = comp[np.array(colors) == YELLOW][0]
        anchor = (int(anchor_arr[0]), int(anchor_arr[1]))
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 <= 2 and c1 - c0 <= 2
        normalized = tuple(sorted((int(r - anchor[0]), int(c - anchor[1])) for r, c in comp))
        assert all(-1 <= dr <= 1 and -1 <= dc <= 1 for dr, dc in normalized)
        anchors.append(anchor)
        normalized_shapes.append(normalized)
    shape0 = normalized_shapes[0]
    assert all(shape == shape0 for shape in normalized_shapes[1:])
    return comps, anchors, shape0


def _stamp_sprite(grid: np.ndarray, anchor: Coord, offsets: tuple[Coord, ...]) -> None:
    ar, ac = anchor
    h, w = grid.shape
    for dr, dc in offsets:
        r = ar + dr
        c = ac + dc
        assert 0 <= r < h and 0 <= c < w
        grid[r, c] = YELLOW if (dr, dc) == (0, 0) else BLUE


def _line_mask_from_anchors(shape: tuple[int, int], anchors: list[Coord], grey_mask: np.ndarray) -> np.ndarray:
    red_mask = np.zeros(shape, dtype=bool)
    for a, b in combinations(anchors, 2):
        for r, c in _bresenham(a, b):
            if grey_mask[r, c]:
                red_mask[r, c] = True
    return red_mask


def _best_endpoint_assignment(endpoints: tuple[Coord, Coord], anchors: list[Coord]) -> tuple[int, int]:
    ranked: list[tuple[int, tuple[int, int]]] = []
    for ia, ib in permutations(range(len(anchors)), 2):
        d0 = abs(endpoints[0][0] - anchors[ia][0]) + abs(endpoints[0][1] - anchors[ia][1])
        d1 = abs(endpoints[1][0] - anchors[ib][0]) + abs(endpoints[1][1] - anchors[ib][1])
        ranked.append((d0 + d1, (ia, ib)))
    ranked.sort(key=lambda item: (item[0], item[1]))
    assert ranked[0][0] < ranked[1][0], "endpoint-anchor pairing must be unique"
    return ranked[0][1]


def _build_output(
    shape: tuple[int, int],
    grey_mask: np.ndarray,
    anchors: list[Coord],
    segment_endpoints: tuple[Coord, Coord],
    sprite_offsets: tuple[Coord, ...],
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    out[grey_mask] = GREY
    red_mask = _line_mask_from_anchors(shape, anchors, grey_mask)
    out[red_mask] = RED
    _stamp_sprite(out, segment_endpoints[0], sprite_offsets)
    _stamp_sprite(out, segment_endpoints[1], sprite_offsets)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap anchored sprites onto the original segment endpoints, reconnect the original anchors, and keep only the line parts that fall inside grey rectangles.

    Input structure
    A black background contains one red horizontal or vertical segment, three identical blue 4-connected sprites whose unique yellow center cell is the anchor, and two or three solid grey rectangles. The sprites and the red segment are disjoint from the grey rectangles.

    Rule
    Find the two endpoints of the red segment. Find the three yellow sprite anchors and the shared sprite shape. Swap roles: place one copy of that sprite at each red endpoint, with the unique nearest-anchor pairing only serving to choose which identical copy moves where. Then draw Bresenham red lines between every pair of original anchors, but keep only those line pixels whose coordinates lie inside the grey rectangles. Erase the original segment and the original sprite locations; preserve the grey rectangles.

    Key insight
    The visible red result is controlled by two independent references: the original sprite anchors determine where the new lines run, while the unrelated grey rectangles decide which fragments of those lines survive.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert set(np.unique(g)).issubset({BLACK, BLUE, RED, YELLOW, GREY})
    assert _get_bg(g) == BLACK

    red_components = _components_from_mask(g == RED)
    assert len(red_components) == 1
    segment_endpoints = _segment_endpoints(red_components[0])

    sprite_components, anchors, sprite_offsets = _sprite_components(g)
    assert len({anchor for anchor in anchors}) == 3
    assert (anchors[1][0] - anchors[0][0]) * (anchors[2][1] - anchors[0][1]) != (
        anchors[2][0] - anchors[0][0]
    ) * (anchors[1][1] - anchors[0][1]), "anchors must be non-collinear"
    _best_endpoint_assignment(segment_endpoints, anchors)

    grey_components = _components_from_mask(g == GREY)
    assert 2 <= len(grey_components) <= 3
    for comp in grey_components:
        r0, c0, r1, c1 = _bbox(comp)
        assert len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1), "grey components must be solid rectangles"
    grey_mask = g == GREY

    return _build_output(g.shape, grey_mask, anchors, segment_endpoints, sprite_offsets)


def _sprite_cells(anchor: Coord, offsets: tuple[Coord, ...]) -> set[Coord]:
    ar, ac = anchor
    return {(ar + dr, ac + dc) for dr, dc in offsets}


def _sprite_fits(anchor: Coord, offsets: tuple[Coord, ...], h: int, w: int) -> bool:
    return all(0 <= anchor[0] + dr < h and 0 <= anchor[1] + dc < w for dr, dc in offsets)


def _sample_anchors(rng: random.Random, h: int, w: int, offsets: tuple[Coord, ...]) -> list[Coord] | None:
    min_dr = min(dr for dr, _ in offsets)
    max_dr = max(dr for dr, _ in offsets)
    min_dc = min(dc for _, dc in offsets)
    max_dc = max(dc for _, dc in offsets)
    rows = range(-min_dr, h - max_dr)
    cols = range(-min_dc, w - max_dc)
    for _ in range(300):
        anchors: list[Coord] = []
        tries = 0
        while len(anchors) < 3 and tries < 300:
            tries += 1
            candidate = (rng.choice(tuple(rows)), rng.choice(tuple(cols)))
            if any(max(abs(candidate[0] - r), abs(candidate[1] - c)) < 5 for r, c in anchors):
                continue
            anchors.append(candidate)
        if len(anchors) < 3:
            continue
        if (anchors[1][0] - anchors[0][0]) * (anchors[2][1] - anchors[0][1]) == (
            anchors[2][0] - anchors[0][0]
        ) * (anchors[1][1] - anchors[0][1]):
            continue
        lines = [_bresenham(a, b) for a, b in combinations(anchors, 2)]
        if min(len(line) for line in lines) < 8:
            continue
        return anchors
    return None


def _place_rectangles(
    rng: random.Random,
    h: int,
    w: int,
    anchors: list[Coord],
    offsets: tuple[Coord, ...],
) -> list[tuple[int, int, int, int]] | None:
    sprite_forbidden: set[Coord] = set()
    for anchor in anchors:
        sprite_forbidden |= _sprite_cells(anchor, offsets)
    pair_lines = {pair: _bresenham(*pair) for pair in combinations(anchors, 2)}
    interior_points = [
        pt
        for line in pair_lines.values()
        for pt in line[2:-2]
        if pt not in sprite_forbidden
    ]
    if not interior_points:
        return None

    for _ in range(500):
        rects: list[tuple[int, int, int, int]] = []
        rect_mask = np.zeros((h, w), dtype=bool)
        for _rect_index in range(rng.choice(RECT_COUNT_CHOICES)):
            placed = False
            for _ in range(100):
                rh = rng.randint(*RECT_SIZE_RANGE)
                rw = rng.randint(*RECT_SIZE_RANGE)
                pr, pc = rng.choice(interior_points)
                r0_min = max(0, pr - rh + 1)
                r0_max = min(pr, h - rh)
                c0_min = max(0, pc - rw + 1)
                c0_max = min(pc, w - rw)
                if r0_min > r0_max or c0_min > c0_max:
                    continue
                r0 = rng.randint(r0_min, r0_max)
                c0 = rng.randint(c0_min, c0_max)
                r1 = r0 + rh - 1
                c1 = c0 + rw - 1
                rect_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
                if rect_cells & sprite_forbidden:
                    continue
                if any(rect_mask[r, c] for r, c in rect_cells):
                    continue
                if any(
                    0 <= nr < h and 0 <= nc < w and rect_mask[nr, nc]
                    for r, c in rect_cells
                    for nr, nc in _neighbors4(r, c)
                ):
                    continue
                for r, c in rect_cells:
                    rect_mask[r, c] = True
                rects.append((r0, c0, r1, c1))
                placed = True
                break
            if not placed:
                break
        if len(rects) not in RECT_COUNT_CHOICES:
            continue
        visible_counts = []
        for line in pair_lines.values():
            visible = sum(1 for r, c in line if rect_mask[r, c])
            visible_counts.append(visible)
        if not all(2 <= count < len(line) for count, line in zip(visible_counts, pair_lines.values())):
            continue
        if sum(visible_counts) < 10:
            continue
        return rects
    return None


def _visible_red_cells(anchors: list[Coord], rects: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> set[Coord]:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in rects:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    red_mask = _line_mask_from_anchors(shape, anchors, mask)
    return {tuple(cell) for cell in np.argwhere(red_mask)}


def _segment_cells(a: Coord, b: Coord) -> set[Coord]:
    if a[0] == b[0]:
        row = a[0]
        c0, c1 = sorted((a[1], b[1]))
        return {(row, c) for c in range(c0, c1 + 1)}
    col = a[1]
    r0, r1 = sorted((a[0], b[0]))
    return {(r, col) for r in range(r0, r1 + 1)}


def _pick_original_segment(
    rng: random.Random,
    h: int,
    w: int,
    anchors: list[Coord],
    offsets: tuple[Coord, ...],
    rects: list[tuple[int, int, int, int]],
) -> tuple[Coord, Coord] | None:
    grey_cells = {(r, c) for r0, c0, r1, c1 in rects for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    input_sprite_cells = set().union(*(_sprite_cells(anchor, offsets) for anchor in anchors))
    visible_output_red = _visible_red_cells(anchors, rects, (h, w))
    candidates: list[tuple[Coord, Coord]] = []
    orientations = ["h", "v"]
    rng.shuffle(orientations)
    for orientation in orientations:
        if orientation == "h":
            for length in range(SEGMENT_LEN_RANGE[0], min(SEGMENT_LEN_RANGE[1], w - 1) + 1):
                span = length - 1
                for r in range(h):
                    for c0 in range(w - span):
                        a = (r, c0)
                        b = (r, c0 + span)
                        cells = _segment_cells(a, b)
                        if cells & grey_cells or cells & input_sprite_cells:
                            continue
                        if not (_sprite_fits(a, offsets, h, w) and _sprite_fits(b, offsets, h, w)):
                            continue
                        footprint = _sprite_cells(a, offsets) | _sprite_cells(b, offsets)
                        if footprint & grey_cells or footprint & visible_output_red:
                            continue
                        if not _best_assignment_is_unique((a, b), anchors):
                            continue
                        candidates.append((a, b))
        else:
            for length in range(SEGMENT_LEN_RANGE[0], min(SEGMENT_LEN_RANGE[1], h - 1) + 1):
                span = length - 1
                for c in range(w):
                    for r0 in range(h - span):
                        a = (r0, c)
                        b = (r0 + span, c)
                        cells = _segment_cells(a, b)
                        if cells & grey_cells or cells & input_sprite_cells:
                            continue
                        if not (_sprite_fits(a, offsets, h, w) and _sprite_fits(b, offsets, h, w)):
                            continue
                        footprint = _sprite_cells(a, offsets) | _sprite_cells(b, offsets)
                        if footprint & grey_cells or footprint & visible_output_red:
                            continue
                        if not _best_assignment_is_unique((a, b), anchors):
                            continue
                        candidates.append((a, b))
    if not candidates:
        return None
    return rng.choice(candidates)


def _best_assignment_is_unique(endpoints: tuple[Coord, Coord], anchors: list[Coord]) -> bool:
    try:
        _best_endpoint_assignment(endpoints, anchors)
        return True
    except AssertionError:
        return False


def _make_input_grid(
    h: int,
    w: int,
    anchors: list[Coord],
    offsets: tuple[Coord, ...],
    rects: list[tuple[int, int, int, int]],
    segment_endpoints: tuple[Coord, Coord],
) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for r0, c0, r1, c1 in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    for anchor in anchors:
        _stamp_sprite(grid, anchor, offsets)
    for r, c in _segment_cells(*segment_endpoints):
        grid[r, c] = RED
    return grid


def _oracle_checks(
    inp: np.ndarray,
    out: np.ndarray,
    anchors: list[Coord],
    offsets: tuple[Coord, ...],
    rects: list[tuple[int, int, int, int]],
    segment_endpoints: tuple[Coord, Coord],
) -> bool:
    solved = main(inp)
    if not np.array_equal(solved, out):
        return False

    grey_mask = inp == GREY
    full_line_out = np.full(inp.shape, BLACK, dtype=int)
    full_line_out[grey_mask] = GREY
    for a, b in combinations(anchors, 2):
        for r, c in _bresenham(a, b):
            full_line_out[r, c] = RED
    _stamp_sprite(full_line_out, segment_endpoints[0], offsets)
    _stamp_sprite(full_line_out, segment_endpoints[1], offsets)
    if np.array_equal(full_line_out, out):
        return False

    sprite_only_out = np.full(inp.shape, BLACK, dtype=int)
    sprite_only_out[grey_mask] = GREY
    _stamp_sprite(sprite_only_out, segment_endpoints[0], offsets)
    _stamp_sprite(sprite_only_out, segment_endpoints[1], offsets)
    if np.array_equal(sprite_only_out, out):
        return False

    masked_original = np.full(inp.shape, BLACK, dtype=int)
    masked_original[grey_mask] = GREY
    for r, c in _segment_cells(*segment_endpoints):
        if grey_mask[r, c]:
            masked_original[r, c] = RED
    _stamp_sprite(masked_original, segment_endpoints[0], offsets)
    _stamp_sprite(masked_original, segment_endpoints[1], offsets)
    if np.array_equal(masked_original, out):
        return False

    for a, b in combinations(anchors, 2):
        line = _bresenham(a, b)
        visible = sum(1 for r, c in line if grey_mask[r, c])
        if not (2 <= visible < len(line)):
            return False

    grey_components = _components_from_mask(grey_mask)
    if len(grey_components) != len(rects):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    offsets = tuple(sorted(rng.choice(SPRITE_SHAPES)))
    anchors = _sample_anchors(rng, h, w, offsets)
    if anchors is None:
        return None
    rects = _place_rectangles(rng, h, w, anchors, offsets)
    if rects is None:
        return None
    segment_endpoints = _pick_original_segment(rng, h, w, anchors, offsets, rects)
    if segment_endpoints is None:
        return None

    inp = _make_input_grid(h, w, anchors, offsets, rects, segment_endpoints)
    out = _build_output(inp.shape, inp == GREY, anchors, segment_endpoints, offsets)
    if not _oracle_checks(inp, out, anchors, offsets, rects, segment_endpoints):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATE_ATTEMPTS} attempts")
