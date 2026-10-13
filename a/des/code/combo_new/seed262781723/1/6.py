import itertools
import random
from collections import deque

import numpy as np


BG_COLOR = 0
PAIR_COLORS = tuple(range(1, 10))
RELATION_KINDS = ("vertical_axis", "horizontal_axis", "rotation180")
GENERATION_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 200
GRID_SIDE_RANGE = (18, 30)
PAIR_COUNT_RANGE = (2, 4)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _connected(cells) -> bool:
    cells = set(cells)
    if not cells:
        return False
    seen = set()
    queue = deque([next(iter(cells))])
    seen.add(queue[0])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return seen == cells


def _transform_coords(coords, kind: str, height: int, width: int):
    if kind == "identity":
        return {(int(r), int(c)) for r, c in coords}
    if kind == "vertical_axis":
        return {(int(r), width - 1 - int(c)) for r, c in coords}
    if kind == "horizontal_axis":
        return {(height - 1 - int(r), int(c)) for r, c in coords}
    if kind == "rotation180":
        return {(height - 1 - int(r), width - 1 - int(c)) for r, c in coords}
    raise ValueError(f"unknown transform: {kind}")


def _build_valid_base_shapes():
    cells = [(r, c) for r in range(3) for c in range(3)]
    shapes = []
    for size in range(3, 6):
        for subset in itertools.combinations(cells, size):
            subset = set(subset)
            rows = {r for r, _c in subset}
            cols = {c for _r, c in subset}
            if rows != {0, 1, 2} or cols != {0, 1, 2}:
                continue
            if not _connected(subset):
                continue
            if any(_transform_coords(subset, kind, 3, 3) == subset for kind in RELATION_KINDS):
                continue
            shapes.append(tuple(sorted(subset)))
    assert shapes
    return tuple(shapes)


VALID_BASE_SHAPES = _build_valid_base_shapes()


def _find_components_of_color(grid: np.ndarray, color: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _component_signature(coords):
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    rel = {(r - r0, c - c0) for r, c in coords}
    return rel, r1 - r0 + 1, c1 - c0 + 1, (r0, c0, r1, c1)


def _midpoint_cell(bbox_a, bbox_b):
    r_num = int(bbox_a[0] + bbox_a[2] + bbox_b[0] + bbox_b[2])
    c_num = int(bbox_a[1] + bbox_a[3] + bbox_b[1] + bbox_b[3])
    assert r_num % 4 == 0 and c_num % 4 == 0
    return r_num // 4, c_num // 4


def _marker_cells(kind: str, center_r: int, center_c: int):
    if kind == "vertical_axis":
        return {(center_r - 1, center_c), (center_r, center_c), (center_r + 1, center_c)}
    if kind == "horizontal_axis":
        return {(center_r, center_c - 1), (center_r, center_c), (center_r, center_c + 1)}
    if kind == "rotation180":
        return {
            (center_r, center_c),
            (center_r - 1, center_c),
            (center_r + 1, center_c),
            (center_r, center_c - 1),
            (center_r, center_c + 1),
        }
    raise ValueError(f"unknown marker kind: {kind}")


def _shape_to_global(shape, center_r: int, center_c: int):
    return {(center_r + r - 1, center_c + c - 1) for r, c in shape}


def _reserve_bbox(shape, h: int, w: int, pad: int = 1):
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    r0 = max(0, min(rows) - pad)
    r1 = min(h - 1, max(rows) + pad)
    c0 = max(0, min(cols) - pad)
    c1 = min(w - 1, max(cols) + pad)
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _paint(grid: np.ndarray, cells, color: int):
    for r, c in cells:
        grid[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Find each same-colored pair of motifs and draw a midpoint symbol that encodes their symmetry type.

    Input structure:
    - The background is a dominant black-like color, with several distinct foreground colors.
    - Each foreground color appears in exactly two disjoint 4-connected motifs of size 3 to 5.
    - Within each pair, one motif is the other motif under either a left-right mirror, an up-down mirror, or a 180-degree rotation.
    - The space around each pair's midpoint is empty so a small marker can be added there.

    Rule:
    - For each foreground color, extract its two motifs and compare their normalized shapes.
    - If the pair matches by left-right mirroring, draw a 3-cell vertical bar of that color at the midpoint between the motifs' geometric centers.
    - If the pair matches by up-down mirroring, draw a 3-cell horizontal bar there.
    - If the pair matches by 180-degree rotation, draw a 5-cell plus sign centered there.
    - Keep the original motifs unchanged.

    Key insight:
    - The task is two-stage: first pair objects by color, then determine which symmetry relation makes the two shapes coincide, because the inserted marker represents the relation itself rather than the objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _background_color(g)
    out = g.copy()

    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors
    for color in colors:
        components = _find_components_of_color(g, color)
        assert len(components) == 2
        assert all(3 <= len(component) <= 5 for component in components)

        sig_a, height_a, width_a, bbox_a = _component_signature(components[0])
        sig_b, height_b, width_b, bbox_b = _component_signature(components[1])
        assert height_a == height_b and width_a == width_b

        matches = [
            kind
            for kind in RELATION_KINDS
            if _transform_coords(sig_a, kind, height_a, width_a) == sig_b
        ]
        assert len(matches) == 1
        kind = matches[0]

        center_r, center_c = _midpoint_cell(bbox_a, bbox_b)
        marker = _marker_cells(kind, center_r, center_c)
        for r, c in marker:
            assert 0 <= r < h and 0 <= c < w
            assert out[r, c] in (bg, color)
            out[r, c] = color

    return out


def _sample_relation_plan(pair_count: int):
    if pair_count == 2:
        plan = random.sample(list(RELATION_KINDS), 2)
    elif pair_count == 3:
        plan = list(RELATION_KINDS)
    else:
        assert pair_count == 4
        plan = list(RELATION_KINDS) + [random.choice(RELATION_KINDS)]
    random.shuffle(plan)
    return plan


def _sample_shape_pair(kind: str):
    base = set(random.choice(VALID_BASE_SHAPES))
    base = _transform_coords(base, random.choice(("identity",) + RELATION_KINDS), 3, 3)
    other = _transform_coords(base, kind, 3, 3)
    if random.random() < 0.5:
        base, other = other, base
    return base, other


def _vertical_layout(h: int, w: int):
    dist_choices = [dist for dist in range(3, 7) if 2 * dist <= w - 3]
    if not dist_choices or h < 5:
        return None
    dist = random.choice(dist_choices)
    center_r = random.randint(1, h - 2)
    center_c = random.randint(dist + 1, w - dist - 2)
    return (center_r, center_c - dist), (center_r, center_c + dist), (center_r, center_c)


def _horizontal_layout(h: int, w: int):
    dist_choices = [dist for dist in range(3, 7) if 2 * dist <= h - 3]
    if not dist_choices or w < 5:
        return None
    dist = random.choice(dist_choices)
    center_r = random.randint(dist + 1, h - dist - 2)
    center_c = random.randint(1, w - 2)
    return (center_r - dist, center_c), (center_r + dist, center_c), (center_r, center_c)


def _rotation_layout(h: int, w: int):
    row_choices = [dist for dist in range(3, 6) if 2 * dist <= h - 3]
    col_choices = [dist for dist in range(3, 6) if 2 * dist <= w - 3]
    if not row_choices or not col_choices:
        return None
    dr = random.choice(row_choices) * random.choice((-1, 1))
    dc = random.choice(col_choices) * random.choice((-1, 1))
    center_r = random.randint(abs(dr) + 1, h - abs(dr) - 2)
    center_c = random.randint(abs(dc) + 1, w - abs(dc) - 2)
    return (center_r - dr, center_c - dc), (center_r + dr, center_c + dc), (center_r, center_c)


def _sample_layout(kind: str, h: int, w: int):
    if kind == "vertical_axis":
        return _vertical_layout(h, w)
    if kind == "horizontal_axis":
        return _horizontal_layout(h, w)
    if kind == "rotation180":
        return _rotation_layout(h, w)
    raise ValueError(f"unknown relation kind: {kind}")


def _gen_attempt():
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    grid = np.full((height, width), BG_COLOR, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    colors = random.sample(PAIR_COLORS, pair_count)
    relations = _sample_relation_plan(pair_count)
    pair_specs = []

    for color, relation in zip(colors, relations):
        placed = False
        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            shape_a, shape_b = _sample_shape_pair(relation)
            layout = _sample_layout(relation, height, width)
            if layout is None:
                return None
            center_a, center_b, marker_center = layout
            cells_a = _shape_to_global(shape_a, *center_a)
            cells_b = _shape_to_global(shape_b, *center_b)
            marker = _marker_cells(relation, *marker_center)
            all_cells = cells_a | cells_b | marker

            if not all(0 <= r < height and 0 <= c < width for r, c in all_cells):
                continue
            if cells_a & cells_b or cells_a & marker or cells_b & marker:
                continue

            reserve = _reserve_bbox(all_cells, height, width, pad=1)
            if any(reserved[r, c] for r, c in reserve):
                continue

            _paint(grid, cells_a | cells_b, color)
            for r, c in reserve:
                reserved[r, c] = True
            pair_specs.append(
                {
                    "color": color,
                    "relation": relation,
                    "marker": marker,
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = grid.copy()
    for spec in pair_specs:
        assert all(expected[r, c] == BG_COLOR for r, c in spec["marker"])
        _paint(expected, spec["marker"], spec["color"])

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-marker ARC input")
