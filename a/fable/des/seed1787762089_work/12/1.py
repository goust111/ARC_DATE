from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

LEGEND_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA)
DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 400
LEGEND_PLACEMENT_ATTEMPTS = 250
OBJECT_PLACEMENT_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        comp: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _normalize(coords) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _bbox(coords) -> tuple[int, int, int, int]:
    pts = [(int(r), int(c)) for r, c in coords]
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _rotate_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h, _ = _shape_dims(shape)
    return _normalize((c, h - 1 - r) for r, c in shape)


def _flip_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    _, w = _shape_dims(shape)
    return _normalize((r, w - 1 - c) for r, c in shape)


def _allowed_directions(shape: tuple[tuple[int, int], ...]) -> tuple[str, ...]:
    occ = set(shape)
    h, w = _shape_dims(shape)
    center_r = (h - 1) // 2
    center_c = (w - 1) // 2
    allowed = []
    if (0, center_c) in occ:
        allowed.append("up")
    if (h - 1, center_c) in occ:
        allowed.append("down")
    if (center_r, 0) in occ:
        allowed.append("left")
    if (center_r, w - 1) in occ:
        allowed.append("right")
    return tuple(allowed)


def _marker_local(shape: tuple[tuple[int, int], ...], direction: str) -> tuple[int, int]:
    h, w = _shape_dims(shape)
    center_r = (h - 1) // 2
    center_c = (w - 1) // 2
    if direction == "up":
        return -1, center_c
    if direction == "down":
        return h, center_c
    if direction == "left":
        return center_r, -1
    if direction == "right":
        return center_r, w
    raise ValueError(direction)


def _expanded(cells, h: int, w: int, min_col: int, max_col: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and min_col <= cc <= max_col:
                    out.add((rr, cc))
    return out


def _parse_legend(grid: np.ndarray, sep_col: int) -> dict[tuple[tuple[int, int], ...], tuple[int, str]]:
    left = grid[:, :sep_col]
    legend_mask = (left != BLACK) & (left != ORANGE)
    comps = _components(legend_mask)
    assert 3 <= len(comps) <= 4, "expected 3-4 legend shapes"

    orange_positions = {tuple(map(int, rc)) for rc in np.argwhere(left == ORANGE)}
    mapping: dict[tuple[tuple[int, int], ...], tuple[int, str]] = {}
    used_colors: set[int] = set()
    for comp in comps:
        color_values = {int(left[r, c]) for r, c in comp}
        assert len(color_values) == 1, "legend shape must be monochrome"
        color = next(iter(color_values))
        assert color in LEGEND_COLORS, "unexpected legend color"
        assert color not in used_colors, "legend colors must be unique"
        used_colors.add(color)

        r0, c0, r1, c1 = _bbox(comp)
        candidate_markers = {
            "up": (r0 - 1, (c0 + c1) // 2),
            "down": (r1 + 1, (c0 + c1) // 2),
            "left": ((r0 + r1) // 2, c0 - 1),
            "right": ((r0 + r1) // 2, c1 + 1),
        }
        matches = [name for name, pos in candidate_markers.items() if pos in orange_positions]
        assert len(matches) == 1, "each legend shape must have exactly one adjacent orange marker"
        direction = matches[0]

        key = _normalize(comp)
        assert key not in mapping, "legend shapes must be distinct after normalization"
        mapping[key] = (color, direction)

    assert len(mapping) == len(comps)
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(sep_cols) == 1, "expected one full-height grey separator"
    sep_col = sep_cols[0]
    assert 0 < sep_col < g.shape[1] - 1, "separator must split legend and canvas"

    mapping = _parse_legend(g, sep_col)

    canvas_mask = (g == GREY).copy()
    canvas_mask[:, sep_col] = False
    comps = _components(canvas_mask)
    assert 3 <= len(comps) <= 5, "expected 3-5 grey placeholder objects"

    out = g.copy()
    out[canvas_mask] = BLACK

    for comp in comps:
        shape = _normalize(comp)
        assert shape in mapping, "canvas object must match exactly one legend shape"
        color, direction = mapping[shape]
        dr, dc = DIR_TO_DELTA[direction]
        steps = len(comp)
        moved = [(r + dr * steps, c + dc * steps) for r, c in comp]
        for r, c in moved:
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1], "moved object left the grid"
            assert c > sep_col, "moved object must remain in the canvas zone"
            assert out[r, c] == BLACK, "moved objects must not overlap"
        for r, c in moved:
            out[r, c] = color

    return out


_BASE_SHAPES = [
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (0, 2), (1, 0), (1, 1), (2, 0)),
]


def _build_shape_pool() -> list[tuple[tuple[int, int], ...]]:
    pool = set()
    for base in _BASE_SHAPES:
        shape = _normalize(base)
        for flipped in (shape, _flip_shape(shape)):
            cur = flipped
            for _ in range(4):
                pool.add(cur)
                cur = _rotate_shape(cur)
    return sorted(shape for shape in pool if _allowed_directions(shape))


SHAPE_POOL = _build_shape_pool()


def _sample_shapes(count: int) -> list[tuple[tuple[int, int], ...]]:
    target_dirs = min(count, 4)
    for _ in range(80):
        shapes = random.sample(SHAPE_POOL, count)
        union = set()
        for shape in shapes:
            union.update(_allowed_directions(shape))
        if len(union) >= target_dirs:
            return shapes
    return random.sample(SHAPE_POOL, count)


def _place_legends(
    grid: np.ndarray,
    sep_col: int,
    shapes: list[tuple[tuple[int, int], ...]],
    colors: list[int],
) -> list[dict] | None:
    h, _ = grid.shape
    reserved: set[tuple[int, int]] = set()
    used_dirs: set[str] = set()
    legends = []

    for shape, color in zip(shapes, colors):
        allowed = _allowed_directions(shape)
        placed = None
        for _ in range(LEGEND_PLACEMENT_ATTEMPTS):
            preferred = [d for d in allowed if d not in used_dirs]
            direction_choices = preferred if preferred and random.random() < 0.8 else list(allowed)
            direction = random.choice(direction_choices)
            marker = _marker_local(shape, direction)
            item_cells = list(shape) + [marker]
            min_r = min(r for r, _ in item_cells)
            max_r = max(r for r, _ in item_cells)
            min_c = min(c for _, c in item_cells)
            max_c = max(c for _, c in item_cells)
            row_lo = -min_r
            row_hi = h - max_r
            col_lo = -min_c
            col_hi = sep_col - max_c
            if row_lo >= row_hi or col_lo >= col_hi:
                continue
            base_r = random.randrange(row_lo, row_hi)
            base_c = random.randrange(col_lo, col_hi)
            shape_cells = {(base_r + r, base_c + c) for r, c in shape}
            marker_cell = (base_r + marker[0], base_c + marker[1])
            all_cells = shape_cells | {marker_cell}
            if all(cell not in reserved for cell in all_cells):
                placed = (shape_cells, marker_cell, direction)
                break
        if placed is None:
            return None

        shape_cells, marker_cell, direction = placed
        for r, c in shape_cells:
            grid[r, c] = color
        grid[marker_cell] = ORANGE
        reserved |= _expanded(shape_cells | {marker_cell}, h, sep_col, 0, sep_col - 1)
        legends.append({"shape": shape, "color": color, "direction": direction})
        used_dirs.add(direction)

    return legends


def _place_objects(
    grid: np.ndarray,
    sep_col: int,
    assignments: list[dict],
) -> bool:
    h, w = grid.shape
    min_canvas_col = sep_col + 1
    init_reserved: set[tuple[int, int]] = set()
    final_reserved: set[tuple[int, int]] = set()

    for legend in assignments:
        shape = legend["shape"]
        direction = legend["direction"]
        dr, dc = DIR_TO_DELTA[direction]
        steps = len(shape)
        placed = None
        for _ in range(OBJECT_PLACEMENT_ATTEMPTS):
            shape_h, shape_w = _shape_dims(shape)
            base_r = random.randrange(0, h - shape_h + 1)
            base_c = random.randrange(min_canvas_col, w - shape_w + 1)
            init_cells = {(base_r + r, base_c + c) for r, c in shape}
            final_cells = {(r + dr * steps, c + dc * steps) for r, c in init_cells}
            if any(not (0 <= r < h and min_canvas_col <= c < w) for r, c in final_cells):
                continue
            if any(cell in init_reserved for cell in init_cells):
                continue
            if any(cell in final_reserved for cell in final_cells):
                continue
            placed = init_cells, final_cells
            break
        if placed is None:
            return False

        init_cells, final_cells = placed
        for r, c in init_cells:
            grid[r, c] = GREY
        init_reserved |= _expanded(init_cells, h, w, min_canvas_col, w - 1)
        final_reserved |= _expanded(final_cells, h, w, min_canvas_col, w - 1)

    return True


def _gen_attempt() -> np.ndarray | None:
    num_legends = random.choice([3, 4])
    num_objects = random.randint(num_legends, 5)

    height = random.randint(14, 19)
    legend_width = random.randint(8, 10)
    canvas_width = random.randint(12, 17)
    width = legend_width + 1 + canvas_width
    sep_col = legend_width

    grid = np.full((height, width), BLACK, dtype=int)
    grid[:, sep_col] = GREY

    shapes = _sample_shapes(num_legends)
    colors = random.sample(list(LEGEND_COLORS), num_legends)
    legends = _place_legends(grid, sep_col, shapes, colors)
    if legends is None:
        return None

    assignments = legends.copy()
    while len(assignments) < num_objects:
        assignments.append(random.choice(legends))
    random.shuffle(assignments)
    if not _place_objects(grid, sep_col, assignments):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(grid == ORANGE) != num_legends:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
