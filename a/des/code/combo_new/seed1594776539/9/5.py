from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG_COLOR = 4
DIVIDER_COLOR = 5
OBJECT_COLORS = [0, 1, 2, 3, 6, 7, 8, 9]
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)
ODD_SIZES = [15, 17, 19, 21, 23, 25, 27, 29]
ARROW_DIRECTIONS = ["up", "down", "left", "right"]
OBJECT_COUNT_CHOICES = [1, 2, 3, 4]
OBJECT_SIZE_RANGE = (3, 6)
SHAPE_ATTEMPTS = 120
PAIR_ATTEMPTS = 240
GENERATE_ATTEMPTS = 2000


Coord = tuple[int, int]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _component_list(grid: np.ndarray, color: int) -> list[list[Coord]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[Coord] | set[Coord]) -> frozenset[Coord]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return frozenset((r - r0, c - c0) for r, c in coords)


def _local_dims(shape: frozenset[Coord]) -> tuple[int, int]:
    return 1 + max(r for r, _ in shape), 1 + max(c for _, c in shape)


def _flip_local(shape: frozenset[Coord], mode: str) -> frozenset[Coord]:
    h, w = _local_dims(shape)
    if mode == "horizontal":
        return frozenset((r, w - 1 - c) for r, c in shape)
    if mode == "vertical":
        return frozenset((h - 1 - r, c) for r, c in shape)
    raise AssertionError(f"unknown flip mode: {mode}")


def _find_divider(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    h, w = grid.shape
    candidates: list[tuple[str, int, int]] = []
    for r in range(h):
        row_color = int(grid[r, 0])
        if row_color != bg and np.all(grid[r, :] == row_color):
            candidates.append(("horizontal", r, row_color))
    for c in range(w):
        col_color = int(grid[0, c])
        if col_color != bg and np.all(grid[:, c] == col_color):
            candidates.append(("vertical", c, col_color))
    assert len(candidates) == 1, "expected one full grey divider"
    return candidates[0]


def _line_arrow(coords: list[Coord]) -> list[Coord] | None:
    if len(coords) != 3:
        return None
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    if len(rows) == 1 and cols == list(range(cols[0], cols[0] + 3)):
        return sorted(coords, key=lambda rc: rc[1])
    if len(cols) == 1 and rows == list(range(rows[0], rows[0] + 3)):
        return sorted(coords, key=lambda rc: rc[0])
    return None


def _endpoint_score(endpoint: Coord, obj_coords: list[Coord]) -> tuple[int, float]:
    min_manhattan = min(abs(endpoint[0] - r) + abs(endpoint[1] - c) for r, c in obj_coords)
    center_r = sum(r for r, _ in obj_coords) / len(obj_coords)
    center_c = sum(c for _, c in obj_coords) / len(obj_coords)
    center_dist = (endpoint[0] - center_r) ** 2 + (endpoint[1] - center_c) ** 2
    return min_manhattan, center_dist


def _arrow_direction(arrow_cells: list[Coord], obj_coords: list[Coord]) -> str:
    ordered = _line_arrow(arrow_cells)
    assert ordered is not None, "arrow must be a contiguous 3-cell line"
    first, middle, last = ordered
    first_score = _endpoint_score(first, obj_coords)
    last_score = _endpoint_score(last, obj_coords)
    assert first_score != last_score, "arrow head must be distinguishable from the object position"
    head = first if first_score > last_score else last
    dr = head[0] - middle[0]
    dc = head[1] - middle[1]
    mapping = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}
    assert (dr, dc) in mapping
    return mapping[(dr, dc)]


def _mirror_bbox(
    bbox: tuple[int, int, int, int],
    divider_orientation: str,
    divider_index: int,
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if divider_orientation == "vertical":
        return r0, 2 * divider_index - c1
    assert divider_orientation == "horizontal"
    return 2 * divider_index - r1, c0


def _copy_coords(
    obj_coords: list[Coord],
    divider_orientation: str,
    divider_index: int,
    direction: str,
) -> list[Coord]:
    r0, c0, r1, c1 = _bbox(obj_coords)
    top, left = _mirror_bbox((r0, c0, r1, c1), divider_orientation, divider_index)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    out: list[Coord] = []
    for r, c in obj_coords:
        lr = r - r0
        lc = c - c0
        if direction in {"left", "right"}:
            nr = top + lr
            nc = left + (w - 1 - lc)
        else:
            nr = top + (h - 1 - lr)
            nc = left + lc
        out.append((nr, nc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each colored object to the mirrored side of the grey divider, flipping the copy according to its same-color arrow.

    Input structure:
    - The background is yellow.
    - One full grey horizontal or vertical line runs through the grid center and splits the canvas into two halves.
    - Each non-grey foreground color appears in exactly two disconnected components: one small irregular object and one 3-pixel arrow mark of the same color.
    - Every arrow is a straight 3-pixel horizontal or vertical line, and all objects and arrows lie on the same side of the grey divider.

    Rule:
    - Pair each arrow with the object of the same color.
    - Read the arrow direction from the endpoint farther from its object; left/right arrows request a left-right flip, while up/down arrows request an up-down flip.
    - Reflect the object's bounding box position across the grey divider, then draw the flipped object inside that mirrored bounding box using the original color.
    - Keep the original objects, arrows, and grey divider unchanged.

    Key insight:
    - The arrow decides only how to flip the shape, while the grey divider decides only where the new copy is placed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    bg = _dominant_color(g)
    divider_orientation, divider_index, divider_color = _find_divider(g, bg)

    out = g.copy()
    colors = sorted(int(color) for color in np.unique(g) if color not in {bg, divider_color})
    for color in colors:
        components = _component_list(g, color)
        assert len(components) == 2, "each color must have exactly one object and one arrow"
        arrow_candidates = [comp for comp in components if _line_arrow(comp) is not None]
        assert len(arrow_candidates) == 1, "exactly one component per color must be a 3-cell arrow"
        arrow = arrow_candidates[0]
        obj = components[0] if components[1] is arrow else components[1]
        direction = _arrow_direction(arrow, obj)
        for r, c in _copy_coords(obj, divider_orientation, divider_index, direction):
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1]
            assert int(out[r, c]) in {bg, color}, "copied object must land in empty mirrored space"
            out[r, c] = color

    return out


def _is_connected(shape: frozenset[Coord]) -> bool:
    start = next(iter(shape))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in shape and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(shape)


def _random_object_shape() -> frozenset[Coord] | None:
    for _ in range(SHAPE_ATTEMPTS):
        target = random.randint(*OBJECT_SIZE_RANGE)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = {
                (r + dr, c + dc)
                for r, c in cells
                for dr, dc in DIRS4
                if (r + dr, c + dc) not in cells
            }
            cells.add(random.choice(tuple(frontier)))
        shape = _normalize(cells)
        h, w = _local_dims(shape)
        if h == 1 or w == 1 or h > 4 or w > 4:
            continue
        if len(shape) == h * w:
            continue
        if shape == _flip_local(shape, "horizontal") or shape == _flip_local(shape, "vertical"):
            continue
        if not _is_connected(shape):
            continue
        return shape
    return None


def _region_bounds(
    shape: tuple[int, int],
    divider_orientation: str,
    side: str,
) -> tuple[int, int, int, int, int]:
    h, w = shape
    if divider_orientation == "vertical":
        divider_index = w // 2
        row_lo, row_hi = 1, h - 2
        if side == "left":
            col_lo, col_hi = 1, divider_index - 2
        else:
            col_lo, col_hi = divider_index + 2, w - 2
    else:
        divider_index = h // 2
        col_lo, col_hi = 1, w - 2
        if side == "top":
            row_lo, row_hi = 1, divider_index - 2
        else:
            row_lo, row_hi = divider_index + 2, h - 2
    return row_lo, row_hi, col_lo, col_hi, divider_index


def _expanded(cells: set[Coord], h: int, w: int) -> set[Coord]:
    expanded = set(cells)
    for r, c in cells:
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                expanded.add((nr, nc))
    return expanded


def _shape_cells(shape: frozenset[Coord], top: int, left: int) -> list[Coord]:
    return [(top + r, left + c) for r, c in shape]


def _arrow_cells_for_object(
    bbox: tuple[int, int, int, int],
    direction: str,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
) -> list[Coord] | None:
    r0, c0, r1, c1 = bbox
    gaps = [1, 2, 3]
    random.shuffle(gaps)
    if direction in {"left", "right"}:
        rows = list(range(r0, r1 + 1))
        random.shuffle(rows)
        for gap in gaps:
            for row in rows:
                if direction == "right":
                    start = c1 + gap + 1
                    cells = [(row, start + dc) for dc in range(3)]
                    if cells[-1][1] > col_hi:
                        continue
                else:
                    start = c0 - gap - 3
                    cells = [(row, start + dc) for dc in range(3)]
                    if cells[0][1] < col_lo:
                        continue
                return cells
    else:
        cols = list(range(c0, c1 + 1))
        random.shuffle(cols)
        for gap in gaps:
            for col in cols:
                if direction == "down":
                    start = r1 + gap + 1
                    cells = [(start + dr, col) for dr in range(3)]
                    if cells[-1][0] > row_hi:
                        continue
                else:
                    start = r0 - gap - 3
                    cells = [(start + dr, col) for dr in range(3)]
                    if cells[0][0] < row_lo:
                        continue
                return cells
    return None


def _copy_expected(
    grid: np.ndarray,
    divider_orientation: str,
    divider_index: int,
    objects: list[tuple[int, list[Coord], str]],
) -> np.ndarray | None:
    out = grid.copy()
    for color, coords, direction in objects:
        for r, c in _copy_coords(coords, divider_orientation, divider_index, direction):
            if int(out[r, c]) not in {BG_COLOR, color}:
                return None
            out[r, c] = color
    return out


def _pair_fits(
    obj_cells: list[Coord],
    arrow_cells: list[Coord],
    blocked: set[Coord],
    h: int,
    w: int,
) -> bool:
    obj_set = set(obj_cells)
    arrow_set = set(arrow_cells)
    if obj_set & arrow_set:
        return False
    if any(cell in blocked for cell in obj_set | arrow_set):
        return False
    if _expanded(obj_set, h, w) & arrow_set:
        return False
    return True


def _try_place_pair(
    shape: frozenset[Coord],
    direction: str,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    blocked: set[Coord],
    grid_shape: tuple[int, int],
) -> tuple[list[Coord], list[Coord]] | None:
    h, w = _local_dims(shape)
    top_candidates = list(range(row_lo, row_hi - h + 2))
    left_candidates = list(range(col_lo, col_hi - w + 2))
    random.shuffle(top_candidates)
    random.shuffle(left_candidates)
    for top in top_candidates[: min(len(top_candidates), 18)]:
        for left in left_candidates[: min(len(left_candidates), 18)]:
            obj_cells = _shape_cells(shape, top, left)
            arrow_cells = _arrow_cells_for_object((top, left, top + h - 1, left + w - 1), direction, row_lo, row_hi, col_lo, col_hi)
            if arrow_cells is None:
                continue
            if _pair_fits(obj_cells, arrow_cells, blocked, *grid_shape):
                return obj_cells, arrow_cells
    return None


def _generate_scene() -> np.ndarray | None:
    h = random.choice(ODD_SIZES)
    w = random.choice(ODD_SIZES)
    divider_orientation = random.choice(["horizontal", "vertical"])
    if divider_orientation == "vertical":
        side = random.choice(["left", "right"])
    else:
        side = random.choice(["top", "bottom"])
    row_lo, row_hi, col_lo, col_hi, divider_index = _region_bounds((h, w), divider_orientation, side)
    if row_hi - row_lo < 3 or col_hi - col_lo < 3:
        return None

    grid = np.full((h, w), BG_COLOR, dtype=int)
    if divider_orientation == "vertical":
        grid[:, divider_index] = DIVIDER_COLOR
    else:
        grid[divider_index, :] = DIVIDER_COLOR

    object_count = random.choice(OBJECT_COUNT_CHOICES)
    colors = random.sample(OBJECT_COLORS, object_count)
    occupied: set[Coord] = set()
    blocked: set[Coord] = set()
    placed_objects: list[tuple[int, list[Coord], str]] = []

    for color in colors:
        pair = None
        for _ in range(PAIR_ATTEMPTS):
            shape = _random_object_shape()
            if shape is None:
                continue
            direction = random.choice(ARROW_DIRECTIONS)
            pair = _try_place_pair(shape, direction, row_lo, row_hi, col_lo, col_hi, blocked, grid.shape)
            if pair is None:
                continue
            obj_cells, arrow_cells = pair
            for r, c in obj_cells:
                grid[r, c] = color
            for r, c in arrow_cells:
                grid[r, c] = color
            pair_cells = set(obj_cells) | set(arrow_cells)
            occupied |= pair_cells
            blocked |= _expanded(pair_cells, h, w)
            placed_objects.append((color, obj_cells, direction))
            break
        if pair is None:
            return None

    expected = _copy_expected(grid, divider_orientation, divider_index, placed_objects)
    if expected is None:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    added = int(np.count_nonzero(actual != grid))
    if added != sum(len(coords) for _color, coords, _direction in placed_objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _generate_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
