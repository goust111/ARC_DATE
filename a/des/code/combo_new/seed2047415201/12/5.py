import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
GREY = 5
RECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 800
PAIR_PLACEMENT_ATTEMPTS = 200


Coord = tuple[int, int]


@dataclass(frozen=True)
class Rect:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def center_r2(self) -> int:
        return self.r0 + self.r1

    @property
    def center_c2(self) -> int:
        return self.c0 + self.c1

    @property
    def center_r(self) -> int:
        assert self.center_r2 % 2 == 0, "rectangle center row must fall on a cell"
        return self.center_r2 // 2

    @property
    def center_c(self) -> int:
        assert self.center_c2 % 2 == 0, "rectangle center column must fall on a cell"
        return self.center_c2 // 2

    def cells(self) -> set[Coord]:
        return {
            (r, c)
            for r in range(self.r0, self.r1 + 1)
            for c in range(self.c0, self.c1 + 1)
        }


def _foreground_colors(grid: np.ndarray) -> list[int]:
    colors: list[int] = []
    seen = {BLACK}
    for value in grid.reshape(-1):
        color = int(value)
        if color not in seen:
            seen.add(color)
            colors.append(color)
    return colors


def _components_of_color(grid: np.ndarray, color: int) -> list[Rect]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[Rect] = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(grid[sr, sc]) != color:
                continue

            seen[sr, sc] = True
            stack = [(sr, sc)]
            coords: list[Coord] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            rect = Rect(int(color), int(r0), int(c0), int(r1), int(c1))
            assert len(coords) == rect.height * rect.width, "object must be a solid rectangle"
            assert bool(np.all(grid[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] == color))
            components.append(rect)

    return components


def _connection_plan(a: Rect, b: Rect) -> tuple[list[Coord], list[Coord]]:
    if a.center_r2 == b.center_r2:
        left, right = (a, b) if a.center_c2 < b.center_c2 else (b, a)
        assert left.c1 < right.c0, "same-row rectangles must face each other"
        row = left.center_r
        line = [(row, c) for c in range(left.c1 + 1, right.c0)]
        assert line, "horizontal connection must contain at least one background cell"
        return line, []

    if a.center_c2 == b.center_c2:
        top, bottom = (a, b) if a.center_r2 < b.center_r2 else (b, a)
        assert top.r1 < bottom.r0, "same-column rectangles must face each other"
        col = top.center_c
        line = [(r, col) for r in range(top.r1 + 1, bottom.r0)]
        assert line, "vertical connection must contain at least one background cell"
        return line, []

    left, right = (a, b) if a.center_c2 < b.center_c2 else (b, a)
    assert left.c1 < right.c0, "diagonal rectangles must be separated left-to-right"

    start_r = left.center_r
    turn_c = right.center_c
    target_r = right.center_r
    assert start_r != target_r

    horizontal = [(start_r, c) for c in range(left.c1 + 1, turn_c + 1)]
    if target_r > start_r:
        vertical = [(r, turn_c) for r in range(start_r, right.r0)]
        corner_r0 = start_r
    else:
        vertical = [(r, turn_c) for r in range(right.r1 + 1, start_r + 1)]
        corner_r0 = start_r - 1

    corner_c0 = turn_c - 1
    corner = [
        (r, c)
        for r in range(corner_r0, corner_r0 + 2)
        for c in range(corner_c0, corner_c0 + 2)
    ]
    assert horizontal and vertical, "L connection must have both legs"
    return horizontal + vertical, corner


def _paint_cells(out: np.ndarray, cells: list[Coord], color: int, original_objects: np.ndarray) -> None:
    h, w = out.shape
    for r, c in cells:
        assert 0 <= r < h and 0 <= c < w, "connection must stay inside the grid"
        assert not bool(original_objects[r, c]), "connection must not cover an input object"
        out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each pair of same-color rectangles with a line style determined by their center alignment.

    Input structure:
    - The grid background is Black.
    - Every non-Black, non-Grey foreground color forms exactly two separated 4-connected solid rectangles.
    - Each rectangle has a cell-centered bbox, so the pair centers can be compared by row and column.
    - Same-color rectangle pairs may be aligned in one row, aligned in one column, or placed diagonally.

    Rule:
    - For each color, find the two rectangle centers and keep all original object pixels unchanged.
    - If the centers share a row, draw a horizontal solid segment of that color between the facing vertical faces.
    - If the centers share a column, draw a vertical solid segment of that color between the facing horizontal faces.
    - Otherwise draw a horizontal-then-vertical L-shaped connection, and mark its turn with a 2x2 Grey square.

    Key insight:
    - The visible spatial relation between the two matching rectangles chooses the connector: row and column alignments make straight lines, while diagonal placement uniquely adds the Grey corner marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must not exceed 30 in either dimension"

    colors = _foreground_colors(g)
    assert colors, "expected at least one pair of rectangles"
    assert GREY not in colors, "Grey is reserved for diagonal corner markers"

    original_objects = g != BLACK
    out = g.copy()
    for color in colors:
        rects = _components_of_color(g, color)
        assert len(rects) == 2, "each foreground color must have exactly two rectangles"
        line, corner = _connection_plan(rects[0], rects[1])
        _paint_cells(out, line, color, original_objects)
        if corner:
            _paint_cells(out, corner, GREY, original_objects)
    return out


def _make_rect(color: int, center_r: int, center_c: int, height: int, width: int) -> Rect:
    rh = height // 2
    rw = width // 2
    return Rect(int(color), center_r - rh, center_c - rw, center_r + rh, center_c + rw)


def _rect_in_bounds(rect: Rect, shape: tuple[int, int]) -> bool:
    h, w = shape
    return 0 <= rect.r0 <= rect.r1 < h and 0 <= rect.c0 <= rect.c1 < w


def _mark_rect_with_margin(mask: np.ndarray, rect: Rect) -> None:
    h, w = mask.shape
    r0 = max(0, rect.r0 - 1)
    c0 = max(0, rect.c0 - 1)
    r1 = min(h - 1, rect.r1 + 1)
    c1 = min(w - 1, rect.c1 + 1)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _fits_rect(rect: Rect, shape: tuple[int, int], blocked: np.ndarray) -> bool:
    if not _rect_in_bounds(rect, shape):
        return False
    return not bool(blocked[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1].any())


def _cells_in_bounds(cells: set[Coord], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _dims() -> tuple[int, int]:
    return random.choice((1, 3, 3, 5)), random.choice((1, 3, 3, 5))


def _sample_pair(shape: tuple[int, int], color: int, branch: str) -> tuple[Rect, Rect] | None:
    height, width = shape
    h1, w1 = _dims()
    h2, w2 = _dims()
    rh1, rw1 = h1 // 2, w1 // 2
    rh2, rw2 = h2 // 2, w2 // 2

    if branch == "h":
        gap = random.randint(3, 9)
        delta_c = rw1 + rw2 + gap + 1
        row_lo = max(rh1, rh2)
        row_hi = height - 1 - max(rh1, rh2)
        col_lo = rw1
        col_hi = width - 1 - rw2 - delta_c
        if row_lo > row_hi or col_lo > col_hi:
            return None
        r = random.randint(row_lo, row_hi)
        c1 = random.randint(col_lo, col_hi)
        c2 = c1 + delta_c
        return _make_rect(color, r, c1, h1, w1), _make_rect(color, r, c2, h2, w2)

    if branch == "v":
        gap = random.randint(3, 9)
        delta_r = rh1 + rh2 + gap + 1
        row_lo = rh1
        row_hi = height - 1 - rh2 - delta_r
        col_lo = max(rw1, rw2)
        col_hi = width - 1 - max(rw1, rw2)
        if row_lo > row_hi or col_lo > col_hi:
            return None
        r1 = random.randint(row_lo, row_hi)
        r2 = r1 + delta_r
        c = random.randint(col_lo, col_hi)
        return _make_rect(color, r1, c, h1, w1), _make_rect(color, r2, c, h2, w2)

    horizontal_gap = random.randint(4, 10)
    delta_c = rw1 + rw2 + horizontal_gap + 1
    col_lo = rw1
    col_hi = width - 1 - rw2 - delta_c
    if col_lo > col_hi:
        return None
    c1 = random.randint(col_lo, col_hi)
    c2 = c1 + delta_c

    vertical_gap = random.randint(4, 10)
    delta_r = rh2 + vertical_gap + 1
    if random.random() < 0.5:
        row_lo = rh1
        row_hi = height - 1 - rh2 - delta_r
        if row_lo > row_hi:
            return None
        r1 = random.randint(row_lo, row_hi)
        r2 = r1 + delta_r
    else:
        row_lo = rh2 + delta_r
        row_hi = height - 1 - rh1
        if row_lo > row_hi:
            return None
        r1 = random.randint(row_lo, row_hi)
        r2 = r1 - delta_r
    return _make_rect(color, r1, c1, h1, w1), _make_rect(color, r2, c2, h2, w2)


def _draw_rect(grid: np.ndarray, rect: Rect) -> None:
    grid[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] = int(rect.color)


def _try_add_pair(
    grid: np.ndarray,
    blocked_for_objects: np.ndarray,
    object_cells: set[Coord],
    reserved_connection: set[Coord],
    color: int,
    branch: str,
) -> bool:
    shape = grid.shape
    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        pair = _sample_pair(shape, color, branch)
        if pair is None:
            continue
        a, b = pair
        if not _fits_rect(a, shape, blocked_for_objects):
            continue
        if not _fits_rect(b, shape, blocked_for_objects):
            continue

        line, corner = _connection_plan(a, b)
        connection_cells = set(line) | set(corner)
        if not _cells_in_bounds(connection_cells, shape):
            continue
        if connection_cells & object_cells:
            continue
        if connection_cells & reserved_connection:
            continue
        if (a.cells() | b.cells()) & reserved_connection:
            continue

        _draw_rect(grid, a)
        _draw_rect(grid, b)
        object_cells.update(a.cells())
        object_cells.update(b.cells())
        reserved_connection.update(connection_cells)
        _mark_rect_with_margin(blocked_for_objects, a)
        _mark_rect_with_margin(blocked_for_objects, b)
        for r, c in connection_cells:
            blocked_for_objects[r, c] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(20, 30)
    width = random.randint(20, 30)
    count = random.randint(3, 5)
    branches = ["h", "v", "d"]
    branches.extend(random.choice(("h", "v", "d")) for _ in range(count - 3))
    random.shuffle(branches)

    colors = random.sample(RECT_COLORS, count)
    grid = np.full((height, width), BLACK, dtype=int)
    blocked_for_objects = np.zeros((height, width), dtype=bool)
    object_cells: set[Coord] = set()
    reserved_connection: set[Coord] = set()

    for color, branch in zip(colors, branches):
        if not _try_add_pair(
            grid,
            blocked_for_objects,
            object_cells,
            reserved_connection,
            int(color),
            branch,
        ):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
