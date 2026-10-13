import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8
BODY_COLORS = (BLUE, GREEN, ORANGE, TEAL)

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class Rectangle:
    r0: int
    c0: int
    r1: int
    c1: int
    color: int
    active: bool

    @property
    def center(self) -> tuple[int, int]:
        return ((self.r0 + self.r1) // 2, (self.c0 + self.c1) // 2)

    def contains(self, r: int, c: int) -> bool:
        return self.r0 <= r <= self.r1 and self.c0 <= c <= self.c1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_divider(grid: np.ndarray) -> tuple[str, int]:
    full_rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    full_cols = np.flatnonzero(np.all(grid == GREY, axis=0))
    grey_mask = grid == GREY

    if len(full_rows) == 1 and len(full_cols) == 0:
        axis = int(full_rows[0])
        expected = np.zeros_like(grey_mask, dtype=bool)
        expected[axis, :] = True
        assert np.array_equal(grey_mask, expected), "expected exactly one full grey divider row"
        return "horizontal", axis

    if len(full_cols) == 1 and len(full_rows) == 0:
        axis = int(full_cols[0])
        expected = np.zeros_like(grey_mask, dtype=bool)
        expected[:, axis] = True
        assert np.array_equal(grey_mask, expected), "expected exactly one full grey divider column"
        return "vertical", axis

    raise AssertionError("expected one full grey divider line")


def _find_rectangles(grid: np.ndarray) -> list[Rectangle]:
    mask = (grid != BLACK) & (grid != GREY)
    comps = _components(mask)
    rects: list[Rectangle] = []
    for coords in comps:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert region.shape[0] >= 3 and region.shape[1] >= 3, "rectangles must be at least 3x3"
        assert not np.any((region == BLACK) | (region == GREY)), "rectangle bbox must be solid"
        body_colors = sorted(set(map(int, np.unique(region))) - {RED})
        assert len(body_colors) == 1, "each rectangle must have one body color plus optional red markers"
        color = int(body_colors[0])

        border = np.zeros(region.shape, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert not np.any((region == RED) & ~border), "red markers must sit on rectangle edges"

        top = bool(np.any(region[0, :] == RED))
        bottom = bool(np.any(region[-1, :] == RED))
        left = bool(np.any(region[:, 0] == RED))
        right = bool(np.any(region[:, -1] == RED))
        has_red = bool(np.any(region == RED))
        active = top and bottom and left and right
        assert (not has_red) or active, "partially marked rectangles are invalid"

        rects.append(Rectangle(int(r0), int(c0), int(r1), int(c1), color, active))

    assert rects, "expected at least one rectangle"
    return rects


def _ray_dirs(rect: Rectangle, divider_orient: str, axis: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if divider_orient == "horizontal":
        if rect.r1 < axis:
            return (1, 0), (-1, 0)
        assert rect.r0 > axis, "rectangles must not cross the divider"
        return (-1, 0), (1, 0)

    if rect.c1 < axis:
        return (0, 1), (0, -1)
    assert rect.c0 > axis, "rectangles must not cross the divider"
    return (0, -1), (0, 1)


def _is_divider_cell(r: int, c: int, divider_orient: str, axis: int) -> bool:
    return r == axis if divider_orient == "horizontal" else c == axis


def _cast_ray(
    out: np.ndarray,
    rect: Rectangle,
    direction: tuple[int, int],
    color: int,
    divider_orient: str,
    axis: int,
) -> None:
    h, w = out.shape
    dr, dc = direction
    r, c = rect.center
    r += dr
    c += dc

    while 0 <= r < h and 0 <= c < w:
        if _is_divider_cell(r, c, divider_orient, axis):
            break
        if not rect.contains(r, c) and out[r, c] != BLACK:
            break
        out[r, c] = int(color)
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every red-marked rectangle to yellow, then emit toward-divider rays in the rectangle's original color and away-divider rays in yellow.

    Input structure:
    - The grid has a black background with exactly one full grey divider row or column.
    - Each side of the divider contains one or more solid axis-aligned rectangles made from one body color in {blue, green, orange, teal}, with some rectangles carrying red edge markers.
    - A rectangle is active exactly when its top, bottom, left, and right edges each contain at least one red pixel; inactive rectangles have no red markers.

    Rule:
    - For every active rectangle, remember its original non-red body color and recolor all of its non-red cells to yellow, leaving red marker cells unchanged.
    - From the rectangle's geometric center, cast two axis-aligned half-rays perpendicular to the divider: the half-ray toward the divider uses the original body color, and the half-ray away from the divider uses yellow.
    - Each half-ray may repaint cells inside its own source rectangle, but once it exits that rectangle it can continue only through black background and must stop before the grid edge, the grey divider, any other rectangle, or any previously drawn ray.

    Key insight:
    - The original body color disappears when the active rectangle is recolored to yellow, so the only way to recover it is to notice that the toward-divider ray preserves the rectangle's pre-recolor color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider_orient, axis = _find_divider(g)
    rects = _find_rectangles(g)
    active_rects = [rect for rect in rects if rect.active]
    inactive_rects = [rect for rect in rects if not rect.active]
    assert len(active_rects) >= 2, "expected at least two active rectangles"
    assert len(inactive_rects) >= 1, "expected at least one inactive rectangle"
    assert len({rect.color for rect in active_rects}) >= 2, "active rectangles must use at least two body colors"

    out = g.copy()
    for rect in active_rects:
        region = out[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1]
        region[region != RED] = YELLOW

    for rect in sorted(active_rects, key=lambda item: (item.center[0], item.center[1], item.r0, item.c0)):
        toward, away = _ray_dirs(rect, divider_orient, axis)
        _cast_ray(out, rect, toward, rect.color, divider_orient, axis)
        _cast_ray(out, rect, away, YELLOW, divider_orient, axis)

    return out


GRID_MIN = 18
GRID_MAX = 24
RECT_MIN = 3
RECT_MAX = 6
CROSS_GAP = 2
GEN_RETRIES = 400


def _pack_spans(start: int, end: int, sizes: list[int], gap: int) -> list[tuple[int, int]] | None:
    length = end - start + 1
    extra = length - sum(sizes) - gap * (len(sizes) - 1)
    if extra < 0:
        return None
    slack = np.random.multinomial(extra, [1.0 / (len(sizes) + 1)] * (len(sizes) + 1))
    spans: list[tuple[int, int]] = []
    pos = start + int(slack[0])
    for i, size in enumerate(sizes):
        spans.append((int(pos), int(pos + size - 1)))
        pos += size
        if i + 1 < len(sizes):
            pos += gap + int(slack[i + 1])
    random.shuffle(spans)
    return spans


def _choose_side_counts() -> tuple[int, int]:
    while True:
        counts = (random.randint(1, 3), random.randint(1, 3))
        if sum(counts) >= 3:
            return counts


def _choose_active_indices(side_counts: tuple[int, int]) -> set[int]:
    total = sum(side_counts)
    offsets = (0, side_counts[0])
    active_count = random.randint(2, total - 1)
    active = {
        random.randrange(offsets[0], offsets[0] + side_counts[0]),
        random.randrange(offsets[1], offsets[1] + side_counts[1]),
    }
    remaining = [idx for idx in range(total) if idx not in active]
    random.shuffle(remaining)
    for idx in remaining[: active_count - len(active)]:
        active.add(idx)
    return active


def _assign_colors(total: int, active_indices: set[int]) -> list[int]:
    colors = [random.choice(BODY_COLORS) for _ in range(total)]
    active_list = sorted(active_indices)
    c0, c1 = random.sample(BODY_COLORS, 2)
    colors[active_list[0]] = c0
    colors[active_list[1]] = c1
    return colors


def _side_row_bounds(h: int, axis: int, side: str) -> tuple[int, int]:
    if side == "top":
        return 1, axis - 2
    if side == "bottom":
        return axis + 2, h - 2
    raise ValueError(side)


def _side_col_bounds(w: int, axis: int, side: str) -> tuple[int, int]:
    if side == "left":
        return 1, axis - 2
    if side == "right":
        return axis + 2, w - 2
    raise ValueError(side)


def _pick_parallel_span(start: int, end: int, size: int) -> tuple[int, int] | None:
    if end - start + 1 < size:
        return None
    origin = random.randint(start, end - size + 1)
    return origin, origin + size - 1


def _place_side_rectangles(
    h: int,
    w: int,
    divider_orient: str,
    axis: int,
    side: str,
    specs: list[tuple[bool, int]],
) -> list[Rectangle] | None:
    count = len(specs)
    if divider_orient == "horizontal":
        widths = [random.randint(RECT_MIN, RECT_MAX) for _ in range(count)]
        spans = _pack_spans(1, w - 2, widths, CROSS_GAP)
        row_start, row_end = _side_row_bounds(h, axis, side)
        if spans is None or row_end - row_start + 1 < RECT_MIN:
            return None
        rects: list[Rectangle] = []
        for (active, color), (c0, c1) in zip(specs, spans):
            max_h = min(RECT_MAX, row_end - row_start + 1)
            height = random.randint(RECT_MIN, max_h)
            rows = _pick_parallel_span(row_start, row_end, height)
            if rows is None:
                return None
            r0, r1 = rows
            rects.append(Rectangle(r0, c0, r1, c1, color, active))
        return rects

    heights = [random.randint(RECT_MIN, RECT_MAX) for _ in range(count)]
    spans = _pack_spans(1, h - 2, heights, CROSS_GAP)
    col_start, col_end = _side_col_bounds(w, axis, side)
    if spans is None or col_end - col_start + 1 < RECT_MIN:
        return None
    rects = []
    for (active, color), (r0, r1) in zip(specs, spans):
        max_w = min(RECT_MAX, col_end - col_start + 1)
        width = random.randint(RECT_MIN, max_w)
        cols = _pick_parallel_span(col_start, col_end, width)
        if cols is None:
            return None
        c0, c1 = cols
        rects.append(Rectangle(r0, c0, r1, c1, color, active))
    return rects


def _preferred_edge_positions(values: list[int]) -> list[int]:
    if len(values) > 2:
        return values[1:-1]
    return values


def _mark_active_rectangle(grid: np.ndarray, rect: Rectangle) -> None:
    edge_cells = {
        "top": [(rect.r0, c) for c in range(rect.c0, rect.c1 + 1)],
        "bottom": [(rect.r1, c) for c in range(rect.c0, rect.c1 + 1)],
        "left": [(r, rect.c0) for r in range(rect.r0, rect.r1 + 1)],
        "right": [(r, rect.c1) for r in range(rect.r0, rect.r1 + 1)],
    }
    chosen: set[tuple[int, int]] = set()

    top_cols = _preferred_edge_positions(list(range(rect.c0, rect.c1 + 1)))
    side_rows = _preferred_edge_positions(list(range(rect.r0, rect.r1 + 1)))
    chosen.add((rect.r0, random.choice(top_cols)))
    chosen.add((rect.r1, random.choice(top_cols)))
    chosen.add((random.choice(side_rows), rect.c0))
    chosen.add((random.choice(side_rows), rect.c1))

    border = sorted(set(edge_cells["top"] + edge_cells["bottom"] + edge_cells["left"] + edge_cells["right"]))
    remaining = [cell for cell in border if cell not in chosen]
    extra_count = random.randint(0, min(3, len(remaining)))
    if extra_count:
        chosen.update(random.sample(remaining, extra_count))

    for r, c in chosen:
        grid[r, c] = RED


def _render_input(h: int, w: int, divider_orient: str, axis: int, rects: list[Rectangle]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    if divider_orient == "horizontal":
        grid[axis, :] = GREY
    else:
        grid[:, axis] = GREY

    for rect in rects:
        grid[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] = rect.color
        if rect.active:
            _mark_active_rectangle(grid, rect)
    return grid


def _divider_axis(length: int) -> int:
    low = max(6, length // 3)
    high = min(length - 7, (2 * length) // 3)
    return random.randint(low, high)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    divider_orient = random.choice(("horizontal", "vertical"))
    axis = _divider_axis(h if divider_orient == "horizontal" else w)
    side_names = ("top", "bottom") if divider_orient == "horizontal" else ("left", "right")
    side_counts = _choose_side_counts()
    active_indices = _choose_active_indices(side_counts)
    colors = _assign_colors(sum(side_counts), active_indices)

    rects: list[Rectangle] = []
    start = 0
    for side, count in zip(side_names, side_counts):
        specs = [
            (idx in active_indices, colors[idx])
            for idx in range(start, start + count)
        ]
        placed = _place_side_rectangles(h, w, divider_orient, axis, side, specs)
        if placed is None:
            return None
        rects.extend(placed)
        start += count

    grid = _render_input(h, w, divider_orient, axis, rects)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.any(out == YELLOW):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 7")
