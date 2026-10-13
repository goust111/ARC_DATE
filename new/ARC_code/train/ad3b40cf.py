from __future__ import annotations

import numpy as np

from arc_utils import colors_present, components_mask, get_bg
import random


Axis = tuple[int, str, int]


def _find_reflection_axis(g: np.ndarray, bg: int) -> Axis:
    h, w = g.shape
    candidates: list[Axis] = []

    for color in colors_present(g, exclude={bg}):
        mask = g == color
        count = int(mask.sum())

        if count == w:
            for r in np.flatnonzero(mask.all(axis=1)):
                candidates.append((int(color), "row", int(r)))

        if count == h:
            for c in np.flatnonzero(mask.all(axis=0)):
                candidates.append((int(color), "col", int(c)))

        if h == w and count == h:
            main_diag = np.eye(h, dtype=bool)
            anti_diag = np.fliplr(main_diag)
            if np.all(mask[main_diag]):
                candidates.append((int(color), "main_diag", 0))
            if np.all(mask[anti_diag]):
                candidates.append((int(color), "anti_diag", 0))

    assert len(candidates) == 1, "expected exactly one full straight reflection axis"
    return candidates[0]


def _reflect(axis: Axis, shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    _axis_color, kind, index = axis
    h, w = shape

    if kind == "row":
        return 2 * index - r, c
    if kind == "col":
        return r, 2 * index - c
    if kind == "main_diag":
        assert h == w, "diagonal reflection axis requires a square grid"
        return c, r
    if kind == "anti_diag":
        assert h == w, "diagonal reflection axis requires a square grid"
        return h - 1 - c, w - 1 - r

    raise AssertionError(f"unknown axis kind: {kind}")


def _smallest_non_axis_color(g: np.ndarray, bg: int, axis_color: int) -> int:
    areas = {
        int(color): int(np.count_nonzero(g == color))
        for color in colors_present(g, exclude={bg, axis_color})
    }
    assert len(areas) >= 2, "expected at least two non-axis foreground colors"

    min_area = min(areas.values())
    colors = [color for color, area in areas.items() if area == min_area]
    assert len(colors) == 1, "smallest non-axis foreground color must be unique"
    return colors[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the smallest non-axis foreground color across the full straight axis.

    Input structure
    - The background is the dominant color.
    - One foreground color forms exactly one full row, column, main diagonal, or anti-diagonal reflection axis.
    - The remaining foreground colors include a unique least-populated color to copy.

    Rule
    - Preserve the input grid.
    - Select every pixel of the unique smallest non-axis foreground color.
    - Reflect those pixels across the full straight axis and paint the reflected positions with the same color.

    Key insight
    - The axis color marks the mirror line, while the color population chooses which set of objects is duplicated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    axis = _find_reflection_axis(g, bg)
    axis_color = axis[0]
    mirror_color = _smallest_non_axis_color(g, bg, axis_color)

    out = g.copy()
    h, w = g.shape
    for r_raw, c_raw in np.argwhere(g == mirror_color):
        r = int(r_raw)
        c = int(c_raw)
        rr, cc = _reflect(axis, g.shape, r, c)
        assert 0 <= rr < h and 0 <= cc < w, "reflected pixel must stay inside the grid"
        assert int(out[rr, cc]) in {bg, mirror_color}, "reflection cannot overwrite another foreground object"
        out[rr, cc] = mirror_color

    return out

# --------------- generator (att2) ---------------
# Canvas and axis options.
_GEN_GRID_SIZE_RANGE = (9, 17)
_GEN_ODD_GRID_SIZES = [9, 11, 13, 15, 17]
_GEN_AXIS_KINDS = ["row", "col", "main_diag", "anti_diag"]
_GEN_CENTER_AXIS_KINDS = {"row", "col"}
_GEN_SIDE_COUNT = 2
# Foreground component size and count bounds.
_GEN_COMPONENT_COUNT_RANGE = range(1, 4)
_GEN_MIRROR_TOTAL_RANGE = (4, 12)
_GEN_MIRROR_COMPONENT_SIZE_RANGE = (2, 7)
_GEN_NONMIRROR_TOTAL_RANGE = (5, 17)
_GEN_NONMIRROR_COMPONENT_SIZE_RANGE = (3, 9)
_GEN_MIN_NONMIRROR_RATIO = (5, 4)
_GEN_MAX_NONMIRROR_MULTIPLIER = 2
# Search and geometry primitives.
_GEN_RETRY_LIMIT = 300
_GEN_COLORS = list(range(10))
_GEN_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

def _gen_draw_axis(grid: np.ndarray, axis_kind: str, color: int) -> None:
    n = grid.shape[0]
    if axis_kind == "row":
        grid[n // 2, :] = color
    elif axis_kind == "col":
        grid[:, n // 2] = color
    elif axis_kind == "main_diag":
        np.fill_diagonal(grid, color)
    else:
        np.fill_diagonal(np.fliplr(grid), color)


def _gen_side_mask(n: int, axis_kind: str, side: int) -> np.ndarray:
    rr, cc = np.indices((n, n))
    if axis_kind == "row":
        return rr < n // 2 if side == 0 else rr > n // 2
    if axis_kind == "col":
        return cc < n // 2 if side == 0 else cc > n // 2
    if axis_kind == "main_diag":
        return cc > rr if side == 0 else cc < rr
    return rr + cc < n - 1 if side == 0 else rr + cc > n - 1


def _gen_reflect(axis_kind: str, n: int, r: int, c: int) -> tuple[int, int]:
    if axis_kind == "row":
        return n - 1 - r, c
    if axis_kind == "col":
        return r, n - 1 - c
    if axis_kind == "main_diag":
        return c, r
    return n - 1 - c, n - 1 - r


def _gen_feasible_counts(total: int, lo: int, hi: int) -> list[int]:
    return [count for count in _GEN_COMPONENT_COUNT_RANGE if count * lo <= total <= count * hi]


def _gen_partition(total: int, count: int, lo: int, hi: int) -> list[int]:
    sizes = []
    remaining = total
    for idx in range(count):
        left = count - idx - 1
        cur_lo = max(lo, remaining - left * hi)
        cur_hi = min(hi, remaining - left * lo)
        size = random.randint(cur_lo, cur_hi)
        sizes.append(size)
        remaining -= size
    random.shuffle(sizes)
    return sizes


def _gen_pick_component_sizes() -> tuple[list[int], list[int]]:
    mirror_total = random.randint(*_GEN_MIRROR_TOTAL_RANGE)
    ratio_num, ratio_den = _GEN_MIN_NONMIRROR_RATIO
    nonmirror_lo = max(
        _GEN_NONMIRROR_TOTAL_RANGE[0],
        (ratio_num * mirror_total + ratio_den - 1) // ratio_den,
    )
    nonmirror_hi = min(_GEN_NONMIRROR_TOTAL_RANGE[1], _GEN_MAX_NONMIRROR_MULTIPLIER * mirror_total)
    nonmirror_total = random.randint(nonmirror_lo, nonmirror_hi)

    mirror_count = random.choice(_gen_feasible_counts(mirror_total, *_GEN_MIRROR_COMPONENT_SIZE_RANGE))
    nonmirror_count = random.choice(_gen_feasible_counts(nonmirror_total, *_GEN_NONMIRROR_COMPONENT_SIZE_RANGE))
    mirror_sizes = _gen_partition(mirror_total, mirror_count, *_GEN_MIRROR_COMPONENT_SIZE_RANGE)
    nonmirror_sizes = _gen_partition(nonmirror_total, nonmirror_count, *_GEN_NONMIRROR_COMPONENT_SIZE_RANGE)
    return mirror_sizes, nonmirror_sizes


def _gen_mark_neighbors(blocked: np.ndarray, coords: np.ndarray) -> None:
    n = blocked.shape[0]
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        blocked[r, c] = True
        for dr, dc in _GEN_DIRS4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < n and 0 <= cc < n:
                blocked[rr, cc] = True


def _gen_grow_component(
    side_mask: np.ndarray, occupied: np.ndarray, same_color_blocked: np.ndarray, size: int
) -> np.ndarray | None:
    available = side_mask & ~occupied & ~same_color_blocked
    choices = [coords for coords in components_mask(available, connectivity=4) if len(coords) >= size]
    if not choices:
        return None

    region = random.choice(choices)
    seed = tuple(int(x) for x in region[random.randrange(len(region))])
    selected = {seed}
    frontier: set[tuple[int, int]] = set()
    n = available.shape[0]

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in _GEN_DIRS4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < n and 0 <= cc < n and available[rr, cc] and (rr, cc) not in selected:
                frontier.add((rr, cc))

    add_frontier(*seed)
    while len(selected) < size:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        selected.add(cell)
        add_frontier(*cell)

    return np.array(list(selected), dtype=int)


def _gen_place_components(
    grid: np.ndarray,
    side_mask: np.ndarray,
    occupied: np.ndarray,
    color: int,
    sizes: list[int],
) -> list[tuple[int, int]] | None:
    same_color_blocked = np.zeros_like(occupied, dtype=bool)
    placed: list[tuple[int, int]] = []
    for size in sizes:
        coords = _gen_grow_component(side_mask, occupied, same_color_blocked, size)
        if coords is None:
            return None
        grid[coords[:, 0], coords[:, 1]] = color
        occupied[coords[:, 0], coords[:, 1]] = True
        _gen_mark_neighbors(same_color_blocked, coords)
        placed.extend((int(r), int(c)) for r, c in coords)
    return placed


def _gen_pick_axis_layout() -> tuple[str, int]:
    axis_kind = random.choice(_GEN_AXIS_KINDS)
    if axis_kind in _GEN_CENTER_AXIS_KINDS:
        n = random.choice(_GEN_ODD_GRID_SIZES)
    else:
        n = random.randint(*_GEN_GRID_SIZE_RANGE)
    return axis_kind, n


def _gen_pick_palette() -> tuple[int, int, int, int]:
    bg = random.choice(_GEN_COLORS)
    axis_color = random.choice([color for color in _GEN_COLORS if color != bg])
    mirror_color, nonmirror_color = random.sample(
        [color for color in _GEN_COLORS if color not in {bg, axis_color}], 2
    )
    return bg, axis_color, mirror_color, nonmirror_color


def _gen_reflect_mirror_cells(
    out: np.ndarray, mirror_cells: list[tuple[int, int]], axis_kind: str, bg: int, mirror_color: int
) -> list[tuple[int, int]] | None:
    n = out.shape[0]
    reflected: list[tuple[int, int]] = []
    for r, c in mirror_cells:
        rr, cc = _gen_reflect(axis_kind, n, r, c)
        if out[rr, cc] not in {bg, mirror_color}:
            return None
        reflected.append((rr, cc))
        out[rr, cc] = mirror_color
    return reflected


def _gen_attempt() -> np.ndarray | None:
    axis_kind, n = _gen_pick_axis_layout()
    bg, axis_color, mirror_color, nonmirror_color = _gen_pick_palette()
    mirror_sizes, nonmirror_sizes = _gen_pick_component_sizes()

    out = np.full((n, n), bg, dtype=int)
    _gen_draw_axis(out, axis_kind, axis_color)
    occupied = out != bg
    side_mask = _gen_side_mask(n, axis_kind, random.randrange(_GEN_SIDE_COUNT))

    if _gen_place_components(out, side_mask, occupied, nonmirror_color, nonmirror_sizes) is None:
        return None
    mirror_cells = _gen_place_components(out, side_mask, occupied, mirror_color, mirror_sizes)
    if mirror_cells is None:
        return None

    reflected = _gen_reflect_mirror_cells(out, mirror_cells, axis_kind, bg, mirror_color)
    if reflected is None:
        return None

    inp = out.copy()
    for rr, cc in reflected:
        inp[rr, cc] = bg

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ad3b40cf input")
