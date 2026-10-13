from __future__ import annotations

import random

import numpy as np


ORANGE = 7
YELLOW = 4
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_COLORS = (0, 1, 2, 3, 5, 6, 8, 9)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components_by_color(grid: np.ndarray, bg: int) -> dict[int, list[set[tuple[int, int]]]]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    groups: dict[int, list[set[tuple[int, int]]]] = {}

    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            seen[r, c] = True
            stack = [(r, c)]
            cells: set[tuple[int, int]] = set()
            while stack:
                rr, cc = stack.pop()
                cells.add((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            groups.setdefault(color, []).append(cells)

    return groups


def _arrow_pixels_from_anchor(anchor: tuple[int, int], dr: int, dc: int) -> set[tuple[int, int]]:
    r, c = anchor
    if dr == 0:
        base_c = c + dc
        return {(r, c + 2 * dc), (r - 1, base_c), (r + 1, base_c)}

    base_r = r + dr
    return {(r + 2 * dr, c), (base_r, c - 1), (base_r, c + 1)}


def _arrow_zone_from_anchor(anchor: tuple[int, int], dr: int, dc: int) -> set[tuple[int, int]]:
    r, c = anchor
    gap = {(r, c + dc)} if dr == 0 else {(r + dr, c)}
    return _arrow_pixels_from_anchor(anchor, dr, dc) | gap


def _pick_anchor(
    cells: set[tuple[int, int]],
    dr: int,
    dc: int,
    target_axis: float,
    shape: tuple[int, int],
) -> tuple[int, int]:
    if dr == 0:
        edge = max(c for _r, c in cells) if dc > 0 else min(c for _r, c in cells)
        strong: list[tuple[int, int]] = []
        weak: list[tuple[int, int]] = []
        for r, c in cells:
            if c != edge:
                continue
            zone = _arrow_zone_from_anchor((r, c), dr, dc)
            if any(not _in_bounds(shape, rr, cc) or (rr, cc) in cells for rr, cc in zone):
                continue
            weak.append((r, c))
            if (r - 1, c) in cells and (r + 1, c) in cells:
                strong.append((r, c))

        pool = strong or weak
        if not pool:
            raise AssertionError("no valid horizontal anchor for arrow")
        return min(pool, key=lambda rc: (abs(rc[0] - target_axis), rc[0], rc[1]))

    edge = max(r for r, _c in cells) if dr > 0 else min(r for r, _c in cells)
    strong = []
    weak = []
    for r, c in cells:
        if r != edge:
            continue
        zone = _arrow_zone_from_anchor((r, c), dr, dc)
        if any(not _in_bounds(shape, rr, cc) or (rr, cc) in cells for rr, cc in zone):
            continue
        weak.append((r, c))
        if (r, c - 1) in cells and (r, c + 1) in cells:
            strong.append((r, c))

    pool = strong or weak
    if not pool:
        raise AssertionError("no valid vertical anchor for arrow")
    return min(pool, key=lambda rc: (abs(rc[1] - target_axis), rc[1], rc[0]))


def _arrow_and_zone(
    cells: set[tuple[int, int]],
    dr: int,
    dc: int,
    target_axis: float,
    shape: tuple[int, int],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    anchor = _pick_anchor(cells, dr, dc, target_axis, shape)
    return _arrow_pixels_from_anchor(anchor, dr, dc), _arrow_zone_from_anchor(anchor, dr, dc)


def _pair_arrow_data(
    cells_a: set[tuple[int, int]],
    cells_b: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    rows_a = {r for r, _c in cells_a}
    rows_b = {r for r, _c in cells_b}
    cols_a = {c for _r, c in cells_a}
    cols_b = {c for _r, c in cells_b}

    shared_rows = sorted(rows_a & rows_b)
    shared_cols = sorted(cols_a & cols_b)

    if shared_rows and not shared_cols:
        target_row = 0.5 * (shared_rows[0] + shared_rows[-1])
        max_a = max(c for _r, c in cells_a)
        min_a = min(c for _r, c in cells_a)
        max_b = max(c for _r, c in cells_b)
        min_b = min(c for _r, c in cells_b)
        if max_a < min_b:
            arrow_a, zone_a = _arrow_and_zone(cells_a, 0, 1, target_row, shape)
            arrow_b, zone_b = _arrow_and_zone(cells_b, 0, -1, target_row, shape)
            return arrow_a, zone_a, arrow_b, zone_b
        if max_b < min_a:
            arrow_b, zone_b = _arrow_and_zone(cells_b, 0, 1, target_row, shape)
            arrow_a, zone_a = _arrow_and_zone(cells_a, 0, -1, target_row, shape)
            return arrow_a, zone_a, arrow_b, zone_b
        raise AssertionError("horizontal pair objects must be separated")

    if shared_cols and not shared_rows:
        target_col = 0.5 * (shared_cols[0] + shared_cols[-1])
        max_a = max(r for r, _c in cells_a)
        min_a = min(r for r, _c in cells_a)
        max_b = max(r for r, _c in cells_b)
        min_b = min(r for r, _c in cells_b)
        if max_a < min_b:
            arrow_a, zone_a = _arrow_and_zone(cells_a, 1, 0, target_col, shape)
            arrow_b, zone_b = _arrow_and_zone(cells_b, -1, 0, target_col, shape)
            return arrow_a, zone_a, arrow_b, zone_b
        if max_b < min_a:
            arrow_b, zone_b = _arrow_and_zone(cells_b, 1, 0, target_col, shape)
            arrow_a, zone_a = _arrow_and_zone(cells_a, -1, 0, target_col, shape)
            return arrow_a, zone_a, arrow_b, zone_b
        raise AssertionError("vertical pair objects must be separated")

    raise AssertionError("paired objects must align in exactly one axis")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a yellow arrow beside each paired object, pointing toward its nearest same-color neighbor.

    Input structure:
    - The background is orange.
    - Foreground objects are monochrome 4-connected components.
    - Some colors appear twice and form an aligned pair; other colors appear once.

    Rule:
    - Only colors with exactly two objects participate.
    - If a pair shares rows, each object gets a left/right arrow on the side facing the other object.
    - If a pair shares columns, each object gets an up/down arrow on the side facing the other object.
    - Each arrow uses exactly three yellow pixels: one tip plus two wings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    out = g.copy()
    by_color = _components_by_color(g, bg)

    for color in sorted(by_color):
        components = by_color[color]
        assert len(components) in (1, 2), "each foreground color must appear once or twice"
        if len(components) == 1:
            continue

        arrow_a, zone_a, arrow_b, zone_b = _pair_arrow_data(components[0], components[1], g.shape)
        for zone in (zone_a, zone_b):
            for r, c in zone:
                assert int(g[r, c]) == bg, "arrow clearance cells must be background"
        for arrow in (arrow_a, arrow_b):
            for r, c in arrow:
                assert int(out[r, c]) == bg, "arrows must not overlap"
                out[r, c] = YELLOW

    return out


GEN_GRID_MIN = 18
GEN_GRID_MAX = 24
GEN_PAIR_COUNT_MIN = 1
GEN_PAIR_COUNT_MAX = 3
GEN_SINGLETON_COUNT_MAX = 2
GEN_PAIR_TRIES = 150
GEN_SINGLETON_TRIES = 150
GEN_MAX_ATTEMPTS = 400


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _dilate(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if _in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _sample_orientations(pair_count: int) -> list[str]:
    if pair_count == 1:
        return [random.choice(("horizontal", "vertical"))]

    orientations = ["horizontal", "vertical"]
    while len(orientations) < pair_count:
        orientations.append(random.choice(("horizontal", "vertical")))
    random.shuffle(orientations)
    return orientations


def _try_place_pair(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    color: int,
    orientation: str,
) -> bool:
    height, width = grid.shape

    for _ in range(GEN_PAIR_TRIES):
        if orientation == "horizontal":
            obj_h = random.randint(3, 5)
            left_w = random.randint(2, 5)
            right_w = random.randint(2, 5)
            gap = random.randint(4, 8)
            total_w = left_w + gap + right_w
            if height < obj_h + 2 or width < total_w + 2:
                continue

            r0 = random.randint(1, height - obj_h - 1)
            c0 = random.randint(1, width - total_w - 1)
            cells_a = _rect_cells(r0, c0, obj_h, left_w)
            cells_b = _rect_cells(r0, c0 + left_w + gap, obj_h, right_w)
        else:
            obj_w = random.randint(3, 5)
            top_h = random.randint(2, 5)
            bot_h = random.randint(2, 5)
            gap = random.randint(4, 8)
            total_h = top_h + gap + bot_h
            if height < total_h + 2 or width < obj_w + 2:
                continue

            r0 = random.randint(1, height - total_h - 1)
            c0 = random.randint(1, width - obj_w - 1)
            cells_a = _rect_cells(r0, c0, top_h, obj_w)
            cells_b = _rect_cells(r0 + top_h + gap, c0, bot_h, obj_w)

        try:
            arrow_a, zone_a, arrow_b, zone_b = _pair_arrow_data(cells_a, cells_b, grid.shape)
        except AssertionError:
            continue

        footprint = cells_a | cells_b | arrow_a | arrow_b
        if len(arrow_a | arrow_b) != 6:
            continue

        clearance = _dilate(footprint | zone_a | zone_b, grid.shape)
        if clearance & reserved:
            continue

        _paint_cells(grid, cells_a | cells_b, color)
        reserved |= clearance
        return True

    return False


def _try_place_singleton(grid: np.ndarray, reserved: set[tuple[int, int]], color: int) -> bool:
    height, width = grid.shape

    for _ in range(GEN_SINGLETON_TRIES):
        obj_h = random.randint(1, 5)
        obj_w = random.randint(1, 5)
        if height < obj_h + 2 or width < obj_w + 2:
            continue

        r0 = random.randint(1, height - obj_h - 1)
        c0 = random.randint(1, width - obj_w - 1)
        cells = _rect_cells(r0, c0, obj_h, obj_w)
        clearance = _dilate(cells, grid.shape)
        if clearance & reserved:
            continue

        _paint_cells(grid, cells, color)
        reserved |= clearance
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    width = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    grid = np.full((height, width), ORANGE, dtype=int)

    pair_count = random.randint(GEN_PAIR_COUNT_MIN, GEN_PAIR_COUNT_MAX)
    singleton_count = random.randint(0, GEN_SINGLETON_COUNT_MAX)
    total_colors = pair_count + singleton_count
    colors = random.sample(GEN_COLORS, total_colors)
    pair_colors = colors[:pair_count]
    singleton_colors = colors[pair_count:]

    reserved: set[tuple[int, int]] = set()
    for color, orientation in zip(pair_colors, _sample_orientations(pair_count)):
        if not _try_place_pair(grid, reserved, color, orientation):
            return None

    for color in singleton_colors:
        if not _try_place_singleton(grid, reserved, color):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.count_nonzero(out == YELLOW) != pair_count * 6:
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
