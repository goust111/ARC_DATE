from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
TARGET_COLORS = (1, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ("up", "down", "left", "right")
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

CELL_SIZE_CHOICES = (5, 7, 9)
TABLE_LAYOUTS = (
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 1),
    (3, 2),
    (3, 3),
)
MAX_GENERATION_ATTEMPTS = 400
MAX_CELL_ATTEMPTS = 120


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each cell's colored target pixels using the axis encoded by its red arrow, then erase the arrow.

    Input structure:
    - The background is black.
    - Full grey rows and columns form a table-like lattice that splits the grid into rectangular cells.
    - Each cell contains exactly one red arrow made of a straight stem and a two-pixel head, plus scattered
      target pixels in colors that are neither black, grey, nor red.

    Rule:
    - Decode the arrow in each cell: the stem direction gives the mirror-axis orientation, and the stem length
      gives the signed offset from the cell center.
    - Up/down arrows define a horizontal mirror axis; left/right arrows define a vertical mirror axis.
    - Keep every original target pixel, add its reflected copy if the mirrored location stays inside the same cell,
      and erase every red arrow pixel to black.

    Key insight:
    - The arrow carries two independent instructions at once: its facing direction selects whether to reflect across
      rows or columns, while its stem length shifts that axis away from the geometric center of the cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    sep_rows = [r for r in range(height) if np.all(g[r, :] == GREY)]
    sep_cols = [c for c in range(width) if np.all(g[:, c] == GREY)]
    row_bands = _bands(height, sep_rows)
    col_bands = _bands(width, sep_cols)
    assert row_bands and col_bands, "expected grey lattice bands"

    out = g.copy()
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            arrow = _decode_arrow(cell)
            axis_mode, axis2 = _axis_spec(cell.shape, arrow["direction"], arrow["length"])

            cell_out = out[r0 : r1 + 1, c0 : c1 + 1]
            cell_out[cell == RED] = BLACK

            targets = np.argwhere((cell != BLACK) & (cell != GREY) & (cell != RED))
            for lr, lc in targets:
                mr, mc = _mirror_point(int(lr), int(lc), cell.shape, axis_mode, axis2)
                if 0 <= mr < cell.shape[0] and 0 <= mc < cell.shape[1]:
                    cell_out[mr, mc] = int(cell[lr, lc])

    return out


def _bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    prev = -1
    for sep in sorted(separators) + [length]:
        if sep - prev > 1:
            bands.append((prev + 1, sep - 1))
        prev = sep
    return bands


def _neighbors4(point: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = point
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _component(points: set[tuple[int, int]], start: tuple[int, int]) -> set[tuple[int, int]]:
    seen = {start}
    stack = [start]
    while stack:
        point = stack.pop()
        for nbr in _neighbors4(point):
            if nbr in points and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return seen


def _decode_arrow(cell: np.ndarray) -> dict[str, object]:
    red_points = {tuple(map(int, rc)) for rc in np.argwhere(cell == RED)}
    assert red_points, "each cell must contain a red arrow"
    comp = _component(red_points, next(iter(red_points)))
    assert len(comp) == len(red_points), "red arrow must be 4-connected"

    degrees = {point: sum(nbr in red_points for nbr in _neighbors4(point)) for point in red_points}
    junctions = [point for point, deg in degrees.items() if deg == 3]
    leaves = [point for point, deg in degrees.items() if deg == 1]
    middles = [point for point, deg in degrees.items() if deg == 2]
    assert len(junctions) == 1, "arrow must have one branching tip pixel"
    assert len(leaves) == 3, "arrow must have exactly three leaf pixels"
    assert len(middles) + len(leaves) + len(junctions) == len(red_points)

    junction = junctions[0]
    head_pair: tuple[tuple[int, int], tuple[int, int]] | None = None
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            a = leaves[i]
            b = leaves[j]
            va = (a[0] - junction[0], a[1] - junction[1])
            vb = (b[0] - junction[0], b[1] - junction[1])
            if abs(va[0]) + abs(va[1]) != 1:
                continue
            if abs(vb[0]) + abs(vb[1]) != 1:
                continue
            if va[0] + vb[0] == 0 and va[1] + vb[1] == 0:
                head_pair = (a, b)
                break
        if head_pair is not None:
            break
    assert head_pair is not None, "arrow must expose a symmetric two-pixel head"

    head_set = set(head_pair)
    tail = next(point for point in leaves if point not in head_set)
    dr = junction[0] - tail[0]
    dc = junction[1] - tail[1]
    assert dr == 0 or dc == 0, "stem must be straight"
    length = abs(dr) + abs(dc)
    assert 1 <= length <= 3, "stem length must be in 1..3"

    step = (
        0 if dr == 0 else dr // abs(dr),
        0 if dc == 0 else dc // abs(dc),
    )
    cursor = tail
    for _ in range(length):
        cursor = (cursor[0] + step[0], cursor[1] + step[1])
        assert cursor in red_points
    assert cursor == junction, "tail-to-tip path must end at the branching pixel"

    perp_a = (-step[1], step[0])
    perp_b = (step[1], -step[0])
    expected_heads = {
        (junction[0] + perp_a[0], junction[1] + perp_a[1]),
        (junction[0] + perp_b[0], junction[1] + perp_b[1]),
    }
    assert expected_heads == head_set, "head pixels must sit on the tip's perpendicular neighbors"

    direction = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}[step]
    return {"direction": direction, "length": length}


def _axis_spec(shape: tuple[int, int], direction: str, length: int) -> tuple[str, int]:
    if direction in ("up", "down"):
        sign = -1 if direction == "up" else 1
        axis2 = (shape[0] - 1) + 2 * sign * length
        return "horizontal", axis2

    sign = -1 if direction == "left" else 1
    axis2 = (shape[1] - 1) + 2 * sign * length
    return "vertical", axis2


def _mirror_point(
    row: int,
    col: int,
    shape: tuple[int, int],
    axis_mode: str,
    axis2: int,
) -> tuple[int, int]:
    if axis_mode == "horizontal":
        return axis2 - row, col
    return row, axis2 - col


def _arrow_template(direction: str, length: int) -> set[tuple[int, int]]:
    if direction == "right":
        return {(1, k) for k in range(length + 1)} | {(0, length), (2, length)}
    if direction == "left":
        return {(1, k) for k in range(length + 1)} | {(0, 0), (2, 0)}
    if direction == "down":
        return {(k, 1) for k in range(length + 1)} | {(length, 0), (length, 2)}
    return {(k, 1) for k in range(length + 1)} | {(0, 0), (0, 2)}


def _bbox_shape(points: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return max(rows) + 1, max(cols) + 1


def _max_length_for_cell(height: int, width: int, direction: str) -> int:
    axis_dim = height if direction in ("up", "down") else width
    return min(3, (axis_dim - 2) // 2)


def _orbit_catalog(
    shape: tuple[int, int],
    axis_mode: str,
    axis2: int,
    blocked: set[tuple[int, int]],
) -> tuple[list[tuple[tuple[int, int], ...]], list[tuple[tuple[int, int], ...]], list[tuple[tuple[int, int], ...]]]:
    height, width = shape
    good: list[tuple[tuple[int, int], ...]] = []
    edge: list[tuple[tuple[int, int], ...]] = []
    fixed: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[int, int]] = set()

    for row in range(height):
        for col in range(width):
            point = (row, col)
            if point in blocked or point in seen:
                continue

            mirror = _mirror_point(row, col, shape, axis_mode, axis2)
            if 0 <= mirror[0] < height and 0 <= mirror[1] < width:
                if mirror in blocked:
                    seen.add(point)
                    continue
                if mirror == point:
                    fixed.append((point,))
                    seen.add(point)
                    continue
                orbit = tuple(sorted((point, mirror)))
                good.append(orbit)
                seen.add(point)
                seen.add(mirror)
                continue

            edge.append((point,))
            seen.add(point)

    return good, edge, fixed


def _sample_table_sizes(rng: random.Random) -> tuple[list[int], list[int]]:
    while True:
        row_count, col_count = rng.choice(TABLE_LAYOUTS)
        row_heights = [rng.choice(CELL_SIZE_CHOICES) for _ in range(row_count)]
        col_widths = [rng.choice(CELL_SIZE_CHOICES) for _ in range(col_count)]
        total_h = sum(row_heights) + row_count + 1
        total_w = sum(col_widths) + col_count + 1
        if total_h <= 30 and total_w <= 30:
            return row_heights, col_widths


def _build_lattice(row_heights: list[int], col_widths: list[int]) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    total_h = sum(row_heights) + len(row_heights) + 1
    total_w = sum(col_widths) + len(col_widths) + 1
    grid = np.full((total_h, total_w), BLACK, dtype=int)

    row_bands: list[tuple[int, int]] = []
    cursor = 0
    grid[cursor, :] = GREY
    for height in row_heights:
        r0 = cursor + 1
        r1 = r0 + height - 1
        row_bands.append((r0, r1))
        cursor = r1 + 1
        grid[cursor, :] = GREY

    col_bands: list[tuple[int, int]] = []
    cursor = 0
    grid[:, cursor] = GREY
    for width in col_widths:
        c0 = cursor + 1
        c1 = c0 + width - 1
        col_bands.append((c0, c1))
        cursor = c1 + 1
        grid[:, cursor] = GREY

    return grid, row_bands, col_bands


def _build_cell(height: int, width: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(MAX_CELL_ATTEMPTS):
        direction = rng.choice(DIRECTIONS)
        max_length = _max_length_for_cell(height, width, direction)
        if max_length < 1:
            continue
        length = rng.randint(1, max_length)
        arrow_template = _arrow_template(direction, length)
        arrow_h, arrow_w = _bbox_shape(arrow_template)
        top = rng.randint(0, height - arrow_h)
        left = rng.randint(0, width - arrow_w)
        arrow_coords = {(top + r, left + c) for r, c in arrow_template}

        axis_mode, axis2 = _axis_spec((height, width), direction, length)
        good, edge, fixed = _orbit_catalog((height, width), axis_mode, axis2, arrow_coords)
        if not good:
            continue

        available = len(good) + len(edge) + len(fixed)
        lower = 2 if available >= 2 else 1
        goal = rng.randint(lower, min(5, available))

        chosen: list[tuple[tuple[int, int], ...]] = []
        used: set[tuple[tuple[int, int], ...]] = set()

        def add_random(pool: list[tuple[tuple[int, int], ...]]) -> None:
            candidates = [orbit for orbit in pool if orbit not in used]
            if not candidates:
                return
            orbit = rng.choice(candidates)
            chosen.append(orbit)
            used.add(orbit)

        add_random(good)
        if edge and rng.random() < 0.45 and len(chosen) < goal:
            add_random(edge)
        if fixed and rng.random() < 0.35 and len(chosen) < goal:
            add_random(fixed)

        remaining = [orbit for orbit in good + edge + fixed if orbit not in used]
        rng.shuffle(remaining)
        for orbit in remaining:
            if len(chosen) >= goal:
                break
            chosen.append(orbit)
            used.add(orbit)

        cell = np.full((height, width), BLACK, dtype=int)
        for row, col in arrow_coords:
            cell[row, col] = RED

        palette = list(TARGET_COLORS)
        rng.shuffle(palette)
        for idx, orbit in enumerate(chosen):
            row, col = rng.choice(orbit)
            cell[row, col] = palette[idx % len(palette)]

        return cell

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_heights, col_widths = _sample_table_sizes(rng)
    grid, row_bands, col_bands = _build_lattice(row_heights, col_widths)

    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            cell = _build_cell(r1 - r0 + 1, c1 - c0 + 1, rng)
            if cell is None:
                return None
            grid[r0 : r1 + 1, c0 : c1 + 1] = cell

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(grid, out):
            continue
        assert grid.ndim == 2 and max(grid.shape) <= 30
        assert out.ndim == 2 and max(out.shape) <= 30
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
