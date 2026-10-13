from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

CELL_GRID_SIZE = 6
CELL_SIZE_CHOICES = (3, 4)
BASE_COLORS = (BLUE, RED, GREEN)
QUARTET_STARTS = {
    "left": (2, 0),
    "right": (2, 4),
    "top": (0, 2),
    "bottom": (4, 2),
}
QUARTET_FAMILIES = {
    "left": "horizontal",
    "right": "horizontal",
    "top": "vertical",
    "bottom": "vertical",
}
CARDINAL_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GEN_TRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _ranges_between(size: int, dividers: list[int]) -> list[tuple[int, int]]:
    points = [-1] + list(dividers) + [int(size)]
    return [
        (points[i] + 1, points[i + 1] - 1)
        for i in range(len(points) - 1)
        if points[i] + 1 <= points[i + 1] - 1
    ]


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    height, width = grid.shape
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        rows = [r for r in range(height) if np.all(grid[r, :] == color)]
        cols = [c for c in range(width) if np.all(grid[:, c] == color)]
        if not rows or not cols:
            continue
        mask = np.zeros_like(grid, dtype=bool)
        mask[rows, :] = True
        mask[:, cols] = True
        if np.array_equal(grid == color, mask):
            candidates.append((int(color), rows, cols))
    assert len(candidates) == 1, "expected exactly one divider color"
    return candidates[0]


def _cell_base_colors(
    grid: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    bg: int,
    divider: int,
) -> np.ndarray:
    cells = np.full((len(row_ranges), len(col_ranges)), -1, dtype=int)
    for cell_r, (r0, r1) in enumerate(row_ranges):
        for cell_c, (c0, c1) in enumerate(col_ranges):
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            vals = set(map(int, cell.ravel())) - {int(bg)}
            assert divider not in vals, "divider color must not appear inside cells"
            vals.discard(ORANGE)
            if vals:
                assert len(vals) == 1, "each occupied cell must have one base sprite color"
                cells[cell_r, cell_c] = vals.pop()
    return cells


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        part: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            part.append((cr, cc))
            for dr, dc in CARDINAL_STEPS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        parts.append(part)
    return parts


def _predict_missing_cell(
    grid: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    have: list[tuple[int, int]],
    missing: tuple[int, int],
) -> np.ndarray:
    missing_r, missing_c = missing
    predictions = []
    for cell_r, cell_c in have:
        r0, r1 = row_ranges[cell_r]
        c0, c1 = col_ranges[cell_c]
        pred = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        if cell_r != missing_r:
            pred = pred[::-1, :]
        if cell_c != missing_c:
            pred = pred[:, ::-1]
        predictions.append(pred)
    first = predictions[0]
    assert all(np.array_equal(first, pred) for pred in predictions[1:]), "reflections must agree"
    return first


def _complete_quartets(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    bg = _get_bg(grid)
    divider, divider_rows, divider_cols = _find_divider(grid, bg)
    row_ranges = _ranges_between(grid.shape[0], divider_rows)
    col_ranges = _ranges_between(grid.shape[1], divider_cols)
    assert len(row_ranges) >= 2 and len(col_ranges) >= 2

    cells = _cell_base_colors(grid, row_ranges, col_ranges, bg, divider)
    out = grid.copy()

    for color in sorted(set(map(int, cells.ravel())) - {-1}):
        for comp in _components(cells == color):
            assert len(comp) == 3, "each quartet group must show exactly three occupied cells"
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            assert max(rs) - min(rs) == 1 and max(cs) - min(cs) == 1, "group must fit a 2x2 block"
            block = {(r, c) for r in range(min(rs), max(rs) + 1) for c in range(min(cs), max(cs) + 1)}
            have = set(comp)
            missing_cells = block - have
            assert len(missing_cells) == 1
            missing = next(iter(missing_cells))
            mr, mc = missing
            assert cells[mr, mc] == -1

            pred = _predict_missing_cell(out, row_ranges, col_ranges, list(have), missing)
            r0, r1 = row_ranges[mr]
            c0, c1 = col_ranges[mc]
            assert np.all(out[r0 : r1 + 1, c0 : c1 + 1] == bg), "missing cell must be empty"
            out[r0 : r1 + 1, c0 : c1 + 1] = pred

    return out, row_ranges, col_ranges


def _collect_markers(
    grid: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    markers: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for cell_r, (r0, r1) in enumerate(row_ranges):
        for cell_c, (c0, c1) in enumerate(col_ranges):
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            for lr, lc in np.argwhere(cell == ORANGE):
                markers[(int(lr), int(lc))].append((cell_r, cell_c))
    return markers


def _paint_local(
    grid: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    cell_r: int,
    cell_c: int,
    offset: tuple[int, int],
    color: int,
) -> None:
    r = row_ranges[cell_r][0] + offset[0]
    c = col_ranges[cell_c][0] + offset[1]
    grid[r, c] = int(color)


def _connect_markers(
    grid: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
) -> tuple[np.ndarray, dict[str, set[tuple[int, int, int, int]]]]:
    markers = _collect_markers(grid, row_ranges, col_ranges)
    out = grid.copy()
    horizontal_offsets: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    debug = {
        "horizontal_draws": set(),
        "vertical_draws": set(),
        "yellow_draws": set(),
    }

    grouped: dict[tuple[int, int], tuple[dict[int, list[int]], dict[int, list[int]]]] = {}
    for offset, cells in markers.items():
        by_row: dict[int, list[int]] = defaultdict(list)
        by_col: dict[int, list[int]] = defaultdict(list)
        for cell_r, cell_c in cells:
            by_row[cell_r].append(cell_c)
            by_col[cell_c].append(cell_r)
        grouped[offset] = (by_row, by_col)

    for offset, (by_row, _by_col) in grouped.items():
        for cell_r, cols in by_row.items():
            cols = sorted(set(cols))
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    for mid_c in range(cols[i] + 1, cols[j]):
                        _paint_local(out, row_ranges, col_ranges, cell_r, mid_c, offset, ORANGE)
                        horizontal_offsets[(cell_r, mid_c)].add(offset)
                        debug["horizontal_draws"].add((cell_r, mid_c, offset[0], offset[1]))

    for offset, (_by_row, by_col) in grouped.items():
        for cell_c, rows in by_col.items():
            rows = sorted(set(rows))
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    for mid_r in range(rows[i] + 1, rows[j]):
                        crosses = any(other != offset for other in horizontal_offsets[(mid_r, cell_c)])
                        color = YELLOW if crosses else ORANGE
                        _paint_local(out, row_ranges, col_ranges, mid_r, cell_c, offset, color)
                        debug["vertical_draws"].add((mid_r, cell_c, offset[0], offset[1]))
                        if color == YELLOW:
                            debug["yellow_draws"].add((mid_r, cell_c, offset[0], offset[1]))

    return out, debug


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each missing reflected sprite, then use the revealed orange endpoints to draw aligned connections and mark crossings.

    Input structure:
    - The background is black, and one grey color forms full-width/full-height divider lines that split the canvas into equal rectangular cells.
    - Some cells are empty, while occupied cells contain a small connected monochrome sprite in blue, red, or green, optionally with one orange marker pixel embedded in the sprite.
    - Each monochrome sprite group occupies exactly three cells of one 2x2 cell block; the fourth cell of that block is empty, and the visible three cells are horizontal/vertical reflection variants of one local sprite.

    Rule:
    - For every three-of-four 2x2 block, infer the missing cell by reflecting the visible cell-local sprite(s) into the empty position, preserving both the base sprite color and any orange marker pixel.
    - After all missing cells are restored, collect every orange marker and index it by its cell row, cell column, and local offset inside its cell.
    - For each local offset, connect every pair of markers that shares a cell row by painting orange pixels at the same local offset in every intermediate cell; do the same for every pair that shares a cell column.
    - Horizontal connections are drawn first. When a later vertical connection paints a cell that already contains a horizontal connection at a different local offset, recolor that vertical connection pixel yellow.

    Key insight:
    - The empty lattice cells hide additional orange endpoints, so the straight-line connection pattern is only recoverable after completing each 2x2 reflection quartet.
    """
    completed, row_ranges, col_ranges = _complete_quartets(np.asarray(grid, dtype=int))
    out, _debug = _connect_markers(completed, row_ranges, col_ranges)
    return out


def _orbit_representatives(cell_size: int) -> list[tuple[int, int]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    reps: list[tuple[int, int]] = []
    for r in range(cell_size):
        for c in range(cell_size):
            orbit = tuple(
                sorted(
                    {
                        (r, c),
                        (r, cell_size - 1 - c),
                        (cell_size - 1 - r, c),
                        (cell_size - 1 - r, cell_size - 1 - c),
                    }
                )
            )
            if len(orbit) <= 1 or orbit in seen:
                continue
            seen.add(orbit)
            reps.append(min(orbit))
    return reps


def _orbit_members(cell_size: int, rep: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = rep
    return sorted(
        {
            (r, c),
            (r, cell_size - 1 - c),
            (cell_size - 1 - r, c),
            (cell_size - 1 - r, cell_size - 1 - c),
        }
    )


def _connected(coords: set[tuple[int, int]]) -> bool:
    start = next(iter(coords))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in CARDINAL_STEPS:
            nxt = (r + dr, c + dc)
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(coords)


def _generate_sprite(cell_size: int, base_color: int, marker: tuple[int, int]) -> np.ndarray:
    for _ in range(300):
        count = random.randint(2, min(4, cell_size * cell_size))
        coords = {marker}
        while len(coords) < count:
            frontier = []
            for r, c in coords:
                for dr, dc in CARDINAL_STEPS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < cell_size and 0 <= nc < cell_size and (nr, nc) not in coords:
                        frontier.append((nr, nc))
            if not frontier:
                break
            coords.add(random.choice(frontier))
        if len(coords) != count or not _connected(coords):
            continue

        mask = np.zeros((cell_size, cell_size), dtype=bool)
        for r, c in coords:
            mask[r, c] = True
        if np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.fliplr(mask)):
            continue

        sprite = np.zeros((cell_size, cell_size), dtype=int)
        for r, c in coords:
            sprite[r, c] = int(base_color)
        sprite[marker] = ORANGE
        return sprite
    raise RuntimeError("failed to generate asymmetric connected sprite")


def _build_lattice(cell_size: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    total = CELL_GRID_SIZE * cell_size + (CELL_GRID_SIZE - 1)
    grid = np.zeros((total, total), dtype=int)
    row_ranges = []
    col_ranges = []
    pos = 0
    for i in range(CELL_GRID_SIZE):
        row_ranges.append((pos, pos + cell_size - 1))
        col_ranges.append((pos, pos + cell_size - 1))
        pos += cell_size
        if i != CELL_GRID_SIZE - 1:
            grid[pos, :] = GREY
            grid[:, pos] = GREY
            pos += 1
    return grid, row_ranges, col_ranges


def _variant(sprite: np.ndarray, local_r: int, local_c: int) -> np.ndarray:
    out = sprite.copy()
    if local_r == 1:
        out = out[::-1, :]
    if local_c == 1:
        out = out[:, ::-1]
    return out


def _paste_cell(
    grid: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    cell_r: int,
    cell_c: int,
    sprite: np.ndarray,
) -> None:
    r0, r1 = row_ranges[cell_r]
    c0, c1 = col_ranges[cell_c]
    grid[r0 : r1 + 1, c0 : c1 + 1] = sprite


def _marker_specs(cell_size: int) -> tuple[tuple[int, int], tuple[int, int]]:
    reps = _orbit_representatives(cell_size)
    assert len(reps) >= 2
    rep_a, rep_b = random.sample(reps, 2)
    marker_a = random.choice(_orbit_members(cell_size, rep_a))
    marker_b = random.choice(_orbit_members(cell_size, rep_b))
    return marker_a, marker_b


def _hidden_markers(
    input_grid: np.ndarray,
    completed_grid: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
) -> set[tuple[int, int, int, int]]:
    hidden = set()
    for cell_r, (r0, r1) in enumerate(row_ranges):
        for cell_c, (c0, c1) in enumerate(col_ranges):
            before = input_grid[r0 : r1 + 1, c0 : c1 + 1]
            after = completed_grid[r0 : r1 + 1, c0 : c1 + 1]
            before_pts = {tuple(map(int, pt)) for pt in np.argwhere(before == ORANGE)}
            for lr, lc in np.argwhere(after == ORANGE):
                key = (cell_r, cell_c, int(lr), int(lc))
                if (int(lr), int(lc)) not in before_pts:
                    hidden.add(key)
    return hidden


def _markers_that_participate(
    completed_grid: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
) -> set[tuple[int, int, int, int]]:
    markers = _collect_markers(completed_grid, row_ranges, col_ranges)
    participating = set()
    for offset, cells in markers.items():
        for cell_r, cell_c in cells:
            if any(other_r == cell_r and other_c != cell_c for other_r, other_c in cells) or any(
                other_c == cell_c and other_r != cell_r for other_r, other_c in cells
            ):
                participating.add((cell_r, cell_c, offset[0], offset[1]))
    return participating


def _generate_instance() -> tuple[np.ndarray, np.ndarray]:
    cell_size = random.choice(CELL_SIZE_CHOICES)
    marker_a, marker_b = _marker_specs(cell_size)
    palette = [random.choice(BASE_COLORS) for _ in range(4)]
    if len(set(palette)) == 1:
        palette[random.randrange(4)] = random.choice([c for c in BASE_COLORS if c != palette[0]])

    completed, row_ranges, col_ranges = _build_lattice(cell_size)
    input_grid = completed.copy()

    for idx, name in enumerate(("left", "right", "top", "bottom")):
        start_r, start_c = QUARTET_STARTS[name]
        marker = marker_a if QUARTET_FAMILIES[name] == "horizontal" else marker_b
        sprite = _generate_sprite(cell_size, palette[idx], marker)
        missing_dr, missing_dc = random.choice(((0, 0), (0, 1), (1, 0), (1, 1)))
        for dr in range(2):
            for dc in range(2):
                variant = _variant(sprite, dr, dc)
                cell_r = start_r + dr
                cell_c = start_c + dc
                _paste_cell(completed, row_ranges, col_ranges, cell_r, cell_c, variant)
                if (dr, dc) != (missing_dr, missing_dc):
                    _paste_cell(input_grid, row_ranges, col_ranges, cell_r, cell_c, variant)

    return input_grid, completed


def _validate_instance(input_grid: np.ndarray, completed_grid: np.ndarray) -> bool:
    completed_from_input, row_ranges, col_ranges = _complete_quartets(input_grid)
    if not np.array_equal(completed_from_input, completed_grid):
        return False

    expected, debug = _connect_markers(completed_grid, row_ranges, col_ranges)
    actual = main(input_grid)
    if not np.array_equal(actual, expected):
        return False

    visible_only, visible_debug = _connect_markers(input_grid, row_ranges, col_ranges)
    if np.array_equal(visible_only, expected):
        return False

    if not debug["horizontal_draws"] or not debug["vertical_draws"] or not debug["yellow_draws"]:
        return False

    hidden = _hidden_markers(input_grid, completed_grid, row_ranges, col_ranges)
    if not hidden:
        return False

    participating = _markers_that_participate(completed_grid, row_ranges, col_ranges)
    if not (hidden & participating):
        return False

    if len({(lr, lc) for _, _, lr, lc in hidden & participating}) == 0:
        return False

    if len({(lr, lc) for _, _, lr, lc in {(r, c, lr, lc) for r, c, lr, lc in debug["horizontal_draws"] | debug["vertical_draws"]}}) < 2:
        return False

    if len(debug["yellow_draws"]) < 2:
        return False

    if np.sum(completed_grid == ORANGE) <= np.sum(input_grid == ORANGE):
        return False

    if len(visible_debug["yellow_draws"]) >= len(debug["yellow_draws"]):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_TRIES):
        input_grid, completed_grid = _generate_instance()
        if _validate_instance(input_grid, completed_grid):
            return input_grid
    raise RuntimeError("failed to generate a valid puzzle instance")
