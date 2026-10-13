from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


# ARC palette constants.
BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7
AZURE = 8

BAR_COLORS = (RED, GREEN, ORANGE, AZURE)
HORIZONTAL = "horizontal"
VERTICAL = "vertical"

# Generator bounds.
GRID_MIN = 14
GRID_MAX = 20
SEED_LEN_RANGE = (2, 3)
REFLECTION_RANGE = (1, 3)
BAR_LENGTH_RANGE = (3, 6)
TOTAL_BARS_RANGE = (3, 6)
PRE_REFLECTION_SEGMENT_RANGE = (1, 5)
PATH_NEW_CELLS_RANGE = (6, 18)
MAX_GENERATION_ATTEMPTS = 600
BAR_PLACEMENT_ATTEMPTS = 80

KEY_CELLS = ((0, 0), (0, 1))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    neighbors = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    comps: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _diagonal_kind(cells: set[tuple[int, int]]) -> str | None:
    if len(cells) < 2:
        return None
    if len({r - c for r, c in cells}) == 1:
        rows = sorted(r for r, _ in cells)
        cols = sorted(c for _, c in cells)
        if rows == list(range(rows[0], rows[-1] + 1)) and cols == list(range(cols[0], cols[-1] + 1)):
            return "down_right"
        return None
    if len({r + c for r, c in cells}) == 1:
        rows = sorted(r for r, _ in cells)
        cols = sorted(c for _, c in cells)
        if rows == list(range(rows[0], rows[-1] + 1)) and cols == list(range(cols[0], cols[-1] + 1)):
            return "down_left"
    return None


def _find_seed(grid: np.ndarray) -> set[tuple[int, int]]:
    blue_mask = grid == BLUE
    blue_mask[0, 0] = False
    comps = _components(blue_mask, connectivity=8)
    candidates = [set(comp) for comp in comps if _diagonal_kind(set(comp)) is not None]
    assert len(candidates) == 1, "expected one blue diagonal seed"
    return candidates[0]


def _initial_state(
    shape: tuple[int, int], seed: set[tuple[int, int]]
) -> tuple[int, int, int, int]:
    kind = _diagonal_kind(seed)
    assert kind is not None, "seed must be a contiguous diagonal segment"
    if kind == "down_right":
        first = min(seed)
        last = max(seed)
        endpoints = ((first, (-1, -1)), (last, (1, 1)))
    else:
        top = min(seed)
        bottom = max(seed)
        endpoints = ((top, (-1, 1)), (bottom, (1, -1)))

    candidates = []
    for (r, c), (dr, dc) in endpoints:
        if _in_bounds(shape, r + dr, c + dc):
            candidates.append((r, c, dr, dc))
    assert len(candidates) == 1, "seed must have exactly one in-bounds extension"
    return candidates[0]


def _component_orientation(cells: list[tuple[int, int]]) -> str:
    rows = sorted({r for r, _ in cells})
    cols = sorted({c for _, c in cells})
    if len(rows) == 1:
        ordered = sorted(c for _, c in cells)
        assert ordered == list(range(ordered[0], ordered[-1] + 1)), "bar must be contiguous"
        return HORIZONTAL
    if len(cols) == 1:
        ordered = sorted(r for r, _ in cells)
        assert ordered == list(range(ordered[0], ordered[-1] + 1)), "bar must be contiguous"
        return VERTICAL
    raise AssertionError("bar must be horizontal or vertical")


def _mirror_orientations(grid: np.ndarray, mirror_color: int) -> dict[tuple[int, int], str]:
    mirror_map: dict[tuple[int, int], str] = {}
    mask = grid == mirror_color
    for comp in _components(mask, connectivity=4):
        cell_set = set(comp)
        if cell_set == {(0, 1)}:
            continue
        orientation = _component_orientation(comp)
        for cell in comp:
            mirror_map[cell] = orientation
    return mirror_map


def _reflect(orientation: str, dr: int, dc: int) -> tuple[int, int]:
    if orientation == HORIZONTAL:
        return -dr, dc
    return dr, -dc


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 2, "expected a 2D grid with a top-left key"
    assert int(g[0, 0]) == BLUE, "top-left key must start with blue"

    mirror_color = int(g[0, 1])
    seed = _find_seed(g)
    r, c, dr, dc = _initial_state(g.shape, seed)
    mirrors = _mirror_orientations(g, mirror_color)

    out = g.copy()
    seen: set[tuple[int, int, int, int]] = set()
    key_cells = set(KEY_CELLS)

    while True:
        state = (r, c, dr, dc)
        assert state not in seen, "ray must not loop"
        seen.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(g.shape, nr, nc):
            break

        if (nr, nc) in mirrors:
            dr, dc = _reflect(mirrors[(nr, nc)], dr, dc)
            continue

        if (nr, nc) not in key_cells:
            out[nr, nc] = BLUE
        r, c = nr, nc

    return out


def _key_reserved(shape: tuple[int, int]) -> set[tuple[int, int]]:
    reserved: set[tuple[int, int]] = set()
    for kr, kc in KEY_CELLS:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = kr + dr, kc + dc
                if _in_bounds(shape, nr, nc):
                    reserved.add((nr, nc))
    return reserved


def _build_seed_cells(
    shape: tuple[int, int], r0: int, c0: int, dr: int, dc: int, seed_len: int
) -> list[tuple[int, int]] | None:
    cells = []
    for step in range(seed_len):
        nr, nc = r0 + step * dr, c0 + step * dc
        if not _in_bounds(shape, nr, nc):
            return None
        cells.append((nr, nc))
    return cells


def _sample_start_state(shape: tuple[int, int], seed_len: int) -> tuple[int, int, int, int]:
    height, width = shape
    choices: list[tuple[int, int, int, int]] = []

    for dc in (-1, 1):
        c_min = max(3, seed_len - 1) if dc == -1 else 3
        c_max = width - 1 if dc == -1 else width - seed_len
        if c_min <= c_max:
            choices.append((0, random.randint(c_min, c_max), 1, dc))

    for dc in (-1, 1):
        c_min = max(0, seed_len - 1) if dc == -1 else 0
        c_max = width - 1 if dc == -1 else width - seed_len
        if c_min <= c_max:
            choices.append((height - 1, random.randint(c_min, c_max), -1, dc))

    for dr in (-1, 1):
        r_min = max(3, seed_len - 1) if dr == -1 else 3
        r_max = height - 1 if dr == -1 else height - seed_len
        if r_min <= r_max:
            choices.append((random.randint(r_min, r_max), 0, dr, 1))

    for dr in (-1, 1):
        r_min = max(0, seed_len - 1) if dr == -1 else 0
        r_max = height - 1 if dr == -1 else height - seed_len
        if r_min <= r_max:
            choices.append((random.randint(r_min, r_max), width - 1, dr, -1))

    return random.choice(choices)


def _walk_segment(
    shape: tuple[int, int],
    r: int,
    c: int,
    dr: int,
    dc: int,
    length: int,
    blocked: set[tuple[int, int]],
    path_cells: list[tuple[int, int]],
) -> tuple[int, int] | None:
    for _ in range(length):
        nr, nc = r + dr, c + dc
        if not _in_bounds(shape, nr, nc) or (nr, nc) in blocked:
            return None
        blocked.add((nr, nc))
        path_cells.append((nr, nc))
        r, c = nr, nc
    return r, c


def _run_to_edge(
    shape: tuple[int, int],
    r: int,
    c: int,
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    cells: list[tuple[int, int]] = []
    while True:
        nr, nc = r + dr, c + dc
        if not _in_bounds(shape, nr, nc):
            return cells if cells else None
        if (nr, nc) in blocked:
            return None
        cells.append((nr, nc))
        r, c = nr, nc


def _bar_cells_through(
    shape: tuple[int, int],
    orientation: str,
    anchor: tuple[int, int],
    length: int,
) -> list[tuple[int, int]] | None:
    height, width = shape
    ar, ac = anchor
    if orientation == HORIZONTAL:
        c_min = max(0, ac - length + 1)
        c_max = min(ac, width - length)
        if c_min > c_max:
            return None
        start = random.randint(c_min, c_max)
        return [(ar, start + offset) for offset in range(length)]

    r_min = max(0, ar - length + 1)
    r_max = min(ar, height - length)
    if r_min > r_max:
        return None
    start = random.randint(r_min, r_max)
    return [(start + offset, ac) for offset in range(length)]


def _random_bar_cells(
    shape: tuple[int, int], orientation: str, length: int
) -> list[tuple[int, int]] | None:
    height, width = shape
    if orientation == HORIZONTAL:
        if length > width:
            return None
        row = random.randint(0, height - 1)
        start = random.randint(0, width - length)
        return [(row, start + offset) for offset in range(length)]
    if length > height:
        return None
    col = random.randint(0, width - 1)
    start = random.randint(0, height - length)
    return [(start + offset, col) for offset in range(length)]


def _touches_same_color(
    cells: set[tuple[int, int]], existing: set[tuple[int, int]]
) -> bool:
    for r, c in cells:
        for dr, dc in NEIGHBORS4:
            neighbor = (r + dr, c + dc)
            if neighbor in existing and neighbor not in cells:
                return True
    return False


def _cells_valid(
    cells: list[tuple[int, int]],
    reserved: set[tuple[int, int]],
    fixed_occupied: set[tuple[int, int]],
    same_color_cells: set[tuple[int, int]],
    path_set: set[tuple[int, int]],
    allow_path_overlap: bool,
) -> bool:
    cell_set = set(cells)
    if cell_set & reserved:
        return False
    if cell_set & fixed_occupied:
        return False
    if _touches_same_color(cell_set, same_color_cells):
        return False
    if not allow_path_overlap and cell_set & path_set:
        return False
    return True


def _try_place_mirror(
    shape: tuple[int, int],
    r: int,
    c: int,
    dr: int,
    dc: int,
    reserved: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    fixed_occupied: set[tuple[int, int]],
    mirror_cells: set[tuple[int, int]],
) -> tuple[str, list[tuple[int, int]], int, int] | None:
    anchor = (r + dr, c + dc)
    if not _in_bounds(shape, *anchor) or anchor in blocked:
        return None

    orientations = [HORIZONTAL, VERTICAL]
    random.shuffle(orientations)
    for orientation in orientations:
        new_dr, new_dc = _reflect(orientation, dr, dc)
        next_after_bounce = (r + new_dr, c + new_dc)
        if not _in_bounds(shape, *next_after_bounce) or next_after_bounce in blocked:
            continue
        for _ in range(BAR_PLACEMENT_ATTEMPTS):
            length = random.randint(*BAR_LENGTH_RANGE)
            cells = _bar_cells_through(shape, orientation, anchor, length)
            if cells is None:
                continue
            if not _cells_valid(
                cells,
                reserved=reserved,
                fixed_occupied=fixed_occupied,
                same_color_cells=mirror_cells,
                path_set=set(),
                allow_path_overlap=False,
            ):
                continue
            return orientation, cells, new_dr, new_dc
    return None


def _place_bar(
    shape: tuple[int, int],
    color: int,
    reserved: set[tuple[int, int]],
    fixed_occupied: set[tuple[int, int]],
    same_color_cells: set[tuple[int, int]],
    path_cells: list[tuple[int, int]],
    allow_path_overlap: bool,
    must_cross_path: bool,
) -> list[tuple[int, int]] | None:
    path_set = set(path_cells)
    for _ in range(BAR_PLACEMENT_ATTEMPTS):
        orientation = random.choice((HORIZONTAL, VERTICAL))
        length = random.randint(*BAR_LENGTH_RANGE)
        if must_cross_path or (allow_path_overlap and path_cells and random.random() < 0.5):
            anchor = random.choice(path_cells)
            cells = _bar_cells_through(shape, orientation, anchor, length)
        else:
            cells = _random_bar_cells(shape, orientation, length)
        if cells is None:
            continue
        cell_set = set(cells)
        if must_cross_path and not (cell_set & path_set):
            continue
        if not allow_path_overlap and cell_set & path_set:
            continue
        if not _cells_valid(
            cells,
            reserved=reserved,
            fixed_occupied=fixed_occupied,
            same_color_cells=same_color_cells,
            path_set=path_set,
            allow_path_overlap=allow_path_overlap,
        ):
            continue
        return cells
    return None


def _render_grid(
    shape: tuple[int, int],
    seed_cells: list[tuple[int, int]],
    mirror_color: int,
    mirror_bars: list[list[tuple[int, int]]],
    decoy_bars: list[tuple[int, list[tuple[int, int]]]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    grid[0, 0] = BLUE
    grid[0, 1] = mirror_color
    for r, c in seed_cells:
        grid[r, c] = BLUE
    for bar in mirror_bars:
        for r, c in bar:
            grid[r, c] = mirror_color
    for color, bar in decoy_bars:
        for r, c in bar:
            grid[r, c] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
    reserved = _key_reserved(shape)

    seed_len = random.randint(*SEED_LEN_RANGE)
    r0, c0, dr, dc = _sample_start_state(shape, seed_len)
    if _in_bounds(shape, r0 - dr, c0 - dc):
        return None
    seed_cells = _build_seed_cells(shape, r0, c0, dr, dc, seed_len)
    if seed_cells is None or any(cell in reserved for cell in seed_cells):
        return None

    n_colors = random.choice((2, 3))
    selected_colors = random.sample(BAR_COLORS, n_colors)
    mirror_color = random.choice(selected_colors)
    other_colors = [color for color in selected_colors if color != mirror_color]
    n_reflections = random.randint(*REFLECTION_RANGE)

    min_bars = max(TOTAL_BARS_RANGE[0], n_reflections + len(other_colors))
    if min_bars > TOTAL_BARS_RANGE[1]:
        return None
    total_bars = random.randint(min_bars, TOTAL_BARS_RANGE[1])

    path_cells: list[tuple[int, int]] = []
    blocked = set(reserved)
    blocked.update(seed_cells)
    fixed_occupied = set(KEY_CELLS)
    fixed_occupied.update(seed_cells)
    mirror_bars: list[list[tuple[int, int]]] = []
    bars_by_color: dict[int, set[tuple[int, int]]] = defaultdict(set)

    r, c = seed_cells[-1]
    cur_dr, cur_dc = dr, dc
    for _ in range(n_reflections):
        seg_len = random.randint(*PRE_REFLECTION_SEGMENT_RANGE)
        endpoint = _walk_segment(shape, r, c, cur_dr, cur_dc, seg_len, blocked, path_cells)
        if endpoint is None:
            return None
        r, c = endpoint

        placed = _try_place_mirror(
            shape,
            r,
            c,
            cur_dr,
            cur_dc,
            reserved=reserved,
            blocked=blocked,
            fixed_occupied=fixed_occupied,
            mirror_cells=bars_by_color[mirror_color],
        )
        if placed is None:
            return None
        _orientation, cells, cur_dr, cur_dc = placed
        mirror_bars.append(cells)
        fixed_occupied.update(cells)
        blocked.update(cells)
        bars_by_color[mirror_color].update(cells)

    final_cells = _run_to_edge(shape, r, c, cur_dr, cur_dc, blocked)
    if final_cells is None:
        return None
    path_cells.extend(final_cells)
    blocked.update(final_cells)

    if not (PATH_NEW_CELLS_RANGE[0] <= len(path_cells) <= PATH_NEW_CELLS_RANGE[1]):
        return None

    decoy_bars: list[tuple[int, list[tuple[int, int]]]] = []
    crossed_decoy = False
    for idx, color in enumerate(other_colors):
        cells = _place_bar(
            shape,
            color=color,
            reserved=reserved,
            fixed_occupied=fixed_occupied,
            same_color_cells=bars_by_color[color],
            path_cells=path_cells,
            allow_path_overlap=True,
            must_cross_path=idx == 0,
        )
        if cells is None:
            return None
        decoy_bars.append((color, cells))
        fixed_occupied.update(cells)
        bars_by_color[color].update(cells)
        crossed_decoy = crossed_decoy or bool(set(cells) & set(path_cells))

    extra_bars = total_bars - (len(mirror_bars) + len(decoy_bars))
    for _ in range(extra_bars):
        color = random.choice(selected_colors)
        cells = _place_bar(
            shape,
            color=color,
            reserved=reserved,
            fixed_occupied=fixed_occupied,
            same_color_cells=bars_by_color[color],
            path_cells=path_cells,
            allow_path_overlap=color != mirror_color,
            must_cross_path=False,
        )
        if cells is None:
            return None
        if color == mirror_color:
            mirror_bars.append(cells)
        else:
            decoy_bars.append((color, cells))
            crossed_decoy = crossed_decoy or bool(set(cells) & set(path_cells))
        fixed_occupied.update(cells)
        bars_by_color[color].update(cells)

    if not crossed_decoy:
        return None

    grid = _render_grid(shape, seed_cells, mirror_color, mirror_bars, decoy_bars)
    expected = grid.copy()
    for rr, cc in path_cells:
        expected[rr, cc] = BLUE

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None

    used_bar_colors = {mirror_color, *[color for color, _ in decoy_bars]}
    if len(used_bar_colors) != n_colors:
        return None
    if not (TOTAL_BARS_RANGE[0] <= len(mirror_bars) + len(decoy_bars) <= TOTAL_BARS_RANGE[1]):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
