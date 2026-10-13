from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
ORANGE = 7

DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GRID_ROWS_RANGE = (13, 18)
GRID_COLS_RANGE = (13, 18)
SEED_LENGTH_RANGE = (2, 3)
MIRROR_COUNT_RANGE = (2, 4)
MIRROR_LENGTH_RANGE = (3, 5)
HIT_MIRROR_WEIGHTS = {1: 1, 2: 4, 3: 2}
SEGMENT_LENGTH_RANGE = (1, 5)
MAX_GENERATION_ATTEMPTS = 1200


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _neighbors(r: int, c: int, connectivity: int) -> tuple[tuple[int, int], ...]:
    dirs = DIR4 if int(connectivity) == 4 else DIR8
    return tuple((int(r + dr), int(c + dc)) for dr, dc in dirs)


def _components_of_color(grid: np.ndarray, color: int, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != int(color) or (r, c) in seen:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen.add((r, c))
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, connectivity):
                    if (
                        _in_bounds(grid.shape, nr, nc)
                        and (nr, nc) not in seen
                        and int(grid[nr, nc]) == int(color)
                    ):
                        seen.add((nr, nc))
                        q.append((nr, nc))
            comps.append(sorted(comp))
    return comps


def _zero_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    regions: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLACK or (r, c) in seen:
                continue
            region: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen.add((r, c))
            while q:
                cr, cc = q.popleft()
                region.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, 4):
                    if (
                        _in_bounds(grid.shape, nr, nc)
                        and (nr, nc) not in seen
                        and int(grid[nr, nc]) == BLACK
                    ):
                        seen.add((nr, nc))
                        q.append((nr, nc))
            regions.append(sorted(region))
    return regions


def _is_diagonal_segment(coords: list[tuple[int, int]]) -> bool:
    pts = set(coords)
    if len(pts) != len(coords) or len(coords) < 2:
        return False
    rows = sorted(r for r, _c in pts)
    cols = sorted(c for _r, c in pts)
    if len({r - c for r, c in pts}) == 1 or len({r + c for r, c in pts}) == 1:
        return (
            rows == list(range(rows[0], rows[-1] + 1))
            and cols == list(range(cols[0], cols[-1] + 1))
            and len(pts) == rows[-1] - rows[0] + 1
        )
    return False


def _find_seed(grid: np.ndarray) -> list[tuple[int, int]]:
    candidates = [
        comp for comp in _components_of_color(grid, BLUE, 8) if _is_diagonal_segment(comp)
    ]
    assert len(candidates) == 1, "expected one diagonal blue seed"
    return candidates[0]


def _seed_state(shape: tuple[int, int], seed: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = set(seed)
    if len({r - c for r, c in pts}) == 1:
        first = min(pts)
        last = max(pts)
        endpoints = ((first, (-1, -1)), (last, (1, 1)))
    else:
        top = min(pts)
        bottom = max(pts)
        endpoints = ((top, (-1, 1)), (bottom, (1, -1)))

    candidates = []
    for (r, c), (dr, dc) in endpoints:
        if _in_bounds(shape, r + dr, c + dc):
            candidates.append((r, c, dr, dc))
    assert len(candidates) == 1, "seed must have a unique inward extension"
    return candidates[0]


def _mirror_bars(grid: np.ndarray) -> tuple[dict[tuple[int, int], str], list[tuple[str, list[tuple[int, int]]]]]:
    mapping: dict[tuple[int, int], str] = {}
    bars: list[tuple[str, list[tuple[int, int]]]] = []
    for comp in _components_of_color(grid, RED, 4):
        rows = sorted({r for r, _c in comp})
        cols = sorted({c for _r, c in comp})
        if len(rows) == 1 and cols == list(range(cols[0], cols[-1] + 1)):
            orient = "horizontal"
        elif len(cols) == 1 and rows == list(range(rows[0], rows[-1] + 1)):
            orient = "vertical"
        else:
            raise AssertionError("every red component must be a straight mirror segment")
        for cell in comp:
            mapping[cell] = orient
        bars.append((orient, comp))
    return mapping, bars


def _trace_ray(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]], list[str]]:
    seed = _find_seed(grid)
    r, c, dr, dc = _seed_state(grid.shape, seed)
    mirror_map, _bars = _mirror_bars(grid)

    out = np.array(grid, dtype=int, copy=True)
    seen_states: set[tuple[int, int, int, int]] = set()
    visited_blue = set(seed)
    new_cells: list[tuple[int, int]] = []
    hit_orients: list[str] = []

    while True:
        state = (r, c, dr, dc)
        assert state not in seen_states, "ray path must not loop"
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(grid.shape, nr, nc):
            break

        if int(grid[nr, nc]) == RED:
            orient = mirror_map.get((nr, nc))
            assert orient is not None, "red cell must belong to a mirror segment"
            hit_orients.append(orient)
            if orient == "horizontal":
                dr = -dr
            else:
                dc = -dc
            continue

        assert (nr, nc) not in visited_blue, "ray must not revisit a cell"
        out[nr, nc] = BLUE
        visited_blue.add((nr, nc))
        new_cells.append((nr, nc))
        r, c = nr, nc

    return out, seed, new_cells, hit_orients


def _rank_and_fill(grid: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    out = np.array(grid, dtype=int, copy=True)
    regions = _zero_regions(out)
    ranked = sorted(regions, key=lambda cells: (-len(cells), min(cells)))
    palette = (GREEN, YELLOW, GRAY)
    for idx, cells in enumerate(ranked):
        color = palette[idx] if idx < len(palette) else ORANGE
        for r, c in cells:
            out[r, c] = color
    return out, ranked


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the reflected diagonal ray, use it to partition the black background, and fill the regions by area rank.

    Input structure:
    - A black background contains exactly one blue diagonal seed near a corner.
    - Red foreground objects are straight horizontal or vertical mirror segments.
    - Aside from the blue seed and red mirrors, all remaining cells are black.

    Rule:
    - Continue the blue seed from the endpoint whose diagonal extension stays inside the grid.
    - Move one diagonal step at a time, painting visited black cells blue; if the next cell is a red mirror, do not paint it and instead reflect by flipping the row direction on a horizontal mirror or the column direction on a vertical mirror.
    - After the ray exits the grid, treat the full blue ray and the red mirrors as barriers, find all 4-connected black regions, sort them by descending area with top-left tie-break, and recolor ranks 1, 2, 3, 4+ with green, yellow, gray, and orange respectively.

    Key insight:
    - The blue ray and red mirrors are only intermediate dividers: the true output depends on how their combined barrier geometry cuts the black background into regions whose relative areas must be compared globally.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    traced, _seed, _new_cells, _hit_orients = _trace_ray(g)
    filled, _regions = _rank_and_fill(traced)
    return filled


def _segment_start_bounds(anchor: int, length: int, limit: int) -> tuple[int, int] | None:
    low = max(0, int(anchor) - int(length) + 1)
    high = min(int(anchor), int(limit) - int(length))
    if low > high:
        return None

    inner_low = max(low, 1)
    inner_high = min(high, int(limit) - int(length) - 1)
    if inner_low <= inner_high:
        return inner_low, inner_high
    return low, high


def _bar_cells_through(
    shape: tuple[int, int], orientation: str, r: int, c: int, length: int
) -> list[tuple[int, int]] | None:
    h, w = shape
    if orientation == "horizontal":
        bounds = _segment_start_bounds(c, length, w)
        if bounds is None:
            return None
        start = random.randint(*bounds)
        return [(r, start + i) for i in range(length)]

    bounds = _segment_start_bounds(r, length, h)
    if bounds is None:
        return None
    start = random.randint(*bounds)
    return [(start + i, c) for i in range(length)]


def _has_orthogonal_contact(cells: set[tuple[int, int]], other: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for nr, nc in _neighbors(r, c, 4):
            if (nr, nc) in other and (nr, nc) not in cells:
                return True
    return False


def _make_seed(shape: tuple[int, int], seed_len: int) -> tuple[list[tuple[int, int]], int, int]:
    h, w = shape
    corner = random.choice(("tl", "tr", "bl", "br"))
    offset = random.randint(0, 2)
    on_primary_edge = random.choice([False, True])

    if corner == "tl":
        start = (0, offset) if on_primary_edge else (offset, 0)
        dr, dc = 1, 1
    elif corner == "tr":
        start = (0, w - 1 - offset) if on_primary_edge else (offset, w - 1)
        dr, dc = 1, -1
    elif corner == "bl":
        start = (h - 1, offset) if on_primary_edge else (h - 1 - offset, 0)
        dr, dc = -1, 1
    else:
        start = (h - 1, w - 1 - offset) if on_primary_edge else (h - 1 - offset, w - 1)
        dr, dc = -1, -1

    cells = [(start[0] + i * dr, start[1] + i * dc) for i in range(seed_len)]
    if not all(_in_bounds(shape, r, c) for r, c in cells):
        raise AssertionError("seed left the grid")
    return cells, dr, dc


def _walk_segment(
    shape: tuple[int, int],
    r: int,
    c: int,
    dr: int,
    dc: int,
    length: int,
    blocked: set[tuple[int, int]],
) -> tuple[int, int, list[tuple[int, int]]] | None:
    painted: list[tuple[int, int]] = []
    for _ in range(length):
        nr, nc = r + dr, c + dc
        if not _in_bounds(shape, nr, nc) or (nr, nc) in blocked:
            return None
        painted.append((nr, nc))
        blocked.add((nr, nc))
        r, c = nr, nc
    return r, c, painted


def _reflected_direction(orientation: str, dr: int, dc: int) -> tuple[int, int]:
    if orientation == "horizontal":
        return -dr, dc
    return dr, -dc


def _walk_to_exit(
    shape: tuple[int, int],
    r: int,
    c: int,
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    trail: list[tuple[int, int]] = []
    while True:
        nr, nc = r + dr, c + dc
        if not _in_bounds(shape, nr, nc):
            return trail
        if (nr, nc) in blocked:
            return None
        trail.append((nr, nc))
        blocked.add((nr, nc))
        r, c = nr, nc


def _try_place_hit_mirror(
    shape: tuple[int, int],
    r: int,
    c: int,
    dr: int,
    dc: int,
    blue_cells: set[tuple[int, int]],
    red_cells: set[tuple[int, int]],
    preferred: str | None = None,
) -> tuple[str, list[tuple[int, int]], int, int] | None:
    mr, mc = r + dr, c + dc
    if not _in_bounds(shape, mr, mc) or (mr, mc) in blue_cells or (mr, mc) in red_cells:
        return None

    orientations = ["horizontal", "vertical"]
    random.shuffle(orientations)
    if preferred in orientations:
        orientations.remove(preferred)
        orientations.insert(0, preferred)

    for orient in orientations:
        new_dr, new_dc = _reflected_direction(orient, dr, dc)
        nr, nc = r + new_dr, c + new_dc
        if not _in_bounds(shape, nr, nc) or (nr, nc) in blue_cells or (nr, nc) in red_cells:
            continue

        lengths = list(range(MIRROR_LENGTH_RANGE[0], MIRROR_LENGTH_RANGE[1] + 1))
        random.shuffle(lengths)
        for length in lengths:
            cells = _bar_cells_through(shape, orient, mr, mc, length)
            if cells is None:
                continue
            cell_set = set(cells)
            if cell_set & (blue_cells | red_cells):
                continue
            if _has_orthogonal_contact(cell_set, red_cells):
                continue
            return orient, cells, new_dr, new_dc
    return None


def _try_place_extra_mirror(
    shape: tuple[int, int], blue_cells: set[tuple[int, int]], red_cells: set[tuple[int, int]]
) -> tuple[str, list[tuple[int, int]]] | None:
    for _ in range(40):
        orient = random.choice(["horizontal", "vertical"])
        length = random.randint(*MIRROR_LENGTH_RANGE)
        if orient == "horizontal":
            r = random.randint(1, shape[0] - 2)
            c = random.randint(1, shape[1] - length - 1)
            cells = [(r, c + i) for i in range(length)]
        else:
            r = random.randint(1, shape[0] - length - 1)
            c = random.randint(1, shape[1] - 2)
            cells = [(r + i, c) for i in range(length)]
        cell_set = set(cells)
        if cell_set & (blue_cells | red_cells):
            continue
        if _has_orthogonal_contact(cell_set, red_cells):
            continue
        return orient, cells
    return None


def _render_input(
    shape: tuple[int, int],
    seed_cells: list[tuple[int, int]],
    mirror_bars: list[tuple[str, list[tuple[int, int]]]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c in seed_cells:
        grid[r, c] = BLUE
    for _orient, cells in mirror_bars:
        for r, c in cells:
            grid[r, c] = RED
    return grid


def _choose_hit_count(total_mirrors: int, need_both: bool) -> int:
    choices = list(range(1, min(3, total_mirrors) + 1))
    weighted = [HIT_MIRROR_WEIGHTS[i] for i in choices]
    hit_count = random.choices(choices, weights=weighted, k=1)[0]
    if need_both and hit_count < 2 and total_mirrors >= 2:
        hit_count = 2
    return hit_count


def _analyze_candidate(
    grid: np.ndarray,
    need_tie: bool,
    need_orange: bool,
    need_both_hits: bool,
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        seed = _find_seed(grid)
        mirror_map, bars = _mirror_bars(grid)
        traced, _seed, new_cells, hit_orients = _trace_ray(grid)
        out, ranked = _rank_and_fill(traced)
        solved = main(grid)
    except Exception:
        return False

    if len(seed) not in SEED_LENGTH_RANGE:
        return False
    if not (MIRROR_COUNT_RANGE[0] <= len(bars) <= MIRROR_COUNT_RANGE[1]):
        return False
    if any(len(cells) not in range(MIRROR_LENGTH_RANGE[0], MIRROR_LENGTH_RANGE[1] + 1) for _o, cells in bars):
        return False
    if any(orient not in {"horizontal", "vertical"} for orient in mirror_map.values()):
        return False
    if len(new_cells) < 5 or len(hit_orients) < 1:
        return False

    areas = [len(region) for region in ranked]
    if len(ranked) < 3 or len(set(areas)) < 3:
        return False
    if need_orange and len(ranked) < 4:
        return False
    if need_tie and max(Counter(areas).values()) < 2:
        return False
    if need_both_hits and set(hit_orients) != {"horizontal", "vertical"}:
        return False

    divider_ratio = np.count_nonzero(traced != BLACK) / float(traced.size)
    if divider_ratio > 0.42:
        return False

    if not np.array_equal(out, solved):
        return False
    return True


def _build_candidate() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_ROWS_RANGE),
        random.randint(*GRID_COLS_RANGE),
    )
    seed_len = random.randint(*SEED_LENGTH_RANGE)
    total_mirrors = random.randint(*MIRROR_COUNT_RANGE)
    need_tie = random.random() < 0.25
    need_orange = random.random() < 0.5
    need_both_hits = random.random() < 0.65
    hit_count = _choose_hit_count(total_mirrors, need_both_hits)

    try:
        seed_cells, dr, dc = _make_seed(shape, seed_len)
    except AssertionError:
        return None

    blue_cells = set(seed_cells)
    red_cells: set[tuple[int, int]] = set()
    mirror_bars: list[tuple[str, list[tuple[int, int]]]] = []
    r, c = seed_cells[-1]

    first_required = None
    if need_both_hits:
        first_required = random.choice(["horizontal", "vertical"])

    for idx in range(hit_count):
        preferred = None
        if need_both_hits:
            if idx == 0:
                preferred = first_required
            elif idx == 1:
                preferred = "vertical" if first_required == "horizontal" else "horizontal"

        seg_lengths = list(range(SEGMENT_LENGTH_RANGE[0], SEGMENT_LENGTH_RANGE[1] + 1))
        random.shuffle(seg_lengths)
        committed = None
        for seg_len in seg_lengths:
            blocked = blue_cells | red_cells
            walked = _walk_segment(shape, r, c, dr, dc, seg_len, blocked.copy())
            if walked is None:
                continue
            nr, nc, cells = walked
            temp_blue = blue_cells | set(cells)
            placed = _try_place_hit_mirror(shape, nr, nc, dr, dc, temp_blue, red_cells, preferred)
            if placed is None:
                continue
            committed = (nr, nc, cells, placed)
            break
        if committed is None:
            return None

        r, c, cells, placed = committed
        blue_cells.update(cells)
        orient, bar_cells, dr, dc = placed
        mirror_bars.append((orient, bar_cells))
        red_cells.update(bar_cells)

    tail = _walk_to_exit(shape, r, c, dr, dc, (blue_cells | red_cells).copy())
    if tail is None or len(tail) < 2:
        return None
    blue_cells.update(tail)

    while len(mirror_bars) < total_mirrors:
        extra = _try_place_extra_mirror(shape, blue_cells, red_cells)
        if extra is None:
            return None
        orient, cells = extra
        mirror_bars.append((orient, cells))
        red_cells.update(cells)

    grid = _render_input(shape, seed_cells, mirror_bars)
    if not _analyze_candidate(grid, need_tie, need_orange, need_both_hits):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reflected-ray region-ranking puzzle")
