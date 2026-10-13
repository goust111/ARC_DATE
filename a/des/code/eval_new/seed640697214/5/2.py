import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

PATH_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, AZURE)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
SOURCE_COUNT_CHOICES = (1, 2)
PRIORITY_LEN_CHOICES = (2, 3)
MAIN_SEGMENT_LEN = (2, 6)
SIDE_SEGMENT_LEN = (2, 5)
COMPONENT_RETRIES = 120
GEN_RETRIES = 300


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _legend_info(grid: np.ndarray) -> tuple[list[int], tuple[int, int, int, int]]:
    h, w = grid.shape
    grey_comps = _components(grid == GREY)
    candidates: list[tuple[list[int], tuple[int, int, int, int]]] = []

    for comp in grey_comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if ww != 3 or not (4 <= hh <= 5):
            continue
        touches_corner = (
            (r0 == 0 and c0 == 0)
            or (r0 == 0 and c1 == w - 1)
            or (r1 == h - 1 and c0 == 0)
            or (r1 == h - 1 and c1 == w - 1)
        )
        if not touches_corner:
            continue
        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                if on_border and int(grid[r, c]) != GREY:
                    ok = False
                if not on_border and c != c0 + 1:
                    ok = False
        if not ok:
            continue
        priority = [int(grid[r, c0 + 1]) for r in range(r0 + 1, r1)]
        if not (2 <= len(priority) <= 3):
            continue
        if any(v in (BLACK, GREY, ORANGE, MAROON) for v in priority):
            continue
        if len(set(priority)) != len(priority):
            continue
        candidates.append((priority, (r0, c0, r1 + 1, c1 + 1)))

    assert len(candidates) == 1, "expected exactly one corner legend"
    return candidates[0]


def _trace_signal(
    grid: np.ndarray,
    path_mask: np.ndarray,
    source: tuple[int, int],
    ranks: dict[int, int],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    trail = [source]
    seen = {source}
    prev: tuple[int, int] | None = None
    cur = source

    while True:
        if cur != source and int(grid[cur]) == MAROON:
            break

        nbrs = [(nr, nc) for nr, nc in _neighbors4(cur[0], cur[1], h, w) if path_mask[nr, nc]]
        options = nbrs if prev is None else [p for p in nbrs if p != prev]
        if not options:
            break
        if len(options) == 1:
            nxt = options[0]
        else:
            nxt = min(
                options,
                key=lambda p: (ranks.get(int(grid[p]), len(ranks)), p[0], p[1]),
            )
        if nxt in seen:
            break
        trail.append(nxt)
        seen.add(nxt)
        prev, cur = cur, nxt

    return trail


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each signal through a colored path network using the corner legend's color priority.

    Input structure:
    - The background is black.
    - A network of single-pixel-wide colored paths occupies the grid, with orange 1x1 signal sources and maroon 1x1 endpoints placed on path endpoints.
    - One corner contains a grey rectangular frame whose interior is a vertical stack of 2-3 colors; top-to-bottom order gives path-color priority.

    Rule:
    - Read the legend colors from top to bottom to obtain the branch priority order.
    - For each orange source, follow the connected path forward; at each junction, ignore the cell you came from and choose the outgoing branch whose first pixel has the highest-priority color.
    - Repaint every visited path cell, including the reached maroon endpoint, to orange while leaving untouched paths and the legend unchanged.

    Key insight:
    - The legend does not encode directions directly; it ranks branch colors, and that ranking resolves every junction choice during path tracing.
    """
    g = np.asarray(grid, dtype=int)
    priority, legend_bbox = _legend_info(g)
    rank = {color: i for i, color in enumerate(priority)}

    legend_mask = np.zeros_like(g, dtype=bool)
    r0, c0, r1, c1 = legend_bbox
    legend_mask[r0:r1, c0:c1] = True

    path_mask = (g != BLACK) & ~legend_mask
    sources = [(int(r), int(c)) for r, c in zip(*np.where((g == ORANGE) & path_mask))]

    painted: set[tuple[int, int]] = set()
    for source in sources:
        painted.update(_trace_signal(g, path_mask, source, rank))

    out = g.copy()
    for r, c in painted:
        out[r, c] = ORANGE
    return out


def _segment_cells(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    dr, dc = direction
    r0, c0 = start
    return [(r0 + dr * step, c0 + dc * step) for step in range(1, length + 1)]


def _can_place_segment(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    occupied: set[tuple[int, int]],
    blocked: np.ndarray,
) -> list[tuple[int, int]] | None:
    h, w = shape
    cells = _segment_cells(start, direction, length)
    cell_set = set(cells)

    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return None
        if blocked[r, c] or (r, c) in occupied:
            return None

    for idx, (r, c) in enumerate(cells):
        prev = start if idx == 0 else cells[idx - 1]
        nxt = cells[idx + 1] if idx + 1 < len(cells) else None
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in cell_set:
                if (nr, nc) == prev or (nxt is not None and (nr, nc) == nxt):
                    continue
                return None
            if (nr, nc) in occupied and (nr, nc) != prev:
                return None

    return cells


def _viable_dirs(
    shape: tuple[int, int],
    node: tuple[int, int],
    forbidden_back: tuple[int, int] | None,
    occupied: set[tuple[int, int]],
    blocked: np.ndarray,
    min_len: int = 2,
) -> list[tuple[int, int]]:
    out = []
    for direction in DIRS4:
        if forbidden_back is not None and direction == forbidden_back:
            continue
        if _can_place_segment(shape, node, direction, min_len, occupied, blocked) is not None:
            out.append(direction)
    return out


def _supports_next_stage(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
    blocked: np.ndarray,
    need_future_junction: bool,
    rng: random.Random,
) -> bool:
    lengths = list(range(MAIN_SEGMENT_LEN[0], MAIN_SEGMENT_LEN[1] + 1))
    rng.shuffle(lengths)
    for length in lengths:
        cells = _can_place_segment(shape, start, direction, length, occupied, blocked)
        if cells is None:
            continue
        if not need_future_junction:
            return True
        end = cells[-1]
        next_occ = occupied | set(cells)
        back = (-direction[0], -direction[1])
        if len(_viable_dirs(shape, end, back, next_occ, blocked, min_len=2)) >= 2:
            return True
    return False


def _choose_stage(
    shape: tuple[int, int],
    start: tuple[int, int],
    directions: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    blocked: np.ndarray,
    need_junction: bool,
    rng: random.Random,
) -> tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    dir_order = directions[:]
    rng.shuffle(dir_order)
    lengths = list(range(MAIN_SEGMENT_LEN[0], MAIN_SEGMENT_LEN[1] + 1))
    rng.shuffle(lengths)
    for direction in dir_order:
        for length in lengths:
            cells = _can_place_segment(shape, start, direction, length, occupied, blocked)
            if cells is None:
                continue
            out_dirs: list[tuple[int, int]] = []
            if need_junction:
                end = cells[-1]
                next_occ = occupied | set(cells)
                back = (-direction[0], -direction[1])
                out_dirs = _viable_dirs(shape, end, back, next_occ, blocked, min_len=2)
                if len(out_dirs) < 2:
                    continue
            return direction, cells, out_dirs
    return None


def _endpoint_candidates(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    blocked: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if blocked[r, c] or (r, c) in occupied:
                continue
            if any((nr, nc) in occupied for nr, nc in _neighbors4(r, c, h, w)):
                continue
            out.append((r, c))
    return out


def _draw_legend(
    grid: np.ndarray,
    blocked: np.ndarray,
    priority: list[int],
    corner: str,
) -> tuple[int, int, int, int]:
    ph = len(priority) + 2
    patch = np.full((ph, 3), GREY, dtype=int)
    for idx, color in enumerate(priority, start=1):
        patch[idx, 1] = color

    h, w = grid.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - 3
    elif corner == "bl":
        r0, c0 = h - ph, 0
    else:
        r0, c0 = h - ph, w - 3

    grid[r0 : r0 + ph, c0 : c0 + 3] = patch
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r0 + ph + 1)
    cc1 = min(w, c0 + 3 + 1)
    blocked[rr0:rr1, cc0:cc1] = True
    return r0, c0, r0 + ph, c0 + 3


def _build_component(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    blocked: np.ndarray,
    priority: list[int],
    rng: random.Random,
) -> tuple[dict[tuple[int, int], int], set[tuple[int, int]]] | None:
    rank = {color: i for i, color in enumerate(priority)}

    for _ in range(COMPONENT_RETRIES):
        temp_occ = set(occupied)
        paints: dict[tuple[int, int], int] = {}
        trail: set[tuple[int, int]] = set()

        endpoints = _endpoint_candidates(shape, temp_occ, blocked)
        if not endpoints:
            return None
        source = rng.choice(endpoints)
        temp_occ.add(source)
        paints[source] = ORANGE
        trail.add(source)

        num_junctions = rng.choice((1, 2, 3))
        current_pos = source
        current_color = rng.choice(priority)
        current_dir: tuple[int, int] | None = None
        ok = True

        for stage in range(num_junctions + 1):
            need_junction = stage < num_junctions
            dirs = DIRS4 if current_dir is None else [current_dir]
            stage_choice = _choose_stage(shape, current_pos, dirs, temp_occ, blocked, need_junction, rng)
            if stage_choice is None:
                ok = False
                break

            direction, cells, outgoing = stage_choice
            current_dir = direction
            temp_occ.update(cells)
            trail.update(cells)

            if need_junction:
                for cell in cells:
                    paints[cell] = current_color

                junction = cells[-1]
                remaining_junctions = num_junctions - stage - 1
                plan_found = False

                branch_sizes = [2]
                if len(priority) >= 3 and len(outgoing) >= 3:
                    branch_sizes.append(3)
                rng.shuffle(branch_sizes)

                for branch_count in branch_sizes:
                    for _ in range(20):
                        branch_dirs = rng.sample(outgoing, branch_count)
                        need_future_junction = remaining_junctions > 0
                        continue_options = [
                            d
                            for d in branch_dirs
                            if _supports_next_stage(
                                shape,
                                junction,
                                d,
                                temp_occ,
                                blocked,
                                need_future_junction,
                                rng,
                            )
                        ]
                        if not continue_options:
                            continue

                        continue_dir = rng.choice(continue_options)
                        colors = rng.sample(priority, branch_count)
                        best_color = min(colors, key=lambda color: rank[color])
                        side_colors = [color for color in colors if color != best_color]
                        rng.shuffle(side_colors)

                        branch_color: dict[tuple[int, int], int] = {continue_dir: best_color}
                        side_dirs = [d for d in branch_dirs if d != continue_dir]
                        for side_dir, side_color in zip(side_dirs, side_colors):
                            branch_color[side_dir] = side_color

                        side_occ = set(temp_occ)
                        side_paints: dict[tuple[int, int], int] = {}
                        side_ok = True
                        for side_dir in side_dirs:
                            branch_cells = None
                            side_lengths = list(range(SIDE_SEGMENT_LEN[0], SIDE_SEGMENT_LEN[1] + 1))
                            rng.shuffle(side_lengths)
                            for length in side_lengths:
                                trial = _can_place_segment(shape, junction, side_dir, length, side_occ, blocked)
                                if trial is not None:
                                    branch_cells = trial
                                    break
                            if branch_cells is None:
                                side_ok = False
                                break
                            side_occ.update(branch_cells)
                            for cell in branch_cells[:-1]:
                                side_paints[cell] = branch_color[side_dir]
                            side_paints[branch_cells[-1]] = MAROON

                        if not side_ok:
                            continue

                        temp_occ = side_occ
                        paints.update(side_paints)
                        current_pos = junction
                        current_dir = continue_dir
                        current_color = best_color
                        plan_found = True
                        break
                    if plan_found:
                        break

                if not plan_found:
                    ok = False
                    break
            else:
                for cell in cells[:-1]:
                    paints[cell] = current_color
                paints[cells[-1]] = MAROON

        if ok:
            return paints, trail

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    priority_len = rng.choice(PRIORITY_LEN_CHOICES)
    priority = rng.sample(PATH_COLORS, priority_len)
    corner = rng.choice(("tl", "tr", "bl", "br"))
    _draw_legend(grid, blocked, priority, corner)

    source_count = rng.choice(SOURCE_COUNT_CHOICES)
    occupied: set[tuple[int, int]] = set()
    trail_union: set[tuple[int, int]] = set()

    for _ in range(source_count):
        built = _build_component(grid.shape, occupied, blocked, priority, rng)
        if built is None:
            return None
        paints, trail = built
        for (r, c), color in paints.items():
            grid[r, c] = color
        occupied.update(paints)
        trail_union.update(trail)

    if len(trail_union) <= source_count:
        return None

    expected = grid.copy()
    for r, c in trail_union:
        expected[r, c] = ORANGE
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
