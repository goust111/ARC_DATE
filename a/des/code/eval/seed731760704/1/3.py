import random
from collections import defaultdict, deque
from itertools import combinations

import numpy as np


BLACK = 0
GREY = 5
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator configuration.
GRID_H_RANGE = (10, 16)
GRID_W_RANGE = (8, 15)
LEGEND_WIDTH = 6
GEN_RETRIES = 300
SCENARIOS = ("cross_split", "triple_chain", "aligned_stop")
TOWER_PALETTE = (1, 2, 3, 4, 6, 8)
MIX_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _perpendicular_dirs(dr: int, dc: int) -> list[tuple[int, int]]:
    return [(0, -1), (0, 1)] if dr != 0 else [(-1, 0), (1, 0)]


def _extract_legend(
    grid: np.ndarray,
) -> tuple[dict[frozenset[int], int], set[int], str, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for side in ("top", "bottom"):
        r0, r1 = (0, 1) if side == "top" else (h - 2, h - 1)
        mask0 = g[r0] != BLACK
        mask1 = g[r1] != BLACK
        if not np.array_equal(mask0, mask1):
            continue
        cols = np.flatnonzero(mask0)
        if cols.size == 0:
            continue
        if np.any(np.diff(cols) != 1):
            continue
        c0, c1 = int(cols[0]), int(cols[-1])
        width = c1 - c0 + 1
        if width < 2 or width % 2 != 0:
            continue

        top = g[r0, c0 : c1 + 1]
        bottom = g[r1, c0 : c1 + 1]
        legend: dict[frozenset[int], int] = {}
        ok = True
        for start in range(0, width, 2):
            a = int(top[start])
            b = int(top[start + 1])
            mix0 = int(bottom[start])
            mix1 = int(bottom[start + 1])
            if BLACK in (a, b, mix0, mix1):
                ok = False
                break
            if a == b or mix0 != mix1:
                ok = False
                break
            key = frozenset((a, b))
            if len(key) != 2:
                ok = False
                break
            prev = legend.get(key)
            if prev is not None and prev != mix0:
                ok = False
                break
            legend[key] = mix0
        if ok:
            blocked_rows = {0, 1} if side == "top" else {h - 2, h - 1}
            candidates.append((legend, blocked_rows, side, (c0, c1)))

    assert len(candidates) == 1, "expected exactly one edge legend strip"
    return candidates[0]


def _collect_objects(
    grid: np.ndarray, blocked_rows: set[int]
) -> tuple[dict[tuple[int, int], int], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    towers: dict[tuple[int, int], int] = {}
    prisms: set[tuple[int, int]] = set()
    tower_counts: dict[int, int] = defaultdict(int)

    for r in range(g.shape[0]):
        if r in blocked_rows:
            continue
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color == BLACK:
                continue
            if color == GREY:
                prisms.add((r, c))
            else:
                towers[(r, c)] = color
                tower_counts[color] += 1

    assert len(towers) >= 2, "expected at least two signal towers"
    assert prisms, "expected at least one prism"
    assert all(count == 1 for count in tower_counts.values()), "tower colors must be unique"
    return towers, prisms


def _simulate_rays(
    grid: np.ndarray, blocked_rows: set[int]
) -> tuple[
    dict[tuple[int, int], int],
    set[tuple[int, int]],
    dict[tuple[int, int], list[tuple[int, int, int]]],
]:
    g = np.asarray(grid, dtype=int)
    towers, prisms = _collect_objects(g, blocked_rows)
    events: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)

    queue: deque[tuple[int, int, int, int, int]] = deque()
    for (r, c), color in sorted(towers.items()):
        for dr, dc in DIRS:
            queue.append((r, c, dr, dc, color))

    seen_segments: set[tuple[int, int, int, int, int]] = set()
    order = 0
    while queue:
        src_r, src_c, dr, dc, color = queue.popleft()
        state = (src_r, src_c, dr, dc, color)
        if state in seen_segments:
            continue
        seen_segments.add(state)

        r, c = src_r, src_c
        step = 0
        while True:
            nr, nc = r + dr, c + dc
            step += 1
            if not _in_bounds(g.shape, nr, nc) or nr in blocked_rows:
                break

            pos = (nr, nc)
            if pos in towers:
                break
            if pos in prisms:
                for ndr, ndc in _perpendicular_dirs(dr, dc):
                    queue.append((nr, nc, ndr, ndc, GREY))
                break

            events[pos].append((step, order, color))
            order += 1
            r, c = nr, nc

    return towers, prisms, events


def _resolve_cell_color(
    hits: list[tuple[int, int, int]], legend: dict[frozenset[int], int]
) -> int:
    ordered = sorted(hits)
    distinct: list[int] = []
    for _step, _order, color in ordered:
        if color not in distinct:
            distinct.append(color)
        if len(distinct) == 2:
            break
    assert distinct, "visited cell must have at least one hit"
    if len(distinct) == 1:
        return distinct[0]
    pair = frozenset(distinct)
    assert pair in legend, "missing color pair in legend"
    return legend[pair]


def _all_needed_pairs(towers: dict[tuple[int, int], int], prisms: set[tuple[int, int]]) -> set[frozenset[int]]:
    active = set(towers.values())
    if prisms:
        active.add(GREY)
    return {frozenset(pair) for pair in combinations(sorted(active), 2)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast tower rays, split them at grey prisms, and mix overlap colors via the edge legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    legend, blocked_rows, _side, _span = _extract_legend(g)
    towers, prisms, events = _simulate_rays(g, blocked_rows)
    assert _all_needed_pairs(towers, prisms) <= set(legend), "legend must cover all active color pairs"

    out = g.copy()
    for pos, hits in events.items():
        if pos in towers or pos in prisms:
            continue
        out[pos] = _resolve_cell_color(hits, legend)
    return out


def _apply_flips(
    cells: list[tuple[int, int]], arena_h: int, arena_w: int
) -> list[tuple[int, int]]:
    flip_r = random.random() < 0.5
    flip_c = random.random() < 0.5
    out = []
    for r, c in cells:
        rr = arena_h - 1 - r if flip_r else r
        cc = arena_w - 1 - c if flip_c else c
        out.append((rr, cc))
    return out


def _scenario_cross_split(arena_h: int, arena_w: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    c_a = random.randint(1, arena_w - 5)
    c_b = random.randint(c_a + 3, arena_w - 2)
    r_a = random.randint(1, arena_h - 5)
    r_b = random.randint(r_a + 3, arena_h - 2)
    r_p = random.randint(r_a + 1, r_b - 1)
    cells = _apply_flips([(r_a, c_a), (r_b, c_b), (r_p, c_a)], arena_h, arena_w)
    return cells[:2], cells[2:]


def _scenario_triple_chain(arena_h: int, arena_w: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    for _ in range(50):
        c_a = random.randint(1, arena_w - 5)
        c_b = random.randint(c_a + 3, arena_w - 2)
        r_b = random.randint(1, arena_h - 6)
        r_a = random.randint(r_b + 2, arena_h - 4)
        r_p = random.randint(r_a + 2, arena_h - 2)
        d_a = c_b - c_a
        d_b = r_a - r_b
        d_g = r_p - r_a
        if len({d_a, d_b, d_g}) < 3:
            continue
        towers = [(r_a, c_a), (r_b, c_b)]
        prisms = [(r_p, c_a), (r_p, c_b)]
        cells = _apply_flips(towers + prisms, arena_h, arena_w)
        return cells[:2], cells[2:]
    raise RuntimeError("failed to sample triple-chain layout")


def _scenario_aligned_stop(arena_h: int, arena_w: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if random.random() < 0.5:
        row = random.randint(1, arena_h - 4)
        c_a = random.randint(1, arena_w - 5)
        c_b = random.randint(c_a + 3, arena_w - 2)
        r_p = random.randint(row + 2, arena_h - 2)
        towers = [(row, c_a), (row, c_b)]
        prisms = [(r_p, c_a)]
    else:
        col = random.randint(1, arena_w - 4)
        r_a = random.randint(1, arena_h - 5)
        r_b = random.randint(r_a + 3, arena_h - 2)
        c_p = random.randint(col + 2, arena_w - 2)
        towers = [(r_a, col), (r_b, col)]
        prisms = [(r_a, c_p)]
    cells = _apply_flips(towers + prisms, arena_h, arena_w)
    return cells[:2], cells[2:]


def _build_layout(
    scenario: str, arena_h: int, arena_w: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if scenario == "cross_split":
        towers, prisms = _scenario_cross_split(arena_h, arena_w)
    elif scenario == "triple_chain":
        towers, prisms = _scenario_triple_chain(arena_h, arena_w)
    else:
        towers, prisms = _scenario_aligned_stop(arena_h, arena_w)
    assert len(set(towers + prisms)) == len(towers) + len(prisms)
    return towers, prisms


def _make_legend(
    tower_colors: list[int],
) -> tuple[dict[frozenset[int], int], list[tuple[int, int, int]]]:
    active = [tower_colors[0], tower_colors[1], GREY]
    pair_keys = [frozenset(pair) for pair in combinations(active, 2)]
    mix_pool = [color for color in MIX_PALETTE if color not in active]
    mix_colors = random.sample(mix_pool, len(pair_keys))
    mapping = dict(zip(pair_keys, mix_colors))

    blocks = []
    for key, mix in mapping.items():
        a, b = tuple(key)
        if random.random() < 0.5:
            a, b = b, a
        blocks.append((a, b, mix))
    random.shuffle(blocks)
    return mapping, blocks


def _place_legend(
    grid: np.ndarray, side: str, start_col: int, blocks: list[tuple[int, int, int]]
) -> None:
    r0, r1 = (0, 1) if side == "top" else (grid.shape[0] - 2, grid.shape[0] - 1)
    col = start_col
    for a, b, mix in blocks:
        grid[r0, col] = a
        grid[r0, col + 1] = b
        grid[r1, col] = mix
        grid[r1, col + 1] = mix
        col += 2


def _cell_summary(
    events: dict[tuple[int, int], list[tuple[int, int, int]]]
) -> tuple[int, int, int]:
    tower_only_mix = 0
    grey_mix = 0
    triple = 0
    for hits in events.values():
        ordered = sorted(hits)
        distinct: list[int] = []
        for _step, _order, color in ordered:
            if color not in distinct:
                distinct.append(color)
        if len(distinct) >= 2:
            pair = frozenset(distinct[:2])
            if GREY in pair:
                grey_mix += 1
            else:
                tower_only_mix += 1
        if len(distinct) >= 3:
            triple += 1
    return tower_only_mix, grey_mix, triple


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    if w < LEGEND_WIDTH or h < 8:
        return None

    side = random.choice(("top", "bottom"))
    arena_h = h - 2
    arena_w = w
    if arena_h < 7 or arena_w < 8:
        return None

    scenario = random.choice(SCENARIOS)
    try:
        tower_cells, prism_cells = _build_layout(scenario, arena_h, arena_w)
    except RuntimeError:
        return None

    tower_colors = random.sample(TOWER_PALETTE, 2)
    legend_map, legend_blocks = _make_legend(tower_colors)

    grid = np.zeros((h, w), dtype=int)
    legend_start = random.randint(0, w - LEGEND_WIDTH)
    _place_legend(grid, side, legend_start, legend_blocks)

    row_offset = 2 if side == "top" else 0
    for (r, c), color in zip(tower_cells, tower_colors):
        grid[row_offset + r, c] = color
    for r, c in prism_cells:
        grid[row_offset + r, c] = GREY

    try:
        parsed_legend, blocked_rows, _parsed_side, _span = _extract_legend(grid)
        if parsed_legend != legend_map:
            return None
        towers, prisms, events = _simulate_rays(grid, blocked_rows)
        if _all_needed_pairs(towers, prisms) != set(parsed_legend):
            return None
        tower_mix, grey_mix, triple = _cell_summary(events)
        if tower_mix + grey_mix == 0:
            return None
        if scenario == "cross_split" and (tower_mix < 1 or grey_mix < 1):
            return None
        if scenario == "triple_chain" and (tower_mix < 1 or grey_mix < 1 or triple < 1):
            return None
        if scenario == "aligned_stop" and grey_mix < 1:
            return None
        out = main(grid)
        if np.array_equal(out, grid):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
