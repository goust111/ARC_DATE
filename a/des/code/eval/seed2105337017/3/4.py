import random
from collections import Counter

import numpy as np


BG = 0
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4
FRAME = 5
KEY_COLORS = (6, 7, 8)
OBSTACLE = 9

ARROWS = (UP, RIGHT, DOWN, LEFT)
ARROW_TO_DELTA = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}

TABLE_W = 7
GEN_RETRIES = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _parse_table(grid: np.ndarray) -> tuple[list[tuple[int, int, list[int]]], tuple[int, int, int, int]]:
    pts = np.argwhere(grid == FRAME)
    assert pts.size > 0, "lookup table frame must exist"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    assert c1 - c0 + 1 >= TABLE_W, "lookup table is too narrow"

    separator_rows = [
        r
        for r in range(r0, r1 + 1)
        if np.all(grid[r, c0 : c1 + 1] == FRAME)
    ]
    assert len(separator_rows) >= 2, "lookup table needs at least one entry"

    entries: list[tuple[int, int, list[int]]] = []
    for top, bottom in zip(separator_rows, separator_rows[1:]):
        if bottom <= top + 1:
            continue
        content_rows = [
            r
            for r in range(top + 1, bottom)
            if np.any(grid[r, c0 + 1 : c1] != FRAME)
        ]
        if not content_rows:
            continue
        row = content_rows[0]
        key = int(grid[row, c0 + 1])
        pattern = [int(v) for v in grid[row, c0 + 2 : c0 + 6]]
        assert len(pattern) == 4, "direction pattern must have four cells"
        assert all(v in ARROW_TO_DELTA for v in pattern), "unknown arrow color"
        entries.append((row, key, pattern))

    assert entries, "failed to parse any lookup table entries"
    return entries, (r0, c0, r1, c1)


def _seed_states(
    grid: np.ndarray, entries: list[tuple[int, int, list[int]]], table_bbox: tuple[int, int, int, int]
) -> list[dict[str, object]]:
    r0, c0, r1, c1 = table_bbox
    states: list[dict[str, object]] = []
    for _row, key, pattern in entries:
        positions = []
        for rr, cc in np.argwhere(grid == key):
            r = int(rr)
            c = int(cc)
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            positions.append((r, c))
        positions.sort()
        for pos in positions:
            states.append(
                {
                    "color": int(key),
                    "pattern": list(pattern),
                    "index": 0,
                    "pos": pos,
                    "alive": True,
                }
            )
    return states


def _simulate_growth(
    grid: np.ndarray,
    entries: list[tuple[int, int, list[int]]],
    table_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, Counter]:
    g = np.asarray(grid, dtype=int)
    if table_bbox is None:
        entries, table_bbox = _parse_table(g)

    states = _seed_states(g, entries, table_bbox)
    out = g.copy()
    stats: Counter = Counter()
    h, w = out.shape

    while any(bool(state["alive"]) for state in states):
        for state in states:
            if not bool(state["alive"]):
                continue

            pattern = state["pattern"]
            assert isinstance(pattern, list)
            success = False
            for _ in range(4):
                arrow = int(pattern[int(state["index"])])
                state["index"] = (int(state["index"]) + 1) % 4
                dr, dc = ARROW_TO_DELTA[arrow]
                r, c = state["pos"]
                nr = int(r) + dr
                nc = int(c) + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    stats["blocked_boundary"] += 1
                    continue
                if int(out[nr, nc]) != BG:
                    if int(out[nr, nc]) == OBSTACLE:
                        stats["blocked_obstacle"] += 1
                    else:
                        stats["blocked_occupied"] += 1
                    continue
                out[nr, nc] = int(state["color"])
                state["pos"] = (nr, nc)
                stats["moves"] += 1
                success = True
                break

            if not success:
                state["alive"] = False
                stats["terminated"] += 1

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a color -> direction-cycle lookup table, then grow every seed in round-robin order.

    Each seed keeps its own pointer into its 4-step cycle. On a turn it tries up to four
    directions, advancing the pointer after every attempt. It can only step onto black cells.
    If all four directions are blocked by the boundary, an obstacle, or an existing path pixel,
    that seed stops permanently.
    """
    g = np.asarray(grid, dtype=int)
    entries, bbox = _parse_table(g)
    out, _stats = _simulate_growth(g, entries, bbox)
    return out


def _random_pattern() -> list[int]:
    if random.random() < 0.35:
        base = random.choice(
            [
                [RIGHT, DOWN, LEFT, UP],
                [DOWN, RIGHT, UP, LEFT],
            ]
        )
        shift = random.randrange(4)
        return base[shift:] + base[:shift]

    while True:
        pattern = [random.choice(ARROWS) for _ in range(4)]
        if len(set(pattern)) >= 2:
            return pattern


def _draw_table(grid: np.ndarray, top: int, left: int, keys: list[int], patterns: list[list[int]]) -> tuple[int, int, int, int]:
    n = len(keys)
    height = 2 * n + 1
    right = left + TABLE_W - 1
    bottom = top + height - 1

    grid[top : bottom + 1, left] = FRAME
    grid[top : bottom + 1, right] = FRAME
    for row in range(top, bottom + 1, 2):
        grid[row, left : right + 1] = FRAME

    for i, key in enumerate(keys):
        row = top + 1 + 2 * i
        grid[row, left + 1] = int(key)
        grid[row, left + 2 : left + 6] = np.array(patterns[i], dtype=int)

    return top, left, bottom, right


def _square_cycle() -> list[int]:
    bases = [
        [RIGHT, DOWN, LEFT, UP],
        [DOWN, LEFT, UP, RIGHT],
        [LEFT, UP, RIGHT, DOWN],
        [UP, RIGHT, DOWN, LEFT],
        [DOWN, RIGHT, UP, LEFT],
        [RIGHT, UP, LEFT, DOWN],
        [UP, LEFT, DOWN, RIGHT],
        [LEFT, DOWN, RIGHT, UP],
    ]
    return list(random.choice(bases))


def _prefix_positions(pattern: list[int], steps: int) -> list[tuple[int, int]]:
    r = 0
    c = 0
    out = []
    for arrow in pattern[:steps]:
        dr, dc = ARROW_TO_DELTA[int(arrow)]
        r += dr
        c += dc
        out.append((r, c))
    return out


def _seed_far_enough(pos: tuple[int, int], seeds: list[tuple[int, int]]) -> bool:
    r, c = pos
    for rr, cc in seeds:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    return True


def _place_shape(
    shape: tuple[int, int],
    min_col: int,
    rel_cells: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> tuple[int, int] | None:
    rows = [r for r, _c in rel_cells]
    cols = [c for _r, c in rel_cells]
    low_r = -min(rows)
    high_r = shape[0] - 1 - max(rows)
    low_c = max(min_col, -min(cols))
    high_c = shape[1] - 1 - max(cols)
    if low_r > high_r or low_c > high_c:
        return None

    candidates = [
        (r, c)
        for r in range(low_r, high_r + 1)
        for c in range(low_c, high_c + 1)
    ]
    random.shuffle(candidates)
    for base_r, base_c in candidates:
        cells = {(base_r + dr, base_c + dc) for dr, dc in rel_cells}
        if cells & forbidden:
            continue
        return base_r, base_c
    return None


def _choose_boundary_pattern() -> list[int]:
    first = random.choice([UP, DOWN, RIGHT])
    second = {UP: DOWN, DOWN: UP, RIGHT: LEFT}[first]
    tail = [random.choice(ARROWS) for _ in range(2)]
    return [first, second] + tail


def _choose_obstacle_pattern() -> tuple[list[int], int, int]:
    first = random.choice(ARROWS)
    choices = [d for d in ARROWS if d != first]
    second = random.choice(choices)
    tail = [random.choice(ARROWS) for _ in range(2)]
    return [first, second] + tail, first, second


def _random_seed_cell(
    shape: tuple[int, int],
    min_col: int,
    forbidden: set[tuple[int, int]],
    seeds: list[tuple[int, int]],
) -> tuple[int, int] | None:
    candidates = [
        (r, c)
        for r in range(shape[0])
        for c in range(min_col, shape[1])
        if (r, c) not in forbidden
    ]
    random.shuffle(candidates)
    for pos in candidates:
        if _seed_far_enough(pos, seeds):
            return pos
    return None


def _attempt_generate() -> np.ndarray | None:
    target = random.choice(["obstacle", "boundary", "occupied", "stuck"])
    n_entries = random.randint(2, len(KEY_COLORS))
    keys = random.sample(list(KEY_COLORS), n_entries)

    table_h = 2 * n_entries + 1
    table_left = random.randint(0, 1)
    gap = random.randint(2, 5)
    min_main_w = 9
    h = random.randint(max(10, table_h + 1), 18)
    w_min = table_left + TABLE_W + gap + min_main_w
    w = random.randint(max(18, w_min), 30)
    table_top = random.randint(0, h - table_h)
    main_left = table_left + TABLE_W + gap

    patterns = [_random_pattern() for _ in range(n_entries)]
    focus_idx = random.randrange(n_entries)
    protected: set[tuple[int, int]] = set()
    forced_obstacles: set[tuple[int, int]] = set()
    seed_positions: dict[int, tuple[int, int]] = {}

    if target == "boundary":
        patterns[focus_idx] = _choose_boundary_pattern()
    elif target == "occupied":
        patterns[focus_idx] = _square_cycle()
    elif target == "obstacle":
        patterns[focus_idx], blocked_dir, escape_dir = _choose_obstacle_pattern()
    else:
        blocked_dir = escape_dir = None

    focus_key = keys[focus_idx]

    if target == "boundary":
        first = patterns[focus_idx][0]
        if first == UP:
            pos = (0, random.randint(main_left, w - 1))
        elif first == DOWN:
            pos = (h - 1, random.randint(main_left, w - 1))
        else:
            pos = (random.randint(0, h - 1), w - 1)
        seed_positions[focus_key] = pos
        protected.add(pos)
        dr, dc = ARROW_TO_DELTA[patterns[focus_idx][1]]
        nr = pos[0] + dr
        nc = pos[1] + dc
        if _in_bounds((h, w), nr, nc) and nc >= main_left:
            protected.add((nr, nc))

    elif target == "occupied":
        rel_cells = [(0, 0)] + _prefix_positions(patterns[focus_idx], 3)
        anchor = _place_shape((h, w), main_left, rel_cells, protected)
        if anchor is None:
            return None
        base_r, base_c = anchor
        pos = (base_r, base_c)
        seed_positions[focus_key] = pos
        for dr, dc in rel_cells:
            protected.add((base_r + dr, base_c + dc))

    elif target == "obstacle":
        dr_block, dc_block = ARROW_TO_DELTA[int(blocked_dir)]
        dr_escape, dc_escape = ARROW_TO_DELTA[int(escape_dir)]
        rel_cells = [(0, 0), (dr_block, dc_block), (dr_escape, dc_escape)]
        anchor = _place_shape((h, w), main_left, rel_cells, protected)
        if anchor is None:
            return None
        base_r, base_c = anchor
        pos = (base_r, base_c)
        seed_positions[focus_key] = pos
        forced_obstacles.add((base_r + dr_block, base_c + dc_block))
        protected.update(
            {
                pos,
                (base_r + dr_block, base_c + dc_block),
                (base_r + dr_escape, base_c + dc_escape),
            }
        )

    else:
        rel_cells = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        anchor = _place_shape((h, w), main_left + 1, rel_cells, protected)
        if anchor is None:
            return None
        base_r, base_c = anchor
        pos = (base_r, base_c)
        seed_positions[focus_key] = pos
        protected.add(pos)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            forced_obstacles.add((base_r + dr, base_c + dc))
            protected.add((base_r + dr, base_c + dc))

    placed_seeds = [seed_positions[focus_key]]
    for key in keys:
        if key == focus_key:
            continue
        if random.random() < 0.2:
            edge = random.choice(["top", "bottom", "right"])
            if edge == "top":
                pos = (0, random.randint(main_left, w - 1))
            elif edge == "bottom":
                pos = (h - 1, random.randint(main_left, w - 1))
            else:
                pos = (random.randint(0, h - 1), w - 1)
            if pos in protected or not _seed_far_enough(pos, placed_seeds):
                pos = _random_seed_cell((h, w), main_left, protected, placed_seeds)
        else:
            pos = _random_seed_cell((h, w), main_left, protected, placed_seeds)
        if pos is None:
            return None
        seed_positions[key] = pos
        placed_seeds.append(pos)
        protected.add(pos)

    grid = np.full((h, w), BG, dtype=int)
    table_bbox = _draw_table(grid, table_top, table_left, keys, patterns)

    for r, c in forced_obstacles:
        if not _in_bounds((h, w), r, c):
            return None
        if c < main_left:
            return None
        grid[r, c] = OBSTACLE

    seed_neighbor_block = set(forced_obstacles)
    for r, c in seed_positions.values():
        seed_neighbor_block.add((r, c))
        for dr, dc in ARROW_TO_DELTA.values():
            rr = r + dr
            cc = c + dc
            if _in_bounds((h, w), rr, cc):
                seed_neighbor_block.add((rr, cc))

    extra_obstacles = random.randint(2, max(4, (h * (w - main_left)) // 18))
    obstacle_candidates = [
        (r, c)
        for r in range(h)
        for c in range(main_left, w)
        if grid[r, c] == BG and (r, c) not in seed_neighbor_block
    ]
    random.shuffle(obstacle_candidates)
    for r, c in obstacle_candidates[:extra_obstacles]:
        grid[r, c] = OBSTACLE

    for key, pos in seed_positions.items():
        r, c = pos
        if grid[r, c] != BG:
            return None
        grid[r, c] = int(key)

    entries, bbox = _parse_table(grid)
    if bbox != table_bbox:
        return None

    out, stats = _simulate_growth(grid, entries, bbox)
    if stats["moves"] < (3 if target == "stuck" else 5):
        return None
    if np.array_equal(grid, out):
        return None
    if target == "obstacle" and stats["blocked_obstacle"] == 0:
        return None
    if target == "boundary" and stats["blocked_boundary"] == 0:
        return None
    if target == "occupied" and stats["blocked_occupied"] == 0:
        return None
    if target == "stuck" and stats["terminated"] == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {GEN_RETRIES} attempts")
