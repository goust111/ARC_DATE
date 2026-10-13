import random
from collections import Counter, defaultdict

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.flat)
    return max(counts, key=counts.get)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _clockwise_cycle(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        pts.append((r, c0))
    for c in range(c0 + 1, c1 + 1):
        pts.append((r1, c))
    for r in range(r1 - 1, r0 - 1, -1):
        pts.append((r, c1))
    for c in range(c1 - 1, c0, -1):
        pts.append((r0, c))
    return pts


def _room_bboxes(grid: np.ndarray, wall: int) -> list[tuple[int, int, int, int]]:
    mask = grid != wall
    rooms: list[tuple[int, int, int, int]] = []
    for comp in _components(mask):
        if any(r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1 for r, c in comp):
            continue
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 <= r0 or c1 <= c0:
            continue
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        rooms.append((r0, c0, r1, c1))
    return sorted(rooms)


def _path_on_cycle(
    cycle: list[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    idx = {pt: i for i, pt in enumerate(cycle)}
    start_i = idx[start]
    end_i = idx[end]
    path: list[tuple[int, int]] = []
    i = start_i
    while True:
        path.append(cycle[i])
        if i == end_i:
            return path
        i = (i + 1) % len(cycle)


def _seed_start(pair: list[tuple[int, int]]) -> tuple[int, int]:
    return min(pair)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-colored seed pixels by walking clockwise along the room's inner wall.

    The pair itself is unordered in the input, so the deterministic visible tie-break is:
    start from the row-major earlier seed, then walk forward on the room's clockwise inner-wall cycle
    until reaching the other seed.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    counts = Counter(int(v) for v in g.flat)
    non_bg = [color for color in counts if color != bg]
    if not non_bg:
        return g.copy()
    wall = max(non_bg, key=lambda color: counts[color])

    out = g.copy()
    for r0, c0, r1, c1 in _room_bboxes(g, wall):
        cycle = _clockwise_cycle(r0, c0, r1, c1)
        cycle_set = set(cycle)
        seeds_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                color = int(g[r, c])
                if color != bg:
                    seeds_by_color[color].append((r, c))

        paint_ops: list[tuple[tuple[int, int], int, list[tuple[int, int]]]] = []
        for color, pts in seeds_by_color.items():
            if len(pts) != 2:
                continue
            if pts[0] not in cycle_set or pts[1] not in cycle_set:
                continue
            start = _seed_start(pts)
            end = pts[0] if pts[1] == start else pts[1]
            paint_ops.append((start, color, _path_on_cycle(cycle, start, end)))

        paint_ops.sort(key=lambda item: item[0])
        for _start, color, path in paint_ops:
            for r, c in path:
                out[r, c] = color
    return out


BG = 0
WALL = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 300


def _edge_kind(pt: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = pt
    r0, c0, r1, c1 = bbox
    if r == r0:
        return "T"
    if c == c1:
        return "R"
    if r == r1:
        return "B"
    return "L"


def _same_edge(a: tuple[int, int], b: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    return _edge_kind(a, bbox) == _edge_kind(b, bbox)


def _opposite_edges(a: tuple[int, int], b: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    ea = _edge_kind(a, bbox)
    eb = _edge_kind(b, bbox)
    return (ea, eb) in {("T", "B"), ("B", "T"), ("L", "R"), ("R", "L")}


def _pair_category(a: tuple[int, int], b: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
        if (a[0] == b[0] == bbox[0]) or (a[1] == b[1] == bbox[3]):
            return "adjacent_long"
        return "adjacent_short"
    if _same_edge(a, b, bbox):
        return "same_edge"
    if _opposite_edges(a, b, bbox):
        return "opposite"
    return "turn"


def _path_indices_for_pair(
    cycle: list[tuple[int, int]],
    pair: tuple[tuple[int, int], tuple[int, int]],
) -> set[int]:
    idx = {pt: i for i, pt in enumerate(cycle)}
    start = min(pair)
    end = pair[0] if pair[1] == start else pair[1]
    cur = idx[start]
    end_i = idx[end]
    used: set[int] = set()
    while True:
        used.add(cur)
        if cur == end_i:
            return used
        cur = (cur + 1) % len(cycle)


def _candidate_pairs(
    cycle: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    available: set[int],
    category: str,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    cells = [cycle[i] for i in sorted(available)]
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            a, b = cells[i], cells[j]
            cat = _pair_category(a, b, bbox)
            if category == "any" or cat == category:
                pairs.append((a, b))
    return pairs


def _choose_room_pairs(
    cycle: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    rng: random.Random,
    num_pairs: int,
) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    available = set(range(len(cycle)))
    idx = {pt: i for i, pt in enumerate(cycle)}
    colors = rng.sample(SEED_COLORS, num_pairs)
    chosen: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    prior_paths: list[set[int]] = []
    room_wants_overlap = num_pairs >= 2 and rng.random() < 0.6

    for color in colors:
        order = ["adjacent_long", "same_edge", "opposite", "turn", "adjacent_short"]
        rng.shuffle(order)
        order.append("any")
        pair: tuple[tuple[int, int], tuple[int, int]] | None = None
        for category in order:
            candidates = _candidate_pairs(cycle, bbox, available, category)
            if not candidates:
                continue
            if room_wants_overlap and prior_paths:
                overlapping = []
                for cand in candidates:
                    cand_path = _path_indices_for_pair(cycle, cand)
                    if any(cand_path & prev for prev in prior_paths):
                        overlapping.append(cand)
                if overlapping:
                    candidates = overlapping
            pair = rng.choice(candidates)
            break
        if pair is None:
            raise ValueError("unable to place all room pairs")

        chosen.append((color, pair[0], pair[1]))
        prior_paths.append(_path_indices_for_pair(cycle, pair))
        used = {idx[pair[0]], idx[pair[1]]}
        available -= used
    return chosen


def _build_partition(rng: random.Random) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    for _ in range(200):
        nrows = rng.randint(1, 3)
        ncols = rng.randint(1, 3)
        if nrows * ncols == 1 and rng.random() < 0.8:
            continue
        heights = [rng.randint(4, 8) for _ in range(nrows)]
        widths = [rng.randint(4, 8) for _ in range(ncols)]
        h = sum(heights) + nrows + 1
        w = sum(widths) + ncols + 1
        if not (14 <= h <= 28 and 14 <= w <= 28):
            continue

        grid = np.full((h, w), BG, dtype=int)
        row_walls = [0]
        cur = 0
        for span in heights:
            cur += span + 1
            row_walls.append(cur)
        col_walls = [0]
        cur = 0
        for span in widths:
            cur += span + 1
            col_walls.append(cur)

        for r in row_walls:
            grid[r, :] = WALL
        for c in col_walls:
            grid[:, c] = WALL

        rooms: list[tuple[int, int, int, int]] = []
        r0 = 1
        for room_h in heights:
            c0 = 1
            for room_w in widths:
                rooms.append((r0, c0, r0 + room_h - 1, c0 + room_w - 1))
                c0 += room_w + 1
            r0 += room_h + 1
        return grid, rooms
    raise RuntimeError("failed to build room partition")


def _validate_input(grid: np.ndarray) -> None:
    counts = Counter(int(v) for v in grid.flat)
    bg = _get_bg(grid)
    assert bg == BG
    non_bg = [color for color in counts if color != bg]
    wall = max(non_bg, key=lambda color: counts[color])
    assert wall == WALL
    assert counts[bg] > counts[wall]
    assert all(counts[wall] > counts[color] for color in non_bg if color != wall)

    rooms = _room_bboxes(grid, wall)
    assert rooms
    total_pairs = 0
    saw_adjacent_long = False
    saw_multi_pair_room = False
    for bbox in rooms:
        r0, c0, r1, c1 = bbox
        cycle = set(_clockwise_cycle(r0, c0, r1, c1))
        by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                color = int(grid[r, c])
                if color != bg:
                    by_color[color].append((r, c))
        pair_count = 0
        for color, pts in by_color.items():
            assert color != wall
            assert len(pts) == 2
            assert all(pt in cycle for pt in pts)
            pair_count += 1
            total_pairs += 1
            if _pair_category(pts[0], pts[1], bbox) == "adjacent_long":
                saw_adjacent_long = True
        if pair_count >= 2:
            saw_multi_pair_room = True

    assert total_pairs >= 2
    assert saw_adjacent_long
    assert saw_multi_pair_room
    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid, rooms = _build_partition(rng)
        total_pairs = 0
        adjacent_long_done = False
        multi_pair_done = False

        room_order = rooms[:]
        rng.shuffle(room_order)
        for room_index, bbox in enumerate(room_order):
            cycle = _clockwise_cycle(*bbox)
            capacity = len(cycle)
            max_pairs = min(3, capacity // 4)
            if max_pairs <= 0:
                continue
            if room_index == 0:
                num_pairs = min(max_pairs, 2)
            else:
                num_pairs = rng.randint(1, max_pairs)
            if num_pairs >= 2:
                multi_pair_done = True
            pairs = _choose_room_pairs(cycle, bbox, rng, num_pairs)
            for color, a, b in pairs:
                grid[a] = color
                grid[b] = color
                total_pairs += 1
                if _pair_category(a, b, bbox) == "adjacent_long":
                    adjacent_long_done = True

        if total_pairs < 2 or not adjacent_long_done or not multi_pair_done:
            continue

        try:
            _validate_input(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
