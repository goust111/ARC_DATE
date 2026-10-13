from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
MAGENTA = 6

SOURCE_COLORS = (RED, BLUE)
SCENARIOS = ("single", "same_overlap", "diff_overlap", "triple_mix")
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _cycle_neighbors(points: set[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    graph: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for pt in points:
        nbrs = [nbr for nbr in _neighbors4(*pt) if nbr in points]
        graph[pt] = nbrs
    return graph


def _trace_cycle(component: list[tuple[int, int]]) -> list[tuple[int, int]]:
    points = set(component)
    graph = _cycle_neighbors(points)
    assert all(len(nbrs) == 2 for nbrs in graph.values()), "component must be a simple 4-cycle"

    start = min(points)
    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start

    while True:
        order.append(cur)
        nbrs = graph[cur]
        if prev is None:
            nxt = min(nbrs)
        else:
            nxt = nbrs[0] if nbrs[1] == prev else nbrs[1]
        prev, cur = cur, nxt
        if cur == start:
            break

    assert len(order) == len(points), "failed to traverse entire contour"
    return order


def _source_runs(order: list[tuple[int, int]], grid: np.ndarray) -> list[tuple[int, int, int]]:
    n = len(order)
    colors = [int(grid[r, c]) for r, c in order]
    starts = [
        i
        for i in range(n)
        if colors[i] in SOURCE_COLORS and colors[(i - 1) % n] != colors[i]
    ]
    if not starts:
        if all(color == GREY for color in colors):
            return []
        if all(color == colors[0] for color in colors) and colors[0] in SOURCE_COLORS:
            return [(0, n, colors[0])]
        raise AssertionError("could not identify source segments")

    runs: list[tuple[int, int, int]] = []
    seen: set[int] = set()
    for start in starts:
        if start in seen:
            continue
        color = colors[start]
        length = 0
        while length < n and colors[(start + length) % n] == color:
            seen.add((start + length) % n)
            length += 1
        runs.append((start, length, color))
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate waves along each closed contour. The source segment length is the
    propagation distance on both sides; red/blue overlaps mix into magenta.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for comp in _components(g != BLACK):
        order = _trace_cycle(comp)
        n = len(order)
        coverage = [set() for _ in range(n)]

        for start, length, color in _source_runs(order, g):
            end = (start + length - 1) % n
            for step in range(1, length + 1):
                coverage[(start - step) % n].add(color)
                coverage[(end + step) % n].add(color)

        for idx, (r, c) in enumerate(order):
            if int(g[r, c]) != GREY:
                continue
            colors = coverage[idx]
            if RED in colors and BLUE in colors:
                out[r, c] = MAGENTA
            elif RED in colors:
                out[r, c] = RED
            elif BLUE in colors:
                out[r, c] = BLUE

    return out


def _build_monotone_path(rows: list[int]) -> list[tuple[int, int]]:
    path = [(int(rows[0]), 0)]
    cur = int(rows[0])
    for c in range(len(rows) - 1):
        nxt = int(rows[c + 1])
        if nxt != cur:
            step = 1 if nxt > cur else -1
            for r in range(cur + step, nxt + step, step):
                path.append((int(r), int(c)))
        path.append((int(nxt), int(c + 1)))
        cur = nxt
    return path


def _rotate_cycle(
    cycle: list[tuple[int, int]], height: int, width: int, turns: int
) -> tuple[list[tuple[int, int]], int, int]:
    turns %= 4
    if turns == 0:
        return list(cycle), height, width
    if turns == 1:
        return [(c, height - 1 - r) for r, c in cycle], width, height
    if turns == 2:
        return [(height - 1 - r, width - 1 - c) for r, c in cycle], height, width
    return [(width - 1 - c, r) for r, c in cycle], width, height


def _make_loop(rng: random.Random) -> tuple[list[tuple[int, int]], int, int] | None:
    height = rng.randint(6, 11)
    width = rng.randint(8, 15)

    upper = [rng.randint(0, max(0, height // 3))]
    lower_min = max(upper[0] + 3, height // 2)
    if lower_min > height - 1:
        return None
    lower = [rng.randint(lower_min, height - 1)]

    for _ in range(width - 1):
        valid: list[tuple[int, int]] = []
        for nu in range(max(0, upper[-1] - 1), min(height - 4, upper[-1] + 1) + 1):
            low_lo = max(nu + 3, lower[-1] - 1)
            low_hi = min(height - 1, lower[-1] + 1)
            for nl in range(low_lo, low_hi + 1):
                if min(nl, lower[-1]) - max(nu, upper[-1]) >= 2:
                    valid.append((int(nu), int(nl)))
        if not valid:
            return None
        nu, nl = rng.choice(valid)
        upper.append(nu)
        lower.append(nl)

    if all(upper[i] == upper[0] for i in range(width)) and all(lower[i] == lower[0] for i in range(width)):
        return None

    top_path = _build_monotone_path(upper)
    bottom_path = _build_monotone_path(lower)

    right_connector = [(r, width - 1) for r in range(upper[-1] + 1, lower[-1] + 1)]
    left_connector = [(r, 0) for r in range(lower[0] - 1, upper[0], -1)]
    cycle = top_path + right_connector + list(reversed(bottom_path[:-1])) + left_connector

    points = set(cycle)
    if len(points) != len(cycle):
        return None
    graph = _cycle_neighbors(points)
    if not all(len(nbrs) == 2 for nbrs in graph.values()):
        return None

    cycle, height, width = _rotate_cycle(cycle, height, width, rng.randrange(4))
    return cycle, height, width


def _rects_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _make_plan(n: int, scenario: str, rng: random.Random) -> tuple[list[int], list[int], list[int]] | None:
    for _ in range(120):
        if scenario == "single":
            max_len = min(5, (n - 3) // 3)
            if max_len < 1:
                return None
            length = rng.randint(1, max_len)
            gap = n - length
            if gap < 2 * length + 2:
                continue
            return [length], [rng.choice(SOURCE_COLORS)], [gap]

        if scenario in {"same_overlap", "diff_overlap"}:
            max_len = min(5, max(1, (n - 4) // 4))
            l1 = rng.randint(1, max_len)
            l2 = rng.randint(1, max_len)
            total = l1 + l2
            max_small = min(total - 1, n - 2 * total - 1)
            if max_small < 1:
                continue
            g1 = rng.randint(1, max_small)
            g2 = n - total - g1
            if g2 < total + 1:
                continue
            if scenario == "same_overlap":
                color = rng.choice(SOURCE_COLORS)
                colors = [color, color]
            else:
                colors = [RED, BLUE]
                rng.shuffle(colors)
            return [l1, l2], colors, [g1, g2]

        if scenario == "triple_mix":
            l1 = rng.randint(1, min(3, max(1, (n - 7) // 6 + 1)))
            l2 = rng.randint(1, min(3, max(1, (n - 7) // 6 + 1)))
            l3 = rng.randint(1, min(3, max(1, (n - 7) // 6 + 1)))
            min_g2 = l2 + l3 + 1
            min_g3 = l3 + l1 + 1
            free = n - (l1 + l2 + l3) - min_g2 - min_g3
            max_g1 = min(l1 + l2 - 1, free)
            if max_g1 < 1:
                continue
            g1 = rng.randint(1, max_g1)
            extra = free - g1
            add2 = rng.randint(0, extra)
            add3 = extra - add2
            colors = [RED, BLUE, rng.choice(SOURCE_COLORS)]
            if rng.random() < 0.5:
                colors.reverse()
            return [l1, l2, l3], colors, [g1, min_g2 + add2, min_g3 + add3]

    return None


def _apply_plan(
    grid: np.ndarray,
    cycle: list[tuple[int, int]],
    lengths: list[int],
    colors: list[int],
    gaps: list[int],
    rng: random.Random,
) -> None:
    n = len(cycle)
    assert sum(lengths) + sum(gaps) == n
    offset = rng.randrange(n)
    pos = offset
    for length, color, gap in zip(lengths, colors, gaps):
        for step in range(length):
            r, c = cycle[(pos + step) % n]
            grid[r, c] = int(color)
        pos = (pos + length + gap) % n


def _scenario_order(num_loops: int, rng: random.Random) -> list[str]:
    pool = list(SCENARIOS)
    rng.shuffle(pool)
    return [pool[i % len(pool)] for i in range(num_loops)]


def _validate_input(grid: np.ndarray) -> bool:
    comps = _components(grid != BLACK)
    if not comps:
        return False
    for comp in comps:
        try:
            order = _trace_cycle(comp)
            runs = _source_runs(order, grid)
        except AssertionError:
            return False
        if not (1 <= len(runs) <= 3):
            return False
        for _start, length, color in runs:
            if color not in SOURCE_COLORS or not (1 <= length <= 5):
                return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(18, 30)
    width = rng.randint(20, 34)
    num_loops = rng.randint(1, 3)
    grid = np.full((height, width), BLACK, dtype=int)

    placed_boxes: list[tuple[int, int, int, int]] = []
    scenarios = _scenario_order(num_loops, rng)
    wants_magenta = False

    for scenario in scenarios:
        loop_spec = None
        for _ in range(120):
            loop_spec = _make_loop(rng)
            if loop_spec is not None:
                break
        if loop_spec is None:
            return None
        local_cycle, box_h, box_w = loop_spec
        plan = _make_plan(len(local_cycle), scenario, rng)
        if plan is None:
            return None

        placed = False
        for _ in range(120):
            if box_h >= height - 2 or box_w >= width - 2:
                return None
            r0 = rng.randint(1, height - box_h - 1)
            c0 = rng.randint(1, width - box_w - 1)
            bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            if all(_rects_separate(bbox, other, 2) for other in placed_boxes):
                placed_boxes.append(bbox)
                global_cycle = [(r0 + r, c0 + c) for r, c in local_cycle]
                for r, c in global_cycle:
                    grid[r, c] = GREY
                _apply_plan(grid, global_cycle, *plan, rng)
                wants_magenta |= scenario in {"diff_overlap", "triple_mix"}
                placed = True
                break
        if not placed:
            return None

    if not _validate_input(grid):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if wants_magenta and not np.any(out == MAGENTA):
        return None
    if not np.any(grid == GREY) or not np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
