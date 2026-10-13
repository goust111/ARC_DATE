from __future__ import annotations

import heapq
import random
from collections import deque

import numpy as np


BG = 8
GREEN = 3
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
OBSTACLE_COLORS = (1, 2, 4, 5, 6, 7, 9)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_specs(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int]]]:
    specs: list[dict] = []
    dots: list[tuple[int, int]] = []

    for color in sorted(set(map(int, grid.flatten())) - {BG, GREEN}):
        comps = _components_mask(grid == color)
        assert comps, "each obstacle color must appear at least once"

        bars = [comp for comp in comps if comp.shape[0] > 1]
        singles = [comp for comp in comps if comp.shape[0] == 1]
        assert len(bars) == 1, "each obstacle color must have exactly one bar component"
        assert singles, "each obstacle color must have at least one isolated dot"

        bar = bars[0]
        box = _bbox(bar)
        r0, c0, r1, c1 = box
        assert r0 == r1 or c0 == c1, "bar must be horizontal or vertical"
        specs.append(
            {
                "color": int(color),
                "bbox": box,
                "radius": int(len(singles)),
            }
        )
        dots.extend((int(comp[0, 0]), int(comp[0, 1])) for comp in singles)

    return specs, dots


def _blocked_mask(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        radius = spec["radius"]
        rr0 = max(0, r0 - radius)
        cc0 = max(0, c0 - radius)
        rr1 = min(h - 1, r1 + radius)
        cc1 = min(w - 1, c1 + radius)
        blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return blocked


def _start_and_goal(grid: np.ndarray):
    points = np.argwhere(grid == GREEN)
    assert points.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, points[0])
    h, w = grid.shape

    if sr == 0:
        init_dir = 2
        is_goal = lambda r, c: r == h - 1
    elif sr == h - 1:
        init_dir = 0
        is_goal = lambda r, c: r == 0
    elif sc == 0:
        init_dir = 1
        is_goal = lambda r, c: c == w - 1
    elif sc == w - 1:
        init_dir = 3
        is_goal = lambda r, c: c == 0
    else:
        raise AssertionError("green start must lie on the border")

    return (sr, sc), init_dir, is_goal


def _best_path_from_state(
    blocked: np.ndarray,
    start: tuple[int, int],
    init_dir: int,
    is_goal,
) -> tuple[tuple[int, int] | None, list[tuple[int, int]], tuple[int, int, int] | None]:
    h, w = blocked.shape
    sr, sc = start
    start_state = (sr, sc, init_dir)

    if is_goal(sr, sc):
        return (0, 0), [(sr, sc)], start_state

    best: dict[tuple[int, int, int], tuple[int, int]] = {start_state: (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[int, int, int, int, int]] = [(0, 0, sr, sc, init_dir)]

    goal_state: tuple[int, int, int] | None = None
    goal_cost: tuple[int, int] | None = None

    while heap:
        turns, steps, r, c, dprev = heapq.heappop(heap)
        cost = (turns, steps)
        if cost != best.get((r, c, dprev)):
            continue
        if is_goal(r, c):
            goal_state = (r, c, dprev)
            goal_cost = cost
            break

        for nd, (dr, dc) in enumerate(DIRS):
            rr, cc = r + dr, c + dc
            if not _in_bounds((h, w), rr, cc) or blocked[rr, cc]:
                continue
            ncost = (turns + int(nd != dprev), steps + 1)
            key = (rr, cc, nd)
            if ncost < best.get(key, (10**9, 10**9)):
                best[key] = ncost
                prev[key] = (r, c, dprev)
                heapq.heappush(heap, (ncost[0], ncost[1], rr, cc, nd))

    if goal_state is None or goal_cost is None:
        return None, [], None

    path: list[tuple[int, int]] = []
    state = goal_state
    while True:
        r, c, _ = state
        path.append((r, c))
        if state == start_state:
            break
        state = prev[state]
    path.reverse()
    return goal_cost, path, goal_state


def _full_path(blocked: np.ndarray, start: tuple[int, int], init_dir: int, is_goal) -> list[tuple[int, int]]:
    initial_cost, _path, _state = _best_path_from_state(blocked, start, init_dir, is_goal)
    assert initial_cost is not None, "a path to the opposite edge must exist"
    target_turns = initial_cost[0]

    dr, dc = DIRS[init_dir]
    sr, sc = start
    best_prefix_len = 0
    best_state = start
    r, c = sr, sc
    prefix_len = 0

    while True:
        cost, _suffix, _goal = _best_path_from_state(blocked, (r, c), init_dir, is_goal)
        if cost is None or cost[0] != target_turns:
            break
        best_prefix_len = prefix_len
        best_state = (r, c)

        nr, nc = r + dr, c + dc
        if not _in_bounds(blocked.shape, nr, nc) or blocked[nr, nc]:
            break
        r, c = nr, nc
        prefix_len += 1

    final_cost, suffix, _goal = _best_path_from_state(blocked, best_state, init_dir, is_goal)
    assert final_cost is not None and final_cost[0] == target_turns

    prefix = [(sr + i * dr, sc + i * dc) for i in range(best_prefix_len + 1)]
    return prefix + suffix[1:]


def _solve_from_specs(grid: np.ndarray, specs: list[dict], dots: list[tuple[int, int]]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    start, init_dir, is_goal = _start_and_goal(g)
    blocked = _blocked_mask(g.shape, specs)
    blocked[start] = False

    path = _full_path(blocked, start, init_dir, is_goal)

    out = g.copy()
    for r, c in path:
        if out[r, c] != BG and out[r, c] != GREEN:
            raise AssertionError("path must stay off obstacle bars")
        out[r, c] = GREEN
    for r, c in dots:
        out[r, c] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route the green start pixel to the opposite edge while remote same-color dots encode each obstacle bar's keep-out radius.

    Input structure:
    - The background is azure (8) with exactly one green (3) start pixel on a border cell.
    - There are 2-3 obstacle bars, each a 1-pixel-thick horizontal or vertical line of a distinct non-azure, non-green color.
    - Each obstacle color also has several isolated same-color single pixels elsewhere in the grid, separated from the bar by 4-connectivity.

    Rule:
    - For each obstacle color, count its isolated single-pixel components; that count is the bar's Chebyshev exclusion radius.
    - Extend the green path from the border start to the opposite edge using 4-neighbor steps, never entering any cell whose Chebyshev distance to a bar pixel is less than or equal to that bar's radius.
    - Among legal paths, use the fewest turns; if several still tie, choose the one whose first turn happens farthest from the start.
    - Draw the path in green and erase all isolated encoding dots back to the azure background, leaving the bars unchanged.

    Key insight:
    - The scattered same-color dots are not noise: their counts are the hidden parameters that determine how much space each distant bar repels.
    """
    g = np.asarray(grid, dtype=int)
    specs, dots = _extract_specs(g)
    return _solve_from_specs(g, specs, dots)


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _min_main_len(radii: list[int]) -> int:
    row = radii[0] + 2
    for prev_radius, radius in zip(radii, radii[1:]):
        row += prev_radius + radius + 2
    return row + radii[-1] + 2


def _choose_lengths(radii: list[int], width: int) -> list[int] | None:
    options = []
    for radius in radii:
        hi = min(6, width - radius - 2)
        opts = list(range(3, hi + 1))
        if not opts:
            return None
        options.append(opts)

    for _ in range(300):
        lengths = [random.choice(opts) for opts in options]
        if all(
            lengths[i] + lengths[i + 1] + radii[i] + radii[i + 1] >= width
            for i in range(len(radii) - 1)
        ):
            return lengths
    return None


def _canonical_specs(
    height: int,
    width: int,
    radii: list[int],
    lengths: list[int],
    colors: list[int],
    first_side: int,
) -> tuple[list[dict], int]:
    slack = height - _min_main_len(radii)
    extras = _random_composition(slack, len(radii) + 1)
    row = radii[0] + 2 + extras[0]
    specs: list[dict] = []

    for i, (radius, length, color) in enumerate(zip(radii, lengths, colors)):
        side = (first_side + i) % 2
        if side == 0:
            c0, c1 = 0, length - 1
        else:
            c0, c1 = width - length, width - 1
        specs.append({"color": color, "bbox": (row, c0, row, c1), "radius": radius})
        if i + 1 < len(radii):
            row += radius + radii[i + 1] + 2 + extras[i + 1]

    if first_side == 0:
        start_col = random.randint(1, lengths[0] - 1)
    else:
        start_col = random.randint(width - lengths[0], width - 2)
    return specs, start_col


def _paint_canonical(height: int, width: int, specs: list[dict], start_col: int) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = spec["color"]
    grid[0, start_col] = GREEN
    return grid


def _path_mask(shape: tuple[int, int], path: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in path:
        mask[r, c] = True
    return mask


def _dilate_chebyshev(mask: np.ndarray, radius: int) -> np.ndarray:
    h, w = mask.shape
    out = np.array(mask, copy=True)
    rows, cols = np.where(mask)
    for r, c in zip(map(int, rows), map(int, cols)):
        rr0 = max(0, r - radius)
        rr1 = min(h - 1, r + radius)
        cc0 = max(0, c - radius)
        cc1 = min(w - 1, c + radius)
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return out


def _place_dots(
    grid: np.ndarray,
    specs: list[dict],
    path: list[tuple[int, int]],
) -> np.ndarray | None:
    h, w = grid.shape
    blocked = _blocked_mask((h, w), specs)
    path_forbidden = _dilate_chebyshev(_path_mask((h, w), path), 1)
    occupied = _dilate_chebyshev(grid != BG, 1)
    forbidden = blocked | path_forbidden | occupied
    forbidden[0, :] = True
    forbidden[-1, :] = True
    forbidden[:, 0] = True
    forbidden[:, -1] = True

    out = grid.copy()
    for spec in random.sample(specs, len(specs)):
        candidates = [tuple(map(int, rc)) for rc in np.argwhere(~forbidden)]
        random.shuffle(candidates)
        need = spec["radius"]
        picked = 0
        for r, c in candidates:
            if forbidden[r, c]:
                continue
            out[r, c] = spec["color"]
            rr0 = max(0, r - 1)
            rr1 = min(h - 1, r + 1)
            cc0 = max(0, c - 1)
            cc1 = min(w - 1, c + 1)
            forbidden[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            picked += 1
            if picked == need:
                break
        if picked != need:
            return None
    return out


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    if edge == "right":
        return np.fliplr(grid.T).copy()
    raise ValueError(f"unknown edge: {edge}")


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    if edge == "left":
        return bool(np.any(out[:, -1] == GREEN))
    if edge == "right":
        return bool(np.any(out[:, 0] == GREEN))
    raise ValueError(f"unknown edge: {edge}")


def _gen_attempt() -> np.ndarray | None:
    count = random.choice((2, 3))
    radii = random.sample([1, 2, 3], count)
    random.shuffle(radii)

    min_height = _min_main_len(radii)
    max_pair_width = min(
        15,
        min(12 + radii[i] + radii[i + 1] for i in range(count - 1)),
    )
    if max_pair_width < 12:
        return None

    height = random.randint(min_height, min(min_height + 5, 24))
    width = random.randint(12, max_pair_width)
    lengths = _choose_lengths(radii, width)
    if lengths is None:
        return None

    colors = random.sample(list(OBSTACLE_COLORS), count)
    first_side = random.randrange(2)
    canonical_specs, start_col = _canonical_specs(
        height,
        width,
        radii,
        lengths,
        colors,
        first_side,
    )
    canonical = _paint_canonical(height, width, canonical_specs, start_col)
    path = _full_path(_blocked_mask((height, width), canonical_specs), (0, start_col), 2, lambda r, c: r == height - 1)
    with_dots = _place_dots(canonical, canonical_specs, path)
    if with_dots is None:
        return None

    edge = random.choice(("top", "bottom", "left", "right"))
    inp = _orient_grid(with_dots, edge)

    try:
        out = main(inp)
    except Exception:
        return None

    if np.array_equal(inp, out):
        return None
    if not _reaches_opposite_edge(out, edge):
        return None
    if not np.array_equal(out, main(inp)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
