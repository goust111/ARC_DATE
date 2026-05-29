# Combo variant of cb2d8a2c (index 33)
# combo_spec: D11/2+D17/2
# Name: 布尔掩码输出+路径填充禁区

from __future__ import annotations

# Design analysis for combo D11/2 + D17/2
# a) Concrete behavior trace:
#    In the source task, the answer is still an overpainted copy of the input:
#    a 1-pixel green polyline snakes from one border to the opposite border,
#    keeping Manhattan distance > blue_count from each red/blue bar, and blue
#    pixels are then recolored to red. This combo changes both the rendered
#    object and the output framing. We still recover the same edge-to-edge route
#    family, but each "vertical detour leg" is expanded into the whole safe
#    corridor on that bar's open side, so the answer is a wide strip rather than
#    a thin line. Then D11/2 removes the original scene entirely and renders a
#    boolean mask: the expanded route band becomes green and every non-band cell
#    becomes red. Example: if a top-start path must go right of a left-attached
#    bar and then left of a right-attached bar, the original output is a thin
#    zig-zag on azure, while the fused output is a two-color region map whose
#    green area occupies those two open-side corridors plus the thin turn rows
#    that connect them.
# b) Complexity calibration:
#    If we only recolored the original 1-pixel route, D17/2 would be visually
#    latent. If we flood every reachable safe cell, the output would ignore the
#    chosen bypass side and collapse into a trivial complement mask. The chosen
#    design sits between those extremes: bar-side corridors become visibly wide,
#    but the turn rows stay thin, so the band still exposes the same zig-zag
#    decisions as the source task.
# c) Hidden assumptions audit:
#    The source solver assumed "output == input size and preserves obstacles",
#    "painting happens only on bg cells", and "blue->red is visible in the
#    answer". All three assumptions are false now, so the new solver computes
#    the route on the input grid but renders onto a fresh binary canvas.
#    Another implicit assumption was "a 1-cell route is enough to reveal the
#    obstacle-side choices"; D17/2 breaks that on purpose, so the generator must
#    enforce corridor widths > 1 or the fused behavior would degenerate back to
#    the original line drawing.
# d) Placement legality:
#    The generator keeps the original bar family: 1-4 horizontal/vertical bars,
#    alternating attached walls, spacing derived from blue counts, and a single
#    border start pixel. It additionally enforces a minimum open-corridor width
#    so the widened mask never overlaps a bar, never leaves the grid, and stays
#    visibly thicker than the original path. Each candidate is validated by the
#    fused solver before acceptance.
# e) Visual observability:
#    Every accepted sample shows a multicolor obstacle scene as input but a
#    strict two-color mask as output. The green region is much larger than the
#    recovered centerline and contains repeated wide strips aligned with the
#    bars' open sides, which makes both D17/2 and D11/2 inferable from only a
#    few examples.
# f) Visual cleanliness:
#    The band occupies one corridor at a time and uses the original turn rows as
#    narrow connectors, so the output reads as one coherent serpentine region
#    instead of a noisy flood over the full canvas. Red is reserved exclusively
#    for the complement class in the output, which keeps the binary partition
#    easy to parse.
# g) Inferability of the change:
#    No ordering uses hidden palette semantics such as numeric color ids. The
#    same visible features as in the source task still drive the route choice:
#    blue count controls keep-out radius, alternating wall attachment controls
#    which corridor opens, and the answer simply re-encodes that recovered route
#    as a thick two-class region.

import heapq
import itertools
import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4


BG = 8
BLUE = 1
RED = 2
GREEN = 3

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# --- Tunable parameters (generator) ---
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 1, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MIN_CORRIDOR_WIDTH = 3
START_OPEN_SLACK = 3
MAX_GEN_RETRIES = 400
MIN_BAND_GAIN = 8
MAX_BAND_FRACTION = 0.78


def _detect_start_edge(grid: np.ndarray) -> tuple[str, tuple[int, int]]:
    starts = np.argwhere(grid == GREEN)
    assert starts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, starts[0])
    h, w = grid.shape
    if sr == 0:
        return "top", (sr, sc)
    if sr == h - 1:
        return "bottom", (sr, sc)
    if sc == 0:
        return "left", (sr, sc)
    if sc == w - 1:
        return "right", (sr, sc)
    raise AssertionError("green start must be on a border edge")


def _to_canonical(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return np.array(grid, copy=True)
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid).T.copy()


def _from_canonical(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return np.array(grid, copy=True)
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _build_blocked(grid: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    bar_mask = (grid == BLUE) | (grid == RED)
    bar_comps = components_mask(bar_mask, connectivity=4)

    forbidden = np.zeros((h, w), dtype=bool)
    for comp in bar_comps:
        blue_count = sum(1 for r, c in comp if grid[int(r), int(c)] == BLUE)
        radius = int(blue_count)

        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in comp:
            rr, cc = int(r), int(c)
            dist[rr, cc] = 0
            q.append((rr, cc))

        while q:
            r, c = q.popleft()
            d = int(dist[r, c])
            forbidden[r, c] = True
            if d == radius:
                continue
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] == -1:
                    dist[rr, cc] = d + 1
                    q.append((rr, cc))

    blocked = bar_mask | forbidden
    blocked[start] = False
    return blocked


def _trace_shortest_path(blocked: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = blocked.shape
    sr, sc = start
    init_dir = 2

    def is_goal(r: int, c: int) -> bool:
        return r == h - 1

    inf = (10**9, 10**9)
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]

    goal_state: tuple[int, int, int] | None = None
    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), inf):
            continue
        if is_goal(r, c) and (r, c) != start:
            goal_state = (r, c, dprev)
            break
        for nd, (dr, dc) in enumerate(_DIRS_URDL):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if blocked[rr, cc]:
                continue
            ncost = (steps + 1, turns + (0 if nd == dprev else 1))
            key = (rr, cc, nd)
            if ncost < best_cost.get(key, inf):
                best_cost[key] = ncost
                prev[key] = (r, c, dprev)
                heapq.heappush(heap, (ncost, rr, cc, nd))

    assert goal_state is not None, "a valid path to the opposite edge must exist"
    path: list[tuple[int, int]] = []
    state = goal_state
    while True:
        r, c, _ = state
        path.append((r, c))
        if (r, c) == start:
            break
        state = prev[state]
    path.reverse()
    return path


def _bar_infos_canonical(grid: np.ndarray) -> list[dict[str, int | tuple[int, int]]]:
    h, w = grid.shape
    bar_mask = (grid == BLUE) | (grid == RED)
    infos: list[dict[str, int | tuple[int, int]]] = []
    for comp in components_mask(bar_mask, connectivity=4):
        rows = sorted({int(r) for r, _c in comp})
        cols = sorted(int(c) for _r, c in comp)
        assert len(rows) == 1, "canonical bars must be horizontal"
        row = rows[0]
        c0, c1 = cols[0], cols[-1]
        blue_count = sum(1 for r, c in comp if grid[int(r), int(c)] == BLUE)
        side = 0 if c0 == 0 else 1
        radius = int(blue_count)
        top = row - radius
        bottom = row + radius
        if side == 0:
            corridor = (c1 + radius + 1, w - 1)
        else:
            corridor = (0, c0 - radius - 1)
        assert 0 <= top <= bottom < h
        assert 0 <= corridor[0] <= corridor[1] < w
        infos.append(
            {
                "row": row,
                "side": side,
                "blue_count": int(blue_count),
                "length": c1 - c0 + 1,
                "top": top,
                "bottom": bottom,
                "corridor": corridor,
            }
        )
    infos.sort(key=lambda item: int(item["row"]))
    return infos


def _solve_canonical_state(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, copy=True)
    h, _w = g.shape
    start = _detect_start_edge(g)[1]
    assert start[0] == 0, "canonical grid must start from the top edge"

    bars = _bar_infos_canonical(g)
    assert bars, "expected at least one bar"

    blocked = _build_blocked(g, start)
    path = _trace_shortest_path(blocked, start)

    band = np.zeros_like(g, dtype=bool)
    for r, c in path:
        band[r, c] = True

    for i, bar in enumerate(bars):
        top = int(bar["top"])
        if i + 1 < len(bars):
            next_top = int(bars[i + 1]["top"])
            active_end = next_top - 2
        else:
            active_end = h - 1
        if top > active_end:
            continue
        c0, c1 = map(int, bar["corridor"])
        band[top : active_end + 1, c0 : c1 + 1] = True

    return {
        "bars": bars,
        "path": path,
        "path_len": len(path),
        "band_mask": band,
        "blocked": blocked,
    }


def _solve_state(grid: np.ndarray) -> dict[str, object]:
    edge, _start = _detect_start_edge(grid)
    canonical = _to_canonical(grid, edge)
    state = _solve_canonical_state(canonical)
    band_mask = _from_canonical(state["band_mask"], edge)
    out = dict(state)
    out["edge"] = edge
    out["band_mask"] = band_mask
    return out


def _max_true_run(mask: np.ndarray) -> int:
    best = 0
    for row in mask:
        run = 0
        for v in row:
            if v:
                run += 1
                best = max(best, run)
            else:
                run = 0
    for col in mask.T:
        run = 0
        for v in col:
            if v:
                run += 1
                best = max(best, run)
            else:
                run = 0
    return int(best)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert the detour route into a binary region map whose foreground is a widened bypass band.

    Input structure:
    - bg = azure (8). Red/blue bars (2/1) are one-cell-thick obstacles attached to alternating side walls.
    - A single green (3) pixel sits on one border edge and indicates the route's start side.

    Rule:
    - Each bar's blue count still defines its Manhattan keep-out radius, and the same minimum-turn edge-to-edge route is recovered first.
    - Instead of leaving that route as a 1-pixel polyline, expand each detour leg into the whole safe corridor on the bar's open side; the turn rows stay as the thin connectors between those widened strips.
    - Render a boolean mask rather than an overpainted scene: the expanded route band is green and every other cell is red.

    Key insight: The obstacle geometry is only used to infer which open-side corridor the original path would choose; once that corridor chain is known, the answer is just its binary mask.
    """
    state = _solve_state(np.array(grid, copy=True))
    out = np.full(np.array(grid).shape, RED, dtype=int)
    out[state["band_mask"]] = GREEN
    return out


def _min_main_len(blues: tuple[int, ...]) -> int:
    last_bar = blues[0] + 2
    for a, b in zip(blues, blues[1:]):
        last_bar += a + b + 2
    return last_bar + 2


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


def _choose_blues(count: int, main_hi: int) -> tuple[int, ...]:
    choices = [
        combo
        for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count)
        if _min_main_len(combo) <= main_hi
    ]
    return random.choice(choices)


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, 2 * blue_count - 1, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(
        BAR_LEN_MAX,
        cross_size - blue_count - MIN_CORRIDOR_WIDTH,
        (BAR_CROSS_PCT_HI * cross_size) // 100,
    )
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int) -> None:
    w = grid.shape[1]
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, w - length :] = colors[::-1]


def _canonical_grid(
    main_size: int,
    cross_size: int,
    blues: tuple[int, ...],
    first_side: int,
) -> np.ndarray | None:
    count = len(blues)
    slack = main_size - _min_main_len(blues)
    if slack < 0:
        return None

    extras = _random_composition(slack, count + 1)
    row = blues[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, count):
        row += blues[i - 1] + blues[i] + 2 + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    lengths: list[int] = []
    sides = [(first_side + i) % 2 for i in range(count)]
    for row, blue_count, side in zip(rows, blues, sides):
        bounds = _length_bounds(cross_size, blue_count)
        if bounds is None:
            return None
        length = random.randint(*bounds)
        lengths.append(length)
        _paint_bar(grid, row, side, length, blue_count)

    first_len = lengths[0]
    if first_side == 0:
        lo = max(1, first_len - START_OPEN_SLACK)
        hi = first_len - 1
    else:
        lo = cross_size - first_len
        hi = min(cross_size - 2, cross_size - first_len + START_OPEN_SLACK - 1)
    if lo > hi:
        return None
    start_col = random.randint(lo, hi)
    grid[0, start_col] = GREEN
    return grid


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    return _from_canonical(grid, edge)


def _reaches_opposite_edge(mask: np.ndarray, edge: str) -> bool:
    if edge == "top":
        return bool(np.any(mask[-1, :]))
    if edge == "bottom":
        return bool(np.any(mask[0, :]))
    if edge == "left":
        return bool(np.any(mask[:, -1]))
    return bool(np.any(mask[:, 0]))


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    count = random.randint(BAR_COUNT_LO, BAR_COUNT_HI)
    blues = _choose_blues(count, main_hi)
    main_size = random.randint(max(main_lo, _min_main_len(blues)), main_hi)
    cross_size = random.randint(cross_lo, cross_hi)
    first_side = random.randrange(2)

    canonical = _canonical_grid(main_size, cross_size, blues, first_side)
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)

    try:
        state = _solve_state(inp)
        band_mask = np.array(state["band_mask"], copy=False)
        path_len = int(state["path_len"])
        band_area = int(band_mask.sum())
        band_fraction = band_area / band_mask.size
        if not _reaches_opposite_edge(band_mask, edge):
            return None
        if band_area <= path_len + MIN_BAND_GAIN:
            return None
        if _max_true_run(band_mask) < MIN_CORRIDOR_WIDTH:
            return None
        if band_fraction >= MAX_BAND_FRACTION:
            return None
        out = main(inp)
        if len(np.unique(out)) != 2:
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c combo input")
