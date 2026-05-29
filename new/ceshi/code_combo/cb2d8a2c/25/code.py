# Combo variant of cb2d8a2c (index 25)
# combo_spec: D1/0+D7/1+D14/0
# Name: 全对角线几何范式
#
# Design analysis for the fused variant
# a) Concrete behavior trace:
#    I model the new task in a virtual square grid where the original orthogonal
#    logic is still easy to express, then embed that grid into the actual ARC
#    canvas by the 45-degree map
#        actual_r = vr + vc
#        actual_c = vc - vr + (n - 1)
#    A virtual right step becomes actual SE, a virtual down step becomes actual SW,
#    and a virtual horizontal bar becomes a diagonal "\" bar. Example with n=5:
#      - virtual path (0,0)->(0,1)->(1,1) maps to actual (0,4)->(1,5)->(2,4):
#        two diagonal moves with a 45-degree turn.
#      - virtual bar cells (2,0)(2,1)(2,2) map to actual (2,2)(3,3)(4,4), i.e.
#        a diagonal staircase obstacle.
#    The key metric identity is exact, not approximate:
#        Manhattan_virtual((a,b),(c,d)) == Chebyshev_actual(F(a,b), F(c,d))
#    so the old radius rule turns into the required D14/0 Chebyshev rule.
#
# b) Complexity calibration:
#    The variant is intentionally not a cosmetic rewrite. Bars are 8-connected but
#    not 4-connected in the actual grid, the path is painted only on one checkerboard
#    parity, and the route is a diagonal zig-zag rather than an L-shaped orthogonal
#    track. I keep 1-3 bars and 1-3 blue counts so examples remain learnable while
#    still providing repeated evidence of alternating diagonal detours.
#
# c) Hidden assumptions audited from the original:
#    - 4-connected obstacle extraction no longer works: a diagonal bar would split
#      into single pixels, so the actual solver canonicalizes and inverse-maps to
#      virtual space before extracting 4-connected bars there.
#    - Manhattan flood-fill on the actual grid is wrong after the fusion; the
#      solver therefore computes forbidden cells in virtual Manhattan space, which
#      is exactly actual Chebyshev space under the transform above.
#    - The original had one inward direction from the border. Here the top-border
#      start has two inward diagonals, so the virtual solver seeds both right/down
#      directions with zero turn cost.
#
# d) Placement legality:
#    The generator only places virtual bars, where overlap and bounds are trivial
#    to check. The 45-degree embedding is injective on valid virtual cells, so no
#    two objects collide after mapping into the actual canvas.
#
# e) Visual observability:
#    A human sees three coupled signatures at once: diagonal green path, diagonal
#    red/blue obstacle bars, and square-style keepout spacing around each bar end.
#    This makes the fused change visible from 2-3 examples instead of hidden in an
#    unobservable internal distance function.
#
# f) Visual cleanliness:
#    Every semantic object lives on the same parity sublattice of the square canvas.
#    That leaves one-cell breathing room between neighboring diagonals and avoids
#    ambiguous overlaps or dense blobs.
#
# g) New solver assumptions and generator guarantees:
#    The solver now assumes the instance is a rotated/checkerboard embedding of a
#    square virtual grid with a unique border start. The generator guarantees that
#    shape, keeps the mapped lattice parity consistent, and validates every sample
#    by running `main()` twice for stability. Within the chosen 13-25 actual-size
#    envelope, 4 bars mostly collapse into rejected layouts, so the realizable
#    range is intentionally tightened to 1-3 bars.

from __future__ import annotations

import heapq
import itertools
import random

import numpy as np

from solvers.arc_utils import components_mask


BG = 8
BLUE = 1
RED = 2
GREEN = 3

_VIRTUAL_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _virtual_to_actual(vr: int, vc: int, n: int) -> tuple[int, int]:
    return vr + vc, vc - vr + (n - 1)


def _actual_to_virtual(ar: int, ac: int, n: int) -> tuple[int, int]:
    x_num = ar - ac + (n - 1)
    y_num = ar + ac - (n - 1)
    assert x_num % 2 == 0 and y_num % 2 == 0, "colored cells must lie on the diagonal lattice"
    vr = x_num // 2
    vc = y_num // 2
    assert 0 <= vr < n and 0 <= vc < n, "colored cell fell outside the virtual diamond"
    return vr, vc


def _start_rotation(g: np.ndarray) -> int:
    pts = np.argwhere(g == GREEN)
    assert pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, pts[0])
    h, w = g.shape
    assert h == w and h % 2 == 1, "diagonal variant uses an odd square canvas"
    if sr == 0:
        return 0
    if sc == w - 1:
        return 1
    if sr == h - 1:
        return 2
    if sc == 0:
        return 3
    raise AssertionError("green start must lie on a border edge")


def _to_virtual_from_canonical(actual: np.ndarray) -> np.ndarray:
    h, w = actual.shape
    assert h == w and h % 2 == 1
    n = (h + 1) // 2
    virt = np.full((n, n), BG, dtype=int)
    for ar, ac in np.argwhere(actual != BG):
        vr, vc = _actual_to_virtual(int(ar), int(ac), n)
        virt[vr, vc] = int(actual[ar, ac])
    return virt


def _paint_virtual_path(virt: np.ndarray) -> list[tuple[int, int]]:
    n = virt.shape[0]
    start = (0, 0)
    goal = (n - 1, n - 1)

    bar_mask = (virt == BLUE) | (virt == RED)
    bar_comps = components_mask(bar_mask, connectivity=4)

    forbidden = np.zeros((n, n), dtype=bool)
    rr, cc = np.indices((n, n))
    for comp in bar_comps:
        blue_count = sum(1 for r, c in comp if virt[r, c] == BLUE)
        radius = int(blue_count)
        if radius < 0:
            continue
        comp_arr = np.asarray(comp, dtype=int)
        dist = np.min(np.abs(rr[..., None] - comp_arr[:, 0]) + np.abs(cc[..., None] - comp_arr[:, 1]), axis=2)
        forbidden |= dist <= radius

    blocked = bar_mask | forbidden
    blocked[start] = False
    blocked[goal] = False

    inf = (10**9, 10**9)
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = []

    for init_dir in (1, 2):  # right, down
        state = (start[0], start[1], init_dir)
        best_cost[state] = (0, 0)
        heapq.heappush(heap, ((0, 0), start[0], start[1], init_dir))

    goal_state: tuple[int, int, int] | None = None
    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        state = (r, c, dprev)
        if (steps, turns) != best_cost.get(state, inf):
            continue
        if (r, c) == goal and (r, c) != start:
            goal_state = state
            break
        for nd, (dr, dc) in enumerate(_VIRTUAL_DIRS_URDL):
            rr1, cc1 = r + dr, c + dc
            if not (0 <= rr1 < n and 0 <= cc1 < n):
                continue
            if blocked[rr1, cc1]:
                continue
            ncost = (steps + 1, turns + (0 if nd == dprev else 1))
            key = (rr1, cc1, nd)
            if ncost < best_cost.get(key, inf):
                best_cost[key] = ncost
                prev[key] = state
                heapq.heappush(heap, (ncost, rr1, cc1, nd))

    assert goal_state is not None, "a valid diagonal route to the opposite edge must exist"

    path: list[tuple[int, int]] = []
    st = goal_state
    while True:
        r, c, _ = st
        path.append((r, c))
        if (r, c) == start:
            break
        st = prev[st]
    path.reverse()
    return path


def _solve_canonical(actual: np.ndarray) -> np.ndarray:
    virt = _to_virtual_from_canonical(actual)
    n = virt.shape[0]
    path = _paint_virtual_path(virt)

    out = np.array(actual, copy=True)
    for vr, vc in path:
        ar, ac = _virtual_to_actual(vr, vc, n)
        if out[ar, ac] == BG:
            out[ar, ac] = GREEN
    out[out == BLUE] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a green border seed through a fully diagonal obstacle field, using a square-distance safety rule encoded by blue pixels.

    Input structure:
    - bg = azure (8). The non-background content lives on a 45-degree checkerboard lattice inside an odd square canvas.
    - A single green (3) pixel sits on one outer border and marks the path origin.
    - Each obstacle bar is a diagonal red/blue line; blue and red cells belong to the same bar.

    Rule:
    - A bar's danger radius equals its number of blue pixels.
    - Distances are measured in Chebyshev terms in the visible grid, so each bar forbids a square-shaped keep-out halo.
    - The green path reaches the opposite border by moving only along diagonal neighbor steps, choosing the shortest route and preferring fewer turns.
    - After routing, all blue pixels are recolored red.

    Key insight: The whole puzzle is the original bar-avoidance idea viewed in a 45-degree coordinate system, so diagonal motion, diagonal bars, and Chebyshev spacing are one coupled geometric rule rather than three independent tweaks.
    """
    g = np.array(grid, copy=True)
    k = _start_rotation(g)
    canonical = np.rot90(g, k)
    solved = _solve_canonical(canonical)
    return np.rot90(solved, -k)


# --------------- generator ---------------

# Virtual-grid parameters. After the 45-degree embedding, actual size is (2*n-1)^2.
VIRTUAL_SIDE_LO, VIRTUAL_SIDE_HI = 7, 13
BAR_COUNT_LO, BAR_COUNT_HI = 1, 3
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 3
BAR_LEN_MIN, BAR_LEN_MAX = 3, 9
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 40, 82
MAX_GEN_RETRIES = 400


def _min_virtual_side(blues: tuple[int, ...]) -> int:
    last_bar = blues[0] + 1
    for a, b in zip(blues, blues[1:]):
        last_bar += a + b + 1
    return last_bar + 2


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_blues(count: int, side_hi: int, blue_hi: int | None = None) -> tuple[int, ...]:
    if blue_hi is None:
        blue_hi = BLUE_PER_BAR_HI
    choices = [
        combo
        for combo in itertools.product(range(BLUE_PER_BAR_LO, blue_hi + 1), repeat=count)
        if _min_virtual_side(combo) <= side_hi
    ]
    if not choices:
        raise ValueError("no blue-count assignment fits the virtual side bound")
    return random.choice(choices)


def _length_bounds(side: int, blue_count: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, 2 * blue_count - 1, (BAR_CROSS_PCT_LO * side + 99) // 100)
    hi = min(BAR_LEN_MAX, side - blue_count - 1, (BAR_CROSS_PCT_HI * side) // 100)
    if lo > hi:
        return None
    return lo, hi


def _paint_virtual_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int) -> None:
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, grid.shape[1] - length :] = colors[::-1]


def _build_virtual_grid(side: int, blues: tuple[int, ...]) -> np.ndarray | None:
    count = len(blues)
    slack = side - _min_virtual_side(blues)
    if slack < 0:
        return None

    extras = _random_composition(slack, count + 1)
    row = blues[0] + 1 + extras[0]
    rows = [row]
    for i in range(1, count):
        row += blues[i - 1] + blues[i] + 1 + extras[i]
        rows.append(row)

    grid = np.full((side, side), BG, dtype=int)
    for row, blue_count, side_flag in zip(rows, blues, (i % 2 for i in range(count))):
        bounds = _length_bounds(side, blue_count)
        if bounds is None:
            return None
        length = random.randint(*bounds)
        _paint_virtual_bar(grid, row, side_flag, length, blue_count)

    grid[0, 0] = GREEN
    return grid


def _virtual_to_actual_grid(virt: np.ndarray) -> np.ndarray:
    n = virt.shape[0]
    actual = np.full((2 * n - 1, 2 * n - 1), BG, dtype=int)
    for vr, vc in np.argwhere(virt != BG):
        ar, ac = _virtual_to_actual(int(vr), int(vc), n)
        actual[ar, ac] = int(virt[vr, vc])
    return actual


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "right":
        return np.rot90(grid, -1).copy()
    if edge == "bottom":
        return np.rot90(grid, 2).copy()
    return np.rot90(grid, 1).copy()  # left


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    if edge == "left":
        return bool(np.any(out[:, -1] == GREEN))
    return bool(np.any(out[:, 0] == GREEN))


def _gen_attempt(count: int) -> np.ndarray | None:
    edge = random.choice(("top", "right", "bottom", "left"))
    try:
        blue_hi = 2 if count >= 3 else BLUE_PER_BAR_HI
        blues = _choose_blues(count, VIRTUAL_SIDE_HI, blue_hi=blue_hi)
    except ValueError:
        return None
    side_lo = max(VIRTUAL_SIDE_LO, _min_virtual_side(blues))
    if count >= 3:
        side_lo = max(side_lo, VIRTUAL_SIDE_HI - 2)
    side = random.randint(side_lo, VIRTUAL_SIDE_HI)

    virt = _build_virtual_grid(side, blues)
    if virt is None:
        return None

    try:
        _ = _paint_virtual_path(virt)
    except Exception:
        return None

    actual = _virtual_to_actual_grid(virt)
    inp = _orient_grid(actual, edge)

    try:
        out = main(inp)
        if not _reaches_opposite_edge(out, edge):
            return None
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    target = random.choices([1, 2, 3], weights=[1, 3, 3], k=1)[0]
    order = [target] + [c for c in (3, 2, 1) if c != target]
    tries_per_count = max(1, MAX_GEN_RETRIES // len(order))
    for count in order:
        for _ in range(tries_per_count):
            grid = _gen_attempt(count)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid diagonal cb2d8a2c input")
