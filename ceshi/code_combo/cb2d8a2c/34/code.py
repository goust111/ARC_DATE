# Combo variant of cb2d8a2c (index 34)
# combo_spec: D1/2+D6/3
# Name: 封闭环路终止+螺旋式绕行

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4


# Design analysis for the fused variant
# a) Behavior trace:
#    - Original cb2d8a2c draws one shortest orthogonal route from a border seed to the opposite edge.
#    - The fused rule must stop by closure instead of by reaching the far edge, and each bypass must look
#      like a clockwise coil rather than a single in-out dodge.
#    - In a coarse ARC grid, chaining many such coils while also forcing a single-pixel simple cycle tends
#      to create self-touches. To keep the fusion legible, this variant uses one dominant wall-attached bar:
#      the green trace goes from the border into a short top cap, drops beside the bar, folds inward at the
#      bar's top side, drops past the bar tip, then returns on a second vertical lane to close at the top.
#      The loop is therefore both closed (D6/3) and visibly spiral/U-shaped around the obstacle (D1/2).
# b) Complexity calibration:
#    - Keeping the original blue-count clearance code is important: the loop's inner lane is anchored at the
#      first safe column beyond the bar's Manhattan keep-out zone, so larger blue counts visibly widen the
#      whole loop.
#    - Using a single dominant bar avoids ambiguous multi-lobe self-intersections that are hard to infer from
#      only 2-4 ARC examples, while still making the fused behavior obvious at a glance.
# c) Broken original assumptions:
#    - No "goal on opposite edge" remains.
#    - The start pixel is no longer a shortest-path source; it is just the border touch-point of the final loop.
#    - Dijkstra/turn minimization is removed because the new geometry is a deterministic closed motif derived
#      from one bar's side, row, and encoded radius.
# d) Placement legality:
#    - The generator enforces a single horizontal bar in canonical top-start orientation.
#    - It leaves at least six cells of open-side slack beyond the encoded keep-out radius, which is enough for
#      the three distinct spiral columns: inner, middle, and outer.
#    - The bar row is kept away from top/bottom borders so the top and bottom folds never clip the frame.
# e) Visual observability:
#    - Input still shows the original bar encoding: red/blue wall-attached bar plus one green border seed.
#    - Output clearly differs from the original task because green no longer extends to the far edge; instead it
#      closes back on itself and creates a boxed spiral lobe around the obstacle tip.
# f) Visual cleanliness:
#    - The chosen loop template is a provable degree-2 cycle on the cell graph, so no thick blobs or touching
#      diagonals appear.
# g) Inferability of changes:
#    - The only color change remains blue->red after solving, so the fusion does not add extra color mappings.
#      The new information is entirely geometric: "same bar code, but now the bypass loops back and closes."

BG = 8
BLUE = 1
RED = 2
GREEN = 3

ROWS_LO, ROWS_HI = 12, 26
COLS_LO, COLS_HI = 18, 26
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_PCT_LO, BAR_PCT_HI = 35, 70
MAX_GEN_RETRIES = 300


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("segments must be axis-aligned")


def _paint_polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        seg = _segment_cells(points[i], points[i + 1])
        path.extend(seg if i == 0 else seg[1:])
    return path


def _components_bar_data(grid: np.ndarray) -> list[dict[str, int]]:
    bar_mask = (grid == BLUE) | (grid == RED)
    comps = components_mask(bar_mask, connectivity=4)
    out: list[dict[str, int]] = []
    for comp in comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        out.append(
            {
                "r0": int(r0),
                "r1": int(r1),
                "c0": int(c0),
                "c1": int(c1),
                "blue": int(sum(1 for r, c in comp if grid[r, c] == BLUE)),
            }
        )
    out.sort(key=lambda d: (d["r0"], d["c0"]))
    return out


def _blocked_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    bar_mask = (grid == BLUE) | (grid == RED)
    comps = components_mask(bar_mask, connectivity=4)
    blocked = np.zeros((h, w), dtype=bool)
    for comp in comps:
        radius = int(sum(1 for r, c in comp if grid[r, c] == BLUE))
        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in comp:
            dist[r, c] = 0
            q.append((r, c))
        while q:
            r, c = q.popleft()
            d = int(dist[r, c])
            blocked[r, c] = True
            if d == radius:
                continue
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] == -1:
                    dist[rr, cc] = d + 1
                    q.append((rr, cc))
    return blocked


def _to_canonical(grid: np.ndarray) -> tuple[np.ndarray, str]:
    g = np.array(grid, copy=True)
    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])
    h, w = g.shape
    if sr == 0:
        return g, "top"
    if sr == h - 1:
        return np.flipud(g).copy(), "bottom"
    if sc == 0:
        return g.T.copy(), "left"
    if sc == w - 1:
        return np.fliplr(g).T.copy(), "right"
    raise AssertionError("green start must lie on a border")


def _from_canonical(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    if edge == "right":
        return np.fliplr(grid.T).copy()
    raise AssertionError(edge)


def _spiral_loop_cells(canonical: np.ndarray) -> set[tuple[int, int]]:
    h, w = canonical.shape
    bars = _components_bar_data(canonical)
    assert len(bars) == 1, "generator/solver variant expects exactly one obstacle bar"
    bar = bars[0]
    assert bar["r0"] == bar["r1"], "canonical obstacle must be horizontal"
    r = bar["r0"]
    b = bar["blue"]
    side_left = bar["c0"] == 0
    assert side_left or bar["c1"] == w - 1, "bar must touch a side wall"

    blocked = _blocked_mask(canonical)

    u = r - b - 1
    v = r + b + 1
    assert 1 <= u < v < h - 1, "bar must leave room for the spiral folds"

    if side_left:
        near = bar["c1"] + b + 1
        mid = near + 2
        outer = near + 4
    else:
        near = bar["c0"] - b - 1
        mid = near - 2
        outer = near - 4

    pts = [(0, outer), (0, mid), (u, mid), (u, near), (v, near), (v, outer), (0, outer)]
    raw = _paint_polyline(pts)
    loop = set(raw[:-1])

    assert (0, outer) in loop, "loop must touch the border at its start cell"
    for rr, cc in loop:
        assert 0 <= rr < h and 0 <= cc < w, "loop must stay in bounds"
        assert not blocked[rr, cc], "loop must stay outside every encoded keep-out zone"

    for rr, cc in loop:
        deg = 0
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if (nr, nc) in loop:
                deg += 1
        assert deg == 2, "loop must be a simple degree-2 cycle"
    return loop


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert the border seed into a closed spiral loop that wraps around a wall-attached bar.

    Input structure:
    - bg = azure (8). A single red/blue bar (colors 2/1) touches one side wall.
    - A single green (3) pixel lies on a border edge and marks where the loop touches the frame.

    Rule:
    - The bar's blue count still encodes its Manhattan keep-out radius, exactly as in the original task.
    - Instead of continuing to the far edge, the green trace uses that clearance to form a clockwise folded
      bypass: top cap -> inward drop -> descent past the bar tip -> outward return -> closure at the top.
    - After drawing the loop, every blue pixel becomes red.

    Key insight:
    - The blue pixels do not merely push the path away once; they shift the whole closed loop, because the
      inner spiral lane is anchored at the first safe column outside the encoded danger zone.
    """

    canonical, edge = _to_canonical(np.asarray(grid, dtype=int))
    loop = _spiral_loop_cells(canonical)

    start = tuple(map(int, np.argwhere(canonical == GREEN)[0]))
    assert start in loop, "input start must lie on the closed loop"

    out = canonical.copy()
    for r, c in loop:
        if out[r, c] == BG or out[r, c] == GREEN:
            out[r, c] = GREEN
    out[out == BLUE] = RED
    return _from_canonical(out, edge)


def _paint_bar(grid: np.ndarray, row: int, side_left: bool, length: int, blue_count: int) -> None:
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    if side_left:
        grid[row, :length] = colors
    else:
        grid[row, -length:] = colors[::-1]


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    if edge == "right":
        return np.fliplr(grid.T).copy()
    raise AssertionError(edge)


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        h = random.randint(ROWS_LO, ROWS_HI)
        w = random.randint(COLS_LO, COLS_HI)
    else:
        h = random.randint(COLS_LO, COLS_HI)
        w = random.randint(ROWS_LO, ROWS_HI)

    blue_count = random.randint(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI)
    side_left = bool(random.getrandbits(1))

    lo_len = max(BAR_LEN_MIN, (BAR_PCT_LO * w + 99) // 100)
    hi_len = min(BAR_LEN_MAX, (BAR_PCT_HI * w) // 100, w - blue_count - 6)
    if lo_len > hi_len:
        return None
    length = random.randint(lo_len, hi_len)

    row_lo = blue_count + 3
    row_hi = h - blue_count - 4
    if row_lo > row_hi:
        return None
    row = random.randint(row_lo, row_hi)

    canonical = np.full((h, w), BG, dtype=int)
    _paint_bar(canonical, row, side_left, length, blue_count)

    bar = _components_bar_data(canonical)[0]
    if side_left:
        near = bar["c1"] + blue_count + 1
        outer = near + 4
    else:
        near = bar["c0"] - blue_count - 1
        outer = near - 4
    if not (0 < outer < w - 1):
        return None

    canonical[0, outer] = GREEN

    try:
        solved = main(canonical)
        if not np.array_equal(main(canonical), solved):
            return None
    except Exception:
        return None
    return _orient_grid(canonical, edge)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused cb2d8a2c input")
