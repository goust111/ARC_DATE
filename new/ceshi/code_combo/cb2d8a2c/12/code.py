# Combo variant of cb2d8a2c (index 12)
# combo_spec: D7/1+D10/0
# Name: 对角线障碍+迷宫式紧密排列
#
# Design analysis for the fused variant:
# a) Concrete behavior and traced example:
#    - Keep the original roles: azure background, red/blue obstacle bars, one green border seed,
#      and an output that extends that seed into a border-to-border path while turning around
#      danger zones.
#    - Original task: each obstacle is a horizontal or vertical 4-connected bar. Blue count b on
#      that bar creates a Manhattan keep-out halo of radius b, so the path routes around a set of
#      separated diamond-shaped forbidden regions.
#    - Fused task: each obstacle is a 45-degree staircase bar attached to the left or right wall.
#      Example canonical bars:
#        left bar  = {(2,0),(3,1),(4,2),(5,3),(6,4)}
#        right bar = {(4,12),(5,11),(6,10),(7,9),(8,8)}
#      With b=2, the old rule would be inapplicable because 4-connectivity would split these
#      staircases into singletons. The new rule treats each staircase as one 8-connected object and
#      expands it by king-distance 2, i.e. every cell within Chebyshev distance <= 2 from any bar
#      pixel is forbidden.
#    - Dense placement matters jointly with the diagonal change. If the right bar begins only two
#      rows below the left bar, their thick diagonal halos overlap in the center and leave a narrow
#      zigzag channel. The path is still 4-neighbor and turn-based, but now it must weave through
#      overlapping diagonal bands rather than pass wide orthogonal detours.
# b) Complexity calibration:
#    - Every sample uses 3-5 diagonal bars, not 1 isolated bar, so the maze effect is visible.
#    - Blue radii stay in a small discrete set 1-3, with at least one bar using radius >= 2. That
#      keeps the rule inferable from few examples while still making halo overlap common.
#    - Bar lengths are limited to roughly 35%-58% of the cross-axis. This is long enough to form
#      obvious diagonal walls, but short enough to preserve a central corridor instead of filling
#      the whole board.
# c) Hidden-assumption audit against the original main():
#    - Original `components_mask(..., connectivity=4)` assumed every obstacle is orthogonally
#      connected. Broken: diagonal staircases would fragment. New solver uses 8-connectivity.
#    - Original forbidden-zone growth via 4-neighbor BFS encoded Manhattan distance. Broken: the
#      variant says the safety radius is diagonal/45-degree oriented. New solver uses king-distance
#      halos (Chebyshev metric), which is the natural 8-neighbor distance on the pixel grid.
#    - Original generator assumed large inter-bar gaps so constraints rarely overlap. Broken:
#      D10/0 requires overlapping halos and narrow channels, so the generator now chooses very
#      small row gaps and rejects samples with too few turns or no halo overlaps.
#    - Original solver did not care whether a 4-connected reinterpretation of the same input would
#      behave similarly. Broken for this combo: the generator explicitly rejects samples where a
#      baseline orthogonal interpretation produces the same output, preventing degeneration.
# d) Placement legality:
#    - Bars are painted as monotone staircases from a side wall toward the center: left bars use
#      (r0+k, k), right bars use (r0+k, W-1-k).
#    - The generator enforces:
#      1. all bar cells stay in bounds,
#      2. distinct bars are 8-separated at the raw-pixel level, so solver parsing remains valid,
#      3. the overall height leaves at least one exit row below the last staircase,
#      4. a start column is chosen only after solve-time validation confirms a legal route.
# e) Visual observability:
#    - The diagonal change is directly visible in the input because every obstacle is a staircase,
#      not a horizontal/vertical segment.
#    - The dense-maze change is also visible: successive left/right staircases sit only 1-3 rows
#      apart, producing a repeated sawtooth corridor rather than isolated bars floating far apart.
#    - Output paths visibly hug those zigzag channels and usually turn multiple times, which is a
#      clear contrast with the original sparse orthogonal detours.
# f) Visual cleanliness:
#    - There are only four colors: azure background, green path, blue radius markers, red obstacle
#      body. No extra symbolic colors are introduced.
#    - Raw bars never overlap and remain one-cell wide. The apparent density comes from their
#      proximity and from overlapping invisible halos, so the visible obstacle geometry stays legible.
# g) Color-change inferability:
#    - Blue still has one structural meaning: count the blue pixels on one staircase to get that
#      staircase's keep-out radius.
#    - Blue->red in the output remains uniform across all bars. The new difficulty comes from shape
#      and spacing, not from a new color mapping.
# Post-change assumptions introduced by the new solver:
#    - Every red/blue 8-connected component is one staircase bar and contains at least one blue
#      pixel; the generator guarantees alternating blue/red prefixes.
#    - The unique green seed lies on a border edge; the generator still enforces this exactly.
#    - The path is searched with 4-neighbor motion, so diagonally adjacent free cells do not count
#      as a passable step. The generator validates solvability under that exact movement model.

from __future__ import annotations

import heapq
import random

import numpy as np

from arc_utils import components_mask

BG = 8
BLUE = 1
RED = 2
GREEN = 3

DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# --- Tunable parameters (CFG) ---
ROWS_LO, ROWS_HI = 16, 29
COLS_LO, COLS_HI = 15, 26
BAR_COUNT_LO, BAR_COUNT_HI = 3, 5
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 3
BAR_LEN_MIN, BAR_LEN_MAX = 5, 10
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 35, 58
ROW_GAP_LO, ROW_GAP_HI = 1, 3
MAX_GEN_RETRIES = 1000


def _find_start(grid: np.ndarray) -> tuple[int, int, int, str, int]:
    start_pts = np.argwhere(grid == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])
    h, w = grid.shape
    if sr == 0:
        return sr, sc, 2, "row", h - 1
    if sr == h - 1:
        return sr, sc, 0, "row", 0
    if sc == 0:
        return sr, sc, 1, "col", w - 1
    if sc == w - 1:
        return sr, sc, 3, "col", 0
    raise AssertionError("green start must be on a border edge")


def _is_goal(r: int, c: int, goal_kind: str, goal_val: int) -> bool:
    return (r == goal_val) if goal_kind == "row" else (c == goal_val)


def _halo_mask(
    shape: tuple[int, int],
    comp: np.ndarray,
    radius: int,
    metric: str,
) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for rr, cc in comp:
        r = int(rr)
        c = int(cc)
        if metric == "king":
            r0 = max(0, r - radius)
            r1 = min(h, r + radius + 1)
            c0 = max(0, c - radius)
            c1 = min(w, c + radius + 1)
            mask[r0:r1, c0:c1] = True
        else:
            for dr in range(-radius, radius + 1):
                rem = radius - abs(dr)
                nr = r + dr
                if not (0 <= nr < h):
                    continue
                c0 = max(0, c - rem)
                c1 = min(w, c + rem + 1)
                mask[nr, c0:c1] = True
    return mask


def _extract_bars(grid: np.ndarray, connectivity: int) -> list[dict]:
    bar_mask = (grid == BLUE) | (grid == RED)
    comps = components_mask(bar_mask, connectivity=connectivity)
    bars: list[dict] = []
    for comp in comps:
        blue_count = int(sum(1 for r, c in comp if int(grid[int(r), int(c)]) == BLUE))
        assert blue_count >= 1, "each obstacle component must contain at least one blue pixel"
        bars.append({"coords": np.array(comp, dtype=int), "radius": blue_count})
    assert bars, "grid must contain at least one red/blue obstacle component"
    return bars


def _solve_variant(
    grid: np.ndarray,
    *,
    connectivity: int,
    metric: str,
) -> tuple[np.ndarray, list[tuple[int, int]], list[np.ndarray]]:
    g = np.array(grid, copy=True)
    h, w = g.shape
    sr, sc, init_dir, goal_kind, goal_val = _find_start(g)
    bars = _extract_bars(g, connectivity=connectivity)

    blocked = np.zeros((h, w), dtype=bool)
    halos: list[np.ndarray] = []
    for bar in bars:
        halo = _halo_mask((h, w), bar["coords"], int(bar["radius"]), metric)
        halos.append(halo)
        blocked |= halo
    blocked[sr, sc] = False

    inf = (10**9, 10**9)
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]
    goal_state: tuple[int, int, int] | None = None

    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), inf):
            continue
        if _is_goal(r, c, goal_kind, goal_val) and (r, c) != (sr, sc):
            goal_state = (r, c, dprev)
            break
        for nd, (dr, dc) in enumerate(DIRS_URDL):
            rr = r + dr
            cc = c + dc
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
        if (r, c) == (sr, sc):
            break
        state = prev[state]
    path.reverse()

    out = g.copy()
    for r, c in path:
        if out[r, c] == BG:
            out[r, c] = GREEN
    out[out == BLUE] = RED
    return out, path, halos


def _count_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    prev_dir: tuple[int, int] | None = None
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        cur_dir = (r1 - r0, c1 - c0)
        if prev_dir is not None and cur_dir != prev_dir:
            turns += 1
        prev_dir = cur_dir
    return turns


def _overlap_pairs(halos: list[np.ndarray]) -> int:
    total = 0
    for i in range(len(halos)):
        for j in range(i + 1, len(halos)):
            if bool(np.any(halos[i] & halos[j])):
                total += 1
    return total


def _bar_is_diagonal(comp: np.ndarray) -> bool:
    rows = sorted(int(r) for r, _ in comp)
    cols = [int(c) for _, c in sorted(((int(r), int(c)) for r, c in comp), key=lambda x: x[0])]
    if len(set(rows)) != len(rows):
        return False
    if rows != list(range(rows[0], rows[0] + len(rows))):
        return False
    if all(cols[i + 1] - cols[i] == 1 for i in range(len(cols) - 1)):
        return True
    if all(cols[i + 1] - cols[i] == -1 for i in range(len(cols) - 1)):
        return True
    return False


def _maze_metrics(grid: np.ndarray) -> dict[str, int]:
    bars = _extract_bars(grid, connectivity=8)
    halos = [_halo_mask(grid.shape, bar["coords"], int(bar["radius"]), "king") for bar in bars]
    diagonal_bars = sum(1 for bar in bars if _bar_is_diagonal(bar["coords"]))
    row_overlap_pairs = 0
    spans = []
    for bar in bars:
        coords = bar["coords"]
        spans.append((int(coords[:, 0].min()), int(coords[:, 0].max())))
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            if not (spans[i][1] < spans[j][0] or spans[j][1] < spans[i][0]):
                row_overlap_pairs += 1
    return {
        "bars": len(bars),
        "diagonal_bars": diagonal_bars,
        "halo_overlap_pairs": _overlap_pairs(halos),
        "row_overlap_pairs": row_overlap_pairs,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the green seed to the opposite border by threading through a dense maze of diagonal bars.

    Input structure:
    - bg = azure (8). Obstacles are red/blue 45-degree staircase bars, each one an 8-connected
      diagonal component touching a side wall.
    - A single green (3) pixel lies on one border edge and marks the route's start.

    Rule:
    - For each staircase bar, count its blue pixels. That count is the bar's keep-out radius.
    - Forbidden cells are all cells within king-distance <= radius from any cell of that bar, so
      closely packed diagonal bars create overlapping zigzag danger zones.
    - Draw the shortest 4-neighbor green route from the seed to the opposite border, breaking ties
      by fewer turns. Then recolor every blue obstacle pixel to red.

    Key insight: Blue pixels no longer describe distance around isolated orthogonal bars; they
    thicken diagonal staircase walls whose 8-neighbor halos overlap to form the actual maze.
    """
    out, _path, _halos = _solve_variant(np.array(grid, copy=False), connectivity=8, metric="king")
    return out


def _length_bounds(width: int, blue_count: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, (BAR_CROSS_PCT_LO * width + 99) // 100, 2 * blue_count + 1)
    hi = min(BAR_LEN_MAX, (BAR_CROSS_PCT_HI * width) // 100, width - blue_count - 3)
    if lo > hi:
        return None
    return lo, hi


def _paint_diag_bar(
    grid: np.ndarray,
    start_row: int,
    side: int,
    length: int,
    blue_count: int,
) -> np.ndarray:
    coords = np.zeros((length, 2), dtype=int)
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    w = grid.shape[1]
    for k in range(length):
        r = start_row + k
        c = k if side == 0 else w - 1 - k
        grid[r, c] = int(colors[k])
        coords[k] = (r, c)
    return coords


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()  # edge == "right"


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    if edge == "left":
        return bool(np.any(out[:, -1] == GREEN))
    return bool(np.any(out[:, 0] == GREEN))


def _too_close(existing: list[np.ndarray], coords: np.ndarray) -> bool:
    for prev in existing:
        for r0, c0 in prev:
            for r1, c1 in coords:
                if max(abs(int(r0) - int(r1)), abs(int(c0) - int(c1))) <= 1:
                    return True
    return False


def _candidate_start_cols(width: int, side: int, length: int, blue_count: int) -> list[int]:
    if side == 0:
        lo = max(1, blue_count)
        hi = min(width - 2, length + blue_count - 1)
        cols = list(range(lo, hi + 1))
    else:
        lo = max(1, width - length - blue_count)
        hi = min(width - 2, width - blue_count - 1)
        cols = list(range(lo, hi + 1))
    random.shuffle(cols)
    return cols


def _build_canonical_candidate() -> np.ndarray | None:
    width = random.randint(COLS_LO, COLS_HI)
    count = random.randint(BAR_COUNT_LO, BAR_COUNT_HI)
    blues = [random.randint(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI) for _ in range(count)]
    if max(blues) < 2:
        blues[random.randrange(count)] = 2

    lengths: list[int] = []
    for blue_count in blues:
        bounds = _length_bounds(width, blue_count)
        if bounds is None:
            return None
        lengths.append(random.randint(*bounds))

    first_side = random.randrange(2)
    start_rows = [random.randint(1, min(3, blues[0]))]
    for i in range(1, count):
        gap_hi = min(ROW_GAP_HI, max(ROW_GAP_LO, blues[i - 1] + 1))
        gap = random.randint(ROW_GAP_LO, gap_hi)
        start_rows.append(start_rows[-1] + gap)

    min_height = max(ROWS_LO, max(r0 + length for r0, length in zip(start_rows, lengths)) + 2)
    if min_height > ROWS_HI:
        return None
    height = random.randint(min_height, ROWS_HI)

    grid = np.full((height, width), BG, dtype=int)
    bar_coords: list[np.ndarray] = []
    for i, (start_row, length, blue_count) in enumerate(zip(start_rows, lengths, blues)):
        side = (first_side + i) % 2
        if start_row + length > height - 1:
            return None
        coords = np.array(
            [[start_row + k, k if side == 0 else width - 1 - k] for k in range(length)],
            dtype=int,
        )
        if _too_close(bar_coords, coords):
            return None
        bar_coords.append(coords)
        _paint_diag_bar(grid, start_row, side, length, blue_count)

    first_length = lengths[0]
    for start_col in _candidate_start_cols(width, first_side, first_length, blues[0]):
        candidate = grid.copy()
        candidate[0, start_col] = GREEN
        try:
            out, path, halos = _solve_variant(candidate, connectivity=8, metric="king")
        except Exception:
            continue
        turns = _count_turns(path)
        overlaps = _overlap_pairs(halos)
        try:
            baseline, _, _ = _solve_variant(candidate, connectivity=4, metric="manhattan")
            same_as_baseline = bool(np.array_equal(out, baseline))
        except Exception:
            same_as_baseline = False
        if turns < max(3, count - 1):
            continue
        if overlaps < max(1, count - 2):
            continue
        if same_as_baseline:
            continue
        if not np.array_equal(main(candidate), out):
            continue
        return candidate
    return None


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom", "left", "right"))
    canonical = _build_canonical_candidate()
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)
    try:
        out = main(inp)
        if not _reaches_opposite_edge(out, edge):
            return None
    except Exception:
        return None
    metrics = _maze_metrics(inp)
    if metrics["diagonal_bars"] != metrics["bars"]:
        return None
    if metrics["halo_overlap_pairs"] < max(1, metrics["bars"] - 2):
        return None
    if metrics["row_overlap_pairs"] < max(1, metrics["bars"] - 2):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c D7/1+D10/0 input")
