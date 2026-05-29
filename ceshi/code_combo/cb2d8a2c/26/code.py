# Combo variant of cb2d8a2c (index 26)
# combo_spec: D3/0+D8/2+D20/1
# Name: 反向距离+聚集蓝色+危险区段标记
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace
#    - Canonical source-task view: a single green pixel starts on the top edge; horizontal
#      wall-attached bars appear below it; the route goes to the opposite edge while staying
#      outside a bar-wide Manhattan halo whose radius equals that bar's blue count.
#    - Fused rule here:
#      1. D8/2 makes each bar's blue cells a contiguous block at one end instead of an
#         alternating scatter.
#      2. D20/1 reinterprets that blue block as the dangerous segment, so only blue cells
#         emit the halo; the red body is still a solid obstacle but does not radiate danger.
#         In generation the blue block is placed on the wall-attached end, leaving a visibly
#         safe red segment on the open side.
#      3. D3/0 inverts the distance code, so the halo radius is the red-count of the bar
#         (`bar_length - blue_count`) rather than the blue-count.
#    - Small trace:
#      in a shared-length-7 puzzle, a left-attached bar with blue block length 4 has a red
#      body length 3. Original task: the whole bar would emit radius 4. Fused task: only the
#      4 blue tip cells emit, and they emit radius 3. So the path can skim the red body more
#      closely and also pass nearer to the blue-rich tip than "more blue = farther" would
#      suggest. A sibling bar with blue length 2 instead emits radius 5 from only its 2-cell
#      tip, so within one sample the solver must infer both locality and the inverted count.
#
# b) Complexity calibration
#    - Blue-tip lengths are restricted to a small visible set {2, 3, 4}; there are only a
#      few distance values to infer.
#    - All bars in one puzzle share the same total length (7). That removes a hidden degree
#      of freedom and makes "more blue -> less red -> smaller radius" directly comparable by
#      eye.
#    - The generator uses exactly two bars with distinct blue-tip lengths and requires one
#      bar to have blue length 4, ensuring the inverse relation is visible rather than latent.
#
# c) Hidden-assumption audit against the original main()
#    - Original assumption: every bar cell is a danger source. Broken by D20/1; now only
#      blue cells seed the distance expansion.
#    - Original assumption: the radius equals blue_count. Broken by D3/0; the radius must be
#      `len(bar) - blue_count`.
#    - Original assumption: blue pixels may be scattered, yet their count alone matters.
#      Broken for human inference by D8/2; the generator must place blue as one end-cluster
#      so people can see the dangerous subsegment.
#    - Pathfinding, goal detection, and blue->red recoloring remain valid once the blocked
#      mask is rebuilt with the new source set and radius.
#
# d) Placement legality enforced by the generator
#    - Bars remain 1-cell-thick, perpendicular to the travel axis, and alternate attached
#      walls exactly as in the source task.
#    - Main-axis gaps are computed from the fused radii (red-counts), so neighboring blue-tip
#      halos leave a guaranteed corridor between bars and between the first bar and the start.
#    - Cross-axis size is capped so the fixed length-7 bars still satisfy the original
#      32%-79% span regime; simultaneously the open-side gap is forced to be at least
#      `max(red_count) + 1`, guaranteeing a legal pass outside each dangerous tip.
#
# e) Visual observability
#    - In 2-4 training examples, a human can compare same-length bars: longer blue tips yield
#      smaller detours, but only near those blue tips. The red bodies stay visually long and
#      are often hugged more closely than the original whole-bar interpretation would allow.
#    - No invisible sort keys or color-id tricks are used. The rule is carried entirely by
#      visible counts, positions, and wall attachments.
#
# f) Visual cleanliness
#    - The grid still uses only azure background, red/blue bars, and a green route.
#    - Blue is concentrated into one contiguous tip per bar, so the danger marker reads as a
#      clean subsegment instead of a noisy interleaving.
#
# g) New solver assumptions and generator support
#    - Solver assumptions:
#      exactly one green border start; obstacle components are straight 1-cell bars; each bar
#      has at least one blue and one red cell; the path reaches the opposite edge.
#    - Generator support:
#      bars are built exactly in that family, the start is placed over the red section of the
#      first bar, and candidates are accepted only if two behavioral metrics are positive:
#      the solved path uses cells forbidden by a whole-bar-source model and also cells
#      forbidden by a non-inverted blue-radius model.

from __future__ import annotations

import heapq
import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4


BG = 8
BLUE = 1
RED = 2
GREEN = 3

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# --- Tunable parameters (CFG) ---
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT = 2
BLUE_CHOICES = ((2, 4), (4, 2), (3, 4), (4, 3))
SHARED_BAR_LENGTH = 7
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 600


def _build_blocked(
    grid: np.ndarray,
    *,
    radius_mode: str,
    source_mode: str,
) -> np.ndarray:
    """
    Build a blocked mask for bars under a chosen radius/source interpretation.

    `radius_mode`:
      - "blue": radius = number of blue cells on the bar
      - "red":  radius = number of red cells on the bar
    `source_mode`:
      - "whole": every bar cell emits the radius
      - "blue":  only blue cells emit the radius
    """
    h, w = grid.shape
    bar_mask = (grid == BLUE) | (grid == RED)
    blocked = bar_mask.copy()

    for comp in components_mask(bar_mask, connectivity=4):
        cells = [(int(r), int(c)) for r, c in comp]
        blue_cells = [(r, c) for r, c in cells if grid[r, c] == BLUE]
        blue_count = len(blue_cells)
        red_count = len(cells) - blue_count
        assert blue_count >= 1 and red_count >= 1, "each bar must contain both blue and red cells"

        radius = blue_count if radius_mode == "blue" else red_count
        sources = cells if source_mode == "whole" else blue_cells

        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in sources:
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


def _solve_path(grid: np.ndarray, blocked: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int, int]]:
    """Find the minimum-(steps, turns) route from the green border start to the opposite edge."""
    g = np.array(grid, copy=False)
    h, w = g.shape

    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])

    if sr == 0:
        init_dir = 2
        goal_kind, goal_val = "row", h - 1
    elif sr == h - 1:
        init_dir = 0
        goal_kind, goal_val = "row", 0
    elif sc == 0:
        init_dir = 1
        goal_kind, goal_val = "col", w - 1
    elif sc == w - 1:
        init_dir = 3
        goal_kind, goal_val = "col", 0
    else:
        raise AssertionError("green start must be on a border edge")

    def is_goal(r: int, c: int) -> bool:
        return (r == goal_val) if goal_kind == "row" else (c == goal_val)

    blocked = np.array(blocked, copy=True)
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
        if is_goal(r, c) and (r, c) != (sr, sc):
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
    st = goal_state
    while True:
        r, c, _ = st
        path.append((r, c))
        if (r, c) == (sr, sc):
            break
        st = prev[st]
    path.reverse()
    return path, goal_state


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    """Paint a path green on background cells and recolor blue to red."""
    out = np.array(grid, copy=True)
    for r, c in path:
        if out[r, c] == BG:
            out[r, c] = GREEN
    out[out == BLUE] = RED
    return out


def _path_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    """Quantify whether the fused behavior differs from simpler interpretations."""
    path_mask = (out == GREEN) & (inp == BG)
    whole_bar_red = _build_blocked(inp, radius_mode="red", source_mode="whole")
    blue_only_blue = _build_blocked(inp, radius_mode="blue", source_mode="blue")
    return {
        "locality_gain": int(np.count_nonzero(path_mask & whole_bar_red)),
        "inversion_gain": int(np.count_nonzero(path_mask & blue_only_blue)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a green route from one border to the opposite, avoiding only blue-marked danger tips and using the red remainder of each bar as the distance code.

    Input structure:
    - The background is azure (8), with exactly one green (3) start pixel on a border edge.
    - Obstacles are 1-cell-thick red/blue bars (2/1); each bar has a contiguous blue block gathered at one end and a red body on the other end.

    Rule:
    - Every bar cell is a solid obstacle, but only the blue block emits the keep-out halo.
    - That halo radius equals the number of red cells on the same bar, so more blue on a fixed-length bar means a smaller required detour.
    - The green route reaches the opposite edge by axis-aligned moves, then all blue pixels turn red.

    Key insight: Blue serves two jobs at once: it marks which part of the bar is actually dangerous, and because the true radius is the bar's red remainder, a longer blue tip means danger is more localized and also closer-pass.
    """
    blocked = _build_blocked(grid, radius_mode="red", source_mode="blue")
    path, _ = _solve_path(grid, blocked)
    return _paint_path(grid, path)


def _min_main_len(reds: tuple[int, ...]) -> int:
    """Minimum main-axis size required by the fused red-count halos."""
    last_bar = reds[0] + 2
    for a, b in zip(reds, reds[1:]):
        last_bar += a + b + 2
    return last_bar + 2


def _random_composition(total: int, parts: int) -> list[int]:
    """Uniform stars-and-bars split of `total` into `parts` non-negative parts."""
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _paint_bar(grid: np.ndarray, row: int, side: int, blue_count: int) -> None:
    """Paint one shared-length bar with a contiguous blue danger block on the wall end."""
    w = grid.shape[1]
    colors = np.full(SHARED_BAR_LENGTH, RED, dtype=int)
    if side == 0:
        colors[:blue_count] = BLUE
        grid[row, :SHARED_BAR_LENGTH] = colors
    else:
        colors[SHARED_BAR_LENGTH - blue_count :] = BLUE
        grid[row, w - SHARED_BAR_LENGTH :] = colors


def _canonical_grid(
    main_size: int,
    cross_size: int,
    blues: tuple[int, ...],
    reds: tuple[int, ...],
    first_side: int,
) -> np.ndarray | None:
    """Build the canonical top-start grid before rotating to the sampled border edge."""
    slack = main_size - _min_main_len(reds)
    if slack < 0:
        return None

    extras = _random_composition(slack, len(reds) + 1)
    row = reds[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, len(reds)):
        row += reds[i - 1] + reds[i] + 2 + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    for row, blue_count, side in zip(rows, blues, ((first_side + i) % 2 for i in range(len(blues)))):
        _paint_bar(grid, row, side, blue_count)

    first_blue = blues[0]
    red_body = SHARED_BAR_LENGTH - first_blue
    assert red_body >= 3
    if first_side == 0:
        lo = first_blue
        hi = SHARED_BAR_LENGTH - 2
    else:
        lo = cross_size - SHARED_BAR_LENGTH + 1
        hi = cross_size - first_blue - 1
    if lo > hi:
        return None

    grid[0, random.randint(lo, hi)] = GREEN
    return grid


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    """Rotate canonical top-start layout to the requested starting edge."""
    if edge == "top":
        return np.array(grid, copy=True)
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    """Check whether the solved path hits the edge opposite to the starting edge."""
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    if edge == "left":
        return bool(np.any(out[:, -1] == GREEN))
    return bool(np.any(out[:, 0] == GREEN))


def sample_metrics(inp: np.ndarray) -> dict[str, int]:
    """Public helper used for post-generation audits of the fused behavior."""
    out = main(inp)
    metrics = _path_metrics(inp, out)
    bar_mask = (inp == BLUE) | (inp == RED)
    bars = components_mask(bar_mask, connectivity=4)
    blue_counts = sorted(int(np.count_nonzero(inp[comp[:, 0], comp[:, 1]] == BLUE)) for comp in bars)
    red_counts = sorted(int(len(comp) - np.count_nonzero(inp[comp[:, 0], comp[:, 1]] == BLUE)) for comp in bars)
    metrics["bar_count"] = len(bars)
    metrics["min_blue"] = blue_counts[0]
    metrics["max_blue"] = blue_counts[-1]
    metrics["min_red"] = red_counts[0]
    metrics["max_red"] = red_counts[-1]
    return metrics


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input and keep it only if the fused behavior is visible."""
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    cross_min = max(cross_lo, SHARED_BAR_LENGTH + 6)
    cross_max = min(cross_hi, (100 * SHARED_BAR_LENGTH) // BAR_CROSS_PCT_LO)
    if cross_min > cross_max:
        return None

    blues = random.choice(BLUE_CHOICES)
    reds = tuple(SHARED_BAR_LENGTH - blue for blue in blues)
    main_need = _min_main_len(reds)
    if main_need > main_hi:
        return None

    main_size = random.randint(max(main_lo, main_need), main_hi)
    cross_size = random.randint(cross_min, cross_max)
    first_side = random.randrange(2)

    canonical = _canonical_grid(main_size, cross_size, blues, reds, first_side)
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)

    try:
        out = main(inp)
    except Exception:
        return None

    if not _reaches_opposite_edge(out, edge):
        return None
    if not np.array_equal(main(inp), out):
        return None

    metrics = _path_metrics(inp, out)
    if metrics["locality_gain"] <= 0:
        return None
    if metrics["inversion_gain"] <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c combo input")
