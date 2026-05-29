# Combo variant of cb2d8a2c (index 20)
# combo_spec: D13/2+D16/0
# Name: 干扰条+穿越型障碍

# Design analysis
# ===============
# a) Concrete behavior and trace:
#    - We keep the original scene grammar: azure background, one green border start, and thin red/blue
#      strips arranged perpendicular to the route in the canonical top-start frame.
#    - The fusion needs three strip classes at once, so using "pure red" alone is not enough:
#      otherwise humans cannot tell "cross this but keep it red" from "ignore this and repaint it green".
#    - We therefore make the class cue two-dimensional:
#      1. blue-marked strip perpendicular to travel -> detour obstacle;
#      2. pure-red strip perpendicular to travel -> pass-through obstacle;
#      3. pure-red strip parallel to travel -> distractor strip.
#    - Small canonical trace:
#      suppose the route goes downward. A blue-marked horizontal strip at row 6 is detoured around
#      because its blue count still defines a Manhattan halo. A pure-red horizontal strip at row 11
#      is crossed directly, but the crossed cell stays red in the output. A pure-red vertical strip
#      attached to the bottom border at column 8 is ignored entirely, so any overlapped cells turn green.
#      Relative to the original task, red no longer means one thing: orientation plus blue markers decide
#      whether a strip repels, is traversed in red, or is visually irrelevant.
# b) Complexity calibration:
#    - The generator always includes at least one detour strip, one pass-through strip, and one distractor.
#    - Only three classes exist, and each has a stable, repeated cue:
#      blue+perpendicular, pure-red+perpendicular, pure-red+parallel.
#    - This is more expressive than either single variant but still learnable from 2-4 ARC examples because
#      the class logic is discrete and visible rather than depending on hidden ordering.
# c) Implicit-assumption audit against the original main():
#    - Original assumption: every red/blue component is an obstacle. Broken: parallel pure-red strips are ignored,
#      and perpendicular pure-red strips are walkable.
#    - Original assumption: blocking `bar_mask | forbidden` is safe. Broken: only blue-marked strips and their
#      halos are blocked now.
#    - Original assumption: painting only background cells is sufficient. Broken twice:
#      distractor cells must become green when crossed, while pass-through cells must remain red.
#    - Original assumption: component color alone classifies behavior. Broken: pure-red components split into two
#      behaviors, so the solver must also inspect orientation relative to the travel axis.
# d) Placement legality:
#    - Perpendicular strips are still spaced with the original conservative radius formula, using radius 0 for
#      pass-through strips.
#    - Parallel distractors are added only after solving the base puzzle, on safe columns/rows that are clear of
#      existing strips and their immediate side neighbors, so components never fuse by 4-connectivity.
#    - Distractors are attached to the start/opposite border in canonical space and leave a gap on the far end,
#      preserving the "wall-attached strip" look without overlaps or out-of-bounds writes.
# e) Visual observability:
#    - Every sample shows all three behaviors: a blue-marked strip is avoided, a pure-red perpendicular strip is
#      crossed while staying red, and a pure-red parallel strip is crossed and recolored green.
#    - The distinction is inferable by eye because the only new cue is orientation for pure-red strips; no hidden
#      numeric ordering or color-value semantics are needed.
# f) Visual cleanliness:
#    - The grid remains sparse and axis-aligned. Distractors live in the open corridor instead of intersecting other
#      strips, so outputs show isolated green cuts rather than tangled overlaps.
#    - There is only one overlay layer at any cell in the final output: background or distractor cells may become
#      green; pass-through cells stay red; blue markers disappear into red.
# g) Color-change inference:
#    - No new colors are introduced.
#    - Blue still predicts "this strip repels and later becomes red".
#    - For pure-red strips, orientation is the structural hint: perpendicular pure-red cells stay red when crossed,
#      while parallel pure-red cells flip to green because they were never functional obstacles.
# h) Degeneracy prevention:
#    - A sample is rejected unless the solved path hits at least one pass-through strip cell and at least one
#      distractor strip cell. This prevents fallback to the original task or either single variant.
# i) Generator contract for the new main():
#    - `main()` assumes red/blue components are isolated 1-cell-thick axis-aligned strips.
#    - `main()` also assumes that a pure-red component's orientation is unambiguous relative to the path axis.
#    - `generate_input()` enforces both assumptions by building only horizontal/vertical strips and by placing
#      distractors in safe, non-touching corridor segments.

import heapq
import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

_CODE_PATH = Path(__file__).resolve()
for _candidate in (_CODE_PATH.parent, *_CODE_PATH.parents):
    if (_candidate / "arc_utils.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.append(str(_candidate))
        break
    if (_candidate / "solvers" / "arc_utils.py").exists():
        solvers_dir = _candidate / "solvers"
        if str(solvers_dir) not in sys.path:
            sys.path.append(str(solvers_dir))
        break

from arc_utils import components_mask, neighbors4

BG = 8
BLUE = 1
RED = 2
GREEN = 3

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _start_and_goal(grid: np.ndarray) -> tuple[int, int, int, str, int]:
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


def _is_goal(goal_kind: str, goal_val: int, r: int, c: int) -> bool:
    return (r == goal_val) if goal_kind == "row" else (c == goal_val)


def _classify_component(
    cells: list[tuple[int, int]],
    blue_count: int,
    main_vertical: bool,
) -> str:
    if blue_count > 0:
        return "detour"

    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    is_horizontal = len(rows) == 1
    is_vertical = len(cols) == 1

    if main_vertical:
        return "pass" if is_horizontal and not is_vertical else "distractor"
    return "pass" if is_vertical and not is_horizontal else "distractor"


def _bar_infos(grid: np.ndarray, main_vertical: bool) -> list[dict[str, object]]:
    infos: list[dict[str, object]] = []
    bar_mask = (grid == BLUE) | (grid == RED)
    for comp in components_mask(bar_mask, connectivity=4):
        cells = [tuple(map(int, cell)) for cell in comp]
        blue_count = sum(1 for r, c in cells if grid[r, c] == BLUE)
        kind = _classify_component(cells, blue_count, main_vertical)
        infos.append(
            {
                "cells": cells,
                "cell_set": set(cells),
                "blue_count": blue_count,
                "kind": kind,
            }
        )
    return infos


def _build_blocked(
    shape: tuple[int, int],
    infos: list[dict[str, object]],
    start: tuple[int, int],
) -> np.ndarray:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)

    for info in infos:
        if info["kind"] != "detour":
            continue
        cells = info["cells"]
        radius = int(info["blue_count"])

        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in cells:
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

    blocked[start] = False
    return blocked


def _shortest_path(
    blocked: np.ndarray,
    start: tuple[int, int],
    init_dir: int,
    goal_kind: str,
    goal_val: int,
) -> list[tuple[int, int]]:
    h, w = blocked.shape
    sr, sc = start

    inf = (10**9, 10**9)
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]

    goal_state: tuple[int, int, int] | None = None
    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), inf):
            continue
        if _is_goal(goal_kind, goal_val, r, c) and (r, c) != (sr, sc):
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
    return path


def _solve_details(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, copy=True)
    sr, sc, init_dir, goal_kind, goal_val = _start_and_goal(g)
    main_vertical = goal_kind == "row"
    infos = _bar_infos(g, main_vertical)
    blocked = _build_blocked(g.shape, infos, (sr, sc))
    path = _shortest_path(blocked, (sr, sc), init_dir, goal_kind, goal_val)

    cell_kind: dict[tuple[int, int], str] = {}
    for info in infos:
        for cell in info["cells"]:
            cell_kind[cell] = str(info["kind"])

    path_set = set(path)
    pass_infos = [info for info in infos if info["kind"] == "pass"]
    distractor_infos = [info for info in infos if info["kind"] == "distractor"]
    detour_infos = [info for info in infos if info["kind"] == "detour"]

    return {
        "grid": g,
        "path": path,
        "cell_kind": cell_kind,
        "infos": infos,
        "detour_bar_count": len(detour_infos),
        "pass_bar_count": len(pass_infos),
        "distractor_bar_count": len(distractor_infos),
        "pass_bar_hits": sum(1 for info in pass_infos if path_set & info["cell_set"]),
        "distractor_bar_hits": sum(1 for info in distractor_infos if path_set & info["cell_set"]),
        "pass_cells": sum(1 for cell in path if cell_kind.get(cell) == "pass"),
        "distractor_cells": sum(1 for cell in path if cell_kind.get(cell) == "distractor"),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the border-started route to the opposite side while resolving three strip roles: detour bars, pass-through bars, and irrelevant distractors.

    Input structure:
    - The background is azure (8), with one green (3) start pixel on a border edge.
    - The grid contains thin side-attached strips built from red (2) and blue (1) cells.
    - Blue-marked strips lie perpendicular to the travel axis; pure-red perpendicular strips look similar; extra pure-red strips run parallel to the travel axis.

    Rule:
    - Any strip containing blue is a real detour obstacle: its blue count sets a Manhattan keep-out radius that the path must stay outside.
    - A pure-red strip perpendicular to the travel axis is traversable, but crossed cells keep the strip's red color in the output.
    - A pure-red strip parallel to the travel axis is just a distractor: it does not constrain routing, and crossed cells are repainted green; all blue cells turn red after routing.

    Key insight:
    - Blue still marks the repelling strips, but pure-red strips now split by orientation: perpendicular red strips are crossed in red, while parallel red strips are ignored and can disappear under the green route.
    """
    details = _solve_details(grid)
    out = np.array(details["grid"], copy=True)
    cell_kind: dict[tuple[int, int], str] = details["cell_kind"]

    for r, c in details["path"]:
        if cell_kind.get((r, c)) == "pass":
            continue
        out[r, c] = GREEN

    out[out == BLUE] = RED
    return out


ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 2, 3
BLUE_PER_BAR_HI = 4
DISTRACTOR_COUNT_LO, DISTRACTOR_COUNT_HI = 1, 2
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 700


def _min_main_len(radii: tuple[int, ...]) -> int:
    last_bar = radii[0] + 2
    for a, b in zip(radii, radii[1:]):
        last_bar += a + b + 2
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


def _choose_radii(count: int, main_hi: int) -> tuple[int, ...]:
    choices = [
        combo
        for combo in itertools.product(range(BLUE_PER_BAR_HI + 1), repeat=count)
        if any(v == 0 for v in combo)
        and any(v > 0 for v in combo)
        and _min_main_len(combo) <= main_hi
    ]
    return random.choice(choices)


def _horizontal_length_bounds(cross_size: int, radius: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    if radius > 0:
        lo = max(lo, 2 * radius - 1)
        hi = min(BAR_LEN_MAX, cross_size - radius - 1, (BAR_CROSS_PCT_HI * cross_size) // 100)
    else:
        hi = min(BAR_LEN_MAX, cross_size - 1, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _vertical_length_bounds(main_size: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, (BAR_CROSS_PCT_LO * main_size + 99) // 100)
    hi = min(BAR_LEN_MAX, main_size - 1, (BAR_CROSS_PCT_HI * main_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _paint_horizontal_bar(grid: np.ndarray, row: int, side: int, length: int, radius: int) -> None:
    w = grid.shape[1]
    colors = np.full(length, RED, dtype=int)
    if radius > 0:
        colors[: 2 * radius - 1 : 2] = BLUE
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, w - length :] = colors[::-1]


def _paint_vertical_distractor(grid: np.ndarray, col: int, from_top: bool, length: int) -> None:
    h = grid.shape[0]
    if from_top:
        grid[:length, col] = RED
    else:
        grid[h - length :, col] = RED


def _canonical_grid(
    main_size: int,
    cross_size: int,
    radii: tuple[int, ...],
    first_side: int,
) -> tuple[np.ndarray, dict[str, object]] | None:
    count = len(radii)
    slack = main_size - _min_main_len(radii)
    if slack < 0:
        return None

    extras = _random_composition(slack, count + 1)
    row = radii[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, count):
        row += radii[i - 1] + radii[i] + 2 + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    lengths: list[int] = []
    sides = [(first_side + i) % 2 for i in range(count)]
    for row, radius, side in zip(rows, radii, sides):
        bounds = _horizontal_length_bounds(cross_size, radius)
        if bounds is None:
            return None
        length = random.randint(*bounds)
        lengths.append(length)
        _paint_horizontal_bar(grid, row, side, length, radius)

    first_len = lengths[0]
    if first_side == 0:
        start_col = random.randint(1, first_len - 1)
    else:
        start_col = random.randint(cross_size - first_len, cross_size - 2)
    grid[0, start_col] = GREEN

    return grid, {"rows": rows, "lengths": lengths, "radii": radii, "sides": sides, "start_col": start_col}


def _safe_vertical_prefix(grid: np.ndarray, col: int) -> int:
    h, w = grid.shape
    length = 0
    for r in range(h):
        if grid[r, col] != BG:
            break
        if col > 0 and grid[r, col - 1] != BG:
            break
        if col + 1 < w and grid[r, col + 1] != BG:
            break
        length += 1
    if length == h:
        return h - 1
    return max(0, length - 1)


def _safe_vertical_suffix(grid: np.ndarray, col: int) -> int:
    h, w = grid.shape
    length = 0
    for r in range(h - 1, -1, -1):
        if grid[r, col] != BG:
            break
        if col > 0 and grid[r, col - 1] != BG:
            break
        if col + 1 < w and grid[r, col + 1] != BG:
            break
        length += 1
    if length == h:
        return h - 1
    return max(0, length - 1)


def _candidate_distractors(
    grid: np.ndarray,
    path: list[tuple[int, int]],
) -> list[dict[str, int | bool]]:
    h, _ = grid.shape
    bounds = _vertical_length_bounds(h)
    if bounds is None:
        return []
    lo, hi = bounds

    path_rows_by_col: dict[int, list[int]] = {}
    for r, c in path:
        path_rows_by_col.setdefault(c, []).append(r)

    candidates: list[dict[str, int | bool]] = []
    for col, rows in path_rows_by_col.items():
        rows = sorted(rows)

        top_len = min(hi, _safe_vertical_prefix(grid, col))
        if top_len >= lo:
            for length in range(top_len, lo - 1, -1):
                hits = sum(1 for r in rows if r < length)
                if hits > 0:
                    candidates.append({"col": col, "from_top": True, "length": length, "hits": hits})
                    break

        bottom_len = min(hi, _safe_vertical_suffix(grid, col))
        if bottom_len >= lo:
            for length in range(bottom_len, lo - 1, -1):
                cutoff = h - length
                hits = sum(1 for r in rows if r >= cutoff)
                if hits > 0:
                    candidates.append({"col": col, "from_top": False, "length": length, "hits": hits})
                    break

    candidates.sort(key=lambda item: (int(item["hits"]), int(item["length"])), reverse=True)
    return candidates


def _add_distractors(canonical: np.ndarray, distractor_target: int) -> np.ndarray | None:
    grid = canonical.copy()
    path = _solve_details(grid)["path"]
    candidates = _candidate_distractors(grid, path)
    if not candidates:
        return None

    used_cols: list[int] = []
    placed = 0
    for cand in candidates:
        col = int(cand["col"])
        if any(abs(col - prev_col) <= 1 for prev_col in used_cols):
            continue
        _paint_vertical_distractor(grid, col, bool(cand["from_top"]), int(cand["length"]))
        used_cols.append(col)
        placed += 1
        if placed >= distractor_target:
            break

    if placed < 1:
        return None
    return grid


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    if edge == "left":
        return bool(np.any(out[:, -1] == GREEN))
    return bool(np.any(out[:, 0] == GREEN))


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    count = random.randint(BAR_COUNT_LO, BAR_COUNT_HI)
    radii = _choose_radii(count, main_hi)
    main_size = random.randint(max(main_lo, _min_main_len(radii)), main_hi)
    cross_size = random.randint(cross_lo, cross_hi)
    first_side = random.randrange(2)
    distractor_target = random.randint(DISTRACTOR_COUNT_LO, DISTRACTOR_COUNT_HI)

    built = _canonical_grid(main_size, cross_size, radii, first_side)
    if built is None:
        return None
    canonical, _ = built
    canonical = _add_distractors(canonical, distractor_target)
    if canonical is None:
        return None

    inp = _orient_grid(canonical, edge)
    try:
        details = _solve_details(inp)
        out = main(inp)
    except Exception:
        return None

    if details["detour_bar_count"] < 1 or details["pass_bar_count"] < 1 or details["distractor_bar_count"] < 1:
        return None
    if details["pass_bar_hits"] < 1 or details["pass_cells"] < 1:
        return None
    if details["distractor_bar_hits"] < 1 or details["distractor_cells"] < 1:
        return None
    if not _reaches_opposite_edge(out, edge):
        return None
    if not np.array_equal(main(inp), out):
        return None
    if int(np.sum((inp == RED) & (out == GREEN))) < int(details["distractor_cells"]):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c combo input")
