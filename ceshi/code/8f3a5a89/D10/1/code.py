# Variant of 8f3a5a89
# Dimension: D10 — 多步处理/迭代逻辑 (multi_step)
# Change: 两步洪填——第一次洪填（仅穿过背景和标记）确定reach0，用于识别结构墙；第二次洪填（还可穿过装饰墙）确定最终可达区域... → 迭代洪填（逐层扩展）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, random_free_location


# Design analysis for the D10 variant
# ----------------------------------
# a) Concrete behavior and trace
# - Input remains visually close to the original task: azure background, blue wall
#   components, and one unique magenta marker fixed at the bottom-left corner.
# - The old output only highlighted the inner boundary of the marker-reachable
#   chamber. The variant keeps the same chamber-detection logic, but renders the
#   whole chamber as step-by-step BFS shells.
# - Small trace:
#   old rule on a corridor marker->A->B->C:
#     reach = {marker,A,B,C}, then only cells touching a wall / edge / unreachable
#     side turn orange.
#   new rule on the same corridor:
#     step 0 = marker stays magenta
#     step 1 = A gets color DIST_COLORS[0]
#     step 2 = B gets color DIST_COLORS[1]
#     step 3 = C gets color DIST_COLORS[2]
#   The structural/non-structural wall distinction still matters, but the final
#   visualization exposes distance-from-marker instead of only the boundary.
# - Reference-task calibration from 271d71e2: that solver makes intermediate
#   step count observable in the output instead of collapsing everything into a
#   one-shot transformation. This variant follows the same spirit by exposing
#   BFS layers directly.
#
# b) Complexity calibration
# - A fixed shell palette of seven colors is used. Seven is enough to show a
#   clear multi-step pattern before wraparound, while still being learnable from
#   a few examples.
# - The generator rejects shallow layouts. Each accepted sample must show at
#   least five visible distance layers and at least four distinct shell colors,
#   so the new behavior is not visually confusable with the original orange rim.
# - Decoration count stays small and patch sizes stay tiny; otherwise blue noise
#   would dominate the striped chamber and obscure the distance rule.
#
# c) Implicit-assumption audit of the original logic
# - Marker detection relied on "corner pixel + unique non-bg color". That still
#   holds because the input is unchanged; only the output gains more colors.
# - Wall detection relied on "most frequent non-bg, non-marker color". This also
#   still holds because the input still contains only bg / wall / marker colors.
# - The original code painted only boundary bg cells. That assumption breaks in
#   the variant because unconditional border-only painting would discard the new
#   distance information. It is replaced with layer-wise coloring of every
#   reachable background cell with positive BFS distance.
# - The original second flood was one-shot and only used reach membership. The
#   variant needs layer index, so the flood helper now returns a full distance
#   map computed frontier-by-frontier.
# - Visible blue decorations were previously passable and stayed blue in output.
#   Keeping that behavior is safe here too, but the generator must place them
#   sparsely so they read as transparent ornaments rather than fake barriers.
#
# d) Placement legality
# - Structural shapes are still border-touching and non-overlapping, and the
#   marker cell may not be covered.
# - Hidden decorations are forced away from the border; otherwise the solver
#   would misclassify them as structural.
# - Visible decorations are placed only on cells that are reachable before any
#   decoration is added, so allowing traversal through them does not change the
#   chamber topology.
#
# e) Visual observability
# - The shell palette depends only on BFS step number, never on invisible color
#   ids or any arbitrary ordering of objects.
# - Because accepted samples must show multiple shells, a human sees outward
#   stripes from the bottom-left marker in 2-3 examples and can infer that the
#   flood expands one taxicab step per band.
#
# f) Visual cleanliness
# - The marker remains a single magenta seed, structural walls remain blue, and
#   shell colors are reserved for reachable background only. This keeps the three
#   semantic roles separated.
# - Decoration patches are small and isolated, avoiding tangled overlaps with
#   the shell pattern.
#
# g) Color-change inferability
# - The palette cycles through a fixed ordered tuple; there is no per-sample
#   permutation. That makes "next layer gets next color" a stable rule.
# - Wraparound can happen on large chambers, but the generator guarantees that
#   several early shells are visible before any repeat, which is the part humans
#   need to infer the iterative process.


BG = 8
WALL = 1
MARKER = 6
DIST_COLORS = (7, 4, 3, 2, 9, 5, 0)

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

NUM_STRUCTS_RANGE = (1, 4)
STRUCT_SHAPE_RETRIES = 8
RECT_MAX_DIM_DIV = 3
TRI_MAX_SIZE = 6
TRI_MIN_DIM_DIV = 3

REACH_RATIO_RANGE = (0.38, 0.87)

NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

MIN_VISIBLE_LAYERS = 5
MIN_DISTINCT_BAND_COLORS = 4
MIN_COLORED_CELLS = 18

GEN_ATTEMPTS = 400


def _iterative_flood(start: tuple[int, int], passable, h: int, w: int) -> np.ndarray:
    """Return 4-neighbor BFS distance map, expanding one frontier layer per step."""
    dist = np.full((h, w), -1, dtype=int)
    frontier = [start]
    dist[start] = 0
    step = 0
    while frontier:
        next_frontier: list[tuple[int, int]] = []
        for r, c in frontier:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] < 0 and passable(nr, nc):
                    dist[nr, nc] = step + 1
                    next_frontier.append((nr, nc))
        frontier = next_frontier
        step += 1
    return dist


def _locate_marker_and_wall(g: np.ndarray) -> tuple[int, tuple[int, int, int], int]:
    bg = get_bg(g)
    counts = Counter(int(v) for v in g.ravel())
    h, w = g.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]

    marker = None
    for r, c in corners:
        v = int(g[r, c])
        if v != bg and counts.get(v, 0) == 1:
            marker = (r, c, v)
            break
    assert marker is not None, "no unique single-pixel corner marker found"

    other_cols = [c for c in counts.keys() if c not in (bg, marker[2])]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda c: counts[c])
    return bg, marker, int(wall_col)


def _analyze_layout(g: np.ndarray) -> dict:
    g = np.asarray(g, dtype=int)
    h, w = g.shape
    bg, marker, wall_col = _locate_marker_and_wall(g)
    mr, mc, marker_col = marker

    strict_dist = _iterative_flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col),
        h,
        w,
    )
    strict_reach = strict_dist >= 0

    walls = np.zeros((h, w), dtype=bool)
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = [(int(r), int(c)) for r, c in comp_arr]
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if not touches_border:
            continue
        adjacent_to_strict = any(
            0 <= nr < h and 0 <= nc < w and strict_reach[nr, nc]
            for r, c in comp
            for nr, nc in neighbors4(r, c)
        )
        if adjacent_to_strict:
            for r, c in comp:
                walls[r, c] = True

    reach_dist = _iterative_flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) == wall_col and not walls[r, c]),
        h,
        w,
    )

    return {
        "grid": g,
        "h": h,
        "w": w,
        "bg": bg,
        "marker": marker,
        "wall_col": wall_col,
        "strict_dist": strict_dist,
        "reach_dist": reach_dist,
        "strict_reach": strict_reach,
        "reach": reach_dist >= 0,
        "walls": walls,
    }


def _distance_color(dist: int) -> int:
    return int(DIST_COLORS[(dist - 1) % len(DIST_COLORS)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Render the marker-reachable chamber as iterative distance shells while removing hidden blue ornaments.

    Input structure:
    - The dominant color is the background. A unique magenta pixel sits on one corner as the flood source.
    - Blue components mix two roles: border-touching pieces can be true chamber walls, while interior pieces may be decorative and traversable.
    - The chamber itself is still defined jointly by blue structural walls and the outer frame of the grid.

    Rule:
    - Run a strict layered flood from the corner marker through only background and marker cells; use that result to decide which blue border-touching components are genuine structural walls.
    - Run a second layered flood that may also pass through non-structural blue pixels, expanding one Manhattan step per iteration from the marker.
    - Erase blue decorations stranded on the blocked side, then color each newly reached background cell by its flood step, producing ordered shell bands around the marker.

    Key insight:
    - The important separation is still structural-wall blue vs decorative blue; once the real barriers are known, the reachable side can be displayed as a BFS distance field instead of a single orange rim.
    """
    state = _analyze_layout(np.asarray(grid, dtype=int))
    g = state["grid"]
    out = g.copy()
    bg = int(state["bg"])
    wall_col = int(state["wall_col"])
    reach = state["reach"]
    reach_dist = state["reach_dist"]
    walls = state["walls"]

    for r in range(state["h"]):
        for c in range(state["w"]):
            if int(out[r, c]) == wall_col and not walls[r, c] and not reach[r, c]:
                out[r, c] = bg

    for r in range(state["h"]):
        for c in range(state["w"]):
            d = int(reach_dist[r, c])
            if d <= 0 or int(out[r, c]) != bg:
                continue
            out[r, c] = _distance_color(d)

    return out


def _shape_vertical_line(h, w, rng):
    c = rng.randint(2, w - 2)
    mode = rng.choice(["top", "bottom", "full"])
    if mode == "full":
        r0, r1 = 0, h - 1
    elif mode == "top":
        r0, r1 = 0, rng.randint(max(1, h // 3), h - 2)
    else:
        r0, r1 = rng.randint(1, h - max(2, h // 3)), h - 1
    return [(r, c) for r in range(r0, r1 + 1)]


def _shape_horizontal_line(h, w, rng):
    r = rng.randint(1, h - 2)
    if rng.choice(["left", "right"]) == "left":
        c0, c1 = 0, rng.randint(max(1, w // 3), w - 2)
    else:
        c0, c1 = rng.randint(1, w - max(2, w // 3)), w - 1
    return [(r, c) for c in range(c0, c1 + 1)]


def _shape_rectangle(h, w, rng):
    rh = rng.randint(2, max(2, h // RECT_MAX_DIM_DIV))
    rw = rng.randint(2, max(2, w // RECT_MAX_DIM_DIV))
    anchor = rng.choice(["top", "right", "tr", "top_mid", "right_mid", "bottom_right"])
    if anchor == "top":
        r0, c0 = 0, rng.randint(1, max(1, w - rw - 1))
    elif anchor == "right":
        r0, c0 = rng.randint(1, max(1, h - rh - 1)), w - rw
    elif anchor == "tr":
        r0, c0 = 0, w - rw
    elif anchor == "top_mid":
        r0, c0 = 0, rng.randint(2, max(2, w - rw - 2))
    elif anchor == "right_mid":
        r0, c0 = rng.randint(2, max(2, h - rh - 2)), w - rw
    else:
        r0, c0 = h - rh, w - rw
    return [(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)]


def _shape_left_diamond(h, w, rng):
    max_size = min(TRI_MAX_SIZE, h // TRI_MIN_DIM_DIV, w // TRI_MIN_DIM_DIV)
    if max_size < 3:
        return None
    size = rng.randint(3, max_size)
    r_center = rng.randint(size, h - size - 1)
    top = r_center - (size - 1)
    cells = []
    for i in range(size):
        cells.extend((top + i, c) for c in range(i + 1))
    for i in range(size - 1):
        cells.extend((top + size + i, c) for c in range(size - 1 - i))
    return cells


def _shape_top_triangle(h, w, rng):
    max_size = min(TRI_MAX_SIZE, h // TRI_MIN_DIM_DIV, w // TRI_MIN_DIM_DIV)
    if max_size < 2:
        return None
    size = rng.randint(2, max_size)
    c_start = rng.randint(1, max(1, w - 2 * size))
    cells = []
    for i in range(size):
        cells.extend((i, c) for c in range(c_start, c_start + size - i))
    return cells


def _shape_right_triangle(h, w, rng):
    max_size = min(TRI_MAX_SIZE, h // 2, w // 2)
    if max_size < 2:
        return None
    size = rng.randint(2, max_size)
    r_start = rng.randint(0, max(0, h - size - 1))
    cells = []
    for i in range(size):
        length = i + 1
        cells.extend((r_start + i, c) for c in range(w - length, w))
    return cells


SHAPE_FACTORIES = [
    _shape_vertical_line,
    _shape_horizontal_line,
    _shape_rectangle,
    _shape_left_diamond,
    _shape_top_triangle,
    _shape_right_triangle,
]


def _flood_mask(start, blocked, h, w):
    dist = _iterative_flood(start, lambda r, c: not blocked[r, c], h, w)
    return dist >= 0


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _place_structural_walls(h, w, marker_rc, rng):
    num = rng.randint(*NUM_STRUCTS_RANGE)
    structural = np.zeros((h, w), dtype=bool)
    placed = 0
    for _ in range(num):
        for _try in range(STRUCT_SHAPE_RETRIES):
            cells = rng.choice(SHAPE_FACTORIES)(h, w, rng)
            if not cells:
                continue
            if marker_rc in cells or any(structural[r, c] for r, c in cells):
                continue
            for r, c in cells:
                structural[r, c] = True
            placed += 1
            break
    return structural if placed > 0 else None


def _validate_structural(structural: np.ndarray, reach: np.ndarray, h: int, w: int) -> bool:
    reach_nbr = _dilate4(reach)
    for comp_arr in components_mask(structural, connectivity=4):
        cmask = np.zeros((h, w), dtype=bool)
        cmask[comp_arr[:, 0], comp_arr[:, 1]] = True
        if not _touches_border(cmask):
            return False
        if not bool((cmask & reach_nbr).any()):
            return False
    return True


def _place_decor(allowed_mask: np.ndarray, count: int, rng):
    h, w = allowed_mask.shape
    placed = np.zeros((h, w), dtype=bool)
    if count == 0:
        return placed
    for _ in range(count):
        area = rng.randint(*DECOR_AREA_RANGE)
        shapes = [(1, area), (area, 1)]
        if area >= 3:
            shapes.append((2, 2))
        rng.shuffle(shapes)
        for ph, pw in shapes:
            if ph > h or pw > w:
                continue
            patch = np.ones((ph, pw), dtype=int)
            canvas = np.where(allowed_mask & ~placed, 0, 9)
            try:
                r0, c0 = random_free_location(
                    canvas,
                    patch,
                    transparent=0,
                    border=DECOR_BORDER,
                    padding=DECOR_PADDING,
                    padding_connectivity=8,
                )
            except ValueError:
                continue
            placed[r0 : r0 + ph, c0 : c0 + pw] = True
            break
        else:
            return None
    return placed


def _variant_metrics(inp: np.ndarray, out: np.ndarray) -> dict:
    state = _analyze_layout(inp)
    reach_dist = state["reach_dist"]
    positive = (reach_dist > 0) & (inp == BG)
    visible_layers = {int(d) for d in reach_dist[positive]}
    band_cells = int(np.count_nonzero(np.isin(out, DIST_COLORS)))
    band_colors = {int(v) for v in out[np.isin(out, DIST_COLORS)]}
    return {
        "max_dist": int(reach_dist.max()),
        "visible_layers": len(visible_layers),
        "band_cells": band_cells,
        "band_colors": len(band_colors),
    }


def _gen_attempt():
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    structural = _place_structural_walls(h, w, marker_rc, rng)
    if structural is None or structural[marker_rc]:
        return None

    base_reach = _flood_mask(marker_rc, structural, h, w)
    reach_frac = float(base_reach.mean())
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    if not _validate_structural(structural, base_reach, h, w):
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = base_reach & (~structural)
    visible_allowed[marker_rc] = False
    hidden_allowed = (~base_reach) & (~structural)
    hidden_allowed[0, :] = False
    hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = False
    hidden_allowed[:, -1] = False

    visible_placed = _place_decor(visible_allowed, n_visible, rng)
    if visible_placed is None:
        return None
    hidden_placed = _place_decor(hidden_allowed, n_hidden, rng)
    if hidden_placed is None:
        return None

    inp = np.full((h, w), BG, dtype=int)
    inp[structural] = WALL
    inp[visible_placed] = WALL
    inp[hidden_placed] = WALL
    inp[marker_rc] = MARKER

    try:
        out = main(inp)
    except AssertionError:
        return None

    metrics = _variant_metrics(inp, out)
    if metrics["visible_layers"] < MIN_VISIBLE_LAYERS:
        return None
    if metrics["band_colors"] < MIN_DISTINCT_BAND_COLORS:
        return None
    if metrics["band_cells"] < MIN_COLORED_CELLS:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D10 variant")
