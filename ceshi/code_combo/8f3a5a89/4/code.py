# Combo variant of 8f3a5a89 (index 4)
# combo_spec: D4/2+D10/1
# Name: 距离限制洪填+迭代渐变

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
# - Base scene stays close to the source task: azure background, blue wall color,
#   and one unique magenta marker fixed at the bottom-left corner.
# - New visible encoding: a horizontal run of red pixels on the top edge. Its
#   length is the radius N. Those red pixels are preserved in the output and do
#   not themselves become shell colors.
# - Structural-wall detection still follows the original two-role blue logic:
#   border-touching blue components adjacent to the marker side are real walls;
#   interior blue fragments are decorations that may be traversed on the marker
#   side and erased if stranded behind a structural wall.
# - Difference trace on a simple open corridor marker -> A -> B -> C -> D -> E,
#   with N = 3:
#     source task: only boundary cells get orange.
#     D10-only task: A/B/C/D/E all get successive shell colors.
#     combo task: A/B/C get successive shell colors, while D/E remain azure even
#     though they are still chamber-reachable. So D10 exposes layer order, and
#     D4 truncates that layer expansion at a visible token-count radius.
# - This makes the two changes interact: the output is not a generic full
#   distance field, but a radius-capped distance field whose cap is read from a
#   counted object in the input.
#
# b) Complexity calibration
# - Radius is encoded by a single repeated token color rather than by a hidden
#   numeric color id. Samples therefore provide repeated evidence that "count of
#   red pixels" matters.
# - Radius is kept in a small learnable range (4..8). That is large enough to
#   show several shell colors, but small enough that the cap is countable by eye.
# - The generator rejects shallow scenes. Accepted grids must show at least four
#   colored shell layers and also a nontrivial set of still-reachable background
#   cells beyond the cap, so both D10 and D4 are visibly active.
#
# c) Implicit-assumption audit
# - Marker detection still relies on "unique non-bg corner color". The token bar
#   never occupies a corner, so this remains safe.
# - Wall inference still relies on "most frequent non-bg, non-marker color". The
#   generator keeps blue wall area comfortably above the red token count.
# - The original output logic assumed every reachable background cell should be
#   treated uniformly once reachability is known. That breaks here: we now need
#   per-cell BFS distance and must leave cells with distance > N untouched.
# - The original hidden-decoration cleanup depended on chamber reach rather than
#   on color output. That assumption still holds, so cleanup is driven by the
#   unrestricted chamber flood; only the shell rendering stage is radius-limited.
# - Non-wall extra colors would have confused the old passability rules. Here the
#   solver explicitly treats token-colored cells as ordinary traversable support
#   cells during analysis, while preserving their color in the final rendering.
#
# d) Placement legality
# - The red token bar is placed on the top edge away from corners, so its count
#   is visually explicit and it never collides with the marker.
# - Structural shapes cannot overlap the token bar or the marker, and decoration
#   patches cannot overlap structural walls, tokens, or each other.
# - Hidden blue decorations stay off the border; otherwise the solver could
#   misclassify them as structural walls.
#
# e) Visual observability
# - A human sees two repeated motifs across examples:
#   1) the number of red top-edge pixels changes,
#   2) the number of outward shell bands from the magenta seed changes with it.
# - Because accepted samples also keep reachable azure cells beyond the last
#   colored shell, the cap reads as "flood stops after N steps", not merely as a
#   palette truncation.
#
# f) Visual cleanliness
# - Semantic roles stay separated by color: blue = walls, magenta = seed, red =
#   radius code, shell colors = distance layers, azure = untouched background.
# - Decorations are few and small so they do not overwhelm the shell pattern.
# - The token bar is isolated on the top edge, keeping the counting cue easy to
#   parse without interfering with the chamber geometry.
#
# g) Change inferability
# - Shell colors follow a fixed cyclic order, never a per-instance permutation.
# - The only new input-side degree of freedom is token count, not token shape or
#   token color choice, so examples repeatedly support the same hypothesis:
#   "count the red cells, then color exactly that many BFS layers from the seed."


BG = 8
WALL = 1
TOKEN = 2
MARKER = 6
DIST_COLORS = (7, 4, 3, 9, 5, 0)

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

RADIUS_RANGE = (4, 8)
MIN_VISIBLE_LAYERS = 4
MIN_DISTINCT_BAND_COLORS = 4
MIN_COLORED_CELLS = 16
MIN_BEYOND_RADIUS_BG = 10
MIN_OUTPUT_DIFF_FROM_UNBOUNDED = 10

GEN_ATTEMPTS = 500


def _iterative_flood(start: tuple[int, int], passable, h: int, w: int) -> np.ndarray:
    """Return 4-neighbor BFS distances from start, or -1 for unreachable cells."""
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


def _is_top_token_bar(g: np.ndarray, color: int) -> bool:
    """Token color is encoded as one contiguous top-edge run away from the corners."""
    h, w = g.shape
    pos = np.argwhere(g == color)
    if len(pos) == 0:
        return False
    if np.any(pos[:, 0] != 0):
        return False
    cols = sorted(int(c) for c in pos[:, 1])
    if cols[0] == 0 or cols[-1] == w - 1:
        return False
    return cols == list(range(cols[0], cols[-1] + 1))


def _locate_roles(g: np.ndarray) -> tuple[int, tuple[int, int, int], int, set[int], int]:
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

    other_cols = [c for c in counts if c not in (bg, marker[2])]
    assert len(other_cols) >= 2, "need wall color plus radius-encoding color"
    token_cols = {int(c) for c in other_cols if _is_top_token_bar(g, int(c))}
    assert len(token_cols) == 1, "need exactly one top-edge token color"
    wall_candidates = [int(c) for c in other_cols if int(c) not in token_cols]
    assert wall_candidates, "need wall color distinct from token color"
    wall_col = int(max(wall_candidates, key=lambda c: counts[c]))
    radius = int(sum(counts[c] for c in token_cols))
    assert radius > 0, "radius encoding missing"
    return int(bg), marker, wall_col, token_cols, radius


def _analyze_layout(g: np.ndarray) -> dict:
    g = np.asarray(g, dtype=int)
    h, w = g.shape
    bg, marker, wall_col, token_cols, radius = _locate_roles(g)
    mr, mc, marker_col = marker

    strict_dist = _iterative_flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in {bg, marker_col} | token_cols,
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

    passable_base = {bg, marker_col} | token_cols
    reach_dist = _iterative_flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in passable_base or (int(g[r, c]) == wall_col and not walls[r, c]),
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
        "token_cols": token_cols,
        "radius": radius,
        "strict_dist": strict_dist,
        "strict_reach": strict_reach,
        "reach_dist": reach_dist,
        "reach": reach_dist >= 0,
        "walls": walls,
    }


def _distance_color(dist: int) -> int:
    return int(DIST_COLORS[(dist - 1) % len(DIST_COLORS)])


def _render_output(state: dict, radius_limit: int | None) -> np.ndarray:
    out = state["grid"].copy()
    bg = int(state["bg"])
    wall_col = int(state["wall_col"])
    reach = state["reach"]
    reach_dist = state["reach_dist"]
    limit = None if radius_limit is None else int(radius_limit)
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
            if limit is not None and d > limit:
                continue
            out[r, c] = _distance_color(d)

    return out


def _unbounded_shell_output(grid: np.ndarray) -> np.ndarray:
    return _render_output(_analyze_layout(np.asarray(grid, dtype=int)), radius_limit=None)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Render the marker-side chamber as a radius-capped layered distance field.

    Input structure:
    - The dominant color is the background. A unique magenta pixel on a corner is the flood source.
    - Blue components still mix structural walls and decorative fragments, exactly as in the base task.
    - An extra repeated color appears as a top-edge token bar; its pixel count encodes the maximum flood radius N.

    Rule:
    - First identify the true structural blue walls by doing the original strict chamber analysis from the marker side.
    - Then run the traversable chamber flood layer by layer, allowing passage through non-structural blue decorations, and record each cell's BFS distance from the marker.
    - Erase decorative blue fragments stranded behind structural walls, but color only the reachable background cells whose distance is between 1 and N, assigning a fixed shell color to each step.

    Key insight:
    - The blue-wall separation still tells you what the chamber is, but the red token count decides how many BFS shells from the magenta seed are actually rendered. Farther chamber cells remain plain background even though the chamber itself continues beyond them.
    """
    state = _analyze_layout(np.asarray(grid, dtype=int))
    return _render_output(state, radius_limit=int(state["radius"]))


def _shape_vertical_line(h: int, w: int, rng: random.Random):
    c = rng.randint(2, w - 2)
    mode = rng.choice(["top", "bottom", "full"])
    if mode == "full":
        r0, r1 = 0, h - 1
    elif mode == "top":
        r0, r1 = 0, rng.randint(max(1, h // 3), h - 2)
    else:
        r0, r1 = rng.randint(1, h - max(2, h // 3)), h - 1
    return [(r, c) for r in range(r0, r1 + 1)]


def _shape_horizontal_line(h: int, w: int, rng: random.Random):
    r = rng.randint(1, h - 2)
    if rng.choice(["left", "right"]) == "left":
        c0, c1 = 0, rng.randint(max(1, w // 3), w - 2)
    else:
        c0, c1 = rng.randint(1, w - max(2, w // 3)), w - 1
    return [(r, c) for c in range(c0, c1 + 1)]


def _shape_rectangle(h: int, w: int, rng: random.Random):
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


def _shape_left_diamond(h: int, w: int, rng: random.Random):
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


def _shape_top_triangle(h: int, w: int, rng: random.Random):
    max_size = min(TRI_MAX_SIZE, h // TRI_MIN_DIM_DIV, w // TRI_MIN_DIM_DIV)
    if max_size < 2:
        return None
    size = rng.randint(2, max_size)
    c_start = rng.randint(1, max(1, w - 2 * size))
    cells = []
    for i in range(size):
        cells.extend((i, c) for c in range(c_start, c_start + size - i))
    return cells


def _shape_right_triangle(h: int, w: int, rng: random.Random):
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


def _flood_mask(start: tuple[int, int], blocked: np.ndarray, h: int, w: int) -> np.ndarray:
    dist = _iterative_flood(start, lambda r, c: not blocked[r, c], h, w)
    return dist >= 0


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _place_radius_tokens(h: int, w: int, radius: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, 1 : 1 + radius] = True
    return mask


def _place_structural_walls(
    h: int,
    w: int,
    marker_rc: tuple[int, int],
    reserved: np.ndarray,
    rng: random.Random,
) -> np.ndarray | None:
    num = rng.randint(*NUM_STRUCTS_RANGE)
    structural = np.zeros((h, w), dtype=bool)
    placed = 0
    for _ in range(num):
        for _try in range(STRUCT_SHAPE_RETRIES):
            cells = rng.choice(SHAPE_FACTORIES)(h, w, rng)
            if not cells:
                continue
            if marker_rc in cells:
                continue
            if any(structural[r, c] or reserved[r, c] for r, c in cells):
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


def _place_decor(allowed_mask: np.ndarray, count: int, rng: random.Random):
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
    radius = int(state["radius"])
    bg_cells = inp == BG
    positive_bg = (reach_dist > 0) & bg_cells
    within = positive_bg & (reach_dist <= radius)
    beyond = positive_bg & (reach_dist > radius)
    band_cells = int(np.count_nonzero(np.isin(out, DIST_COLORS)))
    band_colors = {int(v) for v in out[np.isin(out, DIST_COLORS)]}
    unbounded = _unbounded_shell_output(inp)
    return {
        "radius": radius,
        "max_dist": int(reach_dist.max()),
        "visible_layers": len({int(d) for d in reach_dist[within]}),
        "band_cells": band_cells,
        "band_colors": len(band_colors),
        "beyond_radius_bg": int(np.count_nonzero(beyond)),
        "diff_from_unbounded": int(np.count_nonzero(out != unbounded)),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    radius_hi = min(RADIUS_RANGE[1], w - 2)
    if radius_hi < RADIUS_RANGE[0]:
        return None
    radius = rng.randint(RADIUS_RANGE[0], radius_hi)
    token_mask = _place_radius_tokens(h, w, radius)

    structural = _place_structural_walls(h, w, marker_rc, token_mask, rng)
    if structural is None or structural[marker_rc]:
        return None
    if int(np.count_nonzero(structural)) <= radius:
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

    visible_allowed = base_reach & (~structural) & (~token_mask)
    visible_allowed[marker_rc] = False
    visible_placed = _place_decor(visible_allowed, n_visible, rng)
    if visible_placed is None:
        return None

    hidden_allowed = (~base_reach) & (~structural) & (~token_mask)
    hidden_allowed[0, :] = False
    hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = False
    hidden_allowed[:, -1] = False
    hidden_placed = _place_decor(hidden_allowed, n_hidden, rng)
    if hidden_placed is None:
        return None

    inp = np.full((h, w), BG, dtype=int)
    inp[structural] = WALL
    inp[visible_placed] = WALL
    inp[hidden_placed] = WALL
    inp[token_mask] = TOKEN
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
    if metrics["beyond_radius_bg"] < MIN_BEYOND_RADIUS_BG:
        return None
    if metrics["diff_from_unbounded"] < MIN_OUTPUT_DIFF_FROM_UNBOUNDED:
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 combo variant")
