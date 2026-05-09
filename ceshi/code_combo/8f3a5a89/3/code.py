# Combo variant of 8f3a5a89 (index 3)
# combo_spec: D5/2+D6/1+D14/1
# Name: 多色墙壁识别+装饰变色+按墙色涂边界

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# Design analysis
# ===============
# a) Concretization and behavior trace
# - The base geometry stays the same: a unique magenta marker sits at bottom-left,
#   grid edges plus border-touching wall shapes carve out the marker-side chamber,
#   and the output paints a 1-cell inner rim inside the reachable chamber.
# - The fused change rewires the color roles:
#   * structural walls use two explicit colors: blue `1` and red `2`;
#   * decorations use one separate input color: maroon `9`;
#   * hidden decorations recolor to gray `5` in the output;
#   * boundary paint depends on nearest structural color: blue -> orange `7`,
#     red -> green `3`.
# - Small trace:
#   * Original task: all blue walls are tentatively treated as barriers, hidden
#     non-structural blue fragments are erased, and every boundary cell is orange.
#   * Fused task: border-touching wall colors define the structural palette up
#     front, so blue/red are blocking and maroon is passable immediately. A maroon
#     patch behind the barrier becomes gray instead of disappearing, and boundary
#     cells next to red structure become green rather than orange.
#   * Difference: color now carries a chain of meaning:
#       wall color -> structural vs decorative role -> passability / hidden recolor
#       -> boundary color inherited from the nearest structural wall.
#
# b) Complexity calibration
# - The palette is kept small and repeated: 2 structural colors, 1 decoration color,
#   1 hidden-decoration output color, 2 fill colors. This is enough to show the
#   fusion clearly without turning the task into a large lookup table.
# - Every generated sample is forced to contain:
#   * both structural colors;
#   * both corresponding boundary colors;
#   * at least one reachable decoration and at least one hidden decoration.
#   This prevents silent fallback to a single-color or erase-only behavior.
#
# c) Implicit assumptions broken from the original main()
# - "Most frequent non-bg non-marker color is the wall color" is invalid once blue,
#   red, and maroon coexist.
# - "All non-bg wall pixels should block the first flood" is invalid because maroon
#   decorations must remain traversable.
# - "All hidden decorations disappear into background" is invalid because the fused
#   rule recolors them to gray.
# - "Every boundary pixel receives the same paint color" is invalid because fill is
#   now chosen per pixel from the nearest structural wall color.
# - New assumption introduced here: all structural colors appear on the grid frame,
#   while decoration colors do not. The generator enforces that.
#
# d) Placement legality
# - Structural components remain 4-connected, border-touching, and non-overlapping.
#   The generator now uses a controlled corner-pocket layout so the blue/red wall
#   influence zones are stable instead of relying on random sparse wall placement.
# - Decorations are placed only on legal empty cells:
#   * visible decorations on deep reachable background cells;
#   * hidden decorations on unreachable non-border cells.
# - Hidden decorations never touch the frame, so they cannot masquerade as explicit
#   structural colors.
#
# e) Visual observability
# - A solver can see in one sample that blue and red frame-touching walls act as the
#   real barriers because the rim colors pair with them consistently.
# - The same maroon decoration color is shown on both sides of the barrier: reachable
#   maroon stays maroon, hidden maroon turns gray. That makes the D6 recolor rule
#   visible rather than purely inferred from disappearance.
#
# f) Visual cleanliness
# - Visible decorations are restricted to interior background cells, away from the
#   colored rim, so the cause of orange vs green paint remains attributable to the
#   nearby structural wall instead of a decoration patch.
# - Structural walls are arranged to create a single clean hidden pocket, and visible
#   decorations stay off the painted rim. Some distant edge cells can still land on a
#   distance tie seam, but the generator guarantees long non-tie orange and green runs.
#
# g) Change inferability
# - The mappings are fixed and low-cardinality:
#   * blue structure -> orange rim
#   * red structure -> green rim
#   * hidden maroon decor -> gray
# - These mappings are always accompanied by geometric cues: frame-touching walls for
#   structure, unreachable pockets for hidden decor, and local proximity for rim
#   color attribution.
#
# h) Quantitative anti-degeneration target
# - Generator validation requires nonzero counts for gray hidden decor, orange fill,
#   and green fill, plus a minimum amount of each fill color. Samples that collapse to
#   "all orange" or "hidden decor absent" are rejected.
#
# i) New solver assumptions and generator guarantees
# - main() assumes the corner marker is a unique non-background singleton on a corner.
# - main() assumes every structural color is present on the frame and belongs to the
#   fixed mapping `FILL_BY_STRUCT`.
# - main() assumes decoration colors are exactly the remaining non-bg non-marker
#   colors. Generator guarantees they never appear on the border.
# - main() contains a deterministic tie-breaker for nearest-wall color. Generator does
#   not try to eliminate every tie globally; instead it guarantees clear support for
#   both color pairings through minimum visible orange/green segments.


BG = 8
WALL_BLUE = 1
WALL_RED = 2
STRUCT_COLORS = (WALL_BLUE, WALL_RED)
DECOR = 9
MARKER = 6
HIDDEN_DECOR = 5
FILL_BY_STRUCT = {
    WALL_BLUE: 7,  # orange
    WALL_RED: 3,   # green
}
FILL_COLORS = tuple(sorted(set(FILL_BY_STRUCT.values())))

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

NUM_STRUCTS_RANGE = (2, 4)
STRUCT_SHAPE_RETRIES = 12
RECT_MAX_DIM_DIV = 3
TRI_MAX_SIZE = 6
TRI_MIN_DIM_DIV = 3

REACH_RATIO_RANGE = (0.38, 0.87)

NUM_VISIBLE_DECOR_RANGE = (1, 2)
NUM_HIDDEN_DECOR_RANGE = (1, 1)
VISIBLE_DECOR_AREA_RANGE = (1, 4)
HIDDEN_DECOR_AREA_RANGE = (2, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

MIN_FILL_PIXELS_PER_COLOR = 3
MIN_HIDDEN_DECOR_PIXELS = 2
MAX_AMBIGUOUS_BOUNDARY = 12
MAX_AMBIGUOUS_RATIO = 0.25
GEN_ATTEMPTS = 1200


def _flood(start: tuple[int, int], passable, h: int, w: int) -> set[tuple[int, int]]:
    q = deque([start])
    vis = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and passable(nr, nc):
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis


def _infer_role_colors(g: np.ndarray, bg: int, marker_col: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    counts = Counter(int(v) for v in g.ravel())
    other_cols = [c for c in counts if c not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found"

    h, w = g.shape
    structural_cols = set()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            v = int(g[r, c])
            if v in other_cols:
                structural_cols.add(v)

    assert structural_cols, "no frame-touching structural colors found"
    assert structural_cols.issubset(FILL_BY_STRUCT.keys()), "unexpected structural color on frame"

    decor_cols = tuple(sorted(c for c in other_cols if c not in structural_cols))
    assert decor_cols, "no decoration color found"
    return tuple(sorted(structural_cols)), decor_cols


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = _dilate4(mask)
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _boundary_mask(reach: np.ndarray, structural: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(reach)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    near_struct = _dilate8(structural > 0)
    near_unreach = _dilate4(~reach)
    return reach & (edge | near_struct | near_unreach)


def _nearest_struct_color(
    r: int,
    c: int,
    structural: np.ndarray,
    wall_pixels: list[tuple[int, int, int]],
) -> tuple[int, bool]:
    adjacent_colors = set()
    for nr, nc in neighbors8(r, c):
        if 0 <= nr < structural.shape[0] and 0 <= nc < structural.shape[1]:
            wall_col = int(structural[nr, nc])
            if wall_col:
                adjacent_colors.add(wall_col)
    if len(adjacent_colors) == 1:
        return next(iter(adjacent_colors)), False
    if len(adjacent_colors) > 1:
        return min(adjacent_colors), True

    best_dist = None
    best_colors: set[int] = set()
    for wr, wc, wall_col in wall_pixels:
        dist = max(abs(r - wr), abs(c - wc))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_colors = {wall_col}
        elif dist == best_dist:
            best_colors.add(wall_col)

    assert best_dist is not None, "no structural wall pixels found"
    return min(best_colors), len(best_colors) > 1


def _compute_boundary_fill(structural: np.ndarray, reach: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    boundary = _boundary_mask(reach, structural)
    fills = np.zeros(structural.shape, dtype=int)
    wall_pixels = [(int(r), int(c), int(structural[r, c])) for r, c in np.argwhere(structural > 0)]
    ambiguous = 0
    for r, c in np.argwhere(boundary):
        wall_col, is_ambiguous = _nearest_struct_color(int(r), int(c), structural, wall_pixels)
        fills[r, c] = FILL_BY_STRUCT[wall_col]
        ambiguous += int(is_ambiguous)
    return boundary, fills, ambiguous


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the marker-side chamber, mark hidden decorations, and color the rim from wall colors.

    Input structure:
    - The dominant color is background, and a unique magenta corner pixel marks the flood origin.
    - Border-touching wall pieces come in multiple structural colors (blue and red), while interior
      decorative wall fragments use a separate color.
    - The structural colors are the only non-background colors allowed on the frame; decoration
      patches may lie either inside the reachable side or behind the structural barrier.

    Rule:
    - Infer the structural wall palette from the frame-touching non-marker colors, and treat only
      those colors as blocking during the flood from the marker.
    - Treat the remaining wall-colored cells as decorations: reachable ones stay unchanged, while
      unreachable ones recolor to gray.
    - Paint each reachable background boundary cell with the fill color paired to its nearest
      structural wall: blue walls produce orange, red walls produce green.

    Key insight:
    - Color now encodes role directly. First separate structural colors from decoration colors, then
      apply that role split twice: once for reachability and hidden-decoration recoloring, and again
      to transfer the nearest structural wall color onto the inner boundary.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    counts = Counter(int(v) for v in g.ravel())
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    marker = None
    for r, c in corners:
        v = int(g[r, c])
        if v != bg and counts.get(v, 0) == 1:
            marker = (r, c, v)
            break
    assert marker is not None, "no unique single-pixel corner marker found"

    mr, mc, marker_col = marker
    structural_cols, decor_cols = _infer_role_colors(g, bg, marker_col)
    structural_set = set(structural_cols)
    decor_set = set(decor_cols)

    structural = np.zeros((h, w), dtype=int)
    for wall_col in structural_cols:
        structural[g == wall_col] = wall_col

    reach = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) == bg or int(g[r, c]) == marker_col or int(g[r, c]) in decor_set,
        h,
        w,
    )

    for r in range(h):
        for c in range(w):
            cell = int(out[r, c])
            if cell in decor_set and (r, c) not in reach:
                out[r, c] = HIDDEN_DECOR
            elif cell in structural_set:
                out[r, c] = cell

    reach_mask = np.zeros((h, w), dtype=bool)
    for r, c in reach:
        reach_mask[r, c] = True

    boundary, fills, _ = _compute_boundary_fill(structural, reach_mask)
    for r, c in np.argwhere(boundary):
        if int(out[r, c]) == bg:
            out[r, c] = fills[r, c]

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
    reach = np.zeros((h, w), dtype=bool)
    reach[start] = True
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reach[nr, nc] and not blocked[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def _touches_border(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _cells_have_clearance(occupied: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if occupied[r, c]:
            return False
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                if occupied[nr, nc]:
                    return False
    return True


def _place_structural_walls(h, w, marker_rc, rng):
    for _ in range(200):
        vertical_col, horizontal_col = rng.sample(list(STRUCT_COLORS), 2)
        vert_w = rng.randint(1, 2)
        horiz_h = rng.randint(1, 2)

        joint_c = rng.randint(vert_w + 1, w - 4)
        joint_r = rng.randint(2, h - horiz_h - 3)

        structural = np.zeros((h, w), dtype=int)
        structural[: joint_r + 1, joint_c - vert_w + 1 : joint_c + 1] = vertical_col
        structural[joint_r : joint_r + horiz_h, joint_c + 1 :] = horizontal_col

        if structural[marker_rc] != 0:
            continue
        if not all((structural == wall_col).any() for wall_col in STRUCT_COLORS):
            continue
        return structural
    return None


def _validate_structural(structural: np.ndarray, reach: np.ndarray, h: int, w: int) -> bool:
    reach_nbr = _dilate4(reach)
    for wall_col in STRUCT_COLORS:
        comps = components_mask(structural == wall_col, connectivity=4)
        if not comps:
            return False
        for comp_arr in comps:
            comp = [(int(r), int(c)) for r, c in comp_arr]
            if not _touches_border(comp, h, w):
                return False
            if not any(reach_nbr[r, c] for r, c in comp):
                return False
    return True


def _paint_output(structural: np.ndarray, reach: np.ndarray, marker_rc: tuple[int, int]) -> tuple[np.ndarray, int]:
    out = np.full(structural.shape, BG, dtype=int)
    out[structural > 0] = structural[structural > 0]
    out[marker_rc] = MARKER

    boundary, fills, ambiguous = _compute_boundary_fill(structural, reach)
    paintable = boundary & (out == BG)
    out[paintable] = fills[paintable]
    return out, ambiguous


def _place_decor_mask(allowed_mask: np.ndarray, count: int, rng, area_range: tuple[int, int]) -> np.ndarray | None:
    h, w = allowed_mask.shape
    placed = np.zeros((h, w), dtype=bool)
    if count == 0:
        return placed
    for _ in range(count):
        area = rng.randint(*area_range)
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


def _behavior_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    structural_colors, decor_colors = _infer_role_colors(inp, BG, MARKER)
    structural = np.zeros(inp.shape, dtype=int)
    for wall_col in structural_colors:
        structural[inp == wall_col] = wall_col

    marker_rc = (inp.shape[0] - 1, 0)
    reach = _flood_mask(marker_rc, structural > 0, inp.shape[0], inp.shape[1])
    boundary, fills, ambiguous = _compute_boundary_fill(structural, reach)
    del fills  # boundary colors are already present in `out`

    return {
        "input_blue": int(np.sum(inp == WALL_BLUE)),
        "input_red": int(np.sum(inp == WALL_RED)),
        "input_decor": int(np.sum(np.isin(inp, decor_colors))),
        "out_orange": int(np.sum(out == FILL_BY_STRUCT[WALL_BLUE])),
        "out_green": int(np.sum(out == FILL_BY_STRUCT[WALL_RED])),
        "out_hidden_gray": int(np.sum(out == HIDDEN_DECOR)),
        "boundary_cells": int(boundary.sum()),
        "ambiguous_boundary": int(ambiguous),
    }


def _gen_attempt():
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    structural = _place_structural_walls(h, w, marker_rc, rng)
    if structural is None or structural[marker_rc] != 0:
        return None

    reach = _flood_mask(marker_rc, structural > 0, h, w)
    reach_frac = reach.mean()
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None
    if not _validate_structural(structural, reach, h, w):
        return None

    out, ambiguous = _paint_output(structural, reach, marker_rc)

    fill_mask = np.isin(out, FILL_COLORS)
    if not fill_mask.any() or len(components_mask(fill_mask, connectivity=4)) != 1:
        return None
    if ambiguous > MAX_AMBIGUOUS_BOUNDARY or ambiguous / int(fill_mask.sum()) > MAX_AMBIGUOUS_RATIO:
        return None

    fill_counts = {fill_col: int(np.sum(out == fill_col)) for fill_col in FILL_COLORS}
    if any(fill_counts[fill_col] < MIN_FILL_PIXELS_PER_COLOR for fill_col in FILL_COLORS):
        return None

    n_visible = rng.randint(*NUM_VISIBLE_DECOR_RANGE)
    n_hidden = rng.randint(*NUM_HIDDEN_DECOR_RANGE)

    visible_allowed = reach & (out == BG)
    hidden_allowed = (~reach) & (structural == 0)
    hidden_allowed[0, :] = hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = hidden_allowed[:, -1] = False

    visible_mask = _place_decor_mask(visible_allowed, n_visible, rng, VISIBLE_DECOR_AREA_RANGE)
    if visible_mask is None or not visible_mask.any():
        return None
    hidden_mask = _place_decor_mask(hidden_allowed, n_hidden, rng, HIDDEN_DECOR_AREA_RANGE)
    if hidden_mask is None or int(hidden_mask.sum()) < MIN_HIDDEN_DECOR_PIXELS:
        return None

    out_final = out.copy()
    out_final[visible_mask] = DECOR
    out_final[hidden_mask] = HIDDEN_DECOR

    inp = out_final.copy()
    inp[np.isin(inp, FILL_COLORS)] = BG
    inp[inp == HIDDEN_DECOR] = DECOR

    if not np.array_equal(main(inp), out_final):
        return None

    metrics = _behavior_metrics(inp, out_final)
    if metrics["out_hidden_gray"] < MIN_HIDDEN_DECOR_PIXELS:
        return None
    if metrics["out_orange"] < MIN_FILL_PIXELS_PER_COLOR or metrics["out_green"] < MIN_FILL_PIXELS_PER_COLOR:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 8f3a5a89 combo variant")
