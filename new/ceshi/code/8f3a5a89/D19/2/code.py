# Variant of 8f3a5a89
# Dimension: D19 — 符号编码/图例机制 (symbol_instructions)
# Change: 无显式图例——所有规则从输入输出对中隐式推断... → 墙壁上的特殊标记像素编码涂色方向

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, random_free_location

# ---------------------------------------------------------------------------
# Design analysis for the D19 variant
#
# Reference-task check:
# - The prompt names `e87109e9` as the symbol-instruction inspiration.
# - In this workspace its JSON file exists, but no readable solver `.py` exists,
#   so the "read the source solver first" step is not possible here. The variant
#   below is therefore designed from the D19 brief plus the original 8f3a5a89 spec.
#
# a) Concretization / behavior trace
# - Original behavior: every border-touching structural wall blocks the BL-corner
#   flood, hidden interior wall fragments are erased, and orange is painted around
#   the whole reachable-side inner boundary.
# - New behavior: each structural wall component still blocks reach in the same way,
#   but some of its wall pixels are recolored into one repeated symbol color.
#   Symbol colors are fixed direction codes:
#     red(2)=paint the LEFT side of that wall component
#     green(3)=paint the RIGHT side
#     yellow(4)=paint the UP side
#     maroon(9)=paint the DOWN side
# - Output consequence: orange is no longer a full ring. It appears only on
#   reachable background cells that lie one step on the instructed side of each
#   structural component. Hidden decorations are still erased exactly as before.
# - Small trace:
#   original: a top-edge blue rectangle would orange-paint every reachable cell
#   under/around its reachable-facing perimeter.
#   variant: the same rectangle with yellow symbol pixels paints only the cells
#   directly above it; with maroon symbol pixels it paints only the cells directly
#   below it. The flood and decoration cleanup stay unchanged.
#
# b) Complexity calibration
# - Four absolute directions are enough to make the symbol mechanism meaningful,
#   but not so many that training examples become unreadable.
# - To avoid underdetermined samples, the generator requires at least two distinct
#   structural wall components and at least two different symbol directions per grid.
# - Each component gets at least two symbol-colored pixels on the chosen side-facing
#   edge, so the code is visually repeated instead of being a single accidental dot.
#
# c) Hidden assumptions from the original main()
# - Original wall detection assumes one dominant wall color. That fails once walls
#   may contain red/green/yellow/maroon symbol pixels. The new solver therefore
#   treats every non-background, non-corner-marker color as "wallish" when finding
#   structural components.
# - Original fill logic paints any reachable cell near a wall / unreachable region.
#   That would ignore the new direction code and reproduce the old ring. The new
#   fill logic must be component-local and direction-aware.
# - Original decoration removal only reasoned about blue pixels. In the variant the
#   solver removes any non-structural wallish pixels that lie behind the structure;
#   the generator guarantees symbol colors appear only inside true structural walls.
#
# d) Placement legality
# - Symbol pixels are not separate objects; they replace selected wall cells inside
#   a structural component, so they cannot overlap or go out of bounds.
# - Decorations are placed away from structural walls using an 8-neighborhood gap,
#   preventing accidental mergers that would change component identity.
# - Chosen directions are validated by requiring enough reachable background cells
#   on that side, so every symbol actually causes visible orange output.
#
# e) Visual observability
# - The core visible change is that identical blue wall geometry can produce orange
#   on different sides depending on the embedded symbol color.
# - Requiring multiple components and multiple directions in one sample makes the
#   rule legible from only a few training examples.
#
# f) Visual cleanliness
# - Hidden decorations remain plain blue; only structural walls carry symbol colors.
# - Symbol pixels sit on the relevant wall edge and are repeated, which keeps them
#   readable and avoids mixing many object types in the same small area.
#
# g) Color-change inferability
# - The extra colors are limited to four reusable symbol colors with stable meaning.
# - Their effect is not arbitrary recoloring: each color is coupled to a specific
#   side of a nearby wall component, giving a strong structural cue for humans.
# ---------------------------------------------------------------------------


DIR_TO_SYMBOL = {
    "left": 2,
    "right": 3,
    "up": 4,
    "down": 9,
}
SYMBOL_TO_DIR = {v: k for k, v in DIR_TO_SYMBOL.items()}
DIR_OFFSETS = {
    "left": (0, -1),
    "right": (0, 1),
    "up": (-1, 0),
    "down": (1, 0),
}


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


def _component_side_cells(
    comp: set[tuple[int, int]],
    direction: str,
    reach: set[tuple[int, int]] | np.ndarray,
    out: np.ndarray | None,
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    dr, dc = DIR_OFFSETS[direction]
    cells: set[tuple[int, int]] = set()
    use_mask = isinstance(reach, np.ndarray)
    for r, c in comp:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if use_mask:
            if not reach[nr, nc]:
                continue
        elif (nr, nc) not in reach:
            continue
        if out is not None and int(out[nr, nc]) != BG:
            continue
        cells.add((nr, nc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint direction-selected wall-side strips inside the region reachable from a corner marker.

    Input structure:
    - Background is the most common color; the bottom-left corner still contains a unique magenta start pixel.
    - Border-touching wall components are mostly blue, but each true structural component embeds repeated special-color pixels.
    - Those embedded colors are the only instruction legend: red/green/yellow/maroon encode left/right/up/down painting for that whole wall component.

    Rule:
    - Flood-fill from the corner marker while treating all wallish pixels as blocked to identify the marker-side region.
    - Re-identify structural walls as wallish components that touch a grid edge and touch the initial reachable side; erase non-structural wall fragments hidden behind them.
    - For each structural component, read its embedded symbol color and paint orange only on reachable background cells lying exactly on that instructed side of the component.

    Key insight: The geometry still tells you which wall components are real barriers, but the orange side is no longer implicit from reachability alone; it is explicitly encoded by the colored pixels embedded in each wall.
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

    wallish_colors = [c for c in counts if c not in (bg, marker_col)]
    assert wallish_colors, "no wall colors found"
    wall_mask = np.isin(g, wallish_colors)

    reach0 = _flood(
        (mr, mc),
        lambda r, c: not wall_mask[r, c],
        h,
        w,
    )

    walls: set[tuple[int, int]] = set()
    structural_components: list[set[tuple[int, int]]] = []
    for comp_arr in components_mask(wall_mask, connectivity=4):
        comp = {(int(r), int(c)) for r, c in comp_arr}
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if not touches_border:
            continue
        adjacent_to_reach0 = any(any((nr, nc) in reach0 for nr, nc in neighbors4(r, c)) for r, c in comp)
        if adjacent_to_reach0:
            walls |= comp
            structural_components.append(comp)

    reach = _flood(
        (mr, mc),
        lambda r, c: (not wall_mask[r, c]) or ((r, c) not in walls),
        h,
        w,
    )

    for r in range(h):
        for c in range(w):
            if wall_mask[r, c] and (r, c) not in walls and (r, c) not in reach:
                out[r, c] = bg

    for comp in structural_components:
        symbol_counts = Counter(int(g[r, c]) for r, c in comp if int(g[r, c]) in SYMBOL_TO_DIR)
        assert symbol_counts, "structural component without direction symbol"
        symbol_col = max(symbol_counts, key=symbol_counts.get)
        direction = SYMBOL_TO_DIR[symbol_col]
        for rr, cc in _component_side_cells(comp, direction, reach, out, h, w):
            out[rr, cc] = FILL

    return out


# --------------- generator ---------------

BG = 8
WALL = 1
MARKER = 6
FILL = 7

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

NUM_STRUCTS_RANGE = (2, 4)
STRUCT_SHAPE_RETRIES = 8
RECT_MAX_DIM_DIV = 3
TRI_MAX_SIZE = 6
TRI_MIN_DIM_DIV = 3

REACH_RATIO_RANGE = (0.38, 0.87)

NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

GEN_ATTEMPTS = 400
MIN_SIDE_FILL_CELLS = 3
MIN_SYMBOL_PIXELS = 2
MIN_TOTAL_FILL = 8


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


def _touches_border(mask):
    return mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()


def _dilate8(mask):
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
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


def _validate_structural(structural, reach, h, w):
    for comp_arr in components_mask(structural, connectivity=4):
        cmask = np.zeros((h, w), dtype=bool)
        cmask[comp_arr[:, 0], comp_arr[:, 1]] = True
        if not _touches_border(cmask):
            return False
        adjacent = False
        for r, c in comp_arr:
            for nr, nc in neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and reach[nr, nc]:
                    adjacent = True
                    break
            if adjacent:
                break
        if not adjacent:
            return False
    return True


def _choose_component_instructions(structural, reach, rng):
    h, w = structural.shape
    infos = []
    for comp_arr in components_mask(structural, connectivity=4):
        comp = {(int(r), int(c)) for r, c in comp_arr}
        options = []
        for direction in DIR_OFFSETS:
            fill_cells = _component_side_cells(comp, direction, reach, None, h, w)
            if len(fill_cells) < MIN_SIDE_FILL_CELLS:
                continue
            dr, dc = DIR_OFFSETS[direction]
            symbol_cells = sorted(
                (r, c) for r, c in comp
                if 0 <= r + dr < h and 0 <= c + dc < w and reach[r + dr, c + dc]
            )
            if len(symbol_cells) < MIN_SYMBOL_PIXELS:
                continue
            options.append((direction, fill_cells, symbol_cells))
        if not options:
            return None
        infos.append({"comp": comp, "options": options})

    if len(infos) < 2:
        return None

    for _ in range(30):
        chosen = []
        seen_dirs = set()
        total_fill = set()
        for info in infos:
            direction, fill_cells, symbol_cells = rng.choice(info["options"])
            rng.shuffle(symbol_cells)
            chosen_symbols = set(symbol_cells[:MIN_SYMBOL_PIXELS])
            chosen.append(
                {
                    "comp": info["comp"],
                    "direction": direction,
                    "fill_cells": set(fill_cells),
                    "symbol_cells": chosen_symbols,
                    "symbol_color": DIR_TO_SYMBOL[direction],
                }
            )
            seen_dirs.add(direction)
            total_fill |= set(fill_cells)
        if len(seen_dirs) >= 2 and len(total_fill) >= MIN_TOTAL_FILL:
            return chosen
    return None


def _paint_output(structural, reach, marker_rc, comp_infos):
    out = np.full(structural.shape, BG, dtype=int)
    out[structural] = WALL
    for info in comp_infos:
        for r, c in info["symbol_cells"]:
            out[r, c] = info["symbol_color"]
    out[marker_rc] = MARKER
    for info in comp_infos:
        for r, c in info["fill_cells"]:
            if reach[r, c] and int(out[r, c]) == BG:
                out[r, c] = FILL
    return out


def _place_decor(allowed_mask, count, rng):
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
            placed[r0:r0 + ph, c0:c0 + pw] = True
            break
        else:
            return None
    return placed


def _gen_attempt():
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    structural = _place_structural_walls(h, w, marker_rc, rng)
    if structural is None or structural[marker_rc]:
        return None

    reach = _flood_mask(marker_rc, structural, h, w)
    reach_frac = float(reach.mean())
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    if not _validate_structural(structural, reach, h, w):
        return None

    comp_infos = _choose_component_instructions(structural, reach, rng)
    if comp_infos is None:
        return None

    out = _paint_output(structural, reach, marker_rc, comp_infos)

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    away_from_struct = ~_dilate8(structural)
    visible_allowed = reach & (out == BG) & away_from_struct
    hidden_allowed = (~reach) & (~structural) & away_from_struct
    hidden_allowed[0, :] = hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = hidden_allowed[:, -1] = False

    visible_placed = _place_decor(visible_allowed, n_visible, rng)
    if visible_placed is None:
        return None
    hidden_placed = _place_decor(hidden_allowed, n_hidden, rng)
    if hidden_placed is None:
        return None

    out_final = out.copy()
    out_final[visible_placed] = WALL

    inp = out_final.copy()
    inp[inp == FILL] = BG
    inp[hidden_placed] = WALL

    if not np.array_equal(main(inp), out_final):
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D19 variant")
