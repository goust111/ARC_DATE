# Combo variant of 8f3a5a89 (index 16)
# combo_spec: D2/2+D3/1+D16/2
# Name: 非正方形长条网格+边缘标记+双层边界

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for the combo variant
# -----------------------------------------------------------------------------
# combo_variant_spec:
#   concretization_and_trace:
#     - Base task: detect the unique magenta seed, flood the marker side, keep
#       only border-touching blue components that face the flood as real walls,
#       erase hidden blue decorations, then paint a continuous orange inner ring
#       on reachable background cells adjacent to the chamber boundary.
#     - Combo task:
#       1. The grid is now an extreme strip such as 26x9 or 9x25, so the walls
#          carve long narrow chambers rather than roughly square rooms.
#       2. The magenta seed is still unique, but it lies on a non-corner cell of
#          one long edge, not at a corner. Flood origin therefore moves to the
#          side midpoint band.
#       3. The reachable-side boundary is first computed exactly as before, but
#          only one checkerboard parity class is painted orange, yielding a
#          dashed inner trace.
#       4. A second boundary layer is added on the opposite side of the same
#          structural walls: unreachable background cells adjacent to those walls
#          are painted green continuously.
#     - Example sketch:
#       In a tall 25x10 grid with the marker on the left edge, a right-edge wall
#       rectangle can isolate a long hidden pocket. The base task would produce a
#       continuous orange band only on the reachable side. The combo output shows
#       an orange dashed band on the reachable side and a green continuous band
#       inside the hidden pocket, so the blue wall sits between two differently
#       colored linings.
#   complexity_calibration:
#     - The visual change is obvious in 2-4 examples because all three controls
#       are macroscopic: strip aspect ratio, non-corner edge marker, and green
#       outer lining.
#     - The parity rule is still low-branching because it is anchored to the
#       visible marker parity; there is no hidden enumeration or color-ordering.
#     - The generator keeps the chamber count effectively simple by accepting
#       only layouts where the full pre-dash inner ring is one 4-connected band
#       and the green outer ring is also one 4-connected band.
#   implicit_assumption_audit:
#     - Original assumption "marker is in one of four corners" is invalid; main()
#       must scan only non-corner border cells instead.
#     - Original assumption "all intended boundary cells are painted" is invalid;
#       the orange side now keeps only marker-parity cells from the full inner
#       boundary candidate set.
#     - Adding green means output can no longer be derived from a single reach
#       mask alone; main() must also identify unreachable background cells that
#       lie immediately across structural walls.
#     - Hidden decorations must still not touch the border, otherwise main()
#       could mistake them for structural walls. The generator enforces this.
#     - Visible decorations can no longer occupy any background cell freely:
#       they must avoid the full inner boundary candidate set, or they would mask
#       the dashed rule. Hidden decorations must avoid the future green ring, or
#       they would visually clutter the new outer layer.
#   placement_legality:
#     - Marker is sampled only from the longer pair of edges and never from a
#       corner or near-corner slot.
#     - Structural components are rejected if they overlap each other or cover
#       the marker.
#     - Decoration placement uses random_free_location with border/padding and is
#       additionally masked away from both semantic boundary layers.
#   observability_and_cleanliness:
#     - Orange and green serve different topological sides of the same blue wall,
#       so a viewer can infer "inside reachable side vs outside hidden side"
#       without relying on invisible metadata.
#     - Strip dimensions make the effect read as a long dual-color corridor
#       rather than a small noisy cluster.
#     - The palette stays compact: bg, wall, marker, orange, green.
#   new_solver_assumptions_and_generator_guarantees:
#     - main() assumes exactly one unique non-bg border marker outside corners.
#       generate_input() always creates one.
#     - main() assumes the intended green layer is on unreachable background
#       cells adjacent to structural walls; generate_input() guarantees a hidden
#       chamber large enough for that layer to exist.
#     - main() assumes the orange dashed layer is derived from the full reachable
#       boundary candidate ring; generate_input() guarantees both painted and
#       skipped candidate cells are present in nontrivial counts.


BG = 8
WALL = 1
MARKER = 6
INNER_FILL = 7
OUTER_FILL = 3

LONG_DIM_RANGE = (22, 30)
SHORT_DIM_RANGE = (8, 11)

NUM_STRUCTS_RANGE = (1, 3)
STRUCT_SHAPE_RETRIES = 16
RECT_MAX_DIM_DIV = 3
TRI_MAX_SIZE = 6
TRI_MIN_DIM_DIV = 3

REACH_RATIO_RANGE = (0.32, 0.82)
WALL_RATIO_RANGE = (0.05, 0.22)

NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

MIN_INNER_CANDIDATES = 10
MIN_DASHED_FILL = 5
MIN_SKIPPED_CANDIDATES = 5
MIN_OUTER_CANDIDATES = 8
MIN_HIDDEN_BG = 10

GEN_ATTEMPTS = 500


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


def _coords_mask(coords: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


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


def _noncorner_border_cells(h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(1, w - 1):
        cells.append((0, c))
        cells.append((h - 1, c))
    for r in range(1, h - 1):
        cells.append((r, 0))
        cells.append((r, w - 1))
    return cells


def _boundary_candidates(reach_mask: np.ndarray, walls_mask: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
    candidate = reach_mask & bg_mask

    edge = np.zeros_like(candidate)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True

    near_struct = _dilate8(walls_mask)
    near_unreach = _dilate4(~reach_mask)
    return candidate & (edge | near_struct | near_unreach)


def _checkerboard_subset(mask: np.ndarray, anchor: tuple[int, int]) -> np.ndarray:
    rr, cc = np.indices(mask.shape)
    phase = (int(anchor[0]) + int(anchor[1])) & 1
    return mask & (((rr + cc) & 1) == phase)


def _outer_boundary_candidates(reach_mask: np.ndarray, walls_mask: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
    unreachable_bg = (~reach_mask) & bg_mask
    return unreachable_bg & _dilate8(walls_mask)


def _flood_mask(start: tuple[int, int], blocked: np.ndarray, h: int, w: int) -> np.ndarray:
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


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _strip_dimensions(rng) -> tuple[int, int]:
    if rng.choice([True, False]):
        return rng.randint(*LONG_DIM_RANGE), rng.randint(*SHORT_DIM_RANGE)
    return rng.randint(*SHORT_DIM_RANGE), rng.randint(*LONG_DIM_RANGE)


def _pick_marker(h: int, w: int, rng) -> tuple[int, int]:
    if h > w:
        edge = rng.choice(["left", "right"])
        lo = max(2, h // 4)
        hi = min(h - 3, h - 1 - h // 4)
        r = rng.randint(lo, hi)
        return (r, 0) if edge == "left" else (r, w - 1)
    edge = rng.choice(["top", "bottom"])
    lo = max(2, w // 4)
    hi = min(w - 3, w - 1 - w // 4)
    c = rng.randint(lo, hi)
    return (0, c) if edge == "top" else (h - 1, c)


def _shape_vertical_line(h, w, rng):
    c = rng.randint(1, w - 2)
    mode = rng.choice(["top", "bottom", "full"])
    if mode == "full":
        r0, r1 = 0, h - 1
    elif mode == "top":
        r0, r1 = 0, rng.randint(max(2, h // 3), h - 2)
    else:
        r0, r1 = rng.randint(1, h - max(3, h // 3)), h - 1
    return [(r, c) for r in range(r0, r1 + 1)]


def _shape_horizontal_line(h, w, rng):
    r = rng.randint(1, h - 2)
    if rng.choice(["left", "right"]) == "left":
        c0, c1 = 0, rng.randint(max(2, w // 3), w - 2)
    else:
        c0, c1 = rng.randint(1, w - max(3, w // 3)), w - 1
    return [(r, c) for c in range(c0, c1 + 1)]


def _shape_rectangle(h, w, rng):
    rh = rng.randint(2, max(2, h // RECT_MAX_DIM_DIV))
    rw = rng.randint(2, max(2, w // RECT_MAX_DIM_DIV))
    anchor = rng.choice(["top", "bottom", "left", "right", "tl", "br"])
    if anchor == "top":
        r0, c0 = 0, rng.randint(1, max(1, w - rw - 1))
    elif anchor == "bottom":
        r0, c0 = h - rh, rng.randint(1, max(1, w - rw - 1))
    elif anchor == "left":
        r0, c0 = rng.randint(1, max(1, h - rh - 1)), 0
    elif anchor == "right":
        r0, c0 = rng.randint(1, max(1, h - rh - 1)), w - rw
    elif anchor == "tl":
        r0, c0 = 0, 0
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


def _shape_strip_bar(h, w, rng):
    if h > w:
        side = rng.choice(["left", "right"])
        r = rng.randint(1, h - 2)
        length = rng.randint(max(3, w // 2), w - 1)
        if side == "left":
            return [(r, c) for c in range(length)]
        return [(r, c) for c in range(w - length, w)]
    side = rng.choice(["top", "bottom"])
    c = rng.randint(1, w - 2)
    length = rng.randint(max(3, h // 2), h - 1)
    if side == "top":
        return [(r, c) for r in range(length)]
    return [(r, c) for r in range(h - length, h)]


SHAPE_FACTORIES = [
    _shape_vertical_line,
    _shape_horizontal_line,
    _shape_rectangle,
    _shape_left_diamond,
    _shape_top_triangle,
    _shape_strip_bar,
]


def _place_structural_walls(h: int, w: int, marker_rc: tuple[int, int], rng) -> np.ndarray | None:
    num = rng.randint(*NUM_STRUCTS_RANGE)
    structural = np.zeros((h, w), dtype=bool)
    placed = 0
    for _ in range(num):
        for _ in range(STRUCT_SHAPE_RETRIES):
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
        if not (cmask & reach_nbr).any():
            return False
    return True


def _paint_output(structural: np.ndarray, reach: np.ndarray, marker_rc: tuple[int, int]):
    h, w = structural.shape
    out = np.full((h, w), BG, dtype=int)
    out[structural] = WALL
    out[marker_rc] = MARKER

    bg_mask = out == BG
    inner_full = _boundary_candidates(reach, structural, bg_mask)
    inner_dashed = _checkerboard_subset(inner_full, marker_rc)
    outer_border = _outer_boundary_candidates(reach, structural, bg_mask)

    out[inner_dashed] = INNER_FILL
    out[outer_border] = OUTER_FILL
    return out, inner_full, inner_dashed, outer_border


def _place_decor(allowed_mask: np.ndarray, count: int, rng) -> np.ndarray | None:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a dashed reachable-side border and a continuous hidden-side border in a strip grid.

    Input structure:
    - The dominant color is the background, blue components mix true border-touching walls with small wall-colored decorations.
    - A unique magenta pixel sits on a border cell that is not a corner; in intended instances it is placed on one of the strip's long edges.
    - The structural blue walls split the long rectangular canvas into a marker-reachable side and one or more hidden pockets.

    Rule:
    - Flood-fill from the non-corner border marker, first using all blue pixels as blockers to identify which border-touching blue components really face the marker side.
    - Treat only those facing components as structural walls, erase hidden blue decorations behind them, and recompute the reachable region while allowing pass-through on non-structural blue fragments.
    - On reachable background cells, find the usual inner-boundary ring and paint only the marker-parity half of it orange; on unreachable background cells immediately across structural walls, paint a continuous green outer ring.

    Key insight: The core reasoning is still separating real boundary walls from decorative blue pixels, but now the same wall geometry supports two asymmetric traces: a dashed orange lining on the reachable side and a solid green lining on the hidden side.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    counts = Counter(int(v) for v in g.ravel())

    marker = None
    for r, c in _noncorner_border_cells(h, w):
        v = int(g[r, c])
        if v != bg and counts.get(v, 0) == 1:
            marker = (r, c, v)
            break
    assert marker is not None, "no unique single-pixel non-corner border marker found"

    mr, mc, marker_col = marker
    other_cols = [c for c in counts.keys() if c not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda c: counts[c])

    reach0 = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col),
        h,
        w,
    )

    walls: set[tuple[int, int]] = set()
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = [(int(r), int(c)) for r, c in comp_arr]
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if not touches_border:
            continue
        adjacent_to_reach0 = any(any((nr, nc) in reach0 for nr, nc in neighbors4(r, c)) for r, c in comp)
        if adjacent_to_reach0:
            walls |= set(comp)

    reach = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) == wall_col and (r, c) not in walls),
        h,
        w,
    )

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and (r, c) not in walls and (r, c) not in reach:
                out[r, c] = bg

    reach_mask = _coords_mask(reach, h, w)
    walls_mask = _coords_mask(walls, h, w)
    bg_mask = out == bg
    inner_full = _boundary_candidates(reach_mask, walls_mask, bg_mask)
    outer_border = _outer_boundary_candidates(reach_mask, walls_mask, bg_mask)

    out[_checkerboard_subset(inner_full, (mr, mc))] = INNER_FILL
    out[outer_border] = OUTER_FILL
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random
    h, w = _strip_dimensions(rng)
    marker_rc = _pick_marker(h, w, rng)

    structural = _place_structural_walls(h, w, marker_rc, rng)
    if structural is None or structural[marker_rc]:
        return None

    wall_ratio = float(structural.mean())
    if not (WALL_RATIO_RANGE[0] <= wall_ratio <= WALL_RATIO_RANGE[1]):
        return None

    reach = _flood_mask(marker_rc, structural, h, w)
    reach_frac = float(reach.mean())
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    if not _validate_structural(structural, reach, h, w):
        return None

    hidden_bg = (~reach) & (~structural)
    if int(hidden_bg.sum()) < MIN_HIDDEN_BG:
        return None

    out, inner_full, inner_dashed, outer_border = _paint_output(structural, reach, marker_rc)

    if not inner_full.any() or not outer_border.any():
        return None
    if len(components_mask(inner_full, connectivity=4)) != 1:
        return None
    if len(components_mask(outer_border, connectivity=4)) != 1:
        return None

    inner_count = int(inner_full.sum())
    dashed_count = int(inner_dashed.sum())
    skipped_count = inner_count - dashed_count
    outer_count = int(outer_border.sum())
    if inner_count < MIN_INNER_CANDIDATES:
        return None
    if dashed_count < MIN_DASHED_FILL or skipped_count < MIN_SKIPPED_CANDIDATES:
        return None
    if outer_count < MIN_OUTER_CANDIDATES:
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = reach & (out == BG) & (~inner_full)
    hidden_allowed = (~reach) & (~structural) & (~outer_border)
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
    inp[inp == INNER_FILL] = BG
    inp[inp == OUTER_FILL] = BG
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 combo variant 16")
