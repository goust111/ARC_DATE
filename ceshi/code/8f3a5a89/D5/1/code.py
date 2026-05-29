# Variant of 8f3a5a89
# Dimension: D5 — 结构墙识别规则 (obstacle)
# Change: 结构墙 = 墙色连通分量中满足：(1) 触碰网格边缘 (2) 4-邻接可达区域（第一次洪填的reach0）的分量；非结构... → 按连通分量大小区分

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# Design analysis for the D5 obstacle variant
# -------------------------------------------
# a) Concrete behavior and trace
#    - The visible vocabulary stays the same: azure background, blue wall-color pixels,
#      one magenta marker at bottom-left, and orange output border.
#    - The changed object property is not color or shape type; it is component area.
#      Blue 4-connected components with area >= 6 are true barriers. Smaller blue
#      components are decorative clutter, even if they touch the border.
#    - Concrete trace for the key variant case:
#      a top-edge blue segment of length 3 used to be structural because it touches
#      the border and is adjacent to the first flood's reachable side. In this variant
#      its area is only 3, so the flood is allowed to pass through it. The segment
#      remains blue in the output, but the orange "halo" beneath it disappears because
#      it is no longer a real obstacle.
# b) Complexity calibration
#    - The threshold is fixed at 6. This keeps the rule visually inferable across
#      10x10..20x20 boards: tiny 1-4 cell blobs behave like decorations, while longer
#      lines / larger anchored shapes behave like enclosing walls.
#    - The generator intentionally separates sizes: structural components are always
#      >= 6 cells; decoration components are always <= 4 cells. That prevents ambiguous
#      borderline cases and supports human induction from only a few examples.
# c) Audit of original assumptions that break under D5
#    - Original structural detection depended on "touches border" and "adjacent to
#      reach0". That assumption is no longer valid: a border-touching size-3 component
#      must now be treated as decoration.
#    - Original generator banned border-touching decorations because the old solver
#      would misread them as structural. Under D5 that ban is wrong; the generator must
#      now include small border-touching decorations on purpose.
#    - The second flood still works, but the passability test must depend on component
#      size rather than on border-adjacency.
# d) Placement legality
#    - Structural shapes are still border-anchored so that the reachable region remains
#      enclosed cleanly by walls plus grid edges.
#    - All wall-color components are kept 8-separated from one another at placement
#      time unless they are intentionally part of the same component, so the intended
#      component sizes are not accidentally merged by 4-connectivity.
#    - The marker cell is never covered, and hidden decorations are placed only on the
#      unreachable side of the large walls.
# e) Visual observability
#    - Every generated sample contains at least one small border-touching decoration.
#      This is the key observable difference from the original task because such a
#      fragment looks wall-colored and edge-attached yet does not create an orange rim.
#    - The generator also rejects samples where the old border-based solver would
#      produce exactly the same output, preventing silent regression to the source task.
# f) Visual cleanliness
#    - Small border decorations are restricted to top/right edges and kept away from
#      structural walls, so the grids stay readable instead of collapsing into a dense
#      pile of touching blue shapes.
#    - Decoration count is modest, and hidden decorations are small isolated patches.
# g) Color inference
#    - No color remapping is introduced. The entire variant is communicated only through
#      spatial structure and component size, which is visible in the examples.
# h) New implicit assumptions in main() and how generate_input() guarantees them
#    - main() assumes there is exactly one unique non-background corner marker.
#      The generator fixes the marker at bottom-left and never reuses its color.
#    - main() assumes there is a single wall color distinct from bg and marker.
#      The generator uses only the original palette {8,1,6,7}.
#    - main() assumes structural and decorative wall components are separable by the
#      fixed threshold. The generator enforces structural area >= 6 and decor area <= 4.
#    - main() assumes hidden decorations can exist behind large walls and should be
#      erased only if unreachable after decorative pass-through. The generator places
#      hidden decorations strictly inside the structural unreachable region.


BG = 8
WALL = 1
MARKER = 6
FILL = 7

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

STRUCT_THRESHOLD = 6
NUM_STRUCTS_RANGE = (1, 4)
STRUCT_SHAPE_RETRIES = 12
STRUCT_AREA_RATIO_RANGE = (0.05, 0.18)
REACH_RATIO_RANGE = (0.38, 0.87)

NUM_EDGE_DECOR_RANGE = (1, 2)
NUM_VISIBLE_DECOR_RANGE = (0, 1)
NUM_HIDDEN_DECOR_RANGE = (0, 2)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

RECT_MAX_DIM_DIV = 3
TRI_MAX_SIZE = 6
TRI_MIN_DIM_DIV = 3

GEN_ATTEMPTS = 500


def _size_threshold(h: int, w: int) -> int:
    del h, w
    return STRUCT_THRESHOLD


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


def _detect_marker_and_wall(g: np.ndarray) -> tuple[int, tuple[int, int], int, int]:
    h, w = g.shape
    bg = get_bg(g)
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

    other_cols = [c for c in counts.keys() if c not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda c: counts[c])
    return bg, (mr, mc), marker_col, int(wall_col)


def _classify_size_walls(g: np.ndarray, wall_col: int) -> set[tuple[int, int]]:
    h, w = g.shape
    threshold = _size_threshold(h, w)
    walls: set[tuple[int, int]] = set()
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        if int(comp_arr.shape[0]) < threshold:
            continue
        for r, c in comp_arr:
            walls.add((int(r), int(c)))
    return walls


def _classify_border_walls(
    g: np.ndarray,
    bg: int,
    marker_rc: tuple[int, int],
    marker_col: int,
    wall_col: int,
) -> set[tuple[int, int]]:
    h, w = g.shape
    reach0 = _flood(
        marker_rc,
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
    return walls


def _solve_core(grid: np.ndarray, *, obstacle_mode: str) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, marker_rc, marker_col, wall_col = _detect_marker_and_wall(g)
    mr, mc = marker_rc

    out = g.copy()

    if obstacle_mode == "size":
        walls = _classify_size_walls(g, wall_col)
    elif obstacle_mode == "border":
        walls = _classify_border_walls(g, bg, marker_rc, marker_col, wall_col)
    else:
        raise ValueError(f"unknown obstacle_mode: {obstacle_mode}")

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

    for r, c in reach:
        if int(out[r, c]) != bg:
            continue
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            out[r, c] = FILL
            continue
        if any((nr, nc) in walls for nr, nc in neighbors8(r, c)):
            out[r, c] = FILL
            continue
        if any((nr, nc) not in reach for nr, nc in neighbors4(r, c)):
            out[r, c] = FILL

    return out, walls, reach


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw an orange inner rim around the region reachable from the corner marker while ignoring tiny wall fragments.

    Input structure:
    - The dominant color is the background, while blue cells form several wall-colored 4-connected components.
    - A unique single magenta pixel sits on one corner and marks the flood-fill start.
    - Some blue components are large enough to be true barriers; smaller blue components are decorative clutter even when they touch an edge.

    Rule:
    - Measure every wall-colored 4-connected component and treat only the large ones as structural walls.
    - Flood-fill from the marker through background, the marker cell, and all below-threshold blue components.
    - Erase decorative blue cells that remain on the unreachable side, then paint orange on reachable background cells that lie on an edge, touch a structural wall diagonally/orthogonally, or border an unreachable cell.

    Key insight: Border contact no longer decides which blue pixels are obstacles; the solver must use connected-component size to separate real walls from decorative fragments.
    """
    return _solve_core(grid, obstacle_mode="size")[0]


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


def _shape_vertical_line(h: int, w: int, rng: random.Random) -> list[tuple[int, int]]:
    c = rng.randint(2, w - 2)
    mode = rng.choice(["top", "bottom", "full"])
    if mode == "full":
        r0, r1 = 0, h - 1
    elif mode == "top":
        r0, r1 = 0, rng.randint(max(5, STRUCT_THRESHOLD), h - 2)
    else:
        r0, r1 = rng.randint(1, h - max(2, h // 3)), h - 1
    return [(r, c) for r in range(r0, r1 + 1)]


def _shape_horizontal_line(h: int, w: int, rng: random.Random) -> list[tuple[int, int]]:
    r = rng.randint(1, h - 2)
    if rng.choice(["left", "right"]) == "left":
        c0, c1 = 0, rng.randint(max(5, STRUCT_THRESHOLD), w - 2)
    else:
        c0, c1 = rng.randint(1, w - max(2, w // 3)), w - 1
    return [(r, c) for c in range(c0, c1 + 1)]


def _shape_rectangle(h: int, w: int, rng: random.Random) -> list[tuple[int, int]]:
    rh = rng.randint(2, max(2, h // RECT_MAX_DIM_DIV))
    rw = rng.randint(3, max(3, w // RECT_MAX_DIM_DIV))
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


def _shape_left_diamond(h: int, w: int, rng: random.Random) -> list[tuple[int, int]] | None:
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


def _shape_top_triangle(h: int, w: int, rng: random.Random) -> list[tuple[int, int]] | None:
    max_size = min(TRI_MAX_SIZE, h // TRI_MIN_DIM_DIV, w // TRI_MIN_DIM_DIV)
    if max_size < 2:
        return None
    size = rng.randint(2, max_size)
    c_start = rng.randint(1, max(1, w - 2 * size))
    cells = []
    for i in range(size):
        cells.extend((i, c) for c in range(c_start, c_start + size - i))
    return cells


def _shape_right_triangle(h: int, w: int, rng: random.Random) -> list[tuple[int, int]] | None:
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


def _place_structural_walls(h: int, w: int, marker_rc: tuple[int, int], rng: random.Random) -> np.ndarray | None:
    threshold = _size_threshold(h, w)
    target = rng.randint(*NUM_STRUCTS_RANGE)
    structural = np.zeros((h, w), dtype=bool)
    placed = 0
    for _ in range(target):
        for _try in range(STRUCT_SHAPE_RETRIES):
            cells = rng.choice(SHAPE_FACTORIES)(h, w, rng)
            if not cells or len(cells) < threshold:
                continue
            if marker_rc in cells or any(structural[r, c] for r, c in cells):
                continue
            for r, c in cells:
                structural[r, c] = True
            placed += 1
            break
    if placed == 0:
        return None
    return structural


def _edge_component_candidates(h: int, w: int) -> list[list[tuple[int, int]]]:
    out: list[list[tuple[int, int]]] = []
    for length in range(2, 5):
        for c0 in range(1, w - length):
            out.append([(0, c) for c in range(c0, c0 + length)])
        for r0 in range(1, h - length):
            out.append([(r, w - 1) for r in range(r0, r0 + length)])
    for c0 in range(1, w - 2):
        out.append([(0, c0), (0, c0 + 1), (1, c0), (1, c0 + 1)])
    for r0 in range(1, h - 2):
        out.append([(r0, w - 1), (r0 + 1, w - 1), (r0, w - 2), (r0 + 1, w - 2)])
    return out


def _component_has_clearance(
    cells: list[tuple[int, int]],
    allowed_mask: np.ndarray,
    occupied: np.ndarray,
    marker_rc: tuple[int, int],
) -> bool:
    h, w = allowed_mask.shape
    cell_set = set(cells)
    if marker_rc in cell_set:
        return False
    for r, c in cells:
        if not allowed_mask[r, c] or occupied[r, c]:
            return False
    for r, c in cells:
        for nr, nc in neighbors8(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in cell_set:
                continue
            if occupied[nr, nc]:
                return False
    return True


def _place_edge_decor(
    allowed_mask: np.ndarray,
    occupied: np.ndarray,
    marker_rc: tuple[int, int],
    count: int,
    rng: random.Random,
) -> np.ndarray | None:
    placed = np.zeros_like(allowed_mask, dtype=bool)
    candidates = _edge_component_candidates(*allowed_mask.shape)
    rng.shuffle(candidates)
    for cells in candidates:
        if count == 0:
            break
        if len(cells) >= _size_threshold(*allowed_mask.shape):
            continue
        current_occupied = occupied | placed
        if not _component_has_clearance(cells, allowed_mask, current_occupied, marker_rc):
            continue
        for r, c in cells:
            placed[r, c] = True
        count -= 1
    if count != 0:
        return None
    return placed


def _place_small_decor(
    allowed_mask: np.ndarray,
    occupied: np.ndarray,
    count: int,
    rng: random.Random,
) -> np.ndarray | None:
    h, w = allowed_mask.shape
    placed = np.zeros((h, w), dtype=bool)
    if count == 0:
        return placed
    for _ in range(count):
        area = rng.randint(*DECOR_AREA_RANGE)
        if area >= _size_threshold(h, w):
            area = _size_threshold(h, w) - 1
        shapes = [(1, area), (area, 1)]
        if area >= 3:
            shapes.append((2, 2))
        rng.shuffle(shapes)
        for ph, pw in shapes:
            if ph * pw >= _size_threshold(h, w):
                continue
            if ph > h or pw > w:
                continue
            patch = np.ones((ph, pw), dtype=int)
            blocked = occupied | placed | (~allowed_mask)
            canvas = np.where(blocked, 9, 0)
            try:
                r0, c0 = random_free_location(
                    canvas,
                    patch,
                    transparent=0,
                    border=0,
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


def _validate_structural(structural: np.ndarray) -> bool:
    h, w = structural.shape
    threshold = _size_threshold(h, w)
    area = float(structural.mean())
    if not (STRUCT_AREA_RATIO_RANGE[0] <= area <= STRUCT_AREA_RATIO_RANGE[1]):
        return False
    for comp_arr in components_mask(structural, connectivity=4):
        cmask = np.zeros((h, w), dtype=bool)
        cmask[comp_arr[:, 0], comp_arr[:, 1]] = True
        if int(comp_arr.shape[0]) < threshold:
            return False
        if not _touches_border(cmask):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    structural = _place_structural_walls(h, w, marker_rc, rng)
    if structural is None or structural[marker_rc]:
        return None
    if not _validate_structural(structural):
        return None

    reach = _flood_mask(marker_rc, structural, h, w)
    reach_frac = float(reach.mean())
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    edge_decor_count = rng.randint(*NUM_EDGE_DECOR_RANGE)
    edge_allowed = reach & (~structural)
    edge_allowed[h - 1, 0] = False
    edge_decor = _place_edge_decor(edge_allowed, structural, marker_rc, edge_decor_count, rng)
    if edge_decor is None:
        return None

    occupied = structural | edge_decor

    visible_count = rng.randint(*NUM_VISIBLE_DECOR_RANGE)
    visible_allowed = reach & (~occupied)
    visible_allowed[0, :] = False
    visible_allowed[-1, :] = False
    visible_allowed[:, 0] = False
    visible_allowed[:, -1] = False
    visible_decor = _place_small_decor(visible_allowed, occupied, visible_count, rng)
    if visible_decor is None:
        return None
    occupied |= visible_decor

    hidden_count = rng.randint(*NUM_HIDDEN_DECOR_RANGE)
    hidden_allowed = (~reach) & (~occupied)
    hidden_allowed[0, :] = False
    hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = False
    hidden_allowed[:, -1] = False
    hidden_decor = _place_small_decor(hidden_allowed, occupied, hidden_count, rng)
    if hidden_decor is None:
        return None

    inp = np.full((h, w), BG, dtype=int)
    inp[structural | edge_decor | visible_decor | hidden_decor] = WALL
    inp[marker_rc] = MARKER

    out, walls, reach2 = _solve_core(inp, obstacle_mode="size")
    if not np.array_equal(out, main(inp)):
        return None

    fill_mask = (out == FILL)
    if not fill_mask.any() or len(components_mask(fill_mask, connectivity=4)) != 1:
        return None

    if not edge_decor.any():
        return None
    if not np.all(out[edge_decor] == WALL):
        return None
    if any(coord in walls for coord in map(tuple, np.argwhere(edge_decor))):
        return None
    if not all(tuple(coord) in reach2 for coord in np.argwhere(edge_decor)):
        return None

    old_out, old_walls, _ = _solve_core(inp, obstacle_mode="border")
    if np.array_equal(out, old_out):
        return None
    if len(old_walls) == 0:
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D5 variant")
