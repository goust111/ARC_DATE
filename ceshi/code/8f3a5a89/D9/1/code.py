# Variant of 8f3a5a89
# Dimension: D9 — 输出网格形状 (grid_shape)
# Change: 输出与输入尺寸完全相同（same size），在原网格上修改像素... → 仅输出边界环（移除内部和墙壁）

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# Design analysis for the D9 variant
# a) Concrete behavior and trace:
#    The input still contains the same ingredients as the original task:
#    azure background, a unique magenta marker in the bottom-left corner,
#    blue border-touching structural walls, and optional blue decorative blobs.
#    The original solver produced a full scene: walls stayed blue, hidden
#    decorations were erased, the reachable side got an orange 1-cell inner
#    border, and the remaining reachable interior stayed background.
#    This variant instead outputs only the boundary scaffold itself. That means
#    orange pixels remain only where the reachable-side inner border lies, and
#    blue pixels remain only for the actual structural walls that define that
#    boundary. The marker and every decorative wall disappear.
#    Small trace: if a visible 2x2 blue decoration sits inside the reachable
#    region, the original output would keep that 2x2 block blue. In the
#    variant, the same 2x2 block becomes background because it is not part of
#    the boundary scaffold; only the orange ring and the structural blue walls
#    survive.
# b) Complexity calibration:
#    Keeping structural walls while deleting decorations creates a real visual
#    distinction between "blue that matters" and "blue that does not". That is
#    stronger than merely deleting the single marker pixel, and it directly
#    exercises the intended boundary understanding. Decoration count is kept
#    small (1-3 blobs) so examples stay readable.
# c) Implicit assumptions review:
#    - Marker detection still assumes one unique non-background corner pixel.
#      The generator preserves that.
#    - Wall-color detection still assumes the most common non-bg non-marker
#      color is the wall color. The generator uses only one wall color.
#    - Structural-wall recovery still depends on border-touching blue
#      components being adjacent to the marker-side flood when all blue blocks
#      movement. The generator validates that property.
#    - Boundary painting must still ignore decorations, so the second flood
#      continues to traverse non-structural blue cells. The variant only
#      changes what is emitted, not how reachability is computed.
#    - Because the output is now sparse, writing into a copy of the input would
#      accidentally preserve marker/decor pixels. The solver must instead build
#      a fresh background canvas.
# d) Placement legality:
#    Structural components are placed as non-overlapping border-touching
#    4-connected shapes. Visible decorations are placed only on reachable
#    background cells that are not on the orange ring, so they neither overlap
#    structural walls nor suppress ring pixels. Hidden decorations are placed
#    only in unreachable non-border cells so the solver will never mistake them
#    for structural walls.
# e) Visual observability:
#    Each generated sample contains at least one visible decoration, so the
#    output visibly removes some blue pixels that were present in the input.
#    This makes the new "boundary scaffold only" rule learnable from a few
#    examples instead of collapsing to the original behavior.
# f) Visual cleanliness:
#    Decorations stay small and padded apart. The ring remains 1-cell thick and
#    is required to form a single 4-connected component, which prevents noisy
#    fragmented outputs.
# g) Color inferability:
#    No new colors are introduced. Orange still marks the inner border, blue
#    still marks walls in the input, and the output keeps blue only for genuine
#    boundary walls. The distinction is inferable from spatial role, not from
#    hidden color-id tricks.


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


def _boundary_fill_mask(
    bg_cells: np.ndarray,
    reach: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    h: int,
    w: int,
) -> np.ndarray:
    fill = np.zeros((h, w), dtype=bool)
    for r, c in reach:
        if not bg_cells[r, c]:
            continue
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            fill[r, c] = True
            continue
        if any((nr, nc) in walls for nr, nc in neighbors8(r, c)):
            fill[r, c] = True
            continue
        if any((nr, nc) not in reach for nr, nc in neighbors4(r, c)):
            fill[r, c] = True
    return fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract only the wall-and-ring scaffold around the marker-side region.

    Input structure:
    - The dominant color is background; a single magenta corner pixel marks the flood-fill start.
    - Blue pixels mix two roles: border-anchored structural walls and small decorative blobs.
    - Structural walls, together with the outer frame, carve the grid into marker-reachable and hidden sides.

    Rule:
    - Find the corner marker, infer the wall color, and flood from the marker while treating all blue cells as solid to identify which border-touching blue components truly face the marker side.
    - Flood again while allowing travel through non-structural blue blobs so decorations do not alter the reachable-region geometry.
    - Emit only the boundary scaffold: keep structural wall pixels blue and paint orange on reachable background cells that lie on the inner boundary; erase the marker, all decorations, and all non-boundary space to background.

    Key insight:
    - The output is now a sparse boundary mask rather than the full processed scene, so the crucial step is separating real structural walls from decorative blue pixels before drawing the orange ring.
    """
    g = np.asarray(grid, dtype=int)
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

    other_cols = [col for col in counts if col not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda col: counts[col])

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
            walls.update(comp)

    reach = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) == wall_col and (r, c) not in walls),
        h,
        w,
    )

    out = np.full((h, w), bg, dtype=int)
    for r, c in walls:
        out[r, c] = wall_col

    fill = _boundary_fill_mask(g == bg, reach, walls, h, w)
    out[fill] = 7
    return out


BG = 8
WALL = 1
MARKER = 6
FILL = 7

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

NUM_STRUCTS_RANGE = (1, 4)
STRUCT_SHAPE_RETRIES = 8
RECT_MAX_DIM_DIV = 3
TRI_MAX_SIZE = 6
TRI_MIN_DIM_DIV = 3

REACH_RATIO_RANGE = (0.38, 0.87)

NUM_DECOR_RANGE = (1, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

GEN_ATTEMPTS = 300


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


def _dilate4(mask):
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


def _validate_structural(structural, reach, h, w):
    reach_nbr = _dilate4(reach)
    for comp_arr in components_mask(structural, connectivity=4):
        cmask = np.zeros((h, w), dtype=bool)
        cmask[comp_arr[:, 0], comp_arr[:, 1]] = True
        if not _touches_border(cmask):
            return False
        if not (cmask & reach_nbr).any():
            return False
    return True


def _paint_output(structural, reach, marker_rc):
    h, w = structural.shape
    out = np.full((h, w), BG, dtype=int)
    out[structural] = WALL

    reach_set = {tuple(map(int, rc)) for rc in np.argwhere(reach)}
    wall_set = {tuple(map(int, rc)) for rc in np.argwhere(structural)}
    bg_cells = ~structural
    bg_cells[marker_rc] = False
    fill = _boundary_fill_mask(bg_cells, reach_set, wall_set, h, w)
    out[fill] = FILL
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
            placed[r0 : r0 + ph, c0 : c0 + pw] = True
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
    reach_frac = reach.mean()
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    if not _validate_structural(structural, reach, h, w):
        return None

    out = _paint_output(structural, reach, marker_rc)
    fill_mask = out == FILL
    if not fill_mask.any() or len(components_mask(fill_mask, connectivity=4)) != 1:
        return None

    interior_bg = reach & ~structural & ~fill_mask
    if interior_bg.sum() < 4:
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(1, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = reach & (out == BG)
    hidden_allowed = (~reach) & (~structural)
    hidden_allowed[0, :] = False
    hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = False
    hidden_allowed[:, -1] = False

    visible_placed = _place_decor(visible_allowed, n_visible, rng)
    if visible_placed is None or visible_placed.sum() == 0:
        return None
    hidden_placed = _place_decor(hidden_allowed, n_hidden, rng)
    if hidden_placed is None:
        return None

    inp = out.copy()
    inp[inp == FILL] = BG
    inp[visible_placed] = WALL
    inp[hidden_placed] = WALL
    inp[marker_rc] = MARKER

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None

    if np.count_nonzero(inp == WALL) <= np.count_nonzero(out == WALL):
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D9 variant")
