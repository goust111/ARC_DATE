# Variant of 8f3a5a89
# Dimension: D2 — 内边界涂色规则 (trajectory)
# Change: inner_border——可达区域中，位于网格边缘、8-邻接结构墙、或4-邻接不可达像素的背景格子涂橙色，形成1像素宽... → 渐变厚度边界

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# Design analysis for the D2 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     old: reachable background cells on the inner boundary are always painted with a 1-pixel ring
#     new: the orange boundary keeps a 1-pixel thickness where it follows the outer grid edge,
#       but becomes 2 pixels thick next to structural blue walls by adding one extra inward layer
#   concrete_pixel_behavior:
#     reachable-side boundary detection is still based on the original three visible cues:
#       grid edge, 8-neighborhood of structural walls, and 4-neighborhood of unreachable cells
#     the second layer is not global; it only appears on reachable background cells that lie one
#       more 4-step inward from wall-adjacent boundary cells, and never on the grid edge itself
#   behavior_trace:
#     example:
#       original:
#         a vertical structural wall yields one orange column immediately on its reachable side
#       variant:
#         the same wall yields two orange columns on its reachable side, while the bottom and left
#         frame around the marker remain a single orange stripe
#       difference:
#         solver must distinguish wall-following boundary segments from edge-following segments
#   complexity_calibration:
#     only two thickness levels are used: 1 and 2
#     this is enough to be visually obvious without introducing multiple hard-to-infer cases
#     generator requires both edge-only orange cells and wall-thickened orange cells so the rule
#       cannot collapse back into the original constant-width ring
#   hidden_assumption_audit:
#     original code assumes every fill cell can be assigned independently once it matches a
#       1-layer predicate; that breaks under the variant because a second-layer cell could be
#       overwritten or omitted if thickness is still treated as a one-pass local test
#     the fix is to compute:
#       1. the original boundary seed mask
#       2. an extra inward layer derived only from wall-adjacent seed cells
#     decoration erasure and structural wall identification still rely on the old invariants and
#       remain valid, so those stages are intentionally preserved
#   placement_legality:
#     structural wall placement remains border-touching and non-overlapping
#     visible decorations are still restricted to reachable non-fill interior cells so they never
#       occlude the thickness cue
#     hidden decorations remain non-border-touching so the solver does not mistake them for walls
#   observability:
#     generator rejects samples unless:
#       there are enough extra second-layer cells near walls
#       there are enough edge-border orange cells
#       some reachable azure interior remains after painting
#     this makes the 1-pixel-vs-2-pixel contrast visible across a small number of examples
#   cleanliness:
#     orange is still the only transformed color; no new palette logic is introduced
#     the extra layer is contiguous with the base border, keeping the output legible


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


def _coords_to_mask(coords: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if coords:
        arr = np.array(list(coords), dtype=int)
        mask[arr[:, 0], arr[:, 1]] = True
    return mask


def _edge_mask(h: int, w: int) -> np.ndarray:
    edge = np.zeros((h, w), dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True
    return edge


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


def _boundary_masks(
    reach_mask: np.ndarray,
    walls_mask: np.ndarray,
    candidate_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = reach_mask.shape
    edge = _edge_mask(h, w)
    near_wall = candidate_mask & _dilate8(walls_mask)
    near_unreach = candidate_mask & _dilate4(~reach_mask)
    base = candidate_mask & (edge | near_wall | near_unreach)
    extra = candidate_mask & ~base & ~edge & _dilate4(near_wall)
    return edge, near_wall, near_unreach, base, extra


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint a variable-thickness inner border around the region reachable from the corner marker.

    Input structure:
    - The most frequent color is background, blue components act as either structural walls or small decorations.
    - A unique single-pixel corner marker indicates the flood-fill start, and it stays unchanged.
    - Structural walls are the border-touching blue components on the reachable side; other blue pieces are decorative.

    Rule:
    - Flood from the marker to recover the reachable side, first with all blue blocked and then with only structural walls blocked.
    - Erase decorative blue pixels that lie on the unreachable side of the structural walls.
    - Paint orange on the reachable inner boundary, but use two thicknesses: segments along the grid edge stay 1 pixel thick, while segments adjacent to structural walls gain one extra inward layer.

    Key insight:
    - The same reachability analysis as the original task still isolates the real barrier walls; the new difficulty is that only wall-following boundary segments are thickened, so border type controls orange thickness.
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

    walls_mask = _coords_to_mask(walls, h, w)
    reach_mask = _coords_to_mask(reach, h, w)
    candidate_mask = reach_mask & (out == bg)
    _edge, _near_wall, _near_unreach, base, extra = _boundary_masks(reach_mask, walls_mask, candidate_mask)
    out[base | extra] = 7
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

NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

GEN_ATTEMPTS = 300
MIN_EXTRA_FILL = 5
MIN_EDGE_FILL = 6
MIN_INTERIOR_BG = 6


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
    out[marker_rc] = MARKER

    candidate = reach & (out == BG)
    edge, near_wall, near_unreach, base, extra = _boundary_masks(reach, structural, candidate)
    fill = base | extra
    out[fill] = FILL
    return out, {
        "edge": edge,
        "near_wall": near_wall,
        "near_unreach": near_unreach,
        "base": base,
        "extra": extra,
        "fill": fill,
        "candidate": candidate,
    }


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


def _variant_metrics(masks: dict[str, np.ndarray]) -> dict[str, int]:
    fill = masks["fill"]
    edge = masks["edge"]
    extra = masks["extra"]
    candidate = masks["candidate"]
    return {
        "extra_fill": int(extra.sum()),
        "edge_fill": int((fill & edge).sum()),
        "interior_bg": int((candidate & ~fill).sum()),
    }


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

    out, masks = _paint_output(structural, reach, marker_rc)
    metrics = _variant_metrics(masks)
    if metrics["extra_fill"] < MIN_EXTRA_FILL:
        return None
    if metrics["edge_fill"] < MIN_EDGE_FILL:
        return None
    if metrics["interior_bg"] < MIN_INTERIOR_BG:
        return None

    fill_mask = masks["fill"]
    if not fill_mask.any() or len(components_mask(fill_mask, connectivity=4)) != 1:
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = reach & (out == BG)
    hidden_allowed = (~reach) & (~structural)
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
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 8f3a5a89 variant D2/3")
