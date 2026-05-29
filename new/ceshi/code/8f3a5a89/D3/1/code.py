# Variant of 8f3a5a89
# Dimension: D3 — 源对象（标记像素）位置 (src)
# Change: 单个品红像素固定在底左角 (H-1, 0)，始终为BL角... → 标记在网格边缘（非角落）

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for the D3 variant
# -----------------------------------------------------------------------------
# variant_spec:
#   core_change:
#     from: marker is the only unique non-bg pixel and is always at a corner
#     to:   marker is still a unique single magenta pixel, but now it lies on
#           exactly one border cell that is not a corner
#   concrete_visualization:
#     - Input still contains azure background, blue wall pixels, and one magenta
#       source pixel.
#     - The blue structures still touch the border and partition the grid.
#     - The orange output border is still painted on the marker-reachable side.
#     - The only visible D3 difference is where the flood starts: a side entry
#       point such as top-middle, left-middle, right-middle, or bottom-middle.
#   behavior_trace:
#     - Original:
#       1. Search only the four corners for the unique marker.
#       2. Flood from that corner through bg/marker.
#       3. Re-identify border-touching blue components next to the flood as
#          structural walls, then paint the inner orange boundary.
#     - Variant:
#       1. Search all border cells except the four corners for the unique marker.
#       2. Flood from that edge marker through bg/marker.
#       3. Keep the same wall-identification and border-painting logic.
#     - Difference:
#       the only semantic switch is the flood origin; this changes which side of
#       asymmetric wall layouts is reachable and makes the orange lining start
#       from a non-corner edge position.
#   complexity_calibration:
#     - The rule remains single-mechanism and human-readable: "start from the
#       edge marker instead of a corner marker".
#     - To keep the change visually obvious, the generator places the marker in
#       a mid-band of an edge rather than near corners.
#     - Wall/decor complexity is unchanged; no extra color roles or hidden
#       casework are introduced.
#   implicit_assumption_audit_main:
#     - counts-based marker detection still assumes the marker color occurs once.
#       The generator guarantees exactly one magenta pixel.
#     - Border scan used to assume "marker must be in corners"; this is the one
#       invalidated assumption and is replaced with "marker must be on a
#       non-corner border cell".
#     - Wall-color inference assumes the wall color is the dominant remaining
#       non-bg color. The generator still uses a fixed blue wall color and keeps
#       all decorations blue, so this remains valid.
#     - Structural-wall recovery assumes every true barrier both touches the
#       border and is adjacent to the first flood region. The generator still
#       validates this condition after placing walls.
#     - Orange painting assumes the marker cell itself should not be repainted.
#       This still holds because painting only applies to bg cells.
#   placement_legality:
#     - The generator picks a border marker away from corners.
#     - Structural shapes are rejected if they cover the marker.
#     - Hidden decorations are forbidden from touching the border, preventing
#       them from being mistaken for structural walls.
#     - Output-first validation is preserved: generated inputs are accepted only
#       if main(input) exactly matches the intended output.
#   visual_observability:
#     - In 2-4 examples, the viewer can directly spot that the magenta seed is
#       no longer in a corner.
#     - Because the marker can appear on any of the four sides, the variant has
#       repeated support for the new rule rather than a one-off special case.
#   cleanliness:
#     - No new overlays are introduced.
#     - The marker stays a single pixel and decorations remain small, so the
#       grid is still legible even when the reachable region becomes asymmetric.


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


def _noncorner_border_cells(h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(1, w - 1):
        cells.append((0, c))
        cells.append((h - 1, c))
    for r in range(1, h - 1):
        cells.append((r, 0))
        cells.append((r, w - 1))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint an inner border around the region reachable from an edge marker pixel.

    Input structure:
    - bg = most common color; wall color (blue) forms border-touching barriers.
    - A unique single pixel (magenta) sits on the grid border but not in a corner.
    - Some wall-colored pixels are structural dividers, while others are decorative fragments.

    Rule:
    - Flood-fill from the border marker to find all bg cells reachable without crossing structural walls.
    - Erase decorative wall fragments that lie only on the unreachable side of those barriers.
    - Paint orange (color 7) on each reachable bg cell that lies on the reachable region's inner boundary.

    Key insight: The wall logic is unchanged; the only real difference is that the flood origin is the unique non-corner border marker instead of a corner pixel.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()

    # Marker pixel is a unique (count==1) non-bg pixel on the border, excluding corners.
    counts = Counter(int(v) for v in g.ravel())
    marker = None
    for r, c in _noncorner_border_cells(h, w):
        v = int(g[r, c])
        if v != bg and counts.get(v, 0) == 1:
            marker = (r, c, v)
            break
    assert marker is not None, "no unique single-pixel non-corner border marker found"

    mr, mc, marker_col = marker

    # Wall color: most frequent non-bg, non-marker color.
    other_cols = [c for c in counts.keys() if c not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda c: counts[c])

    # First reach: treat all wall pixels as obstacles.
    reach0 = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col),
        h,
        w,
    )

    # Structural walls are wall-color components that touch the border AND are adjacent to reach0.
    walls: set[tuple[int, int]] = set()
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = [(int(r), int(c)) for r, c in comp_arr]
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if not touches_border:
            continue
        adjacent_to_reach0 = any(any((nr, nc) in reach0 for nr, nc in neighbors4(r, c)) for r, c in comp)
        if adjacent_to_reach0:
            walls |= set(comp)

    # Second reach: allow passing through non-structural wall-color pixels (decorations).
    reach = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) == wall_col and (r, c) not in walls),
        h,
        w,
    )

    # Remove wall-colored decorations that are not reachable (they lie behind structural walls).
    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and (r, c) not in walls and (r, c) not in reach:
                out[r, c] = bg

    fill_col = 7
    # Paint a 1-cell-thick inner border of the reachable region (on background cells only).
    for r, c in reach:
        if int(out[r, c]) != bg:
            continue
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            out[r, c] = fill_col
            continue
        if any((nr, nc) in walls for nr, nc in neighbors8(r, c)):
            out[r, c] = fill_col
            continue
        if any((nr, nc) not in reach for nr, nc in neighbors4(r, c)):
            out[r, c] = fill_col

    return out


# --------------- generator ---------------

# ---- Palette (task colors) ----
BG = 8
WALL = 1
MARKER = 6
FILL = 7

# ---- Canvas size ----
GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

# ---- Structural walls ----
NUM_STRUCTS_RANGE = (1, 4)
STRUCT_SHAPE_RETRIES = 8
RECT_MAX_DIM_DIV = 3
TRI_MAX_SIZE = 6
TRI_MIN_DIM_DIV = 3

# ---- Reachable region gate ----
REACH_RATIO_RANGE = (0.38, 0.87)

# ---- Decorations ----
NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

# ---- Outer retry budget ----
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


def _dilate8(mask):
    m = _dilate4(mask)
    m[1:, 1:] |= mask[:-1, :-1]
    m[1:, :-1] |= mask[:-1, 1:]
    m[:-1, 1:] |= mask[1:, :-1]
    m[:-1, :-1] |= mask[1:, 1:]
    return m


def _pick_marker(h: int, w: int, rng) -> tuple[int, int]:
    edge = rng.choice(["top", "bottom", "left", "right"])
    if edge in ("top", "bottom"):
        lo = max(1, w // 4)
        hi = min(w - 2, w - 1 - w // 4)
        c = rng.randint(lo, hi)
        return (0, c) if edge == "top" else (h - 1, c)
    lo = max(1, h // 4)
    hi = min(h - 2, h - 1 - h // 4)
    r = rng.randint(lo, hi)
    return (r, 0) if edge == "left" else (r, w - 1)


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

    edge = np.zeros_like(candidate)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True

    near_struct = _dilate8(structural)
    near_unreach = _dilate4(~reach)

    fill = candidate & (edge | near_struct | near_unreach)
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
            placed[r0:r0 + ph, c0:c0 + pw] = True
            break
        else:
            return None
    return placed


def _gen_attempt():
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = _pick_marker(h, w, rng)

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

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = reach & (out == BG)
    hidden_allowed = (~reach) & (~structural)
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D3 variant")
