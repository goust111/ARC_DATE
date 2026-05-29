# Variant of 8f3a5a89
# Dimension: D15 — 标记像素唯一性识别 (src)
# Change: 标记像素 = 在角落位置、非背景色、网格中只出现1次的像素... → 标记为特定形状而非单像素

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# Design analysis for the D15 variant
# a) Concreteization and behavior trace:
#    The original task used a single magenta pixel at the bottom-left corner as the flood seed.
#    This variant keeps the same wall / decoration / fill semantics, but replaces that 1x1 seed
#    with a compact corner sprite. The sprite is still magenta, still anchored at bottom-left,
#    but now occupies 3 or 4 connected cells in a tiny 2x2 or 3x2 footprint. The solver must
#    first identify the corner component as one of a few legal marker templates, then treat the
#    whole component as the start region. Example trace:
#    - Old: read (H-1,0), verify its color count is 1, flood from that one coordinate.
#    - New: read the same-color component touching (H-1,0), verify its local shape matches one
#      of the allowed corner sprites, then flood while allowing all sprite cells as seeds/passable.
#    - Difference: color-frequency uniqueness is no longer informative; local corner geometry is.
# b) Complexity calibration:
#    A single fixed 2x2 block would make the change visible but a bit too uniform. Using a small
#    menu of 2-3 compact templates keeps the rule human-readable while preventing trivial collapse
#    back to "just look for one pixel". The shape family stays tiny so examples do not require
#    learning a large dictionary.
# c) Implicit assumption audit:
#    - Original marker detection assumed count(marker_color) == 1. This is invalid now.
#    - Original _place_structural_walls only avoided one marker cell; now it must avoid all marker
#      cells.
#    - Original generator enforced fill to be a single 4-connected component. That was harmless for
#      a 1x1 corner marker, but it is no longer reliable because a multi-cell sprite can interrupt
#      the orange edge border at the corner. The rule itself does not require single-component fill,
#      so this generation-only check must be removed.
#    - Wall-color inference by "most frequent non-bg, non-marker color" remains valid because the
#      marker still uses its own color and walls still dominate that palette share.
# d) Placement legality:
#    Marker cells are generated first, all in-bounds by template construction, and structural walls
#    are forbidden from overlapping them. Decorations are still placed only on allowed masks, with
#    the same padding and border restrictions as the original generator.
# e) Visual observability:
#    Across samples, the bottom-left cue is visibly a small magenta sprite rather than a singleton.
#    That is directly visible from 2-3 examples without any hidden counting logic. The rest of the
#    task stays visually familiar, so the changed factor is easy to isolate.
# f) Visual cleanliness:
#    The sprite stays compact and corner-anchored, so it does not visually compete with wall shapes.
#    No extra colors are introduced, and the interior border / wall layout remains uncluttered.
# g) Color-change inferability:
#    No new color mapping is introduced. Marker stays magenta, walls stay blue, fill stays orange.


def _flood(
    starts: list[tuple[int, int]] | set[tuple[int, int]] | tuple[tuple[int, int], ...],
    passable,
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    q = deque()
    vis: set[tuple[int, int]] = set()
    for start in starts:
        r, c = start
        if 0 <= r < h and 0 <= c < w and passable(r, c) and (r, c) not in vis:
            vis.add((r, c))
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and passable(nr, nc):
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis


def _same_color_component(g: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = g.shape
    sr, sc = start
    col = int(g[sr, sc])
    q = deque([start])
    vis = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) == col:
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis


MARKER_TEMPLATES = {
    frozenset({(0, 0), (0, 1), (1, 0)}),                  # 3-cell corner elbow
    frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),          # 2x2 block
    frozenset({(0, 0), (0, 1), (1, 0), (2, 0)}),          # tall L
}


def _marker_local_shape(cells: set[tuple[int, int]], anchor: tuple[int, int]) -> frozenset[tuple[int, int]]:
    ar, ac = anchor
    return frozenset((ar - r, c - ac) for r, c in cells)


def _find_marker(g: np.ndarray, bg: int) -> tuple[set[tuple[int, int]], int]:
    h, _ = g.shape
    anchor = (h - 1, 0)
    marker_col = int(g[anchor])
    assert marker_col != bg, "bottom-left marker sprite missing"
    marker_cells = _same_color_component(g, anchor)
    local_shape = _marker_local_shape(marker_cells, anchor)
    assert local_shape in MARKER_TEMPLATES, "bottom-left corner component is not a legal marker sprite"
    return marker_cells, marker_col


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint an inner border around the region reachable from a corner marker sprite.

    Input structure:
    - bg is the dominant color; blue cells form barrier-like wall components.
    - The bottom-left corner holds a small connected magenta sprite rather than a single seed pixel.
    - Some blue components are true border-connected walls, while others are decorative fragments.

    Rule:
    - Detect the legal corner sprite at bottom-left and use its full component as the flood source.
    - Flood through background and marker cells, first with all blue cells blocked to identify which
      border-touching blue components truly separate the reachable side from the hidden side.
    - Remove unreachable blue decorations, then paint orange on reachable background cells that lie
      on the inner boundary of the reachable region.

    Key insight: The start cue is no longer defined by global uniqueness of one corner pixel; it is
    defined by a compact corner shape, after which the original wall-vs-decoration logic still applies.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    marker_cells, marker_col = _find_marker(g, bg)

    counts = Counter(int(v) for v in g.ravel())
    other_cols = [c for c in counts.keys() if c not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda c: counts[c])

    reach0 = _flood(
        sorted(marker_cells),
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
        sorted(marker_cells),
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) == wall_col and (r, c) not in walls),
        h,
        w,
    )

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and (r, c) not in walls and (r, c) not in reach:
                out[r, c] = bg

    fill_col = 7
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

MARKER_SHAPES = {
    "elbow3": ((0, 0), (0, 1), (1, 0)),
    "block4": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "tall_l4": ((0, 0), (0, 1), (1, 0), (2, 0)),
}


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


def _flood_mask(starts, blocked, h, w):
    reach = np.zeros((h, w), dtype=bool)
    q = deque()
    for r, c in starts:
        if 0 <= r < h and 0 <= c < w and not blocked[r, c] and not reach[r, c]:
            reach[r, c] = True
            q.append((r, c))
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


def _marker_cells_from_shape(h: int, shape_name: str) -> set[tuple[int, int]]:
    anchor = (h - 1, 0)
    ar, ac = anchor
    return {(ar - dr, ac + dc) for dr, dc in MARKER_SHAPES[shape_name]}


def _place_structural_walls(h, w, marker_cells, rng):
    num = rng.randint(*NUM_STRUCTS_RANGE)
    structural = np.zeros((h, w), dtype=bool)
    placed = 0
    marker_cells = set(marker_cells)
    for _ in range(num):
        for _try in range(STRUCT_SHAPE_RETRIES):
            cells = rng.choice(SHAPE_FACTORIES)(h, w, rng)
            if not cells:
                continue
            if any((r, c) in marker_cells or structural[r, c] for r, c in cells):
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


def _paint_output(structural, reach, marker_cells):
    out = np.full(structural.shape, BG, dtype=int)
    out[structural] = WALL
    for r, c in marker_cells:
        out[r, c] = MARKER

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
    shape_name = rng.choice(sorted(MARKER_SHAPES))
    marker_cells = _marker_cells_from_shape(h, shape_name)

    structural = _place_structural_walls(h, w, marker_cells, rng)
    if structural is None or any(structural[r, c] for r, c in marker_cells):
        return None

    reach = _flood_mask(sorted(marker_cells), structural, h, w)
    reach_frac = reach.mean()
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    if not _validate_structural(structural, reach, h, w):
        return None

    out = _paint_output(structural, reach, marker_cells)

    fill_mask = out == FILL
    if not fill_mask.any():
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D15 variant")
