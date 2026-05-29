# Variant of 8f3a5a89
# Dimension: D19 — 符号编码/图例机制 (symbol_instructions)
# Change: 无显式图例——所有规则从输入输出对中隐式推断... → 标记像素形状编码边界厚度

from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# Design analysis for the D19 variant
# a) Concrete behavior:
#    The old task used a single magenta pixel at the bottom-left corner. Here that
#    cue becomes a small corner-anchored magenta polyomino. Its pixel count is the
#    only new piece of information: 2/3/4 marker pixels mean a 2/3/4-cell-thick
#    orange inner border. The geometry stays simple and nested so the solver can
#    count pixels instead of relying on a legend. Example:
#    - old input: one magenta pixel at (H-1,0) -> old output paints exactly one
#      orange layer along the marker-reachable side of the structural walls.
#    - new input: a 3-pixel L-shaped magenta marker at the same corner -> new
#      output paints the same reachable-side border, but expanded inward to three
#      geodesic layers within the reachable background region.
# b) Complexity calibration:
#    Thickness uses only three values (2, 3, 4). That is enough to be visually
#    obvious and repeatedly supported across examples, while avoiding a large
#    unconstrained numeric space that would be hard to infer from 2-4 examples.
# c) Implicit-assumption audit of the original main():
#    - The old corner scan expected a unique count==1 marker pixel. That fails once
#      the marker is multi-pixel, so the new solver instead extracts the entire
#      bottom-left marker component and decodes thickness from its area.
#    - The old fill thickness was hard-coded by directly painting the first border
#      layer. That is no longer safe; the new solver computes the original border
#      seed first, then performs a 4-neighbor inward expansion for k layers.
#    - The old wall-color inference excluded one marker pixel color; that still
#      works once we exclude the marker component color rather than requiring count
#      1, because the generator keeps the palette limited to bg/wall/marker.
# d) Placement legality:
#    Marker cells are fixed to the bottom-left corner and reserved before wall
#    placement. Structural walls may touch borders but never overlap the marker.
#    Visible decorations are placed only in reachable non-fill interior cells so
#    they do not obscure the thickness cue. Hidden decorations remain off-border so
#    the solver does not misclassify them as structural walls.
# e) Visual observability:
#    The thickness parameter is visible twice: the marker has 2/3/4 magenta cells,
#    and the orange band has matching width. The band is required to have at least
#    one cell in its deepest painted layer and at least one still-unfilled interior
#    background cell, so the width does not collapse into the original 1-layer case
#    or into a fully orange region with no contrast.
# f) Visual cleanliness:
#    The marker stays in one corner, walls stay blue, fill stays orange, and visible
#    decorations are kept away from the orange band. This limits overlap and keeps
#    the new symbol cue easy to read.
# g) Color inferability:
#    No new color mapping is introduced. Magenta still marks the starting side, but
#    now its shape size encodes a number. Blue remains walls, orange remains the
#    computed border, and azure remains background.
# 9) New main() assumptions and generator guarantees:
#    - main() assumes the bottom-left cell belongs to the marker component.
#      generate_input() always anchors the marker there.
#    - main() assumes the marker color forms exactly one meaningful corner component.
#      generate_input() uses marker color only for that shape.
#    - main() assumes the reachable region has enough depth to display the decoded
#      thickness. generate_input() rejects layouts whose border depth is too small.


def _flood(
    starts: list[tuple[int, int]] | set[tuple[int, int]],
    passable,
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    q = deque()
    vis: set[tuple[int, int]] = set()
    for cell in starts:
        if cell in vis:
            continue
        vis.add(cell)
        q.append(cell)
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and passable(nr, nc):
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis


def _distance_from_seed(candidate_mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    h, w = candidate_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q = deque()
    for r, c in np.argwhere(candidate_mask & seed_mask):
        rr, cc = int(r), int(c)
        dist[rr, cc] = 0
        q.append((rr, cc))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and candidate_mask[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _marker_cells_for_thickness(h: int, thickness: int) -> list[tuple[int, int]]:
    shapes = {
        2: [(h - 1, 0), (h - 1, 1)],
        3: [(h - 1, 0), (h - 1, 1), (h - 2, 0)],
        4: [(h - 1, 0), (h - 1, 1), (h - 2, 0), (h - 2, 1)],
    }
    return shapes[thickness]


def _extract_marker(g: np.ndarray, bg: int) -> tuple[set[tuple[int, int]], int, int]:
    h, _w = g.shape
    marker_col = int(g[h - 1, 0])
    assert marker_col != bg, "bottom-left marker block missing"
    marker_comp = None
    for comp_arr in components_mask(g == marker_col, connectivity=4):
        comp = {(int(r), int(c)) for r, c in comp_arr}
        if (h - 1, 0) in comp:
            marker_comp = comp
            break
    assert marker_comp is not None, "no marker component anchored at bottom-left"
    thickness = len(marker_comp)
    assert thickness >= 1, "empty marker component"
    return marker_comp, marker_col, thickness


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint a thickness-coded inner border around the region reachable from a corner marker block.

    Input structure:
    - bg is the dominant color; blue wall pixels divide the canvas into reachable and hidden sides.
    - A magenta marker component is anchored at the bottom-left corner instead of being a single pixel.
    - The marker component's pixel count is the instruction: it encodes how many orange border layers to paint.

    Rule:
    - Treat the marker component as the flood-fill start region and find the bg area reachable from it while blue structural walls block movement.
    - Remove blue decoration pixels that sit behind the structural walls on the unreachable side, but keep reachable decorations.
    - Build the original 1-layer inner border of the reachable region, then expand that border inward by the number of layers encoded by the marker size.

    Key insight: The magenta shape acts like a tiny numeric legend embedded in the puzzle; counting its cells tells you how thick the orange border must be.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    marker_cells, marker_col, thickness = _extract_marker(g, bg)

    counts = Counter(int(v) for v in g.ravel())
    other_cols = [c for c in counts.keys() if c not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda c: counts[c])

    marker_list = sorted(marker_cells)

    reach0 = _flood(
        marker_list,
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
        marker_list,
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) == wall_col and (r, c) not in walls),
        h,
        w,
    )

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and (r, c) not in walls and (r, c) not in reach:
                out[r, c] = bg

    candidate = np.zeros((h, w), dtype=bool)
    seed = np.zeros((h, w), dtype=bool)
    for r, c in reach:
        if int(out[r, c]) != bg:
            continue
        candidate[r, c] = True
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            seed[r, c] = True
            continue
        if any((nr, nc) in walls for nr, nc in neighbors8(r, c)):
            seed[r, c] = True
            continue
        if any((nr, nc) not in reach for nr, nc in neighbors4(r, c)):
            seed[r, c] = True

    fill_col = 7
    dist = _distance_from_seed(candidate, seed)
    fill_mask = candidate & (dist >= 0) & (dist < thickness)
    out[fill_mask] = fill_col
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

MARKER_THICKNESS_CHOICES = (2, 3, 4)
REACH_RATIO_RANGE = (0.42, 0.87)

NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

GEN_ATTEMPTS = 600


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


def _flood_mask(starts: list[tuple[int, int]], blocked: np.ndarray, h: int, w: int) -> np.ndarray:
    reach = np.zeros((h, w), dtype=bool)
    q = deque()
    for r, c in starts:
        if blocked[r, c] or reach[r, c]:
            continue
        reach[r, c] = True
        q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reach[nr, nc] and not blocked[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def _touches_border(mask: np.ndarray) -> bool:
    return mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    m = _dilate4(mask)
    m[1:, 1:] |= mask[:-1, :-1]
    m[1:, :-1] |= mask[:-1, 1:]
    m[:-1, 1:] |= mask[1:, :-1]
    m[:-1, :-1] |= mask[1:, 1:]
    return m


def _place_structural_walls(h: int, w: int, forbidden: np.ndarray, rng) -> np.ndarray | None:
    num = rng.randint(*NUM_STRUCTS_RANGE)
    structural = np.zeros((h, w), dtype=bool)
    placed = 0
    for _ in range(num):
        for _try in range(STRUCT_SHAPE_RETRIES):
            cells = rng.choice(SHAPE_FACTORIES)(h, w, rng)
            if not cells:
                continue
            if any(forbidden[r, c] or structural[r, c] for r, c in cells):
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


def _compute_fill_mask(
    structural: np.ndarray,
    reach: np.ndarray,
    marker_mask: np.ndarray,
    thickness: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate = reach & ~(structural | marker_mask)

    edge = np.zeros_like(candidate)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True

    near_struct = _dilate8(structural)
    near_unreach = _dilate4(~reach)

    seed = candidate & (edge | near_struct | near_unreach)
    dist = _distance_from_seed(candidate, seed)
    fill = candidate & (dist >= 0) & (dist < thickness)
    return seed, fill, dist


def _paint_output(
    structural: np.ndarray,
    reach: np.ndarray,
    marker_mask: np.ndarray,
    thickness: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    out = np.full(structural.shape, BG, dtype=int)
    out[structural] = WALL
    out[marker_mask] = MARKER
    seed, fill, dist = _compute_fill_mask(structural, reach, marker_mask, thickness)
    out[fill] = FILL
    return out, fill, seed, dist


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


def _gen_attempt() -> np.ndarray | None:
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    thickness = rng.choice(MARKER_THICKNESS_CHOICES)
    marker_cells = _marker_cells_for_thickness(h, thickness)
    marker_mask = np.zeros((h, w), dtype=bool)
    for r, c in marker_cells:
        marker_mask[r, c] = True

    structural = _place_structural_walls(h, w, marker_mask, rng)
    if structural is None or (structural & marker_mask).any():
        return None

    reach = _flood_mask(marker_cells, structural, h, w)
    reach_frac = reach.mean()
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    if not _validate_structural(structural, reach, h, w):
        return None

    out, fill_mask, _seed_mask, dist = _paint_output(structural, reach, marker_mask, thickness)
    if not fill_mask.any() or len(components_mask(fill_mask, connectivity=4)) != 1:
        return None

    if not np.any(dist == thickness - 1):
        return None
    if dist.max() < thickness:
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = reach & ~(structural | marker_mask | fill_mask)
    hidden_allowed = (~reach) & (~structural) & (~marker_mask)
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
