# Variant of 8f3a5a89
# Dimension: D14 — 涂色颜色规则 (trajectory)
# Change: 统一涂橙色 (color 7)，所有可达边界像素使用同一种颜色... → 按距离标记不同颜色
#
# Design analysis
# - Reference-task check:
#   - I searched the provided workspace for a standalone solver for source task
#     `195c6913`. Only the ARC JSON exists; no solver file is available here.
#     The implementation below therefore reuses only the visible idea suggested
#     by the prompt: a distance-driven color cycle along a trajectory/boundary.
# - a) Concretization and behavior trace:
#   - The input still shows a blue wall layout on azure background with a single
#     magenta pixel at the bottom-left corner. The output still paints the
#     reachable inner boundary of the marker-side region, but the painted cells
#     are no longer uniform. Instead, each painted boundary cell gets a color
#     from a fixed 3-step cycle based on Manhattan distance from the marker:
#     distance 1 -> 7, distance 2 -> 4, distance 3 -> 2, then repeat.
#   - Example: if boundary cells at (h-2,0), (h-1,1), (h-3,0), (h-2,1) are
#     paintable, their distances from the bottom-left marker are 1, 1, 2, 2, so
#     they become 7, 7, 4, 4. Under the original task they would all become 7.
#   - The changed step is only the final painting pass. Structural-wall
#     detection, flood-fill reachability, and hidden-decoration removal stay the
#     same.
# - b) Complexity calibration:
#   - A binary near/far threshold would be weak because a few examples may not
#     reveal where the threshold sits. A fixed 3-color cycle gives repeated
#     evidence inside each sample and across samples while keeping the rule
#     simple enough for ARC inference.
# - c) Implicit-assumption audit against the original main():
#   - The original `fill_col = 7` assumed every boundary cell could be assigned
#     with one unconditional color. That is no longer safe: the fill color now
#     depends on each cell position relative to the marker.
#   - The earlier code also relied on the fact that output-only fill color never
#     influenced wall identification. That still holds because the input
#     generator removes every trajectory color before validation.
#   - The "do not paint non-bg reachable cells" guard remains necessary because
#     the marker and visible wall decorations are still reachable.
# - d) Placement legality:
#   - Structural walls remain border-touching 4-connected blue components.
#   - Hidden decorations are still forced away from the border so the solver
#     does not misclassify them as structural walls.
#   - Visible decorations are still placed only on reachable non-boundary
#     interior background cells, so they do not obscure the trajectory colors.
# - e) Visual observability:
#   - The generator rejects samples unless the painted boundary spans enough
#     distance values to show all three trajectory colors with repetition.
#     This prevents accidental degeneration back to an almost-uniform border.
# - f) Visual cleanliness:
#   - Only the original one-cell-thick inner border is recolored. No extra
#     overlays are added, so walls, marker, background, and trajectory bands
#     stay visually separable.
# - g) Color-change inferability:
#   - The palette is capped at three trajectory colors (7, 4, 2).
#   - The mapping is tied to an observable spatial cue: Manhattan distance from
#     the visible corner marker, not to hidden ids or arbitrary ordering.
# - 9) New solver assumptions and generator guarantees:
#   - Solver assumption: the marker remains a unique corner singleton. Generator
#     guarantee: marker is always the single magenta pixel at bottom-left.
#   - Solver assumption: at least one non-marker wall color exists. Generator
#     guarantee: at least one structural blue component is placed.
#   - Solver assumption: the trajectory boundary includes several distance
#     classes so the new rule is visible. Generator guarantee: the output is
#     rejected unless all three cycle colors occur at least twice and the
#     boundary spans a nontrivial distance range.

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


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


def _flood_mask(start, blocked, h, w):
    """BFS from `start` on the ~blocked bool grid; return reach as (h,w) bool array."""
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
    """4-connected dilation by 1 via boolean slicing."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    """8-connected dilation by 1 via 4-conn dilation plus diagonals."""
    m = _dilate4(mask)
    m[1:, 1:] |= mask[:-1, :-1]
    m[1:, :-1] |= mask[:-1, 1:]
    m[:-1, 1:] |= mask[1:, :-1]
    m[:-1, :-1] |= mask[1:, 1:]
    return m


TRAJ_COLORS = (7, 4, 2)


def _trajectory_color(marker_rc: tuple[int, int], cell_rc: tuple[int, int]) -> int:
    mr, mc = marker_rc
    r, c = cell_rc
    dist = abs(r - mr) + abs(c - mc)
    assert dist >= 1, "marker cell itself is not a trajectory cell"
    return TRAJ_COLORS[(dist - 1) % len(TRAJ_COLORS)]


def _boundary_mask(reach: np.ndarray, structural: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
    """Reachable background cells on the inner border of the marker-side region."""
    candidate = reach & bg_mask
    edge = np.zeros_like(candidate)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    near_struct = _dilate8(structural)
    near_unreach = _dilate4(~reach)
    return candidate & (edge | near_struct | near_unreach)


def _paint_trajectory(out: np.ndarray, boundary: np.ndarray, marker_rc: tuple[int, int]) -> np.ndarray:
    painted = out.copy()
    coords = np.argwhere(boundary)
    for r, c in coords:
        painted[int(r), int(c)] = _trajectory_color(marker_rc, (int(r), int(c)))
    return painted


def _trajectory_stats(boundary: np.ndarray, marker_rc: tuple[int, int]) -> tuple[Counter, int, int, int]:
    counts: Counter[int] = Counter()
    dmin = None
    dmax = None
    total = 0
    for r, c in np.argwhere(boundary):
        dist = abs(int(r) - marker_rc[0]) + abs(int(c) - marker_rc[1])
        counts[_trajectory_color(marker_rc, (int(r), int(c)))] += 1
        dmin = dist if dmin is None else min(dmin, dist)
        dmax = dist if dmax is None else max(dmax, dist)
        total += 1
    return counts, (dmin or 0), (dmax or 0), total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color the reachable inner border with distance-based trajectory bands from a corner marker.

    Input structure:
    - The background is the dominant color; blue pixels form barrier candidates and a single magenta corner pixel marks the start.
    - Border-touching blue components next to the marker side are structural walls; other blue fragments are decorations.
    - The input contains no trajectory colors yet; those are created only in the output.

    Rule:
    - Flood from the marker corner to find the background region reachable without crossing structural walls.
    - Remove blue decorations that sit on the hidden side of those structural walls.
    - On each reachable background cell lying on the region's inner border, paint a color determined by Manhattan distance from the marker: 7, then 4, then 2, repeating every three steps.

    Key insight:
    - The geometry is solved exactly like the base task; the new work is to map each boundary cell to a distance band instead of assigning one uniform fill color.
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
    marker_rc = (mr, mc)

    other_cols = [c for c in counts.keys() if c not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda c: counts[c])

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

    reach = _flood(
        marker_rc,
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) == wall_col and (r, c) not in walls),
        h,
        w,
    )

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and (r, c) not in walls and (r, c) not in reach:
                out[r, c] = bg

    reach_mask = np.zeros((h, w), dtype=bool)
    for r, c in reach:
        reach_mask[r, c] = True
    wall_mask = np.zeros((h, w), dtype=bool)
    for r, c in walls:
        wall_mask[r, c] = True

    boundary = _boundary_mask(reach_mask, wall_mask, out == bg)
    return _paint_trajectory(out, boundary, marker_rc)


# --------------- generator ---------------

# ---- Palette (task colors) ----
BG = 8
WALL = 1
MARKER = 6

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

# ---- Trajectory visibility gate ----
MIN_BOUNDARY_PIXELS = 12
MIN_DISTANCE_SPAN = 6
MIN_PER_TRAJ_COLOR = 2

# ---- Outer retry budget ----
GEN_ATTEMPTS = 500


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
    """Isoceles diamond attached to left edge."""
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
    """Right triangle on top edge, widest at top."""
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
    """Right triangle on right edge, widest at bottom."""
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


def _place_structural_walls(h, w, marker_rc, rng):
    """Place 1..NUM_STRUCTS_RANGE non-overlapping shapes; skip any that cover marker."""
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
    """Every 4-conn structural component must touch a border AND be 4-adj to reach."""
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
    """Build solver-style output with distance-colored trajectory cells."""
    out = np.full(structural.shape, BG, dtype=int)
    out[structural] = WALL
    out[marker_rc] = MARKER
    boundary = _boundary_mask(reach, structural, out == BG)
    return _paint_trajectory(out, boundary, marker_rc), boundary


def _place_decor(allowed_mask, count, rng):
    """Place `count` decoration patches in `allowed_mask` cells."""
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


def _trajectory_visible_enough(boundary: np.ndarray, marker_rc: tuple[int, int]) -> bool:
    counts, dmin, dmax, total = _trajectory_stats(boundary, marker_rc)
    if total < MIN_BOUNDARY_PIXELS:
        return False
    if dmax - dmin < MIN_DISTANCE_SPAN:
        return False
    return all(counts.get(col, 0) >= MIN_PER_TRAJ_COLOR for col in TRAJ_COLORS)


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

    out, boundary = _paint_output(structural, reach, marker_rc)
    if not _trajectory_visible_enough(boundary, marker_rc):
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
    for color in TRAJ_COLORS:
        inp[inp == color] = BG
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D14 trajectory variant")
