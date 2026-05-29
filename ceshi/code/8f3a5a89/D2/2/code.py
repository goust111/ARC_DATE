# Variant of 8f3a5a89
# Dimension: D2 — 内边界涂色规则 (trajectory)
# Change: inner_border——可达区域中，位于网格边缘、8-邻接结构墙、或4-邻接不可达像素的背景格子涂橙色，形成1像素宽... → 间歇性边界（跳格涂色）

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for the D2 variant
# -----------------------------------------------------------------------------
# variant_spec:
#   source_task_calibration:
#     reference_task: 4a21e3da
#     takeaway: The reference task keeps the geometric detection logic intact and
#       only changes the final painted trace into an intermittent one. This
#       variant follows the same complexity profile: wall detection and
#       reachability stay unchanged; only the boundary painting stage becomes
#       dashed.
#   concretization:
#     original_boundary:
#       Reachable background cells are eligible if they are on the grid edge,
#       8-neighbor a structural wall, or 4-neighbor an unreachable cell.
#       In the base task every eligible cell becomes orange, so the inner border
#       is a continuous 1-pixel ring.
#     new_boundary:
#       Compute the exact same eligible boundary ring first, then only keep the
#       subset whose checkerboard parity matches the marker pixel parity. Because
#       the underlying ring is 4-connected, adjacent ring cells alternate parity,
#       so the visible result is a dashed every-other-cell border.
#     trace_example:
#       Suppose a straight boundary run contains cells
#       [(9,0), (8,0), (7,0), (6,0), (5,0)] and the marker is at (9,0).
#       Base task output paints all five cells.
#       Variant output paints only cells with the same parity as (9,0):
#       (9,0), (7,0), (5,0), leaving (8,0) and (6,0) as background.
#   complexity_calibration:
#     solver_side:
#       No new object type is introduced; the only added reasoning is parity on a
#       pre-existing boundary set. This is visible but not combinatorially large.
#     generator_side:
#       Samples must not degenerate into the original rule. Therefore generation
#       explicitly checks that the continuous candidate boundary exists, that both
#       parities are present on it, and that several candidate cells are skipped.
#   implicit_assumption_audit:
#     - Original code assumed "if cell is a boundary candidate, paint it". That
#       assumption fails in the variant; painting must be gated by parity.
#     - Visible decorations were previously allowed anywhere output stayed bg.
#       After sparsifying the border, skipped boundary slots would also be bg, so
#       the generator must forbid visible decorations on the full candidate ring
#       or they would visually mask the new dashed rule.
#     - The original generator required the orange ring itself to be 4-connected.
#       That is no longer valid; instead the continuous pre-dash candidate ring
#       must be 4-connected.
#   placement_legality:
#     - Structural walls still cannot overlap each other or the corner marker.
#     - Hidden decorations still cannot touch the border, otherwise the solver
#       would misclassify them as structural.
#     - Visible decorations are restricted to reachable non-boundary background
#       cells so the dashed ring remains legible.
#   observability:
#     - The dashed border is human-visible because the skipped cells lie exactly
#       between painted boundary cells along the same ring.
#     - Parity is anchored to the visible marker, not to an invisible color id or
#       arbitrary enumeration, so the pattern is stable across examples.
#   cleanliness:
#     - The output still has only four semantic layers: bg, structural/decoration
#       walls, marker, and orange border. The only visual difference is the
#       orange border becoming sparse.
#   new_solver_assumptions_and_generator_guarantees:
#     - main() now assumes the intended task uses marker-aligned checkerboard
#       parity on the candidate ring.
#     - generate_input() guarantees at least four painted boundary cells and at
#       least four skipped candidate cells, so this parity rule is observable.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a dashed inner border around the region reachable from a corner marker.

    Input structure:
    - The dominant color is background; blue components include true border-touching walls and small wall-colored decorations.
    - A unique magenta pixel sits on one corner and marks the flood-fill start.
    - Structural walls and the grid edges define the reachable chamber, while decoration walls may sit either inside it or behind it.

    Rule:
    - Flood-fill from the marker, treating only border-touching blue components adjacent to the marker side as blocking walls.
    - Erase blue decorations stranded on the unreachable side.
    - Detect the same reachable inner-boundary cells as in the base task, then paint only every other one in marker-aligned checkerboard parity to make a dashed orange loop.

    Key insight: The difficult step is still separating structural walls from decorative blue fragments; once the inner-boundary candidate ring is known, the variant keeps just one parity class of that ring.
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
    border_mask = _boundary_candidates(reach_mask, walls_mask, out == bg)

    fill_col = 7
    out[_checkerboard_subset(border_mask, (mr, mc))] = fill_col
    return out


# --------------- generator (variant-specific output-first) ---------------

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

MIN_BORDER_CANDIDATES = 8
MIN_DASHED_FILL = 4
MIN_SKIPPED_CANDIDATES = 4

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


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


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

    border_mask = _boundary_candidates(reach, structural, out == BG)
    fill_mask = _checkerboard_subset(border_mask, marker_rc)

    out[fill_mask] = FILL
    return out, border_mask, fill_mask


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

    out, border_mask, fill_mask = _paint_output(structural, reach, marker_rc)

    if not border_mask.any():
        return None
    if len(components_mask(border_mask, connectivity=4)) != 1:
        return None

    border_count = int(border_mask.sum())
    fill_count = int(fill_mask.sum())
    skipped_count = border_count - fill_count
    if border_count < MIN_BORDER_CANDIDATES:
        return None
    if fill_count < MIN_DASHED_FILL or skipped_count < MIN_SKIPPED_CANDIDATES:
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = reach & (out == BG) & (~border_mask)
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89")
