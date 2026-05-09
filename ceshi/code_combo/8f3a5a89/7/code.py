# Combo variant of 8f3a5a89 (index 7)
# combo_spec: D2/2+D13/2
# Name: 双层边界+浮动内环

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for the combo variant
# -----------------------------------------------------------------------------
# combo_design:
#   spec_priority:
#     - The authoritative combo text is the fusion description: output must show
#       a green outer ring, a one-cell bg gap, and an orange floating inner ring.
#     - D13/2 contributes the distance logic: the orange band is not the
#       immediate reachable-side border but the layer two 4-steps inward from it.
#     - D2/2 still has to remain visible in the combo, so the orange floating
#       ring is not continuous: only one checkerboard parity of the distance-2
#       ring is painted, creating a dashed inner loop.
#   a_concretization_and_trace:
#     original_task:
#       - Classify blue components into structural walls vs decorations.
#       - Remove unreachable decorations.
#       - Paint the immediate reachable-side inner boundary orange.
#     combo_behavior:
#       - Keep the same wall classification and decoration cleanup.
#       - Paint green on unreachable background cells that sit just outside the
#         structural walls, making the outer side of the divider explicit.
#       - Compute the old immediate reachable-side boundary layer only as a
#         distance seed; distance 1 stays bg as a moat.
#       - Distance 2 inside the reachable chamber forms the orange candidate
#         ring, and then D2/2 keeps only one checkerboard parity of that ring.
#     concrete_trace:
#       - Suppose a vertical structural wall separates the marker side from an
#         unreachable pocket on the right.
#       - The first unreachable bg column right next to the wall turns green.
#       - On the reachable side, the first inward column stays bg and the second
#         inward column becomes an orange checkerboard trace.
#   b_complexity_calibration:
#     - Wall detection remains the main recognition problem; the new reasoning is
#       only about which side gets which color and which distance-2 cells survive.
#     - The generator enforces non-empty green, gap, painted-orange, and skipped
#       orange-candidate cells so the combo cannot collapse to base, pure D13/2,
#       or pure D2/2.
#     - Visible decorations are allowed only deeper than the orange ring so they
#       cannot hide the three-layer pattern.
#   c_implicit_assumption_audit:
#     - Original solver assumption: every detected immediate reachable boundary
#       cell should be colored. Broken here; that layer becomes only a hidden
#       distance seed, while green moves to the unreachable side.
#     - D13/2 assumption: distance-1 from the immediate boundary should be
#       orange. Broken here; distance 1 must stay bg and distance 2 becomes the
#       orange candidate ring.
#     - D2/2 assumption: every eligible ring cell is painted. Broken here; only
#       one checkerboard parity of the distance-2 ring is painted.
#     - Original generator assumption: any remaining reachable bg can host visible
#       decor. Broken here; decor on the green ring, the moat, or the orange ring
#       would erase the visual cue, so visible decor is restricted to dist >= 3.
#     - Original validation assumption: the visible orange ring is continuous.
#       Broken here; generator must verify both painted and skipped distance-2
#       cells explicitly.
#   d_placement_legality:
#     - Structural walls remain non-overlapping border-touching 4-components and
#       cannot cover the bottom-left marker.
#     - Hidden decorations remain off-border so the solver never upgrades them to
#       structural walls.
#     - Visible decorations can only occupy reachable bg cells with inward
#       distance at least 3 from the immediate boundary frontier.
#   e_visual_observability:
#     - Human-solvable evidence comes from the repeated motif:
#       green outer boundary, blank moat, orange dashed inner ring.
#     - The generator requires each visible role to contain multiple cells, and
#       the dashed ring to omit multiple alternating candidates, so the rule is
#       inferable from 2-3 examples.
#   f_visual_cleanliness:
#     - The full distance-2 candidate ring is kept 4-connected even though the
#       painted orange subset is intentionally intermittent.
#     - Decorations are sparse and pushed deeper inside, so they do not compete
#       with the boundary coding.
#   g_color_inference:
#     - Green marks the wall's unreachable side.
#     - Orange keeps its interior-border semantics but is now both offset inward
#       by two 4-neighbor layers and thinned into a checkerboard trace.
#   new_solver_assumptions_and_generator_guarantees:
#     - main() assumes the intended input admits a non-empty unreachable-side
#       green layer and a non-empty distance-2 interior ring with both parities;
#       generate_input() enforces both.
#     - main() assumes reachable visible decorations do not overwrite the
#       outer/moat/orange layers; generate_input() enforces dist >= 3.
# -----------------------------------------------------------------------------


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


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _boundary_frontier(reach_mask: np.ndarray, walls_mask: np.ndarray, bg_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidate = reach_mask & bg_mask

    edge = np.zeros_like(candidate)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True

    near_struct = _dilate8(walls_mask)
    near_unreach = _dilate4(~reach_mask)
    frontier = candidate & (edge | near_struct | near_unreach)
    return frontier, candidate


def _outer_boundary_mask(reach_mask: np.ndarray, walls_mask: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
    return (~reach_mask) & bg_mask & _dilate8(walls_mask)


def _checkerboard_subset(mask: np.ndarray, anchor: tuple[int, int]) -> np.ndarray:
    rr, cc = np.indices(mask.shape)
    phase = (int(anchor[0]) + int(anchor[1])) & 1
    return mask & (((rr + cc) & 1) == phase)


def _distance_layers(candidate: np.ndarray, frontier: np.ndarray) -> np.ndarray:
    dist = np.full(candidate.shape, -1, dtype=int)
    q = deque()
    for r, c in np.argwhere(frontier):
        r = int(r)
        c = int(c)
        dist[r, c] = 0
        q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < candidate.shape[0] and 0 <= nc < candidate.shape[1]:
                if candidate[nr, nc] and dist[nr, nc] == -1:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
    return dist


def _compute_combo_layers(
    out: np.ndarray,
    reach: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    bg: int,
    anchor: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = out.shape
    reach_mask = _coords_mask(reach, h, w)
    walls_mask = _coords_mask(walls, h, w)
    bg_mask = out == bg
    frontier, candidate = _boundary_frontier(reach_mask, walls_mask, bg_mask)
    dist = _distance_layers(candidate, frontier)
    outer = _outer_boundary_mask(reach_mask, walls_mask, bg_mask)
    gap = candidate & (dist == 1)
    inner_candidates = candidate & (dist == 2)
    inner = _checkerboard_subset(inner_candidates, anchor)
    skipped = inner_candidates & ~inner
    return outer, gap, inner_candidates, inner, skipped, dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint a dual-side boundary with a dashed floating inner ring.

    Input structure:
    - The dominant color is background; blue pixels mix true border-touching walls with smaller decorative blue fragments.
    - A unique magenta single pixel sits on one corner and marks the flood-fill start.
    - Border-touching blue components next to the marker-side open area are structural walls; other blue pieces are decorations that may lie inside the chamber or behind it.

    Rule:
    - Flood from the marker to separate structural walls from decorative blue fragments, then erase decorative blue cells stranded on the unreachable side.
    - Paint green on unreachable background cells that sit just outside the structural walls.
    - Inside reachable background cells, reconstruct the old immediate inner-boundary layer only as a distance seed, leave distance 1 untouched as background, and keep only one checkerboard parity of the distance-2 layer in orange.

    Key insight:
    - The hard part is still wall classification; once the true divider is known, the combo exposes the unreachable side in green and a dashed distance-2 reachable-side ring in orange.
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

    outer_col = 3
    inner_col = 7
    outer, _, _, inner, _, _ = _compute_combo_layers(out, reach, walls, bg, (mr, mc))
    out[outer] = outer_col
    out[inner] = inner_col
    return out


BG = 8
WALL = 1
MARKER = 6
OUTER = 3
INNER = 7

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

MIN_OUTER_CELLS = 8
MIN_GAP_CELLS = 6
MIN_INNER_CANDIDATE_CELLS = 8
MIN_INNER_CELLS = 4
MIN_SKIPPED_INNER_CELLS = 4
MIN_DEEPER_INTERIOR_CELLS = 3

GEN_ATTEMPTS = 900


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

    reach_set = {(int(r), int(c)) for r, c in np.argwhere(reach)}
    wall_set = {(int(r), int(c)) for r, c in np.argwhere(structural)}
    outer, gap, inner_candidates, inner, skipped, dist = _compute_combo_layers(out, reach_set, wall_set, BG, marker_rc)
    out[outer] = OUTER
    out[inner] = INNER
    return out, outer, gap, inner_candidates, inner, skipped, dist


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

    out, outer, gap, inner_candidates, inner, skipped, dist = _paint_output(structural, reach, marker_rc)

    if int(outer.sum()) < MIN_OUTER_CELLS:
        return None
    if int(gap.sum()) < MIN_GAP_CELLS:
        return None
    if int(inner_candidates.sum()) < MIN_INNER_CANDIDATE_CELLS:
        return None
    if int(inner.sum()) < MIN_INNER_CELLS:
        return None
    if int(skipped.sum()) < MIN_SKIPPED_INNER_CELLS:
        return None
    if int((dist >= 3).sum()) < MIN_DEEPER_INTERIOR_CELLS:
        return None
    if len(components_mask(inner_candidates, connectivity=4)) != 1:
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = reach & (dist >= 3) & (out == BG)
    hidden_allowed = (~reach) & (~structural) & (~outer)
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
    inp[inp == OUTER] = BG
    inp[inp == INNER] = BG
    inp[hidden_placed] = WALL

    solved = main(inp)
    if not np.array_equal(solved, out_final):
        return None

    if np.array_equal(outer, inner):
        return None
    if np.array_equal(gap, inner):
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 combo variant")
