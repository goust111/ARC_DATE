# Combo variant of 8f3a5a89 (index 14)
# combo_spec: D1/1+D2/3
# Name: 重力水流+渐变厚度边界

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - We keep the same palette and the same "corner marker + blue walls +
#      blue decorations" language, but the semantic region is no longer found
#      by an undirected flood. The marker is now a water source that may move
#      only DOWN, LEFT, or RIGHT.
#    - A concrete 8x9 trace:
#        marker at top-left, one blue curtain hangs from the top near the
#        middle and stops two rows above the floor. Ordinary flood could go
#        down the left side, pass under the curtain, then climb upward into the
#        upper-right chamber. Gravity flow can still go down and pass under the
#        curtain, but it can never take the final upward climb, so that upper
#        chamber stays unreachable.
#    - Boundary painting still happens on the reachable side, but thickness is
#      now type-dependent:
#        * cells that touch the grid frame stay a 1-pixel orange strip;
#        * cells next to structural blue walls gain one extra inward layer,
#          producing a 2-pixel orange band beside those walls.
#    - Compared with the original task, two stages change together:
#        1. the semantic reach set becomes asymmetric because upward recovery is
#           impossible;
#        2. the wall-following part of the inner border becomes thicker than
#           the edge-following part.
# b) Complexity calibration:
#    - If we kept the original always-bottom-left marker, "down/left/right"
#      would almost collapse to a bottom-row crawl. That would make the gravity
#      rule both unnatural and visually weak. The generator therefore places
#      the marker on a top corner, while the solver still accepts any unique
#      corner marker.
#    - Thickness uses only two levels, 1 and 2. That is enough to be obvious
#      without creating many competing cases.
#    - The generator forces both an edge-following orange segment and a
#      wall-thickened orange segment, and it also forces a non-empty region
#      that isotropic flood would reach but gravity cannot.
# c) Implicit assumptions review:
#    - Unique corner marker detection still works unchanged.
#    - "Most frequent non-bg, non-marker color is the wall color" still works
#      because the palette remains the same.
#    - The original assumption that reachability is isotropic is invalid. Every
#      semantic reach computation must use directed gravity moves.
#    - The original 1-pass boundary predicate is also invalid. We now need a
#      base boundary layer plus an extra inward layer derived only from
#      wall-adjacent boundary seeds.
#    - Hidden-decoration erasure must still happen before building the fill
#      mask; otherwise unreachable decorations would incorrectly block or absorb
#      the orange band.
# d) Placement legality:
#    - Structural walls form one border-touching 4-connected curtain component.
#    - The curtain never covers the marker, always leaves a lower pass-under
#      corridor, and always leaves a right-side chamber wide enough to be seen.
#    - Visible decorations are placed only on gravity-reachable background
#      interior cells. Hidden decorations are placed only in unreachable
#      interior cells and never touch the border, so the solver cannot mistake
#      them for structure.
# e) Visual observability:
#    - Accepted samples must satisfy:
#        * isotropic_reach strictly contains gravity_reach;
#        * the combo fill differs from a thin-border baseline;
#        * orange exists on the grid edge and also in an extra wall-thickened
#          layer.
#      This makes both fused controls recoverable from 2-4 examples.
# f) Visual cleanliness:
#    - Only background cells become orange.
#    - Decorations are small and kept off the orange band, so the wall-thick vs
#      edge-thin contrast stays readable.
#    - The unreachable upper chamber is fairly large, so the gravity effect is
#      visible as a coherent missing area rather than as isolated pixels.
# g) New solver assumptions and generator guarantees:
#    - Solver assumption: structural walls can still be re-identified as the
#      border-touching wall-color components adjacent to the first directed
#      reach set. Generator guarantee: the curtain always touches the border and
#      is adjacent to source-side flow.
#    - Solver assumption: extra thickness is one additional 4-neighbor inward
#      layer from wall-adjacent boundary seeds. Generator guarantee: the wall is
#      long enough that this extra layer is present and visually repeated.
#    - Solver assumption: hidden decorations remain unreachable even when
#      decoration-colored wall pixels become traversable. Generator guarantee:
#      hidden decorations live in the suspended chamber that requires an upward
#      move to access.


def _gravity_reach(
    start: tuple[int, int], passable, h: int, w: int
) -> set[tuple[int, int]]:
    q = deque([start])
    vis = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in ((r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and passable(nr, nc):
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis


def _flood(
    start: tuple[int, int], passable, h: int, w: int
) -> set[tuple[int, int]]:
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


def _gravity_reach_mask(
    start: tuple[int, int], blocked: np.ndarray, h: int, w: int
) -> np.ndarray:
    reach = np.zeros((h, w), dtype=bool)
    reach[start] = True
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in ((r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not reach[nr, nc] and not blocked[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


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


def _touches_border(mask: np.ndarray) -> bool:
    return mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()


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
    Simulate gravity-limited water spread from a corner marker and paint a wall-aware inner border.

    Input structure:
    - The dominant color is background, blue pixels are wall-colored objects, and a unique single corner marker is the water source.
    - Some blue components are structural barriers touching the frame; other blue fragments are decorations that should not define the true partition.
    - Generated examples typically place the marker on a top corner so the downward-only flow creates a visible suspended chamber, though the solver accepts any unique corner marker.

    Rule:
    - Starting from the marker, propagate only downward or sideways through background cells; upward motion is illegal.
    - Re-identify structural walls as the border-touching wall-color components adjacent to the first gravity-reachable area, then allow non-structural blue decorations to be traversed and erase the unreachable hidden ones.
    - Paint orange on reachable background boundary cells, but with variable thickness: the boundary stays 1 pixel thick where it follows the grid edge or generic reach/unreach frontier, and becomes 2 pixels thick beside structural walls by adding one more inward layer.

    Key insight:
    - The orange mask cannot be derived from the original isotropic flood or from a single local boundary test. The solver must first recover the directed gravity reach set, then distinguish wall-following boundary segments from edge-following ones so only the wall-adjacent parts thicken.
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

    reach0 = _gravity_reach(
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

    reach = _gravity_reach(
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

REACH_RATIO_RANGE = (0.40, 0.84)
NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

GEN_ATTEMPTS = 500
MIN_GRAVITY_GAP = 6
MIN_EXTRA_FILL = 5
MIN_EDGE_FILL = 6
MIN_INTERIOR_BG = 6
MIN_GRAVITY_FILL_DIFF = 6


def _build_structural_wall(h: int, w: int, rng) -> np.ndarray | None:
    structural = np.zeros((h, w), dtype=bool)

    max_thickness = min(3, max(1, w // 4))
    thickness = rng.randint(1, max_thickness)

    min_c0 = max(2, w // 3)
    max_c0 = w - thickness - 3
    if min_c0 > max_c0:
        return None
    c0 = rng.randint(min_c0, max_c0)

    min_depth = max(4, h // 2)
    max_depth = h - 3
    if min_depth > max_depth:
        return None
    depth = rng.randint(min_depth, max_depth)

    structural[:depth, c0 : c0 + thickness] = True

    # Optional top ledge keeps the wall component irregular without changing
    # the core "hanging curtain with a lower pass-under" geometry.
    max_ledge = min(4, w - (c0 + thickness))
    if max_ledge >= 2 and rng.random() < 0.55:
        ledge_len = rng.randint(2, max_ledge)
        structural[0, c0 : c0 + thickness + ledge_len] = True

    return structural


def _validate_structural(structural: np.ndarray, reach: np.ndarray, h: int, w: int) -> bool:
    reach_nbr = _dilate4(reach)
    comps = components_mask(structural, connectivity=4)
    if len(comps) != 1:
        return False
    for comp_arr in comps:
        cmask = np.zeros((h, w), dtype=bool)
        cmask[comp_arr[:, 0], comp_arr[:, 1]] = True
        if not _touches_border(cmask):
            return False
        if not (cmask & reach_nbr).any():
            return False
    return True


def _paint_output(
    structural: np.ndarray,
    reach: np.ndarray,
    marker_rc: tuple[int, int],
    *,
    thick_walls: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    out = np.full(structural.shape, BG, dtype=int)
    out[structural] = WALL
    out[marker_rc] = MARKER

    candidate = reach & (out == BG)
    edge, near_wall, near_unreach, base, extra = _boundary_masks(reach, structural, candidate)
    fill = base | extra if thick_walls else base
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


def _place_decor(allowed_mask: np.ndarray, count: int, rng):
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


def _variant_metrics(
    structural: np.ndarray,
    gravity_reach: np.ndarray,
    iso_reach: np.ndarray,
    masks: dict[str, np.ndarray],
    thin_fill: np.ndarray,
    iso_fill: np.ndarray,
) -> dict[str, int]:
    gravity_gap = iso_reach & (~gravity_reach) & (~structural)
    fill = masks["fill"]
    edge = masks["edge"]
    extra = masks["extra"]
    candidate = masks["candidate"]
    return {
        "gravity_gap": int(gravity_gap.sum()),
        "extra_fill": int(extra.sum()),
        "edge_fill": int((fill & edge).sum()),
        "interior_bg": int((candidate & ~fill).sum()),
        "gravity_fill_diff": int(np.count_nonzero(fill != iso_fill)),
        "thickness_fill_diff": int(np.count_nonzero(fill != thin_fill)),
    }


def _gen_attempt():
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (0, 0)

    structural = _build_structural_wall(h, w, rng)
    if structural is None or structural[marker_rc]:
        return None

    gravity_reach = _gravity_reach_mask(marker_rc, structural, h, w)
    reach_frac = float(gravity_reach.mean())
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    if not _validate_structural(structural, gravity_reach, h, w):
        return None

    iso_reach = _flood_mask(marker_rc, structural, h, w)

    out, masks = _paint_output(structural, gravity_reach, marker_rc, thick_walls=True)
    thin_out, thin_masks = _paint_output(structural, gravity_reach, marker_rc, thick_walls=False)
    iso_out, iso_masks = _paint_output(structural, iso_reach, marker_rc, thick_walls=True)

    metrics = _variant_metrics(
        structural,
        gravity_reach,
        iso_reach,
        masks,
        thin_masks["fill"],
        iso_masks["fill"],
    )
    if metrics["gravity_gap"] < MIN_GRAVITY_GAP:
        return None
    if metrics["extra_fill"] < MIN_EXTRA_FILL:
        return None
    if metrics["edge_fill"] < MIN_EDGE_FILL:
        return None
    if metrics["interior_bg"] < MIN_INTERIOR_BG:
        return None
    if metrics["gravity_fill_diff"] < MIN_GRAVITY_FILL_DIFF:
        return None
    if metrics["thickness_fill_diff"] < MIN_EXTRA_FILL:
        return None

    fill_mask = masks["fill"]
    if not fill_mask.any() or len(components_mask(fill_mask, connectivity=4)) != 1:
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = gravity_reach & (out == BG)
    hidden_allowed = (~gravity_reach) & (~structural)
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

    if rng.random() < 0.5:
        inp = np.fliplr(inp)
        out_final = np.fliplr(out_final)

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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 combo D1/1+D2/3")
