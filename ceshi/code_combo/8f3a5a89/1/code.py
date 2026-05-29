# Combo variant of 8f3a5a89 (index 1)
# combo_spec: D1/1+D2/0
# Name: 重力水流+外边界涂色

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - We keep the same palette and the same "corner marker + blue walls" scene,
#      but the marker now acts like a water source under gravity.
#    - Small trace: put the marker at the top-left corner and place a blue strip
#      hanging down from the top edge near the middle. Ordinary 4-neighbor flood
#      would go down the left side, pass under the strip, then climb back up into
#      the upper-right chamber. Gravity flow cannot take that last upward step,
#      so the upper-right chamber stays unreachable.
#    - The second change flips the orange placement side. Instead of coloring the
#      reachable side of the frontier, we color unreachable background pixels
#      that lie on the outer side of the reach/unreach divider.
# b) Complexity calibration:
#    - If the marker stayed at bottom-left, "down + left/right only" would almost
#      collapse to a bottom-row crawl, so the generator deliberately places the
#      marker on a top corner and forces a suspended chamber above a pass-under.
#    - We keep one clear structural wall component plus a few small decorations.
#      That is enough variety for ARC while keeping the gravity effect legible.
# c) Implicit assumptions review:
#    - Unique corner marker detection still works.
#    - "Most frequent non-bg, non-marker color is wall color" still works.
#    - The old isotropic flood assumption is invalid: every reach computation in
#      the solver that defines the semantic region must now use directed gravity
#      moves (down/left/right, never up).
#    - Hidden-decoration erasure must still happen before the fill mask, because
#      erased cells belong to the unreachable background that may receive orange.
# d) Placement legality:
#    - Structural walls are border-touching and non-overlapping.
#    - Visible decorations are placed only on reachable background cells.
#    - Hidden decorations are placed only inside unreachable interior cells and
#      never touch the border, so the solver does not misread them as structure.
# e) Visual observability:
#    - The generator rejects cases where gravity reach equals ordinary reach.
#      Each accepted sample contains cells that are reachable only if upward
#      motion were allowed, making the fusion behavior visible from examples.
# f) Visual cleanliness:
#    - Orange is painted only on background cells.
#    - The unreachable chamber is kept reasonably large, so the outer ring reads
#      as a clean contour rather than random speckles.
# g) Inference of the change:
#    - The repeated pattern is "the marker starts high, water descends and can
#      spread sideways but cannot climb back up". The orange ring then appears on
#      the opposite side of that gravity-defined frontier.


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


def _mask_from_coords(coords: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
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


def _outer_fill_mask(bg_mask: np.ndarray, reach_mask: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    candidate = (~reach_mask) & bg_mask

    edge = np.zeros_like(candidate)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True

    near_wall = _dilate8(wall_mask)
    near_reach = _dilate4(reach_mask)
    return candidate & (edge | near_wall | near_reach)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate gravity-limited water spread from a corner marker and color the unreachable-side outline.

    Input structure:
    - The dominant color is background, blue pixels act as wall-colored objects, and a unique corner marker identifies the source.
    - Generated instances place the marker on a top corner so the downward flow has room to develop, though the solver still accepts any unique corner marker.
    - Border-touching blue components adjacent to the source-side flow are the true barriers; other blue fragments are decorative and should not define the partition.

    Rule:
    - Starting at the marker, spread only downward or sideways through background cells; upward motion is forbidden.
    - Re-identify structural walls as the border-touching wall-color components that touch the first gravity-reachable area, then allow non-structural wall-colored decorations to be traversed and erase the hidden ones that still remain unreachable.
    - Paint orange on unreachable background cells that lie on the grid edge, 8-neighbor a structural wall, or 4-neighbor the gravity-reachable region, yielding a 1-cell outer border on the unreachable side.

    Key insight: The solver must first recover a directed, gravity-shaped reachable set rather than an ordinary flood-fill region, and only then mirror the border coloring onto the unreachable side of that directed frontier.
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
    other_cols = [c for c in counts if c not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda c: counts[c])

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

    reach_mask = _mask_from_coords(reach, h, w)
    wall_mask = _mask_from_coords(walls, h, w)
    fill_mask = _outer_fill_mask(out == bg, reach_mask, wall_mask)
    out[fill_mask] = 7
    return out


BG = 8
WALL = 1
MARKER = 6
FILL = 7

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

REACH_RATIO_RANGE = (0.38, 0.87)
NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

GEN_ATTEMPTS = 400
MIN_FILL_CELLS = 6
MIN_GRAVITY_GAP = 6
MIN_BEHAVIOR_DIFF = 6


def _gravity_reach_mask(start, blocked, h, w):
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
    return mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()


def _build_structural_wall(h: int, w: int, rng) -> np.ndarray:
    structural = np.zeros((h, w), dtype=bool)
    thickness = rng.randint(1, min(3, max(1, w // 4)))
    max_c0 = w - thickness - 3
    min_c0 = max(2, w // 4)
    if min_c0 > max_c0:
        return structural
    c0 = rng.randint(min_c0, max_c0)
    depth = rng.randint(max(4, h // 2), h - 2)
    structural[:depth, c0 : c0 + thickness] = True

    max_ledge = min(4, w - c0 - thickness)
    if thickness == 1 and rng.random() < 0.45 and max_ledge >= 2:
        ledge_len = rng.randint(2, max_ledge)
        structural[0, c0 : c0 + ledge_len] = True

    return structural


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


def _paint_output(structural: np.ndarray, reach: np.ndarray, marker_rc: tuple[int, int]) -> np.ndarray:
    out = np.full(structural.shape, BG, dtype=int)
    out[structural] = WALL
    out[marker_rc] = MARKER
    fill = _outer_fill_mask(out == BG, reach, structural)
    out[fill] = FILL
    return out


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


def _gen_attempt():
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (0, 0)

    structural = _build_structural_wall(h, w, rng)
    if not structural.any() or structural[marker_rc]:
        return None

    gravity_reach = _gravity_reach_mask(marker_rc, structural, h, w)
    reach_frac = gravity_reach.mean()
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    if not _validate_structural(structural, gravity_reach, h, w):
        return None

    iso_reach = _flood_mask(marker_rc, structural, h, w)
    gravity_gap = iso_reach & (~gravity_reach) & (~structural)
    if int(gravity_gap.sum()) < MIN_GRAVITY_GAP:
        return None

    out = _paint_output(structural, gravity_reach, marker_rc)
    fill_mask = out == FILL
    if int(fill_mask.sum()) < MIN_FILL_CELLS:
        return None
    if len(components_mask(fill_mask, connectivity=4)) != 1:
        return None

    iso_out = _paint_output(structural, iso_reach, marker_rc)
    behavior_diff = np.count_nonzero((out == FILL) != (iso_out == FILL))
    if behavior_diff < MIN_BEHAVIOR_DIFF:
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = gravity_reach & (out == BG)
    hidden_allowed = (~gravity_reach) & (~structural)
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

    if rng.random() < 0.5:
        inp = np.fliplr(inp)
        out_final = np.fliplr(out_final)

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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 combo D1/1+D2/0")
