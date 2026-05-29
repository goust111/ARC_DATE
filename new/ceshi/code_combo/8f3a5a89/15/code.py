# Combo variant of 8f3a5a89 (index 15)
# combo_spec: D7/1+D18/1
# Name: L形墙壁+横跨对边

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for the fused variant
#
# a) Concrete behavior trace:
#    Original task allowed several unrelated border-touching wall shapes. Here we
#    force one structural wall to be a bent left-to-right barrier:
#      - example: row 3 from col 0..6, then a vertical connector at col 6 down to
#        row 7, then row 7 from col 6..W-1.
#      - the bottom-left marker is below this wall, so flood-fill can only occupy
#        the lower side of the bent barrier.
#      - the orange output border therefore follows the grid edges on the marker
#        side and also hugs the wall's inner corner near the bend.
#    Compared with the original task, the structural-wall test is no longer
#    "touches some border"; it is "touches opposite edges and actually bends".
#
# b) Complexity calibration:
#    Only one structural wall is used. This keeps the rule legible while still
#    making the visual change obvious: every sample shows a cross-grid bent wall
#    with a corner, never a plain straight divider. Decorations stay small.
#
# c) Hidden assumptions audited from the original solver:
#    - "border-touching wall component == structural" is too weak now, so main()
#      upgrades it to "opposite-edge touching + nonzero span in both axes".
#    - "all wall components may block the first flood" still holds.
#    - "hidden decorations can be removed after the second flood" still holds, but
#      the generator must keep decorations disconnected from the structural wall.
#
# d) Placement legality:
#    The generator builds the bent wall from three orthogonal runs that are always
#    inside bounds, 4-connected, and never cover the marker. Decorations are
#    placed by random_free_location on masks that exclude borders and all existing
#    wall pixels, preventing overlap and accidental merging.
#
# e/f/g) Observability and cleanliness:
#    The core observable metric is bend_depth = |left_edge_row - right_edge_row|.
#    The generator enforces bend_depth >= 2, so the wall never degenerates into a
#    straight cross-grid line. The turn column is kept away from the side edges,
#    which produces a visible orange corner and leaves enough plain background on
#    the marker side for humans to infer the rule from a few examples. At least
#    one hidden decoration is also forced so the erase-vs-keep distinction for
#    non-structural wall pixels is visible in every training sample.
# -----------------------------------------------------------------------------


def _flood(start: tuple[int, int], passable, h: int, w: int) -> set[tuple[int, int]]:
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and passable(nr, nc):
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _flood_mask(start: tuple[int, int], blocked: np.ndarray) -> np.ndarray:
    h, w = blocked.shape
    reach = np.zeros((h, w), dtype=bool)
    reach[start] = True
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not reach[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


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


def _is_spanning_bent_component(comp: set[tuple[int, int]], h: int, w: int) -> bool:
    rows = {r for r, _ in comp}
    cols = {c for _, c in comp}
    touches_lr = any(c == 0 for _, c in comp) and any(c == w - 1 for _, c in comp)
    touches_tb = any(r == 0 for r, _ in comp) and any(r == h - 1 for r, _ in comp)
    bends = len(rows) > 1 and len(cols) > 1
    return bends and (touches_lr or touches_tb)


def _find_structural_walls(g: np.ndarray, wall_col: int, reach0: set[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = g.shape
    walls: set[tuple[int, int]] = set()
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = {(int(r), int(c)) for r, c in comp_arr}
        if not _is_spanning_bent_component(comp, h, w):
            continue
        if not any(any((nr, nc) in reach0 for nr, nc in neighbors4(r, c)) for r, c in comp):
            continue
        walls |= comp
    return walls


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the marker-side inner border against a bent wall that spans opposite edges.

    Input structure:
    - Background is the most frequent color, with one unique magenta marker pixel on
      a corner (in generated examples it is bottom-left).
    - Blue pixels contain one structural barrier shaped as a bent left-to-right wall:
      two horizontal arms at different heights joined by a vertical connector.
    - Any other blue fragments are decorations; some lie on the marker side and stay,
      others are trapped behind the spanning wall and must be erased.

    Rule:
    - Flood-fill from the marker through background cells, initially treating every
      blue pixel as blocked.
    - Re-identify the real structural wall as the blue component that both spans a
      pair of opposite edges and bends in two axes, then flood again while allowing
      travel through the remaining blue decoration pixels.
    - Remove unreachable decoration pixels, and color orange on every reachable
      background cell that sits on the reachable region's inner boundary.

    Key insight:
    The decisive signal is not just "touches a border" but "forms a bent separator
    across opposite edges". Once that component is isolated, the rest of the task is
    the same marker-side region tracing as in the original puzzle.
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
    assert other_cols, "no wall-colored pixels found"
    wall_col = max(other_cols, key=lambda c: counts[c])

    reach0 = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col),
        h,
        w,
    )
    walls = _find_structural_walls(g, wall_col, reach0)
    assert walls, "no spanning bent wall found"

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


BG = 8
WALL = 1
MARKER = 6
FILL = 7

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)
REACH_RATIO_RANGE = (0.38, 0.87)
NUM_DECOR_RANGE = (1, 3)
GEN_ATTEMPTS = 500
MIN_BEND_DEPTH = 2

DECOR_PATCHES = [
    np.array([[1]], dtype=int),
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
]


def _build_spanning_bent_wall(h: int, w: int, rng: random.Random) -> tuple[np.ndarray, dict] | None:
    if h < 6 or w < 7:
        return None

    row_choices = list(range(1, h - 1))
    rng.shuffle(row_choices)
    for r_left in row_choices:
        candidates = [r for r in range(1, h - 1) if abs(r - r_left) >= MIN_BEND_DEPTH]
        rng.shuffle(candidates)
        for r_right in candidates:
            c_turn = rng.randint(2, w - 3)
            mask = np.zeros((h, w), dtype=bool)
            mask[r_left, 0 : c_turn + 1] = True
            r0, r1 = sorted((r_left, r_right))
            mask[r0 : r1 + 1, c_turn] = True
            mask[r_right, c_turn:w] = True
            info = {
                "left_row": int(r_left),
                "right_row": int(r_right),
                "turn_col": int(c_turn),
                "bend_depth": int(abs(r_left - r_right)),
            }
            return mask, info
    return None


def _paint_output(structural: np.ndarray, reach: np.ndarray, marker_rc: tuple[int, int]) -> np.ndarray:
    h, w = structural.shape
    out = np.full((h, w), BG, dtype=int)
    out[structural] = WALL
    out[marker_rc] = MARKER

    candidate = reach & (out == BG)
    edge = np.zeros((h, w), dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True

    near_struct = _dilate8(structural)
    near_unreach = _dilate4(~reach)
    fill = candidate & (edge | near_struct | near_unreach)
    out[fill] = FILL
    return out


def _place_decorations(
    allowed_mask: np.ndarray,
    occupied_mask: np.ndarray,
    count: int,
    rng: random.Random,
) -> np.ndarray | None:
    h, w = allowed_mask.shape
    placed = np.zeros((h, w), dtype=bool)
    if count == 0:
        return placed

    for _ in range(count):
        order = list(range(len(DECOR_PATCHES)))
        rng.shuffle(order)
        placed_one = False
        for idx in order:
            patch = DECOR_PATCHES[idx]
            canvas = np.where(allowed_mask & ~occupied_mask & ~placed, 0, 9)
            try:
                r0, c0 = random_free_location(
                    canvas,
                    patch,
                    transparent=0,
                    border=1,
                    padding=1,
                    padding_connectivity=8,
                )
            except ValueError:
                continue
            patch_mask = patch.astype(bool)
            placed[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] |= patch_mask
            placed_one = True
            break
        if not placed_one:
            return None
    return placed


def _interior_reachable_bg_count(out: np.ndarray, reach: np.ndarray) -> int:
    return int(((out == BG) & reach).sum())


def _gen_attempt() -> np.ndarray | None:
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    built = _build_spanning_bent_wall(h, w, rng)
    if built is None:
        return None
    structural, info = built
    if structural[marker_rc]:
        return None

    reach = _flood_mask(marker_rc, structural)
    reach_ratio = float(reach.mean())
    if not (REACH_RATIO_RANGE[0] <= reach_ratio <= REACH_RATIO_RANGE[1]):
        return None

    if info["bend_depth"] < MIN_BEND_DEPTH:
        return None

    out = _paint_output(structural, reach, marker_rc)
    if _interior_reachable_bg_count(out, reach) < max(3, w // 5):
        return None

    fill_mask = out == FILL
    if not fill_mask.any() or len(components_mask(fill_mask, connectivity=4)) != 1:
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_hidden = rng.randint(1, n_decor)
    n_visible = n_decor - n_hidden

    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True

    visible_allowed = reach & (out == BG) & ~edge_mask
    hidden_allowed = (~reach) & (~structural) & ~edge_mask

    visible_placed = _place_decorations(visible_allowed, structural, n_visible, rng)
    if visible_placed is None:
        return None
    hidden_placed = _place_decorations(hidden_allowed, structural | visible_placed, n_hidden, rng)
    if hidden_placed is None:
        return None

    out_final = out.copy()
    out_final[visible_placed] = WALL

    inp = out_final.copy()
    inp[inp == FILL] = BG
    inp[hidden_placed] = WALL

    if not np.array_equal(main(inp), out_final):
        return None

    walls = _find_structural_walls(inp, WALL, _flood(marker_rc, lambda r, c: int(inp[r, c]) in (BG, MARKER), h, w))
    if not walls:
        return None
    wall_mask = np.zeros((h, w), dtype=bool)
    for r, c in walls:
        wall_mask[r, c] = True
    left_rows = np.where(wall_mask[:, 0])[0]
    right_rows = np.where(wall_mask[:, -1])[0]
    if left_rows.size == 0 or right_rows.size == 0:
        return None
    if abs(int(left_rows[0]) - int(right_rows[0])) < MIN_BEND_DEPTH:
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 combo D7/1+D18/1")
