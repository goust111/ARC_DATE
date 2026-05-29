# Combo variant of 8f3a5a89 (index 8)
# combo_spec: D3/0+D5/3+D8/0
# Name: 任意角落标记+反向结构墙+8-连通洪填

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8

"""
Design analysis
===============

a) concretization_and_behavior_trace:
  original_behavior:
    - the marker is a singleton magenta corner pixel, effectively generated at BL
    - 4-connected flood starts from that corner
    - border-touching blue components adjacent to the first reach are treated as true walls
    - floating blue components are decoration and can be erased if hidden
  combo_behavior:
    - the marker may be any one of the four corners, so the solver must detect the start corner first
    - both floods switch to 8-connectivity, so a diagonal one-cell crack is traversable
    - wall semantics invert: every non-border-touching blue 4-component is a real obstacle; every border-touching
      blue component is decoration and becomes traversable in the second flood
  toy_trace:
    - build a floating vertical wall and a floating horizontal wall that miss each other by one diagonal step
    - under old 4-connected travel, the marker-side reach cannot cross that corner and the far side stays untouched
    - under this combo, the marker-side reach slips diagonally across the crack
    - add a bottom-edge blue bar near the marker: old logic would interpret it as structural because it touches the
      border, while the combo logic treats it as decoration and walks through it
    - keep a separate floating L-wall near another corner so the board still has a genuinely unreachable pocket,
      which produces orange inner-border cells and makes the obstacle role visible

b) complexity_calibration:
  - the generator uses one fixed semantic recipe with only a few numeric degrees of freedom: one diagonal leak,
    one sealed floating pocket, and 2-3 border decorations
  - this keeps the rule legible while still making all three dimensions visible in a small number of examples
  - the design avoids many independent object types; all walls remain blue and only their placement relative to the
    border changes their role

c) implicit_assumption_review:
  - marker detection can no longer assume BL, so main() scans all four corners for the unique singleton color
  - the old meaning of "wall" is invalid; border-touching blue components must not be blocked in the second flood
  - first-reach adjacency is no longer part of wall classification in the combo rule, so structural extraction is
    purely based on border contact of 4-connected wall components
  - 4-neighbor flood code is invalid under D8; both the initial and final reachability passes must use neighbors8
  - painting logic can still use edge / 8-near-wall / 4-near-unreachable tests, but it now reads from the new
    reachability mask and the inverted wall set
  - generator samples that do not produce a real diagonal-only gain or a real rule-inversion gain would collapse
    back toward the source task, so generation explicitly rejects those cases by comparing against baseline solvers

d) placement_legality:
  - all floating structural walls are kept at least one cell away from the border, ensuring they remain non-border
    components under 4-connectivity
  - border decorations are placed directly on edges with at least one-cell orthogonal gaps from structural walls,
    preventing accidental merges that would corrupt the intended classification
  - the final whole-grid symmetry transform preserves non-overlap, connectivity class, and corner legality

e) visual_observability:
  - the diagonal leak is visible because a substantial far-side region becomes orange only when diagonal travel is
    allowed
  - the border-decoration inversion is visible because some bold blue edge bars remain in place yet do not behave as
    blocking walls under the combo rule
  - examples vary the start corner by symmetry, so the solver cannot memorize one anchor corner

f) visual_cleanliness:
  - the layout stays sparse: one floating leaky barrier, one floating sealed pocket, and a few edge decorations
  - structural and decorative blue objects occupy distinct geometric regimes (floating vs border-touching), which
    reduces ambiguity
  - orange is only painted on reachable background cells, so the output still reads as a clean inner rim

g) inferability_of_change:
  - no rule depends on hidden color ordering; the only non-background colors remain magenta, blue, and orange
  - the distinction "floating blue blocks, edge blue does not" can be inferred visually from repeated examples
  - generator gating separately enforces a path-connectivity effect and a wall-classification effect, so the combo
    does not silently degenerate into a one-dimension variant

solver_generator_contract:
  - main() assumes exactly one non-background singleton corner marker and at least one other wall-colored pixel
  - main() also assumes the intended wall color is the dominant non-background, non-marker color
  - generate_input() guarantees both assumptions, guarantees at least one floating wall component, and guarantees
    at least one border-touching decoration component
"""


BG = 8
WALL = 1
MARKER = 6
FILL = 7

GRID_H_RANGE = (12, 20)
GRID_W_RANGE = (12, 20)
REACH_RATIO_RANGE = (0.45, 0.94)

MIN_DIAG_LEAK_AREA = 9
MIN_PATH_DIFF = 8
MIN_RULE_DIFF = 8
MIN_TOTAL_DIFF = 16

GEN_ATTEMPTS = 400
CORNER_NAMES = ("BL", "BR", "TL", "TR")


def _find_corner_marker(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    counts = Counter(int(v) for v in g.ravel())
    h, w = g.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    marker = None
    for r, c in corners:
        v = int(g[r, c])
        if v != bg and counts.get(v, 0) == 1:
            assert marker is None, "multiple corner singleton markers found"
            marker = (r, c, v)
    assert marker is not None, "no unique single-pixel corner marker found"
    return marker


def _flood(
    start: tuple[int, int],
    passable,
    h: int,
    w: int,
    connectivity: int,
) -> set[tuple[int, int]]:
    neigh = neighbors8 if connectivity == 8 else neighbors4
    q = deque([start])
    vis = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in neigh(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and passable(nr, nc):
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis


def _touches_border(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


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


def _classify_walls(
    g: np.ndarray,
    wall_col: int,
    reach0: set[tuple[int, int]],
    wall_rule: str,
) -> set[tuple[int, int]]:
    h, w = g.shape
    walls: set[tuple[int, int]] = set()
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = [(int(r), int(c)) for r, c in comp_arr]
        touches_border = _touches_border(comp, h, w)
        if wall_rule == "reverse":
            if not touches_border:
                walls |= set(comp)
            continue
        if not touches_border:
            continue
        adjacent_to_reach0 = any(any((nr, nc) in reach0 for nr, nc in neighbors4(r, c)) for r, c in comp)
        if adjacent_to_reach0:
            walls |= set(comp)
    return walls


def _solve(grid: np.ndarray, connectivity: int, wall_rule: str) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    mr, mc, marker_col = _find_corner_marker(g, bg)
    counts = Counter(int(v) for v in g.ravel())
    other_cols = [col for col in counts if col not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda col: counts[col])

    reach0 = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col),
        h,
        w,
        connectivity=connectivity,
    )

    walls = _classify_walls(g, wall_col, reach0, wall_rule)

    reach = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) == wall_col and (r, c) not in walls),
        h,
        w,
        connectivity=connectivity,
    )

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and (r, c) not in walls and (r, c) not in reach:
                out[r, c] = bg

    reach_mask = np.zeros((h, w), dtype=bool)
    wall_mask = np.zeros((h, w), dtype=bool)
    for r, c in reach:
        reach_mask[r, c] = True
    for r, c in walls:
        wall_mask[r, c] = True

    edge = np.zeros((h, w), dtype=bool)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True

    near_wall = _dilate8(wall_mask)
    near_unreach = _dilate4(~reach_mask)

    for r, c in reach:
        if int(out[r, c]) != bg:
            continue
        if edge[r, c] or near_wall[r, c] or near_unreach[r, c]:
            out[r, c] = FILL

    return {
        "out": out,
        "reach": reach_mask,
        "walls": wall_mask,
        "marker": (mr, mc),
        "wall_col": wall_col,
        "bg": bg,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the marker-side inner rim when the true blockers are the floating blue walls rather than the edge-touching ones.

    Input structure:
    - Azure is the background, blue is the wall-colored material, and a single magenta corner pixel marks the start.
    - The magenta marker may appear on any one of the four corners.
    - Blue components come in two geometric classes: some float fully inside the grid, while others touch the border.

    Rule:
    - Find the marked corner and flood from it with 8-neighbor movement, so diagonal cracks are traversable.
    - Reinterpret the blue components: only the non-border-touching 4-connected components are true walls; the
      border-touching blue components are decorations and may be walked through in the second flood.
    - Erase any unreachable decorative blue cells, then paint orange on each reachable background cell that lies on
      the grid edge, is 8-neighbor-adjacent to a true wall, or is 4-neighbor-adjacent to an unreachable cell.

    Key insight: Two original assumptions flip at once: the start corner is not fixed, and edge-touching blue is now
    the fake material. The real blockers are the floating blue components, while 8-connected travel lets the reach
    squeeze through diagonal one-cell leaks between them.
    """
    return _solve(grid, connectivity=8, wall_rule="reverse")["out"]


def _transform_grid(grid: np.ndarray, target_corner: str) -> np.ndarray:
    if target_corner == "BL":
        return grid.copy()
    if target_corner == "BR":
        return np.fliplr(grid).copy()
    if target_corner == "TL":
        return np.flipud(grid).copy()
    if target_corner == "TR":
        return np.rot90(grid, 2).copy()
    raise ValueError(f"unknown corner: {target_corner}")


def _build_base_input(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    marker_rc = (h - 1, 0)
    grid = np.full((h, w), BG, dtype=int)

    chamber_r0_low = max(2, h // 4)
    chamber_r0_high = max(chamber_r0_low, h // 3)
    chamber_c0_low = max(4, w // 3)
    chamber_c0_high = max(chamber_c0_low, w // 2)
    chamber_r0 = rng.randint(chamber_r0_low, chamber_r0_high)
    chamber_c0 = rng.randint(chamber_c0_low, chamber_c0_high)

    chamber_r1_low = max(chamber_r0 + 4, h // 2)
    chamber_c1_low = max(chamber_c0 + 4, w // 2)
    if chamber_r1_low > h - 3 or chamber_c1_low > w - 3:
        return None
    chamber_r1 = rng.randint(chamber_r1_low, h - 3)
    chamber_c1 = rng.randint(chamber_c1_low, w - 3)

    max_pocket_c = chamber_c0 - 3
    max_pocket_r = min(h - 5, chamber_r0 + 2)
    if max_pocket_c < 2 or max_pocket_r < 3:
        return None
    pocket_c = rng.randint(max(2, max_pocket_c - 2), max_pocket_c)
    pocket_r = rng.randint(3, max_pocket_r)

    structural = np.zeros((h, w), dtype=bool)

    # Sealed pocket that stays unreachable even under 8-connectivity.
    structural[1 : pocket_r + 1, pocket_c] = True
    structural[pocket_r, 1 : pocket_c + 1] = True

    # Large floating chamber whose right side is closed only by diagonal contacts:
    # 4-connectivity cannot enter, 8-connectivity can.
    structural[chamber_r0, chamber_c0:chamber_c1] = True
    structural[chamber_r1, chamber_c0:chamber_c1] = True
    structural[chamber_r0 : chamber_r1 + 1, chamber_c0] = True
    structural[chamber_r0 + 1 : chamber_r1, chamber_c1] = True

    if structural[marker_rc]:
        return None

    decor = np.zeros((h, w), dtype=bool)

    if chamber_c0 <= 3:
        return None
    decor[h - 1, 2 : chamber_c0 + 1] = True

    top_start = min(chamber_c0 + 2, w - 3)
    if top_start < w - 1:
        decor[0, top_start : w - 1] = True

    right_start = min(max(chamber_r0 + 2, 2), h - 3)
    if right_start < h - 1:
        decor[right_start : h - 1, w - 1] = True

    left_start = min(pocket_r + 2, h - 4)
    left_end = max(left_start + 1, h - 3)
    if left_start < left_end:
        decor[left_start:left_end, 0] = True

    if np.any(structural & decor):
        return None

    grid[structural | decor] = WALL
    grid[marker_rc] = MARKER
    return grid


def _valid_fill(mask: np.ndarray) -> bool:
    return int(np.count_nonzero(mask)) >= 12


def _gen_attempt() -> np.ndarray | None:
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)

    base = _build_base_input(h, w, rng)
    if base is None:
        return None

    target_corner = rng.choice(CORNER_NAMES)
    inp = _transform_grid(base, target_corner)

    new_info = _solve(inp, connectivity=8, wall_rule="reverse")
    out = new_info["out"]
    reach8 = new_info["reach"]
    walls_rev = new_info["walls"]

    if float(np.mean(reach8)) < REACH_RATIO_RANGE[0] or float(np.mean(reach8)) > REACH_RATIO_RANGE[1]:
        return None
    if int(np.count_nonzero(walls_rev)) == 0:
        return None
    if not _valid_fill(out == FILL):
        return None

    rev4_info = _solve(inp, connectivity=4, wall_rule="reverse")
    reach4_rev = rev4_info["reach"]
    path_diff = int(np.count_nonzero(out != rev4_info["out"]))
    leak_gain = int(np.count_nonzero(reach8 & ~reach4_rev))
    if leak_gain < MIN_DIAG_LEAK_AREA or path_diff < MIN_PATH_DIFF:
        return None

    orig8_info = _solve(inp, connectivity=8, wall_rule="original")
    rule_diff = int(np.count_nonzero(out != orig8_info["out"]))
    if rule_diff < MIN_RULE_DIFF:
        return None

    orig4_info = _solve(inp, connectivity=4, wall_rule="original")
    total_diff = int(np.count_nonzero(out != orig4_info["out"]))
    if total_diff < MIN_TOTAL_DIFF:
        return None

    wall_mask = inp == WALL
    border_wall = wall_mask.copy()
    border_wall[1:-1, 1:-1] = False
    if int(np.count_nonzero(border_wall)) == 0:
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 combo variant 8")
