# Variant of 8f3a5a89
# Dimension: D1 — 路径/区域发现机制 (path)
# Change: boundary_trace + flood_fill——从标记像素出发做4-连通洪填，找到被结构墙阻挡的可达区域，再在... → 通道宽度约束洪填

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the D1 variant
#
# a) Concretization / behavior trace
#    - The original task treats the marker as a point and floods single cells.
#    - This variant treats reachability as "where a 2x2 probe can move" while
#      still painting cells, not blocks. A region behind a 1-cell slit remains
#      visible in the input but becomes unreachable in the output logic.
#    - Example: a full-height vertical blue barrier at column c with a 1-cell
#      gap. Plain flood crosses the gap and reaches the right chamber. The new
#      reach test cannot move any 2x2 placement through that gap, so the right
#      chamber becomes unreachable. The orange border therefore stays on the
#      marker side, and blue decorations placed in that chamber are erased.
#
# b) Complexity calibration
#    - The visual change is forced by generator construction: every sample has a
#      large chamber that is plain-reachable but width-blocked.
#    - Layout variety is kept small but not trivial: vertical vs horizontal
#      chokepoints, different slit positions, optional extra reachable-side
#      structural shapes, and visible/hidden decorations.
#
# c) Implicit-assumption review of the original main()
#    - Original `_flood(...)` assumes any 4-connected bg path is traversable.
#      That is the exact D1 assumption we must replace.
#    - Structural-wall detection depended on the first reach set. If that set
#      shrinks under width constraints, wall identification must shrink with it
#      as well; otherwise the solver would still reason using plain reachability.
#    - Decoration removal and border painting depend on the final reach set only,
#      so they remain valid once the reach computation is swapped out.
#
# d) Placement legality
#    - Structural walls remain 4-connected blue components touching the border.
#    - The marker corner always has at least one 2x2 passable seed block.
#    - Hidden decorations are placed strictly away from the border so they are
#      not reclassified as structural by the solver.
#
# e) Visual observability
#    - The blocked-vs-open distinction is visible from geometry alone: samples
#      contain a one-pixel slit leading to a larger chamber. Humans can compare
#      examples and infer that some slits are too narrow to count as passable.
#
# f) Visual cleanliness
#    - The generator keeps only a few object layers: structural walls, optional
#      small decorations, a single marker, and the orange border in the output.
#      Hidden decorations live in the width-blocked chamber to avoid clutter.
#
# g) Color inference
#    - No new color mapping is introduced. Blue still means wall, magenta marks
#      the start, orange marks the reachable inner boundary, and azure is bg.
#
# New solver assumptions introduced by this variant and enforced by generator:
#    - Width reach is implemented as BFS over feasible 2x2 placements; the
#      generator guarantees the marker corner participates in such a seed.
#    - At least one large region is plain-reachable but not width-reachable.
#    - Every border-touching wall component is adjacent to the width-reachable
#      side, matching the solver's structural-wall rediscovery logic.
# ---------------------------------------------------------------------------


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


def _flood_mask(start: tuple[int, int], blocked: np.ndarray, h: int, w: int) -> np.ndarray:
    """Plain single-cell flood used only for generator diagnostics/placement."""
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


def _width_reach_mask(start: tuple[int, int], passable: np.ndarray) -> np.ndarray:
    """
    Reachability under a width>=2 constraint, modeled as movement of a 2x2 probe.

    A cell counts as reachable iff it belongs to at least one reachable 2x2
    passable placement. If the start cell has no feasible 2x2 seed, only the
    start itself is marked reachable.
    """
    h, w = passable.shape
    sr, sc = start
    out = np.zeros((h, w), dtype=bool)
    out[sr, sc] = True
    if not passable[sr, sc] or h < 2 or w < 2:
        return out

    feasible = (
        passable[:-1, :-1]
        & passable[1:, :-1]
        & passable[:-1, 1:]
        & passable[1:, 1:]
    )
    ph, pw = feasible.shape

    seeds: list[tuple[int, int]] = []
    for r0 in (sr - 1, sr):
        for c0 in (sc - 1, sc):
            if 0 <= r0 < ph and 0 <= c0 < pw and feasible[r0, c0]:
                seeds.append((r0, c0))
    if not seeds:
        return out

    place_reach = np.zeros((ph, pw), dtype=bool)
    q = deque()
    for seed in sorted(set(seeds)):
        place_reach[seed] = True
        q.append(seed)

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < ph and 0 <= nc < pw and feasible[nr, nc] and not place_reach[nr, nc]:
                place_reach[nr, nc] = True
                q.append((nr, nc))

    for r0, c0 in np.argwhere(place_reach):
        out[r0 : r0 + 2, c0 : c0 + 2] = True
    return out


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint an inner border around the region reachable from a corner marker pixel.

    Input structure:
    - bg is the dominant color, blue components divide space, and one magenta
      corner pixel marks the starting side.
    - Border-touching blue components act as candidate structural walls; small
      interior blue pieces are decorations and may be removed.
    - Some openings look traversable to a normal flood but are only one cell
      wide, so they fail the width test.

    Rule:
    - Starting from the marker corner, search only through positions available
      to a 2x2 passable probe, so 1-cell-wide slits are blocked.
    - Re-identify structural wall components from that width-constrained reach,
      then allow non-structural blue decorations to be traversed/kept exactly as
      in the original task.
    - Erase unreachable decorations and paint orange on reachable bg cells that
      lie on the inner boundary of the width-reachable region.

    Key insight: Reachability is no longer about single-cell connectivity; a
    chamber counts as reachable only if a width-2 passage connects it to the
    marker side.
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

    passable0 = (g == bg) | (g == marker_col)
    reach0_mask = _width_reach_mask((mr, mc), passable0)

    walls_mask = np.zeros((h, w), dtype=bool)
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = [(int(r), int(c)) for r, c in comp_arr]
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if not touches_border:
            continue
        adjacent_to_reach0 = any(
            0 <= nr < h and 0 <= nc < w and reach0_mask[nr, nc]
            for r, c in comp
            for nr, nc in neighbors4(r, c)
        )
        if adjacent_to_reach0:
            for r, c in comp:
                walls_mask[r, c] = True

    passable = (g == bg) | (g == marker_col) | ((g == wall_col) & ~walls_mask)
    reach_mask = _width_reach_mask((mr, mc), passable)

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and not walls_mask[r, c] and not reach_mask[r, c]:
                out[r, c] = bg

    fill_col = 7
    for r, c in np.argwhere(reach_mask):
        if int(out[r, c]) != bg:
            continue
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            out[r, c] = fill_col
            continue
        if any(
            0 <= nr < h and 0 <= nc < w and walls_mask[nr, nc]
            for nr, nc in neighbors8(int(r), int(c))
        ):
            out[r, c] = fill_col
            continue
        if any(
            not (0 <= nr < h and 0 <= nc < w and reach_mask[nr, nc])
            for nr, nc in neighbors4(int(r), int(c))
        ):
            out[r, c] = fill_col

    return out


# --------------- generator ---------------

BG = 8
WALL = 1
MARKER = 6
FILL = 7

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)
REACH_RATIO_RANGE = (0.22, 0.78)
DIFF_RATIO_RANGE = (0.12, 0.60)

NUM_VISIBLE_DECOR_RANGE = (0, 2)
NUM_HIDDEN_DECOR_RANGE = (1, 2)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

GEN_ATTEMPTS = 600


def _paint_output(structural: np.ndarray, reach: np.ndarray, marker_rc: tuple[int, int]) -> np.ndarray:
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


def _place_decor(allowed_mask: np.ndarray, count: int, rng: random.Random) -> np.ndarray | None:
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


def _add_top_stub(structural: np.ndarray, gate_c: int, rng: random.Random) -> None:
    h, _w = structural.shape
    max_width = gate_c - 2
    if max_width < 2:
        return
    rw = rng.randint(2, min(5, max_width))
    c0 = rng.randint(1, gate_c - rw)
    rh = rng.randint(2, min(4, h - 3))
    structural[0:rh, c0 : c0 + rw] = True


def _add_right_stub(structural: np.ndarray, gate_r: int, rng: random.Random) -> None:
    h, w = structural.shape
    max_height = h - gate_r - 3
    if max_height < 2:
        return
    rh = rng.randint(2, min(5, max_height))
    r0 = rng.randint(gate_r + 2, h - rh - 1)
    rw = rng.randint(2, min(4, w - 2))
    structural[r0 : r0 + rh, w - rw : w] = True


def _build_structural(h: int, w: int, marker_rc: tuple[int, int], rng: random.Random) -> np.ndarray | None:
    structural = np.zeros((h, w), dtype=bool)
    orientation = rng.choice(["vertical", "horizontal"])

    if orientation == "vertical":
        if w < 8:
            return None
        gate_c = rng.randint(3, w - 4)
        gate_r = rng.randint(2, h - 3)
        structural[:, gate_c] = True
        structural[gate_r, gate_c] = False
        if rng.random() < 0.65:
            _add_top_stub(structural, gate_c, rng)
    else:
        if h < 8:
            return None
        gate_r = rng.randint(2, h - 4)
        gate_c = rng.randint(2, w - 3)
        structural[gate_r, :] = True
        structural[gate_r, gate_c] = False
        if rng.random() < 0.65:
            _add_right_stub(structural, gate_r, rng)

    if structural[marker_rc]:
        return None
    seed = np.zeros((h, w), dtype=bool)
    seed[h - 2 : h, 0:2] = True
    if (structural & seed).any():
        return None
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


def _gen_attempt() -> np.ndarray | None:
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    structural = _build_structural(h, w, marker_rc, rng)
    if structural is None:
        return None

    plain_reach = _flood_mask(marker_rc, structural, h, w)
    width_reach = _width_reach_mask(marker_rc, ~structural)

    reach_frac = float(width_reach.mean())
    diff_frac = float((plain_reach & ~width_reach).mean())
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None
    if not (DIFF_RATIO_RANGE[0] <= diff_frac <= DIFF_RATIO_RANGE[1]):
        return None
    if not _validate_structural(structural, width_reach, h, w):
        return None

    out = _paint_output(structural, width_reach, marker_rc)
    fill_mask = out == FILL
    if not fill_mask.any() or len(components_mask(fill_mask, connectivity=4)) != 1:
        return None

    hidden_allowed = plain_reach & ~width_reach & ~structural
    visible_allowed = width_reach & (out == BG)
    hidden_allowed[0, :] = hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = hidden_allowed[:, -1] = False

    if hidden_allowed.sum() < 3:
        return None

    n_hidden = rng.randint(*NUM_HIDDEN_DECOR_RANGE)
    n_visible = rng.randint(*NUM_VISIBLE_DECOR_RANGE)
    hidden_placed = _place_decor(hidden_allowed, n_hidden, rng)
    if hidden_placed is None or not hidden_placed.any():
        return None
    visible_placed = _place_decor(visible_allowed, n_visible, rng)
    if visible_placed is None:
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 width-constraint variant")
