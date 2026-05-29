# Variant of 8f3a5a89
# Dimension: D4 — 终止/边界判定条件 (dst)
# Change: 洪填被结构墙和网格边缘阻挡——结构墙定义为触碰网格边缘且4-邻接可达区域的墙色连通分量... → 仅墙壁阻挡（洪填可达网格边缘无阻挡）

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# Design analysis for the D4 variant:
# - Concrete behavior:
#   Structural blue components are now closed fences inside the grid. The flood
#   from the magenta corner may leave the board through any border gap, move
#   through the exterior padding, and re-enter elsewhere, so an open wall or a
#   border-touching wall is bypassable. Only a wall component that separates
#   marker-reachable cells from an enclosed pocket remains structural.
# - Trace on a small example:
#   Old rule: a vertical blue segment touching top and bottom split the board,
#   because the segment plus the two side edges enclosed the left side. New
#   rule: the same segment is no longer a boundary, because the flood can walk
#   through the outside padding above or below the grid and reach the other
#   side. A 1-cell-thick closed rectangle still traps its interior, so orange is
#   painted only around that rectangle.
# - Complexity calibration:
#   To make the difference visible, the generator always creates at least one
#   closed loop and usually also one border-touching open blue decoration. That
#   keeps repeated evidence for both "closed walls matter" and "edge-touching
#   open walls do not matter" without introducing too many wall categories.
# - Implicit assumptions audited in main():
#   `reach0` must be computed with an exterior padding; otherwise the solver
#   would silently fall back to the old edge-blocked behavior. Structural wall
#   detection cannot use "touches border" anymore; it now relies on a component
#   having both reachable and unreachable non-wall neighbors. The final orange
#   paint step must not color all reachable edge cells, because grid edges are
#   no longer boundaries.
# - Placement legality:
#   Structural loops stay strictly inside the canvas, do not overlap, and keep a
#   1-cell moat from one another so their enclosed pockets are unambiguous.
#   Hidden decorations are placed strictly inside those pockets and away from
#   the loop boundary so they do not merge with structural walls.
# - Visual observability and cleanliness:
#   The output shows orange only along enclosed pockets, never as an outer frame
#   around the whole board. Border-touching blue strokes remain blue without an
#   orange lining, which is the intended visible contrast from the source task.
def _flood_with_outside(start: tuple[int, int], blocked: np.ndarray) -> np.ndarray:
    """Flood on a 1-cell padded canvas so the walk may go around the grid exterior."""
    h, w = blocked.shape
    pad = np.zeros((h + 2, w + 2), dtype=bool)
    pad[1:-1, 1:-1] = blocked
    reach = np.zeros_like(pad, dtype=bool)
    sr, sc = start[0] + 1, start[1] + 1
    if pad[sr, sc]:
        raise ValueError("start lies on a blocked cell")
    q = deque([(sr, sc)])
    reach[sr, sc] = True
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and not reach[nr, nc] and not pad[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach[1:-1, 1:-1]


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint an inner border around the marker-reachable region when only closed walls matter.

    Input structure:
    - The background is the most common color; blue cells contain both real fences and fake wall decorations.
    - A unique magenta pixel sits on a corner and marks the flood-fill start.
    - Real fences are wall-colored connected components that actually enclose a pocket; open or edge-touching blue pieces can be bypassed through the grid exterior.

    Rule:
    - Flood-fill from the marker while treating the space outside the grid as open; only fence components can block the fill.
    - Remove wall-colored decorations that lie inside enclosed pockets on the unreachable side of a real fence.
    - Paint orange on reachable background cells that sit immediately inside a fence boundary.

    Key insight: Grid edges are no longer part of the barrier. A blue component counts as structural only if it separates reachable exterior-connected space from a trapped interior pocket.
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

    reach0 = _flood_with_outside((mr, mc), g == wall_col)

    walls: set[tuple[int, int]] = set()
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = [(int(r), int(c)) for r, c in comp_arr]
        adjacent_to_reach0 = False
        adjacent_to_unreach0 = False
        for r, c in comp:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    if reach0[nr, nc]:
                        adjacent_to_reach0 = True
                    elif int(g[nr, nc]) != wall_col:
                        adjacent_to_unreach0 = True
            if adjacent_to_reach0 and adjacent_to_unreach0:
                break
        if adjacent_to_reach0 and adjacent_to_unreach0:
            walls |= set(comp)

    structural_mask = np.zeros((h, w), dtype=bool)
    for r, c in walls:
        structural_mask[r, c] = True

    reach = _flood_with_outside((mr, mc), structural_mask)

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and not structural_mask[r, c] and not reach[r, c]:
                out[r, c] = bg

    fill_col = 7
    for r in range(h):
        for c in range(w):
            if not reach[r, c] or int(out[r, c]) != bg:
                continue
            if any(0 <= nr < h and 0 <= nc < w and structural_mask[nr, nc] for nr, nc in neighbors8(r, c)):
                out[r, c] = fill_col
                continue
            if any(
                0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != wall_col and not reach[nr, nc]
                for nr, nc in neighbors4(r, c)
            ):
                out[r, c] = fill_col

    return out


BG = 8
WALL = 1
MARKER = 6
FILL = 7

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

NUM_LOOPS_RANGE = (1, 2)
LOOP_RETRIES = 80
LOOP_SEPARATION = 1

REACH_RATIO_RANGE = (0.45, 0.90)

NUM_VISIBLE_DECOR_RANGE = (1, 3)
NUM_HIDDEN_DECOR_RANGE = (0, 2)
DECOR_AREA_RANGE = (1, 4)

GEN_ATTEMPTS = 300


def _rect_ring_cells(r0: int, c0: int, r1: int, c1: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    return cells, interior


def _component_masks_from_loops(h: int, w: int, loops: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, list[np.ndarray]]:
    structural = np.zeros((h, w), dtype=bool)
    interiors: list[np.ndarray] = []
    for r0, c0, r1, c1 in loops:
        cells, interior = _rect_ring_cells(r0, c0, r1, c1)
        for r, c in cells:
            structural[r, c] = True
        pocket = np.zeros((h, w), dtype=bool)
        for r, c in interior:
            pocket[r, c] = True
        interiors.append(pocket)
    return structural, interiors


def _loop_candidate(h: int, w: int, rng: random.Random) -> tuple[int, int, int, int] | None:
    max_h = min(8, h - 2)
    max_w = min(8, w - 2)
    if max_h < 4 or max_w < 4:
        return None
    rh = rng.randint(4, max_h)
    rw = rng.randint(4, max_w)
    if h - rh - 1 < 1 or w - rw - 1 < 1:
        return None
    r0 = rng.randint(1, h - rh - 1)
    c0 = rng.randint(1, w - rw - 1)
    return r0, c0, r0 + rh - 1, c0 + rw - 1


def _place_structural_loops(h: int, w: int, marker_rc: tuple[int, int], rng: random.Random) -> tuple[np.ndarray, list[np.ndarray]] | None:
    target = rng.randint(*NUM_LOOPS_RANGE)
    loops: list[tuple[int, int, int, int]] = []
    for _ in range(LOOP_RETRIES):
        if len(loops) >= target:
            break
        cand = _loop_candidate(h, w, rng)
        if cand is None:
            continue
        r0, c0, r1, c1 = cand
        if r0 <= marker_rc[0] <= r1 and c0 <= marker_rc[1] <= c1:
            continue
        if c0 <= 1 and r1 >= h - 2:
            continue
        if any(not (r1 + LOOP_SEPARATION < rr0 or rr1 + LOOP_SEPARATION < r0 or c1 + LOOP_SEPARATION < cc0 or cc1 + LOOP_SEPARATION < c0) for rr0, cc0, rr1, cc1 in loops):
            continue
        loops.append(cand)
    if len(loops) != target:
        return None
    return _component_masks_from_loops(h, w, loops)
def _place_patch_mask(allowed_mask: np.ndarray, count: int, rng: random.Random) -> np.ndarray | None:
    h, w = allowed_mask.shape
    placed = np.zeros((h, w), dtype=bool)
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
                    border=0,
                    padding=1,
                    padding_connectivity=8,
                )
            except ValueError:
                continue
            placed[r0 : r0 + ph, c0 : c0 + pw] = True
            break
        else:
            return None
    return placed


def _place_border_decoration(h: int, w: int, structural: np.ndarray, marker_rc: tuple[int, int], rng: random.Random) -> np.ndarray | None:
    placed = np.zeros((h, w), dtype=bool)
    for _ in range(40):
        mode = rng.choice(["top", "right", "bottom"])
        if mode == "top":
            length = rng.randint(max(2, w // 5), max(2, w // 2))
            c0 = rng.randint(0, w - length)
            cells = [(0, c) for c in range(c0, c0 + length)]
        elif mode == "right":
            length = rng.randint(max(2, h // 5), max(2, h // 2))
            r0 = rng.randint(0, h - length)
            cells = [(r, w - 1) for r in range(r0, r0 + length)]
        else:
            length = rng.randint(max(2, w // 5), max(2, w // 2))
            c0 = rng.randint(1, w - length)
            cells = [(h - 1, c) for c in range(c0, c0 + length)]
        if marker_rc in cells:
            continue
        cand = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            cand[r, c] = True
        if (cand & structural).any():
            continue
        if (_dilate8(cand) & structural).any():
            continue
        placed |= cand
        return placed
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    placed = _place_structural_loops(h, w, marker_rc, rng)
    if placed is None:
        return None
    structural, interiors = placed
    if structural[marker_rc]:
        return None

    reach = _flood_with_outside(marker_rc, structural)
    reach_frac = reach.mean()
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    border_decor = _place_border_decoration(h, w, structural, marker_rc, rng)
    if border_decor is None:
        return None

    visible_decor = border_decor.copy()
    n_visible_extra = rng.randint(NUM_VISIBLE_DECOR_RANGE[0] - 1, NUM_VISIBLE_DECOR_RANGE[1] - 1)
    visible_allowed = reach & ~structural & ~_dilate8(border_decor)
    visible_allowed[marker_rc] = False
    extra_visible = _place_patch_mask(visible_allowed, n_visible_extra, rng)
    if extra_visible is None:
        return None
    visible_decor |= extra_visible

    base = np.full((h, w), BG, dtype=int)
    base[structural | visible_decor] = WALL
    base[marker_rc] = MARKER
    out_final = main(base)
    fill_mask = out_final == FILL
    if not fill_mask.any():
        return None

    hidden_allowed = np.zeros((h, w), dtype=bool)
    for pocket in interiors:
        pocket_inner = pocket.copy()
        pocket_inner &= ~_dilate4(structural)
        hidden_allowed |= pocket_inner
    cap = min(NUM_HIDDEN_DECOR_RANGE[1], int(hidden_allowed.sum()) // 3)
    n_hidden = 0 if cap <= 0 else rng.randint(NUM_HIDDEN_DECOR_RANGE[0], cap)
    hidden_decor = _place_patch_mask(hidden_allowed, n_hidden, rng)
    if hidden_decor is None:
        return None

    inp = out_final.copy()
    inp[inp == FILL] = BG
    inp[hidden_decor] = WALL

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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D4 variant")
