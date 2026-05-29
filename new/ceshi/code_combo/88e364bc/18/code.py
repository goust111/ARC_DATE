# Combo variant of 88e364bc (index 18)
# combo_spec: D3/0+D15/0
# Name: 迭代稳定+重力流动

"""
Design analysis for the combo variant.

fusion_summary:
  - Preserve the original scene grammar:
    border legends, colored wall loops, enclosed background regions, and yellow movers.
  - Replace straight one-pass sliding with synchronous gravity flow:
    every round, each yellow may move at most one cell.
  - Repeat rounds until no yellow can move.

concretization_and_behavior_trace:
  - In the original task, a legend direction was a transport ray. A yellow at (r, c) kept
    advancing along that ray until the wall or another settled yellow stopped it.
  - In this combo, the legend direction is reinterpreted as the gravity axis for its wall color.
    Inside one region and one round:
    1. a yellow first tries to move one cell along gravity
    2. if that cell is blocked, it searches sideways on the current level for the nearest spill
       point that would let it descend later, or else the nearest supported pooling cell
    3. it moves only one sideways cell toward that target
    4. all move decisions use the occupancy from the start of the round
    5. conflicting targets are cancelled for that round
  - Small example in a down-gravity 4x4 cavity with yellow sources on the top row at the two
    leftmost cells:
    original straight slide:
      each yellow stays in its own column and packs vertically against the floor
    D3-only immediate gravity settle:
      deeper particles fully fall first, so the second particle can keep flowing farther right
      in the same outer pass
    combo rule here:
      round 1 both yellows fall one cell
      later one reaches the floor, the other is temporarily blocked and shifts sideways only
      one cell, then falls in a later round
      final pool is lower and wider, and it differs from both the original straight slide and
      the D3 single-variant settle

complexity_calibration:
  - Generated samples use only cardinal legends.
    This keeps "gravity + lateral spread" visually legible and avoids hiding the combo behind
    diagonal corner-cut details.
  - Each legend color gets exactly one rectangular enclosed region in generated samples.
    That is intentionally simpler than the original irregular-maze generator because the new
    behavior should be read from yellow dynamics, not from boundary noise.
  - The generator uses a repeated two-yellow motif per region that guarantees:
    lateral spread happens,
    stabilization takes multiple rounds,
    and the final result differs from a D3-style immediate settle baseline.

implicit_assumption_audit_of_original_main:
  - _legend_directions():
    still valid, because legends remain border-touching filled rectangles with three colors.
  - passable-region extraction via bg/yellow connected components:
    still valid, because only yellow moves and interiors remain background except for yellows.
  - border-color inference by counting adjacent wall colors:
    still valid, because each rectangular cavity is enclosed by a single wall color.
  - original _slide_points():
    invalid after fusion, because movement is no longer a monotone ray that can be collapsed into
    a far-to-near stack. Sideways motion and round-start occupancy now matter.
  - original "clear all yellows, compute final positions, repaint":
    still valid at region scope, provided the replacement transport routine preserves count.

placement_legality:
  - Legends are placed first at distinct corners and reserve a one-cell buffer.
  - Regions are axis-aligned wall rectangles with their own one-cell outer buffer, so they do not
    overlap legends, each other, or the grid border.
  - Yellow sources are placed only on interior cells and never overlap.
  - Topology is verified by checking that enclosed background components match the intended
    region interiors exactly.

visual_observability:
  - The same visual motif repeats across samples:
    yellow pairs start on the gravity-opposite side, then end up pooled against the gravity-facing
    wall with a visible lateral offset.
  - Because each particle moves only one cell per round, the final pattern is not just "drop in
    the same column"; it visibly spreads along a floor or side shelf.
  - The generator rejects samples unless the combo output differs from:
    1. the original directional straight-slide baseline, and
    2. a D3-style immediate gravity-settle baseline.

visual_cleanliness_and_inference:
  - No extra symbols are added inside regions.
  - Each chamber contains only wall color, black background, and yellow movers, so the moved
    objects are unambiguous.
  - No hidden use of color ordering exists. Legend fill color still provides the only mapping from
    border marker to controlled region family.

new_assumptions_added_by_combo_main:
  - Gravity is treated as cardinal in generated samples; diagonal legend vectors are normalized to a
    cardinal axis defensively in the solver.
  - Sideways tie breaks are deterministic (left before right, or up before down in horizontal
    gravity), and the generator uses an asymmetric source motif so humans do not need to infer from
    perfect symmetry.
  - Same-round conflicts are resolved by cancelling all moves into a multiply-targeted cell.
    The generator avoids relying on such collisions, but the solver handles them safely.
"""

from __future__ import annotations

import random
import sys

import numpy as np

for _ARC_UTILS_DIR in (
    "/home/lixinyu/snap/new",
    "/home/lixinyu/snap/new/solvers",
    "/home/lixinyu/snap/new/arc2_solver_select",
):
    if _ARC_UTILS_DIR not in sys.path:
        sys.path.append(_ARC_UTILS_DIR)

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """Decode border legends and map fill color -> centroid-sign direction."""
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for pts in components_mask(non_bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        fill_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != fill_col]
        if len(others) != 2:
            continue
        others.sort()
        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            center_candidates = []
            for _, col in others:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                center_candidates.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            center_candidates.sort()
            center_col = center_candidates[0][1]
            indicator_col = center_candidates[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = [(r, c) for r, c in pts if int(g[r, c]) == center_col]
        indic = [(r, c) for r, c in pts if int(g[r, c]) == indicator_col]
        if not center or not indic:
            continue

        cc = np.mean(np.array(center, dtype=float), axis=0)
        ci = np.mean(np.array(indic, dtype=float), axis=0)
        vr, vc = (ci - cc).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        prev = best.get(fill_col)
        if prev is None or len(pts) > prev[0]:
            best[fill_col] = (len(pts), (dr, dc))

    return {k: v for k, (_, v) in best.items()}


def _normalize_gravity_dir(dr: int, dc: int) -> tuple[int, int]:
    """Generator uses cardinal gravity; diagonal inputs are collapsed defensively."""
    dr = int(dr)
    dc = int(dc)
    if dr != 0 and dc != 0:
        return (1 if dr > 0 else -1), 0
    if dr != 0:
        return (1 if dr > 0 else -1), 0
    if dc != 0:
        return 0, (1 if dc > 0 else -1)
    raise ValueError("gravity direction cannot be zero")


def _depth_key(p: tuple[int, int], dr: int, dc: int) -> int:
    return dr * p[0] + dc * p[1]


def _lateral_key(p: tuple[int, int], dr: int, dc: int) -> int:
    return p[1] if dr != 0 else p[0]


def _lateral_dirs(dr: int, dc: int) -> list[tuple[int, int, int]]:
    if dr != 0:
        return [(0, -1, 0), (0, 1, 1)]
    return [(-1, 0, 0), (1, 0, 1)]


def _best_lateral_target(
    region_set: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    pos: tuple[int, int],
    dr: int,
    dc: int,
) -> tuple[tuple[int, int], bool] | None:
    """
    Search sideways from one cell.

    Returns:
      - (target, True)  for the nearest lateral spill point that can descend later
      - (target, False) for the nearest supported pooling point
      - None            if no lateral motion is available
    """
    r0, c0 = pos
    spill: list[tuple[int, int, int, int]] = []
    supported: list[tuple[int, int, int, int]] = []
    for sdr, sdc, side_rank in _lateral_dirs(dr, dc):
        r, c = r0, c0
        dist = 0
        while True:
            r += sdr
            c += sdc
            dist += 1
            here = (r, c)
            if here not in region_set or here in occupied:
                break
            below = (r + dr, c + dc)
            if below in region_set and below not in occupied:
                spill.append((dist, side_rank, r, c))
                break
            supported.append((dist, side_rank, r, c))
    if spill:
        _, _, r, c = min(spill)
        return (r, c), True
    if supported:
        _, _, r, c = min(supported)
        return (r, c), False
    return None


def _choose_one_step(
    region_set: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    pos: tuple[int, int],
    dr: int,
    dc: int,
) -> tuple[int, int]:
    """Choose the one-cell move for a single round using round-start occupancy."""
    r, c = pos
    below = (r + dr, c + dc)
    if below in region_set and below not in occupied:
        return below

    lateral = _best_lateral_target(region_set, occupied, pos, dr, dc)
    if lateral is None:
        return pos
    (tr, tc), _ = lateral
    if dr != 0:
        step = (r, c + (1 if tc > c else -1))
    else:
        step = (r + (1 if tr > r else -1), c)
    if step not in region_set or step in occupied:
        return pos
    return step


def _iterate_gravity_until_stable(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> tuple[list[tuple[int, int]], int, int]:
    """
    Simulate synchronous one-step gravity rounds until no yellow can move.

    Returns (final_points, rounds, lateral_steps).
    """
    occupied = {tuple(map(int, p)) for p in points}
    rounds = 0
    lateral_steps = 0
    max_iters = len(region_set) * max(1, len(points)) * 8
    for _ in range(max_iters):
        ordered = sorted(occupied)
        intents = {p: _choose_one_step(region_set, occupied, p, dr, dc) for p in ordered}
        target_counts: dict[tuple[int, int], int] = {}
        for p, q in intents.items():
            if q != p:
                target_counts[q] = target_counts.get(q, 0) + 1

        moved = False
        new_occupied: set[tuple[int, int]] = set()
        for p in ordered:
            q = intents[p]
            if q != p and target_counts.get(q, 0) == 1:
                if (q[0] - p[0], q[1] - p[1]) != (dr, dc):
                    lateral_steps += 1
                new_occupied.add(q)
                moved = True
            else:
                new_occupied.add(p)

        if not moved:
            return sorted(new_occupied), rounds, lateral_steps
        occupied = new_occupied
        rounds += 1

    return sorted(occupied), rounds, lateral_steps


def _flow_particle_d3(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    occupied: set[tuple[int, int]],
    dr: int,
    dc: int,
) -> tuple[int, int]:
    """Single-particle local settle used for the D3 baseline comparison."""
    r, c = start
    seen = {(r, c)}
    for _ in range(len(region_set) * 4 + 4):
        below = (r + dr, c + dc)
        if below in region_set and below not in occupied:
            r, c = below
            if (r, c) in seen:
                break
            seen.add((r, c))
            continue

        lateral = _best_lateral_target(region_set, occupied, (r, c), dr, dc)
        if lateral is None:
            break
        (nr, nc), has_drop = lateral
        if (nr, nc) == (r, c):
            break
        r, c = nr, nc
        if (r, c) in seen:
            break
        seen.add((r, c))
        if not has_drop:
            break
    return r, c


def _gravity_settle_points_d3(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    """Immediate-settle gravity baseline from the D3 single variant."""
    if not points:
        return []
    positions = [tuple(map(int, p)) for p in points]
    max_iters = len(region_set) * max(1, len(points)) * 4
    for _ in range(max_iters):
        moved = False
        order = sorted(
            range(len(positions)),
            key=lambda i: (_depth_key(positions[i], dr, dc), _lateral_key(positions[i], dr, dc), positions[i]),
            reverse=True,
        )
        occupied = set(positions)
        for i in order:
            cur = positions[i]
            occupied.remove(cur)
            nxt = _flow_particle_d3(region_set, cur, occupied, dr, dc)
            occupied.add(nxt)
            if nxt != cur:
                positions[i] = nxt
                moved = True
        if not moved:
            return sorted(positions)
    return sorted(positions)


def _directional_slide_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    """Original straight-slide baseline for degeneracy checks."""
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (_depth_key(p, dr, dc), p[0], p[1]), reverse=True)
    occupied = set()
    out_pts = []
    for r0, c0 in points_sorted:
        r, c = r0, c0
        while True:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region_set or (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        out_pts.append((r, c))
    return sorted(out_pts)


def _collect_regions(
    g: np.ndarray,
    bg: int,
    mover: int,
    dirs: dict[int, tuple[int, int]],
) -> list[dict]:
    h, w = g.shape
    passable = (g == bg) | (g == mover)
    infos = []
    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        region_set = {(int(r), int(c)) for r, c in pts}
        border_counts: dict[int, int] = {}
        for r, c in pts:
            for nr, nc in neighbors4(int(r), int(c)):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in region_set:
                    continue
                v = int(g[nr, nc])
                if v == bg or v == mover:
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue
        border_col = max(border_counts, key=border_counts.get)
        if border_col not in dirs:
            continue
        infos.append(
            {
                "region_set": region_set,
                "movers": movers,
                "dir": dirs[border_col],
                "border_color": border_col,
            }
        )
    return infos


def _solve_combo(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    mover = 4
    dirs = {col: _normalize_gravity_dir(*vec) for col, vec in _legend_directions(g, bg).items()}
    assert dirs, "legend_directions must find at least one legend"

    out = g.copy()
    for info in _collect_regions(g, bg, mover, dirs):
        for r, c in info["movers"]:
            out[r, c] = bg
        new_pts, _, _ = _iterate_gravity_until_stable(info["region_set"], info["movers"], *info["dir"])
        for r, c in new_pts:
            out[r, c] = mover
    return out


def _solve_d3_baseline(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    mover = 4
    dirs = {col: _normalize_gravity_dir(*vec) for col, vec in _legend_directions(g, bg).items()}
    out = g.copy()
    for info in _collect_regions(g, bg, mover, dirs):
        for r, c in info["movers"]:
            out[r, c] = bg
        new_pts = _gravity_settle_points_d3(info["region_set"], info["movers"], *info["dir"])
        for r, c in new_pts:
            out[r, c] = mover
    return out


def _solve_directional_baseline(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    mover = 4
    dirs = {col: _normalize_gravity_dir(*vec) for col, vec in _legend_directions(g, bg).items()}
    out = g.copy()
    for info in _collect_regions(g, bg, mover, dirs):
        for r, c in info["movers"]:
            out[r, c] = bg
        new_pts = _directional_slide_points(info["region_set"], info["movers"], *info["dir"])
        for r, c in new_pts:
            out[r, c] = mover
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Settle yellow pixels inside each walled region by repeated gravity-flow rounds.

    Input structure:
    - The background is the most common color, while yellow pixels lie inside enclosed chambers
      bounded by a single wall color.
    - Border legends are solid rectangles on the outer frame; a legend's fill color matches the
      wall color of the regions it controls, and its red indicator gives a gravity axis.

    Rule:
    - For each enclosed region, use the legend with the matching wall color to get a gravity
      direction.
    - Then iterate synchronous rounds inside that region: each yellow tries to move one cell along
      gravity; if blocked, it shifts one lateral step toward the nearest place where it can keep
      descending, or else toward the nearest supported pooling position.
    - All decisions in a round use the occupancy from the start of that round, and if multiple
      yellows target the same cell, those moves are cancelled. Repeat until no yellow moves.

    Key insight:
    - The legend no longer means "slide straight forever"; it picks which side of the chamber acts
      as down. The output is the stable pool formed by iterative gravity and sideways spreading.
    """
    return _solve_combo(grid)


# --------------- generator ---------------

GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1

SPECIAL_SIZES = [(4, 4), (4, 5), (4, 6), (5, 4), (5, 5), (5, 6), (6, 4), (6, 5), (6, 6)]
PLACEMENT_RETRIES = 200
GEN_RETRIES = 500


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = 1
    g[1, 2] = 1
    g[1, 3] = 2
    g[3, 1] = 1
    g[3, 2] = 1
    g[3, 3] = 2
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


def _corner_origin(corner: str, h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - pw
    if corner == "BL":
        return h - ph, 0
    if corner == "BR":
        return h - ph, w - pw
    raise ValueError(corner)


def _place_legends(
    rng: random.Random,
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    reserved = np.zeros((h, w), dtype=bool)
    legends: list[tuple[int, tuple[int, int]]] = []
    for corner, wall_color in zip(corners, wall_colors):
        dr, dc = rng.choice(CARDINAL_DIRS)
        patch = _legend_5x5(wall_color, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo = max(0, r0 - LEGEND_BUFFER)
        r_hi = min(h, r0 + ph + LEGEND_BUFFER)
        c_lo = max(0, c0 - LEGEND_BUFFER)
        c_hi = min(w, c0 + pw + LEGEND_BUFFER)
        reserved[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wall_color, (dr, dc)))
    return reserved, legends


def _can_place_box(
    reserved: np.ndarray,
    outer_r0: int,
    outer_c0: int,
    outer_h: int,
    outer_w: int,
) -> bool:
    h, w = reserved.shape
    if outer_r0 <= 0 or outer_c0 <= 0:
        return False
    if outer_r0 + outer_h >= h or outer_c0 + outer_w >= w:
        return False
    r_lo = outer_r0 - 1
    c_lo = outer_c0 - 1
    r_hi = outer_r0 + outer_h + 1
    c_hi = outer_c0 + outer_w + 1
    return not reserved[r_lo:r_hi, c_lo:c_hi].any()


def _paint_rect_region(
    grid: np.ndarray,
    reserved: np.ndarray,
    outer_r0: int,
    outer_c0: int,
    inner_h: int,
    inner_w: int,
    color: int,
    direction: tuple[int, int],
) -> dict:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    walls = set()
    for c in range(outer_c0, outer_c0 + outer_w):
        walls.add((outer_r0, c))
        walls.add((outer_r0 + outer_h - 1, c))
    for r in range(outer_r0 + 1, outer_r0 + outer_h - 1):
        walls.add((r, outer_c0))
        walls.add((r, outer_c0 + outer_w - 1))
    for r, c in walls:
        grid[r, c] = color

    r_lo = outer_r0 - 1
    c_lo = outer_c0 - 1
    r_hi = outer_r0 + outer_h + 1
    c_hi = outer_c0 + outer_w + 1
    reserved[r_lo:r_hi, c_lo:c_hi] = True

    interior = {
        (r, c)
        for r in range(outer_r0 + 1, outer_r0 + 1 + inner_h)
        for c in range(outer_c0 + 1, outer_c0 + 1 + inner_w)
    }
    return {
        "color": color,
        "dir": direction,
        "interior": interior,
        "walls": walls,
        "inner_r0": outer_r0 + 1,
        "inner_c0": outer_c0 + 1,
        "inner_h": inner_h,
        "inner_w": inner_w,
    }


def _place_special_region(
    rng: random.Random,
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    direction: tuple[int, int],
) -> dict | None:
    h, w = grid.shape
    sizes = SPECIAL_SIZES[:]
    rng.shuffle(sizes)
    for inner_h, inner_w in sizes:
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        for _ in range(PLACEMENT_RETRIES):
            outer_r0 = rng.randint(1, h - outer_h - 1)
            outer_c0 = rng.randint(1, w - outer_w - 1)
            if not _can_place_box(reserved, outer_r0, outer_c0, outer_h, outer_w):
                continue
            return _paint_rect_region(grid, reserved, outer_r0, outer_c0, inner_h, inner_w, color, direction)
    return None


def _generate_regions(
    rng: random.Random,
    grid: np.ndarray,
    reserved: np.ndarray,
    legends: list[tuple[int, tuple[int, int]]],
) -> list[dict] | None:
    regions = []
    for color, direction in legends:
        reg = _place_special_region(rng, grid, reserved, color, direction)
        if reg is None:
            return None
        regions.append(reg)
    return regions


def _validate_topology(grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = grid.shape
    labeled, n_comp = label_mask(grid == 0, connectivity=4)
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    expected = [reg["interior"] for reg in regions]
    if len(bounded) != len(expected):
        return False
    return all(comp in expected for comp in bounded)


def _special_sources(reg: dict) -> list[tuple[int, int]]:
    r0 = reg["inner_r0"]
    c0 = reg["inner_c0"]
    h = reg["inner_h"]
    w = reg["inner_w"]
    dr, dc = reg["dir"]
    if (dr, dc) == (1, 0):
        return [(r0, c0), (r0, c0 + 1)]
    if (dr, dc) == (-1, 0):
        return [(r0 + h - 1, c0), (r0 + h - 1, c0 + 1)]
    if (dr, dc) == (0, 1):
        return [(r0, c0), (r0 + 1, c0)]
    if (dr, dc) == (0, -1):
        return [(r0, c0 + w - 1), (r0 + 1, c0 + w - 1)]
    raise ValueError((dr, dc))


def _place_yellows(regions: list[dict], grid: np.ndarray) -> bool:
    for reg in regions:
        sources = _special_sources(reg)
        reg["sources"] = sources
        for r, c in sources:
            if (r, c) not in reg["interior"] or grid[r, c] != 0:
                return False
            grid[r, c] = 4
    return True


def _combo_metrics(input_grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(input_grid, dtype=int)
    bg = get_bg(g)
    mover = 4
    dirs = {col: _normalize_gravity_dir(*vec) for col, vec in _legend_directions(g, bg).items()}
    infos = _collect_regions(g, bg, mover, dirs)

    multi_round_regions = 0
    lateral_steps = 0
    regions_diff_vs_d3 = 0
    regions_diff_vs_directional = 0
    total_rounds = 0

    for info in infos:
        combo_pts, rounds, lat = _iterate_gravity_until_stable(info["region_set"], info["movers"], *info["dir"])
        d3_pts = _gravity_settle_points_d3(info["region_set"], info["movers"], *info["dir"])
        directional_pts = _directional_slide_points(info["region_set"], info["movers"], *info["dir"])
        total_rounds += rounds
        lateral_steps += lat
        if rounds >= 2:
            multi_round_regions += 1
        if combo_pts != d3_pts:
            regions_diff_vs_d3 += 1
        if combo_pts != directional_pts:
            regions_diff_vs_directional += 1

    combo_out = _solve_combo(g)
    d3_out = _solve_d3_baseline(g)
    directional_out = _solve_directional_baseline(g)
    return {
        "n_regions": len(infos),
        "multi_round_regions": multi_round_regions,
        "lateral_steps": lateral_steps,
        "regions_diff_vs_d3": regions_diff_vs_d3,
        "regions_diff_vs_directional": regions_diff_vs_directional,
        "grid_diff_vs_d3": int(np.sum(combo_out != d3_out)),
        "grid_diff_vs_directional": int(np.sum(combo_out != directional_out)),
        "total_rounds": total_rounds,
    }


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    struct_mask = (input_grid != 4) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if int((solver_out == 4).sum()) != int((input_grid == 4).sum()):
        return False

    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    if not bool(np.all(regions_mask[solver_out == 4])):
        return False

    metrics = _combo_metrics(input_grid)
    if metrics["n_regions"] != len(regions):
        return False
    if metrics["multi_round_regions"] != len(regions):
        return False
    if metrics["regions_diff_vs_d3"] != len(regions):
        return False
    if metrics["regions_diff_vs_directional"] != len(regions):
        return False
    if metrics["lateral_steps"] < len(regions):
        return False
    if metrics["grid_diff_vs_d3"] <= 0 or metrics["grid_diff_vs_directional"] <= 0:
        return False
    if metrics["total_rounds"] < 2 * len(regions):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    reserved, legends = _place_legends(rng, grid)
    regions = _generate_regions(rng, grid, reserved, legends)
    if regions is None:
        return None
    if not _validate_topology(grid, regions):
        return None

    input_grid = grid.copy()
    if not _place_yellows(regions, input_grid):
        return None
    if not _oracle_check(input_grid, regions):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
