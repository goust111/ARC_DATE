# Variant of 88e364bc
# Dimension: D3 — 滑动路径类型 (path)
# Change: directional_movement——黄色像素沿图例指定的固定方向（基数或对角线）直线滑动，遇到区域边界或已占据位... → 重力流动（向下+侧向扩散）

"""
Design analysis for the D3 variant.

Reference-task note:
- The prompt asks to read solver code for source task 28a6681f first. In this workspace I
  could find /home/lixinyu/snap/new/ARC-data/ARC-AGI-2/data/evaluation/28a6681f.json but no
  local solver file named 28a6681f.py under the allowed repo paths. I therefore use the
  reference task's I/O examples plus the prompt's "gravity / flow" description as the baseline.

spec_variant:
  preserved_structure:
    - grid stays 20x20 with bg=0
    - border legends still live on corners and are linked to region-wall colors
    - colored walls still enclose passable regions
    - yellow count is preserved; only yellow positions change
  core_change:
    - original: each region has a legend vector and every yellow slides in one straight ray
    - variant: each region has a gravity direction; yellow behaves like discrete water
    - local rule in gravity coordinates:
      1. try to move one step "down" (along gravity)
      2. if blocked below, search left/right on the current level
      3. prefer the nearest lateral spill point that can drop to a lower level
      4. if no spill exists, occupy the nearest laterally supported cell, so a pool spreads
  concretization_and_trace:
    - input still shows small yellow tokens inside an empty cavity bounded by one wall color
    - output still leaves walls/legends unchanged, but the yellow cluster now sinks to the
      gravity-facing side and can bend around shelves/floors instead of staying on a straight line
    - example in a down-gravity region:
      original logic:
        a yellow at (r,c) would keep moving to (r+1,c), (r+2,c), ... until the wall
      variant logic:
        first yellow falls to the floor; second yellow above it cannot move down, so it
        laterally diffuses to a supported neighboring floor cell; with an opening to one side,
        the particle first slides sideways to that spill point and then falls again
    - this means the changed behavior happens exactly in the region-update phase of main():
      region detection and legend-color matching remain intact; point transport is replaced
  complexity_calibration:
    - gravity is restricted to cardinal directions in generated samples
    - this is intentional: "down + sideways" is visually legible for humans, while diagonal
      gravity would create ambiguous lateral fronts and hide the D3 change behind another axis change
    - 1-2 legend colors and repeated region colors are preserved so the color→rule mapping is
      still learnable from a few examples
    - generator uses 2-3 yellows per non-empty region and tries to stack them in the same
      gravity column so that lateral pooling becomes visible instead of degenerating to a straight drop
  implicit_assumption_review_of_original_main:
    - _legend_directions() itself is still valid: it only decodes a sign direction from the legend
    - original _slide_points() assumes a single fixed ray and a total order "far end to near end"
      that assumption is invalid once motion can turn sideways and re-descend
    - original main() clears movers before transport and then paints final cells; that is still safe
      because yellow remains the only moving color and count is preserved
    - original border-color detection by adjacent wall counts still holds because region topology
      is unchanged by the variant
    - original passable-region detection still holds because the region interiors remain bg/yellow only
  placement_legality:
    - legends are pasted first and buffered so regions never overlap them
    - walls are still one cell thick and never touch the outer border
    - regions are validated against enclosed background components
    - yellow sources are chosen only from region-interior cells
    - generator rejects samples where the gravity solver would match the old straight-slide baseline
  visual_observability:
    - from 2-4 examples, a human can see that yellows always migrate to the gravity-facing side,
      pool on floors, and bend sideways around support, unlike the original straight-line stacking
    - samples are rejected unless at least one particle changes lateral coordinate and the full-grid
      gravity output differs from the old directional-slide output
  visual_cleanliness:
    - no extra internal markers are added; the scene remains "legend + wall + yellow in cavity"
    - only the yellow configuration changes, so the variant is easy to inspect
  color_inference:
    - no new color remapping is introduced
    - legend fill color still identifies which wall color / region family uses the same gravity rule
  new_assumptions_added_by_variant_main:
    - gravity simulation assumes exactly one cardinal gravity axis per legend in generated inputs
    - lateral spill-point tie breaks are deterministic; generator avoids long symmetric tie cases by
      favoring stacked starting positions and rejecting degenerate outputs
    - generate_input() enforces these assumptions by using only cardinal legends and by checking
      that gravity output exhibits lateral-flow behavior
"""

from __future__ import annotations

import random
import sys
from typing import Iterable

import numpy as np

for _ARC_UTILS_DIR in (
    "/home/lixinyu/snap/new/solvers",
    "/home/lixinyu/snap/new/arc2_solver_select",
):
    if _ARC_UTILS_DIR not in sys.path:
        sys.path.append(_ARC_UTILS_DIR)

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Detect border legends.

    The legend's fill color is also the wall color it controls.
    The sign of centroid(indicator) - centroid(center) gives the legend direction.
    In this variant that direction is reinterpreted as the gravity axis.
    """
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

        bg_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != bg_col]
        if len(others) != 2:
            continue
        others.sort()
        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            cols = [others[0][1], others[1][1]]
            cents = []
            for col in cols:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                cents.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            cents.sort()
            center_col = cents[0][1]
            indicator_col = cents[1][1]
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

        prev = best.get(bg_col)
        if prev is None or len(pts) > prev[0]:
            best[bg_col] = (len(pts), (dr, dc))
    return {k: v for k, (_, v) in best.items()}


def _normalize_gravity_dir(dr: int, dc: int) -> tuple[int, int]:
    """
    Generated samples use only cardinal gravity.

    If a diagonal legend appears, collapse it to a cardinal gravity axis so the solver
    still behaves sensibly instead of crashing.
    """
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
    if dr != 0:
        return p[1]
    return p[0]


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
    Search sideways on the current level.

    Returns:
    - (target, True)  for the nearest lateral spill point that can drop lower
    - (target, False) for the nearest laterally supported pooling cell
    - None            if no sideways motion is possible
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


def _flow_particle(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    occupied: set[tuple[int, int]],
    dr: int,
    dc: int,
) -> tuple[int, int]:
    """
    Move one particle until it reaches a local gravity-resting position.

    The path can alternate between downward steps and lateral searches for spill / pooling cells.
    """
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


def _gravity_settle_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    if not points:
        return []
    dr, dc = _normalize_gravity_dir(dr, dc)
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
            nxt = _flow_particle(region_set, cur, occupied, dr, dc)
            occupied.add(nxt)
            if nxt != cur:
                positions[i] = nxt
                moved = True
        if not moved:
            return positions
    return positions


def _directional_slide_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
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
    return out_pts


def _region_infos(g: np.ndarray, bg: int, mover: int) -> list[dict]:
    h, w = g.shape
    passable = (g == bg) | (g == mover)
    infos = []
    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue
        pset = {(int(r), int(c)) for r, c in pts}
        border_counts: dict[int, int] = {}
        for r, c in pts:
            for nr, nc in neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in pset:
                    continue
                v = int(g[nr, nc])
                if v == bg or v == mover:
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue
        border_col = max(border_counts, key=border_counts.get)
        infos.append({"points": pts, "pset": pset, "movers": movers, "border_col": border_col})
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant solver: yellow particles settle under legend-coded gravity inside each enclosed region.

    The legend still links a wall color to a rule, but the rule is now a gravity axis rather than
    a fixed straight-line slide. Yellows fall along gravity, then spread laterally when blocked.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    mover = 4
    dirs = {col: _normalize_gravity_dir(*vec) for col, vec in _legend_directions(g, bg).items()}
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"

    out = g.copy()
    for info in _region_infos(g, bg, mover):
        border_col = info["border_col"]
        if border_col not in dirs:
            continue
        dr, dc = dirs[border_col]
        movers = info["movers"]
        for r, c in movers:
            out[r, c] = bg
        new_pts = _gravity_settle_points(info["pset"], movers, dr, dc)
        for r, c in new_pts:
            out[r, c] = mover
    return out


# --------------- generator ---------------

GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1

REGIONS_PER_COLOR = (1, 3)
MAX_TOTAL_REGIONS = 4
REGION_SIZE_BOUNDS = {
    1: (18, 90),
    2: (16, 56),
    3: (14, 38),
    4: (12, 28),
}
BLOB_GROW_RETRIES = 25

YELLOWS_PER_REGION = [2, 3]
TOP_SOURCE_BAND = 4
GEN_RETRIES = 400


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


def _grow_blob(
    canvas_mask: np.ndarray,
    seed: tuple[int, int],
    target_size: int,
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    h, w = canvas_mask.shape
    sr, sc = seed
    if not canvas_mask[sr, sc]:
        return None
    blob = {(sr, sc)}
    frontier = []

    def push_neighbors(r: int, c: int) -> None:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and canvas_mask[nr, nc] and (nr, nc) not in blob:
                frontier.append((nr, nc))

    push_neighbors(sr, sc)
    while len(blob) < target_size and frontier:
        idx = rng.randrange(len(frontier))
        r, c = frontier.pop(idx)
        if (r, c) in blob or not canvas_mask[r, c]:
            continue
        blob.add((r, c))
        push_neighbors(r, c)
    if len(blob) < target_size:
        return None
    return blob


def _blob_wall_cells(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    wall = set()
    for r, c in blob:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in blob:
                wall.add((nr, nc))
    return wall


def _fill_holes(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    walls = _blob_wall_cells(blob, h, w)
    barrier = np.zeros((h, w), dtype=bool)
    for r, c in blob:
        barrier[r, c] = True
    for r, c in walls:
        barrier[r, c] = True

    reachable = np.zeros((h, w), dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if not barrier[r, c]:
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not barrier[r, c]:
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        if reachable[r, c] or barrier[r, c]:
            continue
        reachable[r, c] = True
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr, nc] and not barrier[nr, nc]:
                stack.append((nr, nc))

    trapped = (~reachable) & (~barrier)
    if not trapped.any():
        return blob
    filled = set(blob)
    for r, c in np.argwhere(trapped):
        filled.add((int(r), int(c)))
    return filled


def _try_grow_valid_blob(
    rng: random.Random,
    canvas: np.ndarray,
    occupied: np.ndarray,
    target_size: int,
    h: int,
    w: int,
) -> set[tuple[int, int]] | None:
    for _ in range(BLOB_GROW_RETRIES):
        free = np.argwhere(canvas)
        if len(free) == 0:
            break
        sr, sc = free[rng.randrange(len(free))]
        cand = _grow_blob(canvas, (int(sr), int(sc)), target_size, rng)
        if cand is None:
            continue
        while True:
            filled = _fill_holes(cand, h, w)
            if filled == cand:
                break
            cand = filled
        if any(not canvas[r, c] for r, c in cand):
            continue
        walls = _blob_wall_cells(cand, h, w)
        if any(occupied[wr, wc] or wr in (0, h - 1) or wc in (0, w - 1) for wr, wc in walls):
            continue
        return cand
    return None


def _place_legends(rng: random.Random, grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)
    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []
    for corner, wall_color in zip(corners, wall_colors):
        dr, dc = rng.choice(CARDINAL_DIRS)
        patch = _legend_5x5(wall_color, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wall_color, (dr, dc)))
    return legend_occupied, legends


def _grow_regions(
    rng: random.Random,
    legends: list[tuple[int, tuple[int, int]]],
    legend_occupied: np.ndarray,
) -> list[dict] | None:
    h, w = legend_occupied.shape
    color_dir = {wc: d for wc, d in legends}
    all_colors = [wc for wc, _ in legends]

    region_colors = []
    for c in all_colors:
        region_colors.extend([c] * rng.randint(*REGIONS_PER_COLOR))
    if len(region_colors) > MAX_TOTAL_REGIONS:
        region_colors = rng.sample(region_colors, MAX_TOTAL_REGIONS)
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions = []

    for region_color in region_colors:
        target_size = rng.randint(size_min, size_max)
        canvas = np.ones((h, w), dtype=bool)
        canvas[0, :] = False
        canvas[-1, :] = False
        canvas[:, 0] = False
        canvas[:, -1] = False
        canvas &= ~occupied
        if not canvas.any():
            continue

        blob = _try_grow_valid_blob(rng, canvas, occupied, target_size, h, w)
        if blob is None:
            continue
        walls = _blob_wall_cells(blob, h, w)
        regions.append(
            {
                "color": region_color,
                "dir": color_dir[region_color],
                "interior": blob,
                "walls": walls,
            }
        )

        reserve = blob | walls
        extra = set()
        for r, c in walls:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reserve:
                    extra.add((nr, nc))
        for r, c in reserve | extra:
            occupied[r, c] = True

    if not regions:
        return None
    if {reg["color"] for reg in regions} != set(all_colors):
        return None
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
    region_sets = [reg["interior"] for reg in regions]
    if len(bounded) != len(region_sets):
        return False
    return all(comp in region_sets for comp in bounded)


def _choose_sources_for_region(
    rng: random.Random,
    blob: set[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]] | None:
    cells = list(blob)
    if len(cells) < 2:
        return None
    grouped: dict[int, list[tuple[int, int]]] = {}
    for p in cells:
        grouped.setdefault(_lateral_key(p, dr, dc), []).append(p)

    best_lat = None
    best_score = None
    for lat, pts in grouped.items():
        pts_sorted = sorted(pts, key=lambda p: (_depth_key(p, dr, dc), p[0], p[1]))
        min_depth = _depth_key(pts_sorted[0], dr, dc)
        top_pts = [p for p in pts_sorted if _depth_key(p, dr, dc) <= min_depth + TOP_SOURCE_BAND]
        score = (-len(top_pts), min_depth, abs(lat), lat)
        if best_score is None or score < best_score:
            best_score = score
            best_lat = lat
    assert best_lat is not None

    min_depth = min(_depth_key(p, dr, dc) for p in cells)
    target = 3 if len(cells) >= 24 and rng.random() < 0.5 else 2
    ordered = sorted(
        cells,
        key=lambda p: (
            _depth_key(p, dr, dc),
            abs(_lateral_key(p, dr, dc) - best_lat),
            p[0],
            p[1],
        ),
    )
    sources = []
    for p in ordered:
        if _depth_key(p, dr, dc) > min_depth + TOP_SOURCE_BAND and len(sources) >= 2:
            break
        if p not in sources:
            sources.append(p)
        if len(sources) >= target:
            break
    if len(sources) < 2:
        return None
    return sources


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    for reg in regions:
        sources = _choose_sources_for_region(rng, reg["interior"], *reg["dir"])
        if sources is None:
            return False
        reg["sources"] = sources
        for r, c in sources:
            grid[r, c] = 4
    return True


def _directional_baseline_main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    mover = 4
    dirs = {col: _normalize_gravity_dir(*vec) for col, vec in _legend_directions(g, bg).items()}
    out = g.copy()
    for info in _region_infos(g, bg, mover):
        border_col = info["border_col"]
        if border_col not in dirs:
            continue
        dr, dc = dirs[border_col]
        movers = info["movers"]
        for r, c in movers:
            out[r, c] = bg
        new_pts = _directional_slide_points(info["pset"], movers, dr, dc)
        for r, c in new_pts:
            out[r, c] = mover
    return out


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    struct_mask = (input_grid != 4) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if (solver_out == 4).sum() != (input_grid == 4).sum():
        return False
    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    return bool(np.all(regions_mask[solver_out == 4]))


def _variant_metrics(input_grid: np.ndarray, regions: list[dict]) -> dict[str, int]:
    gravity_out = main(input_grid)
    slide_out = _directional_baseline_main(input_grid)
    total_yellows = 0
    moved = 0
    lateral_moved = 0
    flow_regions = 0
    for reg in regions:
        sources = reg["sources"]
        dr, dc = reg["dir"]
        finals = _gravity_settle_points(reg["interior"], sources, dr, dc)
        total_yellows += len(sources)
        reg_lateral = 0
        for src, dst in zip(sources, finals):
            if src != dst:
                moved += 1
            if _lateral_key(src, dr, dc) != _lateral_key(dst, dr, dc):
                lateral_moved += 1
                reg_lateral += 1
        if reg_lateral > 0:
            flow_regions += 1
    return {
        "total_yellows": total_yellows,
        "moved": moved,
        "lateral_moved": lateral_moved,
        "flow_regions": flow_regions,
        "gravity_vs_slide_diff": int(np.sum(gravity_out != slide_out)),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    legend_occupied, legends = _place_legends(rng, grid)
    regions = _grow_regions(rng, legends, legend_occupied)
    if regions is None:
        return None

    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = reg["color"]
    if not _validate_topology(grid, regions):
        return None

    input_grid = grid.copy()
    if not _place_yellows(rng, regions, input_grid):
        return None
    if not _oracle_check(input_grid, regions):
        return None

    metrics = _variant_metrics(input_grid, regions)
    if metrics["moved"] != metrics["total_yellows"]:
        return None
    if metrics["lateral_moved"] <= 0:
        return None
    if metrics["flow_regions"] <= 0:
        return None
    if metrics["gravity_vs_slide_diff"] <= 0:
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            g = _gen_attempt(rng)
        except Exception:
            continue
        if g is not None:
            return g
    raise RuntimeError("generate_input: exceeded retries")
