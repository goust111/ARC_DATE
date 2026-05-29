# Combo variant of 88e364bc (index 27)
# combo_spec: D7/2+D17/1
# Name: 共享墙壁+独立颜色判定

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


# Design analysis for the combo variant
# -------------------------------------
# concreteization_and_trace:
#   - The grid always contains exactly two enclosed regions that share one straight wall seam.
#   - The shared seam alternates the two legend-linked wall colors cell by cell.
#   - Example with a vertical seam and colors A/B:
#       left region private walls are mostly A, right region private walls are mostly B,
#       seam rows go A/B/A/B/...
#       A yellow at (r, seam-1) in the left region next to a B-colored seam cell uses legend B,
#       even though the left region's boundary majority is still A.
#   - Old solver path:
#       1. count all boundary colors for the whole region
#       2. pick the regional majority color
#       3. move every yellow in that region the same way
#   - Combo solver path:
#       1. still find enclosed passable regions
#       2. but for each yellow separately, inspect the nearest boundary wall cell
#       3. use that local wall color's legend direction
#       4. simulate all yellows with fixed per-yellow directions
#   - Visible difference:
#       in one shared region we place one yellow next to an A seam cell and one next to a B seam cell,
#       so the pair splits in opposite directions along the seam. The old region-majority solver would
#       move both together, so the combo behavior is unmistakable.
#
# complexity_calibration:
#   - Two wall colors are mandatory because the shared seam cue only exists when both colors repeat.
#   - Only one shared pair is used; extra regions would weaken the main clue.
#   - Directions are restricted to cardinal motion parallel to the seam:
#       vertical seam -> up/down legends, horizontal seam -> left/right legends.
#     This keeps the new boundary-ownership clue readable instead of mixing it with diagonal geometry.
#   - Each region gets two yellows, one assigned to each wall color, so the independent-color rule is
#     repeated twice per sample rather than appearing as a one-off exception.
#
# implicit_assumptions_review:
#   - Original `border_counts[...] += 1` assumed one region-level summary color; invalid now.
#   - Original `border_col = max(border_counts, key=...)` assumed every yellow in one chamber shares a
#     direction; invalid now.
#   - Original `_slide_points(region, movers, dr, dc)` assumed a single `(dr, dc)` per chamber; invalid.
#   - `_legend_directions(...)` still holds because legends are unchanged filled rectangles on the border.
#   - Nearest-wall lookup introduces a new assumption: ties between different colors must be avoided or the
#     task becomes visually ambiguous. The generator therefore places yellows immediately beside the seam,
#     away from corners, so their nearest wall color is unique and human-visible.
#   - Mixed directions introduce motion-order risk. The solver switches to repeated one-step simulation,
#     and the generator only uses opposite directions parallel to the seam so the four yellows separate
#     cleanly instead of deadlocking.
#
# placement_legality:
#   - Legends are placed first in two distinct corners and reserve a 1-cell halo.
#   - The shared pair is then placed wholly inside the remaining area with its own 1-cell halo, so it
#     cannot touch legends or the outer border.
#   - The topology validator confirms that the painted walls create exactly the two intended bounded
#     background components.
#   - Yellow placements are drawn only from cells inside region interiors, are all distinct, and are
#     validated against the solver before acceptance.
#
# visual_observability_and_cleanliness:
#   - The shared seam is long, straight, and alternates only the two legend colors, so the shared-wall
#     phenomenon is easy to spot by eye.
#   - Private walls stay mostly single-color, which preserves the impression that each chamber "belongs"
#     to one color even though local seam cells can override it.
#   - Yellows sit in a single column or row next to the seam and then split apart; this makes the local
#     wall-color reading easy to infer from 2-3 examples.
#
# change_inferability:
#   - There are only two active wall colors, and both repeat in legends, private walls, and shared seam.
#   - The color-to-direction mapping is never hidden behind numeric order; directions come only from the
#     visible legend arrows.
#   - Generator validation rejects any sample where the combo solver matches the old region-majority
#     behavior, so accepted samples always exhibit the fused rule rather than a degenerate original case.
#
# new_solver_assumptions_checked:
#   - `main()` assumes every active yellow has at least one adjacent boundary wall somewhere in its region;
#     the closed rectangular chambers guarantee this.
#   - `main()` assumes nearest-wall colors map to existing legends; all wall colors come from the two
#     legend fill colors, so this is guaranteed.
#   - `main()` assumes the one-step simulation stabilizes quickly; the structured opposite-direction
#     placements ensure monotone movement toward walls and avoid cyclic traffic patterns.


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """Detect border legends and decode fill-color -> movement direction."""
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
            candidates = []
            for _cnt, col in others:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                candidates.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            candidates.sort()
            center_col = candidates[0][1]
            indicator_col = candidates[1][1]
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


def _slide_points(region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    """Legacy same-direction slide, kept for generator validation against the original behavior."""
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied = set()
    out_pts = []
    for r0, c0 in points_sorted:
        r, c = r0, c0
        while True:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region_set:
                break
            if dr != 0 and dc != 0:
                if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
                    break
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        out_pts.append((r, c))
    return out_pts


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _boundary_wall_cells(
    g: np.ndarray, region_set: set[tuple[int, int]], bg: int, mover: int
) -> list[tuple[int, int]]:
    h, w = g.shape
    boundary = set()
    for r, c in region_set:
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in region_set:
                continue
            v = int(g[nr, nc])
            if v == bg or v == mover:
                continue
            boundary.add((nr, nc))
    return sorted(boundary)


def _nearest_wall_colors(g: np.ndarray, boundary: list[tuple[int, int]], pos: tuple[int, int]) -> set[int]:
    if not boundary:
        return set()
    r0, c0 = pos
    best_dist = min(abs(wr - r0) + abs(wc - c0) for wr, wc in boundary)
    return {int(g[wr, wc]) for wr, wc in boundary if abs(wr - r0) + abs(wc - c0) == best_dist}


def _nearest_wall_color(g: np.ndarray, boundary: list[tuple[int, int]], pos: tuple[int, int]) -> int | None:
    if not boundary:
        return None
    nearest_colors = _nearest_wall_colors(g, boundary, pos)
    if len(nearest_colors) != 1:
        return None
    return next(iter(nearest_colors))


def _simulate_individual_slides(
    region_set: set[tuple[int, int]],
    movers: list[tuple[int, int]],
    move_dirs: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Repeated one-step simulation with a fixed direction per yellow."""
    positions = list(movers)
    max_iters = max(1, len(region_set) * 2)
    for _ in range(max_iters):
        current = set(positions)
        proposals: list[tuple[int, int]] = []
        target_counts: dict[tuple[int, int], int] = {}
        for (r, c), (dr, dc) in zip(positions, move_dirs):
            if not _slide_step_valid(region_set, r, c, dr, dc):
                proposals.append((r, c))
                continue
            nr, nc = r + dr, c + dc
            if (nr, nc) in current:
                proposals.append((r, c))
                continue
            proposals.append((nr, nc))
            target_counts[(nr, nc)] = target_counts.get((nr, nc), 0) + 1

        moved = False
        new_positions = []
        for pos, proposal in zip(positions, proposals):
            if proposal != pos and target_counts.get(proposal, 0) == 1:
                new_positions.append(proposal)
                moved = True
            else:
                new_positions.append(pos)
        positions = new_positions
        if not moved:
            break
    return positions


def _region_border_counts(
    g: np.ndarray, region_set: set[tuple[int, int]], bg: int, mover: int
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for r, c in region_set:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in region_set:
                continue
            if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                continue
            v = int(g[nr, nc])
            if v == bg or v == mover:
                continue
            counts[v] = counts.get(v, 0) + 1
    return counts


def _majority_border_color(border_counts: dict[int, int]) -> int | None:
    if not border_counts:
        return None
    ordered = sorted(border_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ordered) >= 2 and ordered[0][1] == ordered[1][1]:
        return None
    return ordered[0][0]


def _legacy_majority_main(grid: np.ndarray) -> np.ndarray:
    """Original region-majority interpretation, used only to reject degenerate generated samples."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    mover = 4
    dirs = _legend_directions(g, bg)
    out = g.copy()
    passable = (g == bg) | (g == mover)

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        region_set = {(int(r), int(c)) for r, c in pts}
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue
        border_counts = _region_border_counts(g, region_set, bg, mover)
        border_col = _majority_border_color(border_counts)
        if border_col is None or border_col not in dirs:
            continue
        dr, dc = dirs[border_col]
        for r, c in movers:
            out[r, c] = bg
        for r, c in _slide_points(region_set, movers, dr, dc):
            out[r, c] = mover
    return out


def _combo_behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether a generated sample really exhibits the fused shared-wall + local-color behavior."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    mover = 4
    dirs = _legend_directions(g, bg)
    passable = (g == bg) | (g == mover)

    active_regions = 0
    regions_with_two_boundary_colors = 0
    regions_with_both_assigned_colors = 0
    minority_assigned_movers = 0
    ambiguous_movers = 0
    total_movers = 0

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        region_set = {(int(r), int(c)) for r, c in pts}
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        active_regions += 1
        boundary = _boundary_wall_cells(g, region_set, bg, mover)
        border_counts = _region_border_counts(g, region_set, bg, mover)
        majority = _majority_border_color(border_counts)
        if len(border_counts) >= 2:
            regions_with_two_boundary_colors += 1

        assigned = []
        for pos in movers:
            total_movers += 1
            nearest_colors = _nearest_wall_colors(g, boundary, pos)
            if len(nearest_colors) != 1:
                ambiguous_movers += 1
            color = _nearest_wall_color(g, boundary, pos)
            if color is None or color not in dirs:
                continue
            assigned.append(color)
            if majority is not None and color != majority:
                minority_assigned_movers += 1
        if len(set(assigned)) >= 2:
            regions_with_both_assigned_colors += 1

    new_out = main(grid)
    legacy_out = _legacy_majority_main(grid)
    legacy_diff_cells = int(np.sum(new_out != legacy_out))
    return {
        "active_regions": active_regions,
        "regions_with_two_boundary_colors": regions_with_two_boundary_colors,
        "regions_with_both_assigned_colors": regions_with_both_assigned_colors,
        "minority_assigned_movers": minority_assigned_movers,
        "ambiguous_movers": ambiguous_movers,
        "total_movers": total_movers,
        "legacy_diff_cells": legacy_diff_cells,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside shared-wall chambers using legend directions chosen from local wall colors.

    Input structure:
    - Background is the most common color; yellow pixels lie inside enclosed passable regions bounded by colored walls.
    - Two regions may share one straight wall seam whose cells alternate between the two legend-linked wall colors.
    - Border legends are solid rectangles on the outer edge; each legend's fill color matches one wall color and its inner marker arrangement encodes a direction.

    Rule:
    - Find each enclosed passable region as usual, but do not assign one direction to the whole region.
    - For every yellow pixel independently, inspect the nearest boundary wall pixel around its own region.
    - Use that wall pixel's color to select the matching legend direction, then move all yellows with their fixed per-pixel directions until blocked by a wall or another yellow.

    Key insight:
    - The shared seam is not owned uniformly: adjacent yellows can read different seam cells, so the same chamber can contain yellows that follow different legends even though the region boundary still looks mostly like one color from afar.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    mover = 4

    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"

    passable = (g == bg) | (g == mover)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        region_set = {(int(r), int(c)) for r, c in pts}
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        boundary = _boundary_wall_cells(g, region_set, bg, mover)
        active_movers = []
        move_dirs = []
        for pos in movers:
            border_col = _nearest_wall_color(g, boundary, pos)
            if border_col is None or border_col not in dirs:
                continue
            active_movers.append(pos)
            move_dirs.append(dirs[border_col])

        for r, c in active_movers:
            out[r, c] = bg
        for r, c in _simulate_individual_slides(region_set, active_movers, move_dirs):
            out[r, c] = mover

    return out


GRID_H, GRID_W = 20, 20
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
CORNERS = ["TL", "TR", "BL", "BR"]
LEGEND_BUFFER = 1
SHARED_PAIR_RETRIES = 80
GEN_RETRIES = 300
MIN_MOVE_STEPS = 2

VERT_DIRS = [(-1, 0), (1, 0)]
HORZ_DIRS = [(0, -1), (0, 1)]
INNER_SPAN_RANGE = (7, 9)
INNER_SIDE_RANGE = (4, 6)


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


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                out.add((rr, cc))
    return out


def _place_legends(
    rng: random.Random, grid: np.ndarray, legends: list[tuple[int, tuple[int, int]]]
) -> np.ndarray:
    h, w = grid.shape
    legend_occupied = np.zeros((h, w), dtype=bool)
    corners = rng.sample(CORNERS, len(legends))
    for corner, (color, direction) in zip(corners, legends):
        patch = _legend_5x5(color, *direction)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
    return legend_occupied


def _make_shared_pair(
    top: int,
    left: int,
    orientation: str,
    color_a: int,
    color_b: int,
    dir_a: tuple[int, int],
    dir_b: tuple[int, int],
    span: int,
    side_a: int,
    side_b: int,
) -> tuple[list[dict], dict[tuple[int, int], int], dict]:
    wall_colors: dict[tuple[int, int], int] = {}

    if orientation == "vertical":
        seam_c = left + side_a + 1
        bottom = top + span + 1
        right = left + side_a + side_b + 2

        interior_a = {(r, c) for r in range(top + 1, bottom) for c in range(left + 1, seam_c)}
        interior_b = {(r, c) for r in range(top + 1, bottom) for c in range(seam_c + 1, right)}

        for c in range(left, seam_c):
            wall_colors[(top, c)] = color_a
            wall_colors[(bottom, c)] = color_a
        for r in range(top, bottom + 1):
            wall_colors[(r, left)] = color_a

        for c in range(seam_c + 1, right + 1):
            wall_colors[(top, c)] = color_b
            wall_colors[(bottom, c)] = color_b
        for r in range(top, bottom + 1):
            wall_colors[(r, right)] = color_b

        seam_inner = []
        for r in range(top, bottom + 1):
            wall_colors[(r, seam_c)] = color_a if (r - top) % 2 == 0 else color_b
            if top < r < bottom:
                seam_inner.append((r, seam_c))

        region_a = {"color": color_a, "dir": dir_a, "interior": interior_a, "side": "left"}
        region_b = {"color": color_b, "dir": dir_b, "interior": interior_b, "side": "right"}
        meta = {
            "orientation": orientation,
            "top": top,
            "left": left,
            "bottom": bottom,
            "right": right,
            "seam_c": seam_c,
            "seam_inner": seam_inner,
        }
    else:
        seam_r = top + side_a + 1
        bottom = top + side_a + side_b + 2
        right = left + span + 1

        interior_a = {(r, c) for r in range(top + 1, seam_r) for c in range(left + 1, right)}
        interior_b = {(r, c) for r in range(seam_r + 1, bottom) for c in range(left + 1, right)}

        for r in range(top, seam_r):
            wall_colors[(r, left)] = color_a
            wall_colors[(r, right)] = color_a
        for c in range(left, right + 1):
            wall_colors[(top, c)] = color_a

        for r in range(seam_r + 1, bottom + 1):
            wall_colors[(r, left)] = color_b
            wall_colors[(r, right)] = color_b
        for c in range(left, right + 1):
            wall_colors[(bottom, c)] = color_b

        seam_inner = []
        for c in range(left, right + 1):
            wall_colors[(seam_r, c)] = color_a if (c - left) % 2 == 0 else color_b
            if left < c < right:
                seam_inner.append((seam_r, c))

        region_a = {"color": color_a, "dir": dir_a, "interior": interior_a, "side": "top"}
        region_b = {"color": color_b, "dir": dir_b, "interior": interior_b, "side": "bottom"}
        meta = {
            "orientation": orientation,
            "top": top,
            "left": left,
            "bottom": bottom,
            "right": right,
            "seam_r": seam_r,
            "seam_inner": seam_inner,
        }

    return [region_a, region_b], wall_colors, meta


def _shared_pair_majorities_ok(grid: np.ndarray, regions: list[dict]) -> bool:
    for reg in regions:
        border_counts = _region_border_counts(grid, reg["interior"], 0, 4)
        if len(border_counts) < 2:
            return False
        if _majority_border_color(border_counts) != reg["color"]:
            return False
    return True


def _try_place_shared_pair(
    rng: random.Random,
    legend_occupied: np.ndarray,
    color_a: int,
    color_b: int,
    dir_a: tuple[int, int],
    dir_b: tuple[int, int],
    orientation: str,
) -> tuple[list[dict], dict[tuple[int, int], int], dict] | None:
    h, w = legend_occupied.shape
    for _ in range(SHARED_PAIR_RETRIES):
        span = rng.randint(*INNER_SPAN_RANGE)
        side_a = rng.randint(*INNER_SIDE_RANGE)
        side_b = rng.randint(*INNER_SIDE_RANGE)
        if orientation == "vertical":
            shape_h = span + 2
            shape_w = side_a + side_b + 3
        else:
            shape_h = side_a + side_b + 3
            shape_w = span + 2

        if shape_h > h - 4 or shape_w > w - 4:
            continue
        top = rng.randint(1, h - shape_h - 1)
        left = rng.randint(1, w - shape_w - 1)
        regions, wall_colors, meta = _make_shared_pair(
            top, left, orientation, color_a, color_b, dir_a, dir_b, span, side_a, side_b
        )
        footprint = set(wall_colors)
        for reg in regions:
            footprint |= reg["interior"]
        halo = _expand_cells(footprint, h, w, pad=1)
        if any(legend_occupied[r, c] for r, c in halo):
            continue

        check_grid = np.zeros((h, w), dtype=int)
        for (r, c), color in wall_colors.items():
            check_grid[r, c] = color
        if not _shared_pair_majorities_ok(check_grid, regions):
            continue
        return regions, wall_colors, meta
    return None


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
    return len(bounded) == len(region_sets) and all(comp in region_sets for comp in bounded)


def _move_distance(region_set: set[tuple[int, int]], start: tuple[int, int], dr: int, dc: int) -> int:
    r, c = start
    steps = 0
    while _slide_step_valid(region_set, r, c, dr, dc):
        r += dr
        c += dc
        steps += 1
    return steps


def _region_boundary(grid: np.ndarray, reg: dict) -> list[tuple[int, int]]:
    return _boundary_wall_cells(grid, reg["interior"], 0, 4)


def _seam_adjacent_cells(meta: dict, reg: dict) -> list[tuple[int, int]]:
    if meta["orientation"] == "vertical":
        col = meta["seam_c"] - 1 if reg["side"] == "left" else meta["seam_c"] + 1
        return [(r, col) for r, _c in meta["seam_inner"]]
    row = meta["seam_r"] - 1 if reg["side"] == "top" else meta["seam_r"] + 1
    return [(row, c) for _r, c in meta["seam_inner"]]


def _candidate_seam_cells_for_color(
    grid: np.ndarray,
    reg: dict,
    meta: dict,
    dirs: dict[int, tuple[int, int]],
    target_color: int,
) -> list[tuple[int, int]]:
    boundary = _region_boundary(grid, reg)
    out = []
    for cell in _seam_adjacent_cells(meta, reg):
        nearest_colors = _nearest_wall_colors(grid, boundary, cell)
        if nearest_colors != {target_color}:
            continue
        dr, dc = dirs[target_color]
        if _move_distance(reg["interior"], cell, dr, dc) < MIN_MOVE_STEPS:
            continue
        out.append(cell)
    return out


def _try_place_yellows(
    rng: random.Random,
    base_grid: np.ndarray,
    regions: list[dict],
    meta: dict,
    dirs: dict[int, tuple[int, int]],
) -> np.ndarray | None:
    color_a = regions[0]["color"]
    color_b = regions[1]["color"]
    candidates: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for idx, reg in enumerate(regions):
        candidates[(idx, color_a)] = _candidate_seam_cells_for_color(base_grid, reg, meta, dirs, color_a)
        candidates[(idx, color_b)] = _candidate_seam_cells_for_color(base_grid, reg, meta, dirs, color_b)
        if not candidates[(idx, color_a)] or not candidates[(idx, color_b)]:
            return None

    for _ in range(200):
        chosen = []
        for idx in range(2):
            cell_a = rng.choice(candidates[(idx, color_a)])
            cell_b = rng.choice(candidates[(idx, color_b)])
            if cell_a == cell_b:
                break
            chosen.extend([cell_a, cell_b])
        else:
            if len(set(chosen)) != 4:
                continue
            input_grid = base_grid.copy()
            for r, c in chosen:
                input_grid[r, c] = 4
            if _oracle_check(input_grid, regions):
                return input_grid
    return None


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    try:
        out = main(input_grid)
    except Exception:
        return False

    struct_mask = (input_grid != 4) & (input_grid != 0)
    if not np.array_equal(out[struct_mask], input_grid[struct_mask]):
        return False
    if int((out == 4).sum()) != int((input_grid == 4).sum()):
        return False

    region_mask = np.zeros_like(input_grid, dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            region_mask[r, c] = True
    if not bool(np.all(region_mask[out == 4])):
        return False

    metrics = _combo_behavior_metrics(input_grid)
    if metrics["active_regions"] != 2:
        return False
    if metrics["regions_with_two_boundary_colors"] != 2:
        return False
    if metrics["regions_with_both_assigned_colors"] != 2:
        return False
    if metrics["minority_assigned_movers"] < 2:
        return False
    if metrics["ambiguous_movers"] != 0:
        return False
    if metrics["legacy_diff_cells"] == 0:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    orientation = rng.choice(["vertical", "horizontal"])
    dir_pair = list(VERT_DIRS if orientation == "vertical" else HORZ_DIRS)
    rng.shuffle(dir_pair)

    color_a, color_b = rng.sample(WALL_COLOR_CHOICES, 2)
    legends = [(color_a, dir_pair[0]), (color_b, dir_pair[1])]
    legend_occupied = _place_legends(rng, grid, legends)

    grown = _try_place_shared_pair(
        rng, legend_occupied, color_a, color_b, dir_pair[0], dir_pair[1], orientation
    )
    if grown is None:
        return None
    regions, wall_colors, meta = grown

    for (r, c), color in wall_colors.items():
        grid[r, c] = color
    if not _validate_topology(grid, regions):
        return None

    dirs = {color: direction for color, direction in legends}
    return _try_place_yellows(rng, grid, regions, meta, dirs)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
