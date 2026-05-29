# Combo variant of 88e364bc (index 32)
# combo_spec: D6/1+D16/2
# Name: 宽通道约束+L/T形区域

"""
design_analysis:
  fused_goal:
    keep:
      - border legends map wall color -> motion direction
      - yellow cells slide monotonically and stack from the far end inward
      - enclosed passable components are still the only movable regions
    change:
      - regions are explicit non-convex L/T silhouettes instead of generic blobs
      - step legality now uses a width rule, so a diagonal move needs both the old corner-cut
        square and a parallel support lane on one side

  a_concretization_and_trace:
    appearance:
      - legends remain 6x7 filled rectangles at grid corners and encode only diagonal directions
      - each wall color surrounds one L or T shaped interior made from unions of axis-aligned
        rectangles, leaving visible concave elbows
      - walls are still the 4-neighbor ring around the interior, so the region stays fully enclosed
    trace_example:
      - consider a down-right legend and an L region whose upper arm meets a lower arm at a tight
        elbow
      - original task:
        - if the destination diagonal cell plus the two orthogonal bridge cells exist, the yellow
          keeps going through the elbow
      - fused task:
        - the same diagonal step also needs a same-side parallel support pair, i.e. a second
          diagonal lane beside the move
        - at a narrow elbow the bridge square exists but the parallel lane disappears, so the
          yellow stops one step before the corner instead of reaching the far outer wall
      - visible difference:
        - two yellows under the same arrow can end at different depths: one reaches the remote
          boundary, one piles up at the inner elbow because width, not occupancy, blocks it

  b_complexity_calibration:
    controls:
      - legend count: 1-2
      - directions: diagonal only
      - region count: one region per legend color
      - region shapes: L or T only
      - yellows: 1-2 per region, with at least one width-sensitive target in every sample
    why_this_is_not_too_simple:
      - L/T geometry creates repeated concave evidence instead of one arbitrary notch
      - diagonal-only legends keep the new width rule central rather than mixing two unrelated
        movement regimes
      - forcing at least one width-blocked target prevents degeneration into the original behavior

  c_implicit_assumption_audit_of_main:
    - _legend_directions assumes border-touching solid rectangles with exactly 3 colors.
      Generator preserves that exactly by reusing the original diagonal legend template.
    - connected passable components are treated as independent regions.
      Generator surrounds every intended interior by a single wall ring and buffers structures so
      no accidental merges occur.
    - border color is inferred by majority adjacency.
      Each region has exactly one wall color, so the majority vote remains unambiguous.
    - sliding order is far-to-near along the legend vector.
      Generator reverse-constructs inputs against that same order.
    - the new _combo_slide_step_valid assumes width can be read from local support cells.
      Generator explicitly builds elbows where the old bridge square exists but the support lane
      vanishes, so this assumption is both meaningful and observable.

  d_placement_legality:
    - legends reserve a 1-cell moat
    - every candidate region is generated as an interior mask first, then converted to wall cells
    - interior, walls, and a 4-neighbor safety buffer must fit strictly inside the 20x20 board
    - topology is validated after painting by relabeling bounded background components and checking
      they equal the intended interiors exactly

  e_visual_observability:
    - each generated task contains at least one region where the fused solver stops at a cell from
      which the original rule would still permit another diagonal step
    - the blocking reason is visible in the elbow geometry, not in hidden color ordering or counts
    - paired yellows in the same region often show both behaviors simultaneously: far-wall stacking
      versus elbow stopping

  f_cleanliness:
    - one region per legend color keeps the legend-to-wall mapping readable
    - only standard ARC colors are used: bg=0, center=1, indicator=2, yellow=4, walls in
      {3,5,6,7,8,9}
    - region buffers prevent walls and legends from touching in confusing ways

  g_change_inferability:
    - colors do not encode any extra state beyond wall-color matching
    - all interesting variation comes from visible shape and channel width
    - the generator rejects samples whose fused output would match the baseline original solver, so
      the training examples actually exhibit the combo behavior

  solver_generator_contract_after_change:
    - solver expects at least one detectable legend and diagonal directions only in generated data
    - solver expects every movable region to be a bounded passable component with one dominant wall
      color
    - generator guarantees at least one width-sensitive terminal whose forward step is legal under
      the baseline rule but illegal under the combo rule
"""

from __future__ import annotations

import random
from typing import Callable, Iterable

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1
GEN_RETRIES = 500
REGION_RETRIES = 180
MAX_SLIDE_STEPS = 6


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """Detect legend blocks on the border and decode their directions."""
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
        if len(others) < 2:
            continue
        others.sort()
        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            tied = [others[0][1], others[1][1]]
            ranked = []
            for col in tied:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                ranked.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            ranked.sort()
            center_col = ranked[0][1]
            indicator_col = ranked[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[-1][1]

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


def _projection(p: tuple[int, int], dr: int, dc: int) -> int:
    return dr * p[0] + dc * p[1]


def _cardinal_width_ok(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    for sr, sc in ((-dc, dr), (dc, -dr)):
        if (r + sr, c + sc) in region_set and (r + dr + sr, c + dc + sc) in region_set:
            return True
    return False


def _diagonal_width_ok(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
        return False
    for sr, sc in ((-dc, dr), (dc, -dr)):
        if (r + sr, c + sc) in region_set and (r + dr + sr, c + dc + sc) in region_set:
            return True
    return False


def _baseline_slide_step_valid(
    region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int
) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        return (r + dr, c) in region_set and (r, c + dc) in region_set
    return True


def _combo_slide_step_valid(
    region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int
) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        return _diagonal_width_ok(region_set, r, c, dr, dc)
    return _cardinal_width_ok(region_set, r, c, dr, dc)


def _slide_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
    step_valid: Callable[[set[tuple[int, int]], int, int, int, int], bool],
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out_pts = []
    for r0, c0 in points_sorted:
        r, c = r0, c0
        while True:
            if not step_valid(region_set, r, c, dr, dc):
                break
            nr, nc = r + dr, c + dc
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        out_pts.append((r, c))
    return out_pts


def _solve_with_step_rule(
    grid: np.ndarray,
    step_valid: Callable[[set[tuple[int, int]], int, int, int, int], bool],
) -> np.ndarray:
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

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        border_counts: dict[int, int] = {}
        pset = {(int(r), int(c)) for r, c in pts}
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
        if border_col not in dirs:
            continue
        dr, dc = dirs[border_col]

        for r, c in movers:
            out[r, c] = bg
        for r, c in _slide_points(pset, movers, dr, dc, step_valid):
            out[r, c] = mover
    return out


def _baseline_main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_step_rule(grid, _baseline_slide_step_valid)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside enclosed regions using border legends, but treat motion as needing
    a visibly wide path rather than a point-sized path.

    Input structure:
    - Background is color 0; yellow movers (4) sit inside colored wall rings.
    - Each wall color has a matching legend rectangle on the outer border, and these legends in
      this combo encode diagonal directions.
    - The enclosed interiors are non-convex L/T shapes, so they contain elbows and concave stops.

    Rule:
    - Find each enclosed passable region and match it to the legend whose fill color equals that
      region's wall color.
    - Move all yellows in the legend direction, processing from far to near so they stack.
    - A step is legal only when the destination is inside the region and the path is at least two
      cells wide: cardinal moves need one supporting 2x2 window, while diagonal moves need both
      the usual bridge square and an extra parallel support lane on one side.

    Key insight:
    - The wall color still chooses the arrow, but the final resting place now depends on both the
      non-convex L/T geometry and local channel width. A yellow can stop before an inner corner
      even though the original point-sized diagonal rule would have let it continue.
    """
    return _solve_with_step_rule(grid, _combo_slide_step_valid)


def _legend_6x7(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((6, 7), fill, dtype=int)
    g[1, 3] = 1
    g[2, 1] = 1
    g[2, 4] = 1
    g[3, 2] = 1
    g[3, 5] = 2
    g[4, 3] = 2
    if (dr, dc) == (1, 1):
        return g
    if (dr, dc) == (-1, 1):
        return np.flipud(g)
    if (dr, dc) == (1, -1):
        return np.fliplr(g)
    if (dr, dc) == (-1, -1):
        return np.rot90(g, 2)
    raise ValueError((dr, dc))


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


def _place_legends(rng: random.Random, grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    colors = rng.sample(WALL_COLOR_CHOICES, n_legends)
    width_idx = rng.randrange(n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []
    for idx, (corner, color) in enumerate(zip(corners, colors)):
        dr, dc = rng.choice(DIAGONAL_DIRS)
        patch = _legend_6x7(color, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo = max(0, r0 - LEGEND_BUFFER)
        r_hi = min(h, r0 + ph + LEGEND_BUFFER)
        c_lo = max(0, c0 - LEGEND_BUFFER)
        c_hi = min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append(
            {
                "color": color,
                "dir": (dr, dc),
                "require_width_stop": idx == width_idx,
            }
        )
    return legend_occupied, legends


def _blob_wall_cells(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    wall = set()
    for r, c in blob:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in blob:
                wall.add((nr, nc))
    return wall


def _reserve_with_buffer(mask: np.ndarray, coords: Iterable[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in coords:
        mask[r, c] = True
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                mask[nr, nc] = True


def _make_l_right(h: int, w: int, stem_w: int, bar_h: int, top_bar: bool) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :stem_w] = True
    if top_bar:
        mask[:bar_h, :] = True
    else:
        mask[h - bar_h :, :] = True
    return mask


def _make_t_right(h: int, w: int, stem_w: int, bar_h: int, bar_r: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :stem_w] = True
    mask[bar_r : bar_r + bar_h, :] = True
    return mask


def _sample_region_mask(rng: random.Random) -> tuple[np.ndarray, str] | None:
    shape_type = rng.choice(["L", "T"])
    if shape_type == "L":
        h = rng.randint(7, 11)
        w = rng.randint(7, 11)
        stem_w = rng.randint(2, min(4, w - 2))
        bar_h = rng.randint(2, min(4, h - 2))
        base = _make_l_right(h, w, stem_w, bar_h, bool(rng.randrange(2)))
    else:
        h = rng.randint(8, 11)
        w = rng.randint(8, 11)
        stem_w = rng.randint(2, min(4, w - 2))
        bar_h = rng.randint(2, min(3, h - 2))
        max_bar_r = h - bar_h - 1
        if max_bar_r <= 1:
            return None
        bar_r = rng.randint(1, max_bar_r - 1)
        base = _make_t_right(h, w, stem_w, bar_h, bar_r)
    rot_k = rng.randrange(4)
    mask = np.rot90(base, rot_k)
    if not (12 <= int(mask.sum()) <= 97):
        return None
    return mask, shape_type


def _shift_points(points: Iterable[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in points]


def _classify_targets(
    blob: set[tuple[int, int]],
    direction: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    dr, dc = direction
    terminals = []
    width_blocked = []
    for p in blob:
        combo_ok = _combo_slide_step_valid(blob, p[0], p[1], dr, dc)
        base_ok = _baseline_slide_step_valid(blob, p[0], p[1], dr, dc)
        if not combo_ok:
            terminals.append(p)
            if base_ok:
                width_blocked.append(p)
    max_proj = max(_projection(p, dr, dc) for p in blob)
    far = [p for p in terminals if _projection(p, dr, dc) == max_proj]
    return width_blocked, far, terminals


def _try_place_region(rng: random.Random, occupied: np.ndarray, spec: dict) -> dict | None:
    h, w = occupied.shape
    direction = spec["dir"]
    need_width = spec["require_width_stop"]

    for _ in range(REGION_RETRIES):
        sampled = _sample_region_mask(rng)
        if sampled is None:
            continue
        mask, shape_type = sampled
        rh, rw = mask.shape
        if rh > h - 4 or rw > w - 4:
            continue

        row_choices = list(range(2, h - rh - 1))
        col_choices = list(range(2, w - rw - 1))
        rng.shuffle(row_choices)
        rng.shuffle(col_choices)

        for r0 in row_choices[:20]:
            for c0 in col_choices[:20]:
                interior = {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)}
                walls = _blob_wall_cells(interior, h, w)
                if not walls:
                    continue
                if any(
                    rr in (0, h - 1) or cc in (0, w - 1) or occupied[rr, cc]
                    for rr, cc in interior | walls
                ):
                    continue

                reserve = set(interior) | walls
                reserve_with_buffer = set(reserve)
                for rr, cc in reserve:
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w:
                            reserve_with_buffer.add((nr, nc))
                if any(occupied[rr, cc] for rr, cc in reserve_with_buffer):
                    continue

                width_targets, far_targets, terminals = _classify_targets(interior, direction)
                if not far_targets:
                    continue
                if need_width and not width_targets:
                    continue

                return {
                    "color": spec["color"],
                    "dir": direction,
                    "shape": shape_type,
                    "interior": interior,
                    "walls": walls,
                    "require_width_stop": need_width,
                    "width_targets": width_targets,
                    "far_targets": far_targets,
                    "terminals": terminals,
                }
    return None


def _build_regions(rng: random.Random, legends: list[dict], legend_occupied: np.ndarray) -> list[dict] | None:
    occupied = legend_occupied.copy()
    regions = []
    for legend in legends:
        placed = _try_place_region(rng, occupied, legend)
        if placed is None:
            return None
        _reserve_with_buffer(occupied, placed["interior"] | placed["walls"])
        regions.append(placed)
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
    return len(bounded) == len(expected) and all(comp in expected for comp in bounded)


def _forward_slide(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
    step_valid: Callable[[set[tuple[int, int]], int, int, int, int], bool],
) -> tuple[int, int]:
    r, c = start
    while True:
        if not step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
    return r, c


def _reverse_start_candidates(
    blob: set[tuple[int, int]],
    target: tuple[int, int],
    dr: int,
    dc: int,
    blocked_outputs: set[tuple[int, int]],
    used_inputs: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    cands = []
    for start in blob:
        if start == target or start in blocked_outputs or start in used_inputs:
            continue
        out = _forward_slide(blob, start, dr, dc, blocked_outputs, _combo_slide_step_valid)
        if out != target:
            continue
        dist = abs(target[0] - start[0]) + abs(target[1] - start[1])
        if 1 <= dist <= MAX_SLIDE_STEPS:
            cands.append(start)
    return cands


def _realize_targets(
    rng: random.Random,
    blob: set[tuple[int, int]],
    direction: tuple[int, int],
    targets: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    dr, dc = direction
    targets_sorted = sorted(targets, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
    blocked_outputs: set[tuple[int, int]] = set()
    used_inputs: set[tuple[int, int]] = set()
    inputs = []
    for target in targets_sorted:
        cands = _reverse_start_candidates(blob, target, dr, dc, blocked_outputs, used_inputs)
        if not cands:
            return None
        start = rng.choice(cands)
        inputs.append(start)
        used_inputs.add(start)
        blocked_outputs.add(target)
    return inputs


def _choose_regular_target_options(rng: random.Random, reg: dict) -> list[list[tuple[int, int]]]:
    blob = reg["interior"]
    dr, dc = reg["dir"]
    far_targets = reg["far_targets"][:]
    terminals = reg["terminals"][:]
    rng.shuffle(far_targets)
    rng.shuffle(terminals)

    options: list[list[tuple[int, int]]] = []
    if far_targets:
        options.append([far_targets[0]])
    if len(far_targets) >= 2:
        options.append([far_targets[0], far_targets[1]])
    else:
        behind = []
        for p in blob:
            if _projection(p, dr, dc) == _projection(far_targets[0], dr, dc) - 1:
                behind.append(p)
        rng.shuffle(behind)
        if far_targets and behind:
            options.append([far_targets[0], behind[0]])
    if terminals:
        options.append([terminals[0]])
    return options


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    width_seen = False
    baseline_diff_seen = False

    for reg in regions:
        blob = reg["interior"]
        dr, dc = reg["dir"]
        width_targets = reg["width_targets"][:]
        far_targets = reg["far_targets"][:]
        rng.shuffle(width_targets)
        rng.shuffle(far_targets)

        target_options: list[list[tuple[int, int]]] = []
        if reg["require_width_stop"]:
            width_seen = True
            for width_target in width_targets[:8]:
                if far_targets:
                    for far_target in far_targets[:4]:
                        if far_target != width_target:
                            target_options.append([far_target, width_target])
                target_options.append([width_target])
        else:
            target_options.extend(_choose_regular_target_options(rng, reg))

        success = None
        for targets in target_options:
            starts = _realize_targets(rng, blob, (dr, dc), targets)
            if starts is None:
                continue
            success = (targets, starts)
            break
        if success is None:
            return False

        targets, starts = success
        for start, target in zip(starts, sorted(targets, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)):
            baseline_target = _forward_slide(blob, start, dr, dc, set(), _baseline_slide_step_valid)
            if baseline_target != target:
                baseline_diff_seen = True
        for r, c in starts:
            grid[r, c] = 4

    return width_seen and baseline_diff_seen


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
    return not np.array_equal(solver_out, _baseline_main(input_grid))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    legend_occupied, legends = _place_legends(rng, grid)
    regions = _build_regions(rng, legends, legend_occupied)
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
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exceeded retries")
