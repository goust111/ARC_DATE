# Combo variant of 88e364bc (index 11)
# combo_spec: D6/1+D7/2
# Name: L/T形区域+共享墙壁

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


"""
design_analysis:
  a_specific_concretization_and_trace:
    original:
      - each enclosed passable component behaves as one sliding chamber
      - border color is read from adjacent wall cells, then matched to a corner legend
      - yellows slide monotonically until the next step leaves the chamber or hits an earlier yellow
    fused_variant:
      - there are exactly two enclosed chambers, and they are a mandatory shared pair
      - each chamber is a non-convex L or side-T silhouette, so the chamber itself contains a visible
        inner notch
      - the two chambers share one straight 1-cell wall segment; that segment alternates the two wall
        colors cell-by-cell
      - each chamber still belongs to one legend, but ownership is now the unique majority color over
        the chamber's whole 4-neighbor boundary
    trace_example:
      - in canonical orientation, the left chamber slides left and the right chamber slides right,
        so both move away from the shared seam
      - an L-top chamber has a long seam-side stem and a wide top bar extending outward
      - a yellow inside the top bar reaches the outer far wall, but a yellow in a lower stem row
        stops at the concave inner corner before that outer wall
      - meanwhile the shared seam contributes both colors to both regions' boundary counts, so the
        solver must aggregate the whole perimeter instead of assuming one uniform wall color
    visible_difference:
      - compared with the original task, a sample now simultaneously shows mixed-color ownership cues
        and non-global terminal positions caused by the notch geometry

  b_complexity_calibration:
    retained:
      - legend decoding from border rectangles
      - 1-cell wall thickness
      - monotone yellow sliding with stacking
    intentionally_restricted:
      - exactly two legends / two regions, because the shared-wall cue already adds ambiguity
      - cardinal directions only, so the new difficulty is dominated by concave geometry + ownership
      - a small region family: mirrored L-top, mirrored L-bottom, or mirrored side-T
    why_this_is_not_too_simple:
      - every sample has both mandatory features, not optional ones
      - at least one region receives a yellow whose output is a concave trap terminal rather than the
        globally farthest projection tier
      - both regions see both wall colors on their boundary, and the shared seam is long enough to
        contribute several minority-color boundary counts to each region

  c_implicit_assumption_audit_of_main:
    - _legend_directions assumes legends are border-touching solid rectangles with exactly 3 colors.
      The generator preserves that unchanged.
    - components_mask(passable) assumes each intended chamber is one bounded 4-connected bg/yellow
      component. The generator constructs exactly two such components and validates topology.
    - the original max(border_counts, key=...) silently assumed a single dominant color and would be
      underspecified in ties. The fused solver replaces that with an explicit strict-majority helper.
    - _slide_points assumes the chamber itself fully determines stopping behavior. This still holds:
      the L/T notch is represented directly in the passable mask, so no special-case path code is
      needed at the corner.
    - the solver now assumes that every active region has a unique majority boundary color that also
      has a matching legend. The generator checks both conditions before accepting a sample.

  d_placement_legality:
    - legends are placed first in two distinct corners and reserve a 1-cell halo
    - the shared pair is built in a padded local canvas, rotated as a whole, then translated so the
      full footprint stays strictly inside the 20x20 grid
    - walls are computed from 4-neighbor adjacency to the two interiors; intersection cells become
      the alternating shared seam
    - a further 1-cell halo around the whole pair prevents legend overlap and keeps the picture clean
    - topology is validated by checking that bounded background components match the generated
      interiors exactly

  e_visual_observability:
    - the alternating seam is straight, contiguous, and length >= 4, so the shared-wall cue is easy
      to spot from a few examples
    - the notch-stop cue is also explicit because one sample region is forced to realize a trap
      terminal with a yellow output that is not on the chamber's max-projection tier
    - directions are tied to visible legend patterns, not hidden color ordering

  f_visual_cleanliness:
    - only one shared pair is drawn per grid
    - both chambers use large simple silhouettes rather than noisy random blobs
    - there is no extra region clutter, so the seam and the notch remain easy to parse

  g_inference_of_change:
    - colors still only select legends; no new color remapping is introduced
    - the new reasoning burden is structural: mixed wall ownership plus concave stopping
    - repeated evidence is provided because both chambers touch the shared seam and at least one
      yellow demonstrates the concave-stop behavior

  h_quantitative_behavior_targets:
    - every accepted sample must satisfy:
        - both regions have boundary counts containing both wall colors
        - the majority color of each region equals its assigned legend color
        - the shared seam length is at least 4
        - at least one output yellow lands on a trap terminal with projection < max projection of
          its region under that region's slide direction

  i_solver_generator_contract_after_change:
    - solver assumes exactly the majority-color ownership rule and ordinary monotone sliding
    - generator guarantees strict majority, valid legends, enclosed chambers, and realizable
      reverse-slide starts for all planted yellows
"""


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """Detect border legends and map legend fill color -> slide direction."""
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
            tied = [others[0][1], others[1][1]]
            dist_rank = []
            for col in tied:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                dist_rank.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            dist_rank.sort()
            center_col = dist_rank[0][1]
            indicator_col = dist_rank[1][1]
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


def _slide_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
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


def _region_border_counts(
    g: np.ndarray,
    region_set: set[tuple[int, int]],
    bg: int,
    mover: int,
) -> dict[int, int]:
    h, w = g.shape
    border_counts: dict[int, int] = {}
    for r, c in region_set:
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in region_set:
                continue
            v = int(g[nr, nc])
            if v == bg or v == mover:
                continue
            border_counts[v] = border_counts.get(v, 0) + 1
    return border_counts


def _majority_border_color(border_counts: dict[int, int]) -> int | None:
    if not border_counts:
        return None
    ordered = sorted(border_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ordered) >= 2 and ordered[0][1] == ordered[1][1]:
        return None
    return ordered[0][0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside non-convex enclosed chambers using border legends.

    Input structure:
    - Border-touching legend rectangles still encode directions by center-vs-indicator placement.
    - The movable yellow pixels live inside two enclosed L/T-shaped chambers made from colored
      1-cell walls.
    - The two chambers share one straight alternating-color wall segment, so a chamber can see both
      wall colors along its perimeter.

    Rule:
    - For each bounded passable chamber, count all non-bg wall colors on its 4-neighbor boundary and
      take the unique majority color as that chamber's owner.
    - Use the legend whose fill color matches that majority wall color to obtain the slide direction.
    - Slide every yellow step-by-step until the next move would leave the chamber, hit a concave
      notch, or collide with an already-stopped yellow.

    Key insight:
    - The shared seam means a chamber no longer has a uniformly colored boundary, so ownership must
      come from the full-boundary majority rather than a local wall patch.
    - Because the chambers are L/T-shaped, some yellows stop at an inner corner before reaching the
      globally farthest edge in the movement direction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    mover = 4

    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend"

    passable = (g == bg) | (g == mover)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        region_set = {(int(r), int(c)) for r, c in pts}
        movers = [(r, c) for r, c in region_set if int(g[r, c]) == mover]
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


GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1

GEN_RETRIES = 400
PLACEMENT_RETRIES = 80
MAX_SLIDE_STEPS = 6


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


def _rotate_dir(dr: int, dc: int, k: int) -> tuple[int, int]:
    for _ in range(k % 4):
        dr, dc = -dc, dr
    return dr, dc


def _rotate_coords(
    cells: set[tuple[int, int]],
    h: int,
    w: int,
    k: int,
) -> tuple[set[tuple[int, int]], int, int]:
    out = set(cells)
    hh, ww = h, w
    for _ in range(k % 4):
        out = {(ww - 1 - c, r) for r, c in out}
        hh, ww = ww, hh
    return out, hh, ww


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                out.add((rr, cc))
    return out


def _make_l_right(h: int, w: int, stem_w: int, bar_h: int, top_bar: bool) -> set[tuple[int, int]]:
    cells = {(r, c) for r in range(h) for c in range(stem_w)}
    rows = range(bar_h) if top_bar else range(h - bar_h, h)
    cells |= {(r, c) for r in rows for c in range(w)}
    return cells


def _make_t_right(h: int, w: int, stem_w: int, bar_h: int, bar_r: int) -> set[tuple[int, int]]:
    cells = {(r, c) for r in range(h) for c in range(stem_w)}
    cells |= {(r, c) for r in range(bar_r, bar_r + bar_h) for c in range(w)}
    return cells


def _mirror_h(cells: set[tuple[int, int]], w: int) -> set[tuple[int, int]]:
    return {(r, w - 1 - c) for r, c in cells}


def _projection(p: tuple[int, int], dr: int, dc: int) -> int:
    return dr * p[0] + dc * p[1]


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _forward_slide(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
) -> tuple[int, int]:
    r, c = start
    while True:
        if not _slide_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
    return (r, c)


def _boundary_wall_map(
    interior_a: set[tuple[int, int]],
    interior_b: set[tuple[int, int]],
    color_a: int,
    color_b: int,
) -> tuple[dict[tuple[int, int], int], list[tuple[int, int]]] | tuple[None, None]:
    interiors = [interior_a, interior_b]
    adjacency: dict[tuple[int, int], set[int]] = {}
    occupied = interior_a | interior_b
    for idx, blob in enumerate(interiors):
        for r, c in blob:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in occupied:
                    continue
                adjacency.setdefault((nr, nc), set()).add(idx)

    wall_map: dict[tuple[int, int], int] = {}
    shared = []
    for cell, owners in adjacency.items():
        if owners == {0}:
            wall_map[cell] = color_a
        elif owners == {1}:
            wall_map[cell] = color_b
        elif owners == {0, 1}:
            r, c = cell
            wall_map[cell] = color_a if (r + c) % 2 == 0 else color_b
            shared.append(cell)
        else:
            return None, None

    if len(shared) < 4:
        return None, None
    rows = {r for r, _ in shared}
    cols = {c for _, c in shared}
    if len(rows) != 1 and len(cols) != 1:
        return None, None
    ordered = sorted(shared)
    if len(cols) == 1:
        ordered = sorted(shared, key=lambda p: p[0])
        if any(ordered[i][0] + 1 != ordered[i + 1][0] for i in range(len(ordered) - 1)):
            return None, None
    else:
        ordered = sorted(shared, key=lambda p: p[1])
        if any(ordered[i][1] + 1 != ordered[i + 1][1] for i in range(len(ordered) - 1)):
            return None, None
    return wall_map, ordered


def _shared_pair_majorities_ok(
    wall_map: dict[tuple[int, int], int],
    regions: list[dict],
    h: int,
    w: int,
) -> bool:
    grid = np.zeros((h, w), dtype=int)
    for (r, c), color in wall_map.items():
        grid[r, c] = color
    for reg in regions:
        counts = _region_border_counts(grid, reg["interior"], 0, 4)
        majority = _majority_border_color(counts)
        if majority != reg["color"]:
            return False
        if len(counts) < 2:
            return False
    vals = [int(grid[r, c]) for r, c in regions[0]["shared_segment"]]
    if any(vals[i] == vals[i + 1] for i in range(len(vals) - 1)):
        return False
    return True


def _build_local_shared_pair(
    rng: random.Random,
    color_a: int,
    color_b: int,
    rot_k: int,
) -> tuple[list[dict], dict[tuple[int, int], int], int, int] | None:
    h_core = rng.randint(8, 10)
    w_left = rng.randint(6, 7)
    w_right = rng.randint(6, 7)
    stem_left = rng.randint(2, 3)
    stem_right = rng.randint(2, 3)
    family = rng.choice(["L_top", "L_bottom", "T_mid"])

    bar_h = rng.randint(4, 5)
    if family == "L_top":
        left = _mirror_h(_make_l_right(h_core, w_left, stem_left, bar_h, True), w_left)
        right = _make_l_right(h_core, w_right, stem_right, bar_h, True)
    elif family == "L_bottom":
        left = _mirror_h(_make_l_right(h_core, w_left, stem_left, bar_h, False), w_left)
        right = _make_l_right(h_core, w_right, stem_right, bar_h, False)
    else:
        max_bar_r = h_core - bar_h - 2
        if max_bar_r < 2:
            return None
        bar_r = rng.randint(2, max_bar_r)
        left = _mirror_h(_make_t_right(h_core, w_left, stem_left, bar_h, bar_r), w_left)
        right = _make_t_right(h_core, w_right, stem_right, bar_h, bar_r)

    total_h = h_core + 2
    total_w = w_left + 1 + w_right + 2
    left_shift = {(r + 1, c + 1) for r, c in left}
    right_shift = {(r + 1, c + 1 + w_left + 1) for r, c in right}

    left_rot, rh, rw = _rotate_coords(left_shift, total_h, total_w, rot_k)
    right_rot, rh2, rw2 = _rotate_coords(right_shift, total_h, total_w, rot_k)
    assert rh == rh2 and rw == rw2

    wall_map, shared_segment = _boundary_wall_map(left_rot, right_rot, color_a, color_b)
    if wall_map is None or shared_segment is None:
        return None

    dir_a = _rotate_dir(0, -1, rot_k)
    dir_b = _rotate_dir(0, 1, rot_k)
    regions = [
        {
            "color": color_a,
            "dir": dir_a,
            "interior": left_rot,
            "shape_family": family,
            "shared_segment": shared_segment,
        },
        {
            "color": color_b,
            "dir": dir_b,
            "interior": right_rot,
            "shape_family": family,
            "shared_segment": shared_segment,
        },
    ]
    if not _shared_pair_majorities_ok(wall_map, regions, rh, rw):
        return None
    return regions, wall_map, rh, rw


def _place_legends(
    rng: random.Random,
    grid: np.ndarray,
    legends: list[tuple[int, tuple[int, int]]],
) -> np.ndarray:
    h, w = grid.shape
    corners = rng.sample(CORNERS, len(legends))
    legend_occupied = np.zeros((h, w), dtype=bool)
    for corner, (color, direction) in zip(corners, legends):
        dr, dc = direction
        patch = _legend_5x5(color, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
    return legend_occupied


def _place_shared_pair(
    rng: random.Random,
    legend_occupied: np.ndarray,
    local_regions: list[dict],
    local_wall_map: dict[tuple[int, int], int],
    pair_h: int,
    pair_w: int,
) -> tuple[list[dict], dict[tuple[int, int], int]] | None:
    h, w = legend_occupied.shape
    footprint = set(local_wall_map)
    for reg in local_regions:
        footprint |= reg["interior"]
    halo = _expand_cells(footprint, pair_h, pair_w, pad=1)

    if pair_h > h - 2 or pair_w > w - 2:
        return None
    for _ in range(PLACEMENT_RETRIES):
        r0 = rng.randint(1, h - pair_h - 1)
        c0 = rng.randint(1, w - pair_w - 1)
        if any(legend_occupied[r0 + r, c0 + c] for r, c in halo):
            continue

        regions = []
        for reg in local_regions:
            regions.append(
                {
                    "color": reg["color"],
                    "dir": reg["dir"],
                    "interior": {(r0 + r, c0 + c) for r, c in reg["interior"]},
                    "shape_family": reg["shape_family"],
                    "shared_segment": [(r0 + r, c0 + c) for r, c in reg["shared_segment"]],
                }
            )
        wall_map = {(r0 + r, c0 + c): color for (r, c), color in local_wall_map.items()}
        return regions, wall_map
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
    expected = [reg["interior"] for reg in regions]
    return len(bounded) == len(expected) and all(comp in expected for comp in bounded)


def _terminal_targets(
    blob: set[tuple[int, int]],
    direction: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    dr, dc = direction
    terminals = [p for p in blob if not _slide_step_valid(blob, p[0], p[1], dr, dc)]
    max_proj = max(_projection(p, dr, dc) for p in blob)
    far = [p for p in terminals if _projection(p, dr, dc) == max_proj]
    trap = [p for p in terminals if _projection(p, dr, dc) < max_proj]
    return far, trap


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
        out = _forward_slide(blob, start, dr, dc, blocked_outputs)
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


def _target_options(
    rng: random.Random,
    blob: set[tuple[int, int]],
    direction: tuple[int, int],
    require_trap: bool,
) -> list[list[tuple[int, int]]]:
    far, trap = _terminal_targets(blob, direction)
    rng.shuffle(far)
    rng.shuffle(trap)
    options: list[list[tuple[int, int]]] = []

    if require_trap:
        for t in trap[:6]:
            if far:
                for f in far[:4]:
                    if f != t:
                        options.append([f, t])
            options.append([t])
        if len(trap) >= 2:
            options.append([trap[0], trap[1]])
    else:
        for f in far[:4]:
            options.append([f])
        for t in trap[:3]:
            options.append([t])
        if len(far) >= 2:
            options.append([far[0], far[1]])
        if far and trap:
            options.append([far[0], trap[0]])
    return options


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    anchor_idx = rng.randrange(len(regions))
    trap_realized = False
    total = 0

    for idx, reg in enumerate(regions):
        require_trap = idx == anchor_idx
        options = _target_options(rng, reg["interior"], reg["dir"], require_trap)
        success = None
        for targets in options:
            starts = _realize_targets(rng, reg["interior"], reg["dir"], targets)
            if starts is None:
                continue
            success = (targets, starts)
            break
        if success is None:
            return False

        targets, starts = success
        dr, dc = reg["dir"]
        max_proj = max(_projection(p, dr, dc) for p in reg["interior"])
        if any(_projection(p, dr, dc) < max_proj for p in targets):
            trap_realized = True
        for r, c in starts:
            grid[r, c] = 4
        total += len(starts)

    return trap_realized and total >= 2


def _oracle_check(input_grid: np.ndarray, regions: list[dict], wall_map: dict[tuple[int, int], int]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    struct_mask = (input_grid != 4) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if int((solver_out == 4).sum()) != int((input_grid == 4).sum()):
        return False

    region_mask = np.zeros_like(input_grid, dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            region_mask[r, c] = True
        counts = _region_border_counts(input_grid, reg["interior"], 0, 4)
        majority = _majority_border_color(counts)
        if majority != reg["color"]:
            return False
        if len(counts) < 2:
            return False
    if not bool(np.all(region_mask[solver_out == 4])):
        return False

    shared_regions = [regions[0], regions[1]]
    if not _shared_pair_majorities_ok(wall_map, shared_regions, *input_grid.shape):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    color_a, color_b = rng.sample(WALL_COLOR_CHOICES, 2)
    rot_k = rng.randrange(4)
    dir_a = _rotate_dir(0, -1, rot_k)
    dir_b = _rotate_dir(0, 1, rot_k)

    legend_occupied = _place_legends(rng, grid, [(color_a, dir_a), (color_b, dir_b)])
    built = _build_local_shared_pair(rng, color_a, color_b, rot_k)
    if built is None:
        return None
    local_regions, local_wall_map, pair_h, pair_w = built

    placed = _place_shared_pair(rng, legend_occupied, local_regions, local_wall_map, pair_h, pair_w)
    if placed is None:
        return None
    regions, wall_map = placed

    for (r, c), color in wall_map.items():
        grid[r, c] = color
    if not _validate_topology(grid, regions):
        return None

    input_grid = grid.copy()
    if not _place_yellows(rng, regions, input_grid):
        return None
    if not _oracle_check(input_grid, regions, wall_map):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
