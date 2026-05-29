# Variant of 88e364bc
# Dimension: D7 — 区域数量与墙壁颜色数 (obstacle)
# Change: 1-2种墙壁颜色，每种形成1-3个封闭区域，总计不超过4个区域；不同颜色墙壁可4-邻接形成迷宫结构... → 共享墙壁（两个区域共用一面墙）

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


# Design analysis for the D7 variant
# ----------------------------------
# spec:
#   original_core:
#     - enclosed passable regions are surrounded by colored walls
#     - a border legend with fill color C gives the slide direction for regions whose wall color is C
#     - the solver identifies each region's controlling wall color from adjacent boundary cells
#   variant_core:
#     - exactly two wall colors are always present, because the new phenomenon requires two different colors
#     - one mandatory pair of enclosed regions shares a full straight wall segment
#     - the shared segment is a 1-cell-thick seam whose cells alternate the two wall colors
#     - each shared region still belongs to one legend by strict majority color over its whole boundary
#
# concreteization_and_trace:
#   - The shared pair is built from two rectangles that touch through a single vertical or horizontal seam.
#   - Example (vertical seam):
#       left region private walls mostly color A, right region private walls mostly color B,
#       seam cells go A/B/A/B/... from top to bottom.
#       Old logic on a single-color wall only ever saw one adjacent wall color.
#       New logic must count both colors on the same region boundary:
#         left region border counts might be {A: 16, B: 3} -> use legend A
#         right region border counts might be {B: 16, A: 3} -> use legend B
#       The visible difference is that a human sees both regions touch the same alternating seam,
#       yet their yellow pixels still follow different legends.
#
# complexity_calibration:
#   - The shared seam is mandatory in every sample, so D7 is always visible.
#   - We keep the wall-color count at exactly two; adding more colors would bury the intended clue.
#   - We optionally add 0-2 isolated extra regions so the task still spans 2-4 regions overall.
#   - Directions are still sampled from the original cardinal/diagonal set, preserving the original
#     movement difficulty while changing only the obstacle/boundary analysis.
#
# implicit_assumptions_review:
#   - _legend_directions only depends on border rectangles; the variant does not change legend parsing.
#   - _slide_points only depends on passable interior geometry; the variant does not change motion.
#   - The original main() used max(border_counts, key=...) which silently relied on dict order if a tie
#     ever happened. With alternating shared walls, tie-handling becomes a real concern, so this variant
#     makes the majority-color step explicit and deterministic.
#   - The generator therefore guarantees a strict majority for each shared region, and the solver skips a
#     region if a malformed input produces no unique majority.
#
# placement_legality:
#   - Legends stay in corners with a 1-cell reservation halo.
#   - The shared pair is placed strictly inside the grid and reserves an external buffer so extra regions
#     cannot accidentally create second shared walls.
#   - Extra regions are isolated rectangles with their own 1-cell halo.
#   - Topology is validated by matching bounded background components to the intended interiors.
#
# visual_observability_and_cleanliness:
#   - The alternating seam is long (at least 4 cells adjacent to both interiors), straight, and uses only
#     the two legend-linked wall colors, so the shared-wall cue is visible in 2-3 examples.
#   - Only one mandatory shared pair is created per grid; this keeps the picture readable.
#   - No invisible numeric ordering is used: all decisions are based on visible counts of boundary colors.


# ── Solver ────────────────────────────────────────────────────────


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Detect border legends and map legend fill color -> slide direction.

    A legend is a border-touching filled rectangle with exactly three colors:
    dominant fill color C, a more central "center" color, and a rarer "indicator" color.
    The sign of centroid(indicator) - centroid(center) defines the direction.
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
            cents = []
            for _, col in others:
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


def _slide_points(
    region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int
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
    g: np.ndarray, region_set: set[tuple[int, int]], bg: int, mover: int
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
    Slide yellow pixels inside enclosed regions using border legends.

    Variant note:
    - Some regions share an alternating two-color wall segment.
    - Region ownership is determined by the strict majority color along the whole 4-neighbor boundary.
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


# ── Generator ─────────────────────────────────────────────────────


GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ALL_DIRS = CARDINAL_DIRS + DIAGONAL_DIRS
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1

SHARED_PAIR_RETRIES = 60
EXTRA_REGION_RETRIES = 60
GEN_RETRIES = 300

SHARED_SIDE_RANGE = (3, 5)
SHARED_SPAN_RANGE = (4, 6)
EXTRA_SIDE_RANGE = (3, 5)
YELLOWS_PER_ACTIVE_REGION = [1, 2]
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


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                out.add((rr, cc))
    return out


def _mark_occupied(mask: np.ndarray, cells: set[tuple[int, int]]) -> None:
    for r, c in cells:
        mask[r, c] = True


def _place_legends(
    rng: random.Random, grid: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    h, w = grid.shape
    corners = rng.sample(CORNERS, 2)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, 2)
    dirs = rng.sample(ALL_DIRS, 2)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []
    for corner, wc, direction in zip(corners, wall_colors, dirs):
        dr, dc = direction
        patch = _legend_5x5(wc, dr, dc) if direction in CARDINAL_DIRS else _legend_6x7(wc, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wc, direction))
    return legend_occupied, legends


def _make_rect_region(
    top: int, left: int, inner_h: int, inner_w: int, color: int, direction: tuple[int, int], kind: str
) -> tuple[dict, dict[tuple[int, int], int]]:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    interior = {
        (r, c) for r in range(top + 1, bottom) for c in range(left + 1, right)
    }
    wall_colors: dict[tuple[int, int], int] = {}
    for c in range(left, right + 1):
        wall_colors[(top, c)] = color
        wall_colors[(bottom, c)] = color
    for r in range(top, bottom + 1):
        wall_colors[(r, left)] = color
        wall_colors[(r, right)] = color
    region = {
        "color": color,
        "dir": direction,
        "interior": interior,
        "kind": kind,
        "shared_segment": [],
    }
    return region, wall_colors


def _make_shared_pair(
    top: int,
    left: int,
    orientation: str,
    major_a: int,
    major_b: int,
    dir_a: tuple[int, int],
    dir_b: tuple[int, int],
    span: int,
    side_a: int,
    side_b: int,
) -> tuple[list[dict], dict[tuple[int, int], int], set[tuple[int, int]]]:
    wall_colors: dict[tuple[int, int], int] = {}

    if orientation == "vertical":
        seam_c = left + side_a + 1
        bottom = top + span + 1
        right = left + side_a + side_b + 2

        interior_a = {(r, c) for r in range(top + 1, bottom) for c in range(left + 1, seam_c)}
        interior_b = {(r, c) for r in range(top + 1, bottom) for c in range(seam_c + 1, right)}

        for c in range(left, seam_c):
            wall_colors[(top, c)] = major_a
            wall_colors[(bottom, c)] = major_a
        for r in range(top, bottom + 1):
            wall_colors[(r, left)] = major_a

        for c in range(seam_c + 1, right + 1):
            wall_colors[(top, c)] = major_b
            wall_colors[(bottom, c)] = major_b
        for r in range(top, bottom + 1):
            wall_colors[(r, right)] = major_b

        shared_segment = []
        for r in range(top, bottom + 1):
            wall_colors[(r, seam_c)] = major_a if (r - top) % 2 == 0 else major_b
            if top < r < bottom:
                shared_segment.append((r, seam_c))
    else:
        seam_r = top + side_a + 1
        bottom = top + side_a + side_b + 2
        right = left + span + 1

        interior_a = {(r, c) for r in range(top + 1, seam_r) for c in range(left + 1, right)}
        interior_b = {(r, c) for r in range(seam_r + 1, bottom) for c in range(left + 1, right)}

        for r in range(top, seam_r):
            wall_colors[(r, left)] = major_a
            wall_colors[(r, right)] = major_a
        for c in range(left, right + 1):
            wall_colors[(top, c)] = major_a

        for r in range(seam_r + 1, bottom + 1):
            wall_colors[(r, left)] = major_b
            wall_colors[(r, right)] = major_b
        for c in range(left, right + 1):
            wall_colors[(bottom, c)] = major_b

        shared_segment = []
        for c in range(left, right + 1):
            wall_colors[(seam_r, c)] = major_a if (c - left) % 2 == 0 else major_b
            if left < c < right:
                shared_segment.append((seam_r, c))

    region_a = {
        "color": major_a,
        "dir": dir_a,
        "interior": interior_a,
        "kind": "shared",
        "shared_segment": shared_segment,
    }
    region_b = {
        "color": major_b,
        "dir": dir_b,
        "interior": interior_b,
        "kind": "shared",
        "shared_segment": shared_segment,
    }
    footprint = set(wall_colors) | interior_a | interior_b
    return [region_a, region_b], wall_colors, footprint


def _try_place_shared_pair(
    rng: random.Random,
    legends: list[tuple[int, tuple[int, int]]],
    legend_occupied: np.ndarray,
    occupied: np.ndarray,
) -> tuple[list[dict], dict[tuple[int, int], int], set[tuple[int, int]]] | None:
    h, w = occupied.shape
    (color_a, dir_a), (color_b, dir_b) = legends

    for _ in range(SHARED_PAIR_RETRIES):
        orientation = rng.choice(["vertical", "horizontal"])
        span = rng.randint(*SHARED_SPAN_RANGE)
        side_a = rng.randint(*SHARED_SIDE_RANGE)
        side_b = rng.randint(*SHARED_SIDE_RANGE)
        if side_a * span < 12 or side_b * span < 12:
            continue

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

        regions, wall_colors, footprint = _make_shared_pair(
            top, left, orientation, color_a, color_b, dir_a, dir_b, span, side_a, side_b
        )
        halo = _expand_cells(footprint, h, w, pad=1)
        if any(legend_occupied[r, c] or occupied[r, c] for r, c in halo):
            continue

        grid = np.zeros((h, w), dtype=int)
        for (r, c), col in wall_colors.items():
            grid[r, c] = col
        if not _shared_pair_majorities_ok(grid, regions):
            continue
        return regions, wall_colors, halo
    return None


def _try_place_extra_region(
    rng: random.Random,
    color: int,
    direction: tuple[int, int],
    occupied: np.ndarray,
) -> tuple[dict, dict[tuple[int, int], int], set[tuple[int, int]]] | None:
    h, w = occupied.shape
    for _ in range(EXTRA_REGION_RETRIES):
        inner_h = rng.randint(*EXTRA_SIDE_RANGE)
        inner_w = rng.randint(*EXTRA_SIDE_RANGE)
        if inner_h * inner_w < 12:
            continue
        shape_h = inner_h + 2
        shape_w = inner_w + 2
        if shape_h > h - 4 or shape_w > w - 4:
            continue
        top = rng.randint(1, h - shape_h - 1)
        left = rng.randint(1, w - shape_w - 1)
        region, wall_colors = _make_rect_region(top, left, inner_h, inner_w, color, direction, "isolated")
        footprint = set(wall_colors) | region["interior"]
        halo = _expand_cells(footprint, h, w, pad=1)
        if any(occupied[r, c] for r, c in halo):
            continue
        return region, wall_colors, halo
    return None


def _grow_regions(
    rng: random.Random, legends: list[tuple[int, tuple[int, int]]], legend_occupied: np.ndarray
) -> tuple[list[dict], dict[tuple[int, int], int]] | None:
    occupied = legend_occupied.copy()
    shared = _try_place_shared_pair(rng, legends, legend_occupied, occupied)
    if shared is None:
        return None
    regions, wall_colors, halo = shared
    _mark_occupied(occupied, halo)

    counts = {legends[0][0]: 1, legends[1][0]: 1}
    total_regions = rng.choice([2, 3, 4])
    extras_needed = total_regions - 2

    color_dir = {color: direction for color, direction in legends}
    for _ in range(extras_needed):
        choices = [color for color in color_dir if counts[color] < 3]
        if not choices:
            break
        color = rng.choice(choices)
        placed = _try_place_extra_region(rng, color, color_dir[color], occupied)
        if placed is None:
            return None
        region, extra_walls, extra_halo = placed
        regions.append(region)
        wall_colors.update(extra_walls)
        _mark_occupied(occupied, extra_halo)
        counts[color] += 1

    return regions, wall_colors


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


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _forward_slide(
    region_set: set[tuple[int, int]], start: tuple[int, int], dr: int, dc: int, blocked: set[tuple[int, int]]
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


def _find_reverse_slide(
    blob: set[tuple[int, int]],
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    already_stopped: set[tuple[int, int]],
    existing_inputs: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    target_steps = rng.randint(1, MAX_SLIDE_STEPS)
    for steps in range(target_steps, 0, -1):
        r, c = out_pos
        path_ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in blob or (pr, pc) in already_stopped or (pr, pc) in existing_inputs:
                path_ok = False
                break
            if dr != 0 and dc != 0:
                if (pr + dr, pc) not in blob or (pr, pc + dc) not in blob:
                    path_ok = False
                    break
            r, c = pr, pc
        if not path_ok:
            continue
        if _forward_slide(blob, (r, c), dr, dc, already_stopped) == out_pos:
            return (r, c)
    return None


def _choose_active_regions(rng: random.Random, regions: list[dict]) -> list[int]:
    shared_ids = [i for i, reg in enumerate(regions) if reg["kind"] == "shared"]
    active = list(shared_ids)
    extra_ids = [i for i, reg in enumerate(regions) if reg["kind"] != "shared"]
    rng.shuffle(extra_ids)
    if extra_ids and rng.random() < 0.6:
        active.append(extra_ids[0])
    return active


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    active_ids = set(_choose_active_regions(rng, regions))
    total_yellows = 0
    for idx, reg in enumerate(regions):
        if idx not in active_ids:
            continue

        blob = reg["interior"]
        dr, dc = reg["dir"]
        n_yellows = rng.choice(YELLOWS_PER_ACTIVE_REGION)
        if total_yellows + n_yellows > 6:
            n_yellows = max(1, 6 - total_yellows)
        if n_yellows <= 0:
            continue

        cells_sorted = sorted(blob, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
        far_key = dr * cells_sorted[0][0] + dc * cells_sorted[0][1]
        tier1 = [p for p in cells_sorted if dr * p[0] + dc * p[1] == far_key]
        first = tier1[rng.randrange(len(tier1))]
        output_positions = [first]

        if n_yellows == 2:
            pr, pc = first[0] - dr, first[1] - dc
            cands = ([(pr, pc)] if (pr, pc) in blob else []) + [p for p in tier1 if p != first]
            cands = [p for p in cands if p != first]
            if cands:
                output_positions.append(cands[rng.randrange(len(cands))])

        out_sorted = sorted(output_positions, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
        already_stopped = set()
        inputs = []
        for out_pos in out_sorted:
            candidate = _find_reverse_slide(blob, out_pos, dr, dc, already_stopped, inputs, rng)
            if candidate is None:
                return False
            inputs.append(candidate)
            already_stopped.add(out_pos)

        for r, c in inputs:
            grid[r, c] = 4
        total_yellows += len(inputs)
    return total_yellows >= 2


def _shared_pair_majorities_ok(grid: np.ndarray, regions: list[dict]) -> bool:
    for reg in regions:
        border_counts = _region_border_counts(grid, reg["interior"], 0, 4)
        majority = _majority_border_color(border_counts)
        if majority != reg["color"]:
            return False
        if len(border_counts) < 2:
            return False
    vals = [int(grid[r, c]) for r, c in regions[0]["shared_segment"]]
    if len(vals) < 4:
        return False
    if any(v not in {regions[0]["color"], regions[1]["color"]} for v in vals):
        return False
    if any(vals[i] == vals[i + 1] for i in range(len(vals) - 1)):
        return False
    return True


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    struct_mask = (input_grid != 4) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if (solver_out == 4).sum() != (input_grid == 4).sum():
        return False

    region_mask = np.zeros_like(input_grid, dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            region_mask[r, c] = True
        border_counts = _region_border_counts(input_grid, reg["interior"], 0, 4)
        majority = _majority_border_color(border_counts)
        if majority != reg["color"]:
            return False
        if reg["kind"] == "shared" and len(border_counts) < 2:
            return False
    if not np.all(region_mask[solver_out == 4]):
        return False

    shared_regions = [reg for reg in regions if reg["kind"] == "shared"]
    if len(shared_regions) != 2:
        return False
    return _shared_pair_majorities_ok(input_grid, shared_regions)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    legend_occupied, legends = _place_legends(rng, grid)
    grown = _grow_regions(rng, legends, legend_occupied)
    if grown is None:
        return None
    regions, wall_colors = grown

    for (r, c), color in wall_colors.items():
        grid[r, c] = color
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
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
