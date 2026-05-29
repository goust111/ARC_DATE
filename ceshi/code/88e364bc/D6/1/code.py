# Variant of 88e364bc
# Dimension: D6 — 区域边界形状 (obstacle)
# Change: 不规则闭合边界——1像素厚的彩色墙壁形成有台阶/凹槽的不规则闭合区域（填充率0.47-0.75）... → L形或T形区域

"""
design_analysis:
  variant_goal:
    keep:
      - legend-on-border mechanism
      - wall color -> direction lookup
      - yellow slides stepwise until blocked by a wall or an already-stopped yellow
    change:
      - enclosed passable regions are no longer generic irregular blobs
      - each region interior is explicitly an L or T shape, so the destination is not always the
        globally farthest wall in the movement direction

  a_concretization_and_trace:
    appearance:
      - legends stay 5x5 rectangles at grid corners
      - each wall color encloses one non-convex interior region
      - interiors are axis-aligned L/T silhouettes made from unions of rectangles
      - wall thickness remains 1 cell because walls are painted on the 4-neighbor ring around the
        interior set
    trace_example:
      - original-like case: a convex-ish region moving right lets every yellow continue until it
        reaches the globally rightmost boundary tier
      - variant case: an L-shape moving right has a short upper branch and a long lower branch;
        a yellow in the short branch stops at the inner concave wall, while a yellow in the long
        branch reaches the true far-right wall
      - solver difference in practice: the same sliding loop is still correct, but the generator
        must now build regions where "cannot step forward" can happen at an interior notch rather
        than only at the outer extreme

  b_complexity_calibration:
    chosen_controls:
      - legend count: 1-2
      - directions: cardinal only
      - region shapes: L or T
      - yellows per region: 1-2
    why_cardinal_only:
      - diagonal motion would add corner-cutting as a second source of complexity
      - that would visually compete with D6; cardinal motion makes the concave-stop behavior obvious
    visual_support:
      - every generated grid forces at least one yellow to end at a non-global-max projection cell
      - this prevents degeneration into the original "always pile at far wall" look

  c_implicit_assumption_audit_of_main:
    - _legend_directions assumes legends are border-touching filled rectangles with exactly 3 colors.
      This remains true because the generator still uses the same legend template family.
    - main() identifies enclosed regions as 4-connected passable components not touching the border.
      This still holds because each L/T interior is fully surrounded by a 1-cell wall ring.
    - border color is inferred by majority adjacent wall color. This remains valid because each
      region is generated with exactly one wall color and wall buffers prevent mixed-color adjacency.
    - _slide_points assumes motion is monotone along one direction and stops when the next cell is
      outside the region or occupied by an already-stopped yellow. This is exactly the behavior we
      want in concave regions; no new solver branch is needed.
    - the original generator relied on generic blobs whose output tiers were usually the far wall.
      That hidden tendency is invalid for the variant, so generator logic must explicitly request
      concave-stop targets.

  d_placement_legality:
    - legends reserve a 1-cell buffer
    - each region is placed from an explicit bbox with a computed wall ring
    - walls must stay off the outer border and may not overlap legend buffers or other reserved cells
    - a further 4-neighbor buffer around walls keeps different structures visually separated
    - topology is validated by checking that every bounded background component equals one generated
      interior set exactly

  e_visual_observability:
    - each sample contains at least one region where two rows/columns in the same region terminate
      at different x/y positions under the same motion direction
    - at least one yellow is forced to stop at a concave notch, so the variant can be inferred from
      the input/output pair without relying on hidden numeric properties

  f_cleanliness:
    - no overlapping walls, legends, or regions
    - one region per wall color keeps the color->legend mapping readable
    - cardinal-only motion avoids diagonal clutter

  g_color_inference:
    - colors do not encode new logic
    - wall color still only selects the legend direction
    - the D6 change is entirely geometric, which is visible directly in the silhouettes

  solver_generator_contract_after_change:
    - solver assumes bounded passable components are the intended regions
    - generator guarantees those components are exactly the L/T interiors
    - solver assumes at least one legend exists and every region's wall color has a legend
    - generator guarantees that bijection
    - solver processes yellows from far to near along the chosen direction
    - generator reverse-constructs input starts against that exact ordering
"""

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """Detect legend direction by center->indicator centroid offset."""
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
        others.sort()
        if len(others) < 2:
            continue
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside enclosed regions according to their matching border legend.

    The variant changes only the region geometry: regions may now be non-convex L/T shapes, so a
    yellow can stop at a concave notch before reaching the region's globally farthest boundary.
    The movement rule itself is unchanged: keep stepping in the legend direction until the next
    cell would leave the region or collide with an already-stopped yellow.
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
        if border_col not in dirs:
            continue
        dr, dc = dirs[border_col]

        for r, c in movers:
            out[r, c] = bg
        for r, c in _slide_points(pset, movers, dr, dc):
            out[r, c] = mover
    return out


GRID_H, GRID_W = 20, 20
CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1
MAX_SLIDE_STEPS = 6
GEN_RETRIES = 400
REGION_RETRIES = 80
YELLOWS_PER_REGION = [1, 2]


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
    colors = rng.sample(WALL_COLOR_CHOICES, n_legends)
    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []
    for corner, color in zip(corners, colors):
        dr, dc = rng.choice(CARDINAL_DIRS)
        patch = _legend_5x5(color, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((color, (dr, dc)))
    return legend_occupied, legends


def _blob_wall_cells(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    wall = set()
    for r, c in blob:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in blob:
                wall.add((nr, nc))
    return wall


def _projection(p: tuple[int, int], dr: int, dc: int) -> int:
    return dr * p[0] + dc * p[1]


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    return (nr, nc) in region_set


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
    return r, c


def _make_l_right(h: int, w: int, stem_w: int, bar_h: int, top_bar: bool) -> set[tuple[int, int]]:
    cells = {(r, c) for r in range(h) for c in range(stem_w)}
    if top_bar:
        cells |= {(r, c) for r in range(bar_h) for c in range(w)}
    else:
        cells |= {(r, c) for r in range(h - bar_h, h) for c in range(w)}
    return cells


def _make_t_right(h: int, w: int, stem_w: int, bar_h: int, bar_r: int) -> set[tuple[int, int]]:
    cells = {(r, c) for r in range(h) for c in range(stem_w)}
    cells |= {(r, c) for r in range(bar_r, bar_r + bar_h) for c in range(w)}
    return cells


def _transform_right_shape(
    coords: set[tuple[int, int]],
    h: int,
    w: int,
    direction: tuple[int, int],
) -> tuple[set[tuple[int, int]], int, int]:
    dr, dc = direction
    if (dr, dc) == (0, 1):
        return set(coords), h, w
    if (dr, dc) == (0, -1):
        return {(r, w - 1 - c) for r, c in coords}, h, w
    if (dr, dc) == (1, 0):
        return {(c, h - 1 - r) for r, c in coords}, w, h
    if (dr, dc) == (-1, 0):
        return {(w - 1 - c, r) for r, c in coords}, w, h
    raise ValueError(direction)


def _sample_region_shape(
    rng: random.Random,
    direction: tuple[int, int],
) -> dict | None:
    shape_type = rng.choice(["L", "T"])
    if shape_type == "L":
        h = rng.randint(6, 10)
        w = rng.randint(6, 10)
        stem_w = rng.randint(2, min(4, w - 2))
        bar_h = rng.randint(2, min(4, h - 2))
        top_bar = bool(rng.randrange(2))
        local = _make_l_right(h, w, stem_w, bar_h, top_bar)
    else:
        h = rng.randint(7, 11)
        w = rng.randint(7, 11)
        stem_w = rng.randint(2, min(4, w - 2))
        bar_h = rng.randint(2, min(3, h - 2))
        max_bar_r = h - bar_h - 1
        if max_bar_r <= 1:
            return None
        bar_r = rng.randint(1, max_bar_r - 1)
        local = _make_t_right(h, w, stem_w, bar_h, bar_r)
    if not (12 <= len(local) <= 97):
        return None
    transformed, th, tw = _transform_right_shape(local, h, w, direction)
    return {"shape": shape_type, "interior_local": transformed, "bbox_h": th, "bbox_w": tw}


def _reserve_with_buffer(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _place_regions(
    rng: random.Random,
    legends: list[tuple[int, tuple[int, int]]],
    legend_occupied: np.ndarray,
) -> list[dict] | None:
    h, w = legend_occupied.shape
    occupied = legend_occupied.copy()
    regions = []
    trap_index = rng.randrange(len(legends))

    for idx, (color, direction) in enumerate(legends):
        placed = None
        for _ in range(REGION_RETRIES):
            shape = _sample_region_shape(rng, direction)
            if shape is None:
                continue
            bh, bw = shape["bbox_h"], shape["bbox_w"]
            if bh > h - 4 or bw > w - 4:
                continue
            r0 = rng.randint(2, h - bh - 2)
            c0 = rng.randint(2, w - bw - 2)
            blob = {(r0 + r, c0 + c) for r, c in shape["interior_local"]}
            walls = _blob_wall_cells(blob, h, w)
            if not walls:
                continue
            if any(r in (0, h - 1) or c in (0, w - 1) for r, c in walls):
                continue
            reserve = _reserve_with_buffer(blob | walls, h, w)
            if any(occupied[r, c] for r, c in reserve):
                continue
            dr, dc = direction
            max_proj = max(_projection(p, dr, dc) for p in blob)
            trap_targets = [
                p for p in blob if not _slide_step_valid(blob, p[0], p[1], dr, dc) and _projection(p, dr, dc) < max_proj
            ]
            far_targets = [
                p for p in blob if not _slide_step_valid(blob, p[0], p[1], dr, dc) and _projection(p, dr, dc) == max_proj
            ]
            if not far_targets or (idx == trap_index and not trap_targets):
                continue
            for r, c in reserve:
                occupied[r, c] = True
            placed = {
                "color": color,
                "dir": direction,
                "shape": shape["shape"],
                "interior": blob,
                "walls": walls,
                "require_trap": idx == trap_index,
            }
            break
        if placed is None:
            return None
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


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    trap_realized = False
    for reg in regions:
        blob = reg["interior"]
        direction = reg["dir"]
        far_targets, trap_targets = _terminal_targets(blob, direction)
        rng.shuffle(far_targets)
        rng.shuffle(trap_targets)

        target_options: list[list[tuple[int, int]]] = []
        if reg["require_trap"]:
            for trap in trap_targets[:8]:
                if far_targets:
                    for far in far_targets[:4]:
                        if far != trap:
                            target_options.append([far, trap])
                target_options.append([trap])
        else:
            n_yellows = rng.choice(YELLOWS_PER_REGION)
            if n_yellows == 2 and len(far_targets) >= 2:
                target_options.extend([[far_targets[0], far_targets[1]], [far_targets[1], far_targets[0]]])
            if n_yellows == 2 and far_targets and trap_targets:
                target_options.append([far_targets[0], trap_targets[0]])
            if far_targets:
                target_options.append([far_targets[0]])
            if trap_targets:
                target_options.append([trap_targets[0]])

        success = None
        for targets in target_options:
            starts = _realize_targets(rng, blob, direction, targets)
            if starts is None:
                continue
            success = (targets, starts)
            break
        if success is None:
            return False

        targets, starts = success
        dr, dc = direction
        max_proj = max(_projection(p, dr, dc) for p in blob)
        if any(_projection(p, dr, dc) < max_proj for p in targets):
            trap_realized = True
        for r, c in starts:
            grid[r, c] = 4

    return trap_realized


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    struct_mask = (input_grid != 4) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if int((solver_out == 4).sum()) != int((input_grid == 4).sum()):
        return False
    regions_mask = np.zeros_like(input_grid, dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    if not bool(np.all(regions_mask[solver_out == 4])):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    legend_occupied, legends = _place_legends(rng, grid)
    regions = _place_regions(rng, legends, legend_occupied)
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
