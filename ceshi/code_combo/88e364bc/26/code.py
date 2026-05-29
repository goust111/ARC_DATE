# Combo variant of 88e364bc (index 26)
# combo_spec: D6/1+D9/0
# Name: L/T形区域+仅基数方向

"""
design_analysis:
  variant_goal:
    keep:
      - border legends still map wall color -> movement direction
      - yellow pixels still slide monotonically until the next step is blocked
      - bounded passable components are still the units that `main()` transforms
    change:
      - every enclosed region is now an explicit non-convex L or T silhouette
      - legends now use only 5x5 cardinal-direction templates
      - at least one generated yellow must stop at a concave inner wall, not only at the
        globally farthest projection tier

  a_concretization_and_trace:
    appearance:
      - the grid remains 20x20 with background 0
      - 1-2 colored legends sit on corners; each legend is a filled 5x5 rectangle with wall color
        as its dominant fill, blue center pixels, and red indicator pixels
      - each wall color surrounds one or more 1-cell-thick closed chambers whose interiors are
        axis-aligned L/T unions of rectangles
    trace_example:
      - original convex behavior: in a rectangular room with rightward motion, every yellow keeps
        stepping right until it reaches the outermost right wall or another stopped yellow
      - fused behavior: in an L-shape moving right, a yellow in the short arm can step right only
        until the concave notch removes the next cell; another yellow lower in the shape can keep
        going farther to the true outer wall
      - difference: the solver loop still only checks "is the next cell inside the region and free",
        but in this variant that local test can fail at an internal corner rather than only at the
        global extreme

  b_complexity_calibration:
    controls:
      - legend count stays small: 1-2
      - directions are restricted to the 4 cardinal offsets
      - region silhouettes are restricted to two families: L and T
      - yellow count per region stays 1-2
    anti-degeneration choices:
      - if there is only one legend, the generator creates two regions of that same color so the
        color->direction mapping has repeated support
      - one region is explicitly forced to realize a concave-stop terminal
      - paired far-stop and concave-stop targets are preferred in that forced region so the visual
        contrast appears inside a single chamber when possible

  c_implicit_assumption_audit_of_main:
    - `_legend_directions()` assumes each legend is a border-touching filled rectangle with exactly
      three colors. The generator preserves that template.
    - `main()` treats every bounded 4-connected passable component as one region. The generator
      therefore must fully seal each L/T interior with a 1-cell wall ring and keep regions
      disconnected from each other and from the outside background.
    - border color is inferred by majority adjacency. The generator assigns exactly one wall color
      per region and reserves buffers so mixed-color adjacency does not compete.
    - `_slide_points()` now assumes cardinal motion, i.e. exactly one of `dr`, `dc` is non-zero.
      The generator guarantees this by using only 5x5 cardinal legends.
    - slide order remains far-to-near along the chosen projection. Reverse construction of yellow
      inputs must use the same ordering or stacked outputs can become unrealizable.
    - because `main()` does not inspect shape type, the generator must encode the D6 change
      geometrically: the non-convex stop behavior has to arise from the region silhouette itself.

  d_placement_legality:
    - legends reserve their patch plus a 1-cell buffer
    - each region is instantiated from a sampled bbox, then converted into wall cells via its
      4-neighbor ring
    - wall cells are forbidden from touching the outer border
    - regions reserve an extra 4-neighbor buffer around interior and wall cells so walls, legends,
      and separate chambers never overlap or visually merge
    - topology is validated by recomputing bounded background components and matching them exactly
      against the generated interiors

  e_visual_observability:
    - the key visible clue is that in the same motion direction, different rows or columns of a
      chamber terminate at different depths
    - at least one sample region contains a yellow whose output projection is strictly less than the
      region's maximum projection, making the concave stop human-visible
    - using only cardinal arrows avoids mixing the D6 geometric effect with diagonal corner-cutting

  f_cleanliness:
    - legends stay in corners, away from chambers
    - regions are buffered apart, so colored walls do not fuse into unreadable mazes
    - each chamber uses a single wall color, and yellow pixels are the only movable objects

  g_color_inference:
    - colors do not encode any new secondary rule
    - wall color still serves only as the lookup key for the border legend
    - the fused change is inferable from shape geometry plus arrow direction, not from numeric color
      ids or hidden ordering

  main_assumptions_after_change:
    - every movable region has at least one adjacent wall color that appears in `dirs`
    - all decoded directions are cardinal
    - all yellow pixels start inside bounded passable components
    - stacked stopping behavior is deterministic under projection sorting
"""

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """Detect border legends and decode their center->indicator directions."""
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
            center_rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            ranked = []
            for _, col in others:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                ranked.append((np.sum((pts_col.mean(axis=0) - center_rc) ** 2), col))
            ranked.sort()
            center_col = ranked[0][1]
            indicator_col = ranked[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = [(r, c) for r, c in pts if int(g[r, c]) == center_col]
        indicator = [(r, c) for r, c in pts if int(g[r, c]) == indicator_col]
        if not center or not indicator:
            continue

        center_mean = np.mean(np.array(center, dtype=float), axis=0)
        indicator_mean = np.mean(np.array(indicator, dtype=float), axis=0)
        vr, vc = (indicator_mean - center_mean).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if (dr == 0) == (dc == 0):
            continue

        prev = best.get(fill_col)
        if prev is None or len(pts) > prev[0]:
            best[fill_col] = (len(pts), (dr, dc))

    return {color: direction for color, (_, direction) in best.items()}


def _projection(p: tuple[int, int], dr: int, dc: int) -> int:
    return dr * p[0] + dc * p[1]


def _slide_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    if not points:
        return []
    if (dr == 0) == (dc == 0):
        raise ValueError("cardinal direction required")

    points_sorted = sorted(points, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move yellow pixels inside closed colored chambers using the matching border legends.

    Input structure:
    - The background is the dominant color, and yellow pixels sit inside enclosed passable regions.
    - Each region is surrounded by a 1-cell colored wall, and the region interior is an L or T
      shape with a concave corner.
    - Border legends are 5x5 filled rectangles on grid corners. Their fill color matches a wall
      color, and their blue center pixels plus red indicator pixels encode one of the four cardinal
      directions.

    Rule:
    - For each bounded passable region, determine which wall color surrounds it and decode that
      wall color's legend direction.
    - Slide all yellow pixels in that region step by step in the indicated cardinal direction,
      processing from the far end inward so earlier stops block later ones.
    - A yellow stops as soon as the next step would leave the region or hit an already-stopped
      yellow; in a non-convex L/T chamber this can happen at an inner notch instead of only at the
      outermost wall.

    Key insight:
    - The legend still tells you only the movement direction. The new difficulty is geometric:
      because the chamber is non-convex, the correct stopping place is the first local wall reached
      along that direction, not necessarily the region's globally farthest boundary tier.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    mover = 4
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend"

    out = g.copy()
    passable = (g == bg) | (g == mover)

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        region_set = {(int(r), int(c)) for r, c in pts}
        border_counts: dict[int, int] = {}
        for r, c in pts:
            for nr, nc in neighbors4(r, c):
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
GEN_RETRIES = 500
REGION_RETRIES = 120
MAX_SLIDE_STEPS = 6
YELLOWS_PER_REGION = [1, 2]


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    patch = np.full((5, 5), fill, dtype=int)
    patch[1, 1] = 1
    patch[1, 2] = 1
    patch[1, 3] = 2
    patch[3, 1] = 1
    patch[3, 2] = 1
    patch[3, 3] = 2
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(patch, rot[(dr, dc)])


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


def _place_legends(rng: random.Random, grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    colors = rng.sample(WALL_COLOR_CHOICES, n_legends)
    legend_occupied = np.zeros((h, w), dtype=bool)
    legends: list[tuple[int, tuple[int, int]]] = []

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


def _sample_region_shape(rng: random.Random, direction: tuple[int, int]) -> dict | None:
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
    reserved = set(cells)
    for r, c in list(cells):
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                reserved.add((nr, nc))
    return reserved


def _region_plan(rng: random.Random, legends: list[tuple[int, tuple[int, int]]]) -> list[tuple[int, tuple[int, int]]]:
    if len(legends) == 1:
        return [legends[0], legends[0]]

    plan = [legends[0], legends[1]]
    if rng.random() < 0.5:
        plan.append(rng.choice(legends))
    rng.shuffle(plan)
    return plan


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    if (dr == 0) == (dc == 0):
        return False
    return (r + dr, c + dc) in region_set


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


def _place_regions(
    rng: random.Random,
    legends: list[tuple[int, tuple[int, int]]],
    legend_occupied: np.ndarray,
) -> list[dict] | None:
    h, w = legend_occupied.shape
    occupied = legend_occupied.copy()
    plan = _region_plan(rng, legends)
    trap_index = rng.randrange(len(plan))
    regions = []

    for idx, (color, direction) in enumerate(plan):
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

            far_targets, trap_targets = _terminal_targets(blob, direction)
            if not far_targets:
                continue
            if idx == trap_index and not trap_targets:
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

    used_colors = {reg["color"] for reg in regions}
    if used_colors != {color for color, _ in legends}:
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
    candidates = []
    for start in blob:
        if start == target or start in blocked_outputs or start in used_inputs:
            continue
        if _forward_slide(blob, start, dr, dc, blocked_outputs) != target:
            continue
        dist = abs(target[0] - start[0]) + abs(target[1] - start[1])
        if 1 <= dist <= MAX_SLIDE_STEPS:
            candidates.append(start)
    return candidates


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
    starts = []
    for target in targets_sorted:
        candidates = _reverse_start_candidates(blob, target, dr, dc, blocked_outputs, used_inputs)
        if not candidates:
            return None
        start = rng.choice(candidates)
        starts.append(start)
        used_inputs.add(start)
        blocked_outputs.add(target)
    return starts


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    trap_realized = False

    for reg in regions:
        blob = reg["interior"]
        direction = reg["dir"]
        dr, dc = direction
        far_targets, trap_targets = _terminal_targets(blob, direction)
        max_proj = max(_projection(p, dr, dc) for p in blob)

        rng.shuffle(far_targets)
        rng.shuffle(trap_targets)

        target_options: list[list[tuple[int, int]]] = []
        if reg["require_trap"]:
            for trap in trap_targets[:8]:
                for far in far_targets[:4]:
                    if far != trap:
                        target_options.append([far, trap])
                target_options.append([trap])
        else:
            n_yellows = rng.choice(YELLOWS_PER_REGION)
            if n_yellows == 2 and len(far_targets) >= 2:
                target_options.append([far_targets[0], far_targets[1]])
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
        if any(_projection(target, dr, dc) < max_proj for target in targets):
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
    return bool(np.all(regions_mask[solver_out == 4]))


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
