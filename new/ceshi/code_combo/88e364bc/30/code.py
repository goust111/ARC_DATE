# Combo variant of 88e364bc (index 30)
# combo_spec: D7/0+D13/2
# Name: 区域独立输出+单色多区域

from __future__ import annotations

# Design analysis for the fused variant
# a) Concrete behavior trace:
#    - Original: possibly 1-2 wall colors, each region reads its own color-matched legend, yellows
#      slide in-place inside the 20x20 grid, and the output keeps the full canvas.
#    - Fused: there is exactly one legend and one wall color, but 3-5 enclosed regions share it.
#      Each region still slides its yellows independently, then each region is cropped together with
#      its surrounding wall ring and emitted as a separate panel. Those panels are packed vertically.
#    - Example: if three same-colored rooms appear in different parts of the input, the solver cannot
#      use color to distinguish them. It must find three disconnected enclosed passable components,
#      apply the same direction to all three, then output three cropped room snippets instead of the
#      original 20x20 board.
# b) Complexity calibration:
#    - The single shared legend removes color-matching, but the generator compensates by forcing 3-5
#      regions so connectivity is the only reliable separator.
#    - Every region receives at least one yellow so the per-region output panels are behaviorally
#      meaningful; otherwise the packed output could contain many unchanged distractor panels.
#    - Regions remain irregular blobs rather than trivial rectangles, so the crop-and-pack change is
#      visually obvious and not reducible to a simple bounding-box trick.
# c) Implicit-assumption audit of the original main():
#    - `_legend_directions` originally supported multiple colors; the fused solver now requires a
#      single detected legend direction and applies it to all regions.
#    - The old loop inferred a border color for each region and kept the original canvas. Border-color
#      lookup is no longer needed for disambiguation, and the final stage now builds a packed grid.
#    - The old code only mattered for regions containing yellow. Here every enclosed region is part of
#      the output, so the solver must retain zero-yellow regions too (generator still avoids them).
# d) Placement legality:
#    - Generator reserves the legend corner with padding, grows each region on free interior cells,
#      rejects overlaps with prior walls/buffers, and keeps all walls off the outer border.
#    - Reverse-sliding only chooses input yellows whose forward slide lands on the intended final cell.
# e) Visual observability:
#    - Across examples, one legend color repeats around multiple disconnected rooms, making it clear
#      that color alone does not identify individual outputs.
#    - Packed outputs omit the legend entirely and show one small cropped room per region, so the
#      D13/2 shape change is obvious from raw dimensions alone.
# f) Visual cleanliness:
#    - Regions are buffered apart in the input; output panels are separated by one blank row and an
#      outer border, preventing adjacent crops from visually merging.
# g) Rule inferability:
#    - The only color relationship is "the lone legend controls every same-colored wall region".
#      Because all walls share one color in each sample, humans can infer the common direction from
#      repeated yellow motion rather than from hidden numeric color IDs.

import random
import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, neighbors4, paste


MOVER = 4
CENTER = 1
INDICATOR = 2

GRID_H, GRID_W = 20, 20
CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
CARDINAL_PROB = 0.6
LEGEND_BUFFER = 1

REGION_COUNT_CHOICES = [3, 4, 5]
REGION_SIZE_BOUNDS = {
    3: (11, 18),
    4: (8, 14),
    5: (6, 11),
}
BLOB_GROW_RETRIES = 40
MAX_SLIDE_STEPS = 4
GEN_RETRIES = 400


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """Detect legend rectangles on the border and decode their directions."""
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
            tie_break = []
            for _cnt, col in others:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                tie_break.append((float(np.sum((pts_col.mean(axis=0) - rc) ** 2)), col))
            tie_break.sort()
            center_col = tie_break[0][1]
            indicator_col = tie_break[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = np.array([(r, c) for r, c in pts if int(g[r, c]) == center_col], dtype=float)
        indicator = np.array([(r, c) for r, c in pts if int(g[r, c]) == indicator_col], dtype=float)
        if len(center) == 0 or len(indicator) == 0:
            continue

        delta = indicator.mean(axis=0) - center.mean(axis=0)
        dr = 0 if abs(float(delta[0])) < 1e-6 else (1 if delta[0] > 0 else -1)
        dc = 0 if abs(float(delta[1])) < 1e-6 else (1 if delta[1] > 0 else -1)
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
    """Slide movers within one region, stacking from the destination side inward."""
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


def _region_records(g: np.ndarray, bg: int) -> list[dict]:
    """Return enclosed passable regions with their adjacent walls and bounding boxes."""
    h, w = g.shape
    passable = (g == bg) | (g == MOVER)
    records = []
    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        interior = {(int(r), int(c)) for r, c in pts}
        walls = set()
        wall_colors = set()
        movers = []
        for r, c in pts:
            rr = int(r)
            cc = int(c)
            if int(g[rr, cc]) == MOVER:
                movers.append((rr, cc))
            for nr, nc in neighbors4(rr, cc):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in interior:
                    continue
                v = int(g[nr, nc])
                if v == bg or v == MOVER:
                    continue
                walls.add((nr, nc))
                wall_colors.add(v)
        if not walls:
            continue
        bbox_pts = np.array(list(interior | walls), dtype=int)
        r0, c0, r1, c1 = bbox_coords(bbox_pts)
        records.append(
            {
                "interior": interior,
                "walls": walls,
                "movers": movers,
                "wall_colors": wall_colors,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    records.sort(key=lambda rec: (rec["bbox"][0], rec["bbox"][1]))
    return records


def _pack_crops_vertical(crops: list[np.ndarray], bg: int) -> np.ndarray:
    """Pack crops top-to-bottom with 1-cell gaps and a 1-cell outer border."""
    assert crops
    heights = [int(c.shape[0]) for c in crops]
    widths = [int(c.shape[1]) for c in crops]
    out_h = 1 + sum(heights) + (len(crops) - 1) + 1
    out_w = 1 + max(widths) + 1
    out = np.full((out_h, out_w), int(bg), dtype=int)
    cursor = 1
    for crop in crops:
        h, w = crop.shape
        paste(out, crop, cursor, 1)
        cursor += int(h) + 1
    return out


def _solve_large_grid(g: np.ndarray) -> tuple[np.ndarray, list[dict], int]:
    """Apply the shared slide rule on the original 20x20 canvas."""
    bg = int(get_bg(g))
    dirs = _legend_directions(g, bg)
    assert len(dirs) == 1, "fused variant expects exactly one legend"
    shared_dir = next(iter(dirs.values()))
    dr, dc = shared_dir

    out = np.array(g, dtype=int, copy=True)
    regions = _region_records(g, bg)
    for rec in regions:
        movers = rec["movers"]
        for r, c in movers:
            out[r, c] = bg
        new_pts = _slide_points(rec["interior"], movers, dr, dc)
        for r, c in new_pts:
            out[r, c] = MOVER
    return out, regions, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside several same-colored enclosed rooms, then crop each room into its own panel.

    Input structure:
    - Background is the most common color.
    - There is one border legend whose fill color matches the single wall color used by every room.
    - Inside the 20x20 board are 3-5 disconnected enclosed passable regions bounded by that same wall color; each region may contain yellow pixels.

    Rule:
    - Decode one shared movement direction from the lone legend.
    - In every enclosed region, slide the yellow pixels independently in that common direction, stacking them against the far wall.
    - After sliding, crop each region together with its surrounding wall ring, sort the crops by reading order, and pack them vertically with 1-cell gaps and a 1-cell outer border.

    Key insight: The legend no longer selects among wall colors; instead it provides a global direction, so the main task is to separate same-colored rooms by connectivity and then emit each solved room as its own cropped output panel.
    """
    solved, regions, bg = _solve_large_grid(np.asarray(grid, dtype=int))
    crops = [crop_bbox(solved, rec["bbox"]) for rec in regions]
    return _pack_crops_vertical(crops, bg)


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = CENTER
    g[1, 2] = CENTER
    g[1, 3] = INDICATOR
    g[3, 1] = CENTER
    g[3, 2] = CENTER
    g[3, 3] = INDICATOR
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


def _legend_6x7(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((6, 7), fill, dtype=int)
    g[1, 3] = CENTER
    g[2, 1] = CENTER
    g[2, 4] = CENTER
    g[3, 2] = CENTER
    g[3, 5] = INDICATOR
    g[4, 3] = INDICATOR
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


def _place_single_legend(rng: random.Random, grid: np.ndarray) -> tuple[np.ndarray, dict]:
    h, w = grid.shape
    wall_color = rng.choice(WALL_COLOR_CHOICES)
    if rng.random() < CARDINAL_PROB:
        dr, dc = rng.choice(CARDINAL_DIRS)
        patch = _legend_5x5(wall_color, dr, dc)
    else:
        dr, dc = rng.choice(DIAGONAL_DIRS)
        patch = _legend_6x7(wall_color, dr, dc)
    corner = rng.choice(CORNERS)
    ph, pw = patch.shape
    r0, c0 = _corner_origin(corner, h, w, ph, pw)
    paste(grid, patch, r0, c0)
    occupied = np.zeros((h, w), dtype=bool)
    r_lo = max(0, r0 - LEGEND_BUFFER)
    r_hi = min(h, r0 + ph + LEGEND_BUFFER)
    c_lo = max(0, c0 - LEGEND_BUFFER)
    c_hi = min(w, c0 + pw + LEGEND_BUFFER)
    occupied[r_lo:r_hi, c_lo:c_hi] = True
    return occupied, {"color": wall_color, "dir": (dr, dc), "corner": corner}


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
    frontier: list[tuple[int, int]] = []

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
    for r, c in blob | walls:
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
    out = set(blob)
    for r, c in np.argwhere(trapped):
        out.add((int(r), int(c)))
    return out


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


def _grow_regions(rng: random.Random, legend_occupied: np.ndarray, wall_color: int, direction: tuple[int, int]) -> list[dict] | None:
    h, w = legend_occupied.shape
    n_regions = rng.choice(REGION_COUNT_CHOICES)
    size_min, size_max = REGION_SIZE_BOUNDS[n_regions]
    occupied = legend_occupied.copy()
    regions = []

    for _ in range(n_regions):
        canvas = np.ones((h, w), dtype=bool)
        canvas[0, :] = False
        canvas[-1, :] = False
        canvas[:, 0] = False
        canvas[:, -1] = False
        canvas &= ~occupied
        if not canvas.any():
            return None
        target_size = rng.randint(size_min, size_max)
        blob = _try_grow_valid_blob(rng, canvas, occupied, target_size, h, w)
        if blob is None:
            return None
        walls = _blob_wall_cells(blob, h, w)
        bbox_pts = np.array(list(blob | walls), dtype=int)
        bbox = bbox_coords(bbox_pts)
        regions.append(
            {
                "color": wall_color,
                "dir": direction,
                "interior": blob,
                "walls": walls,
                "bbox": bbox,
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

    regions.sort(key=lambda rec: (rec["bbox"][0], rec["bbox"][1]))
    return regions


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
    return r, c


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
        ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in blob or (pr, pc) in already_stopped or (pr, pc) in existing_inputs:
                ok = False
                break
            if dr != 0 and dc != 0:
                if (pr + dr, pc) not in blob or (pr, pc + dc) not in blob:
                    ok = False
                    break
            r, c = pr, pc
        if not ok:
            continue
        if _forward_slide(blob, (r, c), dr, dc, already_stopped) == out_pos:
            return r, c
    return None


def _plan_region_yellows(rng: random.Random, region: dict) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    blob = region["interior"]
    dr, dc = region["dir"]
    cells_sorted = sorted(blob, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    first_key = dr * cells_sorted[0][0] + dc * cells_sorted[0][1]
    tier1 = [p for p in cells_sorted if dr * p[0] + dc * p[1] == first_key]
    tier2 = [p for p in cells_sorted if dr * p[0] + dc * p[1] == first_key - 1]
    first_choices = tier1[:]
    rng.shuffle(first_choices)
    desired = 2 if len(blob) >= 8 and rng.random() < 0.45 else 1

    for first in first_choices:
        second_pool = []
        pr, pc = first[0] - dr, first[1] - dc
        if (pr, pc) in blob:
            second_pool.append((pr, pc))
        for p in tier1 + tier2:
            if p != first and p not in second_pool:
                second_pool.append(p)
        outputs_options = [[first]]
        if desired == 2:
            for second in second_pool:
                outputs_options.append([first, second])
        for output_positions in outputs_options:
            out_sorted = sorted(output_positions, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
            already_stopped = set()
            input_positions = []
            ok = True
            for out_pos in out_sorted:
                candidate = _find_reverse_slide(blob, out_pos, dr, dc, already_stopped, input_positions, rng)
                if candidate is None:
                    ok = False
                    break
                input_positions.append(candidate)
                already_stopped.add(out_pos)
            if ok:
                return input_positions, output_positions
    return None


def _paint_walls(grid: np.ndarray, regions: list[dict]) -> None:
    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = int(reg["color"])


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    for reg in regions:
        plan = _plan_region_yellows(rng, reg)
        if plan is None:
            return False
        input_positions, output_positions = plan
        reg["input_yellows"] = input_positions
        reg["final_yellows"] = output_positions
        for r, c in input_positions:
            grid[r, c] = MOVER
    return True


def _expected_output(input_grid: np.ndarray, regions: list[dict]) -> np.ndarray:
    bg = int(get_bg(input_grid))
    solved = np.array(input_grid, dtype=int, copy=True)
    solved[solved == MOVER] = bg
    for reg in regions:
        for r, c in reg["final_yellows"]:
            solved[r, c] = MOVER
    crops = [crop_bbox(solved, reg["bbox"]) for reg in sorted(regions, key=lambda rec: (rec["bbox"][0], rec["bbox"][1]))]
    return _pack_crops_vertical(crops, bg)


def _validate_instance(input_grid: np.ndarray, regions: list[dict], legend: dict) -> bool:
    wall_colors = {int(reg["color"]) for reg in regions}
    if len(regions) < 3 or len(regions) > 5:
        return False
    if len(wall_colors) != 1 or next(iter(wall_colors)) != int(legend["color"]):
        return False
    if not all(len(reg.get("final_yellows", [])) >= 1 for reg in regions):
        return False
    expected = _expected_output(input_grid, regions)
    try:
        actual = main(input_grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    legend_occupied, legend = _place_single_legend(rng, grid)
    regions = _grow_regions(rng, legend_occupied, int(legend["color"]), tuple(legend["dir"]))
    if regions is None:
        return None
    _paint_walls(grid, regions)
    input_grid = grid.copy()
    if not _place_yellows(rng, regions, input_grid):
        return None
    if not _validate_instance(input_grid, regions, legend):
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
