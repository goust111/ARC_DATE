# Combo variant of 88e364bc (index 16)
# combo_spec: D7/0+D13/0
# Name: 裁剪输出+单色多区域

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste

# -----------------------------------------------------------------------------
# Design analysis for the D7/0 + D13/0 combo variant
#
# a) Concrete behavior:
#    - Original task: 1-2 legend colors can exist, each wall color points to its own legend,
#      and the full 20x20 board is returned after yellow pixels slide.
#    - Combo input: there is exactly one border legend, one wall color, and 3-5 separate
#      enclosed regions made from that same wall color. Every yellow pixel in every region
#      therefore shares the same slide direction.
#    - Combo output: perform the same slide first, then crop the result to the minimal bbox
#      covering every enclosed region interior plus its wall ring. The legend and empty outer
#      background disappear.
#    - Small trace:
#      original: two differently colored fenced rooms might use two different arrows, and the
#      output keeps both corner legends visible.
#      combo: one corner legend controls four same-colored rooms; all yellows move in parallel
#      along the same direction; then the board is cropped down to just the cluster of rooms.
#      The visible difference is both structural (same-color multi-region) and geometric
#      (smaller output frame).
#
# b) Complexity calibration:
#    - Using only one wall color simplifies legend matching, so region count is increased to
#      3-5 to keep the puzzle visually nontrivial.
#    - We keep one shared direction across all rooms. This repeated evidence is important:
#      humans can see the same directional effect occur in several compartments in one example.
#    - Crop behavior must stay obvious, so regions are generated in a safe band away from the
#      legend side. That makes the output consistently smaller than the 20x20 input.
#
# c) main() assumptions audit:
#    - `_legend_directions(...)` originally allowed several legend colors. In the combo, the
#      solver still supports a mapping, but the generator guarantees only one usable legend.
#    - Region detection via enclosed passable components still works even though all walls have
#      the same color, because regions are identified by connectivity of their interiors, not
#      by color identity.
#    - The original same-shape assumption no longer holds: after sliding, `main()` must crop.
#    - Crop support cannot depend only on yellow destinations; empty rooms or rooms whose yellow
#      starts and ends on the same column/row must still remain visible. Therefore all enclosed
#      passable cells and their adjacent walls contribute to the crop bbox.
#
# d) Placement legality:
#    - The legend sits in exactly one corner with a one-cell buffer.
#    - Every region interior is grown on a safe mask that avoids the legend side and keeps a
#      margin from the outer border, so walls do not overlap legends or spill out of bounds.
#    - Regions reserve their wall ring plus an extra moat, ensuring 3-5 distinct enclosed
#      components rather than one merged same-color maze.
#
# e) Visual observability:
#    - The fusion is inferable from 2-4 examples because two signals are stable across samples:
#      all rooms use the same wall color and all yellows move in the same direction, while the
#      output frame is always the tight room-cluster crop instead of the original canvas.
#    - No hidden ordering by numeric color id is used. Region handling relies only on visible
#      connectivity, wall contact, and geometric cropping.
#
# f) Visual cleanliness:
#    - Rooms are separated by reserved spacing, so same-colored walls remain readable as several
#      distinct fenced areas instead of one tangled mass.
#    - The crop includes full wall rings, preserving the identity of each room after legends and
#      blank margins are removed.
#
# g) Change inferability:
#    - There is no new color remapping to learn. The only color linkage is the visible one:
#      the single legend fill color matches the single wall color.
#    - The meaningful changes are "one legend controls many same-colored rooms" and "output is
#      cropped to the room cluster"; both are directly visible in the pixel grids.
#
# 9) New solver assumptions and generator guarantees:
#    - Assumption: at least one border legend exists and its fill color matches the wall color.
#      Guarantee: generator always places exactly one valid legend.
#    - Assumption: multiple enclosed regions can be found without ambiguity. Guarantee: generator
#      enforces 3-5 disjoint interiors with wall/buffer separation.
#    - Assumption: crop bbox excludes the legend. Guarantee: regions are restricted to a safe
#      band opposite the legend side, so their union bbox does not intersect the legend patch.
#    - Assumption: the combo should not degenerate back to the original same-size output.
#      Guarantee: oracle rejects uncropped outputs.
# -----------------------------------------------------------------------------


GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
CARDINAL_PROB = 0.5
LEGEND_BUFFER = 1

N_REGIONS_RANGE = (3, 5)
REGION_SIZE_BOUNDS = {
    3: (14, 26),
    4: (10, 18),
    5: (8, 13),
}
BLOB_GROW_RETRIES = 40
GEN_RETRIES = 500

MAX_SLIDE_STEPS = 5


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """Detect border legends and decode the center→indicator direction."""
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for pts in components_mask(non_bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(np.array(pts, dtype=int))
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
            tied = []
            for _cnt, col in others:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                tied.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            tied.sort()
            center_col = tied[0][1]
            indicator_col = tied[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = np.array([(r, c) for r, c in pts if int(g[r, c]) == center_col], dtype=float)
        indic = np.array([(r, c) for r, c in pts if int(g[r, c]) == indicator_col], dtype=float)
        if len(center) == 0 or len(indic) == 0:
            continue

        vr, vc = (indic.mean(axis=0) - center.mean(axis=0)).tolist()
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


def _bounded_regions(g: np.ndarray, bg: int, mover: int) -> list[dict[str, object]]:
    """Return each enclosed passable region with its movers and adjacent walls."""
    h, w = g.shape
    passable = (g == bg) | (g == mover)
    regions: list[dict[str, object]] = []
    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        pset = {(int(r), int(c)) for r, c in pts}
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        walls = set()
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
                walls.add((nr, nc))
                border_counts[v] = border_counts.get(v, 0) + 1
        border_col = max(border_counts, key=border_counts.get) if border_counts else None
        regions.append(
            {
                "pset": pset,
                "movers": movers,
                "walls": walls,
                "border_col": border_col,
            }
        )
    return regions


def _crop_bbox_from_regions(regions: list[dict[str, object]]) -> tuple[int, int, int, int]:
    coords: list[tuple[int, int]] = []
    for reg in regions:
        coords.extend(reg["pset"])
        coords.extend(reg["walls"])
    assert coords, "combo variant requires at least one enclosed region"
    return bbox_coords(np.array(coords, dtype=int))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside multiple same-colored rooms, then return a tight crop.

    Input structure:
    - The board still contains a border legend and several enclosed passable regions, but all
      room walls share one color and there are 3-5 separate rooms using that same wall color.
    - The single legend on the border uses the shared wall color as its fill and encodes one
      movement direction via the center-versus-indicator offset.

    Rule:
    - Detect every enclosed passable region, infer its border color from adjacent wall cells,
      and look up the legend direction for that color.
    - Since all rooms use the same wall color, every yellow pixel in every room slides in the
      same direction and stacks against the far wall of its own room.
    - After all movement is resolved, crop the result to the minimal rectangle containing all
      enclosed regions together with their wall boundaries, discarding the legend and padding.

    Key insight:
    - Color matching becomes globally shared: the one legend direction applies to all rooms of
      the same wall color, so the main extra task is to separate the multiple enclosed regions
      and then crop to their combined footprint.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    mover = 4
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one border legend"

    regions = _bounded_regions(g, bg, mover)
    out = g.copy()

    for reg in regions:
        movers = reg["movers"]
        border_col = reg["border_col"]
        if not movers or border_col not in dirs:
            continue
        dr, dc = dirs[border_col]
        for r, c in movers:
            out[r, c] = bg
        for r, c in _slide_points(reg["pset"], movers, dr, dc):
            out[r, c] = mover

    r0, c0, r1, c1 = _crop_bbox_from_regions(regions)
    return out[r0 : r1 + 1, c0 : c1 + 1]


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


def _legend_corner(side: str, rng: random.Random) -> str:
    if side == "top":
        return rng.choice(["TL", "TR"])
    if side == "bottom":
        return rng.choice(["BL", "BR"])
    if side == "left":
        return rng.choice(["TL", "BL"])
    if side == "right":
        return rng.choice(["TR", "BR"])
    raise ValueError(side)


def _side_safe_mask(shape: tuple[int, int], side: str, patch_shape: tuple[int, int]) -> np.ndarray:
    """Cells where region interiors may live so the crop bbox stays away from the legend."""
    h, w = shape
    ph, pw = patch_shape
    mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        start = ph + LEGEND_BUFFER + 2
        stop = h - 2
        if start < stop:
            mask[start:stop, 2 : w - 2] = True
    elif side == "bottom":
        start = 2
        stop = h - ph - LEGEND_BUFFER - 2
        if start < stop:
            mask[start:stop, 2 : w - 2] = True
    elif side == "left":
        start = pw + LEGEND_BUFFER + 2
        stop = w - 2
        if start < stop:
            mask[2 : h - 2, start:stop] = True
    elif side == "right":
        start = 2
        stop = w - pw - LEGEND_BUFFER - 2
        if start < stop:
            mask[2 : h - 2, start:stop] = True
    else:
        raise ValueError(side)
    return mask


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
    walls = set()
    for r, c in blob:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in blob:
                walls.add((nr, nc))
    return walls


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


def _place_legend(
    rng: random.Random,
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]], np.ndarray]:
    """Place exactly one legend and return occupied cells plus the safe region mask."""
    h, w = grid.shape
    side = rng.choice(["top", "bottom", "left", "right"])
    corner = _legend_corner(side, rng)
    wall_color = rng.choice(WALL_COLOR_CHOICES)

    if rng.random() < CARDINAL_PROB:
        dr, dc = rng.choice(CARDINAL_DIRS)
        patch = _legend_5x5(wall_color, dr, dc)
    else:
        dr, dc = rng.choice(DIAGONAL_DIRS)
        patch = _legend_6x7(wall_color, dr, dc)

    ph, pw = patch.shape
    r0, c0 = _corner_origin(corner, h, w, ph, pw)
    paste(grid, patch, r0, c0)

    legend_occupied = np.zeros((h, w), dtype=bool)
    r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
    c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
    legend_occupied[r_lo:r_hi, c_lo:c_hi] = True

    safe_mask = _side_safe_mask((h, w), side, patch.shape)
    safe_mask &= ~legend_occupied
    return legend_occupied, [(wall_color, (dr, dc))], safe_mask


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


def _grow_regions(
    rng: random.Random,
    legend_occupied: np.ndarray,
    legends: list[tuple[int, tuple[int, int]]],
    safe_mask: np.ndarray,
) -> list[dict[str, object]] | None:
    h, w = legend_occupied.shape
    wall_color, direction = legends[0]
    n_regions = rng.randint(*N_REGIONS_RANGE)
    size_min, size_max = REGION_SIZE_BOUNDS[n_regions]

    occupied = legend_occupied.copy()
    regions: list[dict[str, object]] = []

    for _ in range(n_regions):
        target_size = rng.randint(size_min, size_max)
        canvas = safe_mask.copy()
        canvas &= ~occupied
        canvas[0, :] = False
        canvas[-1, :] = False
        canvas[:, 0] = False
        canvas[:, -1] = False
        if not canvas.any():
            return None

        blob = _try_grow_valid_blob(rng, canvas, occupied, target_size, h, w)
        if blob is None:
            return None

        walls = _blob_wall_cells(blob, h, w)
        regions.append({"color": wall_color, "dir": direction, "interior": blob, "walls": walls})

        reserve = blob | walls
        extra = set()
        for r, c in reserve:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reserve:
                    extra.add((nr, nc))
        for r, c in reserve | extra:
            occupied[r, c] = True

    return regions if len(regions) == n_regions else None


def _validate_topology(grid: np.ndarray, regions: list[dict[str, object]]) -> bool:
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


def _place_yellows(rng: random.Random, regions: list[dict[str, object]], grid: np.ndarray) -> bool:
    """Place 1-2 yellows in every region using reverse simulation from output positions."""
    for reg in regions:
        blob = reg["interior"]
        dr, dc = reg["dir"]
        n_yellows = 1 + int(len(blob) >= 12 and rng.random() < 0.35)

        cells_sorted = sorted(blob, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
        far_key = dr * cells_sorted[0][0] + dc * cells_sorted[0][1]
        tier1 = [p for p in cells_sorted if dr * p[0] + dc * p[1] == far_key]

        first = tier1[rng.randrange(len(tier1))]
        output_positions = [first]
        if n_yellows == 2:
            pr, pc = first[0] - dr, first[1] - dc
            cands = []
            if (pr, pc) in blob:
                cands.append((pr, pc))
            cands.extend(p for p in tier1 if p != first)
            if cands:
                second = cands[rng.randrange(len(cands))]
                if second != first:
                    output_positions.append(second)

        out_sorted = sorted(output_positions, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
        stopped = set()
        inputs = []
        for out_pos in out_sorted:
            candidate = _find_reverse_slide(blob, out_pos, dr, dc, stopped, inputs, rng)
            if candidate is None:
                return False
            inputs.append(candidate)
            stopped.add(out_pos)

        for r, c in inputs:
            grid[r, c] = 4
    return True


def _regions_crop_bbox(regions: list[dict[str, object]]) -> tuple[int, int, int, int]:
    coords = []
    for reg in regions:
        coords.extend(reg["interior"])
        coords.extend(reg["walls"])
    return bbox_coords(np.array(coords, dtype=int))


def _expected_output(input_grid: np.ndarray, regions: list[dict[str, object]]) -> np.ndarray | None:
    try:
        solver_out = main(input_grid)
    except Exception:
        return None

    full_out = input_grid.copy()
    bg = get_bg(input_grid)
    for reg in regions:
        blob = reg["interior"]
        dr, dc = reg["dir"]
        movers = [(r, c) for r, c in blob if full_out[r, c] == 4]
        for r, c in movers:
            full_out[r, c] = bg
        for r, c in _slide_points(blob, movers, dr, dc):
            full_out[r, c] = 4

    r0, c0, r1, c1 = _regions_crop_bbox(regions)
    expected = full_out[r0 : r1 + 1, c0 : c1 + 1]
    return expected if np.array_equal(solver_out, expected) else None


def _oracle_check(input_grid: np.ndarray, regions: list[dict[str, object]]) -> bool:
    expected = _expected_output(input_grid, regions)
    if expected is None:
        return False
    if expected.shape == input_grid.shape:
        return False
    if not (N_REGIONS_RANGE[0] <= len(regions) <= N_REGIONS_RANGE[1]):
        return False
    wall_colors = {reg["color"] for reg in regions}
    dirs = {reg["dir"] for reg in regions}
    if len(wall_colors) != 1 or len(dirs) != 1:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)

    legend_occupied, legends, safe_mask = _place_legend(rng, grid)
    regions = _grow_regions(rng, legend_occupied, legends, safe_mask)
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
        try:
            g = _gen_attempt(rng)
        except Exception:
            continue
        if g is not None:
            return g
    raise RuntimeError("generate_input: exceeded retries")
