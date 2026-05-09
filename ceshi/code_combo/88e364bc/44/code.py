# Combo variant of 88e364bc (index 44)
# combo_spec: D2/3+D6/0
# Name: 墙壁厚度匹配+规则矩形区域

"""
Design analysis for the fused variant.

a) Concrete behavior trace
   - Original task:
     - Border legends are filled rectangles whose dominant fill color matches a room wall color.
     - Interior rooms are irregular fenced blobs.
     - A yellow in wall color C uses the legend with fill color C, then slides to the far side.
   - Fused task:
     - All rooms are strict axis-aligned rectangles.
     - Every room wall uses the same color, so color matching is gone.
     - Instead, room wall thickness is either 1 or 2, and each thickness matches the border legend
       whose own fill frame has the same thickness.
     - Example: a thin-walled room (1-cell wall) may use a 5x5 legend with 1-cell fill margin and
       move right, while a thick-walled room (2-cell wall) uses a 7x7 legend with 2-cell margin
       and moves up. Both rooms can share the same wall/fill color, so thickness is the only cue.
   - Difference:
     - The original "majority adjacent wall color -> legend color" link is invalid.
     - The fused solver must detect a room rectangle, measure its wall thickness, and map that
       thickness to the matching legend before sliding.

b) Complexity calibration
   - Thickness is restricted to exactly two visually measurable values: 1 and 2.
   - Every sample contains both thicknesses, and the two legends always encode different
     directions, so the new cue is necessary rather than decorative.
   - To keep the rule learnable after adding thickness matching, directions are restricted to the
     four cardinal moves. This keeps the legend family compact and leaves board space for several
     rooms.
   - The generator may place multiple rooms per thickness, but it always guarantees at least one
     thin room and one thick room in the same grid, so the contrast is explicit in every example.

c) Implicit assumption audit against the original main()
   - Original `_legend_directions(...)` keyed directions by dominant legend fill color. That fails
     here because both legends deliberately share the same fill color.
   - Original region processing inferred a room's controlling legend from adjacent wall colors.
     That also fails because all walls share one color.
   - The original passable-component scan still works, but the solver now additionally assumes each
     enclosed component is a full rectangle. The generator guarantees that by using only rectangular
     interiors.
   - The new solver assumes the wall around each interior rectangle is a complete 1-cell or 2-cell
     rectangular ring. The generator paints exactly those rings and reserves a moat so nearby walls
     cannot fake a thicker ring.
   - Stack order still matters when multiple yellows share a room, so reverse placement follows the
     same far-to-near order as the solver.

d) Placement legality
   - Legends are placed on distinct corners and buffered by one background cell.
   - Each room reserves its full outer rectangle plus a 1-cell moat, so rooms never overlap,
     never merge into one wall mass, and never touch legends.
   - Room outer rectangles stay away from the border, ensuring enclosed passable components never
     connect to the outside background.

e) Visual observability
   - Humans can directly compare "thin room vs thick room" because room geometry is rectangular and
     wall thickness is uniform around the whole rectangle.
   - The matching legend difference is also visible: one legend is a smaller filled rectangle with
     a 1-cell fill frame, the other is a larger one with a 2-cell frame.
   - The generator rejects samples where a legacy color-only interpretation would produce the same
     output, so the fused behavior is materially exercised.

f) Visual cleanliness
   - Rooms are simple rectangles with one wall color, yellow movers, and black interior.
   - Legends stay in corners; rooms stay in the open field with background moats between them.
   - No distractor markings are added inside rooms, so thickness and movement remain easy to read.

g) New solver assumptions and generator guarantees
   - Assumption: valid border legends are solid rectangles with exactly three colors. Guaranteed.
   - Assumption: legend frame thickness is unambiguously 1 or 2. Guaranteed by fixed 5x5 and 7x7
     legend templates.
   - Assumption: every yellow-containing room has thickness 1 or 2 and that thickness appears in
     the legend map. Guaranteed because the generator always places both legend types and only
     produces those two room types.
   - Assumption: the fused rule is non-degenerate relative to the original color-only rule.
     Guaranteed by requiring the two legend directions to differ and by rejecting samples whose
     fused output matches the color-only baseline.
"""

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste


GRID_H = 20
GRID_W = 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
THICKNESSES = [1, 2]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]

LEGEND_BUFFER = 1
ROOM_COUNT_BY_THICKNESS = {1: (1, 2), 2: (1, 2)}
ROOM_SIZE_BY_THICKNESS = {
    1: ((3, 8), (3, 8)),
    2: ((3, 6), (3, 6)),
}
ROOM_PLACE_RETRIES = 120
YELLOWS_PER_ROOM = [1, 2]
MAX_SLIDE_STEPS = 5
GEN_RETRIES = 500

LAST_GENERATION_INFO: dict[str, int] = {}


def _projection(pt: tuple[int, int], dr: int, dc: int) -> int:
    return dr * int(pt[0]) + dc * int(pt[1])


def _legend_patch(fill: int, thickness: int, dr: int, dc: int) -> np.ndarray:
    """Create a cardinal legend whose visible fill-frame thickness is 1 or 2."""
    size = 3 + 2 * thickness
    g = np.full((size, size), fill, dtype=int)
    r0 = thickness
    c0 = thickness
    g[r0 : r0 + 2, c0 : c0 + 2] = 1
    g[r0 : r0 + 2, c0 + 2] = 2
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


def _legend_records(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    """Extract border legends and decode their direction plus visible frame thickness."""
    h, w = g.shape
    records: list[dict[str, object]] = []
    for pts in components_mask(g != bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        if (r1 - r0 + 1) * (c1 - c0 + 1) != len(pts):
            continue

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        fill_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != fill_col]
        others.sort()
        if len(others) != 2:
            continue
        if others[0][0] == others[1][0]:
            rect_center = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            tie_break = []
            for _, col in others:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                tie_break.append((float(np.sum((pts_col.mean(axis=0) - rect_center) ** 2)), col))
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
        dr = 0 if abs(float(delta[0])) < 1e-6 else (1 if float(delta[0]) > 0 else -1)
        dc = 0 if abs(float(delta[1])) < 1e-6 else (1 if float(delta[1]) > 0 else -1)
        if (dr, dc) not in CARDINAL_DIRS:
            continue

        non_fill = [(int(r), int(c)) for r, c in pts if int(g[r, c]) != fill_col]
        frame = min(
            min(r - r0, r1 - r, c - c0, c1 - c)
            for r, c in non_fill
        )
        if frame not in THICKNESSES:
            continue

        records.append(
            {
                "fill": int(fill_col),
                "frame": int(frame),
                "dir": (int(dr), int(dc)),
                "area": int(len(pts)),
            }
        )
    return records


def _legend_dirs_by_thickness(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for rec in _legend_records(g, bg):
        frame = int(rec["frame"])
        prev = best.get(frame)
        if prev is None or int(rec["area"]) > prev[0]:
            best[frame] = (int(rec["area"]), tuple(rec["dir"]))  # type: ignore[arg-type]
    return {k: v for k, (_, v) in best.items()}


def _legend_dirs_by_fill(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """Legacy baseline: collapse legends by dominant fill color as in the original task."""
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for rec in _legend_records(g, bg):
        fill = int(rec["fill"])
        prev = best.get(fill)
        if prev is None or int(rec["area"]) > prev[0]:
            best[fill] = (int(rec["area"]), tuple(rec["dir"]))  # type: ignore[arg-type]
    return {k: v for k, (_, v) in best.items()}


def _slide_points(region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    if not points:
        return []
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


def _ring_cells(inner_bbox: tuple[int, int, int, int], k: int) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = inner_bbox
    rr0, cc0, rr1, cc1 = r0 - k, c0 - k, r1 + k, c1 + k
    cells: set[tuple[int, int]] = set()
    for c in range(cc0, cc1 + 1):
        cells.add((rr0, c))
        cells.add((rr1, c))
    for r in range(rr0, rr1 + 1):
        cells.add((r, cc0))
        cells.add((r, cc1))
    return cells


def _infer_room_thickness(g: np.ndarray, inner_bbox: tuple[int, int, int, int], bg: int, mover: int) -> tuple[int, int] | None:
    h, w = g.shape
    ring1 = _ring_cells(inner_bbox, 1)
    if any(not (0 <= r < h and 0 <= c < w) for r, c in ring1):
        return None
    colors1 = {int(g[r, c]) for r, c in ring1}
    if len(colors1) != 1:
        return None
    wall_color = next(iter(colors1))
    if wall_color in (bg, mover):
        return None

    thickness = 1
    ring2 = _ring_cells(inner_bbox, 2)
    if all(0 <= r < h and 0 <= c < w for r, c in ring2):
        if all(int(g[r, c]) == wall_color for r, c in ring2):
            thickness = 2
    return thickness, wall_color


def _solve_with_mapping(grid: np.ndarray, dirs_by_thickness: dict[int, tuple[int, int]] | None = None,
                        dirs_by_fill: dict[int, tuple[int, int]] | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    mover = 4
    passable = (g == bg) | (g == mover)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue

        region_set = {(int(r), int(c)) for r, c in pts}
        inferred = _infer_room_thickness(g, (int(r0), int(c0), int(r1), int(c1)), bg, mover)
        if inferred is None:
            continue
        thickness, wall_color = inferred

        if dirs_by_thickness is not None:
            if thickness not in dirs_by_thickness:
                continue
            dr, dc = dirs_by_thickness[thickness]
        else:
            assert dirs_by_fill is not None
            if wall_color not in dirs_by_fill:
                continue
            dr, dc = dirs_by_fill[wall_color]

        for r, c in movers:
            out[r, c] = bg
        for r, c in _slide_points(region_set, movers, dr, dc):
            out[r, c] = mover

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellows inside rectangular rooms using thickness-matched border legends.

    Input structure:
    - The board is a 20x20 black-background grid with two border legends at different corners.
    - All room walls and both legend fills use the same non-black color.
    - Each enclosed room is a strict axis-aligned rectangle whose wall thickness is either 1 or 2.
    - Each legend is itself a filled rectangle; its visible fill-frame thickness is also 1 or 2,
      and its inner blue-vs-red offset encodes a cardinal slide direction.

    Rule:
    - Decode the direction of each border legend from the center-colored pixels to the red
      indicator pixels.
    - For every enclosed rectangular room, measure whether its wall is 1 cell or 2 cells thick.
    - Match the room to the legend with the same thickness, then slide all yellows in that room as
      far as possible in that legend's direction so they stack against the far wall.

    Key insight:
    - Color no longer identifies the controlling legend because all walls share one color. The
      decisive cue is thickness: thin rooms follow the thin-frame legend, and thick rooms follow
      the thick-frame legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    dirs = _legend_dirs_by_thickness(g, bg)
    assert dirs, "grid must contain at least one valid border legend"
    return _solve_with_mapping(g, dirs_by_thickness=dirs)


def _room_sets(region: dict[str, object]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    inner_r0, inner_c0, inner_r1, inner_c1 = region["inner_bbox"]  # type: ignore[misc]
    outer_r0, outer_c0, outer_r1, outer_c1 = region["outer_bbox"]  # type: ignore[misc]
    inner = {
        (r, c)
        for r in range(int(inner_r0), int(inner_r1) + 1)
        for c in range(int(inner_c0), int(inner_c1) + 1)
    }
    walls = {
        (r, c)
        for r in range(int(outer_r0), int(outer_r1) + 1)
        for c in range(int(outer_c0), int(outer_c1) + 1)
        if (r, c) not in inner
    }
    return inner, walls


def _place_legends(rng: random.Random, grid: np.ndarray, wall_color: int) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    h, w = grid.shape
    corners = rng.sample(CORNERS, 2)
    dirs = rng.sample(CARDINAL_DIRS, 2)
    legend_occupied = np.zeros((h, w), dtype=bool)
    thickness_to_dir = {1: dirs[0], 2: dirs[1]}

    for thickness, corner in zip(THICKNESSES, corners):
        patch = _legend_patch(wall_color, thickness, *thickness_to_dir[thickness])
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        rr0 = max(0, r0 - LEGEND_BUFFER)
        cc0 = max(0, c0 - LEGEND_BUFFER)
        rr1 = min(h, r0 + ph + LEGEND_BUFFER)
        cc1 = min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[rr0:rr1, cc0:cc1] = True

    return legend_occupied, thickness_to_dir


def _sample_room(rng: random.Random, occupied: np.ndarray, thickness: int) -> dict[str, object] | None:
    h, w = occupied.shape
    (hmin, hmax), (wmin, wmax) = ROOM_SIZE_BY_THICKNESS[thickness]

    for _ in range(ROOM_PLACE_RETRIES):
        inner_h = rng.randint(hmin, hmax)
        inner_w = rng.randint(wmin, wmax)
        outer_h = inner_h + 2 * thickness
        outer_w = inner_w + 2 * thickness
        max_r0 = h - outer_h - 1
        max_c0 = w - outer_w - 1
        if max_r0 < 1 or max_c0 < 1:
            return None

        outer_r0 = rng.randint(1, max_r0)
        outer_c0 = rng.randint(1, max_c0)
        outer_r1 = outer_r0 + outer_h - 1
        outer_c1 = outer_c0 + outer_w - 1
        inner_r0 = outer_r0 + thickness
        inner_c0 = outer_c0 + thickness
        inner_r1 = outer_r1 - thickness
        inner_c1 = outer_c1 - thickness

        moat_r0 = max(0, outer_r0 - 1)
        moat_c0 = max(0, outer_c0 - 1)
        moat_r1 = min(h - 1, outer_r1 + 1)
        moat_c1 = min(w - 1, outer_c1 + 1)
        if occupied[moat_r0 : moat_r1 + 1, moat_c0 : moat_c1 + 1].any():
            continue

        room = {
            "thickness": thickness,
            "inner_bbox": (inner_r0, inner_c0, inner_r1, inner_c1),
            "outer_bbox": (outer_r0, outer_c0, outer_r1, outer_c1),
        }
        return room
    return None


def _place_rooms(rng: random.Random, legend_occupied: np.ndarray) -> list[dict[str, object]] | None:
    occupied = legend_occupied.copy()
    rooms: list[dict[str, object]] = []

    thicknesses = []
    for t in THICKNESSES:
        count = rng.randint(*ROOM_COUNT_BY_THICKNESS[t])
        thicknesses.extend([t] * count)
    rng.shuffle(thicknesses)

    for thickness in thicknesses:
        room = _sample_room(rng, occupied, thickness)
        if room is None:
            return None
        rooms.append(room)
        outer_r0, outer_c0, outer_r1, outer_c1 = room["outer_bbox"]  # type: ignore[misc]
        rr0 = max(0, int(outer_r0) - 1)
        cc0 = max(0, int(outer_c0) - 1)
        rr1 = min(occupied.shape[0] - 1, int(outer_r1) + 1)
        cc1 = min(occupied.shape[1] - 1, int(outer_c1) + 1)
        occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

    return rooms


def _paint_rooms(grid: np.ndarray, rooms: list[dict[str, object]], wall_color: int) -> list[dict[str, object]]:
    painted = []
    for room in rooms:
        inner, walls = _room_sets(room)
        for r, c in walls:
            grid[r, c] = wall_color
        enriched = dict(room)
        enriched["inner"] = inner
        enriched["walls"] = walls
        painted.append(enriched)
    return painted


def _forward_slide(region_set: set[tuple[int, int]], start: tuple[int, int], dr: int, dc: int,
                   blocked: set[tuple[int, int]]) -> tuple[int, int]:
    r, c = start
    while True:
        nr, nc = r + dr, c + dc
        if (nr, nc) not in region_set or (nr, nc) in blocked:
            break
        r, c = nr, nc
    return r, c


def _find_reverse_slide(region_set: set[tuple[int, int]], out_pos: tuple[int, int], dr: int, dc: int,
                        already_stopped: set[tuple[int, int]], existing_inputs: list[tuple[int, int]],
                        rng: random.Random) -> tuple[int, int] | None:
    target_steps = rng.randint(1, MAX_SLIDE_STEPS)
    for steps in range(target_steps, 0, -1):
        r, c = out_pos
        ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in region_set or (pr, pc) in already_stopped or (pr, pc) in existing_inputs:
                ok = False
                break
            r, c = pr, pc
        if not ok:
            continue
        if _forward_slide(region_set, (r, c), dr, dc, already_stopped) == out_pos:
            return r, c
    return None


def _place_yellows(rng: random.Random, rooms: list[dict[str, object]], thickness_to_dir: dict[int, tuple[int, int]],
                   grid: np.ndarray) -> bool:
    for room in rooms:
        region_set = room["inner"]  # type: ignore[assignment]
        thickness = int(room["thickness"])
        dr, dc = thickness_to_dir[thickness]
        n_yellows = rng.choice(YELLOWS_PER_ROOM)

        cells_sorted = sorted(region_set, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
        far_key = _projection(cells_sorted[0], dr, dc)
        tier1 = [p for p in cells_sorted if _projection(p, dr, dc) == far_key]
        first = tier1[rng.randrange(len(tier1))]
        output_positions = [first]

        if n_yellows == 2:
            pr, pc = first[0] - dr, first[1] - dc
            cands = ([(pr, pc)] if (pr, pc) in region_set else []) + [p for p in tier1 if p != first]
            if cands:
                second = cands[rng.randrange(len(cands))]
                if second != first:
                    output_positions.append(second)

        out_sorted = sorted(output_positions, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
        already_stopped: set[tuple[int, int]] = set()
        input_positions: list[tuple[int, int]] = []

        for out_pos in out_sorted:
            candidate = _find_reverse_slide(region_set, out_pos, dr, dc, already_stopped, input_positions, rng)
            if candidate is None:
                return False
            input_positions.append(candidate)
            already_stopped.add(out_pos)

        for r, c in input_positions:
            grid[r, c] = 4
    return True


def _oracle_check(input_grid: np.ndarray) -> bool:
    try:
        out = main(input_grid)
    except Exception:
        return False
    struct_mask = (input_grid != 4) & (input_grid != 0)
    if not np.array_equal(out[struct_mask], input_grid[struct_mask]):
        return False
    return int((out == 4).sum()) == int((input_grid == 4).sum())


def fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    by_thickness = _legend_dirs_by_thickness(g, bg)
    by_fill = _legend_dirs_by_fill(g, bg)
    fused = _solve_with_mapping(g, dirs_by_thickness=by_thickness)
    baseline = _solve_with_mapping(g, dirs_by_fill=by_fill)

    passable = (g == bg) | (g == 4)
    room_count = 0
    thick1 = 0
    thick2 = 0
    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        if (r1 - r0 + 1) * (c1 - c0 + 1) != len(pts):
            continue
        inferred = _infer_room_thickness(g, (int(r0), int(c0), int(r1), int(c1)), bg, 4)
        if inferred is None:
            continue
        thickness, _wall_color = inferred
        room_count += 1
        thick1 += int(thickness == 1)
        thick2 += int(thickness == 2)

    return {
        "legend_count": len(by_thickness),
        "distinct_legend_dirs": len(set(by_thickness.values())),
        "room_count": room_count,
        "thin_rooms": thick1,
        "thick_rooms": thick2,
        "baseline_diff_cells": int(np.sum(fused != baseline)),
        "moved_yellows": int(np.sum((g == 4) & (fused != g))),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    wall_color = rng.choice(WALL_COLOR_CHOICES)

    legend_occupied, thickness_to_dir = _place_legends(rng, grid, wall_color)
    if len(set(thickness_to_dir.values())) != 2:
        return None

    rooms = _place_rooms(rng, legend_occupied)
    if rooms is None:
        return None
    rooms = _paint_rooms(grid, rooms, wall_color)

    if not _place_yellows(rng, rooms, thickness_to_dir, grid):
        return None
    if not _oracle_check(grid):
        return None

    metrics = fusion_metrics(grid)
    if metrics["legend_count"] != 2:
        return None
    if metrics["thin_rooms"] == 0 or metrics["thick_rooms"] == 0:
        return None
    if metrics["distinct_legend_dirs"] != 2:
        return None
    if metrics["baseline_diff_cells"] == 0:
        return None
    if metrics["moved_yellows"] == 0:
        return None

    LAST_GENERATION_INFO.clear()
    LAST_GENERATION_INFO.update(metrics)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries for combo D2/3+D6/0")
