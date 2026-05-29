# Combo variant of 8b9c3697 (index 25)
# combo_spec: D3/3+D10/0
# Name: 堆叠结构多对一填充

from __future__ import annotations

import random
import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo:
# a) Concrete behavior:
#    The original task has one comb-like structure matched with at most one red blob.
#    This combo replaces that with one large "shelf" structure that contains 2-3
#    parallel slot lanes stacked along the orthogonal axis. Each lane is open on the
#    same side, so multiple red bars can independently slide into the same structure.
#    Example in normalized right-opening form:
#      #######
#      #......
#      #######
#      #......
#      #######
#    The odd rows are slot lanes; one red bar sits outside each lane on the right.
#    In the output, every matched bar slides left into the corresponding lane and its
#    swept cells turn black. Distractor reds vanish.
# b) Complexity calibration:
#    Using 2-3 repeated lanes keeps the multi-slot effect visually obvious without
#    introducing too many slot types. The same big structure color is reused across
#    all lanes, so the human-visible novelty is "several reds feed one structure".
# c) Broken implicit assumptions from the original solver:
#    - "one structure -> one red" is no longer valid.
#    - "choose one best anchor per structure" is no longer valid.
#    - The opening direction is inferred from structure geometry alone, then all slot
#      anchors on that side are processed.
#    - The generator must reserve each structure patch's full bbox; otherwise a second
#      structure could land inside another structure's slide corridor and invalidate
#      the new multi-slot assumptions.
# d) Placement legality:
#    Structures are placed by reserving their entire rectangular patch footprint, not
#    just the visible non-bg cells. This prevents overlap, ray obstruction, and
#    out-of-bounds slides. Distractors are kept off all slot sight-lines.
# e) Visual observability:
#    In 2-4 examples the solver pattern is recoverable by eye: a large shelf/bracket
#    shape has several parallel openings, and each visible red bar on those openings
#    moves inward leaving a black trail. This is visibly different from the original
#    one-red-per-structure behavior.
# f) Visual cleanliness:
#    Slot lanes are one cell thick and separated by solid divider rows, so the lanes,
#    final reds, and black trails remain readable. Full-bbox structure spacing avoids
#    different objects mixing in the same local area.
# g) New solver assumptions audited against the generator:
#    - Every matched red is a thin rectangle aligned with one slot lane.
#    - Only one structure direction has >=2 clear slot anchors.
#    - No distractor lies on any slot ray before the intended matched red.
#    - A red component is not shared by multiple structures.
#    The generator explicitly enforces all four.


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _slot_groups_for_dir(
    grid: np.ndarray,
    struct_cells: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    dirv: tuple[int, int],
    bg: int,
) -> list[tuple[int, int]]:
    """Return one slot-anchor cell per lane for a candidate opening direction."""
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    dr, dc = dirv
    groups: dict[int, list[tuple[int, int]]] = {}

    for r, c in struct_cells:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if int(grid[nr, nc]) != int(bg):
            continue
        if not (r0 <= nr <= r1 and c0 <= nc <= c1):
            continue

        rr, cc = nr, nc
        clear = True
        while r0 <= rr <= r1 and c0 <= cc <= c1:
            if int(grid[rr, cc]) != int(bg):
                clear = False
                break
            rr += dr
            cc += dc
        if not clear:
            continue

        key = int(nr if dr == 0 else nc)
        groups.setdefault(key, []).append((int(nr), int(nc)))

    anchors: list[tuple[int, int]] = []
    for key in sorted(groups):
        pts = groups[key]
        if dr != 0:
            pts.sort(key=lambda p: p[0])
        else:
            pts.sort(key=lambda p: p[1])
        anchors.append((int(pts[0][0]), int(pts[0][1])))
    return anchors


def _best_structure_view(
    grid: np.ndarray,
    struct_cells: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    bg: int,
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    """Choose the unique opening side by the number of valid slot lanes."""
    best_dir = None
    best_slots: list[tuple[int, int]] = []
    for dirv in _DIRS:
        slots = _slot_groups_for_dir(grid, struct_cells, bbox, dirv, bg)
        if len(slots) > len(best_slots):
            best_dir = (int(dirv[0]), int(dirv[1]))
            best_slots = slots
    if best_dir is None or not best_slots:
        return None
    return best_dir, best_slots


def _first_red_on_ray(
    grid: np.ndarray,
    start: tuple[int, int],
    dirv: tuple[int, int],
    bg: int,
    red_at: dict[tuple[int, int], int],
) -> tuple[int | None, int | None]:
    """Scan outward from one slot lane and return the nearest visible red component."""
    h, w = grid.shape
    dr, dc = dirv
    rr, cc = start
    dist = 0
    while 0 <= rr < h and 0 <= cc < w:
        val = int(grid[rr, cc])
        if val == int(bg):
            rr += dr
            cc += dc
            dist += 1
            continue
        if val == int(Color.RED):
            return int(red_at[(int(rr), int(cc))]), int(dist)
        return None, None
    return None, None


def _collect_structure_assignments(
    grid: np.ndarray,
    bg: int,
    red_at: dict[tuple[int, int], int],
) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    """Return (rid, dir, slot_anchor) assignments for every visible lane match."""
    red_val = int(Color.RED)
    struct_mask = (grid != int(bg)) & (grid != red_val)
    struct_comps = components_mask(struct_mask, connectivity=4)
    assignments: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    used_red: set[int] = set()

    for coords in struct_comps:
        pts = {(int(r), int(c)) for r, c in coords}
        bbox = tuple(map(int, bbox_coords(coords)))
        view = _best_structure_view(grid, pts, bbox, bg)
        if view is None:
            continue
        dirv, slots = view
        for slot in slots:
            rid, _dist = _first_red_on_ray(grid, slot, dirv, bg, red_at)
            if rid is None or int(rid) in used_red:
                continue
            used_red.add(int(rid))
            assignments.append((int(rid), (int(dirv[0]), int(dirv[1])), (int(slot[0]), int(slot[1]))))
    return assignments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide multiple red bars into the stacked lanes of a larger open-sided structure.

    Input structure:
    - Background is the most common color.
    - Non-background, non-red pixels form 1-2 large shelf/bracket components.
    - Each matched structure contains 2-3 parallel slot lanes stacked vertically or
      horizontally, depending on rotation.
    - Red rectangles sit outside the opening side; distractor reds may appear elsewhere.

    Rule:
    - Infer each structure's opening direction from the side that exposes several
      interior lane anchors to the bounding-box edge.
    - For every slot lane, look straight outward along that opening direction.
    - If the first visible non-bg object on the ray is a red component, slide that
      red bar into the lane and paint all intermediate positions black.
    - Reds that are not the first visible match for any lane disappear.

    Key insight:
    - The important change from the original task is cardinality: one structure is no
      longer paired with only one red. The geometry now exposes several repeated slot
      lanes on the same side, so the solver must process every lane of the structure,
      not just the single best ray hit.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

    red_val = int(Color.RED)
    black_val = int(Color.BLACK)

    assert int(red_val) in set(map(int, np.unique(g)))

    red_comps = components_mask(g == int(red_val), connectivity=4)
    assert red_comps

    red_coords: list[list[tuple[int, int]]] = []
    red_bbox: list[tuple[int, int, int, int]] = []
    red_at: dict[tuple[int, int], int] = {}

    for rid, coords in enumerate(red_comps):
        pts = [(int(r), int(c)) for r, c in coords]
        red_coords.append(pts)
        red_bbox.append(tuple(map(int, bbox_coords(coords))))
        for rc in pts:
            red_at[rc] = int(rid)

    assignments = _collect_structure_assignments(g, bg, red_at)

    out = g.copy()
    out[out == int(red_val)] = int(bg)

    for rid, (dr, dc), (tr, tc) in assignments:
        coords = red_coords[int(rid)]
        rr0, cc0, rr1, cc1 = red_bbox[int(rid)]

        if dr == 0 and dc == 1:
            shift_r = int(tr - rr0)
            shift_c = int(tc - cc0)
        elif dr == 0 and dc == -1:
            shift_r = int(tr - rr0)
            shift_c = int(tc - cc1)
        elif dr == 1 and dc == 0:
            shift_r = int(tr - rr0)
            shift_c = int(tc - cc0)
        else:
            shift_r = int(tr - rr1)
            shift_c = int(tc - cc0)

        steps = int(max(abs(shift_r), abs(shift_c)))
        step_dr = 0 if shift_r == 0 else (1 if shift_r > 0 else -1)
        step_dc = 0 if shift_c == 0 else (1 if shift_c > 0 else -1)

        for step in range(steps):
            off_r = int(step * step_dr)
            off_c = int(step * step_dc)
            for r, c in coords:
                pr = int(r + off_r)
                pc = int(c + off_c)
                if 0 <= pr < h and 0 <= pc < w:
                    out[pr, pc] = int(black_val)

        for r, c in coords:
            pr = int(r + shift_r)
            pc = int(c + shift_c)
            if 0 <= pr < h and 0 <= pc < w:
                out[pr, pc] = int(red_val)

    return out


# --------------- generator ---------------

GRID_H_RANGE = (22, 28)
GRID_W_RANGE = (24, 32)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

STRUCT_COUNT_RANGE = (1, 2)
UNMATCHED_PROB = 0.25
STRUCT_PADDING = 2

LAYERS_RANGE = (2, 3)
DEPTH_RANGE = (5, 6)
DOCK_LEN_OPTIONS = (2, 3)
TRAVEL_BASE_RANGE = (6, 8)
TRAVEL_GAP_RANGE = (2, 3)

DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1))
DISTRACTOR_COUNT_RANGE = (2, 5)

MAX_RETRIES = 400


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _structure_array(layers: int, depth: int, bg: int, sc: int) -> tuple[np.ndarray, list[int]]:
    """Normalized structure opens to the right; odd rows are slot lanes."""
    h = 2 * int(layers) + 1
    w = int(depth) + 2
    arr = np.full((h, w), bg, dtype=int)
    for r in range(h):
        if r % 2 == 0:
            arr[r, :] = sc
        else:
            arr[r, 0] = sc
    return arr, [1 + 2 * i for i in range(int(layers))]


def _sample_travels(count: int, min_travel: int) -> list[int]:
    base = random.randint(max(int(min_travel), TRAVEL_BASE_RANGE[0]), max(int(min_travel), TRAVEL_BASE_RANGE[1]))
    travels = [int(base)]
    for _ in range(1, int(count)):
        travels.append(int(travels[-1] + random.randint(*TRAVEL_GAP_RANGE)))
    return travels


def _build_structure_spec(layers: int, depth: int, dock_len: int, rot_k: int, matched: bool, bg: int, sc: int) -> dict:
    """Create one large stacked structure, optionally with one matched red per lane."""
    structure, slot_rows = _structure_array(layers, depth, bg, sc)
    sh, sw = structure.shape

    if matched:
        travels = _sample_travels(layers, sw - 1)
        pw = max(sw, 1 + max(travels) + int(dock_len))
        input_patch = np.full((sh, pw), bg, dtype=int)
        paste(input_patch, structure, 0, 0, transparent=bg)
        for row, travel in zip(slot_rows, travels):
            c0 = 1 + int(travel)
            input_patch[int(row), c0:c0 + int(dock_len)] = int(Color.RED)
        structure_patch = np.full((sh, pw), bg, dtype=int)
        paste(structure_patch, structure, 0, 0, transparent=bg)
        expected_matches = int(layers)
    else:
        input_patch = structure.copy()
        structure_patch = structure.copy()
        expected_matches = 0

    return {
        "input": np.rot90(input_patch, k=int(rot_k)),
        "structure": np.rot90(structure_patch, k=int(rot_k)),
        "matched_slots": int(expected_matches),
    }


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int) -> tuple[int, int] | None:
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _place_structures(specs: list[dict], h: int, w: int, bg: int, sc: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Reserve each structure's full footprint to keep slide corridors unobstructed."""
    inp = np.full((h, w), bg, dtype=int)
    struct_grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)

    for spec in specs:
        box_patch = np.full(spec["input"].shape, sc, dtype=int)
        loc = _place_patch(reserve, box_patch, bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        r0, c0 = loc
        paste(reserve, box_patch, r0, c0, transparent=bg)
        paste(inp, spec["input"], r0, c0, transparent=bg)
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)

    return inp, struct_grid


def _structure_views_from_grid(grid: np.ndarray, bg: int) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    """Extract geometry-only views from the structure grid."""
    struct_mask = grid != int(bg)
    comps = components_mask(struct_mask, connectivity=4)
    views: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for coords in comps:
        pts = {(int(r), int(c)) for r, c in coords}
        bbox = tuple(map(int, bbox_coords(coords)))
        view = _best_structure_view(grid, pts, bbox, bg)
        if view is not None:
            views.append(view)
    return views


def _matched_slot_counts(grid: np.ndarray, bg: int) -> list[int]:
    """Count how many slot lanes see a red as their nearest outward object."""
    red_comps = components_mask(grid == int(Color.RED), connectivity=4)
    red_at: dict[tuple[int, int], int] = {}
    for rid, coords in enumerate(red_comps):
        for r, c in coords:
            red_at[(int(r), int(c))] = int(rid)

    counts: list[int] = []
    struct_mask = (grid != int(bg)) & (grid != int(Color.RED))
    struct_comps = components_mask(struct_mask, connectivity=4)
    for coords in struct_comps:
        pts = {(int(r), int(c)) for r, c in coords}
        bbox = tuple(map(int, bbox_coords(coords)))
        view = _best_structure_view(grid, pts, bbox, bg)
        if view is None:
            continue
        dirv, slots = view
        seen: set[int] = set()
        count = 0
        for slot in slots:
            rid, _dist = _first_red_on_ray(grid, slot, dirv, bg, red_at)
            if rid is None or int(rid) in seen:
                continue
            seen.add(int(rid))
            count += 1
        counts.append(int(count))
    return counts


def _slot_rays_mask(grid: np.ndarray, struct_grid: np.ndarray, bg: int) -> np.ndarray:
    """Mark all clear lane sight-lines so distractors cannot fake a match."""
    h, w = grid.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for dirv, slots in _structure_views_from_grid(struct_grid, bg):
        dr, dc = dirv
        for sr, sc in slots:
            rr, cc = sr, sc
            while 0 <= rr < h and 0 <= cc < w:
                if int(struct_grid[rr, cc]) != int(bg):
                    break
                forbidden[rr, cc] = True
                rr += dr
                cc += dc
    return forbidden


def _place_distractors(inp: np.ndarray, struct_grid: np.ndarray, bg: int, sc: int, count: int) -> bool:
    forbidden = _slot_rays_mask(inp, struct_grid, bg)
    red_mask = inp == int(Color.RED)
    struct_block = struct_grid != int(bg)

    shapes = list(DISTRACTOR_SHAPES)
    for _ in range(int(count)):
        blocked = forbidden | struct_block | (inp != int(bg)) | _dilate4(red_mask)
        canvas = np.full(inp.shape, bg, dtype=int)
        canvas[blocked] = sc
        random.shuffle(shapes)
        placed = False
        for rh, rw in shapes:
            patch = np.full((rh, rw), int(Color.RED), dtype=int)
            loc = _place_patch(canvas, patch, bg, padding=0)
            if loc is None:
                continue
            paste(inp, patch, loc[0], loc[1], transparent=bg)
            red_mask[loc[0]:loc[0] + rh, loc[1]:loc[1] + rw] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(Color.RED), int(Color.BLACK))])
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_structs = random.randint(*STRUCT_COUNT_RANGE)
    unmatched = 1 if n_structs > 1 and random.random() < UNMATCHED_PROB else 0
    matched_structs = n_structs - unmatched
    if matched_structs <= 0:
        return None

    specs: list[dict] = []
    expected_counts: list[int] = []
    for idx in range(n_structs):
        layers = random.randint(*LAYERS_RANGE)
        depth = random.randint(*DEPTH_RANGE)
        dock_len = random.choice(DOCK_LEN_OPTIONS)
        rot_k = random.randrange(4)
        matched = idx < matched_structs
        spec = _build_structure_spec(layers, depth, dock_len, rot_k, matched, bg, sc)
        specs.append(spec)
        expected_counts.append(int(spec["matched_slots"]))

    random.shuffle(specs)
    specs.sort(key=lambda s: s["input"].shape[0] * s["input"].shape[1], reverse=True)
    expected_counts = sorted((int(s["matched_slots"]) for s in specs), reverse=True)

    placed = _place_structures(specs, h, w, bg, sc)
    if placed is None:
        return None
    inp, struct_grid = placed

    counts_before = sorted(_matched_slot_counts(inp, bg), reverse=True)
    if counts_before != expected_counts:
        return None
    if max(counts_before, default=0) < 2:
        return None

    distractor_count = random.randint(*DISTRACTOR_COUNT_RANGE)
    if not _place_distractors(inp, struct_grid, bg, sc, distractor_count):
        return None

    counts_after = sorted(_matched_slot_counts(inp, bg), reverse=True)
    if counts_after != expected_counts:
        return None

    try:
        _ = main(inp)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for combo variant 8b9c3697/25")
