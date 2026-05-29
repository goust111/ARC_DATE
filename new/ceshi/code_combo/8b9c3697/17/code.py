# Combo variant of 8b9c3697 (index 17)
# combo_spec: D1/3+D3/2+D5/1
# Name: 最近碰撞全保留

from __future__ import annotations

import os
import random
import sys

import numpy as np

_SOLVER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "solvers")
)
if _SOLVER_DIR not in sys.path:
    sys.path.append(_SOLVER_DIR)

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste


# Design analysis for combo D1/3 + D3/2 + D5/1
# ==================================================
# a) Concretization and behavior trace:
#    - Original 8b9c3697 used cavity combs. A structure first located an interior socket,
#      then ray-casting on the same row/column chose at most one aligned red rectangle,
#      and every other red disappeared.
#    - This combo removes both of those anchors. There is no socket target, and row/column
#      alignment is no longer required. Every red component independently looks for the
#      nearest structure by occupied-pixel distance.
#    - Concrete example: suppose a solid comb sits around rows 8..11, cols 10..16 and a
#      2x1 red is at rows 16..17, cols 19..19. The old task would ignore it because it is
#      not on a socket ray. The combo instead compares that red to all structure pixels,
#      finds the nearest pair near the south-east corner, moves the whole red by (-1,-1)
#      each step, paints the visited placements black, and stops at the last placement
#      before the next diagonal step would overlap the structure. A second red near the
#      same structure also moves; nothing is erased for being "unmatched".
# b) Complexity calibration:
#    - To keep the rule learnable, the generator preserves the original scale and colors,
#      uses two large comb/bracket structures from the original count range, and simplifies
#      the target geometry to the solid collision-stop form from D3/2.
#    - Red shapes are still small rectangles chosen from six repeated size classes. Motion
#      sectors are drawn from the eight cardinal/diagonal approach zones around each
#      structure, but the generator explicitly enforces a diagonal majority so the D1
#      change is still visually obvious without introducing too many path types.
#    - Total reds are always greater than the structure count, so the D5 effect is visible:
#      multiple reds head toward structures instead of only one red per structure surviving.
# c) Implicit assumptions audited against the original solver:
#    - Socket discovery inside a structure bbox is invalid after D3/2, so every socket and
#      anchor calculation is removed.
#    - The greedy "each red used once" assignment is invalid after D5/1 because every red
#      must move. The combo solver therefore plans motion per red component, not per
#      structure.
#    - Single-axis final alignment is invalid after D1/3. The combo solver allows a move
#      vector with both row and column components and paints a staircase-like black trail.
#    - Because red-red collision handling would create order dependence, the solver treats
#      structures as the only blockers. The generator therefore guarantees that final red
#      placements do not overlap and that no final red is hidden under another trail.
# d) Placement legality:
#    - Structures are placed first with padding. Reds are then placed in visible approach
#      sectors outside structure bboxes, always in-bounds and non-overlapping in the input.
#    - A candidate red is accepted only if solver-style nearest-structure analysis assigns
#      it to the intended structure and intended move vector, and only if its final output
#      cells stay distinct from previously reserved finals.
# e) Visual observability:
#    - Every generated red moves, so the disappearance rule from the base task is visibly
#      gone.
#    - A majority of generated moves are diagonal, so the "nearest structure" freedom is
#      visible from the input alone: many reds are not row/column aligned with a socket ray.
#    - Final red placements stay outside structures and stop by collision, making the D3/2
#      change visible in the output.
# f) Visual cleanliness:
#    - Red start cells are kept 4-connected separate.
#    - Final red placements are reserved to stay distinct, and trails are only prevented
#      from obscuring earlier final reds.
#    - Structures retain one shared non-red color and remain well spaced, so the scene
#      still reads as "large targets + many moving red rectangles".
# g) Change inferability:
#    - No hidden ordering by color id is used. The only matching signal is visible spatial
#      distance to large structures, and the only transform colors remain red for movers
#      and black for trails.
# h) New solver assumptions checked against the generator:
#    - Each red has a unique nearest structure and a unique nearest-direction signature.
#    - Each chosen move stops because of structure collision rather than boundary exit.
#    - No two final red placements share cells, and no final red is overwritten by another
#      trail, so independent per-red rendering is valid.


GRID_H_RANGE = (21, 23)
GRID_W_RANGE = (24, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

STRUCT_WIDTH_SPAN = ((7, 3), (8, 4), (9, 3), (9, 5))
STRUCT_COUNT_RANGE = (2, 2)
STRUCT_PADDING = 2

RED_SHAPES = ((1, 1), (1, 2), (2, 1))
TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 6
DIAGONAL_SECTORS = ("nw", "ne", "sw", "se")
AXIAL_SECTORS = ("n", "s", "w", "e")
GAP_RANGE = (3, 5)
GAP_OPTIONS = ((4, 4), (3, 4), (4, 3), (3, 3))
MAX_RETRIES = 400


def _sign(v: int) -> int:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _structure_array(width: int, span: int, bg: int, sc: int) -> np.ndarray:
    """Build the solid collision-stop comb used in the D3/2 reference."""
    g = np.full((4, width), bg, dtype=int)
    start = (int(width) - int(span)) // 2
    g[0, :] = sc
    g[1, 1] = sc
    g[2, 1] = sc
    g[1, width - 2] = sc
    g[2, width - 2] = sc
    g[1:4, start:start + span] = sc
    return g


def _extract_reds(
    grid: np.ndarray,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int, int, int]]]:
    red_val = int(Color.RED)
    red_comps = components_mask(grid == red_val, connectivity=4)
    red_coords: list[list[tuple[int, int]]] = []
    red_bbox: list[tuple[int, int, int, int]] = []
    for coords in red_comps:
        pts = [(int(r), int(c)) for r, c in coords]
        red_coords.append(pts)
        red_bbox.append(tuple(map(int, bbox_coords(coords))))
    return red_coords, red_bbox


def _extract_structures(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    red_val = int(Color.RED)
    struct_mask = (grid != int(bg)) & (grid != red_val)
    struct_comps = components_mask(struct_mask, connectivity=4)

    structures: list[dict] = []
    for sid, coords in enumerate(struct_comps):
        pts = [(int(r), int(c)) for r, c in coords]
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        structures.append(
            {
                "sid": int(sid),
                "coords": pts,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
            }
        )
    return structures, struct_mask


def _nearest_structure_for_red(red_cells: list[tuple[int, int]], structures: list[dict]) -> dict:
    best = None
    for item in structures:
        sid = int(item["sid"])
        for rr, cc in red_cells:
            for sr, sc in item["coords"]:
                delta_r = int(sr - rr)
                delta_c = int(sc - cc)
                dist = abs(delta_r) + abs(delta_c)
                if dist == 0:
                    continue
                step_r = _sign(delta_r)
                step_c = _sign(delta_c)
                cand = (
                    int(dist),
                    max(abs(delta_r), abs(delta_c)),
                    int(sid),
                    int(step_r),
                    int(step_c),
                )
                if best is None or cand < best:
                    best = cand

    assert best is not None, "expected a non-overlapping red and structure"
    dist, cheb, sid, step_r, step_c = best
    return {
        "sid": int(sid),
        "move": (int(step_r), int(step_c)),
        "dist": int(dist),
        "cheb": int(cheb),
    }


def _move_plan(
    coords: list[tuple[int, int]],
    move: tuple[int, int],
    struct_mask: np.ndarray,
) -> dict:
    h, w = struct_mask.shape
    step_r, step_c = map(int, move)
    cur = [(int(r), int(c)) for r, c in coords]
    steps = 0

    while True:
        nxt = [(int(r + step_r), int(c + step_c)) for r, c in cur]
        stop_reason = None
        for rr, cc in nxt:
            if not (0 <= rr < h and 0 <= cc < w):
                stop_reason = "bounds"
                break
            if bool(struct_mask[rr, cc]):
                stop_reason = "structure"
                break
        if stop_reason is not None:
            break
        cur = nxt
        steps += 1

    trail_mask = np.zeros((h, w), dtype=bool)
    final_mask = np.zeros((h, w), dtype=bool)
    for step in range(int(steps)):
        off_r = int(step * step_r)
        off_c = int(step * step_c)
        for r, c in coords:
            trail_mask[int(r + off_r), int(c + off_c)] = True
    for rr, cc in cur:
        final_mask[int(rr), int(cc)] = True

    return {
        "steps": int(steps),
        "move": (int(step_r), int(step_c)),
        "stop_reason": str(stop_reason),
        "final_coords": cur,
        "trail_mask": trail_mask,
        "final_mask": final_mask,
    }


def _scene_analysis(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    red_coords, red_bbox = _extract_reds(g)
    structures, struct_mask = _extract_structures(g, bg)

    plans: list[dict] = []
    for rid, coords in enumerate(red_coords):
        nearest = _nearest_structure_for_red(coords, structures)
        plan = _move_plan(coords, nearest["move"], struct_mask)
        plans.append(
            {
                "rid": int(rid),
                "bbox": tuple(map(int, red_bbox[rid])),
                "sid": int(nearest["sid"]),
                "move": (int(nearest["move"][0]), int(nearest["move"][1])),
                "dist": int(nearest["dist"]),
                "cheb": int(nearest["cheb"]),
                "steps": int(plan["steps"]),
                "stop_reason": str(plan["stop_reason"]),
                "trail_mask": plan["trail_mask"],
                "final_mask": plan["final_mask"],
                "final_coords": plan["final_coords"],
            }
        )

    return {
        "bg": int(bg),
        "red_coords": red_coords,
        "red_bbox": red_bbox,
        "structures": structures,
        "struct_mask": struct_mask,
        "plans": plans,
    }


def _render_output(grid: np.ndarray, analysis: dict) -> np.ndarray:
    out = np.array(grid, dtype=int)
    bg = int(analysis["bg"])
    red_val = int(Color.RED)
    black_val = int(Color.BLACK)
    out[out == red_val] = int(bg)

    for plan in analysis["plans"]:
        out[plan["trail_mask"]] = int(black_val)
        for rr, cc in plan["final_coords"]:
            out[int(rr), int(cc)] = int(red_val)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move every red rectangle toward its nearest large structure and stop just before collision, leaving a black path.

    Input structure:
    - The background is the dominant non-red, non-black color.
    - Large non-background structures are solid comb/bracket shapes with no usable cavity target.
    - Multiple small red rectangular components sit outside the structures, often diagonally offset from them.

    Rule:
    - For each red component independently, find the nearest structure by Manhattan distance to occupied structure pixels.
    - Convert that nearest relation into a row/column sign vector toward the structure, then translate the whole red
      component one step at a time along that vector until the next step would hit the structure or leave the grid.
    - Paint every intermediate placement black and redraw the final collision-stop placement in red. No red is deleted
      just for lacking row/column alignment or a dedicated socket.

    Key insight:
    - The pairing is now "red to nearest structure" rather than "structure to one visible aligned red", and the final
      destination is defined only by collision with structure pixels instead of by an interior socket.
    """
    g = np.array(grid, dtype=int)
    red_val = int(Color.RED)

    assert red_val in set(map(int, np.unique(g)))

    analysis = _scene_analysis(g)
    assert analysis["red_coords"], "expected at least one red component"
    assert analysis["structures"], "expected at least one non-bg non-red structure"

    return _render_output(g, analysis)


def _make_structure_spec(width: int, span: int, rot_k: int, bg: int, sc: int) -> dict:
    structure = np.rot90(_structure_array(width, span, bg, sc), k=rot_k)
    return {"structure": structure}


def _place_structures(h: int, w: int, bg: int, sc: int, n_structs: int) -> tuple[np.ndarray, list[dict]] | None:
    assert n_structs == 2
    struct_grid = np.full((h, w), bg, dtype=int)
    placed: list[dict] = []

    specs = []
    for _ in range(n_structs):
        width, span = random.choice(STRUCT_WIDTH_SPAN)
        rot_k = random.randrange(4)
        specs.append(_make_structure_spec(width, span, rot_k, bg, sc))

    layout = random.choice(("main_diag", "anti_diag"))
    locs: list[tuple[int, int]] = []
    zones: list[str] = []
    for idx, spec in enumerate(specs):
        sh, sw = spec["structure"].shape
        if layout == "main_diag":
            base_r = 7 if idx == 0 else h - sh - 7
            base_c = 5 if idx == 0 else w - sw - 5
            zone = "ul" if idx == 0 else "lr"
        else:
            base_r = 7 if idx == 0 else h - sh - 7
            base_c = w - sw - 5 if idx == 0 else 5
            zone = "ur" if idx == 0 else "ll"
        r0 = min(max(0, base_r), h - sh)
        c0 = min(max(0, base_c), w - sw)
        locs.append((int(r0), int(c0)))
        zones.append(zone)

    for i in range(len(specs)):
        sh1, sw1 = specs[i]["structure"].shape
        r1, c1 = locs[i]
        for j in range(i + 1, len(specs)):
            sh2, sw2 = specs[j]["structure"].shape
            r2, c2 = locs[j]
            if not (
                r1 + sh1 + STRUCT_PADDING <= r2
                or r2 + sh2 + STRUCT_PADDING <= r1
                or c1 + sw1 + STRUCT_PADDING <= c2
                or c2 + sw2 + STRUCT_PADDING <= c1
            ):
                return None

    for spec, (r0, c0), zone in zip(specs, locs, zones):
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)
        coords_arr = np.argwhere(spec["structure"] != bg)
        coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in coords_arr]
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        placed.append(
            {
                "sid": len(placed),
                "coords": coords,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "zone": zone,
            }
        )

    return struct_grid, placed


def _available_sectors_for_structure(
    bbox: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
) -> list[str]:
    h, w = map(int, grid_shape)
    r0, c0, r1, c1 = map(int, bbox)
    max_h = max(shape[0] for shape in RED_SHAPES)
    max_w = max(shape[1] for shape in RED_SHAPES)
    min_gap = int(GAP_RANGE[0])

    can_n = int(r0 - min_gap - max_h) >= 0
    can_s = int(r1 + min_gap + 1 + max_h) <= h
    can_w = int(c0 - min_gap - max_w) >= 0
    can_e = int(c1 + min_gap + 1 + max_w) <= w

    sectors: list[str] = []
    if can_n and can_w:
        sectors.append("nw")
    if can_n:
        sectors.append("n")
    if can_n and can_e:
        sectors.append("ne")
    if can_w:
        sectors.append("w")
    if can_e:
        sectors.append("e")
    if can_s and can_w:
        sectors.append("sw")
    if can_s:
        sectors.append("s")
    if can_s and can_e:
        sectors.append("se")
    return sectors


def _distribute_red_counts(capacities: list[int], total_reds: int) -> list[int]:
    counts = [1] * len(capacities)
    remaining = int(total_reds - len(capacities))
    while remaining > 0:
        choices = [idx for idx, cnt in enumerate(counts) if cnt < capacities[idx]]
        assert choices
        idx = random.choice(choices)
        counts[idx] += 1
        remaining -= 1
    return counts


def _balanced_red_counts(capacities: list[int], total_reds: int) -> list[int] | None:
    counts = [0] * len(capacities)
    for idx, cap in enumerate(capacities):
        if cap <= 0:
            return None
        counts[idx] = 1

    remaining = int(total_reds - len(capacities))
    for idx, cap in enumerate(capacities):
        if remaining <= 0:
            break
        if cap >= 2:
            counts[idx] += 1
            remaining -= 1

    while remaining > 0:
        choices = [idx for idx, cap in enumerate(capacities) if counts[idx] < cap]
        if not choices:
            return None
        min_count = min(counts[idx] for idx in choices)
        choices = [idx for idx in choices if counts[idx] == min_count]
        idx = random.choice(choices)
        counts[idx] += 1
        remaining -= 1

    return counts


def _sector_preference(zone: str) -> list[str]:
    if zone == "ul":
        return ["nw", "ne", "sw", "n", "w", "e", "s", "se"]
    if zone == "ur":
        return ["ne", "nw", "se", "n", "e", "w", "s", "sw"]
    if zone == "ll":
        return ["sw", "nw", "se", "s", "w", "e", "n", "ne"]
    if zone == "lr":
        return ["se", "ne", "sw", "s", "e", "w", "n", "nw"]
    raise ValueError(f"unknown zone: {zone}")


def _candidate_top_left(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    sector: str,
    gap_r: int,
    gap_c: int,
) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    h, w = map(int, shape)

    if sector == "nw":
        return int(r0 - gap_r - h), int(c0 - gap_c - w)
    if sector == "n":
        return int(r0 - gap_r - h), int(random.randint(c0 - w + 1, c1))
    if sector == "ne":
        return int(r0 - gap_r - h), int(c1 + gap_c + 1)
    if sector == "w":
        return int(random.randint(r0 - h + 1, r1)), int(c0 - gap_c - w)
    if sector == "e":
        return int(random.randint(r0 - h + 1, r1)), int(c1 + gap_c + 1)
    if sector == "sw":
        return int(r1 + gap_r + 1), int(c0 - gap_c - w)
    if sector == "s":
        return int(r1 + gap_r + 1), int(random.randint(c0 - w + 1, c1))
    if sector == "se":
        return int(r1 + gap_r + 1), int(c1 + gap_c + 1)
    raise ValueError(f"unknown sector: {sector}")


def _sector_move(sector: str) -> tuple[int, int]:
    if sector == "nw":
        return (1, 1)
    if sector == "n":
        return (1, 0)
    if sector == "ne":
        return (1, -1)
    if sector == "w":
        return (0, 1)
    if sector == "e":
        return (0, -1)
    if sector == "sw":
        return (-1, 1)
    if sector == "s":
        return (-1, 0)
    if sector == "se":
        return (-1, -1)
    raise ValueError(f"unknown sector: {sector}")


def _nearest_margin_and_moves(
    red_cells: list[tuple[int, int]],
    structure: dict,
) -> tuple[int, set[tuple[int, int]]]:
    best = None
    moves: set[tuple[int, int]] = set()
    for rr, cc in red_cells:
        for sr, sc in structure["coords"]:
            delta_r = int(sr - rr)
            delta_c = int(sc - cc)
            dist = abs(delta_r) + abs(delta_c)
            if best is None or int(dist) < int(best):
                best = int(dist)
                moves = {(_sign(delta_r), _sign(delta_c))}
            elif int(dist) == int(best):
                moves.add((_sign(delta_r), _sign(delta_c)))
    assert best is not None
    return int(best), moves


def _select_sectors_for_structure(zone: str, available: list[str], count: int) -> list[str]:
    ordered = [sector for sector in _sector_preference(zone) if sector in available]
    diagonal = [sector for sector in ordered if sector in DIAGONAL_SECTORS]
    axial = [sector for sector in ordered if sector in AXIAL_SECTORS]

    chosen: list[str] = []
    if diagonal:
        diag_target = min(len(diagonal), max(1, count - 1))
        chosen.extend(diagonal[:diag_target])
    if len(chosen) < count and axial:
        chosen.append(axial[0])

    for sector in ordered:
        if len(chosen) >= count:
            break
        if sector not in chosen:
            chosen.append(sector)

    return chosen[:count]


def _place_reds(
    struct_grid: np.ndarray,
    structures: list[dict],
    total_reds: int,
) -> np.ndarray | None:
    bg = int(get_bg(struct_grid))
    h, w = struct_grid.shape
    inp = struct_grid.copy()
    struct_mask = struct_grid != bg
    red_mask = np.zeros((h, w), dtype=bool)
    reserved_finals = np.zeros((h, w), dtype=bool)

    available = [_available_sectors_for_structure(item["bbox"], (h, w)) for item in structures]
    if any(not sectors for sectors in available):
        return None

    capacities = [len(sectors) for sectors in available]
    counts = _balanced_red_counts(capacities, total_reds)
    if counts is None:
        counts = _distribute_red_counts(capacities, total_reds)
    requests: list[tuple[int, str]] = []
    for sid, count in enumerate(counts):
        selected = _select_sectors_for_structure(
            str(structures[sid]["zone"]),
            available[sid],
            int(count),
        )
        if len(selected) != int(count):
            return None
        for sector in selected:
            requests.append((int(sid), sector))
    requests.sort(key=lambda item: (item[1] not in DIAGONAL_SECTORS, item[0]))

    for sid, sector in requests:
        item = structures[int(sid)]
        want_move = _sector_move(sector)
        success = False

        candidates = [(shape, gap_r, gap_c) for shape in RED_SHAPES for gap_r, gap_c in GAP_OPTIONS]
        random.shuffle(candidates)
        for shape, gap_r, gap_c in candidates:
            top, left = _candidate_top_left(item["bbox"], shape, sector, gap_r, gap_c)
            rh, rw = map(int, shape)

            if top < 0 or left < 0 or top + rh > h or left + rw > w:
                continue
            if bool((inp[top:top + rh, left:left + rw] != bg).any()):
                continue
            if bool(_dilate4(red_mask)[top:top + rh, left:left + rw].any()):
                continue

            coords = [(int(r), int(c)) for r in range(top, top + rh) for c in range(left, left + rw)]
            nearest = _nearest_structure_for_red(coords, structures)
            if int(nearest["sid"]) != int(sid):
                continue
            if tuple(map(int, nearest["move"])) != tuple(map(int, want_move)):
                continue

            plan = _move_plan(coords, want_move, struct_mask)
            if plan["stop_reason"] != "structure" or int(plan["steps"]) < GAP_RANGE[0]:
                continue
            if bool((plan["trail_mask"] & reserved_finals).any()):
                continue
            if bool((plan["final_mask"] & reserved_finals).any()):
                continue
            if bool((plan["final_mask"] & red_mask).any()):
                continue

            patch = np.full(shape, int(Color.RED), dtype=int)
            paste(inp, patch, top, left, transparent=bg)
            red_mask[top:top + rh, left:left + rw] = True
            reserved_finals |= plan["final_mask"]
            success = True
            break

        if not success:
            return None

    return inp


def _is_visible_combo_case(grid: np.ndarray) -> bool:
    analysis = _scene_analysis(grid)
    if len(analysis["red_coords"]) <= len(analysis["structures"]):
        return False
    diagonal = 0
    axial = 0
    per_structure: dict[int, set[str]] = {sid: set() for sid in range(len(analysis["structures"]))}
    for plan in analysis["plans"]:
        move_r, move_c = map(int, plan["move"])
        if move_r != 0 and move_c != 0:
            diagonal += 1
            per_structure[int(plan["sid"])].add("diag")
        if plan["stop_reason"] != "structure":
            return False
        if move_r == 0 or move_c == 0:
            axial += 1
            per_structure[int(plan["sid"])].add("axis")
        if plan["steps"] < GAP_RANGE[0]:
            return False
        dist_info = [
            _nearest_margin_and_moves(analysis["red_coords"][plan["rid"]], structure)
            for structure in analysis["structures"]
        ]
        ordered = sorted(dist_info, key=lambda item: item[0])
        if len(ordered) > 1 and int(ordered[0][0]) == int(ordered[1][0]):
            return False
        chosen_moves = dist_info[int(plan["sid"])][1]
        if len(chosen_moves) != 1 or tuple(chosen_moves)[0] != tuple(plan["move"]):
            return False
    if axial == 0 or diagonal <= axial:
        return False
    return all({"diag", "axis"} <= tags for tags in per_structure.values())


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(Color.RED), int(Color.BLACK))])
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_structs = random.randint(*STRUCT_COUNT_RANGE)

    struct_result = _place_structures(h, w, bg, sc, n_structs)
    if struct_result is None:
        return None
    struct_grid, structures = struct_result

    capacities = [
        len(_available_sectors_for_structure(item["bbox"], (h, w)))
        for item in structures
    ]
    if any(cap == 0 for cap in capacities):
        return None

    max_reds = min(TOTAL_RED_MAX, sum(capacities))
    min_reds = max(TOTAL_RED_MIN, len(structures) + 1)
    if min_reds > max_reds:
        return None
    total_reds = random.randint(min_reds, max_reds)

    inp = _place_reds(struct_grid, structures, total_reds)
    if inp is None:
        return None

    analysis = _scene_analysis(inp)
    if len(analysis["red_coords"]) != total_reds:
        return None
    if len(analysis["structures"]) != n_structs:
        return None
    if not _is_visible_combo_case(inp):
        return None
    if not np.array_equal(main(inp), _render_output(inp, analysis)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 combo 17")
