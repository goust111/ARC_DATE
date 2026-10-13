from __future__ import annotations

from collections import Counter
from itertools import product
from pathlib import Path
import random
import sys

import numpy as np


def _import_arc_utils():
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if candidate.exists():
            sys.path.insert(0, str(candidate.parent))
            from arc_utils import Color, bbox_coords, components_mask, in_bounds, neighbors4, paste

            return Color, bbox_coords, components_mask, in_bounds, neighbors4, paste
    raise ImportError("could not locate new/solvers/arc_utils.py")


Color, bbox_coords, components_mask, in_bounds, neighbors4, paste = _import_arc_utils()

BLACK = int(Color.BLACK)
GREY = int(Color.GREY)
PALETTE = [int(c) for c in (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DISTANCE_OPTIONS = (2, 4, 6)

PLANS = (
    {"name": "k3_tie2", "chambers": 3, "slots": (("bridge", 1), ("bridge", 2)), "kind": "tie"},
    {"name": "k3_major3", "chambers": 3, "slots": (("bridge", 1), ("single", 1), ("bridge", 2)), "kind": "majority"},
    {"name": "k4_sparse2", "chambers": 4, "slots": (("bridge", 1), ("bridge", 3)), "kind": "sparse"},
    {"name": "k4_tie3", "chambers": 4, "slots": (("bridge", 1), ("bridge", 2), ("bridge", 3)), "kind": "tie"},
    {"name": "k4_major3", "chambers": 4, "slots": (("bridge", 1), ("single", 2), ("bridge", 3)), "kind": "majority"},
    {"name": "k5_tie3a", "chambers": 5, "slots": (("bridge", 1), ("bridge", 2), ("bridge", 4)), "kind": "tie"},
    {"name": "k5_tie3b", "chambers": 5, "slots": (("bridge", 1), ("bridge", 3), ("bridge", 4)), "kind": "tie"},
)

MAX_ATTEMPTS = 400


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _component_info(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    comps = components_mask((grid != BLACK) & (grid != GREY), connectivity=4)
    info: list[dict] = []
    comp_at: dict[tuple[int, int], int] = {}
    for cid, coords in enumerate(comps):
        pts = [(int(r), int(c)) for r, c in coords]
        for rc in pts:
            comp_at[rc] = cid
        info.append(
            {
                "coords": pts,
                "bbox": tuple(map(int, bbox_coords(coords))),
                "color": int(grid[pts[0]]),
            }
        )
    return info, comp_at


def _discover_assignments(grid: np.ndarray) -> tuple[list[dict], list[dict], tuple[int, int, int, int]]:
    grey_comps = components_mask(grid == GREY, connectivity=4)
    assert grey_comps, "expected at least one grey wall component"
    wall = max(grey_comps, key=lambda coords: int(coords.shape[0]))
    wall_bbox = tuple(map(int, bbox_coords(wall)))

    comps, comp_at = _component_info(grid)
    best_for_comp: dict[int, dict] = {}
    r0, c0, r1, c1 = wall_bbox

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if int(grid[r, c]) != BLACK:
                continue
            if not any(in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GREY for nr, nc in neighbors4(r, c)):
                continue

            hits: list[tuple[int, int, tuple[int, int]]] = []
            for dr, dc in DIRS:
                rr, cc = int(r + dr), int(c + dc)
                dist = 1
                while in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == BLACK:
                    rr += dr
                    cc += dc
                    dist += 1
                if not in_bounds(grid.shape, rr, cc):
                    continue
                if int(grid[rr, cc]) in (BLACK, GREY):
                    continue
                cid = comp_at.get((int(rr), int(cc)))
                if cid is not None:
                    hits.append((int(dist), int(cid), (int(dr), int(dc))))

            if len(hits) != 1:
                continue

            dist, cid, direction = hits[0]
            prev = best_for_comp.get(cid)
            candidate = {"cid": cid, "slot": (int(r), int(c)), "direction": direction, "distance": dist}
            if prev is None or (candidate["distance"], candidate["slot"]) < (prev["distance"], prev["slot"]):
                best_for_comp[cid] = candidate

    assignments = [best_for_comp[cid] for cid in sorted(best_for_comp)]
    return comps, assignments, wall_bbox


def _dock_components(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    comps, assignments, _wall_bbox = _discover_assignments(g)
    out = g.copy()

    for comp in comps:
        for r, c in comp["coords"]:
            out[r, c] = BLACK

    for rec in assignments:
        comp = comps[int(rec["cid"])]
        dr, dc = rec["direction"]
        sr, sc = rec["slot"]
        r0, c0, r1, c1 = comp["bbox"]

        if dr == -1 and dc == 0:
            shift_r, shift_c = int(sr - r1), 0
        elif dr == 1 and dc == 0:
            shift_r, shift_c = int(sr - r0), 0
        elif dr == 0 and dc == -1:
            shift_r, shift_c = 0, int(sc - c1)
        else:
            shift_r, shift_c = 0, int(sc - c0)

        steps = int(max(abs(shift_r), abs(shift_c)))
        step_r = _sign(shift_r)
        step_c = _sign(shift_c)
        for step in range(steps):
            off_r = int(step * step_r)
            off_c = int(step * step_c)
            for r, c in comp["coords"]:
                rr = int(r + off_r)
                cc = int(c + off_c)
                if in_bounds(out.shape, rr, cc):
                    out[rr, cc] = BLACK

        for r, c in comp["coords"]:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if in_bounds(out.shape, rr, cc):
                out[rr, cc] = int(comp["color"])

    return out


def _interior_black_components(grid: np.ndarray) -> list[np.ndarray]:
    chambers = []
    for coords in components_mask(grid == BLACK, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        if any(r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1 for r, c in pts):
            continue
        chambers.append(coords)
    return chambers


def _choose_fill_color(grid: np.ndarray, chamber: np.ndarray) -> int:
    chamber_pts = [(int(r), int(c)) for r, c in chamber]
    top_left = min(chamber_pts)
    adjacent: dict[tuple[int, int], int] = {}
    for r, c in chamber_pts:
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) not in (BLACK, GREY):
                adjacent[(int(nr), int(nc))] = int(grid[nr, nc])
    assert adjacent, "every chamber must touch at least one docked marker"

    counts = Counter(adjacent.values())
    top = max(counts.values())
    winners = {color for color, count in counts.items() if count == top}
    if len(winners) == 1:
        return next(iter(winners))

    best = min(
        (
            abs(r - top_left[0]) + abs(c - top_left[1]),
            int(r),
            int(c),
            int(color),
        )
        for (r, c), color in adjacent.items()
        if int(color) in winners
    )
    return int(best[3])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock outside colored components into wall slots, then fill each enclosed chamber by the majority color of its docked boundary markers.

    Input structure:
    - Black background surrounds one largest grey wall structure that encloses several 4-connected black chambers.
    - The wall bbox also contains black candidate slot cells adjacent to grey, and a small set of external monochrome colored components.
    - Each true slot sees exactly one external component along one orthogonal ray.

    Rule:
    - Find the largest grey connected component as the wall.
    - For every black bbox cell adjacent to grey, ray-cast in the four orthogonal directions; a true slot is the nearest such cell for a uniquely visible colored component.
    - Slide each matched component back along that visible ray until its leading edge lands on the slot, turning its previous positions black.
    - After docking, find every enclosed 4-connected black chamber and recolor it with the majority color among distinct orthogonally adjacent docked colored pixels; if tied, use the tied color whose adjacent pixel is closest to the chamber's top-left cell by Manhattan distance.

    Key insight:
    - The chamber votes are not preset: docking changes where the colored pixels end up on the wall, and those final embedded marker positions determine the chamber fills.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    docked = _dock_components(g)
    out = docked.copy()
    for chamber in _interior_black_components(docked):
        fill = _choose_fill_color(docked, chamber)
        out[chamber[:, 0], chamber[:, 1]] = int(fill)
    return out


def _build_wall_patch(orientation: str, chamber_count: int) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    if orientation == "vertical":
        wall_h, wall_w = 6, 4 * chamber_count + 1
        wall = np.zeros((wall_h, wall_w), dtype=int)
        wall[0, :] = GREY
        wall[-1, :] = GREY
        wall[:, 0] = GREY
        wall[:, -1] = GREY
        chambers: list[list[tuple[int, int]]] = []
        for j in range(chamber_count):
            c0 = 1 + 4 * j
            chambers.append([(r, c) for r in range(1, 5) for c in range(c0, c0 + 3)])
        for divider in range(1, chamber_count):
            wall[:, 4 * divider] = GREY
        return wall, chambers

    wall_h, wall_w = 4 * chamber_count + 1, 6
    wall = np.zeros((wall_h, wall_w), dtype=int)
    wall[0, :] = GREY
    wall[-1, :] = GREY
    wall[:, 0] = GREY
    wall[:, -1] = GREY
    chambers = []
    for j in range(chamber_count):
        r0 = 1 + 4 * j
        chambers.append([(r, c) for r in range(r0, r0 + 3) for c in range(1, 5)])
    for divider in range(1, chamber_count):
        wall[4 * divider, :] = GREY
    return wall, chambers


def _slot_template(orientation: str, slot_def: tuple[str, int]) -> dict:
    kind, idx = slot_def
    if orientation == "vertical":
        if kind == "bridge":
            anchor_axis = 4 * idx
            span = tuple(range(anchor_axis - 1, anchor_axis + 2))
            touches = [idx - 1, idx]
        else:
            anchor_axis = 2 + 4 * idx
            span = tuple(range(anchor_axis - 1, anchor_axis + 2))
            touches = [idx]
    else:
        if kind == "bridge":
            anchor_axis = 4 * idx
            span = tuple(range(anchor_axis - 1, anchor_axis + 2))
            touches = [idx - 1, idx]
        else:
            anchor_axis = 2 + 4 * idx
            span = tuple(range(anchor_axis - 1, anchor_axis + 2))
            touches = [idx]

    return {
        "kind": kind,
        "index": int(idx),
        "anchor_axis": int(anchor_axis),
        "span": span,
        "touches": [int(x) for x in touches],
    }


def _choose_side_bits(slot_templates: list[dict]) -> list[int]:
    overlap_pairs = []
    for i in range(len(slot_templates)):
        ai0, ai1 = min(slot_templates[i]["span"]), max(slot_templates[i]["span"])
        for j in range(i + 1, len(slot_templates)):
            aj0, aj1 = min(slot_templates[j]["span"]), max(slot_templates[j]["span"])
            if max(ai0, aj0) <= min(ai1, aj1):
                overlap_pairs.append((i, j))

    valid = []
    for bits in product((0, 1), repeat=len(slot_templates)):
        if all(bits[i] != bits[j] for i, j in overlap_pairs):
            valid.append(bits)
    assert valid, "slot overlap graph must be two-colorable"
    return list(random.choice(valid))


def _assign_slot_geometry(orientation: str, wall_shape: tuple[int, int], slot_templates: list[dict]) -> list[dict]:
    side_names = ("top", "bottom") if orientation == "vertical" else ("left", "right")
    bits = _choose_side_bits(slot_templates)

    distances_by_bit: dict[int, list[int]] = {}
    for bit in (0, 1):
        count = sum(1 for value in bits if value == bit)
        if count:
            distances_by_bit[bit] = random.sample(list(DISTANCE_OPTIONS), k=count)

    assigned: list[dict] = []
    used_per_bit = {0: 0, 1: 0}
    wall_h, wall_w = wall_shape
    for bit, tmpl in zip(bits, slot_templates):
        side = side_names[bit]
        dist = distances_by_bit[bit][used_per_bit[bit]]
        used_per_bit[bit] += 1

        if orientation == "vertical":
            row = 0 if side == "top" else wall_h - 1
            anchor = (int(row), int(tmpl["anchor_axis"]))
            final_rel = [(int(row), int(c)) for c in tmpl["span"]]
            direction = (-1, 0) if side == "top" else (1, 0)
        else:
            col = 0 if side == "left" else wall_w - 1
            anchor = (int(tmpl["anchor_axis"]), int(col))
            final_rel = [(int(r), int(col)) for r in tmpl["span"]]
            direction = (0, -1) if side == "left" else (0, 1)

        rec = dict(tmpl)
        rec["side"] = side
        rec["distance"] = int(dist)
        rec["anchor_rel"] = anchor
        rec["final_rel"] = final_rel
        rec["direction"] = direction
        assigned.append(rec)
    return assigned


def _place_case(plan: dict, orientation: str) -> tuple[np.ndarray, dict]:
    chamber_count = int(plan["chambers"])
    wall_patch, chamber_rel = _build_wall_patch(orientation, chamber_count)
    slots = _assign_slot_geometry(orientation, wall_patch.shape, [_slot_template(orientation, slot_def) for slot_def in plan["slots"]])

    for slot in slots:
        ar, ac = slot["anchor_rel"]
        wall_patch[ar, ac] = BLACK

    max_top = max((slot["distance"] for slot in slots if slot["side"] in {"top", "left"}), default=0)
    max_bottom = max((slot["distance"] for slot in slots if slot["side"] in {"bottom", "right"}), default=0)
    wall_h, wall_w = wall_patch.shape

    if orientation == "vertical":
        wall_r = random.randint(max_top + 1, max_top + 3)
        wall_c = random.randint(1, 3)
        H = wall_r + wall_h + max_bottom + random.randint(1, 3)
        W = wall_c + wall_w + random.randint(1, 3)
    else:
        wall_r = random.randint(1, 3)
        wall_c = random.randint(max_top + 1, max_top + 3)
        H = wall_r + wall_h + random.randint(1, 3)
        W = wall_c + wall_w + max_bottom + random.randint(1, 3)

    assert max(H, W) <= 30

    grid = np.zeros((H, W), dtype=int)
    paste(grid, wall_patch, wall_r, wall_c)

    colors = random.sample(PALETTE, k=len(slots))
    chamber_global = [[(wall_r + r, wall_c + c) for r, c in cells] for cells in chamber_rel]

    for color, slot in zip(colors, slots):
        slot["color"] = int(color)
        final_coords = [(wall_r + r, wall_c + c) for r, c in slot["final_rel"]]
        anchor = (wall_r + slot["anchor_rel"][0], wall_c + slot["anchor_rel"][1])
        slot["anchor"] = anchor
        slot["final_coords"] = final_coords
        slot["slot_cell"] = anchor

        if orientation == "vertical":
            comp_row = anchor[0] - slot["distance"] if slot["side"] == "top" else anchor[0] + slot["distance"]
            component_coords = [(comp_row, c) for _r, c in final_coords]
        else:
            comp_col = anchor[1] - slot["distance"] if slot["side"] == "left" else anchor[1] + slot["distance"]
            component_coords = [(r, comp_col) for r, _c in final_coords]
        slot["component_coords"] = component_coords
        for r, c in component_coords:
            grid[r, c] = int(color)

    meta = {
        "orientation": orientation,
        "plan": plan["name"],
        "plan_kind": plan["kind"],
        "chamber_count": chamber_count,
        "chambers": chamber_global,
        "slots": slots,
    }
    return grid, meta


def _vote_info(final_pixels: dict[tuple[int, int], int], chamber: list[tuple[int, int]]) -> dict:
    top_left = min(chamber)
    adjacent = {}
    for r, c in chamber:
        for nr, nc in neighbors4(r, c):
            if (int(nr), int(nc)) in final_pixels:
                adjacent[(int(nr), int(nc))] = int(final_pixels[(int(nr), int(nc))])
    assert adjacent
    counts = Counter(adjacent.values())
    top = max(counts.values())
    tied = [color for color, count in counts.items() if count == top]
    if len(tied) == 1:
        winner = tied[0]
        tie_flag = False
    else:
        best = min(
            (
                abs(r - top_left[0]) + abs(c - top_left[1]),
                int(r),
                int(c),
                int(color),
            )
            for (r, c), color in adjacent.items()
            if int(color) in tied
        )
        winner = int(best[3])
        tie_flag = True
    return {
        "winner": int(winner),
        "counts": {int(color): int(count) for color, count in counts.items()},
        "tied": tie_flag,
        "multi_color": len(counts) > 1,
    }


def _expected_output_from_meta(grid: np.ndarray, meta: dict) -> tuple[np.ndarray, list[dict]]:
    out = np.array(grid, dtype=int, copy=True)
    final_pixels: dict[tuple[int, int], int] = {}
    for slot in meta["slots"]:
        for r, c in slot["component_coords"]:
            out[r, c] = BLACK
        for r, c in slot["final_coords"]:
            out[r, c] = int(slot["color"])
            final_pixels[(int(r), int(c))] = int(slot["color"])

    vote_infos = []
    for chamber in meta["chambers"]:
        info = _vote_info(final_pixels, chamber)
        vote_infos.append(info)
        for r, c in chamber:
            out[r, c] = int(info["winner"])
    return out, vote_infos


def _validate_case(grid: np.ndarray, meta: dict) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    comps = components_mask((grid != BLACK) & (grid != GREY), connectivity=4)
    if len(comps) != len(meta["slots"]):
        return False

    internal = _interior_black_components(grid)
    if len(internal) != int(meta["chamber_count"]):
        return False

    expected, votes = _expected_output_from_meta(grid, meta)
    if len({vote["winner"] for vote in votes}) < 2:
        return False
    if meta["plan_kind"] == "tie" and not any(vote["tied"] for vote in votes):
        return False
    if meta["plan_kind"] == "majority":
        if not any(vote["multi_color"] and not vote["tied"] for vote in votes):
            return False

    docked = _dock_components(grid)
    if len(_interior_black_components(docked)) != int(meta["chamber_count"]):
        return False

    discovered_comps, assignments, _bbox = _discover_assignments(grid)
    if len(discovered_comps) != len(meta["slots"]) or len(assignments) != len(meta["slots"]):
        return False
    if {tuple(item["slot"]) for item in assignments} != {tuple(slot["slot_cell"]) for slot in meta["slots"]}:
        return False

    return np.array_equal(main(grid), expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        plan = random.choice(PLANS)
        orientation = random.choice(("vertical", "horizontal"))
        grid, meta = _place_case(plan, orientation)
        if _validate_case(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid input for ARC docking-majority puzzle")
