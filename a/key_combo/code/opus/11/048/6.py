from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
ORANGE = 7
TEAL = 8

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_TO_SIDE = {(-1, 0): "top", (1, 0): "bottom", (0, -1): "left", (0, 1): "right"}
SIDE_TO_DIR = {v: k for k, v in DIR_TO_SIDE.items()}
DIR_TO_ROT = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _rect_coords(r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]


def _shape_dims_for_area(socket_shape: tuple[int, int], open_dir: tuple[int, int], area: int) -> tuple[int, int]:
    sh, sw = socket_shape
    if open_dir in [(-1, 0), (1, 0)]:
        if sw == 2:
            if area == 2:
                return 1, 2
            if area == 4:
                return 2, 2
        if sw == 3 and area == 3:
            return 1, 3
    else:
        if sh == 2:
            if area == 2:
                return 2, 1
            if area == 4:
                return 2, 2
        if sh == 3 and area == 3:
            return 3, 1
    raise ValueError(f"unsupported socket/open/area combination: {socket_shape}, {open_dir}, {area}")


def _rotate_template(base: np.ndarray, sockets: list[tuple[int, int, int, int]], k: int):
    patch = np.rot90(base, k=k)
    labels = np.zeros_like(base, dtype=int)
    for i, (r0, c0, r1, c1) in enumerate(sockets, start=1):
        labels[r0 : r1 + 1, c0 : c1 + 1] = i
    labels = np.rot90(labels, k=k)
    rotated_sockets = []
    for i in range(1, len(sockets) + 1):
        coords = np.argwhere(labels == i)
        rotated_sockets.append(_bbox(coords))
    return patch, rotated_sockets


TEMPLATES = {
    "single22": {
        "patch": np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
            ],
            dtype=int,
        ),
        "sockets": [(2, 1, 3, 2)],
    },
    "single23": {
        "patch": np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
            ],
            dtype=int,
        ),
        "sockets": [(2, 1, 3, 3)],
    },
    "double22": {
        "patch": np.array(
            [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 1],
                [1, 0, 0, 1, 0, 0, 1],
            ],
            dtype=int,
        ),
        "sockets": [(2, 1, 3, 2), (2, 4, 3, 5)],
    },
}


def _parse_structures_and_sockets(grid: np.ndarray):
    structures = []
    for coords in _components_mask(grid == BLUE):
        r0, c0, r1, c1 = _bbox(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        black_comps = _components_mask(sub == BLACK)
        sockets = []
        for comp in black_comps:
            br0, bc0, br1, bc1 = _bbox(comp)
            area = int(comp.shape[0])
            if area != (br1 - br0 + 1) * (bc1 - bc0 + 1):
                continue
            touched = []
            if br0 == 0:
                touched.append("top")
            if br1 == sub.shape[0] - 1:
                touched.append("bottom")
            if bc0 == 0:
                touched.append("left")
            if bc1 == sub.shape[1] - 1:
                touched.append("right")
            if len(touched) != 1:
                continue
            h = br1 - br0 + 1
            w = bc1 - bc0 + 1
            if h < 2 or w < 2 or h > 3 or w > 3:
                continue
            sockets.append(
                {
                    "bbox": (r0 + br0, c0 + bc0, r0 + br1, c0 + bc1),
                    "open_dir": SIDE_TO_DIR[touched[0]],
                }
            )
        structures.append({"bbox": (r0, c0, r1, c1), "coords": coords, "sockets": sockets})
    return structures


def _red_components(grid: np.ndarray):
    comps = _components_mask(grid == RED)
    by_cell: dict[tuple[int, int], int] = {}
    result = []
    for rid, coords in enumerate(comps):
        bbox = _bbox(coords)
        cells = {(int(r), int(c)) for r, c in coords}
        for cell in cells:
            by_cell[cell] = rid
        result.append({"id": rid, "coords": coords, "bbox": bbox, "cells": cells})
    return result, by_cell


def _socket_side_cells(sock_bbox: tuple[int, int, int, int], open_dir: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = sock_bbox
    if open_dir == (-1, 0):
        return [(r0, c) for c in range(c0, c1 + 1)]
    if open_dir == (1, 0):
        return [(r1, c) for c in range(c0, c1 + 1)]
    if open_dir == (0, -1):
        return [(r, c0) for r in range(r0, r1 + 1)]
    return [(r, c1) for r in range(r0, r1 + 1)]


def _candidate_for_direction(
    grid: np.ndarray,
    by_cell: dict[tuple[int, int], int],
    sock_bbox: tuple[int, int, int, int],
    open_dir: tuple[int, int],
):
    hits = []
    for r, c in _socket_side_cells(sock_bbox, open_dir):
        rr = r + open_dir[0]
        cc = c + open_dir[1]
        dist = 0
        while _in_bounds(grid.shape, rr, cc) and grid[rr, cc] == BLACK:
            rr += open_dir[0]
            cc += open_dir[1]
            dist += 1
        if _in_bounds(grid.shape, rr, cc) and grid[rr, cc] == RED:
            hits.append((by_cell[(rr, cc)], dist + 1, r, c))
    if not hits:
        return None
    rid_set = {rid for rid, _dist, _r, _c in hits}
    if len(rid_set) != 1:
        return None
    rid = hits[0][0]
    min_dist = min(dist for _rid, dist, _r, _c in hits)
    if open_dir in [(-1, 0), (1, 0)]:
        projection = sorted({c for _rid, _dist, _r, c in hits})
    else:
        projection = sorted({r for _rid, _dist, r, _c in hits})
    return {"rid": rid, "dist": min_dist, "dir": open_dir, "projection": projection}


def _dock_shift(
    sock_bbox: tuple[int, int, int, int],
    comp_bbox: tuple[int, int, int, int],
    open_dir: tuple[int, int],
    projection: list[int],
) -> tuple[int, int]:
    sr0, sc0, sr1, sc1 = sock_bbox
    cr0, cc0, cr1, cc1 = comp_bbox
    if open_dir == (-1, 0):
        return sr0 - cr0, min(projection) - cc0
    if open_dir == (1, 0):
        return sr1 - cr1, min(projection) - cc0
    if open_dir == (0, -1):
        return min(projection) - cr0, sc0 - cc0
    return min(projection) - cr0, sc1 - cc1


def _simulate_docking(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    red_comps, by_cell = _red_components(g)
    structures = _parse_structures_and_sockets(g)
    sockets = []
    candidates = []
    for structure in structures:
        for sock in structure["sockets"]:
            sock_bbox = sock["bbox"]
            sock_candidates = []
            for direction in DIRS:
                cand = _candidate_for_direction(g, by_cell, sock_bbox, direction)
                if cand is not None:
                    sock_candidates.append(cand)
            if not sock_candidates:
                continue
            sock_candidates.sort(key=lambda x: (x["dist"], -len(x["projection"])))
            if len(sock_candidates) > 1 and sock_candidates[0]["dist"] == sock_candidates[1]["dist"]:
                raise AssertionError("ambiguous nearest direction for a socket")
            chosen = sock_candidates[0]
            sockets.append({"bbox": sock_bbox, "open_dir": chosen["dir"]})
            candidates.append(chosen)

    chosen_ids = [c["rid"] for c in candidates]
    if len(chosen_ids) != len(set(chosen_ids)):
        raise AssertionError("a red component is claimed by multiple sockets")

    out = g.copy()
    out[out == RED] = BLACK
    assignments = []
    for chosen in candidates:
        comp = red_comps[chosen["rid"]]
        shift = _dock_shift(chosen.get("sock_bbox", None) or sockets[len(assignments)]["bbox"], comp["bbox"], chosen["dir"], chosen["projection"])
        for r, c in comp["coords"]:
            out[int(r + shift[0]), int(c + shift[1])] = RED
        assignments.append(
            {
                "rid": chosen["rid"],
                "socket_bbox": sockets[len(assignments)]["bbox"],
                "dir": chosen["dir"],
                "shift": shift,
                "area": int(comp["coords"].shape[0]),
            }
        )
    return out, assignments, structures, red_comps


def _region_components(grid: np.ndarray) -> list[np.ndarray]:
    return _components_mask(grid != GREY)


def _region_red_counts(grid: np.ndarray) -> list[int]:
    counts = []
    for comp in _region_components(grid):
        counts.append(int(np.count_nonzero(grid[comp[:, 0], comp[:, 1]] == RED)))
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock the visible red components into blue interior sockets, then recolor the regions with minimum and maximum docked red area.

    Input structure:
    - A black background contains grey one-cell walls that partition the board into 3 or 4 rectangular regions.
    - Each region contains one or more large blue structures whose bounding boxes contain one or two rectangular black sockets.
    - Small red 4-connected components sit outside the structures on straight unobstructed rays from the sockets.

    Rule:
    - For every blue structure, detect each rectangular socket and cast rays from it in the four cardinal directions.
    - In each direction, the first red component seen on that ray is the candidate; pick the candidate from the uniquely nearest direction and translate that red component into the socket, flush with the socket entrance side. Clear all original red cells so only docked components remain.
    - After all docking, treat grey walls as region boundaries, count the remaining red cells in each region, recolor the unique minimum-red region's red cells orange and the unique maximum-red region's red cells teal, and leave blue walls/background unchanged.

    Key insight:
    - The extreme-color decision depends on which red components actually dock into sockets; counting the input reds directly is insufficient because non-selected candidates disappear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    docked, _assignments, _structures, _red_comps = _simulate_docking(g)
    regions = _region_components(docked)
    counts = [int(np.count_nonzero(docked[comp[:, 0], comp[:, 1]] == RED)) for comp in regions]
    assert len(regions) in (3, 4)
    min_count = min(counts)
    max_count = max(counts)
    min_regions = [i for i, v in enumerate(counts) if v == min_count]
    max_regions = [i for i, v in enumerate(counts) if v == max_count]
    assert len(min_regions) == 1 and len(max_regions) == 1 and min_regions[0] != max_regions[0]

    out = docked.copy()
    for idx, color in [(min_regions[0], ORANGE), (max_regions[0], TEAL)]:
        comp = regions[idx]
        mask = out[comp[:, 0], comp[:, 1]] == RED
        target = comp[mask]
        out[target[:, 0], target[:, 1]] = color
    return out


def _layout_regions(rng: random.Random):
    layout = rng.choice(["4", "4", "4", "4", "3v", "3h", "3h", "3h", "3h", "3h"])
    if layout == "4":
        h = rng.randint(19, 22)
        w = rng.randint(19, 22)
        wr = rng.randint(9, h - 10)
        wc = rng.randint(9, w - 10)
        walls = [("h", wr), ("v", wc)]
        regions = [
            (0, 0, wr - 1, wc - 1),
            (0, wc + 1, wr - 1, w - 1),
            (wr + 1, 0, h - 1, wc - 1),
            (wr + 1, wc + 1, h - 1, w - 1),
        ]
    elif layout == "3v":
        h = rng.randint(17, 22)
        widths = [9, 9, rng.randint(9, 10)]
        rng.shuffle(widths)
        w = sum(widths) + 2
        c1 = widths[0]
        c2 = widths[0] + 1 + widths[1]
        walls = [("v", c1), ("v", c2)]
        regions = [
            (0, 0, h - 1, c1 - 1),
            (0, c1 + 1, h - 1, c2 - 1),
            (0, c2 + 1, h - 1, w - 1),
        ]
    else:
        w = rng.randint(17, 22)
        heights = [9, 9, rng.randint(9, 10)]
        rng.shuffle(heights)
        h = sum(heights) + 2
        r1 = heights[0]
        r2 = heights[0] + 1 + heights[1]
        walls = [("h", r1), ("h", r2)]
        regions = [
            (0, 0, r1 - 1, w - 1),
            (r1 + 1, 0, r2 - 1, w - 1),
            (r2 + 1, 0, h - 1, w - 1),
        ]
    return h, w, walls, regions


def _make_plan_library():
    return {
        2: [[{"template": "single22", "areas": [2], "decoys": [4]}]],
        3: [[{"template": "single23", "areas": [3], "decoys": [None]}]],
        4: [
            [{"template": "single22", "areas": [4], "decoys": [None]}],
            [{"template": "double22", "areas": [2, 2], "decoys": [None, None]}],
        ],
        5: [[
            {"template": "single22", "areas": [2], "decoys": [None]},
            {"template": "single23", "areas": [3], "decoys": [None]},
        ]],
        6: [[{"template": "double22", "areas": [2, 4], "decoys": [None, None]}]],
        7: [[
            {"template": "single22", "areas": [4], "decoys": [None]},
            {"template": "single23", "areas": [3], "decoys": [None]},
        ]],
        8: [[{"template": "double22", "areas": [4, 4], "decoys": [None, None]}]],
    }


def _choose_region_totals(n_regions: int, rng: random.Random) -> list[int]:
    if n_regions == 4:
        others = rng.sample([3, 4, 6, 8], 3)
        totals = [2] + others
    else:
        if rng.random() < 0.6:
            totals = [2, rng.choice([5, 7]), rng.choice([6, 8])]
        else:
            totals = [2, 3, rng.choice([6, 8])]
        if len(set(totals)) < 3:
            return _choose_region_totals(n_regions, rng)
    rng.shuffle(totals)
    return totals


def _build_instance(template_name: str, open_dir: tuple[int, int], areas: list[int], decoys: list[int | None], rng: random.Random):
    base = TEMPLATES[template_name]["patch"]
    sockets = TEMPLATES[template_name]["sockets"]
    patch, rot_sockets = _rotate_template(base, sockets, DIR_TO_ROT[open_dir])

    blue = {(r, c) for r, c in np.argwhere(patch == 1)}
    selected = []
    extra = []
    for sock_bbox, area, decoy_area in zip(rot_sockets, areas, decoys):
        sr0, sc0, sr1, sc1 = sock_bbox
        sh = sr1 - sr0 + 1
        sw = sc1 - sc0 + 1
        ph, pw = _shape_dims_for_area((sh, sw), open_dir, area)
        gap1 = rng.randint(1, 2)
        if open_dir == (1, 0):
            top = sr1 + 1 + gap1
            left = sc0
        elif open_dir == (-1, 0):
            top = sr0 - gap1 - ph
            left = sc0
        elif open_dir == (0, 1):
            top = sr0
            left = sc1 + 1 + gap1
        else:
            top = sr0
            left = sc0 - gap1 - pw
        selected_cells = _rect_coords(top, left, ph, pw)
        selected.append({"cells": selected_cells, "area": area})

        if decoy_area is not None:
            dh, dw = _shape_dims_for_area((sh, sw), open_dir, decoy_area)
            gap2 = rng.randint(1, 2)
            if open_dir == (1, 0):
                top2 = max(r for r, _c in selected_cells) + 1 + gap2
                left2 = sc0
            elif open_dir == (-1, 0):
                top2 = min(r for r, _c in selected_cells) - gap2 - dh
                left2 = sc0
            elif open_dir == (0, 1):
                top2 = sr0
                left2 = max(c for _r, c in selected_cells) + 1 + gap2
            else:
                top2 = sr0
                left2 = min(c for _r, c in selected_cells) - gap2 - dw
            extra_cells = _rect_coords(top2, left2, dh, dw)
            extra.append({"cells": extra_cells, "area": decoy_area})

    all_rel = set(blue)
    for item in selected + extra:
        all_rel.update(item["cells"])
    coords = np.array(sorted(all_rel), dtype=int)
    bbox = _bbox(coords)
    return {
        "blue": blue,
        "selected": selected,
        "extra": extra,
        "sockets": rot_sockets,
        "bbox": bbox,
        "open_dir": open_dir,
    }


def _shift_cells(cells: list[tuple[int, int]] | set[tuple[int, int]], dr: int, dc: int):
    return [(r + dr, c + dc) for r, c in cells]


def _fits_empty(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    for r, c in cells:
        if not _in_bounds(grid.shape, r, c) or grid[r, c] != BLACK:
            return False
    return True


def _place_structure_in_slot(
    grid: np.ndarray,
    slot: tuple[int, int, int, int],
    spec: dict,
    rng: random.Random,
    preferred_dir: tuple[int, int] | None = None,
):
    directions = [preferred_dir] if preferred_dir is not None else DIRS[:]
    if preferred_dir is None:
        rng.shuffle(directions)
    for open_dir in directions:
        instance = _build_instance(spec["template"], open_dir, spec["areas"], spec["decoys"], rng)
        br0, bc0, br1, bc1 = instance["bbox"]
        r0, c0, r1, c1 = slot
        ar_min = r0 - br0
        ac_min = c0 - bc0
        ar_max = r1 - br1
        ac_max = c1 - bc1
        if ar_min > ar_max or ac_min > ac_max:
            continue
        anchors = [(ar, ac) for ar in range(ar_min, ar_max + 1) for ac in range(ac_min, ac_max + 1)]
        rng.shuffle(anchors)
        for ar, ac in anchors:
            blue_cells = _shift_cells(instance["blue"], ar, ac)
            red_cells = []
            for item in instance["selected"] + instance["extra"]:
                red_cells.extend(_shift_cells(item["cells"], ar, ac))
            if not _fits_empty(grid, blue_cells) or not _fits_empty(grid, red_cells):
                continue
            for rr, cc in blue_cells:
                grid[rr, cc] = BLUE
            for rr, cc in red_cells:
                grid[rr, cc] = RED
            return True
    return False


def _region_slots(region: tuple[int, int, int, int], n_structs: int):
    r0, c0, r1, c1 = region
    if n_structs == 1:
        return [region], [None]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h >= w:
        mid = (r0 + r1) // 2
        return [(r0, c0, mid - 1, c1), (mid + 1, c0, r1, c1)], [(-1, 0), (1, 0)]
    mid = (c0 + c1) // 2
    return [(r0, c0, r1, mid - 1), (r0, mid + 1, r1, c1)], [(0, -1), (0, 1)]


def _render_attempt(rng: random.Random):
    h, w, walls, regions = _layout_regions(rng)
    grid = np.zeros((h, w), dtype=int)
    for kind, pos in walls:
        if kind == "h":
            grid[pos, :] = GREY
        else:
            grid[:, pos] = GREY

    plan_library = _make_plan_library()
    totals = _choose_region_totals(len(regions), rng)
    for region, total in zip(regions, totals):
        specs = rng.choice(plan_library[total])
        slots, preferred_dirs = _region_slots(region, len(specs))
        order = list(range(len(specs)))
        if len(order) == 2 and rng.random() < 0.5:
            order.reverse()
        for idx, spec in zip(order, specs):
            slot = slots[idx]
            direction = preferred_dirs[idx]
            if not _place_structure_in_slot(grid, slot, spec, rng, preferred_dir=direction):
                return None
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) - {BLACK, BLUE, RED, GREY}:
        return False

    try:
        docked, assignments, structures, red_comps = _simulate_docking(grid)
    except Exception:
        return False

    if len(structures) < 3:
        return False
    socket_count = sum(len(s["sockets"]) for s in structures)
    if socket_count < 3:
        return False
    if len(assignments) != socket_count:
        return False
    if len(red_comps) <= len(assignments):
        return False

    regions = _region_components(docked)
    if len(regions) not in (3, 4):
        return False
    out_counts = [int(np.count_nonzero(docked[comp[:, 0], comp[:, 1]] == RED)) for comp in regions]
    if len(set(out_counts)) != len(out_counts):
        return False

    in_regions = _region_components(grid)
    if len(in_regions) != len(regions):
        return False
    in_counts = [int(np.count_nonzero(grid[comp[:, 0], comp[:, 1]] == RED)) for comp in in_regions]

    out_min = {i for i, v in enumerate(out_counts) if v == min(out_counts)}
    out_max = {i for i, v in enumerate(out_counts) if v == max(out_counts)}
    in_min = {i for i, v in enumerate(in_counts) if v == min(in_counts)}
    in_max = {i for i, v in enumerate(in_counts) if v == max(in_counts)}
    if out_min == in_min and out_max == in_max:
        return False

    solved = main(grid)
    if np.count_nonzero(solved == ORANGE) != min(out_counts):
        return False
    if np.count_nonzero(solved == TEAL) != max(out_counts):
        return False
    if not np.any(grid == RED):
        return False
    if not np.any(grid == GREY):
        return False
    return True


def generate_input() -> np.ndarray:
    for seed in range(400):
        rng = random.Random(random.randrange(10**9) + seed)
        grid = _render_attempt(rng)
        if grid is not None and _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
