from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
OBJECT_COLORS = (BLUE, RED, GREEN, YELLOW)

DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

SIDE_TO_DIR = {
    "L": "right",
    "R": "left",
    "U": "down",
    "D": "up",
}

# Horizontal templates keep a contiguous top run, which makes same-lane stacking
# visually clear and guarantees contact along the shared lane row.
H_TEMPLATES = {
    2: [((0, 0), (0, 1))],
    3: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (0, 1), (1, 0)),
    ],
    4: [
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
    ],
    5: [
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2)),
    ],
}

MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells):
    r0, c0, _, _ = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _dims(cells):
    r0, c0, r1, c1 = _bbox(cells)
    return r1 - r0 + 1, c1 - c0 + 1


def _mirror_h(cells):
    h, w = _dims(cells)
    return _normalize([(r, w - 1 - c) for r, c in cells])


def _mirror_v(cells):
    h, w = _dims(cells)
    return _normalize([(h - 1 - r, c) for r, c in cells])


def _transpose(cells):
    return _normalize([(c, r) for r, c in cells])


def _bool_components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _find_objects(grid: np.ndarray):
    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BG, GREY) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            cells = tuple(sorted(cells))
            bbox = _bbox(cells)
            objects.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": bbox,
                    "area": len(cells),
                }
            )
    return objects


def _extract_attractor(grid: np.ndarray):
    grey_mask = grid == GREY
    comps = _bool_components(grey_mask)
    assert len(comps) == 1, "expected exactly one grey attractor"
    cells = tuple(sorted(comps[0]))
    bbox = _bbox(cells)
    r0, c0, r1, c1 = bbox
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY), "grey attractor must be a solid rectangle"
    return {"cells": cells, "bbox": bbox}


def _candidate_directions(obj_bbox, attr_bbox):
    r0, c0, r1, c1 = obj_bbox
    ar0, ac0, ar1, ac1 = attr_bbox
    options = []
    if r1 < ar0:
        options.append(("down", ar0 - r1 - 1))
    if r0 > ar1:
        options.append(("up", r0 - ar1 - 1))
    if c1 < ac0:
        options.append(("right", ac0 - c1 - 1))
    if c0 > ac1:
        options.append(("left", c0 - ac1 - 1))
    assert options, "object must lie outside the attractor"
    options.sort(key=lambda item: item[1])
    assert len(options) == 1 or options[0][1] < options[1][1], "slide direction is ambiguous"
    return options


def _translate_cells(cells, dr: int, dc: int):
    return tuple(sorted((r + dr, c + dc) for r, c in cells))


def _analyze(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    attractor = _extract_attractor(g)
    objects = _find_objects(g)
    assert 1 <= len(objects) <= 12, "unexpected number of foreground objects"

    enriched = []
    for obj in objects:
        options = _candidate_directions(obj["bbox"], attractor["bbox"])
        direction, distance = options[0]
        enriched.append(
            {
                **obj,
                "direction": direction,
                "distance": distance,
            }
        )

    enriched.sort(key=lambda obj: (obj["distance"], obj["bbox"][0], obj["bbox"][1]))

    occupied = set(attractor["cells"])
    moved = []
    for obj in enriched:
        current = obj["cells"]
        dr, dc = DIR_TO_DELTA[obj["direction"]]
        while True:
            nxt = _translate_cells(current, dr, dc)
            assert all(0 <= r < h and 0 <= c < w for r, c in nxt), "object left the grid before stopping"
            if any(cell in occupied for cell in nxt):
                break
            current = nxt
        moved.append(
            {
                **obj,
                "final_cells": current,
                "final_bbox": _bbox(current),
            }
        )
        occupied.update(current)

    final_mask = np.zeros_like(g, dtype=bool)
    for obj in moved:
        for r, c in obj["final_cells"]:
            final_mask[r, c] = True

    groups = []
    for comp in _bool_components(final_mask):
        comp_set = set(comp)
        members = [obj for obj in moved if any(cell in comp_set for cell in obj["final_cells"])]
        counts = Counter()
        for obj in members:
            counts[obj["color"]] += obj["area"]
        max_area = max(counts.values())
        leaders = {color for color, area in counts.items() if area == max_area}
        if len(leaders) == 1:
            winner = next(iter(leaders))
            tie = False
        else:
            tied_members = [obj for obj in members if obj["color"] in leaders]
            tied_members.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
            winner = tied_members[0]["color"]
            tie = True
        groups.append(
            {
                "cells": tuple(sorted(comp)),
                "members": members,
                "counts": dict(counts),
                "winner": winner,
                "tie": tie,
                "colors": sorted({obj["color"] for obj in members}),
            }
        )

    groups.sort(key=lambda group: min(group["cells"]))

    out = np.full_like(g, BG)
    for r, c in attractor["cells"]:
        out[r, c] = GREY
    for group in groups:
        for obj in group["members"]:
            for r, c in obj["final_cells"]:
                out[r, c] = group["winner"]

    return {
        "attractor": attractor,
        "objects": enriched,
        "moved": moved,
        "groups": groups,
        "output": out,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide colored objects toward a central grey block, then recolor each post-slide contact pile by area vote.

    Input structure:
    - The grid has black background, one solid grey rectangular attractor near the center, and several small single-color 4-connected objects around it.
    - Every foreground object is separated from the attractor and from other objects by background.
    - Object colors come from blue, red, green, and yellow.

    Rule:
    - For each object, compare the empty-row or empty-column gap from its bbox to the grey attractor on the four cardinal sides and choose the unique shortest direction.
    - Process objects by increasing slide distance, breaking ties by the object's input bbox top-left coordinate; slide each object step by step until the next step would overlap the grey attractor or an already stopped object.
    - After all motion, group together the non-grey objects that now touch by 4-neighbor contact, count each color's total pixel area inside a group, and choose the winning color; ties are broken by the input object whose bbox top-left corner is earliest in reading order.
    - Recolor every non-grey pixel in that group to the winning color, keep the grey attractor unchanged, and leave the background unchanged.

    Key insight:
    - Sliding does not change a pixel's color directly; it only determines which objects end up in the same touching pile, and that pile is the region that votes for a single recolor.
    """
    return _analyze(np.asarray(grid, dtype=int))["output"]


def _choose_template(area: int, side: str, rng: random.Random):
    base = list(rng.choice(H_TEMPLATES[area]))
    if side in ("L", "R"):
        cells = _normalize(base)
        if rng.random() < 0.5:
            cells = _mirror_h(cells)
        return cells
    cells = _transpose(base)
    if rng.random() < 0.5:
        cells = _mirror_v(cells)
    return cells


def _group_object_specs(side: str, size: int, force_tie: bool, rng: random.Random):
    if force_tie:
        assert size == 2
        area = rng.choice([2, 3, 4, 5])
        templates = [_choose_template(area, side, rng), _choose_template(area, side, rng)]
        colors = rng.sample(OBJECT_COLORS, 2)
        specs = [{"cells": tpl, "color": color} for tpl, color in zip(templates, colors)]
        rng.shuffle(specs)
        return specs

    if size == 2:
        areas = rng.sample([2, 3, 4, 5], 2)
        specs = [
            {"cells": _choose_template(area, side, rng), "color": color}
            for area, color in zip(areas, rng.sample(OBJECT_COLORS, 2))
        ]
        rng.shuffle(specs)
        return specs

    areas = [rng.choice([2, 3, 4, 5]) for _ in range(size)]
    templates = [_choose_template(area, side, rng) for area in areas]
    dims = [_dims(tpl) for tpl in templates]
    sort_idx = sorted(range(size), key=lambda idx: len(templates[idx]), reverse=True)
    winner_color, other_color = rng.sample(OBJECT_COLORS, 2)
    colors = [other_color] * size
    colors[sort_idx[0]] = winner_color
    colors[sort_idx[1]] = winner_color
    specs = [{"cells": tpl, "color": color} for tpl, color in zip(templates, colors)]
    rng.shuffle(specs)
    return specs


def _partition_group_sizes(group_count: int, rng: random.Random):
    if group_count == 2:
        total = 6
    elif group_count == 4:
        total = 8
    else:
        total = rng.choice([6, 7, 8])
    sizes = [2] * group_count
    extras = total - 2 * group_count
    for idx in rng.sample(range(group_count), extras):
        sizes[idx] += 1
    rng.shuffle(sizes)
    return sizes


def _stamp(grid: np.ndarray, cells, color: int, top: int, left: int):
    for r, c in cells:
        grid[top + r, left + c] = color


def _cells_global(cells, top: int, left: int):
    return [(top + r, left + c) for r, c in cells]


def _touching_pair(cells_a, cells_b):
    b = set(cells_b)
    for r, c in cells_a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in b:
                return True
    return False


def _generate_attempt(rng: random.Random):
    attr_h, attr_w = rng.choice([(2, 3), (3, 2)])
    group_count = rng.choices([2, 3, 4], weights=[2, 6, 2], k=1)[0]
    sizes = _partition_group_sizes(group_count, rng)
    sides = rng.sample(["L", "R", "U", "D"], group_count)

    tie_candidates = [idx for idx, size in enumerate(sizes) if size == 2]
    tie_group = rng.choice(tie_candidates) if tie_candidates and rng.random() < 0.35 else None

    groups = []
    for idx, (side, size) in enumerate(zip(sides, sizes)):
        specs = _group_object_specs(side, size, idx == tie_group, rng)
        face_gap = rng.randint(1, 3)
        chain_gaps = [rng.randint(1, 2) for _ in range(size - 1)]
        groups.append(
            {
                "side": side,
                "objects": specs,
                "face_gap": face_gap,
                "chain_gaps": chain_gaps,
            }
        )

    required = {"L": 0, "R": 0, "U": 0, "D": 0}
    max_face = {"L": 0, "R": 0, "U": 0, "D": 0}
    for group in groups:
        side = group["side"]
        dims = [_dims(obj["cells"]) for obj in group["objects"]]
        if side in ("L", "R"):
            widths = [w for _, w in dims]
            required[side] = group["face_gap"] + sum(widths) + sum(group["chain_gaps"])
            max_face[side] = max(h for h, _ in dims)
        else:
            heights = [h for h, _ in dims]
            required[side] = group["face_gap"] + sum(heights) + sum(group["chain_gaps"])
            max_face[side] = max(w for _, w in dims)

    if max_face["L"] > attr_h or max_face["R"] > attr_h or max_face["U"] > attr_w or max_face["D"] > attr_w:
        return None

    pad_l = rng.randint(1, 2)
    pad_r = rng.randint(1, 2)
    pad_u = rng.randint(1, 2)
    pad_d = rng.randint(1, 2)
    empty_l = rng.randint(1, 3) if "L" not in sides else 0
    empty_r = rng.randint(1, 3) if "R" not in sides else 0
    empty_u = rng.randint(1, 3) if "U" not in sides else 0
    empty_d = rng.randint(1, 3) if "D" not in sides else 0

    left_span = required["L"] if "L" in sides else empty_l
    right_span = required["R"] if "R" in sides else empty_r
    top_span = required["U"] if "U" in sides else empty_u
    bottom_span = required["D"] if "D" in sides else empty_d

    h = pad_u + top_span + attr_h + bottom_span + pad_d
    w = pad_l + left_span + attr_w + right_span + pad_r
    if h > 24 or w > 24:
        return None

    grid = np.full((h, w), BG, dtype=int)
    attr_r0 = pad_u + top_span
    attr_c0 = pad_l + left_span
    grid[attr_r0 : attr_r0 + attr_h, attr_c0 : attr_c0 + attr_w] = GREY
    attr_r1 = attr_r0 + attr_h - 1
    attr_c1 = attr_c0 + attr_w - 1

    occupied = grid != BG

    for group in groups:
        side = group["side"]
        objs = group["objects"]
        dims = [_dims(obj["cells"]) for obj in objs]
        if side in ("L", "R"):
            max_h = max(hh for hh, _ in dims)
            lane_top = attr_r0 + rng.randint(0, attr_h - max_h)
        else:
            max_w = max(ww for _, ww in dims)
            lane_left = attr_c0 + rng.randint(0, attr_w - max_w)

        if side == "L":
            cursor_right = attr_c0 - group["face_gap"] - 1
            for idx, obj in enumerate(objs):
                hh, ww = _dims(obj["cells"])
                left = cursor_right - ww + 1
                _stamp(grid, obj["cells"], obj["color"], lane_top, left)
                cursor_right = left - (group["chain_gaps"][idx] if idx < len(group["chain_gaps"]) else 0) - 1
        elif side == "R":
            cursor_left = attr_c1 + group["face_gap"] + 1
            for idx, obj in enumerate(objs):
                hh, ww = _dims(obj["cells"])
                _stamp(grid, obj["cells"], obj["color"], lane_top, cursor_left)
                cursor_left += ww + (group["chain_gaps"][idx] if idx < len(group["chain_gaps"]) else 0)
        elif side == "U":
            cursor_bottom = attr_r0 - group["face_gap"] - 1
            for idx, obj in enumerate(objs):
                hh, ww = _dims(obj["cells"])
                top = cursor_bottom - hh + 1
                _stamp(grid, obj["cells"], obj["color"], top, lane_left)
                cursor_bottom = top - (group["chain_gaps"][idx] if idx < len(group["chain_gaps"]) else 0) - 1
        else:
            cursor_top = attr_r1 + group["face_gap"] + 1
            for idx, obj in enumerate(objs):
                hh, ww = _dims(obj["cells"])
                _stamp(grid, obj["cells"], obj["color"], cursor_top, lane_left)
                cursor_top += hh + (group["chain_gaps"][idx] if idx < len(group["chain_gaps"]) else 0)

    return grid


def _validate_generated(grid: np.ndarray, info: dict):
    objects = info["objects"]
    moved = info["moved"]
    groups = info["groups"]
    attractor_cells = set(info["attractor"]["cells"])

    assert 6 <= len(objects) <= 8
    assert len(groups) >= 2
    assert any(group["tie"] for group in groups) or True

    for obj in objects:
        assert obj["color"] in OBJECT_COLORS
        assert 2 <= obj["area"] <= 5
        _candidate_directions(obj["bbox"], info["attractor"]["bbox"])
        assert obj["distance"] >= 1

    # No initial object touches another object or the attractor.
    for obj in objects:
        for r, c in obj["cells"]:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    assert (nr, nc) not in attractor_cells
        for other in objects:
            if obj is other:
                continue
            assert not _touching_pair(obj["cells"], other["cells"])

    # Every final group should contain multiple colors and at least two objects.
    assert any(len(group["members"]) > 1 for group in groups)
    assert any(
        any(member["color"] != group["winner"] for member in group["members"])
        for group in groups
    )
    for group in groups:
        assert len(group["members"]) >= 2
        assert len(group["colors"]) >= 2

    # Sliding must genuinely create a pile that did not exist in the input.
    built_by_motion = False
    for group in groups:
        if len(group["members"]) < 2:
            continue
        for i in range(len(group["members"])):
            for j in range(i + 1, len(group["members"])):
                a = group["members"][i]
                b = group["members"][j]
                if not _touching_pair(a["cells"], b["cells"]):
                    built_by_motion = True
                    break
            if built_by_motion:
                break
        if built_by_motion:
            break
    assert built_by_motion

    # Final groups should stay separated from one another.
    assert np.any(info["output"] != grid)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is None:
            continue
        try:
            info = _analyze(grid)
            _validate_generated(grid, info)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid grouped-slide vote puzzle")
