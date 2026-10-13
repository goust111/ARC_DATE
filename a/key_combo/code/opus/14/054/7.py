import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

SIDE_ORDER = ("top", "right", "bottom", "left")
SIDE_COLORS = {
    "top": RED,
    "right": BLUE,
    "bottom": GREEN,
    "left": YELLOW,
}

MAX_RETRIES = 500
GRID_SIDE_RANGE = (17, 23)
OPTIONAL_TIE_PROB = 0.45

PAIR_PATTERNS = [
    {
        "h": 5,
        "w": 4,
        "outer": [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)],
        "inner": [(1, 2), (1, 3), (2, 3)],
    },
    {
        "h": 4,
        "w": 5,
        "outer": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4)],
        "inner": [(2, 1), (3, 1), (3, 2)],
    },
]

SINGLE_PATTERNS = [
    {"h": 2, "w": 2, "cells": [(0, 0), (1, 0), (1, 1)]},
    {"h": 2, "w": 3, "cells": [(0, 0), (0, 1), (1, 1), (1, 2)]},
    {"h": 2, "w": 3, "cells": [(0, 1), (1, 0), (1, 1), (1, 2)]},
    {"h": 3, "w": 3, "cells": [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]},
    {"h": 2, "w": 3, "cells": [(0, 0), (0, 1), (0, 2), (1, 0)]},
    {"h": 3, "w": 2, "cells": [(0, 1), (1, 0), (1, 1), (2, 1)]},
]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _uniform_side(values: np.ndarray) -> int:
    colors = np.unique(values)
    assert len(colors) == 1
    return int(colors[0])


def _nearest_side(coords: np.ndarray, h: int, w: int) -> tuple[str, dict[str, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    distances = {
        "top": int(r0),
        "right": int(w - 1 - c1),
        "bottom": int(h - 1 - r1),
        "left": int(c0),
    }
    best = min(distances.values())
    for side in SIDE_ORDER:
        if distances[side] == best:
            return side, distances
    raise AssertionError("unreachable")


def _transform_points(
    coords: list[tuple[int, int]], base_h: int, base_w: int, rot: int, flip_h: bool
) -> tuple[list[tuple[int, int]], int, int]:
    arr = np.array(coords, dtype=int)
    h, w = int(base_h), int(base_w)
    if flip_h:
        arr[:, 1] = w - 1 - arr[:, 1]
    for _ in range(rot % 4):
        arr = np.stack([arr[:, 1], h - 1 - arr[:, 0]], axis=1)
        h, w = w, h
    return [tuple(map(int, rc)) for rc in arr], int(h), int(w)


def _all_transforms():
    for flip_h in (False, True):
        for rot in range(4):
            yield rot, flip_h


def _shift_coords(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in coords]


def _cells_clear(occupied: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return False
        if occupied[r, c]:
            return False
        if occupied[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)].any():
            return False
    return True


def _bbox_relation_with_any(bbox: tuple[int, int, int, int], existing: list[tuple[int, int, int, int]]) -> bool:
    for other in existing:
        if _bbox_contains(bbox, other) or _bbox_contains(other, bbox):
            return True
    return False


def _mark_occupied(occupied: np.ndarray, coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        occupied[r, c] = True


def _pair_candidate(
    h: int,
    w: int,
    occupied: np.ndarray,
    existing_bboxes: list[tuple[int, int, int, int]],
    predicate,
):
    pattern_order = PAIR_PATTERNS[:]
    random.shuffle(pattern_order)
    transforms = list(_all_transforms())
    random.shuffle(transforms)
    for pattern in pattern_order:
        for rot, flip_h in transforms:
            outer_rel, hh, ww = _transform_points(pattern["outer"], pattern["h"], pattern["w"], rot, flip_h)
            inner_rel, _, _ = _transform_points(pattern["inner"], pattern["h"], pattern["w"], rot, flip_h)
            max_r = h - hh - 1
            max_c = w - ww - 1
            if max_r < 1 or max_c < 1:
                continue
            anchors = [(r, c) for r in range(1, max_r + 1) for c in range(1, max_c + 1)]
            random.shuffle(anchors)
            for ar, ac in anchors:
                outer_abs = _shift_coords(outer_rel, ar, ac)
                inner_abs = _shift_coords(inner_rel, ar, ac)
                union_abs = outer_abs + inner_abs
                if not _cells_clear(occupied, union_abs):
                    continue
                outer_arr = np.array(outer_abs, dtype=int)
                inner_arr = np.array(inner_abs, dtype=int)
                outer_bbox = _bbox(outer_arr)
                inner_bbox = _bbox(inner_arr)
                if _bbox_relation_with_any(outer_bbox, existing_bboxes):
                    continue
                if _bbox_relation_with_any(inner_bbox, existing_bboxes):
                    continue
                outer_side, _ = _nearest_side(outer_arr, h, w)
                inner_side, _ = _nearest_side(inner_arr, h, w)
                if predicate(outer_side, inner_side):
                    return {
                        "outer": outer_arr,
                        "inner": inner_arr,
                        "outer_side": outer_side,
                        "inner_side": inner_side,
                        "outer_bbox": outer_bbox,
                        "inner_bbox": inner_bbox,
                    }
    return None


def _single_candidate(
    h: int,
    w: int,
    occupied: np.ndarray,
    existing_bboxes: list[tuple[int, int, int, int]],
    predicate,
):
    patterns = SINGLE_PATTERNS[:]
    random.shuffle(patterns)
    transforms = list(_all_transforms())
    random.shuffle(transforms)
    for pattern in patterns:
        for rot, flip_h in transforms:
            rel, hh, ww = _transform_points(pattern["cells"], pattern["h"], pattern["w"], rot, flip_h)
            max_r = h - hh - 1
            max_c = w - ww - 1
            if max_r < 1 or max_c < 1:
                continue
            anchors = [(r, c) for r in range(1, max_r + 1) for c in range(1, max_c + 1)]
            random.shuffle(anchors)
            for ar, ac in anchors:
                coords = _shift_coords(rel, ar, ac)
                if not _cells_clear(occupied, coords):
                    continue
                arr = np.array(coords, dtype=int)
                bbox = _bbox(arr)
                if _bbox_relation_with_any(bbox, existing_bboxes):
                    continue
                side, distances = _nearest_side(arr, h, w)
                minimum = min(distances.values())
                tie = sum(int(v == minimum) for v in distances.values()) >= 2
                if predicate(side, distances, tie):
                    return {"coords": arr, "side": side, "distances": distances, "tie": tie, "bbox": bbox}
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each gray interior object by its nearest frame side, then erase only differently recolored contained objects.

    Input structure:
    - The grid has a one-cell-thick rectangular outer frame whose corners are black and whose four sides are solid colors.
    - Inside the frame, the background is black and the foreground consists of several 4-connected gray objects.
    - Some gray-object bounding boxes fully contain smaller gray-object bounding boxes.

    Rule:
    - For each gray object, compute its minimum Chebyshev distance to each of the four frame sides and choose the nearest side, breaking ties by top > right > bottom > left.
    - Recolor the whole object to that side's frame color.
    - After all recoloring is finished, erase any smaller object whose bounding box is contained in a larger object's bounding box and whose recolored color differs from that larger object's color.
    - Keep same-color contained pairs and leave the frame unchanged.

    Key insight:
    - The object position first becomes a color label via nearest-side recoloring, and only those post-recolor labels decide which bbox-containment pairs are erased.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 7 and w >= 7
    assert int(g[0, 0]) == int(g[0, -1]) == int(g[-1, 0]) == int(g[-1, -1]) == BG

    side_colors = {
        "top": _uniform_side(g[0, 1:-1]),
        "right": _uniform_side(g[1:-1, -1]),
        "bottom": _uniform_side(g[-1, 1:-1]),
        "left": _uniform_side(g[1:-1, 0]),
    }

    objects = _find_components_of_color(g, GREY)
    assert objects

    out = g.copy()
    recolored = []
    for coords in objects:
        side, _ = _nearest_side(coords, h, w)
        color = side_colors[side]
        out[coords[:, 0], coords[:, 1]] = color
        recolored.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "bbox_area": _bbox_area(_bbox(coords)),
                "area": int(coords.shape[0]),
                "color": color,
            }
        )

    to_erase = set()
    for i, inner in enumerate(recolored):
        for j, outer in enumerate(recolored):
            if i == j:
                continue
            if outer["color"] == inner["color"]:
                continue
            if outer["area"] <= inner["area"]:
                continue
            if outer["bbox_area"] <= inner["bbox_area"]:
                continue
            if _bbox_contains(outer["bbox"], inner["bbox"]):
                to_erase.add(i)
                break

    for idx in to_erase:
        coords = recolored[idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
    return out


def _render_grid(h: int, w: int, objects: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[0, 1:-1] = RED
    grid[1:-1, -1] = BLUE
    grid[-1, 1:-1] = GREEN
    grid[1:-1, 0] = YELLOW
    for obj in objects:
        coords = obj["coords"]
        grid[coords[:, 0], coords[:, 1]] = GREY
    return grid


def _semantic_summary(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    items = []
    for coords in _find_components_of_color(grid, GREY):
        side, distances = _nearest_side(coords, h, w)
        bbox = _bbox(coords)
        items.append(
            {
                "coords": coords,
                "bbox": bbox,
                "bbox_area": _bbox_area(bbox),
                "area": int(coords.shape[0]),
                "side": side,
                "distances": distances,
            }
        )
    return items


def _validate_generated(grid: np.ndarray, metadata: list[dict], tie_expected: bool) -> bool:
    h, w = grid.shape
    if not (grid.ndim == 2 and max(grid.shape) <= 30):
        return False
    if not (
        int(grid[0, 0]) == int(grid[0, -1]) == int(grid[-1, 0]) == int(grid[-1, -1]) == BG
        and np.all(grid[0, 1:-1] == RED)
        and np.all(grid[1:-1, -1] == BLUE)
        and np.all(grid[-1, 1:-1] == GREEN)
        and np.all(grid[1:-1, 0] == YELLOW)
    ):
        return False

    expected_names = {item["name"] for item in metadata}
    if len(metadata) < 5 or len(metadata) > 8 or len(expected_names) != len(metadata):
        return False

    observed = _semantic_summary(grid)
    if len(observed) != len(metadata):
        return False

    by_name = {item["name"]: item for item in metadata}
    observed_by_name = {}
    for obj in observed:
        coord_set = {tuple(map(int, rc)) for rc in obj["coords"]}
        matches = [name for name, item in by_name.items() if coord_set == item["coord_set"]]
        if len(matches) != 1:
            return False
        observed_by_name[matches[0]] = obj
    if set(observed_by_name) != expected_names:
        return False

    for name, obj in observed_by_name.items():
        if not (3 <= obj["area"] <= 8):
            return False
        if obj["bbox_area"] == obj["area"]:
            return False
        if obj["side"] != by_name[name]["side"]:
            return False

    if {obj["side"] for obj in observed_by_name.values()} != set(SIDE_ORDER):
        return False

    contained = []
    names = list(observed_by_name)
    for outer_name in names:
        for inner_name in names:
            if outer_name == inner_name:
                continue
            outer = observed_by_name[outer_name]
            inner = observed_by_name[inner_name]
            if outer["area"] <= inner["area"] or outer["bbox_area"] <= inner["bbox_area"]:
                continue
            if _bbox_contains(outer["bbox"], inner["bbox"]):
                contained.append((outer_name, inner_name, outer["side"] == inner["side"]))

    if sorted(contained) != sorted(
        [
            ("same_outer", "same_inner", True),
            ("diff_outer", "diff_inner", False),
        ]
    ):
        return False

    if observed_by_name["same_outer"]["side"] != observed_by_name["same_inner"]["side"]:
        return False
    if observed_by_name["diff_outer"]["side"] == observed_by_name["diff_inner"]["side"]:
        return False

    out = main(grid)

    same_color = SIDE_COLORS[observed_by_name["same_outer"]["side"]]
    diff_outer_color = SIDE_COLORS[observed_by_name["diff_outer"]["side"]]
    diff_inner_coords = observed_by_name["diff_inner"]["coords"]
    same_inner_coords = observed_by_name["same_inner"]["coords"]
    same_outer_coords = observed_by_name["same_outer"]["coords"]
    diff_outer_coords = observed_by_name["diff_outer"]["coords"]

    if not np.all(out[same_outer_coords[:, 0], same_outer_coords[:, 1]] == same_color):
        return False
    if not np.all(out[same_inner_coords[:, 0], same_inner_coords[:, 1]] == same_color):
        return False
    if not np.all(out[diff_outer_coords[:, 0], diff_outer_coords[:, 1]] == diff_outer_color):
        return False
    if not np.all(out[diff_inner_coords[:, 0], diff_inner_coords[:, 1]] == BG):
        return False

    has_tie = any(by_name[name]["tie"] for name in by_name)
    if tie_expected and not has_tie:
        return False
    if has_tie:
        tie_names = [name for name, item in by_name.items() if item["tie"]]
        for name in tie_names:
            distances = by_name[name]["distances"]
            minimum = min(distances.values())
            if sum(int(v == minimum) for v in distances.values()) < 2:
                return False
            chosen = by_name[name]["side"]
            for side in SIDE_ORDER:
                if distances[side] == minimum:
                    if side != chosen:
                        return False
                    break

    return not np.array_equal(grid, out)


def _generate_attempt() -> tuple[np.ndarray, list[dict], bool] | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    occupied = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    existing_bboxes: list[tuple[int, int, int, int]] = []

    same_side = random.choice(SIDE_ORDER)
    same_pair = _pair_candidate(
        h,
        w,
        occupied,
        existing_bboxes,
        predicate=lambda outer_side, inner_side: outer_side == same_side and inner_side == same_side,
    )
    if same_pair is None:
        return None

    for name, coords, side, bbox in (
        ("same_outer", same_pair["outer"], same_pair["outer_side"], same_pair["outer_bbox"]),
        ("same_inner", same_pair["inner"], same_pair["inner_side"], same_pair["inner_bbox"]),
    ):
        coord_set = {tuple(map(int, rc)) for rc in coords}
        objects.append(
            {
                "name": name,
                "coords": coords,
                "coord_set": coord_set,
                "bbox": bbox,
                "side": side,
                "distances": _nearest_side(coords, h, w)[1],
                "tie": False,
            }
        )
        existing_bboxes.append(bbox)
        _mark_occupied(occupied, [tuple(map(int, rc)) for rc in coords])

    diff_pair = _pair_candidate(
        h,
        w,
        occupied,
        existing_bboxes,
        predicate=lambda outer_side, inner_side: outer_side != inner_side,
    )
    if diff_pair is None:
        return None

    for name, coords, side, bbox in (
        ("diff_outer", diff_pair["outer"], diff_pair["outer_side"], diff_pair["outer_bbox"]),
        ("diff_inner", diff_pair["inner"], diff_pair["inner_side"], diff_pair["inner_bbox"]),
    ):
        coord_set = {tuple(map(int, rc)) for rc in coords}
        objects.append(
            {
                "name": name,
                "coords": coords,
                "coord_set": coord_set,
                "bbox": bbox,
                "side": side,
                "distances": _nearest_side(coords, h, w)[1],
                "tie": False,
            }
        )
        existing_bboxes.append(bbox)
        _mark_occupied(occupied, [tuple(map(int, rc)) for rc in coords])

    represented = {obj["side"] for obj in objects}
    for missing_side in SIDE_ORDER:
        if missing_side in represented:
            continue
        cand = _single_candidate(
            h,
            w,
            occupied,
            existing_bboxes,
            predicate=lambda side, distances, tie, target=missing_side: side == target and not tie,
        )
        if cand is None:
            return None
        name = f"cover_{missing_side}"
        coord_set = {tuple(map(int, rc)) for rc in cand["coords"]}
        objects.append(
            {
                "name": name,
                "coords": cand["coords"],
                "coord_set": coord_set,
                "bbox": cand["bbox"],
                "side": cand["side"],
                "distances": cand["distances"],
                "tie": False,
            }
        )
        existing_bboxes.append(cand["bbox"])
        _mark_occupied(occupied, [tuple(map(int, rc)) for rc in cand["coords"]])

    tie_expected = random.random() < OPTIONAL_TIE_PROB and len(objects) < 8
    if tie_expected:
        tie_cand = _single_candidate(
            h,
            w,
            occupied,
            existing_bboxes,
            predicate=lambda side, distances, tie: tie,
        )
        if tie_cand is None:
            return None
        coord_set = {tuple(map(int, rc)) for rc in tie_cand["coords"]}
        objects.append(
            {
                "name": "tie_obj",
                "coords": tie_cand["coords"],
                "coord_set": coord_set,
                "bbox": tie_cand["bbox"],
                "side": tie_cand["side"],
                "distances": tie_cand["distances"],
                "tie": True,
            }
        )
        existing_bboxes.append(tie_cand["bbox"])
        _mark_occupied(occupied, [tuple(map(int, rc)) for rc in tie_cand["coords"]])

    while len(objects) < 8 and random.random() < 0.55:
        extra = _single_candidate(
            h,
            w,
            occupied,
            existing_bboxes,
            predicate=lambda side, distances, tie: not tie,
        )
        if extra is None:
            break
        name = f"extra_{len(objects)}"
        coord_set = {tuple(map(int, rc)) for rc in extra["coords"]}
        objects.append(
            {
                "name": name,
                "coords": extra["coords"],
                "coord_set": coord_set,
                "bbox": extra["bbox"],
                "side": extra["side"],
                "distances": extra["distances"],
                "tie": False,
            }
        )
        existing_bboxes.append(extra["bbox"])
        _mark_occupied(occupied, [tuple(map(int, rc)) for rc in extra["coords"]])

    grid = _render_grid(h, w, objects)
    if not _validate_generated(grid, objects, tie_expected):
        return None
    return grid, objects, tie_expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        attempt = _generate_attempt()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("failed to generate a valid input for puzzle 7")
