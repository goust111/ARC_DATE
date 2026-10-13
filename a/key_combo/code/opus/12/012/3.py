from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
ARROW = 2
OBJECT = 3

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRECTIONS = ("up", "down", "left", "right")
SIDE_TO_DIR = ("top", "bottom", "left", "right")
SIDE_DIRECTION = {"top": "up", "bottom": "down", "left": "left", "right": "right"}


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            out.append(
                {
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    col_gap = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return int(row_gap + col_gap)


def _decode_arrow_direction(coords: np.ndarray) -> str:
    pts = [tuple(map(int, p)) for p in coords]
    row_counts = Counter(r for r, _ in pts)
    col_counts = Counter(c for _, c in pts)

    long_rows = [r for r, n in row_counts.items() if n == 4]
    long_cols = [c for c, n in col_counts.items() if n == 4]
    cross_rows = [r for r, n in row_counts.items() if n == 3]
    cross_cols = [c for c, n in col_counts.items() if n == 3]

    is_horizontal = len(long_rows) == 1 and len(cross_cols) == 1 and not long_cols and not cross_rows
    is_vertical = len(long_cols) == 1 and len(cross_rows) == 1 and not long_rows and not cross_cols
    assert is_horizontal ^ is_vertical, "expected a six-pixel asymmetric T arrow"

    if is_horizontal:
        r = int(long_rows[0])
        cross_c = int(cross_cols[0])
        line_cols = sorted(c for rr, c in pts if rr == r)
        assert line_cols == list(range(line_cols[0], line_cols[0] + 4))
        assert sorted(rr for rr, c in pts if c == cross_c) == [r - 1, r, r + 1]
        left_dist = cross_c - line_cols[0]
        right_dist = line_cols[-1] - cross_c
        assert {left_dist, right_dist} == {1, 2}
        return "left" if left_dist < right_dist else "right"

    c = int(long_cols[0])
    cross_r = int(cross_rows[0])
    line_rows = sorted(r for r, cc in pts if cc == c)
    assert line_rows == list(range(line_rows[0], line_rows[0] + 4))
    assert sorted(cc for r, cc in pts if r == cross_r) == [c - 1, c, c + 1]
    up_dist = cross_r - line_rows[0]
    down_dist = line_rows[-1] - cross_r
    assert {up_dist, down_dist} == {1, 2}
    return "up" if up_dist < down_dist else "down"


def _pair_arrows_to_objects(objects: list[dict], arrows: list[dict]) -> list[int]:
    assignments: list[int] = []
    for arrow in arrows:
        ranked = []
        for obj_idx, obj in enumerate(objects):
            ranked.append((_bbox_distance(arrow["bbox"], obj["bbox"]), obj["bbox"][0], obj["bbox"][1], obj_idx))
        ranked.sort()
        assignments.append(int(ranked[0][3]))

    counts = Counter(assignments)
    assert len(assignments) == len(objects) == len(arrows), "expected one arrow per object"
    assert all(counts[i] == 1 for i in range(len(objects))), "pairing must be bijective"
    return assignments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each nearby red arrow's encoded direction to erase the corresponding side of a green object's local bounding box.

    Input structure:
    - The grid has a black background with several irregular green 4-connected objects.
    - Near each object is one red six-pixel asymmetric T arrow, separated from the object's bounding box by 1-2 cells.
    - The arrow's crossbar is offset toward the pointed end, which encodes one of the four cardinal directions.

    Rule:
    - Find all green 4-connected objects and their bounding boxes.
    - Find all red arrow sprites, decode each arrow's direction from the off-center crossbar, and pair the arrow with the nearest object by bounding-box distance, breaking ties by the object's top row then left column.
    - For each paired object, erase only that object's green pixels inside the two bbox rows or columns on the decoded side: top for up, bottom for down, left for left, right for right.
    - Erase all red arrow pixels to background and leave all other pixels unchanged.

    Key insight:
    - The arrow does not mark a place to extend from; it is a local instruction token whose internal asymmetry encodes which side of the matched object's own bbox must be deleted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    objects = sorted(_components_of_color(g, OBJECT), key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    arrows = sorted(_components_of_color(g, ARROW), key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    assert objects and len(objects) == len(arrows), "expected equal counts of objects and arrows"

    for arrow in arrows:
        assert int(arrow["area"]) == 6, "each arrow must be a six-pixel sprite"
        arrow["direction"] = _decode_arrow_direction(arrow["coords"])

    assignments = _pair_arrows_to_objects(objects, arrows)

    out = g.copy()
    out[out == ARROW] = BG
    for arrow, obj_idx in zip(arrows, assignments, strict=False):
        coords = objects[obj_idx]["coords"]
        r0, c0, r1, c1 = objects[obj_idx]["bbox"]
        direction = arrow["direction"]
        rr = coords[:, 0]
        cc = coords[:, 1]

        if direction == "left":
            mask = cc <= c0 + 1
        elif direction == "right":
            mask = cc >= c1 - 1
        elif direction == "up":
            mask = rr <= r0 + 1
        else:
            assert direction == "down"
            mask = rr >= r1 - 1

        doomed = coords[mask]
        out[doomed[:, 0], doomed[:, 1]] = BG

    return out


def _make_arrow_patch(direction: str) -> np.ndarray:
    if direction in {"left", "right"}:
        patch = np.zeros((3, 4), dtype=int)
        patch[1, :] = ARROW
        patch[:, 1 if direction == "left" else 2] = ARROW
        return patch

    patch = np.zeros((4, 3), dtype=int)
    patch[:, 1] = ARROW
    patch[1 if direction == "up" else 2, :] = ARROW
    return patch


def _grow_object_patch() -> tuple[np.ndarray, tuple[str, ...]]:
    for _ in range(200):
        h = random.randint(4, 5)
        w = random.randint(4, 5)
        target_area = random.randint(8, min(15, h * w - 1))
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target_area:
            frontier = sorted(
                {
                    (r + dr, c + dc)
                    for r, c in cells
                    for dr, dc in DIRS4
                    if 0 <= r + dr < h and 0 <= c + dc < w and (r + dr, c + dc) not in cells
                }
            )
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target_area:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
            continue
        if len(cells) == h * w:
            continue

        mask = np.zeros((h, w), dtype=int)
        for r, c in cells:
            mask[r, c] = OBJECT

        viable: list[str] = []
        if mask[:, 0].any() and mask[:, 1].any():
            viable.append("left")
        if mask[:, -1].any() and mask[:, -2].any():
            viable.append("right")
        if mask[0, :].any() and mask[1, :].any():
            viable.append("up")
        if mask[-1, :].any() and mask[-2, :].any():
            viable.append("down")

        if len(viable) < 2:
            continue

        for direction in viable:
            erased = main(_compose_demo(mask, _make_arrow_patch(direction), direction))
            if np.count_nonzero(erased == OBJECT) == 0:
                break
        else:
            return mask, tuple(viable)
    raise RuntimeError("failed to sample a valid object patch")


def _compose_demo(obj_patch: np.ndarray, arrow_patch: np.ndarray, direction: str) -> np.ndarray:
    """Small helper to test that an object is not fully erased for a given direction."""
    oh, ow = obj_patch.shape
    ah, aw = arrow_patch.shape
    canvas = np.zeros((oh + ah + 3, ow + aw + 3), dtype=int)
    canvas[ah + 2 : ah + 2 + oh, aw + 2 : aw + 2 + ow] = obj_patch
    canvas[0:ah, 0:aw] = arrow_patch
    return canvas


def _interval_overlap_positions(obj_start: int, obj_len: int, arrow_len: int, limit: int) -> list[int]:
    low = max(0, obj_start - arrow_len + 1)
    high = min(limit - arrow_len, obj_start + obj_len - 1)
    return list(range(low, high + 1))


def _place_pair_in_zone(
    grid: np.ndarray,
    zone: tuple[int, int, int, int],
    obj_patch: np.ndarray,
    direction: str,
    anchor_side: str,
    gap: int,
) -> tuple[dict, dict] | None:
    zr0, zc0, zr1, zc1 = zone
    zh = zr1 - zr0 + 1
    zw = zc1 - zc0 + 1
    oh, ow = obj_patch.shape
    arrow_patch = _make_arrow_patch(direction)
    ah, aw = arrow_patch.shape

    placements: list[tuple[int, int, int, int]] = []
    for obj_r in range(0, zh - oh + 1):
        for obj_c in range(0, zw - ow + 1):
            if anchor_side == "top":
                arrow_r = obj_r - gap - ah
                if arrow_r < 0:
                    continue
                for arrow_c in _interval_overlap_positions(obj_c, ow, aw, zw):
                    placements.append((obj_r, obj_c, arrow_r, arrow_c))
            elif anchor_side == "bottom":
                arrow_r = obj_r + oh + gap
                if arrow_r + ah > zh:
                    continue
                for arrow_c in _interval_overlap_positions(obj_c, ow, aw, zw):
                    placements.append((obj_r, obj_c, arrow_r, arrow_c))
            elif anchor_side == "left":
                arrow_c = obj_c - gap - aw
                if arrow_c < 0:
                    continue
                for arrow_r in _interval_overlap_positions(obj_r, oh, ah, zh):
                    placements.append((obj_r, obj_c, arrow_r, arrow_c))
            else:
                assert anchor_side == "right"
                arrow_c = obj_c + ow + gap
                if arrow_c + aw > zw:
                    continue
                for arrow_r in _interval_overlap_positions(obj_r, oh, ah, zh):
                    placements.append((obj_r, obj_c, arrow_r, arrow_c))

    random.shuffle(placements)
    for obj_r, obj_c, arrow_r, arrow_c in placements:
        obj_view = grid[zr0 + obj_r : zr0 + obj_r + oh, zc0 + obj_c : zc0 + obj_c + ow]
        arrow_view = grid[zr0 + arrow_r : zr0 + arrow_r + ah, zc0 + arrow_c : zc0 + arrow_c + aw]
        if np.any((obj_patch != 0) & (obj_view != BG)):
            continue
        if np.any((arrow_patch != 0) & (arrow_view != BG)):
            continue
        obj_view[obj_patch != 0] = OBJECT
        arrow_view[arrow_patch != 0] = ARROW

        obj_cells = np.argwhere(obj_patch == OBJECT) + np.array([[zr0 + obj_r, zc0 + obj_c]])
        arrow_cells = np.argwhere(arrow_patch == ARROW) + np.array([[zr0 + arrow_r, zc0 + arrow_c]])
        return (
            {
                "coords": obj_cells,
                "bbox": _bbox_from_coords(obj_cells),
                "direction": direction,
            },
            {
                "coords": arrow_cells,
                "bbox": _bbox_from_coords(arrow_cells),
                "direction": direction,
                "anchor_side": anchor_side,
            },
        )
    return None


def _erase_expected(grid: np.ndarray, objects: list[dict], arrows: list[dict]) -> np.ndarray:
    out = grid.copy()
    out[out == ARROW] = BG
    for obj, arrow in zip(objects, arrows, strict=False):
        coords = obj["coords"]
        r0, c0, r1, c1 = obj["bbox"]
        rr = coords[:, 0]
        cc = coords[:, 1]
        if arrow["direction"] == "left":
            mask = cc <= c0 + 1
        elif arrow["direction"] == "right":
            mask = cc >= c1 - 1
        elif arrow["direction"] == "up":
            mask = rr <= r0 + 1
        else:
            mask = rr >= r1 - 1
        doomed = coords[mask]
        out[doomed[:, 0], doomed[:, 1]] = BG
    return out


def _zones_for_grid(h: int, w: int, count: int) -> list[tuple[int, int, int, int]]:
    row_cut = h // 2
    col_cut = w // 2
    zones = [
        (1, 1, row_cut - 1, col_cut - 1),
        (1, col_cut, row_cut - 1, w - 2),
        (row_cut, 1, h - 2, col_cut - 1),
        (row_cut, col_cut, h - 2, w - 2),
    ]
    picks = random.sample(range(4), count)
    return [zones[i] for i in picks]


def _sample_directions(count: int, viable_lists: list[tuple[str, ...]]) -> list[str] | None:
    for _ in range(200):
        dirs = [random.choice(vs) for vs in viable_lists]
        if len(set(dirs)) < 2:
            continue
        return dirs
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(22, 26)
    w = random.randint(22, 26)
    count = random.randint(3, 4)
    grid = np.zeros((h, w), dtype=int)

    obj_patches: list[np.ndarray] = []
    viable_lists: list[tuple[str, ...]] = []
    for _ in range(count):
        patch, viable = _grow_object_patch()
        obj_patches.append(patch)
        viable_lists.append(viable)

    directions = _sample_directions(count, viable_lists)
    if directions is None:
        return None

    anchor_sides = [random.choice(SIDE_TO_DIR) for _ in range(count)]
    if all(SIDE_DIRECTION[anchor] == direction for anchor, direction in zip(anchor_sides, directions, strict=False)):
        idx = random.randrange(count)
        choices = [side for side in SIDE_TO_DIR if SIDE_DIRECTION[side] != directions[idx]]
        anchor_sides[idx] = random.choice(choices)

    zones = _zones_for_grid(h, w, count)
    objects: list[dict] = []
    arrows: list[dict] = []
    for zone, patch, direction, anchor_side in zip(zones, obj_patches, directions, anchor_sides, strict=False):
        placed = _place_pair_in_zone(grid, zone, patch, direction, anchor_side, gap=random.randint(1, 2))
        if placed is None:
            return None
        obj, arrow = placed
        objects.append(obj)
        arrows.append(arrow)

    found_objects = sorted(_components_of_color(grid, OBJECT), key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    found_arrows = sorted(_components_of_color(grid, ARROW), key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    if len(found_objects) != count or len(found_arrows) != count:
        return None

    for arrow in found_arrows:
        try:
            arrow["direction"] = _decode_arrow_direction(arrow["coords"])
        except AssertionError:
            return None

    try:
        assignments = _pair_arrows_to_objects(found_objects, found_arrows)
    except AssertionError:
        return None
    expected_by_generator = {}
    for obj, arrow in zip(objects, arrows, strict=False):
        expected_by_generator[arrow["bbox"]] = obj["bbox"]
    for arrow, obj_idx in zip(found_arrows, assignments, strict=False):
        if expected_by_generator.get(arrow["bbox"]) != found_objects[obj_idx]["bbox"]:
            return None

    if len(set(arrow["direction"] for arrow in found_arrows)) < 2:
        return None
    if all(SIDE_DIRECTION[arrow["anchor_side"]] == arrow["direction"] for arrow in arrows):
        return None

    expected = _erase_expected(grid, found_objects, found_arrows)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None

    # Reject overly sparse side erasures that do not visibly use the second band.
    for obj, arrow in zip(found_objects, found_arrows, strict=False):
        r0, c0, r1, c1 = obj["bbox"]
        coords = obj["coords"]
        rr = coords[:, 0]
        cc = coords[:, 1]
        if arrow["direction"] == "left" and not np.any(cc == c0 + 1):
            return None
        if arrow["direction"] == "right" and not np.any(cc == c1 - 1):
            return None
        if arrow["direction"] == "up" and not np.any(rr == r0 + 1):
            return None
        if arrow["direction"] == "down" and not np.any(rr == r1 - 1):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
