from __future__ import annotations

import random
from collections import deque
from itertools import combinations

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
COUNT_TO_TURNS = {1: 1, 2: 2, 3: 3}

GEN_SLOT_HEIGHTS = (6, 7)
GEN_SLOT_WIDTHS = (6, 7)
GEN_GUTTERS = (4, 5)
GEN_MARGINS = (0, 1)
GEN_MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _rotate_local_clockwise(
    coords: tuple[tuple[int, int], ...],
    height: int,
    width: int,
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rotated = tuple((c, height - 1 - r) for r, c in coords)
    return _normalize(rotated), width, height


def _rotation_signatures(
    coords: tuple[tuple[int, int], ...],
    height: int,
    width: int,
) -> list[tuple[tuple[int, int], ...]]:
    out = []
    cur = coords
    h = height
    w = width
    for _ in range(4):
        out.append(_normalize(cur))
        cur, h, w = _rotate_local_clockwise(cur, h, w)
    return out


def _marker_candidates_local(
    coords: tuple[tuple[int, int], ...],
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    shape = {(int(r), int(c)) for r, c in coords}
    candidates = []
    for r in range(-1, height + 1):
        for c in range(-1, width + 1):
            if 0 <= r < height and 0 <= c < width:
                continue
            if any(max(abs(r - sr), abs(c - sc)) <= 1 for sr, sc in shape):
                continue
            candidates.append((r, c))
    return candidates


def _shape_library() -> list[dict]:
    library: list[dict] = []
    for height, width in ((3, 3), (2, 4), (4, 2)):
        cells = [(r, c) for r in range(height) for c in range(width)]
        for area in range(4, min(8, height * width) + 1):
            for pick in combinations(cells, area):
                coords = _normalize(pick)
                if _bbox(np.array(coords, dtype=int)) != (0, 0, height - 1, width - 1):
                    continue
                if len(_components(np.array([[((r, c) in set(coords)) for c in range(width)] for r in range(height)], dtype=bool))) != 1:
                    continue
                if len(set(_rotation_signatures(coords, height, width))) != 4:
                    continue
                marker_candidates = _marker_candidates_local(coords, height, width)
                if len(marker_candidates) < 3:
                    continue
                library.append(
                    {
                        "coords": coords,
                        "height": height,
                        "width": width,
                        "marker_candidates": tuple(marker_candidates),
                    }
                )
    assert library, "shape library must not be empty"
    return library


SHAPE_LIBRARY = _shape_library()


def _sprite_from_coords(coords: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    sprite[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)
    return sprite


def _count_ring_markers(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    total = 0
    for r in range(max(0, r0 - 1), min(h - 1, r1 + 1) + 1):
        for c in range(max(0, c0 - 1), min(w - 1, c1 + 1) + 1):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            total += int(grid[r, c] == GREY)
    return total


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) not in (BLACK, GREY)]
    objects: list[dict] = []
    for color in colors:
        for coords in _components(g == color):
            bbox = _bbox(coords)
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "count": _count_ring_markers(g, bbox),
                    "sprite": _sprite_from_coords(coords, bbox, color),
                }
            )
    objects.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return objects


def _paste_rotated(out: np.ndarray, obj: dict, quarter_turns: int) -> None:
    sprite = obj["sprite"]
    rotated = np.rot90(sprite, k=(-quarter_turns) % 4)
    h, w = sprite.shape
    h2, w2 = rotated.shape
    r0, c0, _r1, _c1 = obj["bbox"]
    dest_r0 = r0 + (h - h2) // 2
    dest_c0 = c0 + (w - w2) // 2
    assert 0 <= dest_r0 and 0 <= dest_c0
    assert dest_r0 + h2 <= out.shape[0] and dest_c0 + w2 <= out.shape[1]
    mask = rotated != BLACK
    patch = out[dest_r0 : dest_r0 + h2, dest_c0 : dest_c0 + w2]
    assert not np.any(mask & (patch != BLACK))
    patch[mask] = rotated[mask]


def _render_output(grid: np.ndarray, turns_by_count: dict[int, int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    objects = _extract_objects(g)
    out = np.where(g == GREY, BLACK, g).copy()
    for obj in objects:
        rr = obj["coords"][:, 0]
        cc = obj["coords"][:, 1]
        out[rr, cc] = BLACK
    for obj in objects:
        turns = int(turns_by_count[obj["count"]])
        _paste_rotated(out, obj, turns)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each colored sprite by the angle encoded in the number of nearby gray markers, then erase the markers.

    Input structure:
    - The grid is a black background with 3-4 differently colored 4-connected monochrome sprites.
    - Around each sprite's bounding box, the one-cell outer ring may contain 1-3 isolated gray marker pixels.
    - Different sprites are well separated, and the gray markers belong to exactly one sprite's ring.

    Rule:
    - For each colored sprite, count the gray pixels in the one-cell ring just outside its bounding box.
    - Rotate that sprite around its bounding-box center using the mapping 1 -> 90 degrees clockwise, 2 -> 180 degrees, 3 -> 270 degrees clockwise.
    - Re-center the rotated sprite on the original bounding-box center, and erase all gray markers to black.

    Key insight:
    - The gray pixels are not decoration: only their local count tells you how far each nearby sprite must rotate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _render_output(g, COUNT_TO_TURNS)


def _rect_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    dc = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return max(dr, dc)


def _in_bbox_ring(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1 and not (r0 <= r <= r1 and c0 <= c <= c1)


def _sample_non_adjacent_markers(
    marker_candidates: tuple[tuple[int, int], ...],
    count: int,
) -> tuple[tuple[int, int], ...] | None:
    valid = []
    for pick in combinations(marker_candidates, count):
        ok = True
        for i in range(len(pick)):
            for j in range(i + 1, len(pick)):
                if max(abs(pick[i][0] - pick[j][0]), abs(pick[i][1] - pick[j][1])) <= 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            valid.append(tuple(pick))
    if not valid:
        return None
    return random.choice(valid)


def _oriented_shape(shape_spec: dict, initial_turns: int) -> dict:
    coords = tuple(shape_spec["coords"])
    h = int(shape_spec["height"])
    w = int(shape_spec["width"])
    for _ in range(initial_turns % 4):
        coords, h, w = _rotate_local_clockwise(coords, h, w)
    return {
        "coords": coords,
        "height": h,
        "width": w,
        "marker_candidates": tuple(_marker_candidates_local(coords, h, w)),
    }


def _placement_candidates(
    slot_top: int,
    slot_left: int,
    slot_height: int,
    slot_width: int,
    obj_height: int,
    obj_width: int,
    out_height: int,
    out_width: int,
) -> list[tuple[int, int]]:
    positions = []
    for r0 in range(slot_top + 1, slot_top + slot_height - obj_height):
        for c0 in range(slot_left + 1, slot_left + slot_width - obj_width):
            out_r0 = r0 + (obj_height - out_height) // 2
            out_c0 = c0 + (obj_width - out_width) // 2
            if out_r0 < slot_top or out_c0 < slot_left:
                continue
            if out_r0 + out_height > slot_top + slot_height:
                continue
            if out_c0 + out_width > slot_left + slot_width:
                continue
            positions.append((r0, c0))
    return positions


def _rotation_nontrivial(sprite: np.ndarray) -> bool:
    signatures = {tuple(map(tuple, np.argwhere(np.rot90(sprite, k=k) != BLACK))) for k in range(4)}
    return len(signatures) == 4


def _generate_candidate() -> np.ndarray | None:
    object_count = random.choice((3, 4))
    counts = [1, 2, 3]
    if object_count == 4:
        counts.append(random.choice((1, 2, 3)))
    random.shuffle(counts)

    slot_height = random.choice(GEN_SLOT_HEIGHTS)
    slot_width = random.choice(GEN_SLOT_WIDTHS)
    gutter_r = random.choice(GEN_GUTTERS)
    gutter_c = random.choice(GEN_GUTTERS)
    top_margin = random.choice(GEN_MARGINS)
    left_margin = random.choice(GEN_MARGINS)
    bottom_margin = random.choice(GEN_MARGINS)
    right_margin = random.choice(GEN_MARGINS)

    height = top_margin + 2 * slot_height + gutter_r + bottom_margin
    width = left_margin + 2 * slot_width + gutter_c + right_margin
    if max(height, width) > 30:
        return None

    slot_origins = [
        (top_margin + sr * (slot_height + gutter_r), left_margin + sc * (slot_width + gutter_c))
        for sr in range(2)
        for sc in range(2)
    ]
    chosen_slots = random.sample(range(4), object_count)
    colors = random.sample(PALETTE, object_count)
    shape_specs = random.sample(SHAPE_LIBRARY, object_count)

    placements = []
    for idx, slot_idx in enumerate(chosen_slots):
        slot_top, slot_left = slot_origins[slot_idx]
        count = counts[idx]
        color = colors[idx]
        oriented = _oriented_shape(shape_specs[idx], random.randint(0, 3))
        coords_local = oriented["coords"]
        obj_height = int(oriented["height"])
        obj_width = int(oriented["width"])
        out_coords, out_height, out_width = coords_local, obj_height, obj_width
        for _ in range(count % 4):
            out_coords, out_height, out_width = _rotate_local_clockwise(out_coords, out_height, out_width)
        placement_options = _placement_candidates(
            slot_top,
            slot_left,
            slot_height,
            slot_width,
            obj_height,
            obj_width,
            out_height,
            out_width,
        )
        if not placement_options:
            return None
        r0, c0 = random.choice(placement_options)
        marker_offsets = _sample_non_adjacent_markers(oriented["marker_candidates"], count)
        if marker_offsets is None:
            return None
        coords = np.array([(r0 + r, c0 + c) for r, c in coords_local], dtype=int)
        bbox = (r0, c0, r0 + obj_height - 1, c0 + obj_width - 1)
        markers = [(r0 + dr, c0 + dc) for dr, dc in marker_offsets]
        sprite = np.zeros((obj_height, obj_width), dtype=int)
        sprite[coords[:, 0] - r0, coords[:, 1] - c0] = color
        if not _rotation_nontrivial(sprite):
            return None
        placements.append(
            {
                "color": color,
                "coords": coords,
                "bbox": bbox,
                "count": count,
                "markers": markers,
            }
        )

    for i in range(len(placements)):
        for j in range(i + 1, len(placements)):
            if _rect_gap(placements[i]["bbox"], placements[j]["bbox"]) < 4:
                return None

    grid = np.zeros((height, width), dtype=int)
    for item in placements:
        rr = item["coords"][:, 0]
        cc = item["coords"][:, 1]
        grid[rr, cc] = int(item["color"])
        for r, c in item["markers"]:
            if not (0 <= r < height and 0 <= c < width):
                return None
            if grid[r, c] != BLACK:
                return None
            grid[r, c] = GREY

    return grid


def _validate_semantics(grid: np.ndarray) -> bool:
    objects = _extract_objects(grid)
    assert len(objects) in (3, 4), "expected 3-4 sprites"
    counts = [int(obj["count"]) for obj in objects]
    assert set(counts) == {1, 2, 3}, "must cover all marker-count branches"
    assert all(count in COUNT_TO_TURNS for count in counts)

    grey_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)]
    assert len(grey_coords) == sum(counts)

    for grey in grey_coords:
        owners = [obj for obj in objects if _in_bbox_ring(grey, obj["bbox"])]
        assert len(owners) == 1, "every marker must belong to exactly one bbox ring"
        owner = owners[0]
        assert all(max(abs(grey[0] - int(r)), abs(grey[1] - int(c))) > 1 for r, c in owner["coords"])

    for i in range(len(grey_coords)):
        for j in range(i + 1, len(grey_coords)):
            assert max(abs(grey_coords[i][0] - grey_coords[j][0]), abs(grey_coords[i][1] - grey_coords[j][1])) > 1

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            assert _rect_gap(objects[i]["bbox"], objects[j]["bbox"]) >= 4

    for obj in objects:
        sprite = obj["sprite"]
        assert _rotation_nontrivial(sprite), "every sprite must change under each rotation"

    solved = main(grid)
    assert solved.shape == grid.shape
    assert not np.any(solved == GREY), "markers must be erased"
    assert not np.array_equal(solved, grid), "rotation must have a visible effect"

    out_objects = _extract_objects(solved)
    assert len(out_objects) == len(objects)
    assert sorted(obj["color"] for obj in out_objects) == sorted(obj["color"] for obj in objects)

    alternative_maps = [
        {1: 0, 2: 0, 3: 0},
        {1: 1, 2: 1, 3: 1},
        {1: 2, 2: 2, 3: 2},
        {1: 3, 2: 3, 3: 3},
        {1: 3, 2: 2, 3: 1},
        {1: 1, 2: 3, 3: 2},
    ]
    for mapping in alternative_maps:
        wrong = _render_output(grid, mapping)
        assert not np.array_equal(solved, wrong), f"shortcut mapping {mapping} must fail"

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_candidate()
        if grid is None:
            continue
        try:
            if _validate_semantics(grid):
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input for puzzle 9")
