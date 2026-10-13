from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _ARC_UTILS_DIR not in sys.path:
    sys.path.append(_ARC_UTILS_DIR)

from arc_utils import colors_present, components_mask, get_bg


DIR_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_RETRIES = 300
GEN_MARGIN = 1
GEN_BG = 0
GEN_COLORS = (1, 2, 3, 4)
GEN_COUNTS = (3, 4)
GEN_GRID_ROWS = (18, 26)
GEN_GRID_COLS = (18, 28)
GEN_AREAS = (6, 8, 9, 12, 16)

HORIZONTAL_BODY_OPTIONS = {
    6: [(3, 2)],
    8: [(2, 4), (4, 2)],
    9: [(3, 3)],
    12: [(3, 4), (4, 3)],
    16: [(4, 4)],
}

VERTICAL_BODY_OPTIONS = {
    6: [(2, 3)],
    8: [(2, 4), (4, 2)],
    9: [(3, 3)],
    12: [(3, 4), (4, 3)],
    16: [(4, 4)],
}


def _rect_sum(integral: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    total = int(integral[r1, c1])
    if r0 > 0:
        total -= int(integral[r0 - 1, c1])
    if c0 > 0:
        total -= int(integral[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(integral[r0 - 1, c0 - 1])
    return total


def _largest_solid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty component"
    rmin, cmin = coords.min(axis=0)
    rmax, cmax = coords.max(axis=0)
    sub = mask[rmin : rmax + 1, cmin : cmax + 1].astype(int)
    integral = sub.cumsum(axis=0).cumsum(axis=1)

    best_area = 0
    best: list[tuple[int, int, int, int]] = []
    h, w = sub.shape
    for r0 in range(h):
        for c0 in range(w):
            if sub[r0, c0] == 0:
                continue
            for r1 in range(r0, h):
                for c1 in range(c0, w):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area < best_area:
                        continue
                    if _rect_sum(integral, r0, c0, r1, c1) != area:
                        continue
                    box = (int(rmin + r0), int(cmin + c0), int(rmin + r1), int(cmin + c1))
                    if area > best_area:
                        best_area = area
                        best = [box]
                    else:
                        best.append(box)

    assert best_area > 0, "expected a solid rectangle"
    assert len(best) == 1, "moving body must be the unique largest solid rectangle"
    return best[0]


def _infer_stem(mask: np.ndarray, body_box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    r0, c0, r1, c1 = body_box
    body_mask = np.zeros_like(mask, dtype=bool)
    body_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    stem_mask = mask & ~body_mask
    rows, cols = np.where(stem_mask)
    assert rows.size > 0, "expected a stem outside the rectangle body"

    unique_rows = sorted(set(map(int, rows)))
    unique_cols = sorted(set(map(int, cols)))
    if len(unique_rows) == 1:
        row = unique_rows[0]
        cs = sorted(map(int, cols))
        assert cs == list(range(cs[0], cs[-1] + 1)), "horizontal stem must be contiguous"
        assert r0 <= row <= r1, "horizontal stem must attach to the body side"
        if cs[-1] < c0:
            assert cs[-1] == c0 - 1
            return 0, -1, cs[-1] - cs[0] + 1
        assert cs[0] == c1 + 1
        return 0, 1, cs[-1] - cs[0] + 1

    assert len(unique_cols) == 1, "stem must be a straight one-cell-wide segment"
    col = unique_cols[0]
    rs = sorted(map(int, rows))
    assert rs == list(range(rs[0], rs[-1] + 1)), "vertical stem must be contiguous"
    assert c0 <= col <= c1, "vertical stem must attach to the body side"
    if rs[-1] < r0:
        assert rs[-1] == r0 - 1
        return -1, 0, rs[-1] - rs[0] + 1
    assert rs[0] == r1 + 1
    return 1, 0, rs[-1] - rs[0] + 1


def _shift_box(box: tuple[int, int, int, int], dr: int, dc: int, steps: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 + dr * steps, c0 + dc * steps, r1 + dr * steps, c1 + dc * steps


def _box_in_bounds(shape: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = box
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _box_clear(occupied: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return not bool(np.any(occupied[r0 : r1 + 1, c0 : c1 + 1]))


def _slide_box(
    shape: tuple[int, int],
    occupied: np.ndarray,
    body_box: tuple[int, int, int, int],
    dr: int,
    dc: int,
) -> tuple[tuple[int, int, int, int], str]:
    steps = 0
    while True:
        candidate = _shift_box(body_box, dr, dc, steps + 1)
        if not _box_in_bounds(shape, candidate):
            return _shift_box(body_box, dr, dc, steps), "wall"
        if not _box_clear(occupied, candidate):
            return _shift_box(body_box, dr, dc, steps), "body"
        steps += 1


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _extract_objects(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    colors = colors_present(grid, exclude={bg})
    assert 3 <= len(colors) <= 4, "expected 3-4 movable colors"

    objects: list[dict] = []
    directions = set()
    for color in colors:
        mask = grid == int(color)
        assert len(components_mask(mask, connectivity=4)) == 1, "each color must form one connected component"
        body_box = _largest_solid_rectangle(mask)
        dr, dc, stem_len = _infer_stem(mask, body_box)
        r0, c0, r1, c1 = body_box
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        objects.append(
            {
                "color": int(color),
                "body_box": body_box,
                "dr": int(dr),
                "dc": int(dc),
                "stem_len": int(stem_len),
                "area": int(area),
            }
        )
        directions.add((int(dr), int(dc)))

    assert len({obj["area"] for obj in objects}) == len(objects), "body areas must be distinct"
    assert len(directions) == 1, "all stems must point toward the same wall"
    return objects


def _solve_objects(shape: tuple[int, int], objects: list[dict]) -> tuple[np.ndarray, list[dict]]:
    placed = np.zeros(shape, dtype=bool)
    out = np.zeros(shape, dtype=int)
    results: list[dict] = []
    for obj in sorted(objects, key=lambda item: item["area"], reverse=True):
        final_box, stop_reason = _slide_box(shape, placed, obj["body_box"], obj["dr"], obj["dc"])
        r0, c0, r1, c1 = final_box
        out[r0 : r1 + 1, c0 : c1 + 1] = int(obj["color"])
        placed[r0 : r1 + 1, c0 : c1 + 1] = True
        result = dict(obj)
        result["final_box"] = final_box
        result["stop_reason"] = stop_reason
        results.append(result)
    return out, results


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each stemmed rectangle toward the common wall in descending area order so earlier arrivals become later obstacles.

    Input structure:
    - A black background contains 3-4 differently colored 4-connected composite objects.
    - Each composite is one solid rectangle body plus one straight one-cell-wide stem pointing toward the same border.
    - Rectangle bodies have distinct areas, while stems only indicate the slide direction and are erased in the output.

    Rule:
    - Identify every composite object, isolate its rectangle body, and compute that body's area.
    - Sort the composites from largest body area to smallest.
    - Process them in that order: erase the whole composite, slide only the rectangle along the stem direction until the next step would leave the grid or overlap any previously placed rectangle, then draw the rectangle at that final position without the stem.

    Key insight:
    - Area ranking is the scheduler for the motion, and the earlier rectangles become dynamic blocking obstacles that change where later rectangles can stop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    output, _results = _solve_objects(g.shape, _extract_objects(g))
    return output


def _direction_spec(direction: str) -> tuple[bool, int]:
    if direction == "right":
        return True, 1
    if direction == "left":
        return True, -1
    if direction == "down":
        return False, 1
    assert direction == "up"
    return False, -1


def _max_stem_length(h: int, w: int, axis_is_col: bool) -> int:
    area = h * w
    axis_len = w if axis_is_col else h
    return min(8, area - axis_len - 1)


def _center_choices(length: int) -> list[int]:
    if length % 2 == 1:
        return [length // 2]
    return [length // 2 - 1, length // 2]


def _make_patch(axis_is_col: bool, sign: int, h: int, w: int, stem_len: int, color: int) -> np.ndarray:
    if axis_is_col:
        patch = np.zeros((h, w + stem_len), dtype=int)
        if sign > 0:
            patch[:, :w] = color
            attach = random.choice(_center_choices(h))
            patch[attach, w:] = color
        else:
            patch[:, stem_len:] = color
            attach = random.choice(_center_choices(h))
            patch[attach, :stem_len] = color
        return patch

    patch = np.zeros((h + stem_len, w), dtype=int)
    if sign > 0:
        patch[:h, :] = color
        attach = random.choice(_center_choices(w))
        patch[h:, attach] = color
    else:
        patch[stem_len:, :] = color
        attach = random.choice(_center_choices(w))
        patch[:stem_len, attach] = color
    return patch


def _choose_object_specs(direction: str, count: int) -> list[dict]:
    axis_is_col, _sign = _direction_spec(direction)
    options = HORIZONTAL_BODY_OPTIONS if axis_is_col else VERTICAL_BODY_OPTIONS
    colors = random.sample(GEN_COLORS, count)
    areas = sorted(random.sample(GEN_AREAS, count), reverse=True)
    specs = []
    for color, area in zip(colors, areas):
        h, w = random.choice(options[area])
        stem_cap = _max_stem_length(h, w, axis_is_col)
        high = min(8, stem_cap)
        assert high >= 3
        if random.random() < 0.2:
            stem_len = high
        else:
            stem_len = random.randint(3, high)
        specs.append({"color": int(color), "area": int(area), "h": int(h), "w": int(w), "stem_len": int(stem_len)})
    return specs


def _pick_overlapping_cross_starts(total: int, lengths: list[int]) -> list[int] | None:
    shared = random.randint(GEN_MARGIN, total - GEN_MARGIN - 1)
    starts: list[int] = []
    for length in lengths:
        lo = max(GEN_MARGIN, shared - length + 1)
        hi = min(shared, total - GEN_MARGIN - length)
        if lo > hi:
            return None
        starts.append(random.randint(lo, hi))
    if len(set(starts)) == 1 and len(lengths) > 1:
        for _ in range(20):
            starts = []
            for length in lengths:
                lo = max(GEN_MARGIN, shared - length + 1)
                hi = min(shared, total - GEN_MARGIN - length)
                starts.append(random.randint(lo, hi))
            if len(set(starts)) > 1:
                break
    return starts


def _pick_disjoint_start(total: int, length: int, reserved: list[tuple[int, int]]) -> int | None:
    candidates = []
    for start in range(GEN_MARGIN, total - GEN_MARGIN - length + 1):
        end = start + length - 1
        if all(end + 1 < r0 or r1 + 1 < start for r0, r1 in reserved):
            candidates.append(start)
    if not candidates:
        return None
    return random.choice(candidates)


def _assign_cross_positions(specs: list[dict], axis_is_col: bool, collision_count: int, total: int) -> bool:
    collision_lengths = [(spec["h"] if axis_is_col else spec["w"]) for spec in specs[:collision_count]]
    starts = _pick_overlapping_cross_starts(total, collision_lengths)
    if starts is None:
        return False

    reserved = []
    for spec, start, length in zip(specs[:collision_count], starts, collision_lengths):
        spec["cross0"] = int(start)
        reserved.append((start, start + length - 1))

    for spec in specs[collision_count:]:
        length = spec["h"] if axis_is_col else spec["w"]
        start = _pick_disjoint_start(total, length, reserved)
        if start is None:
            return False
        spec["cross0"] = int(start)
        reserved.append((start, start + length - 1))
    return True


def _assign_collision_axis_positions(specs: list[dict], axis_is_col: bool, sign: int, total: int, collision_count: int) -> bool:
    gaps = [random.randint(1, 3) for _ in range(collision_count - 1)]

    if sign > 0:
        offset_last = 0
        for gap, spec in zip(gaps, specs[1:collision_count]):
            offset_last += gap + spec["stem_len"] + (spec["w"] if axis_is_col else spec["h"])
        leader = specs[0]
        leader_len = leader["w"] if axis_is_col else leader["h"]
        min_leader = GEN_MARGIN + offset_last
        max_leader = total - GEN_MARGIN - (leader_len + leader["stem_len"])
        if min_leader > max_leader:
            return False
        specs[0]["axis0"] = random.randint(min_leader, max_leader)
        for i in range(1, collision_count):
            prev = specs[i - 1]
            current = specs[i]
            current_len = current["w"] if axis_is_col else current["h"]
            current["axis0"] = prev["axis0"] - gaps[i - 1] - current["stem_len"] - current_len
        return True

    offsets = [0]
    for i in range(1, collision_count):
        prev = specs[i - 1]
        current = specs[i]
        prev_len = prev["w"] if axis_is_col else prev["h"]
        offsets.append(offsets[-1] + prev_len + gaps[i - 1] + current["stem_len"])

    leader = specs[0]
    leader_min = GEN_MARGIN + leader["stem_len"]
    last = specs[collision_count - 1]
    last_len = last["w"] if axis_is_col else last["h"]
    leader_max = total - GEN_MARGIN - (offsets[-1] + last_len)
    if leader_min > leader_max:
        return False
    specs[0]["axis0"] = random.randint(leader_min, leader_max)
    for i in range(1, collision_count):
        specs[i]["axis0"] = specs[0]["axis0"] + offsets[i]
    return True


def _body_box_from_spec(spec: dict, axis_is_col: bool) -> tuple[int, int, int, int]:
    if axis_is_col:
        r0 = spec["cross0"]
        c0 = spec["axis0"]
        return r0, c0, r0 + spec["h"] - 1, c0 + spec["w"] - 1
    r0 = spec["axis0"]
    c0 = spec["cross0"]
    return r0, c0, r0 + spec["h"] - 1, c0 + spec["w"] - 1


def _patch_top_left(spec: dict, axis_is_col: bool, sign: int) -> tuple[int, int]:
    if axis_is_col:
        return spec["cross0"], spec["axis0"] if sign > 0 else spec["axis0"] - spec["stem_len"]
    return spec["axis0"] if sign > 0 else spec["axis0"] - spec["stem_len"], spec["cross0"]


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, top_left: tuple[int, int]) -> bool:
    r0, c0 = top_left
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False
    region = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != 0
    return not bool(np.any(region[mask] != 0))


def _assign_independent_axis_positions(grid: np.ndarray, specs: list[dict], axis_is_col: bool, sign: int, start: int) -> bool:
    axis_total = grid.shape[1] if axis_is_col else grid.shape[0]
    for spec in specs[start:]:
        axis_len = spec["w"] if axis_is_col else spec["h"]
        if sign > 0:
            lo = GEN_MARGIN
            hi = axis_total - GEN_MARGIN - (axis_len + spec["stem_len"])
        else:
            lo = GEN_MARGIN + spec["stem_len"]
            hi = axis_total - GEN_MARGIN - axis_len
        if lo > hi:
            return False
        candidates = list(range(lo, hi + 1))
        random.shuffle(candidates)
        patch = _make_patch(axis_is_col, sign, spec["h"], spec["w"], spec["stem_len"], spec["color"])
        placed = False
        for axis0 in candidates:
            spec["axis0"] = int(axis0)
            top_left = _patch_top_left(spec, axis_is_col, sign)
            if _can_place_patch(grid, patch, top_left):
                r0, c0 = top_left
                mask = patch != 0
                grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]
                placed = True
                break
        if not placed:
            return False
    return True


def _render_collision_group(grid: np.ndarray, specs: list[dict], axis_is_col: bool, sign: int, collision_count: int) -> bool:
    for spec in specs[:collision_count]:
        patch = _make_patch(axis_is_col, sign, spec["h"], spec["w"], spec["stem_len"], spec["color"])
        top_left = _patch_top_left(spec, axis_is_col, sign)
        if not _can_place_patch(grid, patch, top_left):
            return False
        r0, c0 = top_left
        mask = patch != 0
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]
    return True


def _projection_interval(spec: dict, axis_is_col: bool) -> tuple[int, int]:
    if axis_is_col:
        return spec["cross0"], spec["cross0"] + spec["h"] - 1
    return spec["cross0"], spec["cross0"] + spec["w"] - 1


def _attempt_generate() -> np.ndarray | None:
    direction = random.choice(tuple(DIR_TO_VEC))
    axis_is_col, sign = _direction_spec(direction)
    count = random.choice(GEN_COUNTS)
    specs = _choose_object_specs(direction, count)
    collision_count = random.randint(2, min(3, count))

    grid_rows = random.randint(*GEN_GRID_ROWS)
    grid_cols = random.randint(*GEN_GRID_COLS)
    grid = np.zeros((grid_rows, grid_cols), dtype=int)
    cross_total = grid_rows if axis_is_col else grid_cols
    axis_total = grid_cols if axis_is_col else grid_rows

    if not _assign_cross_positions(specs, axis_is_col, collision_count, cross_total):
        return None
    if not _assign_collision_axis_positions(specs, axis_is_col, sign, axis_total, collision_count):
        return None
    if not _render_collision_group(grid, specs, axis_is_col, sign, collision_count):
        return None
    if not _assign_independent_axis_positions(grid, specs, axis_is_col, sign, collision_count):
        return None

    for spec in specs:
        spec["body_box"] = _body_box_from_spec(spec, axis_is_col)
        dr, dc = DIR_TO_VEC[direction]
        spec["dr"] = dr
        spec["dc"] = dc

    intervals = [_projection_interval(spec, axis_is_col) for spec in specs]
    overlaps = [
        _intervals_overlap(a0, a1, b0, b1)
        for i, (a0, a1) in enumerate(intervals)
        for b0, b1 in intervals[i + 1 :]
    ]
    if not any(overlaps):
        return None

    try:
        parsed = _extract_objects(grid)
        output, results = _solve_objects(grid.shape, parsed)
    except Exception:
        return None

    if not any(result["stop_reason"] == "body" for result in results):
        return None
    if not any(result["stop_reason"] == "wall" for result in results):
        return None
    if np.array_equal(grid, output):
        return None
    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-priority ARC input")
