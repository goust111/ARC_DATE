from __future__ import annotations

import random
from typing import Iterable

import numpy as np


PALETTE = [1, 2, 3, 4]
SIZE_OPTIONS = [
    (2, 3, 6),
    (2, 4, 8),
    (3, 3, 9),
    (2, 5, 10),
    (3, 4, 12),
    (3, 5, 15),
    (4, 4, 16),
    (3, 6, 18),
    (4, 5, 20),
    (4, 6, 24),
    (5, 5, 25),
]
BIG_OPTIONS = [(4, 5, 20), (4, 6, 24), (5, 5, 25)]
MAX_GENERATION_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _corners(box: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = box
    return ((r0, c0), (r0, c1), (r1, c0), (r1, c1))


def _perimeter_coords(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _interior_coords(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}


def _incremental_fill_coords(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    height = r1 - r0 - 1
    width = c1 - c0 - 1
    cells: set[tuple[int, int]] = set()
    for dr in range(height):
        fill_width = min(dr + 1, width)
        rr = r0 + 1 + dr
        for dc in range(fill_width):
            cells.add((rr, c0 + 1 + dc))
    return cells


def _interior_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return int((r1 - r0 - 1) * (c1 - c0 - 1))


def _rect_overlap_area(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    r0 = max(ar0, br0)
    c0 = max(ac0, bc0)
    r1 = min(ar1, br1)
    c1 = min(ac1, bc1)
    if r0 > r1 or c0 > c1:
        return 0
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _frame_signature(frame: dict) -> tuple[int, tuple[int, int, int, int]]:
    return int(frame["color"]), tuple(map(int, frame["bbox"]))


def _render_frames(
    shape: tuple[int, int],
    frames: Iterable[dict],
    *,
    include_fill: bool,
    full_fill: bool = False,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    border_area = np.full(shape, -1, dtype=int)
    fill_area = np.full(shape, -1, dtype=int)

    for frame in frames:
        color = int(frame["color"])
        area = int(frame["area"])

        for r, c in frame["perimeter"]:
            if area > border_area[r, c]:
                border_area[r, c] = area
                out[r, c] = color

        if not include_fill:
            continue

        fill_cells = frame["interior"] if full_fill else frame["fill"]
        for r, c in fill_cells:
            if area > fill_area[r, c]:
                fill_area[r, c] = area
                if border_area[r, c] < 0:
                    out[r, c] = color

    for r, c in np.argwhere(border_area >= 0):
        out[int(r), int(c)] = out[int(r), int(c)]
    return out


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for color in sorted(c for c in np.unique(g) if int(c) != 0):
        mask = g == int(color)
        box = _bbox(mask)
        assert box is not None, "every non-black color must appear"
        r0, c0, r1, c1 = box
        assert r1 - r0 >= 3 and c1 - c0 >= 4, "frame must have interior height >=2 and width >=3"

        perimeter = _perimeter_coords(box)
        coords = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
        assert coords.issubset(perimeter), "visible frame pixels must lie on their rectangle perimeter"
        for corner in _corners(box):
            assert g[corner] == int(color), "frame corners must stay visible"

        interior = _interior_coords(box)
        fill = _incremental_fill_coords(box)
        frames.append(
            {
                "color": int(color),
                "bbox": box,
                "area": _interior_area(box),
                "perimeter": perimeter,
                "interior": interior,
                "fill": fill,
            }
        )

    areas = [int(frame["area"]) for frame in frames]
    assert len(set(areas)) == len(areas), "interior areas must be unique"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular frame interior with a row-by-row growing wedge, resolving overlaps by larger interior area.

    Input structure:
    - The grid has a black background and 3 to 4 distinct non-black frame colors.
    - Each non-black color belongs to one one-cell-thick axis-aligned rectangular frame.
    - Frames may overlap in interior or border cells, but each frame still exposes its four corners.

    Rule:
    - Recover every rectangular frame from the bounding box of its visible color pixels.
    - For each frame, compute its interior area and its interior rectangle.
    - Inside that rectangle, fill row `k` from the interior top with `min(k, width)` cells starting at the interior left, using the frame color.
    - If multiple fills claim one interior cell, keep the color of the frame with the larger interior area.
    - Border cells stay frame-colored, and if multiple borders coincide, the larger-area frame border stays on top.

    Key insight:
    - The frame bbox fixes both the bounded fill coordinate system and the area-based depth order, so overlap resolution depends on geometry, not on color ids.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    frames = _find_frames(g)
    return _render_frames(g.shape, frames, include_fill=True)


def _make_frame(color: int, box: tuple[int, int, int, int]) -> dict:
    return {
        "color": int(color),
        "bbox": tuple(map(int, box)),
        "area": _interior_area(box),
        "perimeter": _perimeter_coords(box),
        "interior": _interior_coords(box),
        "fill": _incremental_fill_coords(box),
    }


def _flip_box_ud(box: tuple[int, int, int, int], height: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return height - 1 - r1, c0, height - 1 - r0, c1


def _flip_box_lr(box: tuple[int, int, int, int], width: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0, width - 1 - c1, r1, width - 1 - c0


def _transform_frames(shape: tuple[int, int], frames: list[dict]) -> list[dict]:
    height, width = shape
    boxes = [frame["bbox"] for frame in frames]
    if random.random() < 0.5:
        boxes = [_flip_box_ud(box, height) for box in boxes]
    if random.random() < 0.5:
        boxes = [_flip_box_lr(box, width) for box in boxes]
    return [_make_frame(frame["color"], box) for frame, box in zip(frames, boxes)]


def _point_on_perimeter(box: tuple[int, int, int, int], pt: tuple[int, int]) -> bool:
    return pt in _perimeter_coords(box)


def _corners_clear(candidate: dict, frames: list[dict]) -> bool:
    cand_corners = _corners(candidate["bbox"])
    for frame in frames:
        other_perimeter = frame["perimeter"]
        if any(corner in other_perimeter for corner in cand_corners):
            return False
        if any(corner in candidate["perimeter"] for corner in _corners(frame["bbox"])):
            return False
    return True


def _place_nested_pair(
    shape: tuple[int, int], colors: list[int]
) -> tuple[list[dict], set[int]] | None:
    height, width = shape
    big_h, big_w, big_area = random.choice(BIG_OPTIONS)
    small_choices = [
        option
        for option in SIZE_OPTIONS
        if option[2] < big_area and option[0] + 2 <= big_h and option[1] + 2 <= big_w
    ]
    if not small_choices:
        return None

    small_h, small_w, _ = random.choice(small_choices)
    big_outer_h = big_h + 2
    big_outer_w = big_w + 2
    if big_outer_h > height or big_outer_w > width:
        return None

    big_r0 = random.randint(1, height - big_outer_h - 1)
    big_c0 = random.randint(1, width - big_outer_w - 1)
    big_box = (big_r0, big_c0, big_r0 + big_outer_h - 1, big_c0 + big_outer_w - 1)

    small_outer_h = small_h + 2
    small_outer_w = small_w + 2
    max_r_offset = big_h - small_outer_h
    max_c_offset = big_w - small_outer_w
    small_r0 = big_r0 + 1 + random.randint(0, max(0, min(1, max_r_offset)))
    small_c0 = big_c0 + 1 + random.randint(0, max(0, min(1, max_c_offset)))
    small_box = (
        small_r0,
        small_c0,
        small_r0 + small_outer_h - 1,
        small_c0 + small_outer_w - 1,
    )

    frames = [_make_frame(colors[0], big_box), _make_frame(colors[1], small_box)]
    if not _corners_clear(frames[0], [frames[1]]) or not _corners_clear(frames[1], [frames[0]]):
        return None
    return frames, {big_area, _interior_area(small_box)}


def _random_box(shape: tuple[int, int], inner_h: int, inner_w: int) -> tuple[int, int, int, int]:
    height, width = shape
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    r0 = random.randint(0, height - outer_h)
    c0 = random.randint(0, width - outer_w)
    return r0, c0, r0 + outer_h - 1, c0 + outer_w - 1


def _place_separate(
    shape: tuple[int, int], color: int, size: tuple[int, int, int], frames: list[dict]
) -> dict | None:
    inner_h, inner_w, _ = size
    for _ in range(200):
        candidate = _make_frame(color, _random_box(shape, inner_h, inner_w))
        if not _corners_clear(candidate, frames):
            continue
        if any(candidate["perimeter"] & frame["perimeter"] for frame in frames):
            continue
        if any(candidate["interior"] & frame["interior"] for frame in frames):
            continue
        if any(candidate["perimeter"] & frame["interior"] for frame in frames):
            continue
        if any(candidate["interior"] & frame["perimeter"] for frame in frames):
            continue
        return candidate
    return None


def _place_crossing(
    shape: tuple[int, int], color: int, size: tuple[int, int, int], frames: list[dict]
) -> dict | None:
    inner_h, inner_w, _ = size
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    larger_targets = [frame for frame in frames if frame["area"] > size[2]]
    random.shuffle(larger_targets)

    for target in larger_targets:
        tr0, tc0, tr1, tc1 = target["bbox"]
        target_rows = [tr0, tr1]
        random.shuffle(target_rows)
        target_cols = list(range(tc0 + 1, tc1))
        random.shuffle(target_cols)
        offsets = list(range(1, outer_h - 1))
        random.shuffle(offsets)
        for row in target_rows:
            for col in target_cols:
                for offset in offsets:
                    r0 = row - offset
                    c0 = col
                    box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
                    if box[0] < 0 or box[1] < 0 or box[2] >= shape[0] or box[3] >= shape[1]:
                        continue
                    candidate = _make_frame(color, box)
                    if not _corners_clear(candidate, frames):
                        continue
                    if candidate["perimeter"] & target["perimeter"] != {(row, col)}:
                        continue
                    others = [frame for frame in frames if frame is not target]
                    if any(candidate["perimeter"] & frame["perimeter"] for frame in others):
                        continue
                    if any(candidate["interior"] & frame["interior"] for frame in others):
                        continue
                    if any(candidate["perimeter"] & frame["interior"] for frame in others):
                        continue
                    if any(candidate["interior"] & frame["perimeter"] for frame in others):
                        continue
                    return candidate
    return None


def _border_overlap_count(frames: list[dict]) -> int:
    count = 0
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            count += len(frames[i]["perimeter"] & frames[j]["perimeter"])
    return count


def _has_fill_priority_overlap(frames: list[dict]) -> bool:
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            if frames[i]["fill"] & frames[j]["fill"]:
                return True
    return False


def _has_border_over_fill_need(frames: list[dict]) -> bool:
    for i in range(len(frames)):
        for j in range(len(frames)):
            if i == j or frames[i]["area"] >= frames[j]["area"]:
                continue
            if frames[i]["perimeter"] & frames[j]["fill"]:
                return True
    return False


def _build_candidate_frames(shape: tuple[int, int]) -> tuple[list[dict], bool] | None:
    n_frames = random.choice([3, 4])
    colors = random.sample(PALETTE, n_frames)

    base = _place_nested_pair(shape, colors[:2])
    if base is None:
        return None
    frames, used_areas = base

    remaining_sizes = [option for option in SIZE_OPTIONS if option[2] not in used_areas]
    random.shuffle(remaining_sizes)
    need_cross = random.random() < 0.6
    extra_modes = []
    if n_frames == 3:
        extra_modes = ["cross" if need_cross else "separate"]
    else:
        extra_modes = ["separate", "cross" if need_cross else "separate"]

    for color, mode in zip(colors[2:], extra_modes):
        pool = remaining_sizes[:]
        if mode == "cross":
            pool.sort(key=lambda item: item[2])
        else:
            random.shuffle(pool)
        placed = None
        for size in pool:
            candidate = (
                _place_crossing(shape, color, size, frames)
                if mode == "cross"
                else _place_separate(shape, color, size, frames)
            )
            if candidate is None:
                continue
            placed = candidate
            remaining_sizes.remove(size)
            frames.append(candidate)
            break
        if placed is None:
            return None

    frames = _transform_frames(shape, frames)
    return frames, need_cross


def _validate_specs(shape: tuple[int, int], frames: list[dict], need_cross: bool) -> bool:
    if len(frames) not in {3, 4}:
        return False

    areas = [frame["area"] for frame in frames]
    if len(set(areas)) != len(areas):
        return False
    if min(areas) < 6 or max(areas) > 25:
        return False

    overlap_pairs = 0
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            a = frames[i]["bbox"]
            b = frames[j]["bbox"]
            interior_a = (a[0] + 1, a[1] + 1, a[2] - 1, a[3] - 1)
            interior_b = (b[0] + 1, b[1] + 1, b[2] - 1, b[3] - 1)
            if _rect_overlap_area(interior_a, interior_b) >= 2:
                overlap_pairs += 1
    if overlap_pairs == 0:
        return False

    if not _has_fill_priority_overlap(frames):
        return False
    if not _has_border_over_fill_need(frames):
        return False

    if need_cross and _border_overlap_count(frames) == 0:
        return False

    inp = _render_frames(shape, frames, include_fill=False)
    out = _render_frames(shape, frames, include_fill=True)
    full_fill = _render_frames(shape, frames, include_fill=True, full_fill=True)
    if np.array_equal(out, full_fill):
        return False

    try:
        recovered = _find_frames(inp)
    except AssertionError:
        return False

    want = sorted(_frame_signature(frame) for frame in frames)
    got = sorted(_frame_signature(frame) for frame in recovered)
    if want != got:
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return np.array_equal(solved, out)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        shape = (random.randint(16, 22), random.randint(16, 22))
        built = _build_candidate_frames(shape)
        if built is None:
            continue
        frames, need_cross = built
        if not _validate_specs(shape, frames, need_cross):
            continue
        return _render_frames(shape, frames, include_fill=False)
    raise RuntimeError("failed to generate a valid overlapping-frame input")
