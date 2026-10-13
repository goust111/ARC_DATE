import random
from typing import Iterable

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
PIXEL_COUNTS = (2, 3, 4, 5)
GEN_RETRIES = 500


def _colors_present(grid: np.ndarray) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) != BG]


def _bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _intersection_area(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _contains(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _touches_4_neighbor(cell: tuple[int, int], chosen: Iterable[tuple[int, int]]) -> bool:
    r, c = cell
    for rr, cc in chosen:
        if abs(r - rr) + abs(c - cc) == 1:
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered pixels by color, fill each color's bounding box, and stack
    the rectangles from lowest pixel-count to highest pixel-count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    groups: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in _colors_present(g):
        pts = np.argwhere(g == color)
        count = int(len(pts))
        assert 2 <= count <= 5
        groups.append((count, int(color), _bbox(pts)))

    counts = [count for count, _color, _box in groups]
    assert groups
    assert len(counts) == len(set(counts))

    out = np.full_like(g, BG)
    for count, color, (r0, c0, r1, c1) in sorted(groups, key=lambda item: item[0]):
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _sample_box_size(rng: random.Random, count: int) -> tuple[int, int]:
    choices: list[tuple[int, int]] = []
    for h in range(2, 7):
        for w in range(2, 7):
            if h * w <= count:
                continue
            if ((h * w) + 1) // 2 < count:
                continue
            choices.append((h, w))
    assert choices
    return rng.choice(choices)


def _random_box(
    rng: random.Random, height: int, width: int, box_h: int, box_w: int
) -> tuple[int, int, int, int]:
    r0 = rng.randint(0, height - box_h)
    c0 = rng.randint(0, width - box_w)
    return (r0, c0, r0 + box_h - 1, c0 + box_w - 1)


def _random_overlapping_box(
    rng: random.Random,
    height: int,
    width: int,
    box_h: int,
    box_w: int,
    anchor: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    min_r0 = max(0, anchor[0] - box_h + 1)
    max_r0 = min(height - box_h, anchor[2])
    min_c0 = max(0, anchor[1] - box_w + 1)
    max_c0 = min(width - box_w, anchor[3])
    if min_r0 > max_r0 or min_c0 > max_c0:
        return None
    r0 = rng.randint(min_r0, max_r0)
    c0 = rng.randint(min_c0, max_c0)
    return (r0, c0, r0 + box_h - 1, c0 + box_w - 1)


def _box_pair_count(boxes: list[tuple[int, int, int, int]]) -> int:
    total = 0
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            total += int(_intersection_area(a, b) > 0)
    return total


def _place_boxes(
    rng: random.Random, height: int, width: int, counts: list[int], need_overlap: bool
) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    sizes = [_sample_box_size(rng, count) for count in counts]

    for idx, (box_h, box_w) in enumerate(sizes):
        placed = None
        for _ in range(200):
            want_overlap = False
            anchor = None
            if idx > 0 and need_overlap:
                if idx == 1 or rng.random() < 0.6:
                    want_overlap = True
                    anchor = rng.choice(boxes)

            if want_overlap:
                candidate = _random_overlapping_box(
                    rng, height, width, box_h, box_w, anchor
                )
                if candidate is None:
                    continue
            else:
                candidate = _random_box(rng, height, width, box_h, box_w)

            if candidate in boxes:
                continue

            overlaps = [_intersection_area(candidate, other) for other in boxes]
            if not need_overlap and any(area > 0 for area in overlaps):
                continue

            if want_overlap and overlaps:
                anchor_area = _rect_area(anchor)
                inter = _intersection_area(candidate, anchor)
                if inter == 0:
                    continue
                if inter >= min(_rect_area(candidate), anchor_area):
                    continue
                if inter > (3 * min(_rect_area(candidate), anchor_area)) // 4:
                    continue

            if any(
                _contains(candidate, other) or _contains(other, candidate) for other in boxes
            ):
                continue

            placed = candidate
            break

        if placed is None:
            return None
        boxes.append(placed)

    overlap_pairs = _box_pair_count(boxes)
    if need_overlap and overlap_pairs == 0:
        return None
    if not need_overlap and overlap_pairs != 0:
        return None
    return boxes


def _search_extra_cells(
    candidates: list[tuple[int, int]],
    need: int,
    start: int,
    chosen: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    if len(chosen) == need:
        return list(chosen)
    if len(candidates) - start < need - len(chosen):
        return None

    for idx in range(start, len(candidates)):
        cell = candidates[idx]
        if _touches_4_neighbor(cell, chosen):
            continue
        chosen.append(cell)
        result = _search_extra_cells(candidates, need, idx + 1, chosen)
        if result is not None:
            return result
        chosen.pop()
    return None


def _sample_pixels_for_box(
    rng: random.Random,
    box: tuple[int, int, int, int],
    count: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = box
    diagonals = [
        [(r0, c0), (r1, c1)],
        [(r0, c1), (r1, c0)],
    ]
    rng.shuffle(diagonals)

    all_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    for required in diagonals:
        if any(cell in occupied for cell in required):
            continue
        if required[0] == required[1]:
            continue
        if _touches_4_neighbor(required[0], required[1:]):
            continue

        extras = [cell for cell in all_cells if cell not in occupied and cell not in required]
        rng.shuffle(extras)
        result = _search_extra_cells(extras, count, 0, list(required))
        if result is not None:
            return result
    return None


def _validate_grid(
    grid: np.ndarray,
    expected_boxes: dict[int, tuple[int, int, int, int]],
    expected_counts: dict[int, int],
    need_overlap: bool,
) -> bool:
    colors = _colors_present(grid)
    if len(colors) != len(expected_boxes):
        return False

    actual_boxes: dict[int, tuple[int, int, int, int]] = {}
    actual_counts: dict[int, int] = {}
    for color in colors:
        pts = np.argwhere(grid == color)
        actual_counts[color] = int(len(pts))
        actual_boxes[color] = _bbox(pts)

        seen = set()
        for cell in map(tuple, pts):
            if _touches_4_neighbor(cell, seen):
                return False
            seen.add(cell)

    if actual_counts != expected_counts:
        return False
    if actual_boxes != expected_boxes:
        return False
    if len(set(actual_counts.values())) != len(colors):
        return False

    overlap_pairs = _box_pair_count(list(actual_boxes.values()))
    if need_overlap and overlap_pairs == 0:
        return False
    if not need_overlap and overlap_pairs != 0:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.any(out != grid):
        return False
    if np.count_nonzero(out) <= np.count_nonzero(grid):
        return False
    if np.count_nonzero(out) >= int(0.8 * grid.size):
        return False
    if any(not np.any(out == color) for color in colors):
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    color_count = rng.randint(2, 4)
    counts = rng.sample(list(PIXEL_COUNTS), k=color_count)
    colors = rng.sample(PALETTE, k=color_count)
    need_overlap = rng.random() < 0.75

    min_side = 9 if need_overlap else 10
    height = rng.randint(min_side, 18)
    width = rng.randint(min_side, 18)

    boxes = _place_boxes(rng, height, width, counts, need_overlap)
    if boxes is None:
        return None

    items = list(zip(colors, counts, boxes))
    rng.shuffle(items)

    grid = np.full((height, width), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()
    expected_boxes: dict[int, tuple[int, int, int, int]] = {}
    expected_counts: dict[int, int] = {}

    draw_order = sorted(items, key=lambda item: (_rect_area(item[2]), -item[1]))
    for color, count, box in draw_order:
        pixels = _sample_pixels_for_box(rng, box, count, occupied)
        if pixels is None:
            return None
        for r, c in pixels:
            grid[r, c] = int(color)
            occupied.add((r, c))
        expected_boxes[int(color)] = box
        expected_counts[int(color)] = int(count)

    if not _validate_grid(grid, expected_boxes, expected_counts, need_overlap):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate a valid input after {GEN_RETRIES} attempts")
