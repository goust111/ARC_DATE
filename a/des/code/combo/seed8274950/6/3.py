import random
from typing import Iterable

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bounds(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, h, w = box
    return r0, c0, r0 + h - 1, c0 + w - 1


def _boxes_overlap(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    gap: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = _bounds(box_a)
    br0, bc0, br1, bc1 = _bounds(box_b)
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _intersection_area(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> int:
    ar0, ac0, ar1, ac1 = _bounds(box_a)
    br0, bc0, br1, bc1 = _bounds(box_b)
    h = min(ar1, br1) - max(ar0, br0) + 1
    w = min(ac1, bc1) - max(ac0, bc0) + 1
    if h <= 0 or w <= 0:
        return 0
    return int(h * w)


def _draw_ring(
    grid: np.ndarray, color: int, top: int, left: int, bottom: int, right: int
) -> None:
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered single-pixel markers by color, take each color's bbox, and
    draw count-many concentric rectangle rings inside that bbox.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    out = np.full((h, w), bg, dtype=int)
    groups: list[tuple[int, tuple[int, int], int, int, int, int, int]] = []

    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        pts = np.argwhere(g == color)
        if pts.size == 0:
            continue
        count = int(len(pts))
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        anchor = min((int(r), int(c)) for r, c in pts)
        groups.append((count, anchor, color, r0, c0, r1, c1))

    groups.sort(key=lambda item: (item[0], item[1][0], item[1][1]))

    for count, _anchor, color, r0, c0, r1, c1 in groups:
        for layer in range(count):
            inset = 2 * layer
            top = r0 + inset
            left = c0 + inset
            bottom = r1 - inset
            right = c1 - inset
            if top > bottom or left > right:
                break
            _draw_ring(out, color, top, left, bottom, right)

    return out


BASE_SIZE = 30
MIN_GROUPS = 2
MAX_GROUPS = 3
COUNT_CHOICES = (2, 3, 4, 5)
MARGIN_MAX = 3
MAX_GEN_ATTEMPTS = 400


def _sample_box_size(count: int, rng: random.Random) -> tuple[int, int]:
    min_side = 4 * count - 3
    extra_h = rng.randint(0, 2)
    extra_w = rng.randint(0, 2)
    if rng.random() < 0.5:
        extra_h += rng.randint(0, 2)
    else:
        extra_w += rng.randint(0, 2)
    return min_side + extra_h, min_side + extra_w


def _random_box(
    height: int, width: int, rng: random.Random
) -> tuple[int, int, int, int]:
    r0 = rng.randint(0, BASE_SIZE - height)
    c0 = rng.randint(0, BASE_SIZE - width)
    return (r0, c0, height, width)


def _place_non_overlapping(
    sizes: list[tuple[int, int]], rng: random.Random
) -> list[tuple[int, int, int, int]] | None:
    order = sorted(
        range(len(sizes)),
        key=lambda idx: sizes[idx][0] * sizes[idx][1],
        reverse=True,
    )

    for gap in (1, 0):
        boxes: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
        ok = True
        for idx in order:
            height, width = sizes[idx]
            placed = False
            for _ in range(400):
                candidate = _random_box(height, width, rng)
                if all(
                    other is None or not _boxes_overlap(candidate, other, gap=gap)
                    for other in boxes
                ):
                    boxes[idx] = candidate
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return [box for box in boxes if box is not None]
    return None


def _place_overlapping(
    sizes: list[tuple[int, int]], rng: random.Random
) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
    order = sorted(
        range(len(sizes)),
        key=lambda idx: sizes[idx][0] * sizes[idx][1],
        reverse=True,
    )
    lead, overlap_idx = order[:2]
    lead_h, lead_w = sizes[lead]
    boxes[lead] = _random_box(lead_h, lead_w, rng)

    other_h, other_w = sizes[overlap_idx]
    for _ in range(600):
        candidate = _random_box(other_h, other_w, rng)
        if not _boxes_overlap(candidate, boxes[lead], gap=0):
            continue
        overlap = _intersection_area(candidate, boxes[lead])
        if overlap < 9:
            continue
        if candidate[:2] == boxes[lead][:2]:
            continue
        boxes[overlap_idx] = candidate
        break
    if boxes[overlap_idx] is None:
        return None

    for idx in order[2:]:
        height, width = sizes[idx]
        placed = False
        for _ in range(400):
            candidate = _random_box(height, width, rng)
            if all(
                other is None or not _boxes_overlap(candidate, other, gap=1)
                for other in boxes
            ):
                boxes[idx] = candidate
                placed = True
                break
        if not placed:
            return None

    return [box for box in boxes if box is not None]


def _sample_markers(
    box: tuple[int, int, int, int],
    count: int,
    occupied: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    r0, c0, h, w = box
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]

    if count == 2:
        patterns = [
            [corners[0], corners[3]],
            [corners[1], corners[2]],
        ]
        rng.shuffle(patterns)
        for pts in patterns:
            if all(pt not in occupied for pt in pts):
                return pts
        return None

    if count == 3:
        missing = [0, 1, 2, 3]
        rng.shuffle(missing)
        for skip in missing:
            pts = [corners[i] for i in range(4) if i != skip]
            if all(pt not in occupied for pt in pts):
                return pts
        return None

    for _ in range(200):
        pts = [
            (r0, rng.randint(c0, c1)),
            (r1, rng.randint(c0, c1)),
            (rng.randint(r0, r1), c0),
            (rng.randint(r0, r1), c1),
        ]
        if len(set(pts)) != 4:
            continue
        if any(pt in occupied for pt in pts):
            continue
        if count == 5:
            extras = [
                (r, c)
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)
                if (r, c) not in occupied and (r, c) not in pts
            ]
            if not extras:
                continue
            pts.append(rng.choice(extras))
        return pts

    return None


def _has_touching_markers(markers: Iterable[tuple[int, int]]) -> bool:
    pts = list(markers)
    for i, (r0, c0) in enumerate(pts):
        for r1, c1 in pts[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return True
    return False


def _crop_canvas(canvas: np.ndarray, markers: list[tuple[int, int]], rng: random.Random) -> np.ndarray:
    rows = [r for r, _ in markers]
    cols = [c for _, c in markers]
    r0 = min(rows)
    r1 = max(rows)
    c0 = min(cols)
    c1 = max(cols)
    top = rng.randint(0, min(MARGIN_MAX, r0))
    left = rng.randint(0, min(MARGIN_MAX, c0))
    bottom = rng.randint(0, min(MARGIN_MAX, BASE_SIZE - 1 - r1))
    right = rng.randint(0, min(MARGIN_MAX, BASE_SIZE - 1 - c1))
    return canvas[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randrange(10)
    num_groups = rng.randint(MIN_GROUPS, MAX_GROUPS)
    overlap_mode = rng.random() < 0.5

    counts = rng.sample(COUNT_CHOICES, num_groups)
    colors = rng.sample([color for color in range(10) if color != bg], num_groups)
    sizes = [_sample_box_size(count, rng) for count in counts]

    boxes = (
        _place_overlapping(sizes, rng)
        if overlap_mode
        else _place_non_overlapping(sizes, rng)
    )
    if boxes is None:
        return None

    occupied: set[tuple[int, int]] = set()
    group_markers: list[list[tuple[int, int]]] = []
    for box, count in zip(boxes, counts):
        pts = _sample_markers(box, count, occupied, rng)
        if pts is None:
            return None
        occupied.update(pts)
        group_markers.append(pts)

    flat_markers = [pt for pts in group_markers for pt in pts]
    if _has_touching_markers(flat_markers):
        return None

    overlap_pairs = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _boxes_overlap(boxes[i], boxes[j], gap=0):
                overlap_pairs += 1
    if overlap_mode and overlap_pairs == 0:
        return None
    if not overlap_mode and overlap_pairs != 0:
        return None

    canvas = np.full((BASE_SIZE, BASE_SIZE), bg, dtype=int)
    for color, pts in zip(colors, group_markers):
        for r, c in pts:
            canvas[r, c] = color

    cropped = _crop_canvas(canvas, flat_markers, rng).astype(int)
    out = main(cropped)

    if np.array_equal(out, cropped):
        return None

    unique_colors = [int(color) for color in np.unique(cropped) if int(color) != bg]
    if len(unique_colors) != num_groups:
        return None
    if sorted(int(np.count_nonzero(cropped == color)) for color in unique_colors) != sorted(counts):
        return None

    return cropped


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
