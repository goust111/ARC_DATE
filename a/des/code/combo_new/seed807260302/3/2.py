from __future__ import annotations

import random

import numpy as np


BLACK = 0
COLORS = list(range(1, 10))
GEN_RETRIES = 500


def _group_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int], count: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    if count % 2 == 0:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    else:
        mask[r0, c0 : c1 + 1] = True
        mask[r1, c0 : c1 + 1] = True
        mask[r0 : r1 + 1, c0] = True
        mask[r0 : r1 + 1, c1] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a count-prioritized bounding rectangle for each color group.

    Input structure:
    - Black background with several non-black colors.
    - Each non-black color appears as a loose cluster of 2 to 6 pixels.
    - Pixels of the same color form one logical group even when not adjacent.

    Rule:
    - Group all non-black pixels by color.
    - For each color, find the smallest axis-aligned bounding box containing its pixels.
    - Draw a one-pixel-thick rectangle border in that color on the bounding box.
    - If the group has an even number of pixels, fill the rectangle interior with that color.
    - If rectangles overlap, cells from the group with more input pixels cover cells from groups with fewer pixels.

    Key insight: The visible scattered pixels define bounding boxes, and the hidden count parity decides whether each rectangle is solid or hollow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.full(g.shape, BLACK, dtype=int)
    priority = np.full(g.shape, -1, dtype=int)

    for color in np.unique(g):
        color = int(color)
        if color == BLACK:
            continue

        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue

        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        count = int(coords.shape[0])
        mask = _group_mask(g.shape, (int(r0), int(c0), int(r1), int(c1)), count)

        write = mask & (count > priority)
        out[write] = color
        priority[write] = count

    return out


def _inter_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    h = min(ar1, br1) - max(ar0, br0) + 1
    w = min(ac1, bc1) - max(ac0, bc0) + 1
    return max(0, h) * max(0, w)


def _box_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _expanded_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _rendered_overlap(
    shape: tuple[int, int],
    box_a: tuple[int, int, int, int],
    count_a: int,
    box_b: tuple[int, int, int, int],
    count_b: int,
) -> bool:
    return bool(
        np.any(
            _group_mask(shape, box_a, count_a)
            & _group_mask(shape, box_b, count_b)
        )
    )


def _random_box(
    rng: random.Random,
    height: int,
    width: int,
    min_side: int = 4,
    max_side: int = 7,
) -> tuple[int, int, int, int]:
    bh = rng.randint(min_side, min(max_side, height - 2))
    bw = rng.randint(min_side, min(max_side, width - 2))
    r0 = rng.randint(1, height - bh - 1)
    c0 = rng.randint(1, width - bw - 1)
    return r0, c0, r0 + bh - 1, c0 + bw - 1


def _place_overlapping_pair(
    rng: random.Random,
    height: int,
    width: int,
    count_a: int,
    count_b: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    shape = (height, width)
    for _ in range(200):
        box_a = _random_box(rng, height, width, 5, 8)
        ar0, ac0, ar1, ac1 = box_a
        ah = ar1 - ar0 + 1
        aw = ac1 - ac0 + 1

        bh = rng.randint(5, min(8, height - 2))
        bw = rng.randint(5, min(8, width - 2))

        row_shifts = [s for s in range(-bh + 2, ah - 1) if s != 0]
        col_shifts = [s for s in range(-bw + 2, aw - 1) if s != 0]
        rng.shuffle(row_shifts)
        rng.shuffle(col_shifts)

        for sr in row_shifts:
            for sc in col_shifts:
                br0 = ar0 + sr
                bc0 = ac0 + sc
                box_b = (br0, bc0, br0 + bh - 1, bc0 + bw - 1)
                if br0 < 1 or bc0 < 1 or box_b[2] >= height - 1 or box_b[3] >= width - 1:
                    continue
                inter = _inter_area(box_a, box_b)
                if inter < 4:
                    continue
                if inter > int(0.65 * min(_box_area(box_a), _box_area(box_b))):
                    continue
                if not _rendered_overlap(shape, box_a, count_a, box_b, count_b):
                    continue
                return box_a, box_b
    return None


def _sample_points(
    rng: random.Random,
    box: tuple[int, int, int, int],
    count: int,
    occupied: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    r0, c0, r1, c1 = box
    diagonals = [
        [(r0, c0), (r1, c1)],
        [(r0, c1), (r1, c0)],
    ]
    rng.shuffle(diagonals)

    for base in diagonals:
        if any(p in occupied for p in base):
            continue
        points = set(base)
        candidates = [
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if (r, c) not in occupied and (r, c) not in points
        ]
        rng.shuffle(candidates)
        if len(candidates) < count - len(points):
            continue
        points.update(candidates[: count - len(points)])
        return points
    return None


def _choose_counts(rng: random.Random, n_groups: int) -> list[int]:
    even_counts = [2, 4, 6]
    odd_counts = [3, 5]
    parities = [0, 1]
    for _ in range(n_groups - 2):
        parities.append(rng.randint(0, 1))
    rng.shuffle(parities)
    return [
        rng.choice(even_counts if parity == 0 else odd_counts)
        for parity in parities
    ]


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    height = rng.randint(15, 25)
    width = rng.randint(15, 25)
    n_groups = rng.randint(3, 5)

    counts = _choose_counts(rng, n_groups)
    pair = None
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            if counts[i] != counts[j] and counts[i] % 2 != counts[j] % 2:
                pair = (i, j)
                break
        if pair is not None:
            break
    if pair is None:
        return None

    boxes: list[tuple[int, int, int, int] | None] = [None] * n_groups
    box_pair = _place_overlapping_pair(rng, height, width, counts[pair[0]], counts[pair[1]])
    if box_pair is None:
        return None
    boxes[pair[0]], boxes[pair[1]] = box_pair

    for idx in range(n_groups):
        if boxes[idx] is not None:
            continue
        for _ in range(200):
            box = _random_box(rng, height, width, 4, 7)
            if all(
                existing is None or not _expanded_overlap(box, existing, gap=1)
                for existing in boxes
            ):
                boxes[idx] = box
                break
        if boxes[idx] is None:
            return None

    colors = rng.sample(COLORS, n_groups)
    grid = np.full((height, width), BLACK, dtype=int)
    occupied: set[tuple[int, int]] = set()

    order = list(range(n_groups))
    rng.shuffle(order)
    for idx in order:
        box = boxes[idx]
        assert box is not None
        points = _sample_points(rng, box, counts[idx], occupied)
        if points is None:
            return None
        for r, c in points:
            grid[r, c] = colors[idx]
        occupied.update(points)

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if np.array_equal(out, grid):
        return None
    if not any(c % 2 == 0 for c in counts) or not any(c % 2 == 1 for c in counts):
        return None
    if not _rendered_overlap(
        grid.shape,
        boxes[pair[0]],  # type: ignore[arg-type]
        counts[pair[0]],
        boxes[pair[1]],  # type: ignore[arg-type]
        counts[pair[1]],
    ):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
