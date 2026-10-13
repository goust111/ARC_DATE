from __future__ import annotations

import random

import numpy as np


BLACK = 0
ARC_COLORS = tuple(range(1, 10))
GRID_SIZE_RANGE = (12, 26)
COLOR_COUNT_RANGE = (2, 5)
POINT_COUNT_RANGE = (2, 8)
BOX_SIDE_RANGE = (3, 14)
MIN_OVERLAP_AREA = 4
GEN_RETRIES = 300


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each color's point-defined bounding box as a layered solid rectangle.

    Input structure:
    - The background is black.
    - Several non-background colors appear as scattered singleton pixels.
    - All pixels of the same color define one minimum bounding rectangle, and
      the colors have distinct pixel counts.

    Rule:
    - Group all non-black pixels by color.
    - For each color, compute the minimum bounding box containing its pixels.
    - Paint each bounding box as a solid rectangle on a black canvas of the same
      shape as the input.
    - Paint colors from largest pixel count to smallest pixel count, so later
      smaller-count rectangles overwrite earlier larger-count rectangles.

    Key insight:
    - The sparse points are rectangle corners/markers rather than final shapes,
      and overlap priority is encoded by how many visible points each color has.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    records: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in np.unique(g):
        color = int(color)
        if color == BLACK:
            continue
        coords = np.argwhere(g == color)
        assert coords.size > 0
        records.append((int(coords.shape[0]), color, _bbox_from_coords(coords)))

    assert records, "expected at least one foreground color"
    counts = [count for count, _color, _box in records]
    assert len(counts) == len(set(counts)), "foreground pixel counts must be distinct"

    out = np.full(g.shape, BLACK, dtype=int)
    for _count, color, (r0, c0, r1, c1) in sorted(records, key=lambda item: item[0], reverse=True):
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _independent_capacity(height: int, width: int) -> int:
    return ((int(height) + 1) // 2) * ((int(width) + 1) // 2)


def _box_height(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def _box_width(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def _box_contains(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _sample_box(height: int, width: int, point_count: int, rng: random.Random) -> tuple[int, int, int, int] | None:
    max_h = min(BOX_SIDE_RANGE[1], int(height))
    max_w = min(BOX_SIDE_RANGE[1], int(width))
    choices: list[tuple[int, int]] = []
    for box_h in range(BOX_SIDE_RANGE[0], max_h + 1):
        for box_w in range(BOX_SIDE_RANGE[0], max_w + 1):
            if _independent_capacity(box_h, box_w) >= int(point_count):
                choices.append((box_h, box_w))
    if not choices:
        return None

    box_h, box_w = rng.choice(choices)
    r0 = rng.randint(0, int(height) - box_h)
    c0 = rng.randint(0, int(width) - box_w)
    return r0, c0, r0 + box_h - 1, c0 + box_w - 1


def _sample_boxes(height: int, width: int, counts: list[int], rng: random.Random) -> list[tuple[int, int, int, int]] | None:
    for _ in range(120):
        first = _sample_box(height, width, counts[0], rng)
        if first is None:
            return None

        second = None
        for _try in range(180):
            candidate = _sample_box(height, width, counts[1], rng)
            if candidate is None:
                return None
            if _intersection_area(first, candidate) < MIN_OVERLAP_AREA:
                continue
            if _box_contains(first, candidate) or _box_contains(candidate, first):
                continue
            second = candidate
            break
        if second is None:
            continue

        boxes = [first, second]
        ok = True
        for count in counts[2:]:
            chosen = None
            prefer_overlap = rng.random() < 0.6
            for _try in range(160):
                candidate = _sample_box(height, width, count, rng)
                if candidate is None:
                    return None
                if any(candidate == box for box in boxes):
                    continue
                if any(_box_contains(candidate, box) or _box_contains(box, candidate) for box in boxes):
                    continue
                if prefer_overlap and not any(_intersection_area(candidate, box) >= MIN_OVERLAP_AREA for box in boxes):
                    continue
                chosen = candidate
                break
            if chosen is None:
                ok = False
                break
            boxes.append(chosen)
        if ok:
            return boxes
    return None


def _near_existing(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    row, col = cell
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (row + dr, col + dc) in occupied:
                return True
    return False


def _can_place(cell: tuple[int, int], occupied: set[tuple[int, int]], local: set[tuple[int, int]]) -> bool:
    return cell not in occupied and cell not in local and not _near_existing(cell, occupied | local)


def _sample_points_for_box(
    box: tuple[int, int, int, int],
    point_count: int,
    occupied: set[tuple[int, int]],
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    r0, c0, r1, c1 = box
    diagonal_pairs = [
        ((r0, c0), (r1, c1)),
        ((r0, c1), (r1, c0)),
    ]

    all_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    for _ in range(140):
        local: set[tuple[int, int]] = set()
        anchors = list(rng.choice(diagonal_pairs))
        if not all(_can_place(cell, occupied, local) or cell in local for cell in anchors):
            continue
        valid_anchors = True
        for cell in anchors:
            if not _can_place(cell, occupied, local):
                valid_anchors = False
                break
            local.add(cell)
        if not valid_anchors:
            continue

        cells = all_cells[:]
        rng.shuffle(cells)
        for cell in cells:
            if len(local) >= int(point_count):
                break
            if _can_place(cell, occupied, local):
                local.add(cell)

        if len(local) != int(point_count):
            continue
        coords = np.array(sorted(local), dtype=int)
        if _bbox_from_coords(coords) == box:
            return local
    return None


def _draw_records(
    shape: tuple[int, int],
    records: list[tuple[int, int, tuple[int, int, int, int]]],
) -> tuple[np.ndarray, int]:
    out = np.full(shape, BLACK, dtype=int)
    overwritten = 0
    for count, color, (r0, c0, r1, c1) in sorted(records, key=lambda item: item[0], reverse=True):
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        overwritten += int(np.count_nonzero(region != BLACK))
        region[:, :] = int(color)
    return out, overwritten


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_SIZE_RANGE)
    width = rng.randint(*GRID_SIZE_RANGE)
    n_colors = rng.randint(*COLOR_COUNT_RANGE)
    colors = rng.sample(ARC_COLORS, n_colors)
    counts = rng.sample(range(POINT_COUNT_RANGE[0], POINT_COUNT_RANGE[1] + 1), n_colors)

    boxes = _sample_boxes(height, width, counts, rng)
    if boxes is None:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    occupied: set[tuple[int, int]] = set()
    records: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color, count, box in zip(colors, counts, boxes, strict=False):
        pts = _sample_points_for_box(box, count, occupied, rng)
        if pts is None:
            return None
        for r, c in pts:
            grid[r, c] = int(color)
        occupied |= pts
        records.append((int(count), int(color), box))

    expected, overwritten = _draw_records(grid.shape, records)
    if overwritten == 0:
        return None
    if not all(np.any(expected == int(color)) for color in colors):
        return None
    if np.array_equal(expected, grid):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
