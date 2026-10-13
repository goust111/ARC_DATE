from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

RING_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
PLUS_OFFSETS = ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0))

GEN_HEIGHT_RANGE = (16, 21)
GEN_WIDTH_RANGE = (14, 19)
GEN_COUNT_CHOICES = (5, 6, 7)
GEN_MAX_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _shape_cells(center: tuple[int, int], offsets: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr, dc in offsets]


def _boxes_are_separate(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) > 3 or abs(a[1] - b[1]) > 3


def _wall_color(grid: np.ndarray, bg: int) -> int:
    assert grid.ndim == 2 and grid.size > 0
    color = int(grid[-1, 0])
    assert color != bg and np.all(grid[-1, :] == color), "expected one full bottom wall"
    return color


def _find_hollow_boxes(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    wall = _wall_color(g, bg)
    boxes: list[dict] = []
    used: set[tuple[int, int]] = set()

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != bg:
                continue
            ring = [int(g[r + dr, c + dc]) for dr, dc in RING_OFFSETS]
            color = ring[0]
            if color in (bg, wall) or any(v != color for v in ring):
                continue
            cells = set(_shape_cells((r, c), RING_OFFSETS + ((0, 0),)))
            assert used.isdisjoint(cells), "target boxes must not overlap"
            boxes.append({"center": (r, c), "color": color})
            used.update(cells)

    assert boxes, "expected at least one hollow 3x3 box"
    return boxes


def _settle_objects(
    shape_offsets: tuple[tuple[int, int], ...],
    objects: list[dict],
    shape: tuple[int, int],
    wall_color: int,
    bg: int,
) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    out[-1, :] = wall_color
    occupied = np.zeros((h, w), dtype=bool)
    occupied[-1, :] = True
    settled_centers: dict[int, tuple[int, int]] = {}

    for obj in sorted(objects, key=lambda item: (-item["center"][0], item["center"][1])):
        center_r, center_c = obj["center"]
        color = int(obj["color"])
        while True:
            nxt = [(center_r + dr + 1, center_c + dc) for dr, dc in shape_offsets]
            if any(rr < 0 or rr >= h or cc < 0 or cc >= w or occupied[rr, cc] for rr, cc in nxt):
                break
            center_r += 1

        cells = _shape_cells((center_r, center_c), shape_offsets)
        assert all(0 <= rr < h and 0 <= cc < w and not occupied[rr, cc] for rr, cc in cells)
        for rr, cc in cells:
            out[rr, cc] = color
            occupied[rr, cc] = True
        settled_centers[color] = (center_r, center_c)

    return out, settled_centers


def _oracle_output(grid: np.ndarray) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    wall = _wall_color(g, bg)
    objects = _find_hollow_boxes(g)
    return _settle_objects(PLUS_OFFSETS, objects, g.shape, wall, bg)


def _settle_boxes_baseline(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    wall = _wall_color(g, bg)
    objects = _find_hollow_boxes(g)
    _, centers = _settle_objects(RING_OFFSETS, objects, g.shape, wall, bg)
    return centers


def _cross_centers_from_output(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    wall = _wall_color(g, bg)
    centers: dict[int, tuple[int, int]] = {}
    for color in sorted(int(v) for v in np.unique(g) if v not in (bg, wall)):
        coords = np.argwhere(g == color)
        assert coords.shape[0] == 5, "each output object must be a 5-cell cross"
        for r, c in coords:
            r = int(r)
            c = int(c)
            nbrs = {(r + dr, c + dc) for dr, dc in PLUS_OFFSETS if (dr, dc) != (0, 0)}
            if all(0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and int(g[rr, cc]) == color for rr, cc in nbrs):
                centers[color] = (r, c)
                break
        assert color in centers, "failed to recover a cross center"
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn each hollow 3x3 colored square into a same-color plus, then gravity-pack the pluses onto the bottom wall.

    Input structure:
    - The grid has black background and a full grey wall on the bottom row.
    - Above the wall are several separated monochrome 3x3 hollow squares with black centers.
    - Each square is at least two rows above the wall and does not overlap another square's 3x3 window.

    Rule:
    - Rewrite every hollow square in place as a same-color plus: erase the four corners and fill the center with that color.
    - Remove the original square outlines from their scattered positions and keep only the bottom wall fixed.
    - Settle the pluses downward as rigid bodies, processing the ones with larger original center row first; ties are resolved by smaller center column first.
    - A plus stops when its next downward step would hit the wall, a previously settled plus, or the grid boundary.

    Key insight:
    - Converting a ring into a plus changes the collision geometry: empty corners let neighboring plus arms interlock, so the final packed layout cannot be reproduced by stacking the original squares.
    """
    out, _ = _oracle_output(np.asarray(grid, dtype=int))
    return out


def _all_centers(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(1, h - 4) for c in range(1, w - 1)]


def _choose_patterned_centers(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    all_centers = _all_centers(h, w)
    row_slots = [r for r in range(1, h - 4, 4)]
    col_slots = [c for c in range(1, w - 1, 4)]
    if len(row_slots) < 2 or w < 6:
        return None

    for _ in range(300):
        centers: list[tuple[int, int]] = []

        upper_row = random.choice(row_slots[:-1]) if len(row_slots) > 1 else row_slots[0]
        lower_choices = [r for r in row_slots if r >= upper_row + 4]
        if not lower_choices:
            continue
        lower_row = random.choice(lower_choices)
        interlock_col = random.randint(1, w - 4)
        pair = [(upper_row, interlock_col + 2), (lower_row, interlock_col)]
        centers.extend(pair)

        tie_options = []
        for row in row_slots:
            for c1 in col_slots:
                for c2 in col_slots:
                    if c2 <= c1 or c2 - c1 < 4:
                        continue
                    cand1 = (row, c1)
                    cand2 = (row, c2)
                    if all(_boxes_are_separate(cand1, e) and _boxes_are_separate(cand2, e) for e in centers):
                        tie_options.append((cand1, cand2))
        if not tie_options:
            continue
        tie_a, tie_b = random.choice(tie_options)
        centers.extend([tie_a, tie_b])

        random.shuffle(all_centers)
        for cand in all_centers:
            if len(centers) >= count:
                break
            if all(_boxes_are_separate(cand, e) for e in centers):
                centers.append(cand)

        if len(centers) == count:
            return centers
    return None


def _render_input(h: int, w: int, centers: list[tuple[int, int]], colors: list[int]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[-1, :] = WALL
    for (r, c), color in zip(centers, colors, strict=True):
        for dr, dc in RING_OFFSETS:
            grid[r + dr, c + dc] = color
    return grid


def _has_same_row_pair(centers: list[tuple[int, int]]) -> bool:
    rows = Counter(r for r, _ in centers)
    return any(v >= 2 for v in rows.values())


def _interlock_pair_count(centers: dict[int, tuple[int, int]]) -> int:
    items = list(centers.values())
    total = 0
    for i, (r1, c1) in enumerate(items):
        for r2, c2 in items[i + 1 :]:
            if sorted((abs(r1 - r2), abs(c1 - c2))) == [1, 2]:
                total += 1
    return total


def _validate_generated(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if h > 30 or w > 30:
        return False

    boxes = _find_hollow_boxes(g)
    if len(boxes) not in GEN_COUNT_CHOICES:
        return False
    if len({obj["color"] for obj in boxes}) != len(boxes):
        return False
    if any(center[0] > h - 5 for center in (obj["center"] for obj in boxes)):
        return False

    centers = [obj["center"] for obj in boxes]
    if not _has_same_row_pair(centers):
        return False
    for i, a in enumerate(centers):
        for b in centers[i + 1 :]:
            if not _boxes_are_separate(a, b):
                return False

    oracle_out, oracle_centers = _oracle_output(g)
    try:
        actual = main(g)
    except Exception:
        return False
    if not np.array_equal(actual, oracle_out):
        return False

    actual_centers = _cross_centers_from_output(actual)
    if actual_centers != oracle_centers:
        return False
    if _interlock_pair_count(actual_centers) < 1:
        return False

    box_centers = _settle_boxes_baseline(g)
    if all(box_centers[color] == actual_centers[color] for color in actual_centers):
        return False

    distinct_output_rows = len({r for r, _ in actual_centers.values()})
    if distinct_output_rows < 3:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    count = random.choice(GEN_COUNT_CHOICES)
    centers = _choose_patterned_centers(h, w, count)
    if centers is None:
        return None
    colors = random.sample(PALETTE, count)
    random.shuffle(centers)
    grid = _render_input(h, w, centers, colors)
    if _validate_generated(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
