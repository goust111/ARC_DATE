from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
PALETTE = tuple(color for color in range(10) if color not in (BG, FRAME))

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 28)
INNER_H_RANGE = (1, 8)
INNER_W_RANGE = (1, 8)
PLACEMENT_TRIES = 80
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_bbox(grid: np.ndarray, comp: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(comp) != expected:
        return None

    for r in range(r0, r1 + 1):
        if int(grid[r, c0]) != FRAME or int(grid[r, c1]) != FRAME:
            return None
    for c in range(c0, c1 + 1):
        if int(grid[r0, c]) != FRAME or int(grid[r1, c]) != FRAME:
            return None
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == FRAME):
        return None
    return r0, c0, r1, c1


def _vote_colors(inner: np.ndarray) -> tuple[int, int]:
    order: list[int] = []
    counts: dict[int, int] = {}
    for value in inner.ravel():
        color = int(value)
        if color == BG:
            continue
        if color not in counts:
            order.append(color)
            counts[color] = 0
        counts[color] += 1

    assert len(order) == 2, "each frame must contain exactly two vote colors"
    first, second = order
    if counts[first] > counts[second]:
        return first, second
    if counts[second] > counts[first]:
        return second, first
    return first, second


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey frame by majority vote and keep the minority vote as a center marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    comps = _components(g == FRAME)
    assert comps, "expected at least one grey frame"

    for comp in comps:
        bbox = _frame_bbox(g, comp)
        assert bbox is not None, "every grey component must be a 1-cell-thick rectangular frame"
        r0, c0, r1, c1 = bbox
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        majority, minority = _vote_colors(inner)

        out[r0, c0 : c1 + 1] = majority
        out[r1, c0 : c1 + 1] = majority
        out[r0 : r1 + 1, c0] = majority
        out[r0 : r1 + 1, c1] = majority
        out[r0 + 1 : r1, c0 + 1 : c1] = BG

        center_r = r0 + 1 + (inner.shape[0] - 1) // 2
        center_c = c0 + 1 + (inner.shape[1] - 1) // 2
        out[center_r, center_c] = minority

    return out


def _paint_frame(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int) -> None:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _choose_vote_layout(
    inner_h: int, inner_w: int, color_a: int, color_b: int
) -> tuple[dict[tuple[int, int], int], int, int]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    area = len(cells)
    assert area >= 2

    tie_possible = area >= 2
    majority_possible = area >= 3
    if majority_possible and random.random() < 0.5:
        branch = "majority"
    else:
        branch = "tie" if tie_possible else "majority"

    votes: dict[tuple[int, int], int] = {}
    majority = random.choice((color_a, color_b))
    minority = color_b if majority == color_a else color_a

    if branch == "tie":
        max_each = max(1, area // 2)
        each = random.randint(1, max_each)
        total = 2 * each

        earliest_index = random.randint(0, area - total)
        chosen = [cells[earliest_index]]
        remaining = cells[earliest_index + 1 :]
        chosen.extend(random.sample(remaining, total - 1))
        chosen.sort()

        votes[chosen[0]] = majority
        rest = chosen[1:]
        majority_slots = random.sample(rest, each - 1)
        majority_set = set(majority_slots)
        for cell in rest:
            votes[cell] = majority if cell in majority_set else minority
        return votes, majority, minority

    total = random.randint(3, area)
    majority_count = random.randint(total // 2 + 1, total - 1)
    minority_count = total - majority_count
    chosen = random.sample(cells, total)
    majority_slots = set(random.sample(chosen, majority_count))
    for cell in chosen:
        votes[cell] = majority if cell in majority_slots else minority
    assert sum(color == majority for color in votes.values()) == majority_count
    assert sum(color == minority for color in votes.values()) == minority_count
    return votes, majority, minority


def _place_frames(height: int, width: int) -> list[tuple[int, int, int, int]]:
    area = height * width
    max_frames = 2
    if area >= 240:
        max_frames = 3
    if area >= 360:
        max_frames = 4
    if area >= 520:
        max_frames = 5

    frame_count = random.randint(1, max_frames)
    reserved = np.zeros((height, width), dtype=bool)
    frames: list[tuple[int, int, int, int]] = []

    for _ in range(frame_count):
        placed = False
        for _ in range(PLACEMENT_TRIES):
            inner_h = random.randint(*INNER_H_RANGE)
            inner_w = random.randint(*INNER_W_RANGE)
            if inner_h * inner_w < 2:
                continue

            box_h = inner_h + 2
            box_w = inner_w + 2
            if box_h + 2 > height or box_w + 2 > width:
                continue

            r0 = random.randint(1, height - box_h - 1)
            c0 = random.randint(1, width - box_w - 1)
            r1 = r0 + box_h - 1
            c1 = c0 + box_w - 1
            if reserved[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
                continue

            reserved[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
            frames.append((r0, c0, inner_h, inner_w))
            placed = True
            break
        if not placed:
            return []
    return frames


def _expected_output(
    grid: np.ndarray, specs: list[tuple[int, int, int, int, int, int]]
) -> np.ndarray:
    out = grid.copy()
    for r0, c0, inner_h, inner_w, majority, minority in specs:
        r1 = r0 + inner_h + 1
        c1 = c0 + inner_w + 1
        out[r0, c0 : c1 + 1] = majority
        out[r1, c0 : c1 + 1] = majority
        out[r0 : r1 + 1, c0] = majority
        out[r0 : r1 + 1, c1] = majority
        out[r0 + 1 : r1, c0 + 1 : c1] = BG
        center_r = r0 + 1 + (inner_h - 1) // 2
        center_c = c0 + 1 + (inner_w - 1) // 2
        out[center_r, center_c] = minority
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    frames = _place_frames(height, width)
    if not frames:
        return None

    grid = np.full((height, width), BG, dtype=int)
    specs: list[tuple[int, int, int, int, int, int]] = []

    for r0, c0, inner_h, inner_w in frames:
        _paint_frame(grid, r0, c0, inner_h, inner_w)
        color_a, color_b = random.sample(PALETTE, 2)
        votes, majority, minority = _choose_vote_layout(inner_h, inner_w, color_a, color_b)
        for (dr, dc), color in votes.items():
            grid[r0 + 1 + dr, c0 + 1 + dc] = color
        specs.append((r0, c0, inner_h, inner_w, majority, minority))

    expected = _expected_output(grid, specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
