from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 6)
GREY = 5

GEN_ATTEMPTS = 500
GAP = 2

LENGTHS = (2, 3, 4, 5, 6)
ORIENTATIONS = ((1, 1), (1, -1))


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    neighbor_fn = _neighbors4 if connectivity == 4 else _neighbors8

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbor_fn(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boxes_far_enough(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(br0 - ar1 - 1, ar0 - br1 - 1, 0)
    col_gap = max(bc0 - ac1 - 1, ac0 - bc1 - 1, 0)
    return row_gap >= gap or col_gap >= gap


def _analyze_diagonal(cells: list[tuple[int, int]]) -> tuple[int, tuple[int, int]]:
    if len(cells) < 2:
        raise AssertionError("diagonal segment must have length >= 2")

    cell_set = set(cells)
    if len(_components(_cells_to_mask(cells), connectivity=8)) != 1:
        raise AssertionError("grey cells must form one 8-connected segment")

    if len({r - c for r, c in cells}) == 1:
        ordered = sorted(cells)
        deltas = [(ordered[i + 1][0] - ordered[i][0], ordered[i + 1][1] - ordered[i][1]) for i in range(len(ordered) - 1)]
        if all(delta == (1, 1) for delta in deltas):
            return len(cells), (1, 1)

    if len({r + c for r, c in cells}) == 1:
        ordered = sorted(cells)
        deltas = [(ordered[i + 1][0] - ordered[i][0], ordered[i + 1][1] - ordered[i][1]) for i in range(len(ordered) - 1)]
        if all(delta == (1, -1) for delta in deltas):
            return len(cells), (1, -1)

    raise AssertionError("grey cells must form a contiguous 45-degree diagonal segment")


def _cells_to_mask(cells: list[tuple[int, int]]) -> np.ndarray:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    r1, c1 = max(rs), max(cs)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _describe_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rectangles = []
    used_grey = set()

    for color in PALETTE:
        comps = _components(g == color, connectivity=4)
        for coords in comps:
            r0, c0, r1, c1 = _bbox(coords)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 5 or width < 5:
                raise AssertionError("every rectangle must be at least 5x5")

            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            allowed = (sub == color) | (sub == GREY)
            if not bool(allowed.all()):
                raise AssertionError("rectangle bbox may only contain its color and the grey segment")

            if not bool((sub[0, :] == color).all() and (sub[-1, :] == color).all() and (sub[:, 0] == color).all() and (sub[:, -1] == color).all()):
                raise AssertionError("rectangle border must stay solid")

            grey_local = np.argwhere(sub == GREY)
            if grey_local.size == 0:
                raise AssertionError("each rectangle needs one grey diagonal segment")

            grey_cells = [(int(r0 + rr), int(c0 + cc)) for rr, cc in grey_local]
            if any(rr in (r0, r1) or cc in (c0, c1) for rr, cc in grey_cells):
                raise AssertionError("grey segment must stay strictly inside the rectangle")

            length, orientation = _analyze_diagonal(grey_cells)
            used_grey.update(grey_cells)
            rectangles.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "height": height,
                    "width": width,
                    "area": height * width,
                    "grey_cells": grey_cells,
                    "length": length,
                    "orientation": orientation,
                }
            )

    all_grey = {(int(r), int(c)) for r, c in np.argwhere(g == GREY)}
    if used_grey != all_grey:
        raise AssertionError("every grey cell must belong to exactly one rectangle")

    if not 3 <= len(rectangles) <= 5:
        raise AssertionError("expected 3 to 5 rectangles")

    for i, rect_a in enumerate(rectangles):
        for rect_b in rectangles[i + 1 :]:
            if not _boxes_far_enough(rect_a["bbox"], rect_b["bbox"], gap=GAP):
                raise AssertionError("rectangles must be separated by at least two background cells")

    lengths = [rect["length"] for rect in rectangles]
    if len(set(lengths)) != len(lengths):
        raise AssertionError("diagonal lengths must be distinct")

    return rectangles


def _apply_winner_transform(grid: np.ndarray, winner: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = winner["bbox"]
    color = int(winner["color"])
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if ((r - (r0 + 1)) + (c - (c0 + 1))) % 2 == 0:
                out[r, c] = BG
            else:
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace the interior of the rectangle containing the longest grey diagonal segment with a checkerboard.

    Input structure:
    - The grid has a black background plus 3-5 separated solid colored rectangles.
    - Each rectangle uses one color from {blue, red, green, yellow, magenta} and contains
      exactly one strictly interior grey diagonal segment.
    - Rectangle borders stay solid, while the grey segment may interrupt only interior cells.

    Rule:
    - Recover every rectangle and measure the length of its interior grey diagonal segment.
    - Select the unique rectangle whose grey segment is longest.
    - Keep that rectangle's 1-cell border unchanged, then checkerboard its strict interior:
      interior cells at even parity from the interior top-left become black, and odd-parity
      interior cells become the rectangle's own color. All other rectangles stay unchanged.

    Key insight:
    - The grey diagonals are only a hidden selection signal: the longest one chooses which
      rectangle gets transformed, and the winning diagonal itself disappears under the checkerboard.
    """
    rectangles = _describe_rectangles(np.asarray(grid, dtype=int))
    winner = max(rectangles, key=lambda rect: rect["length"])
    return _apply_winner_transform(grid, winner)


def _sample_count(rng: random.Random) -> int:
    return rng.choice((3, 4, 5))


def _sample_orientations(rng: random.Random, count: int, winner_idx: int) -> list[tuple[int, int]]:
    winner_orientation = rng.choice(ORIENTATIONS)
    others = [winner_orientation, (1, -winner_orientation[1])]
    while len(others) < count - 1:
        others.append(rng.choice(ORIENTATIONS))
    rng.shuffle(others)
    orientations = others[: count - 1]
    orientations.insert(winner_idx, winner_orientation)
    return orientations


def _sample_sizes(
    rng: random.Random,
    lengths: list[int],
    winner_idx: int,
) -> list[tuple[int, int]]:
    winner_length = lengths[winner_idx]
    winner_min = max(5, winner_length + 2)
    winner_h = rng.randint(winner_min, min(8, winner_min + 1))
    winner_w = rng.randint(winner_min, min(8, winner_min + 1))
    sizes: list[tuple[int, int] | None] = [None] * len(lengths)
    sizes[winner_idx] = (winner_h, winner_w)

    losers = [idx for idx in range(len(lengths)) if idx != winner_idx]
    big_idx = rng.choice(losers)

    for idx, length in enumerate(lengths):
        if idx == winner_idx:
            continue
        min_side = max(5, length + 2)
        if idx == big_idx:
            h_lo = max(min_side, winner_h)
            w_lo = max(min_side, winner_w)
            h = rng.randint(h_lo, min(9, max(h_lo, winner_h + 1)))
            w = rng.randint(w_lo, min(9, max(w_lo, winner_w + 1)))
            if h * w <= winner_h * winner_w:
                if h < 9:
                    h += 1
                elif w < 9:
                    w += 1
        else:
            h = rng.randint(min_side, min(8, min_side + 1))
            w = rng.randint(min_side, min(8, min_side + 1))
        sizes[idx] = (h, w)

    return [(int(h), int(w)) for h, w in sizes]


def _layout_slots(rng: random.Random, count: int) -> tuple[int, int, list[tuple[int, int]]]:
    if count == 3:
        kind = rng.choice(("row", "col", "corner"))
        if kind == "row":
            return 1, 3, [(0, 0), (0, 1), (0, 2)]
        if kind == "col":
            return 3, 1, [(0, 0), (1, 0), (2, 0)]
        missing = rng.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
        slots = [(r, c) for r in range(2) for c in range(2) if (r, c) != missing]
        return 2, 2, slots
    if count == 4:
        return 2, 2, [(r, c) for r in range(2) for c in range(2)]
    if rng.random() < 0.5:
        missing = rng.choice([(r, c) for r in range(2) for c in range(3)])
        slots = [(r, c) for r in range(2) for c in range(3) if (r, c) != missing]
        return 2, 3, slots
    missing = rng.choice([(r, c) for r in range(3) for c in range(2)])
    slots = [(r, c) for r in range(3) for c in range(2) if (r, c) != missing]
    return 3, 2, slots


def _place_bboxes_from_template(
    rng: random.Random,
    sizes: list[tuple[int, int]],
) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]] | None:
    rows, cols, slots = _layout_slots(rng, len(sizes))
    assignment = list(range(len(sizes)))
    rng.shuffle(assignment)

    row_needs = [0] * rows
    col_needs = [0] * cols
    for rect_idx, (slot_r, slot_c) in zip(assignment, slots):
        rh, rw = sizes[rect_idx]
        row_needs[slot_r] = max(row_needs[slot_r], rh)
        col_needs[slot_c] = max(col_needs[slot_c], rw)

    row_sizes = [need + rng.randint(0, 2) for need in row_needs]
    col_sizes = [need + rng.randint(0, 2) for need in col_needs]
    top_pad = rng.randint(0, 1)
    left_pad = rng.randint(0, 1)
    bottom_pad = rng.randint(0, 1)
    right_pad = rng.randint(0, 1)
    height = top_pad + sum(row_sizes) + GAP * (rows - 1) + bottom_pad
    width = left_pad + sum(col_sizes) + GAP * (cols - 1) + right_pad
    if height > 30 or width > 30:
        return None

    row_starts = []
    cur = top_pad
    for size in row_sizes:
        row_starts.append(cur)
        cur += size + GAP

    col_starts = []
    cur = left_pad
    for size in col_sizes:
        col_starts.append(cur)
        cur += size + GAP

    bboxes: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
    for rect_idx, (slot_r, slot_c) in zip(assignment, slots):
        rh, rw = sizes[rect_idx]
        max_r_off = row_sizes[slot_r] - rh
        max_c_off = col_sizes[slot_c] - rw
        r0 = row_starts[slot_r] + rng.randint(0, max_r_off)
        c0 = col_starts[slot_c] + rng.randint(0, max_c_off)
        bboxes[rect_idx] = (r0, c0, r0 + rh - 1, c0 + rw - 1)

    placed = [bbox for bbox in bboxes if bbox is not None]
    for i, box_a in enumerate(placed):
        for box_b in placed[i + 1 :]:
            if not _boxes_far_enough(box_a, box_b, gap=GAP):
                return None
    return (height, width), placed


def _diagonal_cells_for_bbox(
    rng: random.Random,
    bbox: tuple[int, int, int, int],
    length: int,
    orientation: tuple[int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    assert inner_h >= length and inner_w >= length

    start_r = rng.randint(r0 + 1, r1 - length)
    if orientation == (1, 1):
        start_c = rng.randint(c0 + 1, c1 - length)
        return [(start_r + step, start_c + step) for step in range(length)]

    start_c = rng.randint(c0 + length, c1 - 1)
    return [(start_r + step, start_c - step) for step in range(length)]


def _render_input(
    shape: tuple[int, int],
    bboxes: list[tuple[int, int, int, int]],
    colors: list[int],
    lengths: list[int],
    orientations: list[tuple[int, int]],
    rng: random.Random,
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for bbox, color, length, orientation in zip(bboxes, colors, lengths, orientations):
        r0, c0, r1, c1 = bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        for r, c in _diagonal_cells_for_bbox(rng, bbox, length, orientation):
            grid[r, c] = GREY
    return grid


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        rectangles = _describe_rectangles(grid)
    except AssertionError:
        return False

    winner = max(rectangles, key=lambda rect: rect["length"])
    others = [rect for rect in rectangles if rect is not winner]
    if max(rect["area"] for rect in others) <= winner["area"]:
        return False

    same_orientation = [rect for rect in others if rect["orientation"] == winner["orientation"]]
    other_orientation = [rect for rect in others if rect["orientation"] != winner["orientation"]]
    if not same_orientation or not other_orientation:
        return False

    total_fg = sum(rect["area"] for rect in rectangles)
    if total_fg >= grid.size * 0.7:
        return False

    out = _apply_winner_transform(grid, winner)
    if not np.array_equal(main(grid), out):
        return False

    diff = out != grid
    changed = {(int(r), int(c)) for r, c in np.argwhere(diff)}
    if not changed:
        return False

    r0, c0, r1, c1 = winner["bbox"]
    allowed = {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}
    if not changed.issubset(allowed):
        return False

    for rect in others:
        rr0, cc0, rr1, cc1 = rect["bbox"]
        if bool(diff[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
            return False

    return True


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    count = _sample_count(rng)
    lengths = rng.sample(LENGTHS, count)
    winner_idx = max(range(count), key=lambda idx: lengths[idx])
    orientations = _sample_orientations(rng, count, winner_idx)
    sizes = _sample_sizes(rng, lengths, winner_idx)
    layout = _place_bboxes_from_template(rng, sizes)
    if layout is None:
        return None

    shape, bboxes = layout
    colors = rng.sample(PALETTE, count)
    grid = _render_input(shape, bboxes, colors, lengths, orientations, rng)
    if _semantic_ok(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
