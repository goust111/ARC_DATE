from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

TOP_ARROW = frozenset({(0, 0), (0, 1), (1, 0)})
BOTTOM_ARROW = frozenset({(0, 1), (1, 0), (1, 1)})
LEFT_ARROW = frozenset({(0, 0), (1, 0), (1, 1)})
RIGHT_ARROW = frozenset({(0, 0), (0, 1), (1, 1)})

GRID_MARGIN = 3
GEN_ATTEMPTS = 300
PARTITION_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 120

GEN_RNG = random.Random()


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((rr, cc))
            for dr, dc in DIRS4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _dilate(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _find_arrows(grid: np.ndarray) -> dict[int, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    arrows: dict[int, dict[str, object]] = {}

    for comp in _components4(g != BG):
        if len(comp) != 3:
            continue

        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if (r1 - r0) != 1 or (c1 - c0) != 1:
            continue

        touches: list[str] = []
        if r0 == 0:
            touches.append("top")
        if r1 == h - 1:
            touches.append("bottom")
        if c0 == 0:
            touches.append("left")
        if c1 == w - 1:
            touches.append("right")
        if len(touches) != 1:
            continue

        values = [int(g[r, c]) for r, c in comp]
        non_grey = sorted({v for v in values if v != GREY})
        if len(non_grey) != 1 or values.count(GREY) != 2 or values.count(non_grey[0]) != 1:
            continue
        color = int(non_grey[0])
        corner = next((r, c) for r, c in comp if int(g[r, c]) == color)

        comp_set = set(comp)
        if sum((corner[0] + dr, corner[1] + dc) in comp_set for dr, dc in DIRS4) != 2:
            continue

        norm = frozenset((r - r0, c - c0) for r, c in comp)
        corner_norm = (corner[0] - r0, corner[1] - c0)

        direction: str | None = None
        edge = touches[0]
        if edge == "top" and norm == TOP_ARROW and corner_norm == (0, 0):
            direction = "down"
        elif edge == "bottom" and norm == BOTTOM_ARROW and corner_norm == (1, 1):
            direction = "up"
        elif edge == "left" and norm == LEFT_ARROW and corner_norm == (1, 0):
            direction = "right"
        elif edge == "right" and norm == RIGHT_ARROW and corner_norm == (0, 1):
            direction = "left"

        if direction is None:
            continue
        assert color not in arrows, "each fragment color must have exactly one arrow"
        arrows[color] = {"direction": direction, "corner": corner, "cells": tuple(sorted(comp))}

    assert arrows, "expected at least one colored arrow"
    return arrows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each color's fragments along its border arrow direction until they merge into a solid rectangle.

    The colored elbow pixel of each 3-cell L marker identifies which fragments it controls.
    Fragments move only along the arrow axis; their perpendicular span stays fixed, so the
    completed rectangle is recovered from total area and the unchanged orthogonal extent.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    arrows = _find_arrows(g)
    out = g.copy()

    for color, info in arrows.items():
        mask = g == int(color)
        corner_r, corner_c = map(int, info["corner"])
        mask[corner_r, corner_c] = False

        comps = _components4(mask)
        assert 2 <= len(comps) <= 3, "each color must have 2-3 fragment components"
        assert int(mask.sum()) > 0, "each arrow color must control at least one fragment"

        rows, cols = np.where(mask)
        direction = str(info["direction"])
        area = int(mask.sum())

        if direction in {"left", "right"}:
            r0 = int(rows.min())
            r1 = int(rows.max())
            assert np.all(mask[r0 : r1 + 1, :].any(axis=1)), "horizontal slide keeps a fixed full row span"
            height = int(r1 - r0 + 1)
            assert area % height == 0
            width = int(area // height)
            if direction == "left":
                c0 = int(cols.min())
            else:
                c0 = int(cols.max()) - width + 1
        else:
            c0 = int(cols.min())
            c1 = int(cols.max())
            assert np.all(mask[:, c0 : c1 + 1].any(axis=0)), "vertical slide keeps a fixed full column span"
            width = int(c1 - c0 + 1)
            assert area % width == 0
            height = int(area // width)
            if direction == "up":
                r0 = int(rows.min())
            else:
                r0 = int(rows.max()) - height + 1

        out[mask] = BG
        out[r0 : r0 + height, c0 : c0 + width] = int(color)

    return out


def _random_cut_walk(slice_count: int, span: int, pieces: int, rng: random.Random) -> list[list[int]] | None:
    assert pieces >= 2
    for _ in range(PARTITION_ATTEMPTS):
        cuts = sorted(rng.sample(range(1, span), pieces - 1))
        seq = [cuts]
        ok = True
        for _slice in range(1, slice_count):
            prev = seq[-1]
            cur: list[int] = []
            for j in range(pieces - 1):
                lo = max(prev[j] - 1, 1 if j == 0 else cur[-1] + 1)
                hi = min(prev[j] + 1, span - (pieces - 1 - j))
                if hi < lo:
                    ok = False
                    break
                cur.append(rng.randint(lo, hi))
            if not ok:
                break
            seq.append(cur)
        if not ok:
            continue
        if all(c == seq[0] for c in seq[1:]) and rng.random() < 0.75:
            continue
        return seq
    return None


def _partition_rectangle(height: int, width: int, pieces: int, axis: str, rng: random.Random) -> list[np.ndarray] | None:
    if axis == "horizontal":
        seq = _random_cut_walk(height, width, pieces, rng)
        if seq is None:
            return None
        labels = np.full((height, width), -1, dtype=int)
        for r, cuts in enumerate(seq):
            bounds = [0, *cuts, width]
            for idx in range(pieces):
                labels[r, bounds[idx] : bounds[idx + 1]] = idx
    else:
        seq = _random_cut_walk(width, height, pieces, rng)
        if seq is None:
            return None
        labels = np.full((height, width), -1, dtype=int)
        for c, cuts in enumerate(seq):
            bounds = [0, *cuts, height]
            for idx in range(pieces):
                labels[bounds[idx] : bounds[idx + 1], c] = idx

    masks = [labels == idx for idx in range(pieces)]
    if any(int(mask.sum()) == 0 for mask in masks):
        return None
    if any(len(_components4(mask)) != 1 for mask in masks):
        return None
    return masks


def _piece_offsets(direction: str, piece_count: int, gaps: list[int]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    for idx in range(piece_count):
        if direction == "left":
            offsets.append((0, sum(gaps[:idx])))
        elif direction == "right":
            offsets.append((0, -sum(gaps[idx:])))
        elif direction == "up":
            offsets.append((sum(gaps[:idx]), 0))
        else:
            offsets.append((-sum(gaps[idx:]), 0))
    return offsets


def _relative_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _make_arrow(
    direction: str,
    rect_top: int,
    rect_left: int,
    rect_height: int,
    rect_width: int,
    grid_shape: tuple[int, int],
    rng: random.Random,
) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    h, w = grid_shape
    if direction == "down":
        split = rng.randint(rect_left, rect_left + rect_width - 2)
        cells = {(0, split), (0, split + 1), (1, split)}
        corner = (0, split)
    elif direction == "up":
        split = rng.randint(rect_left + 1, rect_left + rect_width - 1)
        cells = {(h - 1, split - 1), (h - 1, split), (h - 2, split)}
        corner = (h - 1, split)
    elif direction == "right":
        split = rng.randint(rect_top + 1, rect_top + rect_height - 1)
        cells = {(split - 1, 0), (split, 0), (split, 1)}
        corner = (split, 0)
    else:
        split = rng.randint(rect_top, rect_top + rect_height - 2)
        cells = {(split, w - 2), (split, w - 1), (split + 1, w - 1)}
        corner = (split, w - 1)
    return cells, corner


def _valid_top_left_ranges(
    direction: str,
    rect_height: int,
    rect_width: int,
    total_gap: int,
    grid_shape: tuple[int, int],
) -> tuple[range, range] | None:
    h, w = grid_shape
    if direction == "left":
        row_lo = GRID_MARGIN
        row_hi = h - GRID_MARGIN - rect_height
        col_lo = GRID_MARGIN
        col_hi = w - GRID_MARGIN - rect_width - total_gap
    elif direction == "right":
        row_lo = GRID_MARGIN
        row_hi = h - GRID_MARGIN - rect_height
        col_lo = GRID_MARGIN + total_gap
        col_hi = w - GRID_MARGIN - rect_width
    elif direction == "up":
        row_lo = GRID_MARGIN
        row_hi = h - GRID_MARGIN - rect_height - total_gap
        col_lo = GRID_MARGIN
        col_hi = w - GRID_MARGIN - rect_width
    else:
        row_lo = GRID_MARGIN + total_gap
        row_hi = h - GRID_MARGIN - rect_height
        col_lo = GRID_MARGIN
        col_hi = w - GRID_MARGIN - rect_width

    if row_lo > row_hi or col_lo > col_hi:
        return None
    return range(row_lo, row_hi + 1), range(col_lo, col_hi + 1)


def _gen_one_object(
    color: int,
    direction: str,
    grid_shape: tuple[int, int],
    occupied_input: set[tuple[int, int]],
    forbidden_input: set[tuple[int, int]],
    occupied_output: set[tuple[int, int]],
    forbidden_output: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[set[tuple[int, int]], tuple[int, int]]] | None:
    piece_count = rng.choice((2, 3))
    if direction in {"left", "right"}:
        rect_height = rng.randint(3, 6)
        rect_width = rng.randint(piece_count + 2, 8)
        axis = "horizontal"
    else:
        rect_height = rng.randint(piece_count + 2, 8)
        rect_width = rng.randint(3, 6)
        axis = "vertical"

    masks = _partition_rectangle(rect_height, rect_width, piece_count, axis, rng)
    if masks is None:
        return None

    gaps = [rng.randint(1, 3) for _ in range(piece_count - 1)]
    total_gap = int(sum(gaps))
    ranges = _valid_top_left_ranges(direction, rect_height, rect_width, total_gap, grid_shape)
    if ranges is None:
        return None
    row_range, col_range = ranges

    offsets = _piece_offsets(direction, piece_count, gaps)
    rel_cells = [_relative_cells(mask) for mask in masks]

    for _ in range(PLACEMENT_ATTEMPTS):
        rect_top = rng.choice(tuple(row_range))
        rect_left = rng.choice(tuple(col_range))

        input_cells: set[tuple[int, int]] = set()
        for piece_cells, (dr, dc) in zip(rel_cells, offsets):
            for rr, cc in piece_cells:
                input_cells.add((rect_top + rr + dr, rect_left + cc + dc))

        output_cells = {
            (rect_top + rr, rect_left + cc)
            for rr in range(rect_height)
            for cc in range(rect_width)
        }

        arrow_cells, arrow_corner = _make_arrow(
            direction, rect_top, rect_left, rect_height, rect_width, grid_shape, rng
        )
        new_input = input_cells | arrow_cells
        new_output = output_cells | arrow_cells

        if new_input & occupied_input:
            continue
        if new_output & occupied_output:
            continue
        if any(cell in forbidden_input for cell in new_input):
            continue
        if any(cell in forbidden_output for cell in new_output):
            continue

        return input_cells, output_cells, (arrow_cells, arrow_corner)

    return None


def _paint_arrow(grid: np.ndarray, arrow_cells: set[tuple[int, int]], arrow_corner: tuple[int, int], color: int) -> None:
    for cell in arrow_cells:
        grid[cell] = GREY
    grid[arrow_corner] = int(color)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    color_count = rng.choice((1, 2, 3))
    base = 18 + 2 * (color_count - 1)
    height = rng.randint(base, base + 4)
    width = rng.randint(base, base + 4)
    shape = (height, width)

    colors = rng.sample(PALETTE, color_count)
    directions = rng.sample(("up", "down", "left", "right"), color_count)

    grid = np.full(shape, BG, dtype=int)
    expected = np.full(shape, BG, dtype=int)

    occupied_input: set[tuple[int, int]] = set()
    forbidden_input: set[tuple[int, int]] = set()
    occupied_output: set[tuple[int, int]] = set()
    forbidden_output: set[tuple[int, int]] = set()

    for color, direction in zip(colors, directions):
        placed = _gen_one_object(
            color,
            direction,
            shape,
            occupied_input,
            forbidden_input,
            occupied_output,
            forbidden_output,
            rng,
        )
        if placed is None:
            return None

        input_cells, output_cells, (arrow_cells, arrow_corner) = placed
        for r, c in input_cells:
            grid[r, c] = int(color)
        _paint_arrow(grid, arrow_cells, arrow_corner, color)

        for r, c in output_cells:
            expected[r, c] = int(color)
        _paint_arrow(expected, arrow_cells, arrow_corner, color)

        input_presence = input_cells | arrow_cells
        output_presence = output_cells | arrow_cells
        occupied_input |= input_presence
        forbidden_input |= _dilate(input_presence, shape)
        occupied_output |= output_presence
        forbidden_output |= _dilate(output_presence, shape)

    arrows = _find_arrows(grid)
    if set(arrows) != set(colors):
        return None

    for color, info in arrows.items():
        mask = grid == int(color)
        corner_r, corner_c = map(int, info["corner"])
        mask[corner_r, corner_c] = False
        comps = _components4(mask)
        if not (2 <= len(comps) <= 3):
            return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(GEN_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fragment-slide rectangle input")
