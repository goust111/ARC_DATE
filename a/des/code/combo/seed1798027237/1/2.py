from __future__ import annotations

import random

import numpy as np


BG = 0
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _find_components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    assert connectivity in (4, 8)
    neighbors = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    visited = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if visited[r, c]:
            continue
        visited[r, c] = True
        stack = [(r, c)]
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for dr, dc in neighbors:
                nr = cr + dr
                nc = cc + dc
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(sorted(coords), dtype=int))

    return comps


def _perp(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _centered_offsets(length: int) -> range:
    left = length // 2
    right = length - left - 1
    return range(-left, right + 1)


def _arrow_cells(
    shaft_front: tuple[int, int],
    direction: tuple[int, int],
    shaft_len: int,
) -> tuple[set[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]:
    fr, fc = shaft_front
    dr, dc = direction
    pr, pc = _perp(direction)

    shaft = [(fr - k * dr, fc - k * dc) for k in range(shaft_len)]
    tip = (fr + dr, fc + dc)
    cells = set(shaft)
    cells.add(tip)
    cells.add((fr + pr, fc + pc))
    cells.add((fr - pr, fc - pc))
    return cells, shaft, tip


def _parse_line(coords: np.ndarray) -> dict[str, object] | None:
    pts = _coord_set(coords)
    rows = sorted({r for r, _c in pts})
    cols = sorted({c for _r, c in pts})

    if len(rows) == 1 and cols == list(range(cols[0], cols[-1] + 1)):
        return {"orientation": "horizontal", "cells": pts, "length": len(pts)}
    if len(cols) == 1 and rows == list(range(rows[0], rows[-1] + 1)):
        return {"orientation": "vertical", "cells": pts, "length": len(pts)}
    return None


def _parse_arrow(coords: np.ndarray) -> dict[str, object] | None:
    pts = _coord_set(coords)
    matches: list[dict[str, object]] = []

    for direction in DIRS:
        dr, dc = direction
        pr, pc = _perp(direction)
        for fr, fc in pts:
            if (fr + dr, fc + dc) not in pts:
                continue
            if (fr + pr, fc + pc) not in pts or (fr - pr, fc - pc) not in pts:
                continue

            shaft_len = 0
            while (fr - shaft_len * dr, fc - shaft_len * dc) in pts:
                shaft_len += 1
            if shaft_len < 2:
                continue

            expected, shaft, tip = _arrow_cells((fr, fc), direction, shaft_len)
            if expected == pts:
                matches.append(
                    {
                        "cells": pts,
                        "direction": direction,
                        "shaft": shaft,
                        "shaft_front": (fr, fc),
                        "shaft_len": shaft_len,
                        "tip": tip,
                    }
                )

    if len(matches) == 1:
        return matches[0]
    return None


def _translation_steps(
    tip: tuple[int, int],
    direction: tuple[int, int],
    barrier_cells: set[tuple[int, int]],
    shape: tuple[int, int],
) -> int:
    tr, tc = tip
    dr, dc = direction

    for steps in range(max(shape) * 2):
        ahead = (tr + (steps + 1) * dr, tc + (steps + 1) * dc)
        if ahead in barrier_cells:
            return steps
        if not _in_bounds(shape, ahead[0], ahead[1]):
            break
    raise AssertionError("arrow path must meet the matching barrier")


def _parse_scene(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    pairs: list[dict[str, object]] = []

    for color in colors:
        comps = _find_components(g == color, connectivity=8)
        arrows: list[dict[str, object]] = []
        barriers: list[dict[str, object]] = []
        for comp in comps:
            arrow = _parse_arrow(comp)
            if arrow is not None:
                arrows.append(arrow)
                continue

            barrier = _parse_line(comp)
            if barrier is not None:
                barriers.append(barrier)
                continue

            raise AssertionError(f"color {color} must decompose into an arrow and a line segment")

        assert len(arrows) == 1, "each color must contain exactly one arrow"
        assert len(barriers) == 1, "each color must contain exactly one barrier"

        arrow = arrows[0]
        barrier = barriers[0]
        direction = arrow["direction"]
        expected_orientation = "vertical" if direction[0] == 0 else "horizontal"
        assert barrier["orientation"] == expected_orientation, "barrier must be perpendicular to motion"
        steps = _translation_steps(arrow["tip"], direction, barrier["cells"], g.shape)
        pairs.append({"color": color, "arrow": arrow, "barrier": barrier, "steps": steps})

    assert pairs, "expected at least one colored pair"
    return pairs


def _unfolded_arrow_cells(
    shaft_front: tuple[int, int],
    direction: tuple[int, int],
    shaft_len: int,
) -> set[tuple[int, int]]:
    arrow_cells, shaft, tip = _arrow_cells(shaft_front, direction, shaft_len)
    pr, pc = _perp(direction)
    out = {tip}

    for r, c in shaft:
        for offset in _centered_offsets(shaft_len):
            out.add((r + offset * pr, c + offset * pc))

    # Keep the side pixels explicitly; for odd lengths they already belong to the rectangle.
    fr, fc = shaft_front
    out.add((fr + pr, fc + pc))
    out.add((fr - pr, fc - pc))
    assert tip in out and len(out) >= len(arrow_cells)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire each colored arrow forward to its same-color barrier, then unfold its shaft
    into a rectangle perpendicular to the motion direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    pairs = _parse_scene(g)
    out = g.copy()

    for pair in pairs:
        for r, c in pair["arrow"]["cells"] | pair["barrier"]["cells"]:
            out[r, c] = BG

    for pair in pairs:
        color = int(pair["color"])
        arrow = pair["arrow"]
        steps = int(pair["steps"])
        dr, dc = arrow["direction"]
        fr, fc = arrow["shaft_front"]
        moved_front = (fr + steps * dr, fc + steps * dc)
        unfolded = _unfolded_arrow_cells(moved_front, arrow["direction"], int(arrow["shaft_len"]))

        for r, c in unfolded:
            assert _in_bounds(g.shape, r, c), "unfolded arrow must remain inside the grid"
            assert int(out[r, c]) in (BG, color), "transformed arrows must not overlap across colors"
            out[r, c] = color

    return out


# --------------- generator ---------------

GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
PAIR_COUNT_RANGE = (1, 3)
PAIR_ATTEMPTS = 180
GEN_ATTEMPTS = 300

# Odd shaft lengths keep the centered unfolding visually unambiguous.
SHAFT_LENGTH_CHOICES = (3, 5, 7)
BARRIER_LENGTH_RANGE = (2, 4)
TRAVEL_RANGE = (2, 8)
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def _shift_cells(
    cells: set[tuple[int, int]],
    direction: tuple[int, int],
    steps: int,
) -> set[tuple[int, int]]:
    dr, dc = direction
    return {(r + steps * dr, c + steps * dc) for r, c in cells}


def _translation_sweep(
    shaft_front: tuple[int, int],
    direction: tuple[int, int],
    shaft_len: int,
    steps: int,
) -> set[tuple[int, int]]:
    start_cells, _shaft, _tip = _arrow_cells(shaft_front, direction, shaft_len)
    sweep: set[tuple[int, int]] = set()
    for delta in range(steps + 1):
        sweep |= _shift_cells(start_cells, direction, delta)
    return sweep


def _dilate(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in NEIGHBORS8 + [(0, 0)]:
            nr = r + dr
            nc = c + dc
            if _in_bounds(shape, nr, nc):
                out.add((nr, nc))
    return out


def _all_in_bounds(shape: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    return all(_in_bounds(shape, r, c) for r, c in cells)


def _place_pair(
    shape: tuple[int, int],
    input_occ: set[tuple[int, int]],
    input_reserved: set[tuple[int, int]],
    output_reserved: set[tuple[int, int]],
) -> dict[str, object] | None:
    H, W = shape
    feasible_lengths = [length for length in SHAFT_LENGTH_CHOICES if length + 4 <= min(H, W)]
    if not feasible_lengths:
        return None

    max_travel = min(TRAVEL_RANGE[1], min(H, W) - 5)
    if max_travel < TRAVEL_RANGE[0]:
        return None

    for _ in range(PAIR_ATTEMPTS):
        direction = random.choice(DIRS)
        shaft_len = random.choice(feasible_lengths)
        barrier_len = random.randint(*BARRIER_LENGTH_RANGE)
        travel = random.randint(TRAVEL_RANGE[0], max_travel)
        anchor_index = random.randrange(barrier_len)

        final_front = (random.randrange(H), random.randrange(W))
        final_arrow, _shaft, _tip = _arrow_cells(final_front, direction, shaft_len)
        input_front = (final_front[0] - travel * direction[0], final_front[1] - travel * direction[1])
        input_arrow, _in_shaft, input_tip = _arrow_cells(input_front, direction, shaft_len)

        dr, dc = direction
        pr, pc = _perp(direction)
        anchor = (final_front[0] + 2 * dr, final_front[1] + 2 * dc)
        barrier = {
            (anchor[0] + (idx - anchor_index) * pr, anchor[1] + (idx - anchor_index) * pc)
            for idx in range(barrier_len)
        }

        output_cells = _unfolded_arrow_cells(final_front, direction, shaft_len)
        sweep = _translation_sweep(input_front, direction, shaft_len, travel)
        input_cells = input_arrow | barrier

        if not _all_in_bounds(shape, input_cells | output_cells):
            continue
        if input_cells & input_reserved:
            continue
        if sweep & input_occ:
            continue
        if output_cells & output_reserved:
            continue
        if len(input_arrow & barrier) != 0:
            continue

        return {
            "direction": direction,
            "shaft_len": shaft_len,
            "travel": travel,
            "input_arrow": input_arrow,
            "barrier": barrier,
            "output_cells": output_cells,
        }

    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    if pair_count > len(PALETTE):
        return None

    grid = np.full((H, W), BG, dtype=int)
    colors = random.sample(PALETTE, pair_count)
    input_occ: set[tuple[int, int]] = set()
    input_reserved: set[tuple[int, int]] = set()
    output_reserved: set[tuple[int, int]] = set()

    for color in colors:
        placed = _place_pair((H, W), input_occ, input_reserved, output_reserved)
        if placed is None:
            return None

        all_input = placed["input_arrow"] | placed["barrier"]
        for r, c in placed["input_arrow"]:
            grid[r, c] = color
        for r, c in placed["barrier"]:
            grid[r, c] = color

        input_occ |= all_input
        input_reserved |= _dilate(all_input, (H, W))
        output_reserved |= _dilate(placed["output_cells"], (H, W))

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
