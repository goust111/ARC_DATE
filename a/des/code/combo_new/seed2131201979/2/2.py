from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
YELLOW = 4
LINE_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

MIN_SIZE = 8
MAX_SIZE = 30
MAX_HUBS = 5
MAX_ARM_LEN = 8
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 160


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _trace_arm(grid: np.ndarray, hub: tuple[int, int], direction: tuple[int, int]) -> dict[str, object]:
    h, w = grid.shape
    r, c = hub
    dr, dc = direction
    nr, nc = r + dr, c + dc
    assert 0 <= nr < h and 0 <= nc < w, "hub arm must start inside the grid"

    color = int(grid[nr, nc])
    assert color not in (BG, YELLOW), "arm must start with a non-background non-hub color"

    cells: list[tuple[int, int]] = []
    while 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == color:
        cells.append((int(nr), int(nc)))
        nr += dr
        nc += dc

    if 0 <= nr < h and 0 <= nc < w:
        assert int(grid[nr, nc]) == BG, "arm must terminate into background"

    return {"color": color, "cells": cells, "direction": direction}


def _hub_arms(grid: np.ndarray, hub: tuple[int, int]) -> tuple[dict[str, object], dict[str, object]]:
    h, w = grid.shape
    r, c = hub
    starts = []
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) not in (BG, YELLOW):
            starts.append((dr, dc))

    assert len(starts) == 2, "each hub must touch exactly two colored arm starts"
    dirs = set(starts)
    assert dirs in ({(-1, 0), (1, 0)}, {(0, -1), (0, 1)}), "arms must extend in opposite directions"

    first = _trace_arm(grid, hub, starts[0])
    second = _trace_arm(grid, hub, starts[1])
    assert int(first["color"]) != int(second["color"]), "paired arms must use different colors"
    return first, second


def main(grid: np.ndarray) -> np.ndarray:
    """Swap the colors of the two opposite line segments attached to each yellow hub.

    Input structure:
    - The background is black.
    - One or more single-cell yellow hubs appear on the grid.
    - Each yellow hub has exactly two straight one-cell-wide colored line segments attached to its four-neighbors.
    - The two segments attached to one hub lie on the same row or the same column, extend in opposite directions, and do not include the yellow hub cell itself.

    Rule:
    - For each yellow hub independently, identify the two opposite attached segments by tracing outward from the hub.
    - Keep each segment's position, direction, and length unchanged.
    - Recolor every cell of the first segment with the second segment's color, and recolor every cell of the second segment with the first segment's color.
    - Leave the yellow hub cells and the black background unchanged.

    Key insight:
    - The yellow hub is the local pairing cue: only the two segments that meet the same hub exchange colors, so the transformation is a hub-wise swap rather than a global color remapping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    hubs = [tuple(map(int, cell)) for cell in np.argwhere(g == YELLOW)]
    assert hubs, "expected at least one yellow hub"

    out = g.copy()
    traced: set[tuple[int, int]] = set()
    for hub in hubs:
        arm_a, arm_b = _hub_arms(g, hub)
        cells_a = arm_a["cells"]
        cells_b = arm_b["cells"]
        color_a = int(arm_a["color"])
        color_b = int(arm_b["color"])

        for cell in cells_a:
            assert cell not in traced, "line segments should not belong to multiple hubs"
            traced.add(cell)
            out[cell] = color_b
        for cell in cells_b:
            assert cell not in traced, "line segments should not belong to multiple hubs"
            traced.add(cell)
            out[cell] = color_a

    foreground = {
        (int(r), int(c))
        for r, c in np.argwhere((g != BG) & (g != YELLOW))
    }
    assert traced == foreground, "all non-hub foreground cells must belong to traced arms"
    return out


def _all_pairs() -> list[tuple[int, int]]:
    return [tuple(map(int, pair)) for pair in itertools.combinations(LINE_COLORS, 2)]


def _orientation_plan(hub_count: int) -> list[str]:
    if hub_count == 1:
        return [random.choice(("horizontal", "vertical"))]

    mode = random.choice(("all_horizontal", "all_vertical", "mixed"))
    if mode == "all_horizontal":
        return ["horizontal"] * hub_count
    if mode == "all_vertical":
        return ["vertical"] * hub_count

    horizontal_count = random.randint(1, hub_count - 1)
    plan = ["horizontal"] * horizontal_count + ["vertical"] * (hub_count - horizontal_count)
    random.shuffle(plan)
    return plan


def _clear_extent(blocked: np.ndarray, start: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = blocked.shape
    r, c = start
    dr, dc = direction
    length = 0
    nr, nc = r + dr, c + dc
    while 0 <= nr < h and 0 <= nc < w and not bool(blocked[nr, nc]):
        length += 1
        nr += dr
        nc += dc
    return length


def _object_cells(hub: tuple[int, int], orientation: str, neg_len: int, pos_len: int) -> list[tuple[int, int]]:
    r, c = hub
    cells = [(r, c)]
    if orientation == "horizontal":
        for step in range(1, neg_len + 1):
            cells.append((r, c - step))
        for step in range(1, pos_len + 1):
            cells.append((r, c + step))
    else:
        for step in range(1, neg_len + 1):
            cells.append((r - step, c))
        for step in range(1, pos_len + 1):
            cells.append((r + step, c))
    return cells


def _reserve(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True


def _place_object(
    shape: tuple[int, int], blocked: np.ndarray, orientation: str
) -> tuple[tuple[int, int], int, int] | None:
    h, w = shape
    for _ in range(PLACEMENT_ATTEMPTS):
        r = random.randint(0, h - 1)
        c = random.randint(0, w - 1)
        if bool(blocked[r, c]):
            continue

        if orientation == "horizontal":
            neg = _clear_extent(blocked, (r, c), (0, -1))
            pos = _clear_extent(blocked, (r, c), (0, 1))
        else:
            neg = _clear_extent(blocked, (r, c), (-1, 0))
            pos = _clear_extent(blocked, (r, c), (1, 0))

        neg = min(neg, MAX_ARM_LEN)
        pos = min(pos, MAX_ARM_LEN)
        if neg < 1 or pos < 1:
            continue

        neg_len = random.randint(1, neg)
        pos_len = random.randint(1, pos)
        return (int(r), int(c)), int(neg_len), int(pos_len)
    return None


def _paint_pair(
    grid: np.ndarray,
    hub: tuple[int, int],
    orientation: str,
    neg_len: int,
    pos_len: int,
    neg_color: int,
    pos_color: int,
) -> None:
    r, c = hub
    grid[r, c] = YELLOW
    if orientation == "horizontal":
        grid[r, c - neg_len : c] = neg_color
        grid[r, c + 1 : c + pos_len + 1] = pos_color
    else:
        grid[r - neg_len : r, c] = neg_color
        grid[r + 1 : r + pos_len + 1, c] = pos_color


def _expected_from_specs(
    shape: tuple[int, int],
    specs: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BG, dtype=int)
    out = np.full(shape, BG, dtype=int)
    for spec in specs:
        hub = spec["hub"]
        orientation = spec["orientation"]
        neg_len = int(spec["neg_len"])
        pos_len = int(spec["pos_len"])
        neg_color = int(spec["neg_color"])
        pos_color = int(spec["pos_color"])
        _paint_pair(inp, hub, orientation, neg_len, pos_len, neg_color, pos_color)
        _paint_pair(out, hub, orientation, neg_len, pos_len, pos_color, neg_color)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_SIZE, MAX_SIZE)
    width = random.randint(MIN_SIZE, MAX_SIZE)
    shape = (height, width)
    blocked = np.zeros(shape, dtype=bool)

    max_hubs_by_colors = len(_all_pairs())
    hub_cap = min(MAX_HUBS, max_hubs_by_colors, max(1, (height * width) // 30))
    hub_count = random.randint(1, hub_cap)
    orientations = _orientation_plan(hub_count)

    pairs = random.sample(_all_pairs(), hub_count)
    specs: list[dict[str, object]] = []

    for orientation, pair in zip(orientations, pairs):
        placed = _place_object(shape, blocked, orientation)
        if placed is None:
            return None
        hub, neg_len, pos_len = placed
        neg_color, pos_color = pair
        if random.random() < 0.5:
            neg_color, pos_color = pos_color, neg_color

        cells = _object_cells(hub, orientation, neg_len, pos_len)
        _reserve(blocked, cells)
        specs.append(
            {
                "hub": hub,
                "orientation": orientation,
                "neg_len": neg_len,
                "pos_len": pos_len,
                "neg_color": int(neg_color),
                "pos_color": int(pos_color),
            }
        )

    inp, expected = _expected_from_specs(shape, specs)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hub-swap ARC input")
