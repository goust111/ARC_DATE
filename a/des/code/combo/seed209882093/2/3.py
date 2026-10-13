from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
TARGET_COLOR = 6
ARROW_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RECT_SHAPES = [(2, 2), (1, 3), (3, 1), (2, 3), (3, 2), (1, 2), (2, 1)]
OUTER_RETRIES = 400
PLACE_RETRIES = 160
MAX_ARROW_STEPS = 8

ARROW_PATTERNS = {
    frozenset({(0, 0), (-1, 0), (-1, -1), (-1, 1), (1, 0)}): (-1, 0),
    frozenset({(0, 0), (1, 0), (1, -1), (1, 1), (-1, 0)}): (1, 0),
    frozenset({(0, 0), (0, -1), (-1, -1), (1, -1), (0, 1)}): (0, -1),
    frozenset({(0, 0), (0, 1), (-1, 1), (1, 1), (0, -1)}): (0, 1),
}


def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if _in_bounds(nr, nc, h, w):
            yield nr, nc


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _shift_cells(cells: Iterable[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in cells]


def _arrow_cells(center: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    cr, cc = center
    dr, dc = direction
    if (dr, dc) == (-1, 0):
        offsets = [(0, 0), (-1, 0), (-1, -1), (-1, 1), (1, 0)]
    elif (dr, dc) == (1, 0):
        offsets = [(0, 0), (1, 0), (1, -1), (1, 1), (-1, 0)]
    elif (dr, dc) == (0, -1):
        offsets = [(0, 0), (0, -1), (-1, -1), (1, -1), (0, 1)]
    else:
        offsets = [(0, 0), (0, 1), (-1, 1), (1, 1), (0, -1)]
    return [(cr + rr, cc + cc2) for rr, cc2 in offsets]


def _monochrome_components(grid: np.ndarray, bg: int = BACKGROUND) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append({"color": color, "cells": cells})
    return comps


def _detect_arrow(comp: dict) -> dict | None:
    cells = list(comp["cells"])
    if len(cells) != 5:
        return None
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 != 2 or c1 - c0 != 2:
        return None
    center = (r0 + 1, c0 + 1)
    if center not in set(cells):
        return None
    offsets = frozenset((r - center[0], c - center[1]) for r, c in cells)
    direction = ARROW_PATTERNS.get(offsets)
    if direction is None:
        return None
    return {
        "color": int(comp["color"]),
        "cells": sorted(cells),
        "center": center,
        "dir": direction,
    }


def _collect_arrows(grid: np.ndarray) -> list[dict]:
    arrows: list[dict] = []
    for comp in _monochrome_components(grid):
        arrow = _detect_arrow(comp)
        if arrow is not None:
            arrows.append(arrow)
    arrows.sort(key=lambda a: (a["center"][0], a["center"][1], a["color"]))
    return arrows


def _slide_steps(grid: np.ndarray, arrow: dict) -> int:
    h, w = grid.shape
    cr, cc = arrow["center"]
    dr, dc = arrow["dir"]
    self_cells = set(arrow["cells"])
    steps = 0
    while True:
        fr = cr + (steps + 2) * dr
        fc = cc + (steps + 2) * dc
        if not _in_bounds(fr, fc, h, w):
            return steps
        if int(grid[fr, fc]) != BACKGROUND and (fr, fc) not in self_cells:
            return steps
        steps += 1


def _stop_kind(grid: np.ndarray, arrow: dict) -> str:
    steps = _slide_steps(grid, arrow)
    cr, cc = arrow["center"]
    dr, dc = arrow["dir"]
    fr = cr + (steps + 2) * dr
    fc = cc + (steps + 2) * dc
    if not _in_bounds(fr, fc, grid.shape[0], grid.shape[1]):
        return "boundary"
    if int(grid[fr, fc]) == TARGET_COLOR:
        return "target"
    return "object"


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    arrows = _collect_arrows(g)

    for arrow in arrows:
        for r, c in arrow["cells"]:
            out[r, c] = BACKGROUND

    for arrow in arrows:
        cr, cc = arrow["center"]
        out[cr, cc] = arrow["color"]

    for arrow in arrows:
        steps = _slide_steps(g, arrow)
        dr, dc = arrow["dir"]
        for r, c in _shift_cells(arrow["cells"], dr * steps, dc * steps):
            out[r, c] = arrow["color"]

    return out


def _expand_cells(cells: Iterable[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if _in_bounds(rr, cc, h, w):
                    expanded.add((rr, cc))
    return expanded


def _cells_in_bounds(cells: Iterable[tuple[int, int]], h: int, w: int) -> bool:
    return all(_in_bounds(r, c, h, w) for r, c in cells)


def _input_clear(cells: Iterable[tuple[int, int]], reserved: set[tuple[int, int]], h: int, w: int) -> bool:
    cell_list = list(cells)
    return _cells_in_bounds(cell_list, h, w) and all(cell not in reserved for cell in cell_list)


def _paint(grid: np.ndarray, cells: Iterable[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _path_clear(
    center: tuple[int, int],
    direction: tuple[int, int],
    steps: int,
    occupied: set[tuple[int, int]],
    blocker: set[tuple[int, int]],
    h: int,
    w: int,
    stop_mode: str,
) -> bool:
    dr, dc = direction
    base_cells = _arrow_cells(center, direction)

    for t in range(steps + 1):
        moved = _shift_cells(base_cells, dr * t, dc * t)
        if not _cells_in_bounds(moved, h, w):
            return False
        if any(cell in occupied or cell in blocker for cell in moved):
            return False

    for t in range(steps):
        fr = center[0] + (t + 2) * dr
        fc = center[1] + (t + 2) * dc
        if not _in_bounds(fr, fc, h, w):
            return False
        if (fr, fc) in occupied or (fr, fc) in blocker:
            return False

    fr = center[0] + (steps + 2) * dr
    fc = center[1] + (steps + 2) * dc
    if stop_mode == "boundary":
        return not _in_bounds(fr, fc, h, w)
    return _in_bounds(fr, fc, h, w) and (fr, fc) in blocker


def _max_back_steps(center: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> int:
    r, c = center
    dr, dc = direction
    if (dr, dc) == (0, 1):
        return c - 1
    if (dr, dc) == (0, -1):
        return w - 2 - c
    if (dr, dc) == (1, 0):
        return r - 1
    return h - 2 - r


def _rect_cells(top: int, left: int, height: int, width: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(top, top + height) for c in range(left, left + width)]


def _choose_target_layout(
    rng: random.Random,
    h: int,
    w: int,
    direction: tuple[int, int],
    rect_h: int,
    rect_w: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    dr, dc = direction
    if (dr, dc) == (0, 1):
        left_lo, left_hi = 4, w - rect_w
        top_lo, top_hi = 0, h - rect_h
        if left_lo > left_hi or top_lo > top_hi:
            return None
        left = rng.randint(left_lo, left_hi)
        top = rng.randint(top_lo, top_hi)
        center = (top + rng.randrange(rect_h), left - 2)
    elif (dr, dc) == (0, -1):
        left_lo, left_hi = 0, w - rect_w - 4
        top_lo, top_hi = 0, h - rect_h
        if left_lo > left_hi or top_lo > top_hi:
            return None
        left = rng.randint(left_lo, left_hi)
        top = rng.randint(top_lo, top_hi)
        center = (top + rng.randrange(rect_h), left + rect_w + 1)
    elif (dr, dc) == (1, 0):
        top_lo, top_hi = 4, h - rect_h
        left_lo, left_hi = 0, w - rect_w
        if top_lo > top_hi or left_lo > left_hi:
            return None
        top = rng.randint(top_lo, top_hi)
        left = rng.randint(left_lo, left_hi)
        center = (top - 2, left + rng.randrange(rect_w))
    else:
        top_lo, top_hi = 0, h - rect_h - 4
        left_lo, left_hi = 0, w - rect_w
        if top_lo > top_hi or left_lo > left_hi:
            return None
        top = rng.randint(top_lo, top_hi)
        left = rng.randint(left_lo, left_hi)
        center = (top + rect_h + 1, left + rng.randrange(rect_w))
    return _rect_cells(top, left, rect_h, rect_w), center


def _place_target_arrow(
    rng: random.Random,
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    color: int,
) -> dict | None:
    h, w = grid.shape
    for _ in range(PLACE_RETRIES):
        direction = rng.choice(DIRS4)
        rect_h, rect_w = rng.choice(RECT_SHAPES)
        layout = _choose_target_layout(rng, h, w, direction, rect_h, rect_w)
        if layout is None:
            continue
        rect_cells, center = layout
        if not _input_clear(rect_cells, reserved, h, w):
            continue
        arrow_reserved = reserved | _expand_cells(rect_cells, h, w)
        max_steps = min(_max_back_steps(center, direction, h, w), MAX_ARROW_STEPS)
        if max_steps < 1:
            continue
        steps_options = list(range(1, max_steps + 1))
        rng.shuffle(steps_options)
        blocker = set(rect_cells)
        for steps in steps_options:
            origin = (center[0] - steps * direction[0], center[1] - steps * direction[1])
            arrow_cells = _arrow_cells(origin, direction)
            if not _input_clear(arrow_cells, arrow_reserved, h, w):
                continue
            if not _path_clear(origin, direction, steps, occupied, blocker, h, w, "object"):
                continue
            _paint(grid, rect_cells, TARGET_COLOR)
            _paint(grid, arrow_cells, color)
            occupied.update(rect_cells)
            occupied.update(arrow_cells)
            reserved.update(_expand_cells(rect_cells, h, w))
            reserved.update(_expand_cells(arrow_cells, h, w))
            return {"color": color, "cells": arrow_cells, "center": origin, "dir": direction}
    return None


def _place_boundary_arrow(
    rng: random.Random,
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    color: int,
) -> dict | None:
    h, w = grid.shape
    for _ in range(PLACE_RETRIES):
        direction = rng.choice(DIRS4)
        dr, dc = direction
        if (dr, dc) == (0, 1):
            center = (rng.randint(1, h - 2), w - 2)
        elif (dr, dc) == (0, -1):
            center = (rng.randint(1, h - 2), 1)
        elif (dr, dc) == (1, 0):
            center = (h - 2, rng.randint(1, w - 2))
        else:
            center = (1, rng.randint(1, w - 2))

        max_steps = min(_max_back_steps(center, direction, h, w), MAX_ARROW_STEPS)
        if max_steps < 1:
            continue
        steps_options = list(range(1, max_steps + 1))
        rng.shuffle(steps_options)
        for steps in steps_options:
            origin = (center[0] - steps * dr, center[1] - steps * dc)
            arrow_cells = _arrow_cells(origin, direction)
            if not _input_clear(arrow_cells, reserved, h, w):
                continue
            if not _path_clear(origin, direction, steps, occupied, set(), h, w, "boundary"):
                continue
            _paint(grid, arrow_cells, color)
            occupied.update(arrow_cells)
            reserved.update(_expand_cells(arrow_cells, h, w))
            return {"color": color, "cells": arrow_cells, "center": origin, "dir": direction}
    return None


def _place_extra_target(
    rng: random.Random,
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    for _ in range(PLACE_RETRIES):
        rect_h, rect_w = rng.choice(RECT_SHAPES)
        if rect_h > h or rect_w > w:
            continue
        top = rng.randint(0, h - rect_h)
        left = rng.randint(0, w - rect_w)
        rect_cells = _rect_cells(top, left, rect_h, rect_w)
        if not _input_clear(rect_cells, reserved, h, w):
            continue
        _paint(grid, rect_cells, TARGET_COLOR)
        occupied.update(rect_cells)
        reserved.update(_expand_cells(rect_cells, h, w))
        return True
    return False


def _validate_generated(grid: np.ndarray, expected_arrows: int) -> bool:
    arrows = _collect_arrows(grid)
    if len(arrows) != expected_arrows:
        return False
    if not np.any(grid == TARGET_COLOR):
        return False

    finals: list[set[tuple[int, int]]] = []
    trails: set[tuple[int, int]] = set()
    stop_kinds: list[str] = []
    out = main(grid)

    for arrow in arrows:
        steps = _slide_steps(grid, arrow)
        if steps < 1:
            return False
        dr, dc = arrow["dir"]
        moved = set(_shift_cells(arrow["cells"], dr * steps, dc * steps))
        if any(cell in moved for cell in trails):
            return False
        if any(moved & other for other in finals):
            return False
        finals.append(moved)
        trails.add(arrow["center"])
        stop_kinds.append(_stop_kind(grid, arrow))
        if out[arrow["center"][0], arrow["center"][1]] != arrow["color"]:
            return False
        for r, c in moved:
            if out[r, c] != arrow["color"]:
                return False

    if len(arrows) > 1:
        if "boundary" not in stop_kinds or "target" not in stop_kinds:
            return False
    elif stop_kinds[0] not in {"boundary", "target"}:
        return False

    return not np.array_equal(grid, out)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 20)
    w = rng.randint(12, 20)
    grid = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()

    arrow_count = rng.randint(2, 5)
    colors = rng.sample(ARROW_COLORS, arrow_count)

    if arrow_count == 1:
        branch_plan = [rng.choice(["target", "boundary"])]
    else:
        target_count = rng.randint(1, arrow_count - 1)
        branch_plan = ["target"] * target_count + ["boundary"] * (arrow_count - target_count)
        rng.shuffle(branch_plan)

    arrows: list[dict] = []
    for color, branch in zip(colors, branch_plan):
        if branch == "target":
            placed = _place_target_arrow(rng, grid, occupied, reserved, color)
        else:
            placed = _place_boundary_arrow(rng, grid, occupied, reserved, color)
        if placed is None:
            return None
        arrows.append(placed)

    for _ in range(rng.randint(0, 2)):
        if rng.random() < 0.65:
            _place_extra_target(rng, grid, occupied, reserved)

    if not _validate_generated(grid, len(arrows)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(OUTER_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to create a valid puzzle input")
