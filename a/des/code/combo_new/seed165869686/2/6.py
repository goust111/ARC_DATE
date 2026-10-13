from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
ARROW_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


ARROW_TEMPLATES = {
    "up": {
        "cells": ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
        "head": ((0, 0), (0, 1), (0, 2)),
        "tail": (2, 1),
        "dir": (-1, 0),
    },
    "down": {
        "cells": ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2)),
        "head": ((2, 0), (2, 1), (2, 2)),
        "tail": (0, 1),
        "dir": (1, 0),
    },
    "left": {
        "cells": ((0, 0), (1, 0), (2, 0), (1, 1), (1, 2)),
        "head": ((0, 0), (1, 0), (2, 0)),
        "tail": (1, 2),
        "dir": (0, -1),
    },
    "right": {
        "cells": ((0, 2), (1, 0), (1, 1), (1, 2), (2, 2)),
        "head": ((0, 2), (1, 2), (2, 2)),
        "tail": (1, 0),
        "dir": (0, 1),
    },
}

NORMALIZED_SHAPES = {
    frozenset(template["cells"]): name for name, template in ARROW_TEMPLATES.items()
}


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _shift_cells(cells: Iterable[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in cells]


def _paint(grid: np.ndarray, cells: Iterable[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _component_cells(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    h, w = grid.shape
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _identify_arrow(color: int, cells: list[tuple[int, int]]) -> dict[str, object]:
    if len(cells) != 5:
        raise ValueError("each arrow must contain exactly 5 pixels")
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
        raise ValueError("each arrow must fit a 3x3 bounding box")
    normalized = frozenset((r - r0, c - c0) for r, c in cells)
    if normalized not in NORMALIZED_SHAPES:
        raise ValueError("component is not a valid arrow shape")
    name = NORMALIZED_SHAPES[normalized]
    template = ARROW_TEMPLATES[name]
    return {
        "color": color,
        "name": name,
        "cells": _shift_cells(template["cells"], r0, c0),
        "head": _shift_cells(template["head"], r0, c0),
        "tail": (r0 + template["tail"][0], c0 + template["tail"][1]),
        "dir": template["dir"],
    }


def _find_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    arrows: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BLACK, GREY)):
        for comp in _component_cells(grid, color):
            arrows.append(_identify_arrow(color, comp))
    arrows.sort(key=lambda arrow: min(arrow["cells"]))  # deterministic only
    return arrows


def _distance_to_wall(grid: np.ndarray, arrow: dict[str, object]) -> int:
    head = arrow["head"]
    dr, dc = arrow["dir"]
    h, w = grid.shape

    if dr != 0:
        head_row = head[0][0]
        cols = sorted(c for _, c in head)
        if dr < 0:
            for wall_r in range(head_row - 1, -1, -1):
                if all(int(grid[wall_r, c]) == GREY for c in cols):
                    return head_row - wall_r - 1
        else:
            for wall_r in range(head_row + 1, h):
                if all(int(grid[wall_r, c]) == GREY for c in cols):
                    return wall_r - head_row - 1
    else:
        head_col = head[0][1]
        rows = sorted(r for r, _ in head)
        if dc < 0:
            for wall_c in range(head_col - 1, -1, -1):
                if all(int(grid[r, wall_c]) == GREY for r in rows):
                    return head_col - wall_c - 1
        else:
            for wall_c in range(head_col + 1, w):
                if all(int(grid[r, wall_c]) == GREY for r in rows):
                    return wall_c - head_col - 1
    raise ValueError("arrow has no reachable aligned wall")


def main(grid: np.ndarray) -> np.ndarray:
    """Push each arrow toward its facing wall and leave a same-colored tail track of matching travel length.

    Input structure:
    - The background is black.
    - Grey pixels form straight horizontal or vertical wall segments.
    - Each colored foreground object is a 5-pixel arrow: a 3-pixel head and a 2-pixel shaft extending backward from the head center.
    - Every arrow faces a grey wall somewhere ahead on its movement axis.

    Rule:
    - Read each arrow's facing direction from which side contains the 3-pixel head.
    - Measure D, the number of empty cells between the head and the nearest aligned grey wall in that direction.
    - Translate the whole arrow forward by D cells so the head becomes adjacent to the wall.
    - If D > 0, draw a same-colored trail of length D starting at the arrow's original tail cell and extending in the movement direction.

    Key insight:
    - The asymmetric 5-pixel arrow shape encodes both the travel direction and the single-cell-wide trail axis, while the wall distance directly determines both the translation amount and the trail length.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    arrows = _find_arrows(g)

    for arrow in arrows:
        for r, c in arrow["cells"]:
            out[r, c] = BLACK

    for arrow in arrows:
        distance = _distance_to_wall(g, arrow)
        dr, dc = arrow["dir"]
        color = arrow["color"]
        tail_r, tail_c = arrow["tail"]

        for step in range(distance):
            out[tail_r + step * dr, tail_c + step * dc] = color
        for r, c in arrow["cells"]:
            out[r + distance * dr, c + distance * dc] = color

    return out


def _max_arrow_count(h: int, w: int) -> int:
    area = h * w
    if area < 170:
        return 2
    if area < 280:
        return 3
    return 4


def _sample_gap(axis_size: int) -> int:
    max_gap = min(7, axis_size - 4)
    if max_gap <= 0:
        return 0
    if random.random() < 0.5:
        return 0
    return random.randint(1, max_gap)


def _bbox_with_padding(
    cells: Iterable[tuple[int, int]], h: int, w: int, pad: int = 1
) -> tuple[int, int, int, int]:
    cells = list(cells)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return (
        max(0, min(rows) - pad),
        max(0, min(cols) - pad),
        min(h - 1, max(rows) + pad),
        min(w - 1, max(cols) + pad),
    )


def _plan_arrow(
    h: int,
    w: int,
    color: int,
    orientation: str,
    reserved: np.ndarray,
) -> dict[str, object] | None:
    template = ARROW_TEMPLATES[orientation]
    gap = _sample_gap(h if orientation in ("up", "down") else w)
    if orientation in ("up", "down"):
        wall_len = random.randint(3, min(8, w))
    else:
        wall_len = random.randint(3, min(8, h))

    if orientation == "up":
        max_wall_r = h - gap - 4
        if max_wall_r < 0:
            return None
        wall_r = random.randint(0, max_wall_r)
        wall_c0 = random.randint(0, w - wall_len)
        head_offset = random.randint(0, wall_len - 3)
        origin = (wall_r + gap + 1, wall_c0 + head_offset)
        wall = [(wall_r, wall_c0 + dc) for dc in range(wall_len)]
    elif orientation == "down":
        min_wall_r = gap + 3
        if min_wall_r > h - 1:
            return None
        wall_r = random.randint(min_wall_r, h - 1)
        wall_c0 = random.randint(0, w - wall_len)
        head_offset = random.randint(0, wall_len - 3)
        origin = (wall_r - gap - 3, wall_c0 + head_offset)
        wall = [(wall_r, wall_c0 + dc) for dc in range(wall_len)]
    elif orientation == "left":
        max_wall_c = w - gap - 4
        if max_wall_c < 0:
            return None
        wall_c = random.randint(0, max_wall_c)
        wall_r0 = random.randint(0, h - wall_len)
        head_offset = random.randint(0, wall_len - 3)
        origin = (wall_r0 + head_offset, wall_c + gap + 1)
        wall = [(wall_r0 + dr, wall_c) for dr in range(wall_len)]
    else:
        min_wall_c = gap + 3
        if min_wall_c > w - 1:
            return None
        wall_c = random.randint(min_wall_c, w - 1)
        wall_r0 = random.randint(0, h - wall_len)
        head_offset = random.randint(0, wall_len - 3)
        origin = (wall_r0 + head_offset, wall_c - gap - 3)
        wall = [(wall_r0 + dr, wall_c) for dr in range(wall_len)]

    input_cells = _shift_cells(template["cells"], origin[0], origin[1])
    dr, dc = template["dir"]
    output_cells = _shift_cells(input_cells, gap * dr, gap * dc)
    tail_r = origin[0] + template["tail"][0]
    tail_c = origin[1] + template["tail"][1]
    trail = [(tail_r + step * dr, tail_c + step * dc) for step in range(gap)]

    union = input_cells + output_cells + trail + wall
    if any(r < 0 or c < 0 or r >= h or c >= w for r, c in union):
        return None
    r0, c0, r1, c1 = _bbox_with_padding(union, h, w, pad=1)
    if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
        return None

    return {
        "color": color,
        "orientation": orientation,
        "gap": gap,
        "input_cells": input_cells,
        "output_cells": output_cells,
        "trail": trail,
        "wall": wall,
        "reserve": (r0, c0, r1, c1),
    }


def _render_input(h: int, w: int, plans: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for plan in plans:
        _paint(grid, plan["wall"], GREY)
        _paint(grid, plan["input_cells"], plan["color"])
    return grid


def _render_output(h: int, w: int, plans: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for plan in plans:
        _paint(grid, plan["wall"], GREY)
        _paint(grid, plan["trail"], plan["color"])
        _paint(grid, plan["output_cells"], plan["color"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 24)
    w = random.randint(12, 24)
    arrow_count = random.randint(1, _max_arrow_count(h, w))
    colors = random.sample(ARROW_COLORS, arrow_count)
    reserved = np.zeros((h, w), dtype=bool)
    plans: list[dict[str, object]] = []

    for color in colors:
        placed = None
        for _ in range(200):
            orientation = random.choice(tuple(ARROW_TEMPLATES))
            candidate = _plan_arrow(h, w, color, orientation, reserved)
            if candidate is None:
                continue
            r0, c0, r1, c1 = candidate["reserve"]
            reserved[r0 : r1 + 1, c0 : c1 + 1] = True
            plans.append(candidate)
            placed = candidate
            break
        if placed is None:
            return None

    inp = _render_input(h, w, plans)
    expected = _render_output(h, w, plans)

    arrows = _find_arrows(inp)
    if len(arrows) != arrow_count:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-wall input")
