import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9
SEGMENT_COLORS = (1, 2, 3, 4, 6, 7, 8)
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

PAIR_COUNT_RANGE = (1, 4)
GRID_SIZE_MIN = 14
GRID_SIZE_MAX = 30
SEGMENT_LENGTH_RANGE = (2, 4)
LEG_LENGTH_RANGE = (2, 8)
GENERATION_ATTEMPTS = 400
PAIR_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in CARDINALS:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_segment(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    colors = {int(grid[r, c]) for r, c in coords}
    assert GREY in colors and len(colors) == 2

    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    assert len(rows) == 1 or len(cols) == 1
    length = len(coords)
    assert SEGMENT_LENGTH_RANGE[0] <= length <= SEGMENT_LENGTH_RANGE[1]

    if len(rows) == 1:
        ordered = sorted(coords, key=lambda rc: rc[1])
        axis = 1
    else:
        ordered = sorted(coords, key=lambda rc: rc[0])
        axis = 0

    grey_cells = [cell for cell in ordered if int(grid[cell]) == GREY]
    assert len(grey_cells) == 1
    grey = grey_cells[0]
    assert grey == ordered[0] or grey == ordered[-1]

    color = next(color for color in colors if color != GREY)
    assert sum(int(grid[r, c]) == color for r, c in coords) == length - 1

    if axis == 1:
        direction = (0, 1) if grey == ordered[-1] else (0, -1)
    else:
        direction = (1, 0) if grey == ordered[-1] else (-1, 0)

    return {
        "color": int(color),
        "grey": tuple(map(int, grey)),
        "dir": direction,
        "cells": [tuple(map(int, cell)) for cell in ordered],
    }


def _extract_segments(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    comps = _component_coords(g != BG)
    assert comps
    segments = [_extract_segment(g, comp) for comp in comps]
    return segments


def _path_plan(seg_a: dict, seg_b: dict, grid: np.ndarray) -> dict | None:
    ga = seg_a["grey"]
    gb = seg_b["grey"]
    dr1, dc1 = seg_a["dir"]

    if dr1 != 0:
        if ga[1] == gb[1]:
            return None
        leg1_steps = (gb[0] - ga[0]) * dr1
        if leg1_steps <= 0:
            return None
        corner = (gb[0], ga[1])
        dr2, dc2 = 0, 1 if gb[1] > ga[1] else -1
        leg2_steps = abs(gb[1] - ga[1])
    else:
        if ga[0] == gb[0]:
            return None
        leg1_steps = (gb[1] - ga[1]) * dc1
        if leg1_steps <= 0:
            return None
        corner = (ga[0], gb[1])
        dr2, dc2 = 1 if gb[0] > ga[0] else -1, 0
        leg2_steps = abs(gb[0] - ga[0])

    if leg2_steps <= 0:
        return None

    h, w = grid.shape
    cells_to_check = []
    leg1_cells = []
    for k in range(1, leg1_steps):
        cell = (ga[0] + dr1 * k, ga[1] + dc1 * k)
        leg1_cells.append(cell)
        cells_to_check.append(cell)
    leg2_cells = []
    cells_to_check.append(corner)
    for k in range(1, leg2_steps):
        cell = (corner[0] + dr2 * k, corner[1] + dc2 * k)
        leg2_cells.append(cell)
        cells_to_check.append(cell)

    for r, c in cells_to_check:
        if not (0 <= r < h and 0 <= c < w):
            return None
        if int(grid[r, c]) != BG:
            return None

    return {
        "corner": corner,
        "leg1_cells": leg1_cells,
        "leg2_cells": leg2_cells,
    }


def _build_segment(grey: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    return [
        (grey[0] - direction[0] * k, grey[1] - direction[1] * k)
        for k in range(length)
    ]


def _perpendiculars(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(-dc, dr), (dc, -dr)]


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        out.update(_neighbors4(r, c, h, w))
    return out


def _segments_touch(cells_a: list[tuple[int, int]], cells_b: list[tuple[int, int]], h: int, w: int) -> bool:
    set_b = set(cells_b)
    halo_b = _halo(set_b, h, w)
    return any(cell in halo_b for cell in cells_a)


def main(grid: np.ndarray) -> np.ndarray:
    """Connect each color-matched pair of marked line segments with a two-color elbow path.

    Input structure
    - The background is black.
    - Each foreground object is a straight horizontal or vertical segment of length
      2-4, with exactly one grey endpoint and the remaining cells in one color.
    - Every segment color appears on exactly two separate segments.

    Rule
    - For each repeated color, identify its two segments and the grey endpoint on each.
    - Exactly one ordering of the two segments yields a valid L-shaped connection:
      start from segment A's grey endpoint, continue straight in the segment's own
      outward direction until aligned with segment B's grey endpoint, then turn once
      and go straight to B.
    - Recolor all grey endpoints back to their segment color, paint the first leg
      with segment A's color, paint the second leg (excluding B's endpoint) with
      segment B's color, and mark the corner cell maroon.

    Key insight
    - The grey marker does not just show which endpoint connects; combined with the
      segment orientation it fixes the first leg's direction, so the correct ordered
      pairing is the only one whose elbow path stays in empty space.
    """
    g = np.asarray(grid, dtype=int)
    segments = _extract_segments(g)

    by_color: dict[int, list[dict]] = defaultdict(list)
    for seg in segments:
        by_color[seg["color"]].append(seg)

    assert all(len(group) == 2 for group in by_color.values())

    plans: list[tuple[dict, dict, dict]] = []
    for color, group in by_color.items():
        first, second = group
        plan_forward = _path_plan(first, second, g)
        plan_backward = _path_plan(second, first, g)

        valid = []
        if plan_forward is not None:
            valid.append((first, second, plan_forward))
        if plan_backward is not None:
            valid.append((second, first, plan_backward))
        assert len(valid) == 1, f"color {color} must have exactly one valid ordered path"
        plans.append(valid[0])

    out = g.copy()
    for seg in segments:
        gr, gc = seg["grey"]
        out[gr, gc] = seg["color"]

    for seg_a, seg_b, plan in plans:
        for r, c in plan["leg1_cells"]:
            assert int(out[r, c]) == BG
            out[r, c] = seg_a["color"]
        corner_r, corner_c = plan["corner"]
        assert int(out[corner_r, corner_c]) == BG
        out[corner_r, corner_c] = MAROON
        for r, c in plan["leg2_cells"]:
            assert int(out[r, c]) == BG
            out[r, c] = seg_b["color"]

    return out


def _candidate_offsets(
    dir_a: tuple[int, int],
    dir_b: tuple[int, int],
    turn_dir: tuple[int, int],
    len_a: int,
    len_b: int,
    leg1: int,
    leg2: int,
) -> dict[str, list[tuple[int, int]] | tuple[int, int]]:
    a_cells = [(-dir_a[0] * k, -dir_a[1] * k) for k in range(len_a)]
    leg1_cells = [(dir_a[0] * k, dir_a[1] * k) for k in range(1, leg1)]
    corner = (dir_a[0] * leg1, dir_a[1] * leg1)
    leg2_cells = [
        (corner[0] + turn_dir[0] * k, corner[1] + turn_dir[1] * k)
        for k in range(1, leg2)
    ]
    grey_b = (corner[0] + turn_dir[0] * leg2, corner[1] + turn_dir[1] * leg2)
    b_cells = [
        (grey_b[0] - dir_b[0] * k, grey_b[1] - dir_b[1] * k)
        for k in range(len_b)
    ]
    return {
        "a_cells": a_cells,
        "leg1_cells": leg1_cells,
        "corner": corner,
        "leg2_cells": leg2_cells,
        "b_cells": b_cells,
    }


def _translate(cells: list[tuple[int, int]], anchor: tuple[int, int]) -> list[tuple[int, int]]:
    return [(anchor[0] + r, anchor[1] + c) for r, c in cells]


def _make_pair(
    h: int,
    w: int,
    color: int,
    reserved_halo: set[tuple[int, int]],
) -> dict | None:
    for _ in range(PAIR_ATTEMPTS):
        dir_a = random.choice(CARDINALS)
        turn_dir = random.choice(_perpendiculars(dir_a))
        dir_b = random.choice([d for d in CARDINALS if d != turn_dir])
        len_a = random.randint(*SEGMENT_LENGTH_RANGE)
        len_b = random.randint(*SEGMENT_LENGTH_RANGE)
        leg1 = random.randint(*LEG_LENGTH_RANGE)
        leg2 = random.randint(*LEG_LENGTH_RANGE)

        offsets = _candidate_offsets(dir_a, dir_b, turn_dir, len_a, len_b, leg1, leg2)
        all_offsets = (
            offsets["a_cells"]
            + offsets["leg1_cells"]
            + [offsets["corner"]]
            + offsets["leg2_cells"]
            + offsets["b_cells"]
        )

        rows = [r for r, _ in all_offsets]
        cols = [c for _, c in all_offsets]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        low_r, high_r = -min_r, h - 1 - max_r
        low_c, high_c = -min_c, w - 1 - max_c
        if low_r > high_r or low_c > high_c:
            continue

        grey_a = (random.randint(low_r, high_r), random.randint(low_c, high_c))
        a_cells = _translate(offsets["a_cells"], grey_a)
        leg1_cells = _translate(offsets["leg1_cells"], grey_a)
        corner = (grey_a[0] + offsets["corner"][0], grey_a[1] + offsets["corner"][1])
        leg2_cells = _translate(offsets["leg2_cells"], grey_a)
        b_cells = _translate(offsets["b_cells"], grey_a)
        grey_b = b_cells[0]

        output_cells = set(a_cells) | set(b_cells) | set(leg1_cells) | set(leg2_cells) | {corner}
        if len(output_cells) != len(a_cells) + len(b_cells) + len(leg1_cells) + len(leg2_cells) + 1:
            continue
        if any(cell in reserved_halo for cell in output_cells):
            continue
        if _segments_touch(a_cells, b_cells, h, w):
            continue

        temp = np.zeros((h, w), dtype=int)
        for r, c in a_cells:
            temp[r, c] = color
        for r, c in b_cells:
            temp[r, c] = color
        temp[grey_a] = GREY
        temp[grey_b] = GREY

        seg_a = {"color": color, "grey": grey_a, "dir": dir_a, "cells": a_cells}
        seg_b = {"color": color, "grey": grey_b, "dir": dir_b, "cells": b_cells}
        if _path_plan(seg_a, seg_b, temp) is None:
            continue
        if _path_plan(seg_b, seg_a, temp) is not None:
            continue

        return {
            "color": color,
            "grey_a": grey_a,
            "grey_b": grey_b,
            "a_cells": a_cells,
            "b_cells": b_cells,
            "output_cells": output_cells,
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    min_size = max(GRID_SIZE_MIN, 8 + 4 * pair_count)
    h = random.randint(min_size, GRID_SIZE_MAX)
    w = random.randint(min_size, GRID_SIZE_MAX)
    colors = random.sample(SEGMENT_COLORS, pair_count)

    grid = np.zeros((h, w), dtype=int)
    reserved_halo: set[tuple[int, int]] = set()

    for color in colors:
        pair = _make_pair(h, w, color, reserved_halo)
        if pair is None:
            return None

        for r, c in pair["a_cells"]:
            grid[r, c] = color
        for r, c in pair["b_cells"]:
            grid[r, c] = color
        grid[pair["grey_a"]] = GREY
        grid[pair["grey_b"]] = GREY
        reserved_halo |= _halo(pair["output_cells"], h, w)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
