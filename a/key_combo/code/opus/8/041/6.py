from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
FILL_COLORS = (1, 2, 3)
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
INNER = "inner"
OUTER = "outer"
BLOCK_SIZE = 4
LOCAL_RADIUS = 4
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    comps.sort(key=lambda arr: (int(arr[:, 0].min()), int(arr[:, 1].min())))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _square_top_left(marker_r: int, marker_c: int, dr: int, dc: int) -> tuple[int, int]:
    return min(marker_r, marker_r + dr), min(marker_c, marker_c + dc)


def _hat_patterns(square_r0: int, square_c0: int) -> list[dict]:
    base = [
        {"point_dir": (-1, -1), "name": "top_left", "offsets": [(-2, 0), (-2, 1), (-1, 1)]},
        {"point_dir": (-1, 1), "name": "top_right", "offsets": [(-2, 0), (-2, 1), (-1, 0)]},
        {"point_dir": (1, -1), "name": "bottom_left", "offsets": [(2, 1), (3, 0), (3, 1)]},
        {"point_dir": (1, 1), "name": "bottom_right", "offsets": [(2, 0), (3, 0), (3, 1)]},
        {"point_dir": (-1, -1), "name": "left_top", "offsets": [(0, -2), (1, -2), (1, -1)]},
        {"point_dir": (1, -1), "name": "left_bottom", "offsets": [(0, -2), (0, -1), (1, -2)]},
        {"point_dir": (-1, 1), "name": "right_top", "offsets": [(0, 3), (1, 2), (1, 3)]},
        {"point_dir": (1, 1), "name": "right_bottom", "offsets": [(0, 2), (0, 3), (1, 3)]},
    ]
    out = []
    for item in base:
        cells = {
            (square_r0 + dr, square_c0 + dc)
            for dr, dc in item["offsets"]
        }
        out.append({"point_dir": item["point_dir"], "name": item["name"], "cells": cells})
    return out


def _outer_start(step: int) -> int:
    return 0 if step < 0 else 5


def _inner_start(step: int) -> int:
    return 4 if step < 0 else 1


def _draw_block(out: np.ndarray, r0: int, c0: int, color: int, solid: bool) -> None:
    if solid:
        out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = color
        return
    out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = color
    out[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3] = BG


def _parse_group(grid: np.ndarray, coords: np.ndarray) -> dict:
    colors = {int(grid[r, c]) for r, c in coords}
    assert BG not in colors
    assert GREY in colors and YELLOW in colors

    grey_cells = [(int(r), int(c)) for r, c in coords if int(grid[r, c]) == GREY]
    yellow_cells = {(int(r), int(c)) for r, c in coords if int(grid[r, c]) == YELLOW}
    fill_colors = sorted(colors - {GREY, YELLOW})

    assert len(grey_cells) == 1
    assert len(yellow_cells) == 3
    assert len(fill_colors) == 1

    fill = int(fill_colors[0])
    marker_r, marker_c = grey_cells[0]
    match: tuple[int, int] | None = None
    for dr, dc in DIAGONALS:
        fill_cells = {
            (marker_r + dr, marker_c),
            (marker_r, marker_c + dc),
            (marker_r + dr, marker_c + dc),
        }
        if all(_in_bounds(grid.shape, r, c) and int(grid[r, c]) == fill for r, c in fill_cells):
            assert match is None
            match = (dr, dc)
    assert match is not None
    dr, dc = match

    square_r0, square_c0 = _square_top_left(marker_r, marker_c, dr, dc)
    hat_matches = [
        pattern for pattern in _hat_patterns(square_r0, square_c0)
        if pattern["cells"] == yellow_cells
    ]
    assert len(hat_matches) == 1
    point_dir = hat_matches[0]["point_dir"]

    if point_dir == (dr, dc):
        solid_kind = OUTER
    else:
        assert point_dir == (-dr, -dc)
        solid_kind = INNER

    return {
        "marker": (marker_r, marker_c),
        "fill": fill,
        "dr": dr,
        "dc": dc,
        "solid_kind": solid_kind,
        "yellow_cells": yellow_cells,
    }


def _render_output(shape: tuple[int, int], groups: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for group in sorted(groups, key=lambda g: g["marker"]):
        marker_r, marker_c = group["marker"]
        fill = int(group["fill"])
        dr = int(group["dr"])
        dc = int(group["dc"])
        top = marker_r - LOCAL_RADIUS
        left = marker_c - LOCAL_RADIUS
        assert 0 <= top and top + 9 <= shape[0]
        assert 0 <= left and left + 9 <= shape[1]

        outer_r = top + _outer_start(dr)
        outer_c = left + _outer_start(dc)
        inner_r = top + _inner_start(dr)
        inner_c = left + _inner_start(dc)

        outer_solid = group["solid_kind"] == OUTER
        _draw_block(out, outer_r, outer_c, fill, solid=outer_solid)
        _draw_block(out, inner_r, inner_c, fill, solid=not outer_solid)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a grey-corner diagonal cue and a yellow-notch cue to decide where two enlarged blocks go and which one is solid.

    Input structure:
    - The background is black.
    - Each disconnected clue group contains one 2x2 corner made of three same-color fill pixels and one grey marker corner.
    - The grey corner acts like the 4522001f anchor: the three fill pixels occupy the marker's south/north, east/west, and diagonal neighbor for one diagonal direction.
    - Next to that 2x2 corner sits a three-pixel yellow hat whose missing side biases one diagonal direction.

    Rule:
    - For each clue group, read the diagonal direction from the grey marker to the opposite fill corner.
    - Around that marker, treat the output as the same 9x9 local canvas used by the diagonal block-expansion task: draw one 4x4 block in the outer corner of that diagonal and another 4x4 block one step inward on the opposite side.
    - The yellow hat does not move anything; instead, its notch points to exactly one of those two diagonal blocks. The pointed block is solid, and the other is a one-pixel-thick 4x4 frame with a black 2x2 hole.
    - Remove the original grey and yellow clue pixels by producing only the enlarged blocks on black background. If enlarged blocks overlap, later groups in row-major marker order overwrite earlier ones.

    Key insight:
    - The grey corner answers where the diagonal pair of enlarged blocks belongs, while the yellow notch independently answers which of the two blocks is solid versus hollow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _components(g != BG)
    assert components
    groups = [_parse_group(g, coords) for coords in components]
    return _render_output(g.shape, groups)


def _input_cells(spec: dict) -> dict[str, set[tuple[int, int]]]:
    marker_r, marker_c = spec["marker"]
    dr = int(spec["dr"])
    dc = int(spec["dc"])
    square_r0, square_c0 = _square_top_left(marker_r, marker_c, dr, dc)
    fill_cells = {
        (marker_r + dr, marker_c),
        (marker_r, marker_c + dc),
        (marker_r + dr, marker_c + dc),
    }
    pointer_dir = (dr, dc) if spec["solid_kind"] == OUTER else (-dr, -dc)
    pattern_choices = [
        pattern for pattern in _hat_patterns(square_r0, square_c0)
        if pattern["point_dir"] == pointer_dir
    ]
    pattern = random.choice(pattern_choices)
    return {
        "marker": {(marker_r, marker_c)},
        "fill": fill_cells,
        "yellow": set(pattern["cells"]),
    }


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        cells = _input_cells(spec)
        marker_r, marker_c = next(iter(cells["marker"]))
        grid[marker_r, marker_c] = GREY
        for r, c in cells["fill"]:
            grid[r, c] = int(spec["fill"])
        for r, c in cells["yellow"]:
            grid[r, c] = YELLOW
    return grid


def _output_bbox(marker: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = marker
    return r - LOCAL_RADIUS, c - LOCAL_RADIUS, r + LOCAL_RADIUS, c + LOCAL_RADIUS


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _cells_touch(existing: set[tuple[int, int]], new_cells: set[tuple[int, int]]) -> bool:
    forbidden = set(existing)
    for r, c in existing:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                forbidden.add((nr, nc))
    return any(cell in forbidden for cell in new_cells)


def _spec_signature(spec: dict) -> tuple:
    return (spec["marker"], int(spec["fill"]), int(spec["dr"]), int(spec["dc"]), spec["solid_kind"])


def _random_specs() -> tuple[tuple[int, int], list[dict]] | None:
    group_count = random.choice((2, 3))
    min_side = 18 if group_count == 2 else 23
    shape = (random.randint(min_side, 30), random.randint(min_side, 30))

    fills = random.sample(FILL_COLORS, group_count)
    directions = random.sample(list(DIAGONALS), group_count)
    solids = [INNER, OUTER] if group_count == 2 else [INNER, OUTER, random.choice((INNER, OUTER))]
    random.shuffle(solids)

    specs: list[dict] = []
    occupied_boxes: list[tuple[int, int, int, int]] = []
    occupied_input: set[tuple[int, int]] = set()

    for fill, direction, solid_kind in zip(fills, directions, solids):
        dr, dc = direction
        placed = False
        for _ in range(120):
            marker = (
                random.randint(LOCAL_RADIUS, shape[0] - LOCAL_RADIUS - 1),
                random.randint(LOCAL_RADIUS, shape[1] - LOCAL_RADIUS - 1),
            )
            box = _output_bbox(marker)
            if any(_boxes_overlap(box, old) for old in occupied_boxes):
                continue
            spec = {
                "marker": marker,
                "fill": fill,
                "dr": dr,
                "dc": dc,
                "solid_kind": solid_kind,
            }
            parts = _input_cells(spec)
            new_cells = parts["marker"] | parts["fill"] | parts["yellow"]
            if not all(_in_bounds(shape, r, c) for r, c in new_cells):
                continue
            if _cells_touch(occupied_input, new_cells):
                continue
            specs.append(spec)
            occupied_boxes.append(box)
            occupied_input |= new_cells
            placed = True
            break
        if not placed:
            return None

    if len({(spec["dr"], spec["dc"]) for spec in specs}) < 2:
        return None
    if INNER not in {spec["solid_kind"] for spec in specs} or OUTER not in {spec["solid_kind"] for spec in specs}:
        return None
    return shape, specs


def _validate_input(grid: np.ndarray, specs: list[dict]) -> bool:
    try:
        parsed = [_parse_group(grid, comp) for comp in _components(grid != BG)]
    except AssertionError:
        return False
    return sorted(_spec_signature(spec) for spec in specs) == sorted(_spec_signature(spec) for spec in parsed)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        generated = _random_specs()
        if generated is None:
            continue
        shape, specs = generated
        inp = _render_input(shape, specs)
        if not _validate_input(inp, specs):
            continue
        expected = _render_output(shape, specs)
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        if np.array_equal(inp, actual):
            continue
        return inp
    raise RuntimeError("failed to generate a valid input for puzzle 6")
