from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GENERATION_ATTEMPTS = 400
BLOCK_PLACEMENT_ATTEMPTS = 300
DISTRACTOR_PLACEMENT_ATTEMPTS = 200

GRID_SIDE_RANGE = (18, 28)
OBJECT_COUNT_RANGE = (2, 4)
RECT_SIDE_RANGE = (3, 6)
DISTRACTOR_COUNT_MAX = 3


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords_arr = np.asarray(coords, dtype=int)
            r0 = int(coords_arr[:, 0].min())
            c0 = int(coords_arr[:, 1].min())
            r1 = int(coords_arr[:, 0].max())
            c1 = int(coords_arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "area": int(coords_arr.shape[0]),
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return components


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _sweep_cells(r0: int, c0: int, h: int, w: int, dr: int, dc: int) -> set[tuple[int, int]]:
    rr0 = min(r0, r0 + dr)
    cc0 = min(c0, c0 + dc)
    rr1 = max(r0 + h - 1, r0 + dr + h - 1)
    cc1 = max(c0 + w - 1, c0 + dc + w - 1)
    return _rect_cells(rr0, cc0, rr1 - rr0 + 1, cc1 - cc0 + 1)


def _touches_occupied(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    for r, c in cells:
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                if (nr, nc) in occupied:
                    return True
    return False


def _cell_touches_any(cell: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    r, c = cell
    for rr, cc in cells:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each holed rectangle onto its same-colored key pixel, swallowing the key and sealing the hole.

    Input structure:
    - The grid has a black background with several monochrome rectangular blocks.
    - Each block is solid except for one black interior cell acting as a lockhole.
    - The remaining foreground components are isolated single pixels: one same-colored key for each block plus unmatched distractors.

    Rule:
    - For each non-singleton component, identify its bounding rectangle and unique interior background hole.
    - Match it to the same-colored singleton that lies in the hole's row or column, then translate the whole rectangle until the hole lands on that pixel.
    - Clear the old block location, consume the matched key, and draw the destination rectangle as fully solid while leaving distractor pixels unchanged.

    Key insight:
    - The motion vector is not marked directly; it is implied by the only same-colored isolated pixel that can be captured by aligning the hole with it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _find_components(g)
    singletons_by_color: dict[int, list[tuple[int, int]]] = {}
    blocks = []

    for comp in components:
        color = int(comp["color"])
        if int(comp["area"]) == 1:
            coord = tuple(map(int, comp["coords"][0]))
            singletons_by_color.setdefault(color, []).append(coord)
            continue

        r0, c0, r1, c1 = comp["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        hole_coords = np.argwhere(patch == BACKGROUND)
        assert hole_coords.shape[0] == 1, "each block must have exactly one hole"
        hole_r, hole_c = map(int, hole_coords[0])
        assert 0 < hole_r < patch.shape[0] - 1 and 0 < hole_c < patch.shape[1] - 1, "hole must be interior"
        assert np.all((patch == color) | (patch == BACKGROUND)), "block bbox must be monochrome except for its hole"
        assert int(comp["area"]) == patch.size - 1, "block must be a full rectangle minus one cell"

        blocks.append(
            {
                "color": color,
                "coords": comp["coords"],
                "bbox": comp["bbox"],
                "hole": (r0 + hole_r, c0 + hole_c),
            }
        )

    out = g.copy()
    used_keys: set[tuple[int, int]] = set()

    for block in blocks:
        color = int(block["color"])
        hole_r, hole_c = block["hole"]
        candidates = [
            key
            for key in singletons_by_color.get(color, [])
            if key not in used_keys and ((key[0] == hole_r) ^ (key[1] == hole_c))
        ]
        assert len(candidates) == 1, "each block must have one uniquely aligned same-colored key"
        key_r, key_c = candidates[0]
        used_keys.add((key_r, key_c))

        dr = key_r - hole_r
        dc = key_c - hole_c
        assert (dr == 0) ^ (dc == 0), "key must align horizontally or vertically"

        coords = block["coords"]
        out[coords[:, 0], coords[:, 1]] = BACKGROUND
        out[key_r, key_c] = BACKGROUND

        r0, c0, r1, c1 = block["bbox"]
        out[r0 + dr : r1 + dr + 1, c0 + dc : c1 + dc + 1] = color

    return out


def _sample_axes(count: int) -> list[str]:
    if count == 1:
        return [random.choice(("h", "v"))]
    axes = ["h", "v"]
    axes.extend(random.choice(("h", "v")) for _ in range(count - 2))
    random.shuffle(axes)
    return axes


def _gen_block_spec(height: int, width: int, color: int, axis: str) -> dict | None:
    h = random.randint(*RECT_SIDE_RANGE)
    w = random.randint(*RECT_SIDE_RANGE)
    hole_r = random.randint(1, h - 2)
    hole_c = random.randint(1, w - 2)
    sign = random.choice((-1, 1))

    if axis == "h":
        min_shift = (w - 1 - hole_c) + 2 if sign > 0 else hole_c + 2
        max_shift = width - w
        if min_shift > max_shift:
            return None
        shift = random.randint(min_shift, max_shift)
        dr, dc = 0, sign * shift
        r0 = random.randint(0, height - h)
        if dc > 0:
            c0 = random.randint(0, width - w - dc)
        else:
            c0 = random.randint(-dc, width - w)
    else:
        min_shift = (h - 1 - hole_r) + 2 if sign > 0 else hole_r + 2
        max_shift = height - h
        if min_shift > max_shift:
            return None
        shift = random.randint(min_shift, max_shift)
        dr, dc = sign * shift, 0
        c0 = random.randint(0, width - w)
        if dr > 0:
            r0 = random.randint(0, height - h - dr)
        else:
            r0 = random.randint(-dr, height - h)

    hole = (r0 + hole_r, c0 + hole_c)
    key = (hole[0] + dr, hole[1] + dc)
    rect = _rect_cells(r0, c0, h, w)
    block_cells = rect - {hole}
    final_rect = _rect_cells(r0 + dr, c0 + dc, h, w)
    sweep = _sweep_cells(r0, c0, h, w, dr, dc)

    return {
        "color": color,
        "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
        "hole": hole,
        "key": key,
        "block_cells": block_cells,
        "final_rect": final_rect,
        "sweep": sweep,
    }


def _paint_input(shape: tuple[int, int], blocks: list[dict], distractors: list[tuple[int, tuple[int, int]]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for block in blocks:
        color = int(block["color"])
        for r, c in block["block_cells"]:
            grid[r, c] = color
        key_r, key_c = block["key"]
        grid[key_r, key_c] = color
    for color, (r, c) in distractors:
        grid[r, c] = int(color)
    return grid


def _paint_output(shape: tuple[int, int], blocks: list[dict], distractors: list[tuple[int, tuple[int, int]]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for block in blocks:
        color = int(block["color"])
        for r, c in block["final_rect"]:
            grid[r, c] = color
    for color, (r, c) in distractors:
        grid[r, c] = int(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    block_count = random.randint(*OBJECT_COUNT_RANGE)

    colors = random.sample(PALETTE, block_count)
    axes = _sample_axes(block_count)

    blocks: list[dict] = []
    input_occupied: set[tuple[int, int]] = set()
    output_occupied: set[tuple[int, int]] = set()
    sweep_occupied: set[tuple[int, int]] = set()

    for color, axis in zip(colors, axes):
        placed = False
        for _ in range(BLOCK_PLACEMENT_ATTEMPTS):
            block = _gen_block_spec(height, width, color, axis)
            if block is None:
                continue

            input_cells = set(block["block_cells"])
            input_cells.add(block["key"])
            if _touches_occupied(input_cells, input_occupied, (height, width)):
                continue
            if _cell_touches_any(block["key"], block["block_cells"]):
                continue
            if input_cells & sweep_occupied:
                continue
            if block["sweep"] & input_occupied:
                continue
            if block["sweep"] & sweep_occupied:
                continue
            if _touches_occupied(block["final_rect"], output_occupied, (height, width)):
                continue

            blocks.append(block)
            input_occupied |= input_cells
            output_occupied |= block["final_rect"]
            sweep_occupied |= block["sweep"]
            placed = True
            break

        if not placed:
            return None

    remaining_colors = [color for color in PALETTE if color not in colors]
    max_distractors = min(DISTRACTOR_COUNT_MAX, len(remaining_colors))
    if max_distractors == 0 or random.random() < 0.25:
        distractor_count = 0
    else:
        distractor_count = random.randint(1, max_distractors)
    distractor_colors = random.sample(remaining_colors, distractor_count)

    distractors: list[tuple[int, tuple[int, int]]] = []
    for color in distractor_colors:
        placed = False
        for _ in range(DISTRACTOR_PLACEMENT_ATTEMPTS):
            cell = (random.randrange(height), random.randrange(width))
            if cell in sweep_occupied:
                continue
            if _touches_occupied({cell}, input_occupied, (height, width)):
                continue
            if _touches_occupied({cell}, output_occupied, (height, width)):
                continue

            distractors.append((color, cell))
            input_occupied.add(cell)
            output_occupied.add(cell)
            placed = True
            break

        if not placed:
            return None

    inp = _paint_input((height, width), blocks, distractors)
    expected = _paint_output((height, width), blocks, distractors)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
