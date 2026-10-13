import random
from typing import Iterable

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
SIDE_TO_STEP = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GRID_H_RANGE = (12, 30)
GRID_W_RANGE = (12, 30)
RECT_COUNT_OPTIONS = (1, 2, 3, 4)
RECT_COUNT_WEIGHTS = (2, 4, 3, 1)
RECT_H_RANGE = (2, 7)
RECT_W_RANGE = (2, 7)
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250


def _edge_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _ray_cells(start: tuple[int, int], step: tuple[int, int], length: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = start
    dr, dc = step
    cells: list[tuple[int, int]] = []
    for _ in range(length):
        if not (0 <= r < h and 0 <= c < w):
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _buffer_cells(cells: Iterable[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _analyze_color_object(grid: np.ndarray, color: int) -> dict:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, f"missing object for color {color}"

    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    box = grid[r0 : r1 + 1, c0 : c1 + 1]
    box_h, box_w = box.shape

    assert np.count_nonzero(box == color) == coords.shape[0], "object color must stay inside one bbox"
    assert np.all((box == color) | (box == BG)), "bbox may only contain the object color and the black gap"
    assert int(np.count_nonzero(box == BG)) == 1, "expected exactly one missing edge pixel"
    assert coords.shape[0] == box_h * box_w - 1, "object must be a full rectangle minus one pixel"

    gap_r_rel, gap_c_rel = map(int, np.argwhere(box == BG)[0])
    on_top = gap_r_rel == 0
    on_bottom = gap_r_rel == box_h - 1
    on_left = gap_c_rel == 0
    on_right = gap_c_rel == box_w - 1
    assert sum((on_top, on_bottom, on_left, on_right)) == 1, "gap must lie on exactly one side, not a corner"

    if on_top:
        step = SIDE_TO_STEP["top"]
    elif on_bottom:
        step = SIDE_TO_STEP["bottom"]
    elif on_left:
        step = SIDE_TO_STEP["left"]
    else:
        step = SIDE_TO_STEP["right"]

    gap = (r0 + gap_r_rel, c0 + gap_c_rel)
    edge_pixels = int(np.count_nonzero(box[_edge_mask(box_h, box_w)] == color))
    assert edge_pixels >= 1

    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "step": step,
        "length": edge_pixels,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a same-color straight ray outward from the unique edge gap of each almost-solid rectangle.

    Input structure:
    - The background is black.
    - Each non-black color forms exactly one filled axis-aligned rectangle whose bbox contains exactly one black gap on a non-corner edge cell.
    - Different rectangles use different colors and do not touch.

    Rule:
    - Keep every original colored rectangle pixel unchanged.
    - For each rectangle, find the unique black gap on its border and use that side to determine the outward ray direction.
    - Draw a same-color straight ray starting at the gap cell and continuing outward from the rectangle.
    - The ray length equals the count of remaining colored border pixels of that rectangle, and it is truncated only by the grid boundary.

    Key insight:
    - The single missing border pixel simultaneously encodes the ray's launch position and its direction, while the rectangle's own border count determines the ray length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        info = _analyze_color_object(g, color)
        for r, c in _ray_cells(info["gap"], info["step"], info["length"], g.shape):
            out[r, c] = color
    return out


def _make_spec(shape: tuple[int, int], blocked: set[tuple[int, int]]) -> dict | None:
    h, w = shape
    for _ in range(PLACEMENT_ATTEMPTS):
        side = random.choice(tuple(SIDE_TO_STEP))
        rect_h = random.randint(*RECT_H_RANGE)
        rect_w = random.randint(*RECT_W_RANGE)
        if side in ("top", "bottom") and rect_w < 3:
            continue
        if side in ("left", "right") and rect_h < 3:
            continue

        ray_len = 2 * rect_h + 2 * rect_w - 5
        fully_visible = random.random() < 0.6

        if side == "top":
            min_r0 = ray_len - 1 if fully_visible else 1
            max_r0 = h - rect_h
            if min_r0 > max_r0:
                continue
            r0 = random.randint(min_r0, max_r0)
            c0 = random.randint(0, w - rect_w)
            gap = (r0, random.randint(c0 + 1, c0 + rect_w - 2))
        elif side == "bottom":
            min_r0 = 0
            max_r0 = h - rect_h - (ray_len - 1 if fully_visible else 1)
            if min_r0 > max_r0:
                continue
            r0 = random.randint(min_r0, max_r0)
            c0 = random.randint(0, w - rect_w)
            gap = (r0 + rect_h - 1, random.randint(c0 + 1, c0 + rect_w - 2))
        elif side == "left":
            min_c0 = ray_len - 1 if fully_visible else 1
            max_c0 = w - rect_w
            if min_c0 > max_c0:
                continue
            r0 = random.randint(0, h - rect_h)
            c0 = random.randint(min_c0, max_c0)
            gap = (random.randint(r0 + 1, r0 + rect_h - 2), c0)
        else:
            min_c0 = 0
            max_c0 = w - rect_w - (ray_len - 1 if fully_visible else 1)
            if min_c0 > max_c0:
                continue
            r0 = random.randint(0, h - rect_h)
            c0 = random.randint(min_c0, max_c0)
            gap = (random.randint(r0 + 1, r0 + rect_h - 2), c0 + rect_w - 1)

        rect_cells = {
            (r, c)
            for r in range(r0, r0 + rect_h)
            for c in range(c0, c0 + rect_w)
            if (r, c) != gap
        }
        ray_cells = _ray_cells(gap, SIDE_TO_STEP[side], ray_len, shape)
        if len(ray_cells) < 2:
            continue

        colored_output = rect_cells | set(ray_cells)
        if colored_output & blocked:
            continue

        return {
            "bbox": (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1),
            "gap": gap,
            "side": side,
            "step": SIDE_TO_STEP[side],
            "length": ray_len,
            "rect_cells": rect_cells,
            "ray_cells": ray_cells,
            "output_cells": colored_output,
        }
    return None


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        for r, c in spec["rect_cells"]:
            grid[r, c] = spec["color"]
    return grid


def _render_output(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = _render_input(shape, specs)
    for spec in specs:
        for r, c in spec["ray_cells"]:
            out[r, c] = spec["color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_H_RANGE),
        random.randint(*GRID_W_RANGE),
    )
    rect_count = random.choices(RECT_COUNT_OPTIONS, weights=RECT_COUNT_WEIGHTS, k=1)[0]
    colors = random.sample(PALETTE, rect_count)

    specs: list[dict] = []
    blocked: set[tuple[int, int]] = set()
    for color in colors:
        spec = _make_spec(shape, blocked)
        if spec is None:
            return None
        spec["color"] = color
        specs.append(spec)
        blocked |= _buffer_cells(spec["output_cells"], shape)

    inp = _render_input(shape, specs)
    expected = _render_output(shape, specs)
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
