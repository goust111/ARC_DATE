from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, find_components


BG = Color.BLACK
PALETTE = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY)
GEN_RETRY_LIMIT = 400


def _is_line_component(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    area = int(component["area"])
    return (r0 == r1 and area == c1 - c0 + 1) or (c0 == c1 and area == r1 - r0 + 1)


def _extract_segments(grid: np.ndarray) -> list[tuple[int, int, int]]:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert comps, "expected at least one foreground segment"

    segments: list[tuple[int, int, int]] = []
    top_rows: set[int] = set()
    for comp in comps:
        assert _is_line_component(comp), "each component must be a straight single-pixel-wide segment"
        top_row = int(comp["bbox"][0])
        assert top_row not in top_rows, "segment highest rows must be unique"
        top_rows.add(top_row)
        segments.append((top_row, int(comp["color"]), int(comp["area"])))
    segments.sort(key=lambda item: item[0])
    return segments


def main(input_grid: np.ndarray) -> np.ndarray:
    """Sort the line segments by topmost row and expand them into a stacked left-aligned wedge tower.

    Input structure:
    - The grid has a black background and 3 to 5 separated colored line segments.
    - Each segment is a monochromatic horizontal or vertical run with single-pixel thickness.
    - Different segments have different highest occupied rows, and their lengths range from 3 to 8.

    Rule:
    - For each segment, record its color, total pixel count, and highest occupied row.
    - Sort the segments from smaller highest row to larger highest row.
    - Build an output grid with the same width as the input; for each sorted segment, draw a left-aligned wedge whose first row has the full segment length and whose width decreases by 1 each row until 1, then stack the next wedge immediately below it.

    Key insight:
    - Each segment contributes two independent pieces of information: vertical position determines stack order, while segment length determines the wedge's full geometry.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    segments = _extract_segments(grid)
    width = int(grid.shape[1])
    total_height = sum(length for _top, _color, length in segments)
    assert total_height <= 30, "output height exceeds ARC limit"

    out = np.full((total_height, width), BG, dtype=int)
    cursor = 0
    for _top, color, length in segments:
        assert 1 <= length <= width <= 30, "segment length must fit within the output width"
        for dr in range(length):
            out[cursor + dr, : length - dr] = color
        cursor += length
    return out


def _line_cells(length: int, orientation: str, top: int, left: int) -> list[tuple[int, int]]:
    if orientation == "h":
        return [(top, left + dc) for dc in range(length)]
    return [(top + dr, left) for dr in range(length)]


def _expand_forbidden(cells: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    blocked: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    blocked.add((rr, cc))
    return blocked


def _orientation_plan(count: int) -> list[str]:
    roll = random.random()
    if roll < 0.4:
        orientations = [random.choice(("h", "v")) for _ in range(count)]
        if len(set(orientations)) == 1:
            orientations[random.randrange(count)] = "h" if orientations[0] == "v" else "v"
        return orientations
    if roll < 0.7:
        orientations = ["h" if random.random() < 0.75 else "v" for _ in range(count)]
        if "v" not in orientations:
            orientations[random.randrange(count)] = "v"
        return orientations
    orientations = ["v" if random.random() < 0.75 else "h" for _ in range(count)]
    if "h" not in orientations:
        orientations[random.randrange(count)] = "h"
    return orientations


def _sample_lengths(count: int) -> list[int]:
    for _ in range(200):
        lengths = [random.randint(3, 8) for _ in range(count)]
        if sum(lengths) <= 30:
            return lengths
    raise RuntimeError("failed to sample valid segment lengths")


def _range_with_border(limit: int, span: int) -> range:
    if limit - span >= 2:
        return range(1, limit - span)
    return range(0, limit - span + 1)


def _place_segments(
    height: int,
    width: int,
    specs: list[tuple[int, int, str]],
) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    forbidden: set[tuple[int, int]] = set()
    used_top_rows: set[int] = set()

    order = list(range(len(specs)))
    random.shuffle(order)
    order.sort(key=lambda idx: specs[idx][1], reverse=True)

    for idx in order:
        color, length, orientation = specs[idx]
        patch_h, patch_w = (1, length) if orientation == "h" else (length, 1)
        candidates: list[tuple[int, int]] = []
        for top in _range_with_border(height, patch_h):
            if top in used_top_rows:
                continue
            for left in _range_with_border(width, patch_w):
                cells = _line_cells(length, orientation, top, left)
                if any(cell in forbidden for cell in cells):
                    continue
                candidates.append((top, left))
        if not candidates:
            return None

        top, left = random.choice(candidates)
        cells = _line_cells(length, orientation, top, left)
        for r, c in cells:
            grid[r, c] = color
        forbidden |= _expand_forbidden(cells, grid.shape)
        used_top_rows.add(top)

    return grid


def _generate_attempt() -> np.ndarray | None:
    count = random.randint(3, 5)
    colors = random.sample(PALETTE, count)
    lengths = _sample_lengths(count)
    orientations = _orientation_plan(count)

    max_vertical = max(length for length, orient in zip(lengths, orientations) if orient == "v") if "v" in orientations else 1
    max_horizontal = max(length for length, orient in zip(lengths, orientations) if orient == "h") if "h" in orientations else 1
    height = random.randint(max(12, max_vertical + 3), min(20, 30))
    width = random.randint(max(16, max_horizontal + 3), min(26, 30))

    specs = list(zip(colors, lengths, orientations))
    grid = _place_segments(height, width, specs)
    if grid is None:
        return None

    segments = _extract_segments(grid)
    if len(segments) != count:
        return None
    if sum(length for _top, _color, length in segments) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
