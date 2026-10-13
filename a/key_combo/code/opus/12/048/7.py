from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _bbox_nonzero(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != BACKGROUND)
    if coords.size == 0:
        raise AssertionError("expected at least one non-background pixel")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    seen = {next(iter(cells))}
    stack = [next(iter(seen))]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen == cells


def _alternating_recolor(core: np.ndarray, marker_color: int) -> np.ndarray:
    out = core.copy()
    coords = np.argwhere(out != BACKGROUND)
    for index, (r, c) in enumerate(coords, start=1):
        if index % 2 == 1:
            out[int(r), int(c)] = marker_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop away the marked frame and recolor only odd-position foreground pixels to the corner-marker color.

    Input structure:
    - The background is black.
    - One rectangular one-cell frame surrounds a central region; the frame color and the four corner-marker color are distinct.
    - Inside the frame is a single-color irregular foreground pattern on black, and the pattern does not touch the central region boundary.

    Rule:
    - Crop to the bounding box of all non-black pixels and read the shared corner-marker color from the four crop corners.
    - Remove the outer one-cell frame to obtain the central region.
    - Scan the central region row by row, left to right, numbering only non-black pixels.
    - Recolor odd-numbered foreground pixels to the corner-marker color and leave even-numbered foreground pixels unchanged.

    Key insight:
    - The corner markers say which color to paint, but not every interior foreground cell is repainted: only odd positions in the row-major foreground sequence change.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    cropped = _crop_bbox(g, _bbox_nonzero(g))
    assert min(cropped.shape) >= 3, "cropped object must contain a border and an interior"

    corner_values = {
        int(cropped[0, 0]),
        int(cropped[0, -1]),
        int(cropped[-1, 0]),
        int(cropped[-1, -1]),
    }
    assert len(corner_values) == 1, "expected four matching corner markers"
    marker_color = corner_values.pop()
    assert marker_color != BACKGROUND, "corner markers must be non-background"

    core = cropped[1:-1, 1:-1]
    assert core.size > 0, "frame removal must leave a central region"
    return _alternating_recolor(core, marker_color)


def _grow_shape(height: int, width: int, area: int) -> set[tuple[int, int]]:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        frontier = sorted(
            {
                (nr, nc)
                for r, c in cells
                for nr, nc in _neighbors4(r, c)
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells
            }
        )
        cells.add(random.choice(frontier))
    return cells


def _build_core() -> tuple[np.ndarray, int, int]:
    core_h = random.randint(5, 8)
    core_w = random.randint(5, 8)
    inner_h = core_h - 2
    inner_w = core_w - 2
    max_area = min(15, inner_h * inner_w - 1)
    area = random.randint(6, max_area)

    pattern_color, frame_color, marker_color = random.sample(COLORS, 3)

    for _ in range(120):
        inner_cells = _grow_shape(inner_h, inner_w, area)
        rows = [r for r, _ in inner_cells]
        cols = [c for _, c in inner_cells]
        if max(rows) == min(rows) or max(cols) == min(cols):
            continue
        if not _is_connected(inner_cells):
            continue

        core = np.zeros((core_h, core_w), dtype=int)
        for r, c in inner_cells:
            core[r + 1, c + 1] = pattern_color

        if np.count_nonzero(core) < 6:
            continue
        if np.any(core[0, :] != BACKGROUND) or np.any(core[-1, :] != BACKGROUND):
            continue
        if np.any(core[:, 0] != BACKGROUND) or np.any(core[:, -1] != BACKGROUND):
            continue

        recolored = _alternating_recolor(core, marker_color)
        fully_recolored = core.copy()
        fully_recolored[fully_recolored == pattern_color] = marker_color
        if np.array_equal(recolored, fully_recolored):
            continue
        if pattern_color not in recolored or marker_color not in recolored:
            continue
        return core, frame_color, marker_color

    raise RuntimeError("failed to build a valid central region")


def _embed_with_frame(core: np.ndarray, frame_color: int, marker_color: int) -> np.ndarray:
    core_h, core_w = core.shape
    framed = np.full((core_h + 2, core_w + 2), frame_color, dtype=int)
    framed[1:-1, 1:-1] = core
    framed[0, 0] = marker_color
    framed[0, -1] = marker_color
    framed[-1, 0] = marker_color
    framed[-1, -1] = marker_color
    return framed


def _validate_instance(inp: np.ndarray, expected: np.ndarray, frame_color: int, marker_color: int) -> bool:
    try:
        cropped = _crop_bbox(inp, _bbox_nonzero(inp))
    except AssertionError:
        return False

    if marker_color == frame_color or marker_color == BACKGROUND or frame_color == BACKGROUND:
        return False
    if cropped.shape[0] < 3 or cropped.shape[1] < 3:
        return False
    if {int(cropped[0, 0]), int(cropped[0, -1]), int(cropped[-1, 0]), int(cropped[-1, -1])} != {marker_color}:
        return False
    if np.any(cropped[0, 1:-1] != frame_color) or np.any(cropped[-1, 1:-1] != frame_color):
        return False
    if np.any(cropped[1:-1, 0] != frame_color) or np.any(cropped[1:-1, -1] != frame_color):
        return False

    core = cropped[1:-1, 1:-1]
    if not np.array_equal(main(inp), expected):
        return False
    fg = np.argwhere(core != BACKGROUND)
    if fg.shape[0] < 6:
        return False
    if np.any(core[0, :] != BACKGROUND) or np.any(core[-1, :] != BACKGROUND):
        return False
    if np.any(core[:, 0] != BACKGROUND) or np.any(core[:, -1] != BACKGROUND):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    core, frame_color, marker_color = _build_core()
    framed = _embed_with_frame(core, frame_color, marker_color)
    expected = _alternating_recolor(core, marker_color)

    pad_h = random.randint(0, 6)
    pad_w = random.randint(0, 6)
    grid_h = framed.shape[0] + pad_h
    grid_w = framed.shape[1] + pad_w
    grid = np.zeros((grid_h, grid_w), dtype=int)

    top = random.randint(0, grid_h - framed.shape[0])
    left = random.randint(0, grid_w - framed.shape[1])
    grid[top : top + framed.shape[0], left : left + framed.shape[1]] = framed

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _validate_instance(grid, expected, frame_color, marker_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
