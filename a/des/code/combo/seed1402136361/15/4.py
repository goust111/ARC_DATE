from __future__ import annotations

import random

import numpy as np


CORNER_DIRS = {
    "tl": (-1, -1),
    "tr": (-1, 1),
    "bl": (1, -1),
    "br": (1, 1),
}

GRID_HEIGHT_RANGE = (12, 24)
GRID_WIDTH_RANGE = (12, 24)
FRAME_HEIGHT_RANGE = (3, 8)
FRAME_WIDTH_RANGE = (3, 8)
FRAME_BBOX_GAP = 1
FRAME_COUNT_CAP = 4
MIN_RAY_LENGTH = 2
PLACEMENT_TRIES = 300
GENERATION_TRIES = 400
PALETTE = tuple(range(10))


def _background_color(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r0, c0 in np.argwhere(mask):
        r0 = int(r0)
        c0 = int(c0)
        if seen[r0, c0]:
            continue

        stack = [(r0, c0)]
        seen[r0, c0] = True
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        comps.append(np.asarray(cells, dtype=int))

    return comps


def _frame_from_component(coords: np.ndarray) -> dict[str, object]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, c0 = int(rows.min()), int(cols.min())
    r1, c1 = int(rows.max()), int(cols.max())
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height >= 2 and width >= 2, "frame bbox must be at least 2x2"

    corners = {
        "tl": (r0, c0),
        "tr": (r0, c1),
        "bl": (r1, c0),
        "br": (r1, c1),
    }
    cells = {tuple(map(int, cell)) for cell in coords}
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))

    missing = [name for name, cell in corners.items() if cell not in cells]
    assert len(missing) == 1, "expected exactly one missing frame corner"
    missing_name = missing[0]
    assert cells == expected - {corners[missing_name]}, "component must be a rectangular frame with one missing corner"

    return {
        "bbox": (r0, c0, r1, c1),
        "missing_corner": missing_name,
        "missing_cell": corners[missing_name],
        "frame_cells": cells,
    }


def _extract_frames(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(g == color):
            frame = _frame_from_component(comp)
            frame["color"] = color
            frames.append(frame)
    assert frames, "expected at least one colored frame"
    return frames


def _paint_rays(
    base: np.ndarray,
    frames: list[dict[str, object]],
    bg: int,
) -> np.ndarray:
    out = base.copy()
    h, w = base.shape

    for frame in frames:
        color = int(frame["color"])
        missing_corner = str(frame["missing_corner"])
        mr, mc = frame["missing_cell"]
        dr, dc = CORNER_DIRS[missing_corner]
        r, c = mr + dr, mc + dc
        while 0 <= r < h and 0 <= c < w:
            if int(base[r, c]) == bg:
                prior = int(out[r, c])
                assert prior in (bg, color), "overlapping rays must not disagree on color"
                out[r, c] = color
            r += dr
            c += dc

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat each missing frame corner as a direction signal and cast an outward diagonal ray.

    Input structure:
    - The dominant color is the background.
    - Each foreground object is a 1-pixel-thick rectangular frame.
    - Every frame has exactly one missing corner pixel.

    Rule:
    - Identify the missing corner of each frame.
    - Cast a 45-degree ray outward from that corner, away from the frame center.
    - Paint only background cells; keep every frame cell, including the missing corner, unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    frames = _extract_frames(g, bg)
    return _paint_rays(g, frames, bg)


def _frame_cells(
    r0: int,
    c0: int,
    height: int,
    width: int,
    missing_corner: str,
) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int, int, int]]:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    corners = {
        "tl": (r0, c0),
        "tr": (r0, c1),
        "bl": (r1, c0),
        "br": (r1, c1),
    }

    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    cells.remove(corners[missing_corner])
    return cells, corners[missing_corner], (r0, c0, r1, c1)


def _ray_cells(
    height: int,
    width: int,
    missing_cell: tuple[int, int],
    missing_corner: str,
) -> list[tuple[int, int]]:
    dr, dc = CORNER_DIRS[missing_corner]
    r, c = missing_cell[0] + dr, missing_cell[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < height and 0 <= c < width:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _boxes_too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _sample_frame_count(height: int, width: int) -> int:
    area_cap = max(1, height * width // 140 + 1)
    return random.randint(1, min(FRAME_COUNT_CAP, area_cap))


def _render_input(
    height: int,
    width: int,
    bg: int,
    frames: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for frame in frames:
        color = int(frame["color"])
        for r, c in frame["frame_cells"]:
            grid[r, c] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    max_frame_h = min(FRAME_HEIGHT_RANGE[1], height - 3)
    max_frame_w = min(FRAME_WIDTH_RANGE[1], width - 3)
    if max_frame_h < FRAME_HEIGHT_RANGE[0] or max_frame_w < FRAME_WIDTH_RANGE[0]:
        return None

    bg = random.choice(PALETTE)
    colors = [color for color in PALETTE if color != bg]
    frame_count = _sample_frame_count(height, width)
    frame_colors = random.sample(colors, frame_count)

    size_options = [
        (fh, fw)
        for fh in range(FRAME_HEIGHT_RANGE[0], max_frame_h + 1)
        for fw in range(FRAME_WIDTH_RANGE[0], max_frame_w + 1)
    ]
    frames: list[dict[str, object]] = []
    boxes: list[tuple[int, int, int, int]] = []
    reserved: set[tuple[int, int]] = set()

    for color in frame_colors:
        placed = False
        for _ in range(PLACEMENT_TRIES):
            frame_h, frame_w = random.choice(size_options)
            r0 = random.randint(1, height - frame_h - 2)
            c0 = random.randint(1, width - frame_w - 2)
            missing_corner = random.choice(tuple(CORNER_DIRS))

            frame_cells, missing_cell, bbox = _frame_cells(r0, c0, frame_h, frame_w, missing_corner)
            if any(_boxes_too_close(bbox, other, FRAME_BBOX_GAP) for other in boxes):
                continue

            ray = _ray_cells(height, width, missing_cell, missing_corner)
            if len(ray) < MIN_RAY_LENGTH:
                continue

            footprint = frame_cells | set(ray) | {missing_cell}
            if footprint & reserved:
                continue

            frame = {
                "bbox": bbox,
                "color": color,
                "missing_corner": missing_corner,
                "missing_cell": missing_cell,
                "frame_cells": frame_cells,
                "ray_cells": ray,
            }
            frames.append(frame)
            boxes.append(bbox)
            reserved |= footprint
            placed = True
            break

        if not placed:
            return None

    inp = _render_input(height, width, bg, frames)
    out = _paint_rays(inp, frames, bg)

    values, counts = np.unique(inp, return_counts=True)
    bg_count = int(counts[np.where(values == bg)[0][0]])
    if any(int(count) >= bg_count for value, count in zip(values, counts) if int(value) != bg):
        return None

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid missing-corner frame puzzle input")
