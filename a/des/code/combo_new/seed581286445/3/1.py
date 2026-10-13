from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
DIRECTION_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
FRAME_COUNT_CHOICES = (1, 2, 3)
PALETTE_EDGES = ("top", "bottom")
FRAME_SIZE_OPTIONS = (5, 7, 9)
MARKER_COUNT_CAP = 6
FRAME_RETRIES = 240
GENERATION_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(max(counts, key=counts.get))


def _contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _frame_gap_direction(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    gap: tuple[int, int],
) -> str | None:
    gap_r, gap_c = gap
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if width % 2 == 1:
        mid_c = (c0 + c1) // 2
        if gap == (r0, mid_c):
            return "up"
        if gap == (r1, mid_c):
            return "down"
    if height % 2 == 1:
        mid_r = (r0 + r1) // 2
        if gap == (mid_r, c0):
            return "left"
        if gap == (mid_r, c1):
            return "right"
    return None


def _find_palette(
    grid: np.ndarray,
    bg: int,
) -> tuple[tuple[int, int], tuple[int, int], dict[int, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for row0 in (0, h - 2):
        if row0 < 0 or row0 + 1 >= h:
            continue
        both_rows = (grid[row0] != bg) & (grid[row0 + 1] != bg)
        for c0, c1 in _contiguous_segments(both_rows):
            if np.all(grid[row0, c0 : c1 + 1] != bg) and np.all(grid[row0 + 1, c0 : c1 + 1] != bg):
                candidates.append((row0, c0, c1))

    assert len(candidates) == 1, "expected exactly one edge palette"
    row0, c0, c1 = candidates[0]
    mapping: dict[int, int] = {}
    for border_color, ray_color in zip(grid[row0, c0 : c1 + 1], grid[row0 + 1, c0 : c1 + 1]):
        border_color = int(border_color)
        ray_color = int(ray_color)
        if border_color in mapping:
            assert mapping[border_color] == ray_color, "palette mapping must be consistent"
        mapping[border_color] = ray_color
    return (row0, row0 + 1), (c0, c1), mapping


def _extract_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    palette_rows, palette_cols, palette = _find_palette(grid, bg)
    scene = grid.copy()
    row0, row1 = palette_rows
    col0, col1 = palette_cols
    scene[row0 : row1 + 1, col0 : col1 + 1] = bg

    frames: list[dict[str, object]] = []
    for color in sorted(int(value) for value in np.unique(scene) if int(value) != bg):
        if color not in palette:
            continue
        for comp in _components(scene == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            cells = set(comp)
            perimeter = _perimeter_cells(r0, c0, r1, c1)
            if not cells <= perimeter:
                continue
            missing = perimeter - cells
            if len(missing) != 1:
                continue
            gap = next(iter(missing))
            direction = _frame_gap_direction(r0, c0, r1, c1, gap)
            if direction is None:
                continue

            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            marker_mask = inner != bg
            if not np.any(marker_mask):
                continue
            marker_colors = np.unique(inner[marker_mask])
            if marker_colors.size != 1 or int(marker_colors[0]) == color:
                continue

            frames.append(
                {
                    "border_color": color,
                    "ray_color": palette[color],
                    "gap": gap,
                    "direction": direction,
                    "marker_count": int(np.count_nonzero(marker_mask)),
                }
            )
    assert frames, "expected at least one valid frame"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a color-mapped ray outward through each frame gap, with length set by the number of interior markers.

    Input structure:
    - The background is a dominant black field.
    - One edge contains a contiguous 2-row palette whose top row lists frame colors and whose bottom row lists their ray colors.
    - Away from the palette, one or more single-color rectangular frame outlines each miss exactly one midpoint border pixel, and each frame encloses scattered marker pixels of one other color.

    Rule:
    - Read the palette as a frame-color to ray-color mapping.
    - For every frame, find which side contains the missing midpoint border pixel, then count all non-background marker pixels inside that frame.
    - Starting at the missing border cell itself, draw that many cells outward in the gap direction using the ray color paired with the frame color.

    Key insight:
    - The output depends on combining three independent cues: the palette chooses the ray color, the gap side chooses the growth direction, and the interior marker count chooses the ray length.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for frame in _extract_frames(g, bg):
        gap_r, gap_c = frame["gap"]
        dr, dc = DIRECTION_VECTORS[frame["direction"]]
        ray_color = int(frame["ray_color"])
        marker_count = int(frame["marker_count"])
        for step in range(marker_count):
            rr = gap_r + dr * step
            cc = gap_c + dc * step
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "ray must stay in bounds"
            assert int(g[rr, cc]) == bg, "ray path must start on background cells in the input"
            out[rr, cc] = ray_color

    return out


def _cells_with_radius(
    cells: list[tuple[int, int]] | set[tuple[int, int]],
    shape: tuple[int, int],
    radius: int,
) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - radius), min(h - 1, r + radius) + 1):
            for cc in range(max(0, c - radius), min(w - 1, c + radius) + 1):
                expanded.add((rr, cc))
    return expanded


def _path_cells(gap: tuple[int, int], direction: str, length: int) -> list[tuple[int, int]]:
    dr, dc = DIRECTION_VECTORS[direction]
    return [(gap[0] + dr * step, gap[1] + dc * step) for step in range(length)]


def _gap_from_bbox(r0: int, c0: int, height: int, width: int, direction: str) -> tuple[int, int]:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    if direction == "up":
        return r0, (c0 + c1) // 2
    if direction == "down":
        return r1, (c0 + c1) // 2
    if direction == "left":
        return (r0 + r1) // 2, c0
    return (r0 + r1) // 2, c1


def _max_path_length(shape: tuple[int, int], gap: tuple[int, int], direction: str) -> int:
    h, w = shape
    r, c = gap
    if direction == "up":
        return r + 1
    if direction == "down":
        return h - r
    if direction == "left":
        return c + 1
    return w - c


def _draw_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    height: int,
    width: int,
    border_color: int,
    marker_color: int,
    direction: str,
    marker_cells: list[tuple[int, int]],
) -> dict[str, object]:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    gap = _gap_from_bbox(r0, c0, height, width, direction)
    border_cells = sorted(_perimeter_cells(r0, c0, r1, c1) - {gap})
    for r, c in border_cells:
        grid[r, c] = border_color
    for r, c in marker_cells:
        grid[r, c] = marker_color
    return {"gap": gap, "border_cells": border_cells}


def _candidate_clearance(
    shape: tuple[int, int],
    r0: int,
    c0: int,
    height: int,
    width: int,
    path: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    h, w = shape
    bbox_cells = {
        (r, c)
        for r in range(max(0, r0 - 1), min(h - 1, r0 + height) + 1)
        for c in range(max(0, c0 - 1), min(w - 1, c0 + width) + 1)
    }
    return bbox_cells | _cells_with_radius(path, shape, radius=1)


def _try_place_frame(
    grid: np.ndarray,
    blocked: np.ndarray,
    border_color: int,
    ray_color: int,
    marker_color: int,
    direction: str,
    rng: random.Random,
) -> dict[str, object] | None:
    h, w = grid.shape
    viable_sizes = [size for size in FRAME_SIZE_OPTIONS if size + 2 < h and size + 2 < w]
    if not viable_sizes:
        return None

    for _ in range(FRAME_RETRIES):
        height = rng.choice(viable_sizes)
        width = rng.choice(viable_sizes)
        r0 = rng.randint(1, h - height - 1)
        c0 = rng.randint(1, w - width - 1)
        gap = _gap_from_bbox(r0, c0, height, width, direction)
        max_length = min(
            _max_path_length(grid.shape, gap, direction),
            (height - 2) * (width - 2),
            MARKER_COUNT_CAP,
        )
        if max_length < 1:
            continue

        marker_count = rng.randint(1, max_length)
        path = _path_cells(gap, direction, marker_count)
        if any(not (0 <= r < h and 0 <= c < w) for r, c in path):
            continue

        clearance = _candidate_clearance(grid.shape, r0, c0, height, width, path)
        if any(blocked[r, c] for r, c in clearance):
            continue
        if any(int(grid[r, c]) != BACKGROUND for r, c in path):
            continue

        interior = [
            (r, c)
            for r in range(r0 + 1, r0 + height - 1)
            for c in range(c0 + 1, c0 + width - 1)
        ]
        marker_cells = rng.sample(interior, marker_count)
        placement = _draw_frame(
            grid,
            r0,
            c0,
            height,
            width,
            border_color,
            marker_color,
            direction,
            marker_cells,
        )
        for r, c in clearance:
            blocked[r, c] = True
        return {
            "border_color": border_color,
            "ray_color": ray_color,
            "marker_color": marker_color,
            "direction": direction,
            "marker_count": marker_count,
            "gap": placement["gap"],
        }
    return None


def _place_palette(
    grid: np.ndarray,
    blocked: np.ndarray,
    border_colors: list[int],
    ray_colors: list[int],
    edge: str,
    rng: random.Random,
) -> None:
    h, w = grid.shape
    length = len(border_colors)
    row0 = 0 if edge == "top" else h - 2
    col0 = rng.randint(0, w - length)
    order = list(range(length))
    rng.shuffle(order)
    for offset, idx in enumerate(order):
        grid[row0, col0 + offset] = border_colors[idx]
        grid[row0 + 1, col0 + offset] = ray_colors[idx]

    palette_cells = [
        (row0 + dr, col0 + dc)
        for dr in range(2)
        for dc in range(length)
    ]
    for r, c in _cells_with_radius(palette_cells, grid.shape, radius=1):
        blocked[r, c] = True


def _expected_output(grid: np.ndarray, frames: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for frame in frames:
        for r, c in _path_cells(frame["gap"], frame["direction"], int(frame["marker_count"])):
            out[r, c] = int(frame["ray_color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.choice(FRAME_COUNT_CHOICES)
    height = rng.randint(max(16, 8 + 4 * frame_count), 30)
    width = rng.randint(max(16, 8 + 4 * frame_count), 30)
    grid = np.full((height, width), BACKGROUND, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    colors = list(range(1, 10))
    rng.shuffle(colors)
    border_colors = colors[:frame_count]
    ray_colors = colors[frame_count : 2 * frame_count]
    marker_colors = colors[2 * frame_count : 3 * frame_count]
    edge = rng.choice(PALETTE_EDGES)
    _place_palette(grid, blocked, border_colors, ray_colors, edge, rng)

    if frame_count == 1:
        directions = [rng.choice(list(DIRECTION_VECTORS))]
    else:
        directions = rng.sample(list(DIRECTION_VECTORS), frame_count)

    frames: list[dict[str, object]] = []
    triplets = list(zip(border_colors, ray_colors, marker_colors, directions))
    rng.shuffle(triplets)
    for border_color, ray_color, marker_color, direction in triplets:
        placed = _try_place_frame(
            grid,
            blocked,
            border_color,
            ray_color,
            marker_color,
            direction,
            rng,
        )
        if placed is None:
            return None
        frames.append(placed)

    expected = _expected_output(grid, frames)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, grid) or not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
