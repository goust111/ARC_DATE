from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_ROWS_RANGE = (14, 19)
GRID_COLS_RANGE = (14, 19)
FRAME_COUNT_RANGE = (2, 4)
FRAME_GAP = 1
MIN_NOISE_DENSITY = 0.15
MAX_NOISE_DENSITY = 0.23
MAX_GENERATION_ATTEMPTS = 400
FRAME_PLACEMENT_ATTEMPTS = 120

INTERIOR_SIZES = tuple(
    (h, w)
    for h in range(2, 6)
    for w in range(2, 6)
    if 4 <= h * w <= 15
)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not _in_bounds((h, w), nr, nc):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != int(color):
                        continue
                    seen[nr, nc] = True
                    q.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _frame_info(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> dict | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    expected_area = 2 * (height + width) - 4
    if len(coords) != expected_area:
        return None

    for cc in range(c0, c1 + 1):
        if int(grid[r0, cc]) != int(color) or int(grid[r1, cc]) != int(color):
            return None
    for rr in range(r0, r1 + 1):
        if int(grid[rr, c0]) != int(color) or int(grid[rr, c1]) != int(color):
            return None

    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "interior_bbox": (int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)),
        "interior_area": int((height - 2) * (width - 2)),
    }


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for color in sorted(set(map(int, np.unique(g))) - {BLACK, GREY}):
        for coords in _components_of_color(g, color):
            info = _frame_info(g, coords, color)
            if info is not None:
                frames.append(info)
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each colored rectangular frame as a filter that preserves and recolors only the grey noise trapped inside it.

    Input structure:
    - Black background contains 2-4 one-pixel-thick rectangular frames in distinct non-grey colors.
    - Grey single-pixel noise is scattered both inside frame interiors and outside all frames.
    - Frames are axis-aligned, non-overlapping, and their interior cells are not part of the frame itself.

    Rule:
    - Detect every monochrome rectangular border and read its enclosed interior rectangle.
    - For each frame, keep only the grey pixels lying strictly inside that interior and recolor them to the frame's color.
    - Erase the frame border pixels themselves.
    - Erase every grey noise pixel that is not inside any detected frame interior.
    - Leave all remaining cells black.

    Key insight:
    - The frame is not an output object; it is a temporary geometric filter that defines which grey pixels survive and which color they inherit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _extract_frames(g)
    assert frames, "expected at least one rectangular frame"

    out = np.full_like(g, BLACK)
    for frame in frames:
        r0, c0, r1, c1 = frame["interior_bbox"]
        interior = g[r0 : r1 + 1, c0 : c1 + 1]
        grey_mask = interior == GREY
        if not bool(grey_mask.any()):
            continue
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[grey_mask] = int(frame["color"])
    return out


def _expanded_bbox(bbox: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _choose_internal_counts(rng: random.Random, specs: list[dict]) -> list[int] | None:
    if len(specs) < 2:
        return None

    single_idx, multi_idx = rng.sample(range(len(specs)), 2)
    counts: list[int] = []
    for idx, spec in enumerate(specs):
        area = len(spec["interior"])
        max_count = min(area - 1, max(2, (area + 1) // 2))
        if max_count < 1:
            return None
        if idx == single_idx:
            counts.append(1)
        elif idx == multi_idx:
            if max_count < 2:
                return None
            counts.append(rng.randint(2, max_count))
        else:
            counts.append(rng.randint(1, max_count))
    return counts


def _expected_output(shape: tuple[int, int], specs: list[dict], internal_noise: list[list[tuple[int, int]]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for spec, cells in zip(specs, internal_noise):
        for r, c in cells:
            out[r, c] = int(spec["color"])
    return out


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_ROWS_RANGE)
    w = rng.randint(*GRID_COLS_RANGE)
    frame_count = rng.randint(*FRAME_COUNT_RANGE)
    colors = rng.sample(FRAME_COLORS, frame_count)

    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    specs: list[dict] = []

    for color in colors:
        placed = False
        for _ in range(FRAME_PLACEMENT_ATTEMPTS):
            inner_h, inner_w = rng.choice(INTERIOR_SIZES)
            frame_h, frame_w = inner_h + 2, inner_w + 2
            if frame_h > h or frame_w > w:
                continue
            r0 = rng.randint(0, h - frame_h)
            c0 = rng.randint(0, w - frame_w)
            bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
            er0, ec0, er1, ec1 = _expanded_bbox(bbox, FRAME_GAP, (h, w))
            if bool(blocked[er0 : er1 + 1, ec0 : ec1 + 1].any()):
                continue

            _draw_frame(grid, bbox, color)
            blocked[er0 : er1 + 1, ec0 : ec1 + 1] = True
            interior = [(r, c) for r in range(r0 + 1, r0 + frame_h - 1) for c in range(c0 + 1, c0 + frame_w - 1)]
            specs.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "interior": interior,
                }
            )
            placed = True
            break
        if not placed:
            return None

    internal_counts = _choose_internal_counts(rng, specs)
    if internal_counts is None:
        return None

    internal_noise: list[list[tuple[int, int]]] = []
    used_internal: set[tuple[int, int]] = set()
    for spec, count in zip(specs, internal_counts):
        cells = rng.sample(spec["interior"], count)
        internal_noise.append(cells)
        used_internal.update(cells)

    all_interiors = {cell for spec in specs for cell in spec["interior"]}
    outside_cells = [(r, c) for r in range(h) for c in range(w) if int(grid[r, c]) == BLACK and (r, c) not in all_interiors]
    if len(outside_cells) < 2:
        return None

    noise_candidates = int(np.count_nonzero(grid == BLACK))
    total_internal = sum(len(cells) for cells in internal_noise)
    outside_low = max(2, math.ceil(MIN_NOISE_DENSITY * noise_candidates) - total_internal)
    outside_high = min(len(outside_cells), math.floor(MAX_NOISE_DENSITY * noise_candidates) - total_internal)
    if outside_high < outside_low:
        return None

    outside_noise = rng.sample(outside_cells, rng.randint(outside_low, outside_high))
    for cells in internal_noise:
        for r, c in cells:
            grid[r, c] = GREY
    for r, c in outside_noise:
        grid[r, c] = GREY

    expected = _expected_output((h, w), specs, internal_noise)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    frames = _extract_frames(grid)
    if len(frames) != frame_count:
        return None
    if any(len(cells) == len(spec["interior"]) for spec, cells in zip(specs, internal_noise)):
        return None
    if sum(int(np.count_nonzero(expected == spec["color"])) for spec in specs) != int(np.count_nonzero(expected)):
        return None
    if len(set(map(int, np.unique(actual))) - {BLACK}) != frame_count:
        return None
    if int(np.count_nonzero(grid == GREY)) != total_internal + len(outside_noise):
        return None
    if int(np.count_nonzero(actual)) != total_internal:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
