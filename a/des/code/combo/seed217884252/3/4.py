from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME_COLOR = 5
CORNER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (12, 26)
GRID_W_RANGE = (12, 26)
FRAME_COUNT_CHOICES = (1, 2, 2, 3, 3, 4)
INNER_SIZE_RANGE = (1, 9)
PLACEMENT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 300

PARITY_MODES = ((0, 0), (0, 1), (1, 0), (1, 1))

_RNG = random.Random()


FrameInfo = tuple[int, int, int, int, tuple[int, int, int, int]]


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_from_component(grid: np.ndarray, points: list[tuple[int, int]]) -> FrameInfo | None:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    point_set = set(points)
    corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            occupied = (r, c) in point_set
            if on_border:
                if not occupied:
                    return None
                if (r, c) not in corners and int(grid[r, c]) != FRAME_COLOR:
                    return None
            elif occupied:
                return None

    corner_colors = (
        int(grid[r0, c0]),
        int(grid[r0, c1]),
        int(grid[r1, c0]),
        int(grid[r1, c1]),
    )
    if any(color == BG for color in corner_colors):
        return None
    return r0, c0, r1, c1, corner_colors


def _find_frames(grid: np.ndarray) -> list[FrameInfo]:
    g = np.asarray(grid, dtype=int)
    frames: list[FrameInfo] = []
    for comp in _components4(g != BG):
        frame = _frame_from_component(g, comp)
        if frame is not None:
            frames.append(frame)
    return frames


def _quadrant_patch(inner_h: int, inner_w: int, corners: tuple[int, int, int, int]) -> np.ndarray:
    tl, tr, bl, br = corners
    row_cut = (inner_h + 1) // 2
    col_cut = (inner_w + 1) // 2
    patch = np.empty((inner_h, inner_w), dtype=int)
    patch[:row_cut, :col_cut] = tl
    patch[:row_cut, col_cut:] = tr
    patch[row_cut:, :col_cut] = bl
    patch[row_cut:, col_cut:] = br
    return patch


def _apply_frame_fill(out: np.ndarray, src: np.ndarray, frame: FrameInfo) -> None:
    r0, c0, r1, c1, corners = frame
    inner = _quadrant_patch(r1 - r0 - 1, c1 - c0 - 1, corners)
    src_region = src[r0 + 1 : r1, c0 + 1 : c1]
    dst_region = out[r0 + 1 : r1, c0 + 1 : c1]
    bg_mask = src_region == BG
    dst_region[bg_mask] = inner[bg_mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular frame's interior with a four-quadrant color pattern
    determined by the colors at the frame's four corners.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for frame in _find_frames(g):
        _apply_frame_fill(out, g, frame)
    return out


def _sample_canvas_shape(rng: random.Random, n_frames: int) -> tuple[int, int]:
    low_h = max(GRID_H_RANGE[0], 10 + 2 * n_frames)
    low_w = max(GRID_W_RANGE[0], 10 + 2 * n_frames)
    high_h = GRID_H_RANGE[1]
    high_w = GRID_W_RANGE[1]
    rows = rng.randint(low_h, high_h)
    cols = rng.randint(low_w, high_w)
    if rng.random() < 0.45:
        delta = rng.randint(-4, 4)
        cols = max(low_w, min(high_w, rows + delta))
    return rows, cols


def _sample_with_parity(rng: random.Random, lo: int, hi: int, parity: int) -> int | None:
    values = [v for v in range(lo, hi + 1) if v % 2 == parity]
    if not values:
        return None
    return rng.choice(values)


def _sample_corner_colors(rng: random.Random, force_mixed: bool = False) -> tuple[int, int, int, int]:
    modes = ["all_same", "row_pairs", "col_pairs", "diag_pairs", "all_distinct"]
    if force_mixed:
        modes = [m for m in modes if m != "all_same"]
    mode = rng.choice(modes)
    colors = list(CORNER_COLORS)
    rng.shuffle(colors)

    if mode == "all_same":
        return (colors[0],) * 4
    if mode == "row_pairs":
        return colors[0], colors[0], colors[1], colors[1]
    if mode == "col_pairs":
        return colors[0], colors[1], colors[0], colors[1]
    if mode == "diag_pairs":
        return colors[0], colors[1], colors[1], colors[0]
    return colors[0], colors[1], colors[2], colors[3]


def _sample_frame_specs(
    rng: random.Random, rows: int, cols: int, n_frames: int
) -> list[tuple[int, int, tuple[int, int, int, int]]] | None:
    max_inner_h = min(INNER_SIZE_RANGE[1], rows - 4)
    max_inner_w = min(INNER_SIZE_RANGE[1], cols - 4)
    if n_frames >= 4:
        max_inner_h = min(max_inner_h, 6)
        max_inner_w = min(max_inner_w, 6)
    elif n_frames == 3:
        max_inner_h = min(max_inner_h, 7)
        max_inner_w = min(max_inner_w, 7)
    if max_inner_h < INNER_SIZE_RANGE[0] or max_inner_w < INNER_SIZE_RANGE[0]:
        return None

    parity_modes = list(PARITY_MODES)
    rng.shuffle(parity_modes)
    parity_modes = parity_modes[:n_frames]

    specs: list[tuple[int, int, tuple[int, int, int, int]]] = []
    mixed_seen = False
    for index, (parity_h, parity_w) in enumerate(parity_modes):
        inner_h = _sample_with_parity(rng, INNER_SIZE_RANGE[0], max_inner_h, parity_h)
        inner_w = _sample_with_parity(rng, INNER_SIZE_RANGE[0], max_inner_w, parity_w)
        if inner_h is None or inner_w is None:
            return None

        force_mixed = n_frames > 1 and index == n_frames - 1 and not mixed_seen
        corners = _sample_corner_colors(rng, force_mixed=force_mixed)
        mixed_seen = mixed_seen or len(set(corners)) > 1
        specs.append((inner_h + 2, inner_w + 2, corners))

    specs.sort(key=lambda item: item[0] * item[1], reverse=True)
    return specs


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rows, cols = occupied.shape
    if r0 < 1 or c0 < 1 or r0 + h > rows - 1 or c0 + w > cols - 1:
        return False
    rr0 = max(0, r0 - PLACEMENT_PADDING)
    cc0 = max(0, c0 - PLACEMENT_PADDING)
    rr1 = min(rows, r0 + h + PLACEMENT_PADDING)
    cc1 = min(cols, c0 + w + PLACEMENT_PADDING)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rows, cols = occupied.shape
    rr0 = max(0, r0 - PLACEMENT_PADDING)
    cc0 = max(0, c0 - PLACEMENT_PADDING)
    rr1 = min(rows, r0 + h + PLACEMENT_PADDING)
    cc1 = min(cols, c0 + w + PLACEMENT_PADDING)
    occupied[rr0:rr1, cc0:cc1] = True


def _paint_frame(grid: np.ndarray, frame: FrameInfo) -> None:
    r0, c0, r1, c1, corners = frame
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0, c0], grid[r0, c1], grid[r1, c0], grid[r1, c1] = corners


def _expected_output(grid: np.ndarray, frames: list[FrameInfo]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for frame in frames:
        _apply_frame_fill(out, np.asarray(grid, dtype=int), frame)
    return out


def _place_frames(
    rng: random.Random, rows: int, cols: int, specs: list[tuple[int, int, tuple[int, int, int, int]]]
) -> list[FrameInfo] | None:
    occupied = np.zeros((rows, cols), dtype=bool)
    frames: list[FrameInfo] = []

    for h, w, corners in specs:
        candidates = [(r, c) for r in range(1, rows - h) for c in range(1, cols - w)]
        rng.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            if not _can_place(occupied, r0, c0, h, w):
                continue
            frame = (r0, c0, r0 + h - 1, c0 + w - 1, corners)
            frames.append(frame)
            _mark_occupied(occupied, r0, c0, h, w)
            placed = True
            break
        if not placed:
            return None

    return frames


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.choice(FRAME_COUNT_CHOICES)
    rows, cols = _sample_canvas_shape(rng, n_frames)
    specs = _sample_frame_specs(rng, rows, cols, n_frames)
    if specs is None:
        return None

    frames = _place_frames(rng, rows, cols, specs)
    if frames is None:
        return None

    grid = np.full((rows, cols), BG, dtype=int)
    for frame in frames:
        _paint_frame(grid, frame)

    parsed = _find_frames(grid)
    expected_boxes = {(r0, c0, r1, c1) for r0, c0, r1, c1, _ in frames}
    parsed_boxes = {(r0, c0, r1, c1) for r0, c0, r1, c1, _ in parsed}
    if parsed_boxes != expected_boxes:
        return None

    expected = _expected_output(grid, frames)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, actual):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
