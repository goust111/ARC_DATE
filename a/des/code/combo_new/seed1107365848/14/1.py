from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

FRAME_COUNT_RANGE = (2, 5)
FRAME_STEP_RANGE = (2, 4)
OUTER_HEIGHT_MAX = 27
OUTER_WIDTH_MAX = 28
MARGIN_RANGE = (1, 5)
TOP_GAP_RANGE = (1, 5)
BOTTOM_GAP_RANGE = (1, 5)
MAX_EXTRA_COLORS = 3
GENERATION_RETRIES = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            pts: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                pts.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append((rr, cc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_perimeter_points(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    pts = {(r0, c) for c in range(c0, c1 + 1)}
    pts.update((r1, c) for c in range(c0, c1 + 1))
    pts.update((r, c0) for r in range(r0 + 1, r1))
    pts.update((r, c1) for r in range(r0 + 1, r1))
    return pts


def _is_rectangular_frame(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    return set(points) == _frame_perimeter_points((r0, c0, r1, c1))


def _find_frames(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    body = np.asarray(grid, dtype=int).copy()
    body[0, :] = BG
    frames = [pts for pts in _components(body != BG) if _is_rectangular_frame(pts)]
    frames.sort(
        key=lambda pts: (
            -((_bbox(pts)[2] - _bbox(pts)[0] + 1) * (_bbox(pts)[3] - _bbox(pts)[1] + 1)),
            _bbox(pts)[0],
            _bbox(pts)[1],
        )
    )
    return frames


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested monochrome rectangular frames according to the top-row color strip, then erase the strip.

    Input structure:
    - The background is Black.
    - The top row contains a left-to-right strip of distinct non-black color pixels that acts as a legend.
    - Below the strip sits a set of concentric one-cell-thick rectangular ring frames; all frames share one input color and are separated by Black gaps.

    Rule:
    - Read the non-black pixels on the top row from left to right to obtain the legend sequence.
    - Detect every hollow rectangular frame below the top row and sort the frames from outermost to innermost.
    - Recolor the outermost frame with the first legend color, the next frame with the second, and continue inward; unused extra legend colors are ignored.
    - Erase the entire top row to Black in the output.

    Key insight:
    - Color identity is not present on the frames themselves; the solver must recover the nesting order from geometry first, then apply the separate border legend in that order.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    legend = [int(g[0, c]) for c in range(g.shape[1]) if int(g[0, c]) != BG]
    frames = _find_frames(g)

    out[0, :] = BG
    for color, pts in zip(legend, frames, strict=False):
        for r, c in pts:
            out[r, c] = color
    return out


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_frames = random.randint(*FRAME_COUNT_RANGE)
    steps = [random.randint(*FRAME_STEP_RANGE) for _ in range(n_frames - 1)]
    total_inset = sum(steps)

    min_outer = 3 + 2 * total_inset
    if min_outer > OUTER_HEIGHT_MAX or min_outer > OUTER_WIDTH_MAX:
        return None

    outer_h = random.randint(min_outer, OUTER_HEIGHT_MAX)
    outer_w = random.randint(min_outer, OUTER_WIDTH_MAX)

    max_top_gap = min(TOP_GAP_RANGE[1], 30 - 1 - outer_h - 1)
    if max_top_gap < TOP_GAP_RANGE[0]:
        return None
    top_gap = random.randint(TOP_GAP_RANGE[0], max_top_gap)

    max_bottom_gap = min(BOTTOM_GAP_RANGE[1], 30 - 1 - top_gap - outer_h)
    if max_bottom_gap < BOTTOM_GAP_RANGE[0]:
        return None
    bottom_gap = random.randint(BOTTOM_GAP_RANGE[0], max_bottom_gap)

    max_left = min(MARGIN_RANGE[1], 30 - outer_w - 1)
    if max_left < MARGIN_RANGE[0]:
        return None
    left = random.randint(MARGIN_RANGE[0], max_left)

    max_right = min(MARGIN_RANGE[1], 30 - outer_w - left)
    if max_right < MARGIN_RANGE[0]:
        return None
    right = random.randint(MARGIN_RANGE[0], max_right)

    h = 1 + top_gap + outer_h + bottom_gap
    w = left + outer_w + right
    if h > 30 or w > 30:
        return None

    frame_color = random.choice(PALETTE)
    available_legend = [c for c in PALETTE if c != frame_color]

    if random.random() < 0.5:
        extra = 0
    else:
        extra = random.randint(1, min(MAX_EXTRA_COLORS, len(available_legend) - n_frames))
    strip_len = n_frames + extra
    if strip_len > len(available_legend) or strip_len > w:
        return None
    legend = random.sample(available_legend, strip_len)

    grid = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)
    strip_start = random.randint(0, w - strip_len)
    grid[0, strip_start : strip_start + strip_len] = legend

    top = 1 + top_gap
    inset = 0
    used_bboxes: list[tuple[int, int, int, int]] = []
    for i in range(n_frames):
        bbox = (top + inset, left + inset, top + outer_h - inset - 1, left + outer_w - inset - 1)
        used_bboxes.append(bbox)
        _draw_frame(grid, bbox, frame_color)
        _draw_frame(expected, bbox, legend[i])
        if i < n_frames - 1:
            inset += steps[i]

    if len(_find_frames(grid)) != n_frames:
        return None

    if any((r1 - r0) < 2 or (c1 - c0) < 2 for r0, c0, r1, c1 in used_bboxes):
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        case = _sample_case()
        if case is None:
            continue
        grid, expected = case
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid input for the nested-frame recoloring task")
