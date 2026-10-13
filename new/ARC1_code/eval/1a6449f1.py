from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg, paste, random_free_location
import random


def _full_frame(g: np.ndarray, color: int, r0: int, c0: int, r1: int, c1: int) -> bool:
    return (
        np.all(g[r0, c0 : c1 + 1] == color)
        and np.all(g[r1, c0 : c1 + 1] == color)
        and np.all(g[r0 : r1 + 1, c0] == color)
        and np.all(g[r0 : r1 + 1, c1] == color)
    )


def _rectangular_frames(g: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    bg = get_bg(g)
    frames = []
    for color in colors_present(g, exclude={bg}):
        rows = np.flatnonzero(np.any(g == color, axis=1))
        cols = np.flatnonzero(np.any(g == color, axis=0))
        for i, r0 in enumerate(rows):
            for r1 in rows[i + 1 :]:
                if r1 - r0 < 2:
                    continue
                for j, c0 in enumerate(cols):
                    for c1 in cols[j + 1 :]:
                        if c1 - c0 < 2:
                            continue
                        if _full_frame(g, int(color), int(r0), int(c0), int(r1), int(c1)):
                            frames.append((int(color), int(r0), int(c0), int(r1), int(c1)))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the contents enclosed by the largest complete rectangular frame.

    Input structure
    - The grid is mostly background noise plus several monochrome rectangular frames.
    - Each frame has a one-pixel-thick border; the interior may contain any colors.

    Rule
    - Find all complete same-color rectangular frame borders.
    - Select the unique frame with the largest interior area.
    - Output the strict interior of that frame, removing the border.

    Key insight
    - The frame border is only a locator; the desired output is the raw enclosed region of the largest frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _rectangular_frames(g)
    assert frames, "expected at least one complete rectangular frame"

    interior_areas = [(r1 - r0 - 1) * (c1 - c0 - 1) for _, r0, c0, r1, c1 in frames]
    largest_area = max(interior_areas)
    largest = [frame for frame, area in zip(frames, interior_areas) if area == largest_area]
    assert len(largest) == 1, "expected a unique largest rectangular frame"

    _, r0, c0, r1, c1 = largest[0]
    return g[r0 + 1 : r1, c0 + 1 : c1].copy()

# --------------- generator (o5x) ---------------

# Grid, palette, and retry limits.
GRID_H_RANGE = (13, 27)
GRID_W_RANGE = (13, 25)
FRAME_COUNT_RANGE = (2, 3)
GENERATION_ATTEMPTS = 300
BG_COLOR, COLOR_MIN, COLOR_MAX = 0, 1, 9
NON_BG_COLORS = range(COLOR_MIN, COLOR_MAX + 1)

# Frame size bounds and largest-frame separation.
OUTER_H_RANGE = (4, 12)
OUTER_W_RANGE = (5, 12)
LARGEST_MIN_AREA = 18
LARGEST_RATIO_DEN, LARGEST_RATIO_NUM = 10, 19

# Fill density ranges.
INTERIOR_FILL_RANGE = (0.25, 0.35)
OUTSIDE_FILL_RANGE = (0.29, 0.33)


def _gen_sparse_pixels(h: int, w: int) -> np.ndarray:
    out = np.zeros((int(h), int(w)), dtype=int)
    total = int(h) * int(w)
    lo, hi = INTERIOR_FILL_RANGE
    lo_count = max(1, int(np.ceil(lo * total)))
    hi_count = min(total, int(np.floor(hi * total)))
    count = random.randint(lo_count, max(lo_count, hi_count))
    for idx in random.sample(range(total), count):
        out.flat[idx] = random.randint(COLOR_MIN, COLOR_MAX)
    return out


def _gen_frame_patch(interior: np.ndarray, border_color: int) -> np.ndarray:
    h, w = interior.shape
    patch = np.zeros((h + 2, w + 2), dtype=int)
    patch[1:-1, 1:-1] = interior
    patch[[0, -1], :] = border_color
    patch[:, [0, -1]] = border_color
    return patch


def _gen_size_pool(H: int, W: int) -> list[tuple[int, int, int]]:
    pool = []
    for interior_h in range(OUTER_H_RANGE[0] - 2, min(OUTER_H_RANGE[1], H) - 1):
        for interior_w in range(OUTER_W_RANGE[0] - 2, min(OUTER_W_RANGE[1], W) - 1):
            pool.append((interior_h, interior_w, interior_h * interior_w))
    return pool


def _select_frame_sizes(H: int, W: int, n_frames: int) -> list[tuple[int, int, int]] | None:
    size_pool = _gen_size_pool(H, W)
    largest_choices = [s for s in size_pool if s[2] >= LARGEST_MIN_AREA]
    if not largest_choices:
        return None
    largest_h, largest_w, largest_area = random.choice(largest_choices)

    small_pool = [
        s for s in size_pool
        if s[2] != largest_area
        and LARGEST_RATIO_DEN * largest_area >= LARGEST_RATIO_NUM * s[2]
    ]
    if len({s[2] for s in small_pool}) < n_frames - 1:
        return None

    frame_sizes = [(largest_h, largest_w, largest_area)]
    used_areas = {largest_area}
    for _ in range(n_frames - 1):
        choices = [s for s in small_pool if s[2] not in used_areas]
        if not choices:
            return None
        chosen = random.choice(choices)
        frame_sizes.append(chosen)
        used_areas.add(chosen[2])
    return frame_sizes


def _gen_scatter_outside(g: np.ndarray, frame_bboxes: list[tuple[int, int, int, int]]) -> bool:
    outside = np.ones(g.shape, dtype=bool)
    for r0, c0, r1, c1 in frame_bboxes:
        outside[r0 : r1 + 1, c0 : c1 + 1] = False

    coords = np.argwhere(outside)
    if len(coords) == 0:
        return False

    lo, hi = OUTSIDE_FILL_RANGE
    lo_count = int(np.ceil(lo * len(coords)))
    hi_count = int(np.floor(hi * len(coords)))
    count_min = max(lo_count, min(len(NON_BG_COLORS), len(coords)))
    if count_min > hi_count:
        return False
    count = random.randint(count_min, hi_count)
    chosen = random.sample(range(len(coords)), count)
    values = list(NON_BG_COLORS)[:count]
    values += [random.randint(COLOR_MIN, COLOR_MAX) for _ in range(count - len(values))]
    random.shuffle(values)

    for idx, color in zip(chosen, values):
        r, c = coords[idx]
        g[int(r), int(c)] = int(color)
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    n_frames = random.randint(*FRAME_COUNT_RANGE)

    frame_sizes = _select_frame_sizes(H, W, n_frames)
    if frame_sizes is None:
        return None

    colors = random.sample(NON_BG_COLORS, n_frames)
    g = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=int)
    expected = None
    frame_bboxes = []

    for i, ((interior_h, interior_w, _area), color) in enumerate(zip(frame_sizes, colors)):
        interior = _gen_sparse_pixels(interior_h, interior_w)
        if i == 0:
            expected = interior.copy()
        patch = _gen_frame_patch(interior, color)
        solid = np.ones(patch.shape, dtype=int)
        try:
            r0, c0 = random_free_location(occupied, solid, transparent=0)
        except ValueError:
            return None
        paste(g, patch, r0, c0)
        paste(occupied, solid, r0, c0)
        frame_bboxes.append((r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1))

    if not _gen_scatter_outside(g, frame_bboxes):
        return None

    bg_count = int(np.sum(g == BG_COLOR))
    if any(int(np.sum(g == color)) >= bg_count for color in NON_BG_COLORS):
        return None

    try:
        actual = main(g)
    except Exception:
        return None
    if expected is not None and np.array_equal(actual, expected):
        return g
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("failed to generate valid input for 1a6449f1")
