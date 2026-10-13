import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME = 5
PATTERN = 8
LEGEND_COLORS = (1, 2, 3, 4, 6, 7, 9)
INNER_SIZE_CHOICES = (8, 10, 12)
FRAME_COUNT_CHOICES = (2, 3)
GRID_MIN = 22
GRID_MAX = 30
PATTERN_SIZE_RANGE = (6, 15)
PLACEMENT_GAP = 2
GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_rectangular_frame(grid: np.ndarray, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False
    expected = 2 * (h + w) - 4
    if int(coords.shape[0]) != expected:
        return False
    box = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(box[0, :] == FRAME) or not np.all(box[-1, :] == FRAME):
        return False
    if not np.all(box[:, 0] == FRAME) or not np.all(box[:, -1] == FRAME):
        return False
    if np.any(box[1:-1, 1:-1] == FRAME):
        return False
    return True


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames = []
    for comp in _components_mask(grid == FRAME):
        if _is_rectangular_frame(grid, comp):
            frames.append(_bbox_coords(comp))
    return sorted(frames)


def _legend_for_frame(grid: np.ndarray, frame: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, _, _ = frame
    legend = np.asarray(grid[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3], dtype=int)
    assert legend.shape == (2, 2)
    vals = list(map(int, legend.ravel()))
    assert len(set(vals)) == 4
    assert all(v not in (BG, FRAME) for v in vals)
    return legend


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_frames(g)
    assert frames, "expected at least one hollow frame"
    for r0, c0, r1, c1 in frames:
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        assert inner_h % 2 == 0 and inner_w % 2 == 0
        half_h = inner_h // 2
        half_w = inner_w // 2
        legend = _legend_for_frame(g, (r0, c0, r1, c1))
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if r < r0 + 3 and c < c0 + 3:
                    continue
                val = int(g[r, c])
                if val in (BG, FRAME):
                    continue
                qr = 0 if (r - (r0 + 1)) < half_h else 1
                qc = 0 if (c - (c0 + 1)) < half_w else 1
                out[r, c] = int(legend[qr, qc])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each foreground pixel inside a hollow frame using the frame's local 2x2 quadrant legend.

    Input structure:
    - The grid has a black background and two or three gray one-cell-thick hollow rectangular frames.
    - Each frame has an even-sized interior, a fixed 2x2 legend in the interior's top-left corner, and a separate connected single-color foreground pattern elsewhere inside.
    - The four legend cells are distinct non-background colors, while the frame border and the legend remain visible in place.

    Rule:
    - Detect every gray hollow frame and read the 2x2 legend at the frame interior's top-left corner.
    - Split that frame's interior into equal top-left, top-right, bottom-left, and bottom-right quadrants.
    - For each non-background, non-frame pixel inside the frame that is not part of the legend, replace its color with the legend color from the matching quadrant.

    Key insight:
    - The legend is local to its own frame, and each pattern pixel must be colored from a combination of two facts: which frame contains it and which quadrant of that frame it occupies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _expected_output(g)


def _boxes_separated(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = PLACEMENT_GAP,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _candidate_frames(h: int, w: int, rng: random.Random) -> list[tuple[int, int, int, int]] | None:
    n_frames = rng.choice(FRAME_COUNT_CHOICES)
    frames: list[tuple[int, int, int, int]] = []
    size_choices = list(INNER_SIZE_CHOICES)
    for _ in range(n_frames):
        placed = False
        rng.shuffle(size_choices)
        for _ in range(150):
            inner_h = rng.choice(size_choices)
            inner_w = rng.choice(size_choices)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if outer_h + 2 > h or outer_w + 2 > w:
                continue
            r0 = rng.randint(1, h - outer_h - 1)
            c0 = rng.randint(1, w - outer_w - 1)
            frame = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if all(_boxes_separated(frame, other) for other in frames):
                frames.append(frame)
                placed = True
                break
        if not placed:
            return None
    return sorted(frames)


def _quadrant_index(local_r: int, local_c: int, inner_h: int, inner_w: int) -> int:
    half_h = inner_h // 2
    half_w = inner_w // 2
    qr = 0 if local_r < half_h else 1
    qc = 0 if local_c < half_w else 1
    return qr * 2 + qc


def _grow_pattern(
    inner_h: int,
    inner_w: int,
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    allowed = {
        (r, c)
        for r in range(inner_h)
        for c in range(inner_w)
        if not (r < 2 and c < 2)
    }
    quadrants = [
        (inner_h // 2 - 1, inner_w // 2 - 1),
        (inner_h // 2 - 1, inner_w // 2),
        (inner_h // 2, inner_w // 2 - 1),
        (inner_h // 2, inner_w // 2),
    ]
    cover_count = rng.choice((3, 4))
    chosen_ids = set(rng.sample(range(4), cover_count))
    cells = {quadrants[i] for i in chosen_ids}
    if not cells.issubset(allowed):
        return None

    target = rng.randint(PATTERN_SIZE_RANGE[0], PATTERN_SIZE_RANGE[1])
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in allowed and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        rng.shuffle(frontier)
        cells.add(frontier[0])

    coverage = {_quadrant_index(r, c, inner_h, inner_w) for r, c in cells}
    if len(coverage) < 3:
        return None

    comp = _components_mask(_mask_from_cells(inner_h, inner_w, cells))
    if len(comp) != 1:
        return None
    return cells


def _mask_from_cells(h: int, w: int, cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _draw_frame_and_contents(
    grid: np.ndarray,
    frame: tuple[int, int, int, int],
    legend: np.ndarray,
    pattern_cells: set[tuple[int, int]],
) -> None:
    r0, c0, r1, c1 = frame
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    grid[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3] = legend
    for lr, lc in pattern_cells:
        grid[r0 + 1 + lr, c0 + 1 + lc] = PATTERN


def _frame_details(
    grid: np.ndarray,
    frame: tuple[int, int, int, int],
) -> dict[str, object]:
    r0, c0, r1, c1 = frame
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    inner_h, inner_w = inner.shape
    legend = inner[:2, :2].copy()
    pattern_cells = {
        (r, c)
        for r in range(inner_h)
        for c in range(inner_w)
        if not (r < 2 and c < 2) and int(inner[r, c]) == PATTERN
    }
    coverage = {_quadrant_index(r, c, inner_h, inner_w) for r, c in pattern_cells}
    return {
        "frame": frame,
        "legend": legend,
        "pattern_cells": pattern_cells,
        "coverage": coverage,
        "inner_shape": (inner_h, inner_w),
    }


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        frames = _find_frames(grid)
        if len(frames) not in FRAME_COUNT_CHOICES:
            return False
        details = [_frame_details(grid, frame) for frame in frames]
        if any(len(d["pattern_cells"]) < PATTERN_SIZE_RANGE[0] for d in details):
            return False
        if any(len(d["coverage"]) < 3 for d in details):
            return False
        if len(details) > 1:
            legends = {tuple(map(int, np.asarray(d["legend"]).ravel())) for d in details}
            if len(legends) < 2:
                return False
        expected = _expected_output(grid)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False
        if np.array_equal(actual, grid):
            return False
        for d in details:
            frame = d["frame"]
            legend = np.asarray(d["legend"])
            pattern_cells = d["pattern_cells"]
            inner_h, inner_w = d["inner_shape"]
            r0, c0, _, _ = frame
            for lr, lc in pattern_cells:
                qr = 0 if lr < inner_h // 2 else 1
                qc = 0 if lc < inner_w // 2 else 1
                rr = r0 + 1 + lr
                cc = c0 + 1 + lc
                if int(actual[rr, cc]) != int(legend[qr, qc]):
                    return False
        return True
    except AssertionError:
        return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    frames = _candidate_frames(h, w, rng)
    if frames is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    used_legends: set[tuple[int, int, int, int]] = set()
    saw_three = False
    saw_four = False

    for frame in frames:
        r0, c0, r1, c1 = frame
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        legend_vals = tuple(rng.sample(LEGEND_COLORS, 4))
        if len(frames) > 1:
            tries = 0
            while legend_vals in used_legends and tries < 20:
                legend_vals = tuple(rng.sample(LEGEND_COLORS, 4))
                tries += 1
        used_legends.add(legend_vals)
        legend = np.array(legend_vals, dtype=int).reshape(2, 2)

        pattern_cells = None
        for _ in range(80):
            pattern_cells = _grow_pattern(inner_h, inner_w, rng)
            if pattern_cells is not None:
                break
        if pattern_cells is None:
            return None

        coverage = {_quadrant_index(r, c, inner_h, inner_w) for r, c in pattern_cells}
        saw_three |= len(coverage) == 3
        saw_four |= len(coverage) == 4
        _draw_frame_and_contents(grid, frame, legend, pattern_cells)

    if len(frames) >= 2 and len(used_legends) < 2:
        return None
    if len(frames) == 3 and not (saw_three and saw_four):
        return None
    if not _semantic_check(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
