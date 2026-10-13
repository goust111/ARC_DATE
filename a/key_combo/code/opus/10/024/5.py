from __future__ import annotations

import random
from collections import deque

import numpy as np


FRAME_COLOR = 2
BLACK = 0
NON_FRAME_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
GEN_ATTEMPTS = 500


def _run_start_indices(lines: np.ndarray) -> list[int]:
    keep = [0]
    for i in range(1, lines.shape[0]):
        if not np.array_equal(lines[i], lines[i - 1]):
            keep.append(i)
    return keep


def _compress_grid(grid: np.ndarray) -> np.ndarray:
    row_starts = _run_start_indices(grid)
    col_starts = _run_start_indices(grid.T)
    return grid[np.ix_(row_starts, col_starts)].copy()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_hollow_rectangle_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    cells = {tuple(map(int, rc)) for rc in coords}
    expected_border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    return cells == expected_border


def _frame_candidates(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], np.ndarray]]:
    candidates = []
    for color in sorted(set(map(int, np.unique(grid)))):
        for coords in _components_of_color(grid, color):
            if _is_hollow_rectangle_component(coords):
                candidates.append((color, _bbox(coords), coords))
    return candidates


def _analyze(grid: np.ndarray) -> tuple[np.ndarray, int, tuple[int, int, int, int]]:
    compressed = _compress_grid(grid)
    candidates = _frame_candidates(compressed)
    assert len(candidates) == 1, "expected exactly one hollow rectangle component after compression"
    color, bbox, _coords = candidates[0]
    return compressed, int(color), bbox


def main(grid: np.ndarray) -> np.ndarray:
    """Compress duplicate row/column runs, recover the hidden rectangular frame, and return its cleaned interior.

    Input structure:
    - The input is a fully filled stretched color grid with no obvious background; it comes from a smaller grid whose rows and columns were each repeated 1-3 times in unequal adjacent runs.
    - In the compressed grid, exactly one 4-connected monochrome component is a hollow rectangular frame, and additional same-color distractor components may exist elsewhere.

    Rule:
    - Merge each maximal run of identical adjacent rows into one row, and each maximal run of identical adjacent columns into one column.
    - In the compressed grid, find the unique monochrome component whose cells are exactly the border of its bounding box.
    - Crop the interior of that frame, excluding the frame border itself.
    - Replace any pixels of the frame color that remain in the crop with black.

    Key insight:
    - The frame is only geometrically recognizable after undoing the unequal row/column stretching; the crop coordinates must be taken in that recovered compressed coordinate system.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] > 0 and g.shape[1] > 0

    compressed, frame_color, (r0, c0, r1, c1) = _analyze(g)
    cropped = compressed[r0 + 1 : r1, c0 + 1 : c1].copy()
    cropped[cropped == frame_color] = BLACK
    return cropped


def _adjacent_rows_or_cols_repeat(grid: np.ndarray) -> bool:
    for i in range(1, grid.shape[0]):
        if np.array_equal(grid[i], grid[i - 1]):
            return True
    for i in range(1, grid.shape[1]):
        if np.array_equal(grid[:, i], grid[:, i - 1]):
            return True
    return False


def _choose_base_shape() -> tuple[int, int]:
    return random.randint(7, 9), random.randint(9, 11)


def _choose_palette() -> list[int]:
    extra_count = random.randint(3, 4)
    return [FRAME_COLOR] + random.sample(NON_FRAME_COLORS, extra_count)


def _random_mosaic(shape: tuple[int, int], palette: list[int]) -> np.ndarray:
    colors = np.array([c for c in palette if c != FRAME_COLOR], dtype=int)
    return np.random.choice(colors, size=shape, replace=True)


def _choose_frame_bbox(h: int, w: int) -> tuple[int, int, int, int]:
    inner_h = random.randint(2, min(5, h - 4))
    inner_w = random.randint(3, min(6, w - 4))
    bbox_h = inner_h + 2
    bbox_w = inner_w + 2
    r0 = random.randint(1, h - bbox_h - 1)
    c0 = random.randint(1, w - bbox_w - 1)
    return r0, c0, r0 + bbox_h - 1, c0 + bbox_w - 1


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR


def _can_place_red_shape(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    cells: list[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    red_mask = grid == FRAME_COLOR
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if r0 < r < r1 and c0 < c < c1:
            return False
        if red_mask[r, c]:
            return False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and red_mask[nr, nc] and (nr, nc) not in cells:
                return False
    return True


def _safe_add_red_distractors(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    shape_templates = [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (1, 0), (1, 1)],
    ]
    painted_any = False
    for _ in range(random.randint(1, 3)):
        for _ in range(60):
            tmpl = random.choice(shape_templates)
            anchor_r = random.randrange(grid.shape[0])
            anchor_c = random.randrange(grid.shape[1])
            cells = [(anchor_r + dr, anchor_c + dc) for dr, dc in tmpl]
            if not _can_place_red_shape(grid, bbox, cells):
                continue
            original = [int(grid[r, c]) for r, c in cells]
            for r, c in cells:
                grid[r, c] = FRAME_COLOR
            if len(_frame_candidates(grid)) == 1:
                painted_any = True
                break
            for (r, c), color in zip(cells, original):
                grid[r, c] = color
    return painted_any


def _build_compressed_grid() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = _choose_base_shape()
    palette = _choose_palette()
    grid = _random_mosaic((h, w), palette)
    bbox = _choose_frame_bbox(h, w)
    _paint_frame(grid, bbox)
    if not _safe_add_red_distractors(grid, bbox):
        raise RuntimeError("failed to add red distractors")
    if _adjacent_rows_or_cols_repeat(grid):
        raise RuntimeError("compressed grid already contains duplicate adjacent rows or columns")
    candidates = _frame_candidates(grid)
    if len(candidates) != 1:
        raise RuntimeError("compressed grid must have exactly one frame candidate")
    color, actual_bbox, _coords = candidates[0]
    if color != FRAME_COLOR or actual_bbox != bbox:
        raise RuntimeError("compressed grid does not preserve the intended frame")
    return grid, bbox


def _choose_run_lengths(count: int) -> list[int]:
    return [random.randint(1, 3) for _ in range(count)]


def _stretch_grid(
    grid: np.ndarray,
    row_runs: list[int],
    col_runs: list[int],
) -> np.ndarray:
    return np.repeat(np.repeat(grid, row_runs, axis=0), col_runs, axis=1)


def _compression_is_necessary(
    stretched: np.ndarray,
    compressed: np.ndarray,
    bbox: tuple[int, int, int, int],
    row_runs: list[int],
    col_runs: list[int],
) -> bool:
    if np.array_equal(stretched, compressed):
        return False
    if _frame_candidates(stretched):
        return False
    r0, c0, r1, c1 = bbox
    border_run_lengths = [row_runs[r0], row_runs[r1], col_runs[c0], col_runs[c1]]
    return max(border_run_lengths) > 1


def _independent_expected_output(
    compressed: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = compressed[r0 + 1 : r1, c0 + 1 : c1].copy()
    out[out == FRAME_COLOR] = BLACK
    return out


def _semantic_checks(
    stretched: np.ndarray,
    compressed: np.ndarray,
    bbox: tuple[int, int, int, int],
    row_runs: list[int],
    col_runs: list[int],
) -> bool:
    if stretched.ndim != 2 or max(stretched.shape) > 30:
        return False
    if compressed.ndim != 2 or max(compressed.shape) > 30:
        return False
    if not np.array_equal(_compress_grid(stretched), compressed):
        return False
    if not _compression_is_necessary(stretched, compressed, bbox, row_runs, col_runs):
        return False
    red_components = _components_of_color(compressed, FRAME_COLOR)
    if len(red_components) < 2:
        return False
    expected = _independent_expected_output(compressed, bbox)
    if expected.shape[0] < 2 or expected.shape[1] < 3:
        return False
    if len(set(map(int, np.unique(expected)))) < 2:
        return False
    try:
        actual = main(stretched)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    try:
        compressed, bbox = _build_compressed_grid()
    except RuntimeError:
        return None

    for _ in range(80):
        row_runs = _choose_run_lengths(compressed.shape[0])
        col_runs = _choose_run_lengths(compressed.shape[1])
        if max(row_runs) == 1 and max(col_runs) == 1:
            continue
        stretched = _stretch_grid(compressed, row_runs, col_runs)
        if max(stretched.shape) > 30:
            continue
        if not _semantic_checks(stretched, compressed, bbox, row_runs, col_runs):
            continue
        return stretched
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
