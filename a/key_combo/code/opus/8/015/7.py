from __future__ import annotations

from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

LEFT_PANEL_SIZE = 11
RIGHT_PANEL_SIZE = 3
OUTPUT_SIZE = 9
LEFT_CELL_OFFSETS = (0, 4, 8)
DIAMOND_COORDS = {(0, 1), (1, 0), (1, 2), (2, 1)}
RIGHT_COLORS = np.array([BLUE, GREEN, YELLOW, GREY, TEAL, MAROON], dtype=int)

_GEN_CALLS = 0
_SEEN_INPUT_SIGS: set[bytes] = set()
_SEEN_OUTPUT_SIGS: set[bytes] = set()


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _grid_sig(grid: np.ndarray) -> bytes:
    return grid.shape[0].to_bytes(1, "little") + grid.shape[1].to_bytes(1, "little") + grid.tobytes()


def _is_left_panel(window: np.ndarray) -> bool:
    if window.shape != (LEFT_PANEL_SIZE, LEFT_PANEL_SIZE):
        return False
    if not np.all(window[3, :] == PINK) or not np.all(window[7, :] == PINK):
        return False
    if not np.all(window[:, 3] == PINK) or not np.all(window[:, 7] == PINK):
        return False
    if np.any(~np.isin(window, [ORANGE, PINK, RED])):
        return False

    for r0 in LEFT_CELL_OFFSETS:
        for c0 in LEFT_CELL_OFFSETS:
            sub = window[r0 : r0 + 3, c0 : c0 + 3]
            reds = {tuple(pos) for pos in np.argwhere(sub == RED)}
            if reds and reds != DIAMOND_COORDS:
                return False
            if np.any(~np.isin(sub, [ORANGE, RED])):
                return False
    return True


def _find_left_panel(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    hits = []
    for r0 in range(h - LEFT_PANEL_SIZE + 1):
        for c0 in range(w - LEFT_PANEL_SIZE + 1):
            if _is_left_panel(grid[r0 : r0 + LEFT_PANEL_SIZE, c0 : c0 + LEFT_PANEL_SIZE]):
                hits.append((r0, c0))
    assert len(hits) == 1, "expected a unique 11x11 left panel"
    return hits[0]


def _count_sprites(left_panel: np.ndarray) -> int:
    count = 0
    for r0 in LEFT_CELL_OFFSETS:
        for c0 in LEFT_CELL_OFFSETS:
            sub = left_panel[r0 : r0 + 3, c0 : c0 + 3]
            reds = {tuple(pos) for pos in np.argwhere(sub == RED)}
            if reds:
                assert reds == DIAMOND_COORDS, "occupied cell must contain a red diamond sprite"
                count += 1
    assert 1 <= count <= 4, "generator guarantees 1..4 occupied cells"
    return count


def _find_right_panel(grid: np.ndarray, left_top: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    left_r, left_c = left_top
    right_c = left_c + LEFT_PANEL_SIZE + 1
    assert right_c + RIGHT_PANEL_SIZE <= w, "right panel must sit one column to the right of the left panel"
    assert np.all(grid[:, left_c + LEFT_PANEL_SIZE] == BLACK), "expected a black separator column"

    hits = []
    for r0 in range(h - RIGHT_PANEL_SIZE + 1):
        block = grid[r0 : r0 + RIGHT_PANEL_SIZE, right_c : right_c + RIGHT_PANEL_SIZE]
        if np.all(block != BLACK):
            hits.append((r0, right_c))
    assert len(hits) == 1, "expected a unique 3x3 right panel"
    r0, c0 = hits[0]
    return _crop_bbox(grid, (r0, c0, r0 + 2, c0 + 2))


def _dominant_color(pattern: np.ndarray) -> int:
    items = sorted(Counter(map(int, pattern.flat)).items(), key=lambda item: item[1], reverse=True)
    assert len(items) in (2, 3), "right panel must use 2 or 3 colors"
    assert items[0][1] > items[1][1], "right panel must have a unique most-frequent color"
    return int(items[0][0])


def _stamp_output(rotated: np.ndarray) -> np.ndarray:
    dominant = _dominant_color(rotated)
    out = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)
    for rr, cc in np.argwhere(rotated == dominant):
        _paste(out, rotated, int(rr) * 3, int(cc) * 3)
    return out


def _solve_from_count_and_pattern(count: int, right_panel: np.ndarray) -> np.ndarray:
    rotated = np.rot90(np.asarray(right_panel, dtype=int), k=count - 1)
    return _stamp_output(rotated)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the occupied cells in the left nine-cell panel, rotate the right 3x3 pattern, and stamp it into selected 3x3 slots.

    Input structure:
    - A black background contains two horizontal panels separated by one black column.
    - The left panel is an 11x11 orange board with magenta separators forming a 3x3 grid of 3x3 cells.
    - Between one and four cells contain the same red diamond sprite; the other left-panel cells are plain orange.
    - The right panel is a full 3x3 non-black color pattern using two or three colors with one unique most-frequent color.

    Rule:
    - Count how many left-panel cells contain a red diamond sprite; call this count N.
    - Rotate the right 3x3 pattern counterclockwise by (N-1) quarter-turns.
    - In the rotated pattern, find the unique most-frequent color C.
    - Produce a 9x9 black output divided into 3x3 block slots; for every position of color C in the rotated 3x3 pattern, copy the full rotated 3x3 pattern into the corresponding block slot.

    Key insight:
    - The left panel supplies only the count, but that count controls the right panel's rotation, which changes both the stamp's orientation and which 3x3 output slots are activated by the most-frequent color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    left_top = _find_left_panel(g)
    left_r, left_c = left_top
    left_panel = _crop_bbox(g, (left_r, left_c, left_r + 10, left_c + 10))
    count = _count_sprites(left_panel)
    right_panel = _find_right_panel(g, left_top)
    return _solve_from_count_and_pattern(count, right_panel)


def _build_left_panel(occupied_indices: np.ndarray) -> np.ndarray:
    panel = np.full((LEFT_PANEL_SIZE, LEFT_PANEL_SIZE), ORANGE, dtype=int)
    panel[3, :] = PINK
    panel[7, :] = PINK
    panel[:, 3] = PINK
    panel[:, 7] = PINK

    for idx in occupied_indices:
        rr = int(idx) // 3
        cc = int(idx) % 3
        r0 = rr * 4
        c0 = cc * 4
        for dr, dc in DIAMOND_COORDS:
            panel[r0 + dr, c0 + dc] = RED
    return panel


def _sample_right_pattern(rng: np.random.Generator, color_count: int) -> np.ndarray:
    colors = rng.choice(RIGHT_COLORS, size=color_count, replace=False)
    if color_count == 2:
        counts = [5, 4]
    else:
        counts = [4, 3, 2] if int(rng.integers(0, 2)) == 0 else [5, 2, 2]

    cells = np.arange(9)
    rng.shuffle(cells)
    flat = np.empty(9, dtype=int)
    start = 0
    for color, count in zip(colors, counts):
        flat[cells[start : start + count]] = int(color)
        start += count
    return flat.reshape(3, 3)


def _mask_sig(mask: np.ndarray) -> bytes:
    return mask.astype(np.uint8).tobytes()


def _right_pattern_is_informative(pattern: np.ndarray) -> bool:
    dominant = _dominant_color(pattern)
    mask_sigs = set()
    out_sigs = set()
    for count in range(1, 5):
        rotated = np.rot90(pattern, k=count - 1)
        mask_sigs.add(_mask_sig(rotated == dominant))
        out_sigs.add(_grid_sig(_stamp_output(rotated)))
    return len(mask_sigs) >= 2 and len(out_sigs) == 4


def _validate_generated_sample(grid: np.ndarray, expected_count: int, right_pattern: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    left_top = _find_left_panel(grid)
    left_r, left_c = left_top
    left_panel = _crop_bbox(grid, (left_r, left_c, left_r + 10, left_c + 10))
    if _count_sprites(left_panel) != expected_count:
        return False

    parsed_right = _find_right_panel(grid, left_top)
    if not np.array_equal(parsed_right, right_pattern):
        return False

    expected = _solve_from_count_and_pattern(expected_count, right_pattern)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    wrong_outputs = {
        _grid_sig(_solve_from_count_and_pattern(other, right_pattern))
        for other in range(1, 5)
        if other != expected_count
    }
    if _grid_sig(actual) in wrong_outputs:
        return False
    return True


def generate_input() -> np.ndarray:
    global _GEN_CALLS
    rng = np.random.default_rng()

    target_count = (_GEN_CALLS % 4) + 1
    target_color_count = 2 if ((_GEN_CALLS // 4) % 2 == 0) else 3
    _GEN_CALLS += 1

    for _ in range(600):
        occupied = rng.choice(9, size=target_count, replace=False)
        left_panel = _build_left_panel(occupied)

        right_pattern = _sample_right_pattern(rng, target_color_count)
        if not _right_pattern_is_informative(right_pattern):
            continue

        top_margin = int(rng.integers(0, 20))
        bottom_margin = int(rng.integers(0, 20 - top_margin))
        left_margin = int(rng.integers(0, 16))
        right_margin = int(rng.integers(0, 16 - left_margin))
        right_row_offset = int(rng.integers(0, 9))

        height = top_margin + LEFT_PANEL_SIZE + bottom_margin
        width = left_margin + LEFT_PANEL_SIZE + 1 + RIGHT_PANEL_SIZE + right_margin
        grid = np.zeros((height, width), dtype=int)

        _paste(grid, left_panel, top_margin, left_margin)
        _paste(grid, right_pattern, top_margin + right_row_offset, left_margin + LEFT_PANEL_SIZE + 1)

        if not _validate_generated_sample(grid, target_count, right_pattern):
            continue

        in_sig = _grid_sig(grid)
        out_sig = _grid_sig(main(grid))
        if in_sig in _SEEN_INPUT_SIGS or out_sig in _SEEN_OUTPUT_SIGS:
            continue
        _SEEN_INPUT_SIGS.add(in_sig)
        _SEEN_OUTPUT_SIGS.add(out_sig)
        return grid

    raise RuntimeError("failed to generate a valid cross-panel counting stamp task")
