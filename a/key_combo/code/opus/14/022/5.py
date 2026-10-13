from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

PANEL_HEIGHT = 5
GRID_HEIGHT = PANEL_HEIGHT * 2 + 1
WIDTH_MIN = 8
WIDTH_MAX = 16
BAR_LENGTHS = (3, 4, 5)  # generator always forces wrapping; solver still accepts n=2
MASK_AREA_RANGE = (8, 20)
GEN_ATTEMPTS = 500

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_SEEN_INPUTS: set[bytes] = set()
_SEEN_OUTPUTS: set[bytes] = set()


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _sequence_values(n: int, fg: int = BLUE, bg: int = BG) -> list[int]:
    run_lengths = list(range(1, n + 1)) + list(range(n - 1, 0, -1))
    seq: list[int] = []
    for i, run_len in enumerate(run_lengths):
        seq.extend([fg] * int(run_len))
        if i != len(run_lengths) - 1:
            seq.append(bg)
    return seq


def _build_sequence_mask(n: int, height: int, width: int) -> np.ndarray:
    seq = _sequence_values(n)
    assert len(seq) <= height * width, "panel must fit the wrapped sequence"
    out = np.full((height, width), BG, dtype=int)
    out.ravel()[: len(seq)] = np.asarray(seq, dtype=int)
    return out


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    colors, counts = np.unique(grid, return_counts=True)
    count_by_color = {int(color): int(count) for color, count in zip(colors, counts)}
    candidates = []
    for r in range(h):
        color = int(grid[r, 0])
        if color == bg:
            continue
        if np.all(grid[r] == color) and count_by_color[color] == w:
            candidates.append(r)
    assert len(candidates) == 1, "expected exactly one unique full-width separator row"
    return int(candidates[0])


def _read_bar_length(top: np.ndarray) -> int:
    first_row = top[0]
    n = 0
    while n < first_row.size and int(first_row[n]) == BLUE:
        n += 1
    assert 2 <= n <= first_row.size, "expected a left-aligned blue control bar"
    assert np.all(first_row[:n] == BLUE), "control bar must be contiguous"
    assert np.all(first_row[n:] == BG), "top-row pixels after the control bar must be background"
    assert np.all(top[1:] == BG), "the rest of the upper panel must be background"
    return n


def main(grid: np.ndarray) -> np.ndarray:
    """
    Intersect the wrapped control-bar sequence mask with the lower-panel mask.

    Input structure:
    - The grid has black background and one full-width red separator row splitting it into two equal panels.
    - The upper panel contains only a left-aligned blue bar in its first row; the bar length is the control value.
    - The lower panel contains a yellow foreground mask on black background, aligned positionally with the upper panel.

    Rule:
    - Read the blue bar length n from the upper panel.
    - Build the one-dimensional run sequence 1, 2, ..., n, n-1, ..., 1 with one black cell between adjacent blue runs, then wrap it row-major using the panel width to form a blue mask of the upper-panel size.
    - Remove the separator and output a panel-sized grid with green at cells where the wrapped blue mask and the lower yellow mask are both foreground, and black elsewhere.

    Key insight:
    - The output is neither the wrapped sequence alone nor the lower mask alone: the panel width determines the 2D wrapped sequence shape, and only its boolean intersection with the aligned lower mask survives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    assert bg == BG, "expected black background"

    sep_row = _find_separator_row(g, bg)
    top = g[:sep_row]
    bottom = g[sep_row + 1 :]
    assert top.shape == bottom.shape and top.size > 0, "separator must split two equal non-empty panels"
    assert int(g[sep_row, 0]) == RED, "separator row must be red"

    top_nonbg = set(map(int, np.unique(top))) - {BG}
    bottom_nonbg = set(map(int, np.unique(bottom))) - {BG}
    assert top_nonbg <= {BLUE}, "upper panel may only contain the blue control bar"
    assert bottom_nonbg <= {YELLOW}, "lower panel may only contain the yellow mask"

    n = _read_bar_length(top)
    seq_mask = _build_sequence_mask(n, top.shape[0], top.shape[1]) == BLUE
    lower_mask = bottom == YELLOW

    out = np.full(top.shape, BG, dtype=int)
    out[seq_mask & lower_mask] = GREEN
    return out


def _bbox_area(cells: set[tuple[int, int]]) -> int:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)


def _sample_width_for_n(n: int) -> int:
    seq_len = len(_sequence_values(n))
    low = max(WIDTH_MIN, n + 3)
    high = min(WIDTH_MAX, seq_len - 1)
    assert low <= high, "generator must force at least one wrap"
    return random.randint(low, high)


def _grow_mask(seq_mask: np.ndarray, target_area: int, target_overlap: int) -> set[tuple[int, int]] | None:
    h, w = seq_mask.shape
    seq_cells = [(int(r), int(c)) for r, c in np.argwhere(seq_mask)]
    seed = random.choice(seq_cells)
    cells = {seed}
    overlap = 1

    while len(cells) < target_area:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None

        candidates = list(frontier)
        weights = []
        for cand in candidates:
            in_seq = bool(seq_mask[cand])
            weight = 1.0
            if overlap < target_overlap and in_seq:
                weight += 4.0
            if overlap >= target_overlap and not in_seq:
                weight += 3.0
            shared = sum((nr, nc) in cells for nr, nc in _neighbors4(*cand))
            weight += 0.35 * shared
            if cand[0] > 0:
                weight += 0.2
            weights.append(weight)

        choice = random.choices(candidates, weights=weights, k=1)[0]
        cells.add(choice)
        overlap += int(seq_mask[choice])

    return cells


def _shape_is_irregular(mask: np.ndarray) -> bool:
    cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    if len(cells) < MASK_AREA_RANGE[0]:
        return False
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) < 2 or len(cols) < 3:
        return False
    bbox_area = _bbox_area(cells)
    if bbox_area - len(cells) < 2:
        return False
    for r in rows:
        row_cells = sorted(c for rr, c in cells if rr == r)
        if len(row_cells) >= 3 and row_cells[-1] - row_cells[0] + 1 > len(row_cells):
            return True
    return len(rows) >= 3


def _gap_negative_count(seq_mask: np.ndarray, lower_mask: np.ndarray) -> int:
    count = 0
    for r in range(seq_mask.shape[0]):
        cols = np.flatnonzero(seq_mask[r])
        if len(cols) < 2:
            continue
        for c in range(int(cols[0]), int(cols[-1]) + 1):
            if not seq_mask[r, c] and lower_mask[r, c]:
                count += 1
    return count


def _validate_candidate(grid: np.ndarray) -> bool:
    if grid.shape[0] != GRID_HEIGHT:
        return False
    out = main(grid)
    sep = PANEL_HEIGHT
    top = grid[:sep]
    bottom = grid[sep + 1 :]
    n = _read_bar_length(top)
    seq_mask = _build_sequence_mask(n, PANEL_HEIGHT, grid.shape[1]) == BLUE
    lower_mask = bottom == YELLOW
    overlap = seq_mask & lower_mask

    if len(_sequence_values(n)) <= grid.shape[1]:
        return False
    if _component_count(lower_mask) != 1:
        return False
    if not _shape_is_irregular(lower_mask):
        return False

    overlap_count = int(np.count_nonzero(overlap))
    lower_count = int(np.count_nonzero(lower_mask))
    seq_count = int(np.count_nonzero(seq_mask))
    if not (3 <= overlap_count < lower_count and overlap_count < seq_count):
        return False
    if np.count_nonzero(lower_mask & ~seq_mask) < 3:
        return False
    if np.count_nonzero(seq_mask & ~lower_mask) < 3:
        return False
    if not np.any(np.argwhere(overlap)[:, 0] > 0):
        return False
    if _gap_negative_count(seq_mask, lower_mask) < 1:
        return False
    if np.array_equal(out == GREEN, seq_mask) or np.array_equal(out == GREEN, lower_mask):
        return False
    if np.count_nonzero(out) == 0 or np.count_nonzero(out) == out.size:
        return False

    return True


def _render_input(width: int, n: int, lower_mask: np.ndarray) -> np.ndarray:
    grid = np.zeros((GRID_HEIGHT, width), dtype=int)
    grid[PANEL_HEIGHT, :] = RED
    grid[0, :n] = BLUE
    lower = grid[PANEL_HEIGHT + 1 :, :]
    lower[lower_mask] = YELLOW
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(BAR_LENGTHS)
    width = _sample_width_for_n(n)
    seq_mask = _build_sequence_mask(n, PANEL_HEIGHT, width) == BLUE
    seq_count = int(np.count_nonzero(seq_mask))

    for _ in range(100):
        target_area = random.randint(*MASK_AREA_RANGE)
        max_overlap = min(seq_count - 3, target_area - 3, 10)
        if max_overlap < 3:
            continue
        target_overlap = random.randint(3, max_overlap)
        cells = _grow_mask(seq_mask, target_area, target_overlap)
        if cells is None:
            continue

        lower_mask = np.zeros((PANEL_HEIGHT, width), dtype=bool)
        for r, c in cells:
            lower_mask[r, c] = True

        grid = _render_input(width, n, lower_mask)
        if not _validate_candidate(grid):
            continue

        inp_sig = grid.tobytes() + str(grid.shape).encode()
        out = main(grid)
        out_sig = out.tobytes() + str(out.shape).encode()
        if inp_sig in _SEEN_INPUTS or out_sig in _SEEN_OUTPUTS:
            continue

        _SEEN_INPUTS.add(inp_sig)
        _SEEN_OUTPUTS.add(out_sig)
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid

    _SEEN_INPUTS.clear()
    _SEEN_OUTPUTS.clear()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wrapped-mask intersection example")
