import itertools
import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
CELL_COUNT = 4
CELL_SIZE_OPTIONS = (5, 6)
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _runs_without_stops(length: int, stops: list[int]) -> list[tuple[int, int]]:
    stop_set = set(int(x) for x in stops)
    runs: list[tuple[int, int]] = []
    start = None
    for idx in range(length):
        if idx in stop_set:
            if start is not None:
                runs.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        runs.append((start, length - 1))
    return runs


def _find_lattice(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for color in np.unique(g):
        row_lines = [r for r in range(h) if np.all(g[r, :] == color)]
        col_lines = [c for c in range(w) if np.all(g[:, c] == color)]
        if len(row_lines) < 3 or len(col_lines) < 3:
            continue
        row_cells = _runs_without_stops(h, row_lines)
        col_cells = _runs_without_stops(w, col_lines)
        if len(row_cells) != CELL_COUNT or len(col_cells) != CELL_COUNT:
            continue
        if len({r1 - r0 + 1 for r0, r1 in row_cells}) != 1:
            continue
        if len({c1 - c0 + 1 for c0, c1 in col_cells}) != 1:
            continue
        candidates.append((int(color), row_cells, col_cells))
    assert len(candidates) == 1, "expected a unique separator lattice"
    return candidates[0]


def _cell_view(grid: np.ndarray, row_cells, col_cells, row_idx: int, col_idx: int) -> np.ndarray:
    r0, r1 = row_cells[row_idx]
    c0, c1 = col_cells[col_idx]
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _local_blue_offsets(cell: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = sorted(tuple(map(int, rc)) for rc in np.argwhere(cell == BLUE))
    return tuple(coords)


def _detect_source_frame(cell: np.ndarray) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]] | None:
    allowed = {BLACK, BLUE, RED}
    if not set(map(int, np.unique(cell))).issubset(allowed):
        return None
    red_coords = np.argwhere(cell == RED)
    blue_coords = np.argwhere(cell == BLUE)
    if len(red_coords) == 0 or len(blue_coords) == 0:
        return None

    r0, c0 = red_coords.min(axis=0)
    r1, c1 = red_coords.max(axis=0)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    expected = set()
    for r in range(int(r0), int(r1) + 1):
        expected.add((r, int(c0)))
        expected.add((r, int(c1)))
    for c in range(int(c0), int(c1) + 1):
        expected.add((int(r0), c))
        expected.add((int(r1), c))

    red_set = {tuple(map(int, rc)) for rc in red_coords}
    if red_set != expected:
        return None

    for r, c in blue_coords:
        if not (int(r0) < int(r) < int(r1) and int(c0) < int(c) < int(c1)):
            return None

    blue_offsets = tuple(sorted((int(r), int(c)) for r, c in blue_coords))
    red_offsets = tuple(sorted(red_set))
    return blue_offsets, red_offsets


def main(grid: np.ndarray) -> np.ndarray:
    """Broadcast the framed sprite into same-parity empty lattice cells, then erase the frame.

    Input structure:
    - Grey full-width and full-height separator lines partition the board into a 4x4 lattice of equal open cells on a black background.
    - Exactly 3 or 4 cells contain the same blue 4-connected sprite at the same local offsets.
    - Exactly one occupied cell also contains a red hollow rectangle that fully encloses its blue sprite without overlapping it.

    Rule:
    - Recover the open-cell lattice from the separator lines.
    - Identify the unique occupied cell whose red pixels form a complete rectangular frame around the blue sprite; this is the source cell.
    - Copy the source sprite's local blue offsets into every empty cell whose lattice parity `(row_index + col_index) % 2` matches the source cell.
    - Erase the red frame back to black while leaving all blue sprites and the grey separators unchanged.

    Key insight:
    - The red frame is a one-use control signal: it simultaneously selects the source sprite and selects the checkerboard parity class that receives the broadcasts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep_color, row_cells, col_cells = _find_lattice(g)
    assert sep_color == GREY, "expected grey separator lines"

    source_candidates = []
    occupied_offsets = []
    for row_idx in range(CELL_COUNT):
        for col_idx in range(CELL_COUNT):
            cell = _cell_view(g, row_cells, col_cells, row_idx, col_idx)
            blues = _local_blue_offsets(cell)
            if blues:
                occupied_offsets.append(blues)
            source = _detect_source_frame(cell)
            if source is not None:
                source_candidates.append((row_idx, col_idx, *source))

    assert 3 <= len(occupied_offsets) <= 4, "expected 3 or 4 occupied cells"
    assert len(source_candidates) == 1, "expected exactly one framed source cell"
    source_row, source_col, sprite_offsets, frame_offsets = source_candidates[0]
    assert all(offsets == sprite_offsets for offsets in occupied_offsets), "all sprites must share the same local offsets"

    out = g.copy()
    src_r0 = row_cells[source_row][0]
    src_c0 = col_cells[source_col][0]
    for dr, dc in frame_offsets:
        out[src_r0 + dr, src_c0 + dc] = BLACK

    for row_idx in range(CELL_COUNT):
        for col_idx in range(CELL_COUNT):
            cell = _cell_view(g, row_cells, col_cells, row_idx, col_idx)
            if np.any(cell != BLACK):
                continue
            if (row_idx + col_idx) % 2 != (source_row + source_col) % 2:
                continue
            r0, _r1 = row_cells[row_idx]
            c0, _c1 = col_cells[col_idx]
            for dr, dc in sprite_offsets:
                out[r0 + dr, c0 + dc] = BLUE

    return out


def _is_connected(coords: tuple[tuple[int, int], ...]) -> bool:
    remaining = set(coords)
    start = next(iter(remaining))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in remaining and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen == remaining


def _normalize(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _build_pattern_bank() -> tuple[tuple[tuple[int, int], ...], ...]:
    cells = [(r, c) for r in range(3) for c in range(3)]
    patterns = set()
    for size in range(3, 6):
        for subset in itertools.combinations(cells, size):
            if not _is_connected(subset):
                continue
            norm = _normalize(subset)
            rows = [r for r, _ in norm]
            cols = [c for _, c in norm]
            if max(rows) == 0 or max(cols) == 0:
                continue
            patterns.add(norm)
    return tuple(sorted(patterns))


PATTERN_BANK = _build_pattern_bank()


def _frame_placements(cell_size: int, pattern: tuple[tuple[int, int], ...]) -> list[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]]:
    rows = [r for r, _ in pattern]
    cols = [c for _, c in pattern]
    pat_h = max(rows) + 1
    pat_w = max(cols) + 1
    placements = []
    for gap_top, gap_bottom, gap_left, gap_right in itertools.product((0, 1), repeat=4):
        frame_h = pat_h + 2 + gap_top + gap_bottom
        frame_w = pat_w + 2 + gap_left + gap_right
        if frame_h > cell_size or frame_w > cell_size:
            continue
        for frame_r0 in range(cell_size - frame_h + 1):
            for frame_c0 in range(cell_size - frame_w + 1):
                sprite_r0 = frame_r0 + 1 + gap_top
                sprite_c0 = frame_c0 + 1 + gap_left
                sprite = tuple(sorted((sprite_r0 + r, sprite_c0 + c) for r, c in pattern))
                frame = set()
                frame_r1 = frame_r0 + frame_h - 1
                frame_c1 = frame_c0 + frame_w - 1
                for r in range(frame_r0, frame_r1 + 1):
                    frame.add((r, frame_c0))
                    frame.add((r, frame_c1))
                for c in range(frame_c0, frame_c1 + 1):
                    frame.add((frame_r0, c))
                    frame.add((frame_r1, c))
                placements.append((sprite, tuple(sorted(frame))))
    return placements


def _sample_sprite_layout() -> tuple[int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    while True:
        cell_size = random.choice(CELL_SIZE_OPTIONS)
        pattern = random.choice(PATTERN_BANK)
        placements = _frame_placements(cell_size, pattern)
        if placements:
            sprite, frame = random.choice(placements)
            return cell_size, sprite, frame


def _cell_manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _sample_occupied_layout() -> tuple[list[tuple[int, int]], tuple[int, int]]:
    cells = [(r, c) for r in range(CELL_COUNT) for c in range(CELL_COUNT)]
    for _ in range(1000):
        count = random.choice((3, 4))
        occupied = random.sample(cells, count)
        if any(_cell_manhattan(a, b) < 2 for a, b in itertools.combinations(occupied, 2)):
            continue
        source = random.choice(occupied)
        empty = [cell for cell in cells if cell not in occupied]
        same_empty = [cell for cell in empty if (cell[0] + cell[1]) % 2 == (source[0] + source[1]) % 2]
        opp_empty = [cell for cell in empty if (cell[0] + cell[1]) % 2 != (source[0] + source[1]) % 2]
        same_occupied = [cell for cell in occupied if cell != source and (cell[0] + cell[1]) % 2 == (source[0] + source[1]) % 2]
        opp_occupied = [cell for cell in occupied if (cell[0] + cell[1]) % 2 != (source[0] + source[1]) % 2]
        if len(same_empty) < 2 or len(opp_empty) < 1:
            continue
        if not same_occupied or not opp_occupied:
            continue
        return sorted(occupied), source
    raise RuntimeError("failed to sample a non-degenerate occupied layout")


def _cell_spans(cell_size: int) -> tuple[list[tuple[int, int]], list[int]]:
    spans = []
    separators = []
    cursor = 0
    separators.append(cursor)
    cursor += 1
    for _ in range(CELL_COUNT):
        spans.append((cursor, cursor + cell_size - 1))
        cursor += cell_size
        separators.append(cursor)
        cursor += 1
    return spans, separators


def _render_case(
    cell_size: int,
    sprite_offsets: tuple[tuple[int, int], ...],
    frame_offsets: tuple[tuple[int, int], ...],
    occupied: list[tuple[int, int]],
    source: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    spans, separators = _cell_spans(cell_size)
    side = separators[-1] + 1
    grid = np.full((side, side), BLACK, dtype=int)
    grid[separators, :] = GREY
    grid[:, separators] = GREY

    for row_idx, col_idx in occupied:
        r0, _r1 = spans[row_idx]
        c0, _c1 = spans[col_idx]
        for dr, dc in sprite_offsets:
            grid[r0 + dr, c0 + dc] = BLUE

    src_r0 = spans[source[0]][0]
    src_c0 = spans[source[1]][0]
    for dr, dc in frame_offsets:
        grid[src_r0 + dr, src_c0 + dc] = RED

    out = grid.copy()
    for dr, dc in frame_offsets:
        out[src_r0 + dr, src_c0 + dc] = BLACK

    for row_idx in range(CELL_COUNT):
        for col_idx in range(CELL_COUNT):
            if (row_idx, col_idx) in occupied:
                continue
            if (row_idx + col_idx) % 2 != (source[0] + source[1]) % 2:
                continue
            r0, _r1 = spans[row_idx]
            c0, _c1 = spans[col_idx]
            for dr, dc in sprite_offsets:
                out[r0 + dr, c0 + dc] = BLUE

    return grid, out


def _semantic_checks(grid: np.ndarray, expected: np.ndarray, source: tuple[int, int], occupied: list[tuple[int, int]]) -> None:
    actual = main(grid)
    assert np.array_equal(actual, expected), "main(grid) must match the constructed target"

    sep_color, row_cells, col_cells = _find_lattice(grid)
    assert sep_color == GREY
    source_parity = (source[0] + source[1]) % 2
    empty_same = 0
    empty_opp = 0
    source_count = 0
    source_offsets = None

    for row_idx in range(CELL_COUNT):
        for col_idx in range(CELL_COUNT):
            cell = _cell_view(grid, row_cells, col_cells, row_idx, col_idx)
            if (row_idx, col_idx) not in occupied:
                if np.all(cell == BLACK):
                    if (row_idx + col_idx) % 2 == source_parity:
                        empty_same += 1
                        out_cell = _cell_view(expected, row_cells, col_cells, row_idx, col_idx)
                        assert _local_blue_offsets(out_cell), "same-parity empty cell must receive the sprite"
                    else:
                        empty_opp += 1
                        out_cell = _cell_view(expected, row_cells, col_cells, row_idx, col_idx)
                        assert np.all(out_cell == BLACK), "opposite-parity empty cell must stay empty"
            source_info = _detect_source_frame(cell)
            if source_info is not None:
                source_count += 1
                source_offsets = source_info[0]

    assert source_count == 1, "generated input must contain exactly one framed source"
    assert empty_same >= 2 and empty_opp >= 1
    assert source_offsets is not None


def _gen_attempt() -> np.ndarray | None:
    cell_size, sprite_offsets, frame_offsets = _sample_sprite_layout()
    occupied, source = _sample_occupied_layout()
    grid, expected = _render_case(cell_size, sprite_offsets, frame_offsets, occupied, source)
    _semantic_checks(grid, expected, source, occupied)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid broadcast-with-frame input")
