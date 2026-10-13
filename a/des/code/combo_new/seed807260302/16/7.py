from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
PATH_COLORS = (BLUE, RED, GREY)
STRIP_COLORS = (3, 4, 6, 7, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for nr, nc in _neighbors(mask.shape, cr, cc):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_color_strip(g: np.ndarray) -> list[int]:
    h, w = g.shape
    strip_mask = np.zeros_like(g, dtype=bool)
    forbidden = {BLACK, BLUE, RED, GREY}

    for c in range(w):
        if int(g[0, c]) not in forbidden:
            strip_mask[0, c] = True
        if int(g[h - 1, c]) not in forbidden:
            strip_mask[h - 1, c] = True
    for r in range(h):
        if int(g[r, 0]) not in forbidden:
            strip_mask[r, 0] = True
        if int(g[r, w - 1]) not in forbidden:
            strip_mask[r, w - 1] = True

    sequences: list[list[int]] = []
    for comp in _components(strip_mask):
        rows = {r for r, _c in comp}
        cols = {c for _r, c in comp}

        if len(rows) == 1:
            row = next(iter(rows))
            ordered_cols = sorted(c for _r, c in comp)
            assert ordered_cols == list(range(ordered_cols[0], ordered_cols[-1] + 1))
            sequences.append([int(g[row, c]) for c in ordered_cols])
            continue

        if len(cols) == 1:
            col = next(iter(cols))
            ordered_rows = sorted(r for r, _c in comp)
            assert ordered_rows == list(range(ordered_rows[0], ordered_rows[-1] + 1))
            sequences.append([int(g[r, col]) for r in ordered_rows])
            continue

        raise AssertionError("color strip must be a straight edge segment")

    assert len(sequences) == 1, "expected exactly one edge color strip"
    sequence = sequences[0]
    assert sequence and len(sequence) == len(set(sequence))
    return sequence


def _ordered_path(g: np.ndarray, comp: list[tuple[int, int]]) -> list[tuple[int, int]]:
    coord_set = set(comp)
    blue_cells = [(r, c) for r, c in comp if int(g[r, c]) == BLUE]
    assert len(blue_cells) == 1, "each path must have one blue start endpoint"

    degrees: dict[tuple[int, int], int] = {}
    for r, c in comp:
        degree = sum((nr, nc) in coord_set for nr, nc in _neighbors(g.shape, r, c))
        assert degree <= 2, "paths may not branch"
        degrees[(r, c)] = degree

    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(comp) >= 2 and len(endpoints) == 2, "path must have exactly two endpoints"
    assert blue_cells[0] in endpoints, "blue start must lie at an endpoint"
    assert all(int(g[r, c]) != RED for r, c in endpoints), "red markers must be interior path cells"

    ordered: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    current = blue_cells[0]
    while True:
        ordered.append(current)
        choices = [
            nbr
            for nbr in _neighbors(g.shape, current[0], current[1])
            if nbr in coord_set and nbr != prev
        ]
        if not choices:
            break
        assert len(choices) == 1, "path traversal must be unambiguous"
        prev, current = current, choices[0]

    assert len(ordered) == len(comp), "path traversal must visit every path cell exactly once"
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each marked path into ordered color segments using the edge strip as the palette.

    Input structure:
    - The background is black.
    - One straight strip of distinct non-black, non-grey, non-blue, non-red pixels sits on the outer edge and gives the segment color order.
    - The remaining foreground consists of 4-connected, one-pixel-wide grey paths with exactly one blue endpoint and zero or more interior red marker pixels on each path.

    Rule:
    - For every path, start at its blue endpoint and traverse the unique 4-connected path order until the opposite endpoint.
    - Recolor all cells before the first red marker with the first strip color; whenever a red marker is reached, switch to the next strip color and recolor that red cell with the new segment color.
    - Continue switching colors at each later red marker so grey, blue, and red path cells all disappear, while the edge color strip stays unchanged.

    Key insight:
    - Blue fixes the traversal direction, red marks segment boundaries, and the edge strip supplies the visible sequence of colors used to replace the entire path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sequence = _find_color_strip(g)

    path_mask = np.isin(g, PATH_COLORS)
    components = _components(path_mask)
    assert components, "expected at least one marked path"

    out = g.copy()
    for comp in components:
        ordered = _ordered_path(g, comp)
        segment = 0
        for r, c in ordered:
            if int(g[r, c]) == RED:
                segment += 1
            assert segment < len(sequence), "color strip must be long enough for every path"
            out[r, c] = sequence[segment]

    return out


# --------------- generator ---------------

GEN_MIN_SIZE = 12
GEN_MAX_SIZE = 30
GEN_MAX_ATTEMPTS = 400
GEN_SEGMENT_LEN_MIN = 2
GEN_SEGMENT_LEN_MAX = 4
GEN_START_TRIES = 80
GEN_PATH_COUNT_MIN = 1
GEN_PATH_COUNT_MAX = 3


def _usable_ranges(h: int, w: int, strip_side: str) -> tuple[range, range]:
    if strip_side == "top":
        rows = range(2, h - 1)
    else:
        rows = range(1, h - 2)
    cols = range(1, w - 1)
    return rows, cols


def _touches_other_path(cell: tuple[int, int], occupied_other: set[tuple[int, int]]) -> bool:
    r, c = cell
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if (rr, cc) in occupied_other:
                return True
    return False


def _is_usable_cell(
    cell: tuple[int, int],
    usable_rows: set[int],
    usable_cols: set[int],
    occupied_other: set[tuple[int, int]],
) -> bool:
    r, c = cell
    return r in usable_rows and c in usable_cols and not _touches_other_path(cell, occupied_other)


def _valid_next_cell(
    cell: tuple[int, int],
    current: tuple[int, int],
    path_set: set[tuple[int, int]],
    occupied_other: set[tuple[int, int]],
    usable_rows: set[int],
    usable_cols: set[int],
) -> bool:
    if cell in path_set or not _is_usable_cell(cell, usable_rows, usable_cols, occupied_other):
        return False
    for nbr in _neighbors((max(usable_rows) + 2, max(usable_cols) + 2), cell[0], cell[1]):
        if nbr in path_set and nbr != current:
            return False
    return True


def _candidate_steps(
    current: tuple[int, int],
    path_set: set[tuple[int, int]],
    occupied_other: set[tuple[int, int]],
    usable_rows: set[int],
    usable_cols: set[int],
    prev_dir: tuple[int, int] | None,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    dirs = list(DIRS)
    random.shuffle(dirs)
    for dr, dc in dirs:
        nxt = (current[0] + dr, current[1] + dc)
        if _valid_next_cell(nxt, current, path_set, occupied_other, usable_rows, usable_cols):
            candidates.append((nxt, (dr, dc)))

    def onward_score(item: tuple[tuple[int, int], tuple[int, int]]) -> tuple[int, int, float]:
        nxt, direction = item
        next_path = path_set | {nxt}
        onward = 0
        for dr, dc in DIRS:
            probe = (nxt[0] + dr, nxt[1] + dc)
            if _valid_next_cell(probe, nxt, next_path, occupied_other, usable_rows, usable_cols):
                onward += 1
        straight = int(prev_dir is not None and direction == prev_dir)
        return (onward, straight, random.random())

    candidates.sort(key=onward_score, reverse=True)
    return candidates


def _search_path(
    current: tuple[int, int],
    path: list[tuple[int, int]],
    path_set: set[tuple[int, int]],
    target_len: int,
    occupied_other: set[tuple[int, int]],
    usable_rows: set[int],
    usable_cols: set[int],
    prev_dir: tuple[int, int] | None,
) -> bool:
    if len(path) == target_len:
        return True

    for nxt, direction in _candidate_steps(
        current, path_set, occupied_other, usable_rows, usable_cols, prev_dir
    ):
        path.append(nxt)
        path_set.add(nxt)
        if _search_path(
            nxt,
            path,
            path_set,
            target_len,
            occupied_other,
            usable_rows,
            usable_cols,
            direction,
        ):
            return True
        path.pop()
        path_set.remove(nxt)
    return False


def _build_path(
    h: int,
    w: int,
    strip_side: str,
    target_len: int,
    occupied_other: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    row_range, col_range = _usable_ranges(h, w, strip_side)
    usable_rows = set(row_range)
    usable_cols = set(col_range)

    starts = [
        (r, c)
        for r in row_range
        for c in col_range
        if _is_usable_cell((r, c), usable_rows, usable_cols, occupied_other)
    ]
    random.shuffle(starts)
    for start in starts[:GEN_START_TRIES]:
        path = [start]
        path_set = {start}
        if _search_path(
            start,
            path,
            path_set,
            target_len,
            occupied_other,
            usable_rows,
            usable_cols,
            None,
        ):
            return path
    return None


def _sample_markers(marker_count: int) -> list[int]:
    segment_lengths = [
        random.randint(GEN_SEGMENT_LEN_MIN, GEN_SEGMENT_LEN_MAX)
        for _ in range(marker_count + 1)
    ]
    total = 0
    markers: list[int] = []
    for seg_len in segment_lengths[:-1]:
        total += seg_len
        markers.append(total)
    return markers


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], marker_indices: list[int]) -> None:
    marker_set = set(marker_indices)
    for i, (r, c) in enumerate(path):
        if i == 0:
            grid[r, c] = BLUE
        elif i in marker_set:
            grid[r, c] = RED
        else:
            grid[r, c] = GREY


def _expected_from_specs(
    grid: np.ndarray,
    sequence: list[int],
    path_specs: list[tuple[list[tuple[int, int]], list[int]]],
) -> np.ndarray:
    out = grid.copy()
    for path, marker_indices in path_specs:
        marker_set = set(marker_indices)
        segment = 0
        for i, (r, c) in enumerate(path):
            if i in marker_set:
                segment += 1
            out[r, c] = sequence[segment]
    return out


def _build_attempt() -> np.ndarray | None:
    h = random.randint(GEN_MIN_SIZE, GEN_MAX_SIZE)
    w = random.randint(GEN_MIN_SIZE, GEN_MAX_SIZE)
    strip_side = random.choice(("top", "bottom"))

    max_seq_len = min(len(STRIP_COLORS), 6)
    seq_len = random.randint(2, max_seq_len)
    sequence = random.sample(list(STRIP_COLORS), seq_len)
    path_count = random.randint(GEN_PATH_COUNT_MIN, GEN_PATH_COUNT_MAX)
    marker_counts = [seq_len - 1]
    for _ in range(path_count - 1):
        marker_counts.append(random.randint(1, seq_len - 1))
    random.shuffle(marker_counts)

    grid = np.full((h, w), BLACK, dtype=int)
    strip_row = 0 if strip_side == "top" else h - 1
    strip_start = random.randint(1, w - seq_len - 1)
    grid[strip_row, strip_start : strip_start + seq_len] = np.array(sequence, dtype=int)

    occupied: set[tuple[int, int]] = set()
    path_specs: list[tuple[list[tuple[int, int]], list[int]]] = []
    for marker_count in sorted(marker_counts, reverse=True):
        marker_indices = _sample_markers(marker_count)
        target_len = marker_indices[-1] + random.randint(GEN_SEGMENT_LEN_MIN, GEN_SEGMENT_LEN_MAX)
        path = _build_path(h, w, strip_side, target_len, occupied)
        if path is None:
            return None
        _paint_path(grid, path, marker_indices)
        occupied.update(path)
        path_specs.append((path, marker_indices))

    expected = _expected_from_specs(grid, sequence, path_specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid segmented-path ARC input")
