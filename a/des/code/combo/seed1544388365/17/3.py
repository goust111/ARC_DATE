from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _separator_color(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    best_color = int(g[0, 0])
    best_score = -1
    best_count = -1
    for color in np.unique(g):
        color = int(color)
        row_hits = int(np.count_nonzero(np.all(g == color, axis=1)))
        col_hits = int(np.count_nonzero(np.all(g == color, axis=0)))
        score = row_hits + col_hits
        count = int(np.count_nonzero(g == color))
        if (score, count) > (best_score, best_count):
            best_color = color
            best_score = score
            best_count = count
    assert best_score >= 4, "expected separator color with multiple full rows/cols"
    return best_color


def _region_bboxes(grid: np.ndarray, sep: int) -> list[tuple[int, int, int, int]]:
    rows = [int(r) for r in np.where(np.all(grid == sep, axis=1))[0]]
    cols = [int(c) for c in np.where(np.all(grid == sep, axis=0))[0]]
    assert len(rows) >= 3 and len(cols) >= 3

    regions: list[tuple[int, int, int, int]] = []
    for r0_line, r1_line in zip(rows, rows[1:]):
        if r1_line - r0_line <= 1:
            continue
        for c0_line, c1_line in zip(cols, cols[1:]):
            if c1_line - c0_line <= 1:
                continue
            regions.append((r0_line + 1, c0_line + 1, r1_line - 1, c1_line - 1))
    assert regions
    return regions


def _extract_dictionary_patterns(
    patch: np.ndarray,
    sep: int,
    bg: int,
    pattern_color: int,
) -> list[np.ndarray]:
    boxes: list[tuple[int, int, int, int]] = []
    for comp in _components(patch == sep):
        r0, c0, r1, c1 = _bbox(comp)
        sub = patch[r0 : r1 + 1, c0 : c1 + 1]
        h, w = sub.shape
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if h == 4 and w == 4 and np.array_equal(sub == sep, border):
            boxes.append((r0, c0, r1, c1))

    assert len(boxes) == 3, f"expected 3 dictionary boxes, got {len(boxes)}"
    boxes.sort(key=lambda b: (b[1], b[0]))

    patterns: list[np.ndarray] = []
    for r0, c0, r1, c1 in boxes:
        inner = patch[r0 + 1 : r1, c0 + 1 : c1]
        assert inner.shape == (2, 2)
        assert set(map(int, np.unique(inner))) <= {bg, pattern_color}
        patterns.append(inner == pattern_color)
    return patterns


def _paint_partition(
    shape: tuple[int, int],
    pattern: np.ndarray,
    bg: int,
    pattern_color: int,
    seed_mask: np.ndarray,
    seed_color: int,
) -> np.ndarray:
    h, w = map(int, shape)
    assert h % 2 == 0 and w % 2 == 0
    tiled = np.tile(pattern, (h // 2, w // 2))
    out = np.full((h, w), bg, dtype=int)
    out[tiled] = pattern_color
    out[seed_mask] = seed_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select a 2x2 dictionary pattern by seed count, then tile it across each partition.

    Input structure:
    - A dominant background color is split by one-cell-wide separator lines into rectangular partitions.
    - One corner partition is a dictionary area containing three framed 2x2 patterns arranged left-to-right.
    - Every other partition contains 1-3 single-pixel seeds of one shared color.

    Rule:
    - Count the seeds in each non-dictionary partition.
    - Use that count (1, 2, or 3) to pick the corresponding dictionary pattern.
    - Tile the chosen 2x2 pattern across the whole partition interior in the dictionary color.
    - Leave seed pixels unchanged.
    """
    g = np.asarray(grid, dtype=int)
    sep = _separator_color(g)

    counts = Counter(map(int, g[g != sep].reshape(-1)))
    assert counts
    bg = int(counts.most_common(1)[0][0])

    regions = _region_bboxes(g, sep)
    dictionary_idx = None
    for idx, (r0, c0, r1, c1) in enumerate(regions):
        if bool(np.any(g[r0 : r1 + 1, c0 : c1 + 1] == sep)):
            assert dictionary_idx is None, "dictionary partition must be unique"
            dictionary_idx = idx
    assert dictionary_idx is not None

    dict_r0, dict_c0, dict_r1, dict_c1 = regions[dictionary_idx]
    dict_patch = g[dict_r0 : dict_r1 + 1, dict_c0 : dict_c1 + 1]
    dict_colors = set(map(int, np.unique(dict_patch))) - {bg, sep}
    assert len(dict_colors) == 1, "dictionary must use exactly one drawing color"
    pattern_color = dict_colors.pop()

    all_colors = set(map(int, np.unique(g))) - {bg, sep}
    assert len(all_colors) == 2, "expected exactly seed color and pattern color"
    seed_color = next(color for color in all_colors if color != pattern_color)

    patterns = _extract_dictionary_patterns(dict_patch, sep, bg, pattern_color)

    out = g.copy()
    for idx, (r0, c0, r1, c1) in enumerate(regions):
        if idx == dictionary_idx:
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        seed_mask = patch == seed_color
        seed_count = int(seed_mask.sum())
        assert 1 <= seed_count <= 3
        out[r0 : r1 + 1, c0 : c1 + 1] = _paint_partition(
            patch.shape,
            patterns[seed_count - 1],
            bg,
            pattern_color,
            seed_mask,
            seed_color,
        )
    return out


_PATTERN_LIBRARY = [
    np.array([[bool((bits >> 0) & 1), bool((bits >> 1) & 1)],
              [bool((bits >> 2) & 1), bool((bits >> 3) & 1)]], dtype=bool)
    for bits in range(1, 16)
]

_GRID_LAYOUTS = [(2, 2), (2, 3), (3, 2), (3, 3)]
_DICT_CORNERS = ("tl", "tr", "bl", "br")
_MAX_ATTEMPTS = 300


def _sample_even(lo: int, hi: int) -> int:
    choices = [value for value in range(lo, hi + 1) if value % 2 == 0]
    return random.choice(choices)


def _split_total(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _layout_sizes() -> tuple[int, int, list[int], list[int], tuple[int, int]]:
    for _ in range(100):
        n_rows, n_cols = random.choice(_GRID_LAYOUTS)
        corner = random.choice(_DICT_CORNERS)
        dict_row = 0 if corner[0] == "t" else n_rows - 1
        dict_col = 0 if corner[1] == "l" else n_cols - 1

        row_heights = []
        for r in range(n_rows):
            if r == dict_row:
                row_heights.append(_sample_even(6, 10))
            else:
                row_heights.append(_sample_even(4, 8))

        col_widths = []
        for c in range(n_cols):
            if c == dict_col:
                col_widths.append(_sample_even(16, 18))
            else:
                col_widths.append(_sample_even(4, 8))

        height = sum(row_heights) + n_rows + 1
        width = sum(col_widths) + n_cols + 1
        if height <= 30 and width <= 30:
            return n_rows, n_cols, row_heights, col_widths, (dict_row, dict_col)
    raise RuntimeError("failed to sample a valid partition layout")


def _cell_bboxes(row_heights: list[int], col_widths: list[int]) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    row_start = 1
    for r, rh in enumerate(row_heights):
        col_start = 1
        for c, cw in enumerate(col_widths):
            boxes[(r, c)] = (row_start, col_start, row_start + rh - 1, col_start + cw - 1)
            col_start += cw + 1
        row_start += rh + 1
    return boxes


def _draw_dictionary(
    patch: np.ndarray,
    sep: int,
    bg: int,
    pattern_color: int,
    patterns: list[np.ndarray],
) -> None:
    h, w = patch.shape
    patch[:, :] = bg

    extra_h = h - 6
    top_extra, bottom_extra = _split_total(extra_h, 2)
    top = 1 + top_extra
    _bottom = 1 + bottom_extra
    assert top + 4 + _bottom == h

    extra_w = w - 16
    left_extra, gap1_extra, gap2_extra, right_extra = _split_total(extra_w, 4)
    left = 1 + left_extra
    gap1 = 1 + gap1_extra
    gap2 = 1 + gap2_extra
    right = 1 + right_extra
    assert left + 4 + gap1 + 4 + gap2 + 4 + right == w

    starts = [
        left,
        left + 4 + gap1,
        left + 4 + gap1 + 4 + gap2,
    ]
    for c0, pattern in zip(starts, patterns):
        patch[top : top + 4, c0 : c0 + 4] = bg
        patch[top, c0 : c0 + 4] = sep
        patch[top + 3, c0 : c0 + 4] = sep
        patch[top : top + 4, c0] = sep
        patch[top : top + 4, c0 + 3] = sep
        inner = patch[top + 1 : top + 3, c0 + 1 : c0 + 3]
        inner[:, :] = bg
        inner[pattern] = pattern_color


def _seed_positions(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(100):
        random.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        for r, c in cells:
            if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == count:
                    return chosen
    return None


def _expected_output(
    grid: np.ndarray,
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    dict_cell: tuple[int, int],
    bg: int,
    seed_color: int,
    pattern_color: int,
    patterns: list[np.ndarray],
) -> np.ndarray:
    out = grid.copy()
    for cell, (r0, c0, r1, c1) in boxes.items():
        if cell == dict_cell:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        seed_mask = patch == seed_color
        count = int(seed_mask.sum())
        out[r0 : r1 + 1, c0 : c1 + 1] = _paint_partition(
            patch.shape,
            patterns[count - 1],
            bg,
            pattern_color,
            seed_mask,
            seed_color,
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    sep = 0
    colors = random.sample([c for c in range(10) if c != sep], 3)
    bg, seed_color, pattern_color = map(int, colors)

    n_rows, n_cols, row_heights, col_widths, dict_cell = _layout_sizes()
    boxes = _cell_bboxes(row_heights, col_widths)

    height = sum(row_heights) + n_rows + 1
    width = sum(col_widths) + n_cols + 1
    grid = np.full((height, width), bg, dtype=int)

    row_lines = [0]
    pos = 0
    for rh in row_heights:
        pos += rh + 1
        row_lines.append(pos)
    col_lines = [0]
    pos = 0
    for cw in col_widths:
        pos += cw + 1
        col_lines.append(pos)

    grid[row_lines, :] = sep
    grid[:, col_lines] = sep

    patterns = random.sample(_PATTERN_LIBRARY, 3)

    dict_r0, dict_c0, dict_r1, dict_c1 = boxes[dict_cell]
    _draw_dictionary(
        grid[dict_r0 : dict_r1 + 1, dict_c0 : dict_c1 + 1],
        sep,
        bg,
        pattern_color,
        patterns,
    )

    normal_cells = [cell for cell in boxes if cell != dict_cell]
    counts = [1, 2, 3]
    counts.extend(random.randint(1, 3) for _ in range(len(normal_cells) - 3))
    random.shuffle(counts)

    for cell, seed_count in zip(normal_cells, counts):
        r0, c0, r1, c1 = boxes[cell]
        positions = _seed_positions(r1 - r0 + 1, c1 - c0 + 1, seed_count)
        if positions is None:
            return None
        for rr, cc in positions:
            grid[r0 + rr, c0 + cc] = seed_color

    expected = _expected_output(grid, boxes, dict_cell, bg, seed_color, pattern_color, patterns)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
