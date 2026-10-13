from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
NOTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

BAND_HEIGHT = 4
COLOR_COUNT_RANGE = (2, 6)
LEFT_WIDTH_RANGE = (5, 8)
TOP_PAD_RANGE = (0, 2)
BOTTOM_PAD_RANGE = (0, 2)
INTER_BAND_GAP_RANGE = (1, 2)
SCORE_WIDTH_RANGE = (6, 18)
GENERATION_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if np.all(col == col[0]) and int(col[0]) != bg:
            candidates.append((c, int(col[0])))
    assert len(candidates) == 1, "expected a unique full-height separator column"
    return candidates[0]


def _decode_mapping(left: np.ndarray, bg: int, grey: int) -> list[tuple[int, int]]:
    color_points: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere((left != bg) & (left != grey)):
        color = int(left[r, c])
        color_points.setdefault(color, []).append((int(r), int(c)))

    assert color_points, "expected at least one mapping entry"
    entries = []
    for color, pts in color_points.items():
        assert len(pts) == 1, "each mapping color should appear exactly once on the left"
        r, c = pts[0]
        entries.append((r, c, color))
    entries.sort()

    square_starts = []
    for comp in _components(left == grey):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0 = min(rows)
        r1 = max(rows)
        c0 = min(cols)
        c1 = max(cols)
        assert len(comp) == 4 and r1 == r0 + 1 and c1 == c0 + 1, "grey legend marks must be 2x2 squares"
        square_starts.append((r0, c0))
    square_starts.sort()

    assert len(square_starts) == len(entries), "expected one grey square per color entry"

    mapping: list[tuple[int, int]] = []
    for (pix_r, pix_c, color), (sq_r, sq_c) in zip(entries, square_starts):
        assert pix_c < sq_c, "color marker should be left of its grey square"
        level = sq_r - pix_r + 1
        assert 0 <= level <= 2, "square offset must encode high/mid/low"
        mapping.append((color, int(level)))

    return mapping


def _render_output(mapping: list[tuple[int, int]], score: np.ndarray, bg: int, grey: int) -> np.ndarray:
    width = score.shape[1]
    out = np.full((3 * len(mapping), width), BLACK, dtype=int)
    by_color = {color: (i, level) for i, (color, level) in enumerate(mapping)}

    for c in range(width):
        notes = {int(v) for v in score[:, c] if int(v) not in (bg, grey)}
        assert notes <= set(by_color), "score uses colors not declared in the left legend"
        for color in notes:
            band, level = by_color[color]
            out[3 * band + level, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a left-side color-to-pitch legend, then translate the right-side score into piano-key hits.

    The input has a grey separator column. On the left, each color appears once and is paired with
    one grey 2x2 square; the square's vertical offset relative to the color marker encodes pitch
    level 0/1/2. On the right, each column is a time step containing zero or more colored notes.
    The output has three rows per legend color, ordered by the legend top-to-bottom.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    sep_col, grey = _find_separator(g, bg)
    left = g[:, :sep_col]
    score = g[:, sep_col + 1 :]

    assert left.size > 0 and score.size > 0, "separator must split legend and score areas"

    mapping = _decode_mapping(left, bg, grey)
    return _render_output(mapping, score, bg, grey)


def _sample_levels(count: int) -> list[int]:
    if count == 1:
        return [random.randrange(3)]
    if count == 2:
        return random.sample([0, 1, 2], 2)

    levels = [0, 1, 2]
    levels.extend(random.randrange(3) for _ in range(count - 3))
    random.shuffle(levels)
    return levels


def _make_column_subsets(colors: list[int], width: int) -> list[set[int]]:
    subsets = [set() for _ in range(width)]

    empty_idx = random.randrange(width)
    single_idx = random.randrange(width)
    while single_idx == empty_idx:
        single_idx = random.randrange(width)

    multi_idx = random.randrange(width)
    while multi_idx in (empty_idx, single_idx):
        multi_idx = random.randrange(width)

    single_color = random.choice(colors)
    subsets[single_idx] = {single_color}
    if len(colors) > 1:
        size = random.randint(2, min(len(colors), 4))
        subsets[multi_idx] = set(random.sample(colors, size))
    else:
        subsets[multi_idx].add(colors[0])

    for color in colors:
        if color == single_color:
            continue
        choices = [idx for idx in range(width) if idx not in (empty_idx, single_idx)]
        idx = random.choice(choices)
        subsets[idx].add(color)

    for idx in range(width):
        if idx == empty_idx:
            subsets[idx] = set()
            continue
        if idx == single_idx:
            subsets[idx] = {single_color}
            continue
        if subsets[idx]:
            if len(colors) > 1 and random.random() < 0.35:
                target = random.randint(2, min(len(colors), 4))
                subsets[idx].update(random.sample(colors, target))
            continue

        roll = random.random()
        if roll < 0.25:
            subsets[idx] = set()
        elif roll < 0.65:
            subsets[idx] = {random.choice(colors)}
        else:
            size = random.randint(2, min(len(colors), 4))
            subsets[idx] = set(random.sample(colors, size))

    assert any(len(notes) == 0 for notes in subsets)
    assert any(len(notes) == 1 for notes in subsets)
    if len(colors) > 1:
        assert any(len(notes) >= 2 for notes in subsets)
    assert set().union(*subsets) == set(colors)
    return subsets


def _expected_from_layout(mapping: list[tuple[int, int]], column_subsets: list[set[int]]) -> np.ndarray:
    out = np.full((3 * len(mapping), len(column_subsets)), BLACK, dtype=int)
    by_color = {color: (i, level) for i, (color, level) in enumerate(mapping)}
    for c, notes in enumerate(column_subsets):
        for color in notes:
            band, level = by_color[color]
            out[3 * band + level, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*COLOR_COUNT_RANGE)
    colors = random.sample(list(NOTE_COLORS), count)
    levels = _sample_levels(count)
    mapping = list(zip(colors, levels, strict=True))

    left_width = random.randint(*LEFT_WIDTH_RANGE)
    top_pad = random.randint(*TOP_PAD_RANGE)
    bottom_pad = random.randint(*BOTTOM_PAD_RANGE)
    gaps = [random.randint(*INTER_BAND_GAP_RANGE) for _ in range(count - 1)]
    height = top_pad + count * BAND_HEIGHT + sum(gaps) + bottom_pad
    score_width = random.randint(*SCORE_WIDTH_RANGE)

    grid = np.full((height, left_width + 1 + score_width), BLACK, dtype=int)
    grid[:, left_width] = GREY

    band_top = top_pad
    max_square_col = left_width - 3
    if max_square_col < 2:
        return None

    for index, (color, level) in enumerate(mapping):
        grid[band_top + 1, 0] = color
        square_col = random.randint(2, max_square_col)
        grid[band_top + level : band_top + level + 2, square_col : square_col + 2] = GREY
        if index < count - 1:
            band_top += BAND_HEIGHT + gaps[index]

    column_subsets = _make_column_subsets(colors, score_width)
    for c, notes in enumerate(column_subsets, start=left_width + 1):
        if not notes:
            continue
        rows = random.sample(range(height), len(notes))
        for r, color in zip(rows, notes, strict=True):
            grid[r, c] = color

    expected = _expected_from_layout([(color, level) for color, level in zip(colors, levels, strict=True)], column_subsets)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid score-to-keyboard puzzle")
