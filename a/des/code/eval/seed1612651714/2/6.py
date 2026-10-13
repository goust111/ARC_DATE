from __future__ import annotations

import random

import numpy as np


BLACK = 0
SELECTOR = 5
SEPARATOR = 9
TEMPLATE_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)

TEMPLATE_COUNT_RANGE = (2, 5)
PATTERN_WIDTH_RANGE = (3, 8)
TEMPLATE_GAP_RANGE = (1, 2)
LEFT_MARGIN_RANGE = (0, 2)
RIGHT_PAD_RANGE = (0, 3)
TOP_BOTTOM_MARGIN_RANGE = (0, 3)
OUTPUT_EXTRA_ROWS_RANGE = (0, 8)

MAX_GENERATION_ATTEMPTS = 300


def _find_separator_col(g: np.ndarray) -> int:
    candidates: list[int] = []
    height, width = g.shape
    for c in range(width):
        col = g[:, c]
        if int(col[0]) == BLACK:
            continue
        if not np.all(col == col[0]):
            continue
        if c == 0 or c == width - 1:
            continue
        if not np.any(g[:, :c] != BLACK):
            continue
        if not np.any(g[:, c + 1 :] != BLACK):
            continue
        candidates.append(int(c))
    assert len(candidates) == 1, "expected exactly one full-height separator column"
    return candidates[0]


def _extract_templates(left: np.ndarray) -> list[np.ndarray]:
    rows = [int(r) for r in range(left.shape[0]) if np.any(left[r] != BLACK)]
    assert rows, "expected at least one template row"

    templates: list[np.ndarray] = []
    widths: set[int] = set()
    prev_row: int | None = None
    for r in rows:
        if prev_row is not None:
            assert r - prev_row >= 2, "template rows must be separated by black rows"
        prev_row = r

        cols = np.flatnonzero(left[r] != BLACK)
        assert cols.size >= 2, "each template row needs a label plus content"
        start, end = int(cols[0]), int(cols[-1])
        row = left[r, start : end + 1]
        label = int(row[0])
        content = np.asarray(row[1:], dtype=int)

        assert np.all(content != BLACK), "template content must be colored"
        assert int(np.sum(row == label)) == 1, "label color must be unique within the row"
        widths.add(int(content.size))
        templates.append(content.copy())

    assert len(widths) == 1, "all templates must have the same width"
    return templates


def _extract_instruction_matrix(right: np.ndarray, n_templates: int) -> np.ndarray:
    pts = np.argwhere(right != BLACK)
    assert pts.size > 0, "expected a selector matrix on the right side"
    r0, c0 = map(int, pts.min(axis=0))
    r1, c1 = map(int, pts.max(axis=0))
    mat = np.asarray(right[r0 : r1 + 1, c0 : c1 + 1], dtype=int)

    non_black = {int(v) for v in np.unique(mat) if int(v) != BLACK}
    assert len(non_black) == 1, "selector matrix must use a single foreground color"
    assert mat.shape[1] == n_templates, "selector columns must match template count"

    for r in range(mat.shape[0]):
        picks = np.flatnonzero(mat[r] != BLACK)
        assert picks.size == 1, "each selector row must choose exactly one template"
    return mat


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read template rows on the left and a one-hot selector matrix on the right.

    The grid is split by one full-height non-black separator column. Each non-empty
    row on the left is a template row: its first colored cell is a unique label,
    and the remaining colored cells form the template content. On the right, each
    selector row contains exactly one non-black cell whose column chooses the
    corresponding template row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator_col(g)
    templates = _extract_templates(g[:, :sep])
    selectors = _extract_instruction_matrix(g[:, sep + 1 :], len(templates))

    out = np.zeros((selectors.shape[0], templates[0].size), dtype=int)
    for r in range(selectors.shape[0]):
        choice = int(np.flatnonzero(selectors[r] != BLACK)[0])
        out[r] = templates[choice]
    return out


def _random_composition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _build_pattern(width: int, label: int, existing: set[tuple[int, ...]]) -> np.ndarray | None:
    usable = [color for color in TEMPLATE_COLOR_POOL if color != label]
    for _ in range(80):
        segment_count = random.randint(2, min(4, width))
        lengths = _random_composition(width, segment_count)
        colors: list[int] = []
        prev: int | None = None
        for _ in range(segment_count):
            choices = [color for color in usable if color != prev]
            color = random.choice(choices)
            colors.append(color)
            prev = color

        values: list[int] = []
        for color, length in zip(colors, lengths):
            values.extend([color] * length)

        pattern = tuple(values)
        if len(set(pattern)) < 2:
            continue
        if pattern in existing:
            continue
        existing.add(pattern)
        return np.array(pattern, dtype=int)
    return None


def _template_row_positions(count: int) -> list[int]:
    top = random.randint(*TOP_BOTTOM_MARGIN_RANGE)
    gap = random.randint(*TEMPLATE_GAP_RANGE)
    return [top + i * (gap + 1) for i in range(count)]


def _instruction_rows(n_templates: int) -> list[int]:
    out_h = n_templates + random.randint(*OUTPUT_EXTRA_ROWS_RANGE)
    order = list(range(n_templates))
    order.extend(random.randrange(n_templates) for _ in range(out_h - n_templates))
    random.shuffle(order)
    return order


def _expected_output(templates: list[np.ndarray], picks: list[int]) -> np.ndarray:
    return np.array([templates[idx] for idx in picks], dtype=int)


def _gen_attempt() -> np.ndarray | None:
    n_templates = random.randint(*TEMPLATE_COUNT_RANGE)
    pattern_w = random.randint(*PATTERN_WIDTH_RANGE)

    labels = random.sample(TEMPLATE_COLOR_POOL, n_templates)
    patterns_seen: set[tuple[int, ...]] = set()
    templates: list[np.ndarray] = []
    for label in labels:
        pattern = _build_pattern(pattern_w, label, patterns_seen)
        if pattern is None:
            return None
        templates.append(pattern)

    template_rows = _template_row_positions(n_templates)
    picks = _instruction_rows(n_templates)
    instr_top = random.randint(*TOP_BOTTOM_MARGIN_RANGE)

    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    left_right_pad = random.randint(*RIGHT_PAD_RANGE)
    right_left_pad = random.randint(*RIGHT_PAD_RANGE)
    right_right_pad = random.randint(*RIGHT_PAD_RANGE)

    left_w = left_margin + 1 + pattern_w + left_right_pad
    right_w = right_left_pad + n_templates + right_right_pad

    left_h = template_rows[-1] + 1 + random.randint(*TOP_BOTTOM_MARGIN_RANGE)
    right_h = instr_top + len(picks) + random.randint(*TOP_BOTTOM_MARGIN_RANGE)
    height = max(left_h, right_h)
    width = left_w + 1 + right_w

    grid = np.full((height, width), BLACK, dtype=int)
    sep = left_w
    grid[:, sep] = SEPARATOR

    for row, label, pattern in zip(template_rows, labels, templates):
        start = left_margin
        grid[row, start] = int(label)
        grid[row, start + 1 : start + 1 + pattern_w] = pattern

    for i, choice in enumerate(picks):
        grid[instr_top + i, sep + 1 + right_left_pad + choice] = SELECTOR

    expected = _expected_output(templates, picks)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
