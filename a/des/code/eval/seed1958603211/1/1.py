from __future__ import annotations

import random

import numpy as np


BG = 0
DIVIDER = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
TEMPLATE_SIZES = (3, 5, 7)
TEMPLATE_COUNT_RANGE = (3, 5)
ANCHOR_COUNT_RANGE = (2, 4)
MIDDLE_HEIGHT_RANGE = (4, 8)
MASK_RETRIES = 80
ANCHOR_LAYOUT_RETRIES = 160
GENERATION_RETRIES = 300


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _horizontal_midpoint(width: int) -> float:
    return (width - 1) / 2.0


def _find_divider_rows(grid: np.ndarray, bg: int) -> tuple[list[int], int]:
    by_color: dict[int, list[int]] = {}
    for r in range(grid.shape[0]):
        row = grid[r]
        color = int(row[0])
        if color != bg and np.all(row == color):
            by_color.setdefault(color, []).append(r)

    candidates = [(color, rows) for color, rows in by_color.items() if len(rows) == 2]
    assert len(candidates) == 1, "expected exactly one divider color with two full rows"
    divider_color, rows = candidates[0]
    rows = sorted(rows)
    assert 0 < rows[0] < rows[1] < grid.shape[0] - 1
    return rows, divider_color


def _extract_templates(top: np.ndarray, bg: int, divider_color: int) -> tuple[dict[int, np.ndarray], int]:
    h, w = top.shape
    separator_cols = [c for c in range(w) if np.all(top[:, c] == divider_color)]
    intervals: list[tuple[int, int]] = []
    start = 0
    for c in separator_cols + [w]:
        if c > start:
            patch = top[:, start:c]
            colors = set(map(int, np.unique(patch))) - {bg}
            if colors:
                intervals.append((start, c))
        start = c + 1

    assert len(intervals) >= 2, "expected multiple template slots"
    size = intervals[0][1] - intervals[0][0]
    assert size == h, "template slots must be square"

    templates: dict[int, np.ndarray] = {}
    for c0, c1 in intervals:
        assert c1 - c0 == size, "template slots must share one size"
        patch = top[:, c0:c1]
        colors = set(map(int, np.unique(patch))) - {bg}
        assert len(colors) == 1, "each template must use a single foreground color"
        color = colors.pop()
        assert color != divider_color
        mask = patch == color
        assert mask.any()
        templates[color] = mask

    return templates, size


def _stamp_row(anchor_row: int, middle_h: int, bottom_h: int, size: int) -> int:
    if middle_h <= 1 or bottom_h <= size:
        return 0
    span = bottom_h - size
    return int(round(anchor_row * span / (middle_h - 1)))


def _render_output(
    templates: dict[int, np.ndarray],
    anchors: list[tuple[int, int, int]],
    width: int,
    middle_h: int,
    bottom_h: int,
    bg: int,
    size: int,
) -> np.ndarray:
    out = np.full((bottom_h, width), bg, dtype=int)
    midpoint = _horizontal_midpoint(width)
    half = size // 2

    for anchor_row, anchor_col, color in sorted(anchors):
        mask = templates[color]
        if anchor_col < midpoint:
            mask = np.fliplr(mask)
        top = _stamp_row(anchor_row, middle_h, bottom_h, size)
        left = anchor_col - half
        assert 0 <= top <= bottom_h - size
        assert 0 <= left <= width - size
        window = out[top : top + size, left : left + size]
        window[mask] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each middle-band anchor to the same-colored top template and stamp it
    into the bottom band. Left-half anchors flip the template horizontally;
    right-half anchors keep it unchanged. Anchor height maps linearly to the
    template's top row inside the output canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    divider_rows, divider_color = _find_divider_rows(g, bg)
    top_end, mid_end = divider_rows

    top = g[:top_end, :]
    middle = g[top_end + 1 : mid_end, :]
    bottom = g[mid_end + 1 :, :]
    assert top.size and middle.size and bottom.size

    templates, size = _extract_templates(top, bg, divider_color)
    anchors = [
        (int(r), int(c), int(middle[r, c]))
        for r, c in np.argwhere(middle != bg)
    ]
    assert anchors, "expected at least one colored anchor"
    for _, _, color in anchors:
        assert color in templates, "anchor color must match a top template"

    return _render_output(templates, anchors, g.shape[1], middle.shape[0], bottom.shape[0], bg, size)


def _neighbors4(r: int, c: int, n: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < n:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < n:
        out.append((r, c + 1))
    return out


def _make_mask(size: int) -> np.ndarray:
    min_area = max(size + 1, (size * size) // 3)
    max_area = min(size * size - 1, (2 * size * size) // 3)

    for _ in range(MASK_RETRIES):
        target = random.randint(min_area, max_area)
        seed = (random.randrange(size), random.randrange(size))
        cells = {seed}
        frontier = set(_neighbors4(seed[0], seed[1], size))

        while frontier and len(cells) < target:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nb in _neighbors4(cell[0], cell[1], size):
                if nb not in cells:
                    frontier.add(nb)

        mask = np.zeros((size, size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if mask.sum() < min_area:
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if size >= 3 and not (mask[:, : size // 2].any() and mask[:, size // 2 + 1 :].any()):
            continue
        return mask

    raise RuntimeError("failed to build an asymmetric template")


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])


def _choose_anchor_layout(
    width: int,
    middle_h: int,
    bottom_h: int,
    size: int,
    template_colors: list[int],
) -> list[tuple[int, int, int]] | None:
    half = size // 2
    midpoint = _horizontal_midpoint(width)
    valid_cols = list(range(half, width - half))
    left_cols = [c for c in valid_cols if c < midpoint]
    right_cols = [c for c in valid_cols if c > midpoint]
    if not left_cols or not right_cols:
        return None

    anchor_count = random.randint(ANCHOR_COUNT_RANGE[0], min(ANCHOR_COUNT_RANGE[1], len(template_colors), middle_h))
    extra_left = random.choice((False, True))
    left_count = anchor_count // 2 + int(anchor_count % 2 == 1 and extra_left)
    right_count = anchor_count - left_count
    if left_count == 0 or right_count == 0:
        return None

    for _ in range(ANCHOR_LAYOUT_RETRIES):
        rows = random.sample(range(middle_h), anchor_count)
        cols = random.sample(left_cols, left_count) + random.sample(right_cols, right_count)
        colors = random.sample(template_colors, anchor_count)
        random.shuffle(cols)
        anchors = [(rows[i], cols[i], colors[i]) for i in range(anchor_count)]

        boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for anchor_row, anchor_col, _ in anchors:
            top = _stamp_row(anchor_row, middle_h, bottom_h, size)
            left = anchor_col - half
            box = (top, top + size, left, left + size)
            if any(_boxes_overlap(box, other) for other in boxes):
                ok = False
                break
            boxes.append(box)

        if ok:
            return anchors

    return None


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(TEMPLATE_SIZES)
    template_count = random.randint(*TEMPLATE_COUNT_RANGE)
    colors = random.sample(PALETTE, template_count)
    width = template_count * size + (template_count - 1)
    middle_h = random.randint(*MIDDLE_HEIGHT_RANGE)
    bottom_h = random.randint(size + middle_h - 1, size + middle_h + 4)

    templates = {color: _make_mask(size) for color in colors}
    anchors = _choose_anchor_layout(width, middle_h, bottom_h, size, colors)
    if anchors is None:
        return None

    height = size + 1 + middle_h + 1 + bottom_h
    grid = np.full((height, width), BG, dtype=int)

    for idx, color in enumerate(colors):
        c0 = idx * (size + 1)
        patch = np.where(templates[color], color, BG)
        grid[:size, c0 : c0 + size] = patch
        if idx + 1 < template_count:
            grid[:size, c0 + size] = DIVIDER

    grid[size, :] = DIVIDER
    grid[size + 1 + middle_h, :] = DIVIDER

    middle_r0 = size + 1
    for anchor_row, anchor_col, color in anchors:
        grid[middle_r0 + anchor_row, anchor_col] = color

    expected = _render_output(templates, anchors, width, middle_h, bottom_h, BG, size)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
