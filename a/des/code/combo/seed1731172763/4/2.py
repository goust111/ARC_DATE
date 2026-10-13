from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GAP = 1
SLOT_COLS = 2
CELL_SIZES = (2, 3)
SLOT_ROWS_CHOICES = (2, 3)
MAX_GENERATION_ATTEMPTS = 300

MOTIFS = {
    2: (
        ((1, 1), (1, 1)),
        ((1, 0), (1, 0)),
        ((0, 1), (0, 1)),
        ((1, 1), (0, 0)),
        ((0, 0), (1, 1)),
        ((1, 1), (1, 0)),
        ((1, 1), (0, 1)),
        ((1, 0), (1, 1)),
        ((0, 1), (1, 1)),
        ((1, 0), (0, 1)),
        ((0, 1), (1, 0)),
    ),
    3: (
        ((1, 1, 1), (1, 1, 1), (1, 1, 1)),
        ((0, 1, 0), (1, 1, 1), (0, 1, 0)),
        ((1, 0, 0), (1, 0, 0), (1, 1, 1)),
        ((0, 0, 1), (0, 0, 1), (1, 1, 1)),
        ((1, 1, 1), (1, 0, 0), (1, 0, 0)),
        ((1, 1, 1), (0, 0, 1), (0, 0, 1)),
        ((1, 1, 1), (0, 1, 0), (0, 1, 0)),
        ((0, 1, 0), (0, 1, 0), (1, 1, 1)),
        ((1, 0, 0), (1, 1, 1), (1, 0, 0)),
        ((0, 0, 1), (1, 1, 1), (0, 0, 1)),
        ((1, 1, 1), (1, 0, 1), (1, 1, 1)),
    ),
}


def _find_panel_bounds(g: np.ndarray) -> tuple[int, int]:
    h, w = g.shape
    candidates = [
        c
        for c in range(w)
        if int(g[0, c]) != BG and bool(np.all(g[:, c] == g[0, c]))
    ]

    pairs: list[tuple[int, int]] = []
    for i, c1 in enumerate(candidates):
        for c2 in candidates[i + 1 :]:
            if int(g[0, c1]) != int(g[0, c2]):
                continue
            left_w = c1
            mid_w = c2 - c1 - 1
            right_w = w - c2 - 1
            if left_w > 0 and left_w == mid_w == right_w:
                pairs.append((int(c1), int(c2)))

    assert len(pairs) == 1, "expected exactly two equal-spacing separator columns"
    c1, c2 = pairs[0]
    assert int(g[0, c1]) == SEP and int(g[0, c2]) == SEP, "expected grey separators"
    assert h > 0
    return c1, c2


def _recolor(panel: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    # Apply all replacements against the original panel so chained mappings stay simultaneous.
    src = np.asarray(panel, dtype=int)
    out = src.copy()
    for old, new in mapping.items():
        out[src == int(old)] = int(new)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the left and middle panels to infer color substitutions, then apply
    the same substitutions to the right panel and return only that panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    c1, c2 = _find_panel_bounds(g)
    left = g[:, :c1]
    middle = g[:, c1 + 1 : c2]
    right = g[:, c2 + 1 :]
    assert left.shape == middle.shape == right.shape

    mapping: dict[int, int] = {}
    for r, c in np.argwhere(left != middle):
        old = int(left[r, c])
        new = int(middle[r, c])
        if old in mapping:
            assert mapping[old] == new, "inconsistent color mapping"
        else:
            mapping[old] = new

    return _recolor(right, mapping)


def _panel_shape(slot_rows: int, cell_size: int) -> tuple[int, int]:
    return (
        slot_rows * cell_size + (slot_rows + 1) * GAP,
        SLOT_COLS * cell_size + (SLOT_COLS + 1) * GAP,
    )


def _slot_origin(slot_index: int, cell_size: int) -> tuple[int, int]:
    slot_r, slot_c = divmod(slot_index, SLOT_COLS)
    return (
        GAP + slot_r * (cell_size + GAP),
        GAP + slot_c * (cell_size + GAP),
    )


def _paint_slots(
    slot_rows: int,
    cell_size: int,
    active_slots: list[int],
    colors: list[int],
) -> np.ndarray:
    h, w = _panel_shape(slot_rows, cell_size)
    panel = np.full((h, w), BG, dtype=int)
    motifs = MOTIFS[cell_size]

    for slot_index, color in zip(active_slots, colors, strict=True):
        r0, c0 = _slot_origin(slot_index, cell_size)
        motif = np.array(random.choice(motifs), dtype=bool)
        panel[r0 : r0 + cell_size, c0 : c0 + cell_size][motif] = int(color)

    return panel


def _expand_colors(base_colors: list[int], count: int) -> list[int]:
    colors = list(base_colors)
    while len(colors) < count:
        colors.append(random.choice(base_colors))
    random.shuffle(colors)
    return colors


def _gen_left_panel(slot_rows: int, cell_size: int) -> tuple[np.ndarray, list[int]]:
    total_slots = slot_rows * SLOT_COLS
    active_count = random.randint(3, total_slots)
    unique_count = random.randint(2, min(5, active_count, len(PALETTE) - 1))
    left_colors = random.sample(PALETTE, unique_count)
    active_slots = random.sample(range(total_slots), active_count)
    colors = _expand_colors(left_colors, active_count)
    return _paint_slots(slot_rows, cell_size, active_slots, colors), left_colors


def _gen_right_panel(
    slot_rows: int,
    cell_size: int,
    source_colors: list[int],
) -> np.ndarray:
    total_slots = slot_rows * SLOT_COLS
    min_active = max(3, len(source_colors) + 1)
    active_count = random.randint(min_active, total_slots)
    active_slots = random.sample(range(total_slots), active_count)

    extra_pool = [color for color in PALETTE if color not in source_colors]
    max_extra = min(3, len(extra_pool), active_count - len(source_colors))
    extra_count = random.randint(1, max_extra)
    right_palette = list(source_colors) + random.sample(extra_pool, extra_count)
    colors = _expand_colors(list(source_colors), active_count)
    for i in range(len(source_colors), active_count):
        colors[i] = random.choice(right_palette)
    random.shuffle(colors)
    return _paint_slots(slot_rows, cell_size, active_slots, colors)


def _gen_attempt() -> np.ndarray | None:
    cell_size = random.choice(CELL_SIZES)
    slot_rows = random.choice(SLOT_ROWS_CHOICES)

    left, left_colors = _gen_left_panel(slot_rows, cell_size)
    map_count = random.randint(1, min(3, len(left_colors)))
    source_colors = random.sample(left_colors, map_count)
    target_pool = [color for color in PALETTE if color not in left_colors]
    target_colors = random.sample(target_pool, map_count)
    mapping = dict(zip(source_colors, target_colors, strict=True))

    middle = _recolor(left, mapping)
    right = _gen_right_panel(slot_rows, cell_size, source_colors)
    expected = _recolor(right, mapping)

    h, w = left.shape
    grid = np.full((h, 3 * w + 2), BG, dtype=int)
    grid[:, :w] = left
    grid[:, w] = SEP
    grid[:, w + 1 : 2 * w + 1] = middle
    grid[:, 2 * w + 1] = SEP
    grid[:, 2 * w + 2 :] = right

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
