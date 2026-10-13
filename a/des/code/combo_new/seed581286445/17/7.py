from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
ARC_MAX = 30
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400

TEMPLATE_LIBRARY = [
    np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 1],
        ],
        dtype=bool,
    ),
]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    rows = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != bg and bool(np.all(grid[r] == int(grid[r, 0])))
    ]
    assert len(rows) == 1, "expected exactly one solid separator row"
    sep = int(rows[0])
    assert 0 < sep < grid.shape[0] - 1, "separator must split top and bottom regions"
    return sep, int(grid[sep, 0])


def _read_instruction_rows(top: np.ndarray, bg: int, neutral: int) -> list[list[int]]:
    row_ids = [r for r in range(top.shape[0]) if bool(np.any(top[r] != bg))]
    assert row_ids, "expected at least one instruction row"

    instructions: list[list[int]] = []
    prev = None
    for r in row_ids:
        if prev is not None:
            assert r - prev >= 2, "instruction rows must be separated by blank rows"
        prev = r

        cols = np.flatnonzero(top[r] != bg)
        assert cols.size >= 1, "instruction row cannot be empty"
        assert bool(np.all(np.diff(cols) == 1)), "instruction pixels must form one horizontal run"
        colors = [int(top[r, c]) for c in cols]
        assert all(color != neutral for color in colors), "instructions cannot use the template color"
        instructions.append(colors)

    return instructions


def _extract_template(lower: np.ndarray, bg: int, neutral: int) -> np.ndarray:
    assert bool(np.any(lower != bg)), "expected a template below the separator"
    assert bool(np.all((lower == bg) | (lower == neutral))), "lower area must contain only template pixels"

    coords = np.argwhere(lower == neutral)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = lower[r0 : r1 + 1, c0 : c1 + 1] == neutral
    assert bool(mask.any()), "template bbox cannot be empty"
    return mask


def _render_output(template: np.ndarray, instructions: list[list[int]], bg: int) -> np.ndarray:
    height, width = map(int, template.shape)
    offsets = [len(row) - 1 for row in instructions]
    out = np.full((height * len(instructions), width + max(offsets)), bg, dtype=int)

    for layer, row in enumerate(instructions):
        color = int(row[0])
        shift = int(len(row) - 1)
        r0 = out.shape[0] - (layer + 1) * height
        block = out[r0 : r0 + height, shift : shift + width]
        block[template] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Build a vertical stack of recolored template copies from row-wise instruction runs.

    Input structure:
    - The background is black, and one solid grey row splits the grid into an upper
      instruction area and a lower template area.
    - The lower area contains a single small grey template shape surrounded by black.
    - The upper area contains several horizontal instruction rows; each row is one
      contiguous run of 1 to 4 colored pixels, and blank rows separate consecutive instructions.

    Rule:
    - Read the instruction rows from top to bottom.
    - The first color of instruction row k determines the color of output layer k,
      counted from the bottom upward.
    - The number of pixels in that instruction row determines the layer's right shift:
      a row of length M shifts the template by M-1 columns.
    - Copy the grey template once per instruction row, stack the copies bottom-to-top
      without blank rows between their bounding boxes, recolor each copy, and fill the
      remaining cells with black.

    Key insight:
    - Each instruction row encodes two independent attributes at once: its leftmost
      color selects the layer color, while the run length selects the horizontal offset.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    sep_row, neutral = _find_separator(g, bg)
    instructions = _read_instruction_rows(g[:sep_row], bg, neutral)
    template = _extract_template(g[sep_row + 1 :], bg, neutral)
    return _render_output(template, instructions, bg)


def _sample_instruction_rows(count: int) -> list[list[int]]:
    mode = random.choice(["mono", "mixed"])
    for _ in range(100):
        lengths = [1] + [random.randint(1, 4) for _ in range(count - 1)]
        if len(set(lengths)) < 2:
            continue

        first_colors = [random.choice(PALETTE) for _ in range(count)]
        if len(set(first_colors)) < 2:
            continue

        rows = []
        for first, length in zip(first_colors, lengths, strict=False):
            row = [int(first)]
            for _ in range(length - 1):
                row.append(int(first if mode == "mono" else random.choice(PALETTE)))
            rows.append(row)

        if mode == "mixed":
            mixed_rows = [
                idx
                for idx, row in enumerate(rows)
                if len(row) > 1 and any(color != row[0] for color in row[1:])
            ]
            if not mixed_rows:
                candidates = [idx for idx, row in enumerate(rows) if len(row) > 1]
                if not candidates:
                    continue
                idx = random.choice(candidates)
                choices = [color for color in PALETTE if color != rows[idx][0]]
                rows[idx][1] = random.choice(choices)

        return rows
    raise RuntimeError("failed to sample non-degenerate instruction rows")


def _build_input(template: np.ndarray, instructions: list[list[int]]) -> np.ndarray | None:
    template_h, template_w = map(int, template.shape)
    top_lead = random.randint(0, 2)
    bottom_gap = random.randint(1, 2)
    gaps = [random.randint(1, 2) for _ in range(len(instructions) - 1)]
    lower_pad = random.randint(1, 3)
    lower_top = lower_pad + random.randint(0, 1)
    lower_bottom = lower_pad + random.randint(0, 1)

    top_height = top_lead + len(instructions) + sum(gaps) + bottom_gap
    lower_height = lower_top + template_h + lower_bottom

    min_width = max(8, template_w + 4, max(len(row) for row in instructions) + 4)
    max_width = min(18, min_width + 6)
    width = random.randint(min_width, max_width)
    height = top_height + 1 + lower_height
    if height > ARC_MAX or width > ARC_MAX:
        return None

    grid = np.full((height, width), BG, dtype=int)
    sep_row = top_height
    grid[sep_row, :] = GREY

    r = top_lead
    for idx, row in enumerate(instructions):
        start = random.randint(1, width - len(row) - 1)
        grid[r, start : start + len(row)] = np.array(row, dtype=int)
        if idx < len(gaps):
            r += 1 + gaps[idx]

    template_r0 = sep_row + 1 + lower_top
    template_c0 = (width - template_w) // 2
    block = grid[template_r0 : template_r0 + template_h, template_c0 : template_c0 + template_w]
    block[template] = GREY

    return grid


def _oracle_input() -> np.ndarray | None:
    template = random.choice(TEMPLATE_LIBRARY)
    max_layers = min(5, ARC_MAX // template.shape[0])
    layer_count = random.randint(2, max_layers)
    instructions = _sample_instruction_rows(layer_count)
    inp = _build_input(template, instructions)
    if inp is None:
        return None

    expected = _render_output(template, instructions, BG)
    if max(expected.shape) > ARC_MAX:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _oracle_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stacked-template input")
