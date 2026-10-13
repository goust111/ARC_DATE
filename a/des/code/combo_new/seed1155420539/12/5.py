from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

GEN_RETRIES = 400
PATTERN_RETRIES = 200
TEMPLATE_COUNT_RANGE = (2, 6)
INSTRUCTION_COUNT_RANGE = (1, 5)
TEMPLATE_WIDTH_MIN = 4
TEMPLATE_WIDTH_CAP = 10
MARGIN_MAX = 2
ROW_GAP_MAX = 2
PART_GAP_RANGE = (1, 3)
COLUMN_GAP_RANGE = (1, 3)


def _non_bg_colors(grid: np.ndarray) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) != BG]


def _pattern_signature(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in mask.astype(int).tolist())


def _try_parse_with_template_color(
    grid: np.ndarray, template_color: int
) -> dict[str, object] | None:
    h, w = grid.shape
    template_mask = grid == template_color
    template_rows = [r for r in range(h) if bool(template_mask[r].any())]
    if not template_rows:
        return None

    other_mask = (grid != BG) & (grid != template_color)
    if not bool(other_mask.any()):
        return None

    row_set = set(template_rows)
    for r, _ in np.argwhere(other_mask):
        if int(r) not in row_set:
            return None

    template_cols = [c for c in range(w) if bool(template_mask[:, c].any())]
    instruction_cols = [c for c in range(w) if bool(other_mask[:, c].any())]
    if not template_cols or not instruction_cols:
        return None
    if min(instruction_cols) <= max(template_cols):
        return None
    if any(instruction_cols[i + 1] - instruction_cols[i] < 2 for i in range(len(instruction_cols) - 1)):
        return None

    left = min(template_cols)
    right = max(template_cols) + 1

    templates: list[np.ndarray] = []
    seen_patterns: set[tuple[int, ...]] = set()
    for r in template_rows:
        window = grid[r, left:right]
        if np.any((window != BG) & (window != template_color)):
            return None
        pattern = window == template_color
        if not bool(pattern.any()):
            return None
        sig = _pattern_signature(pattern)
        if sig in seen_patterns:
            return None
        seen_patterns.add(sig)
        templates.append(pattern.copy())

    row_to_idx = {row: idx for idx, row in enumerate(template_rows)}
    used_instruction_colors: set[int] = set()
    instructions: list[tuple[int, list[int]]] = []

    for c in instruction_cols:
        vals = [int(v) for v in np.unique(grid[:, c]) if int(v) != BG]
        if len(vals) != 1:
            return None
        marker_color = vals[0]
        if marker_color == template_color or marker_color in used_instruction_colors:
            return None
        used_instruction_colors.add(marker_color)

        selected: list[int] = []
        for r in range(h):
            value = int(grid[r, c])
            if value == BG:
                continue
            if r not in row_set or value != marker_color:
                return None
            selected.append(row_to_idx[r])
        if not selected:
            return None
        instructions.append((marker_color, selected))

    return {
        "template_rows": templates,
        "instructions": instructions,
        "template_pixels": int(template_mask.sum()),
        "template_width": right - left,
    }


def _parse_puzzle(grid: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, list[int]]]]:
    candidates: list[dict[str, object]] = []
    for color in _non_bg_colors(grid):
        spec = _try_parse_with_template_color(grid, color)
        if spec is not None:
            candidates.append(spec)

    assert candidates, "expected one template color plus at least one instruction column"
    candidates.sort(
        key=lambda spec: (int(spec["template_pixels"]), int(spec["template_width"])),
        reverse=True,
    )
    best = candidates[0]
    return best["template_rows"], best["instructions"]


def _assemble_output(
    template_rows: list[np.ndarray], instructions: list[tuple[int, list[int]]]
) -> np.ndarray:
    assert template_rows, "need at least one template row"
    assert instructions, "need at least one instruction column"

    width = int(template_rows[0].shape[0])
    for pattern in template_rows:
        assert pattern.ndim == 1 and int(pattern.shape[0]) == width

    blocks: list[np.ndarray] = []
    for marker_color, selected in instructions:
        assert selected, "every instruction column must select at least one template row"
        block = np.full((len(selected), width), BG, dtype=int)
        for out_r, template_idx in enumerate(selected):
            block[out_r, template_rows[template_idx]] = marker_color
        blocks.append(block)

    out_h = max(block.shape[0] for block in blocks)
    out_w = len(blocks) * width + len(blocks) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for block in blocks:
        block_h, block_w = block.shape
        out[:block_h, col : col + block_w] = block
        col += block_w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild colorized row stacks by reading template rows and their selecting instruction columns.

    Input structure:
    - The background is black.
    - On the left side, several single-color template rows encode different horizontal pixel patterns, with optional blank rows between them.
    - On the right side, one or more instruction columns contain colored marker pixels aligned with some of the template rows.
    - Each instruction column uses one marker color, and neighboring instruction columns are separated by at least one black column.

    Rule:
    - Identify the common template color and record every non-empty template row pattern from top to bottom.
    - For each instruction column, read which template-row positions contain marker pixels and keep that top-to-bottom order.
    - Recolor the selected template patterns with the instruction column's marker color and stack the chosen rows tightly with no blank rows.
    - Concatenate the resulting blocks from left to right with one black separator column, padding shorter blocks with black rows at the bottom.

    Key insight:
    The left side is a vocabulary of row-shaped templates, while each right-side marker column simultaneously chooses a subset, fixes its row order, and supplies the output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    template_rows, instructions = _parse_puzzle(g)
    return _assemble_output(template_rows, instructions)


def _random_pattern(width: int, rng: random.Random) -> np.ndarray:
    while True:
        mask = np.zeros(width, dtype=bool)
        kind = rng.choice(["segment", "two_segments", "random", "alternating"])

        if kind == "segment":
            length = rng.randint(2, width)
            start = rng.randint(0, width - length)
            mask[start : start + length] = True
        elif kind == "two_segments" and width >= 5:
            gap_col = rng.randint(1, width - 2)
            left_space = gap_col
            right_space = width - gap_col - 1
            left_len = rng.randint(1, left_space)
            right_len = rng.randint(1, right_space)
            left_start = rng.randint(0, left_space - left_len)
            right_start = rng.randint(gap_col + 1, width - right_len)
            mask[left_start : left_start + left_len] = True
            mask[right_start : right_start + right_len] = True
        elif kind == "alternating":
            start = rng.randint(0, 1)
            mask[start:width:2] = True
            if not bool(mask.any()):
                mask[rng.randrange(width)] = True
        else:
            count = rng.randint(2, width)
            cols = rng.sample(range(width), count)
            mask[cols] = True

        if bool(mask.any()):
            return mask


def _generate_patterns(count: int, width: int, rng: random.Random) -> list[np.ndarray] | None:
    patterns: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(count):
        for _ in range(PATTERN_RETRIES):
            pattern = _random_pattern(width, rng)
            sig = _pattern_signature(pattern)
            if sig in seen:
                continue
            seen.add(sig)
            patterns.append(pattern)
            break
        else:
            return None
    return patterns


def _sample_selections(template_count: int, instruction_count: int, rng: random.Random) -> list[list[int]]:
    selections: list[list[int]] = []
    for _ in range(instruction_count):
        size = rng.randint(1, template_count)
        picked = sorted(rng.sample(range(template_count), size))
        selections.append(picked)

    if instruction_count > 1 and len({tuple(sel) for sel in selections}) == 1:
        idx = rng.randrange(instruction_count)
        size = rng.randint(1, template_count)
        selections[idx] = sorted(rng.sample(range(template_count), size))
    return selections


def _make_row_positions(template_count: int, rng: random.Random) -> list[int]:
    top_margin = rng.randint(0, MARGIN_MAX)
    positions: list[int] = []
    row = top_margin
    for idx in range(template_count):
        positions.append(row)
        if idx + 1 < template_count:
            row += 1 + rng.randint(0, ROW_GAP_MAX)
    return positions


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    template_count = rng.randint(*TEMPLATE_COUNT_RANGE)
    instruction_count = rng.randint(*INSTRUCTION_COUNT_RANGE)

    max_width = min(
        TEMPLATE_WIDTH_CAP,
        (30 - (instruction_count - 1)) // instruction_count,
    )
    if max_width < TEMPLATE_WIDTH_MIN:
        return None
    template_width = rng.randint(TEMPLATE_WIDTH_MIN, max_width)

    template_color, *instruction_colors = rng.sample(PALETTE, instruction_count + 1)
    patterns = _generate_patterns(template_count, template_width, rng)
    if patterns is None:
        return None
    selections = _sample_selections(template_count, instruction_count, rng)

    row_positions = _make_row_positions(template_count, rng)
    bottom_margin = rng.randint(0, MARGIN_MAX)
    height = row_positions[-1] + 1 + bottom_margin
    if height > 30:
        return None

    left_margin = rng.randint(0, MARGIN_MAX)
    template_start = left_margin

    gap_to_instructions = rng.randint(*PART_GAP_RANGE)
    instruction_cols: list[int] = []
    col = template_start + template_width + gap_to_instructions
    for idx in range(instruction_count):
        instruction_cols.append(col)
        if idx + 1 < instruction_count:
            col += 1 + rng.randint(*COLUMN_GAP_RANGE)

    right_margin = rng.randint(0, MARGIN_MAX)
    width = instruction_cols[-1] + 1 + right_margin
    if width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    for row, pattern in zip(row_positions, patterns):
        grid[row, template_start : template_start + template_width][pattern] = template_color

    for marker_color, col, selected in zip(instruction_colors, instruction_cols, selections):
        for template_idx in selected:
            grid[row_positions[template_idx], col] = marker_color

    expected = _assemble_output(patterns, list(zip(instruction_colors, selections)))
    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if max(grid.shape) > 30 or max(actual.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
