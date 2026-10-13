from __future__ import annotations

from collections import Counter

import numpy as np


BACKGROUND = 0
CONTROL_COLORS = (1, 2, 3)
PATTERN_COLOR = 5
GEN_RETRIES = 400


def _constant_step(values: list[int]) -> int:
    assert len(values) >= 2, "expected at least two lattice positions"
    diffs = [b - a for a, b in zip(values, values[1:])]
    assert len(set(diffs)) == 1, "lattice spacing must be uniform"
    step = diffs[0]
    assert step >= 4, "slots must be at least 3x3 with a 1-cell gap"
    return step


def _control_positions(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(np.isin(grid, CONTROL_COLORS))
    assert coords.size > 0, "expected control pixels"
    return coords


def _infer_slot_array(grid: np.ndarray) -> tuple[list[int], list[int], int, int]:
    coords = _control_positions(grid)
    row_starts = sorted({int(r) for r, _ in coords})
    col_starts = sorted({int(c) for _, c in coords})
    assert len(row_starts) >= 2 and len(col_starts) >= 2, "slot array must have multiple rows and columns"
    assert len(coords) == len(row_starts) * len(col_starts), "every slot must have exactly one control pixel"

    slot_h = _constant_step(row_starts) - 1
    slot_w = _constant_step(col_starts) - 1
    assert slot_h in (3, 4) and slot_w in (3, 4), "slot size must be 3x3 or 4x4"

    arr_r0 = row_starts[0]
    arr_r1 = row_starts[-1] + slot_h - 1
    arr_c0 = col_starts[0]
    arr_c1 = col_starts[-1] + slot_w - 1
    for r0 in row_starts[:-1]:
        gap_r = r0 + slot_h
        assert np.all(grid[gap_r, arr_c0 : arr_c1 + 1] == BACKGROUND), "row gaps must be background"
    for c0 in col_starts[:-1]:
        gap_c = c0 + slot_w
        assert np.all(grid[arr_r0 : arr_r1 + 1, gap_c] == BACKGROUND), "col gaps must be background"

    return row_starts, col_starts, slot_h, slot_w


def _slot_blocks(grid: np.ndarray) -> tuple[dict[tuple[int, int], np.ndarray], list[int], list[int], int, int]:
    row_starts, col_starts, slot_h, slot_w = _infer_slot_array(grid)
    blocks: dict[tuple[int, int], np.ndarray] = {}
    for r0 in row_starts:
        for c0 in col_starts:
            block = grid[r0 : r0 + slot_h, c0 : c0 + slot_w]
            assert block.shape == (slot_h, slot_w), "slot must fit in the grid"
            control = int(block[0, 0])
            assert control in CONTROL_COLORS, "slot top-left must be a control pixel"
            assert set(map(int, np.unique(block))) <= {BACKGROUND, PATTERN_COLOR, *CONTROL_COLORS}
            blocks[(r0, c0)] = block.copy()
    return blocks, row_starts, col_starts, slot_h, slot_w


def _majority_binary(values: list[int]) -> int:
    counts = Counter(values)
    zeros = counts.get(BACKGROUND, 0)
    greys = counts.get(PATTERN_COLOR, 0)
    assert zeros != greys, "majority must be unique"
    return PATTERN_COLOR if greys > zeros else BACKGROUND


def _recover_template(blocks: list[np.ndarray]) -> np.ndarray:
    slot_h, slot_w = blocks[0].shape
    template = np.zeros((slot_h, slot_w), dtype=int)
    for r in range(slot_h):
        for c in range(slot_w):
            if (r, c) == (0, 0):
                continue
            template[r, c] = _majority_binary([int(block[r, c]) for block in blocks])
    return template


def _recover_templates_by_color(grid: np.ndarray) -> tuple[dict[int, np.ndarray], dict[tuple[int, int], np.ndarray], list[int], list[int], int, int]:
    blocks, row_starts, col_starts, slot_h, slot_w = _slot_blocks(grid)
    by_color: dict[int, list[np.ndarray]] = {}
    for key, block in blocks.items():
        by_color.setdefault(int(block[0, 0]), []).append(block)

    assert len(by_color) >= 2, "expected multiple control-color groups"
    templates = {color: _recover_template(group_blocks) for color, group_blocks in by_color.items()}
    assert len({tuple(t.ravel()) for t in templates.values()}) == len(templates), "group templates must differ"
    return templates, blocks, row_starts, col_starts, slot_h, slot_w


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each noisy slot in a regular array by majority-voting within its control-color group.

    Input structure:
    - A black background contains a full rectangular lattice of 3x3 or 4x4 slots separated by 1-cell black gaps.
    - The top-left cell of every slot is a colored control pixel: blue, red, or green.
    - All other slot cells are black or grey, forming noisy copies of one of several hidden templates.

    Rule:
    - Detect the slot lattice from the regularly spaced control pixels.
    - Group slots by the color of their top-left control pixel.
    - For each color group and each relative slot position except the control cell, take the majority of black vs grey across that group's slots.
    - Replace every slot with its group's recovered template while keeping the original control pixel unchanged.

    Key insight:
    - There is not one global template: the control pixel color decides which repeated template a slot belongs to, so voting must happen after grouping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    templates, blocks, row_starts, col_starts, slot_h, slot_w = _recover_templates_by_color(g)
    out = np.full_like(g, BACKGROUND)
    for r0 in row_starts:
        for c0 in col_starts:
            block = blocks[(r0, c0)]
            control = int(block[0, 0])
            restored = templates[control].copy()
            restored[0, 0] = control
            out[r0 : r0 + slot_h, c0 : c0 + slot_w] = restored
    return out


def _sample_layout(rng: np.random.Generator, n_groups: int) -> tuple[int, int]:
    options = []
    for slot_rows in range(3, 5):
        for slot_cols in range(3, 5):
            if slot_rows * slot_cols >= 3 * n_groups:
                options.append((slot_rows, slot_cols))
    idx = int(rng.integers(len(options)))
    return options[idx]


def _sample_template(slot_h: int, slot_w: int, grey_count: int, rng: np.random.Generator) -> np.ndarray | None:
    payload = [(r, c) for r in range(slot_h) for c in range(slot_w) if (r, c) != (0, 0)]
    if grey_count > len(payload):
        return None
    for _ in range(100):
        chosen_idx = rng.choice(len(payload), size=grey_count, replace=False)
        chosen = {payload[int(i)] for i in np.atleast_1d(chosen_idx)}
        rows_hit = {r for r, _ in chosen}
        cols_hit = {c for _, c in chosen}
        if len(rows_hit) != slot_h or len(cols_hit) != slot_w:
            continue
        template = np.zeros((slot_h, slot_w), dtype=int)
        for r, c in chosen:
            template[r, c] = PATTERN_COLOR
        if np.array_equal(template[:, 1:], np.fliplr(template[:, 1:])):
            continue
        return template
    return None


def _sample_templates(slot_h: int, slot_w: int, colors: list[int], rng: np.random.Generator) -> dict[int, np.ndarray] | None:
    payload_size = slot_h * slot_w - 1
    lo = max(slot_h, slot_w, payload_size // 3)
    hi = min(payload_size - 1, (2 * payload_size) // 3)
    counts = list(range(lo, hi + 1))
    rng.shuffle(counts)
    for grey_count in counts:
        templates: dict[int, np.ndarray] = {}
        for color in colors:
            placed = False
            for _ in range(120):
                template = _sample_template(slot_h, slot_w, grey_count, rng)
                if template is None:
                    continue
                if any(np.sum(template != other) < 3 for other in templates.values()):
                    continue
                templates[color] = template
                placed = True
                break
            if not placed:
                break
        if len(templates) == len(colors):
            return templates
    return None


def _mixed_assignment(
    slot_rows: int,
    slot_cols: int,
    colors: list[int],
    counts: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    total = slot_rows * slot_cols
    flat = np.array(sum(([color] * count for color, count in zip(colors, counts)), []), dtype=int)
    assert flat.size == total
    for _ in range(200):
        rng.shuffle(flat)
        layout = flat.reshape(slot_rows, slot_cols)
        if not any(len(set(map(int, row))) >= 2 for row in layout):
            continue
        if not any(len(set(map(int, layout[:, c]))) >= 2 for c in range(slot_cols)):
            continue
        ok = True
        for color in colors:
            pos = np.argwhere(layout == color)
            if len({int(r) for r, _ in pos}) < 2 or len({int(c) for _, c in pos}) < 2:
                ok = False
                break
        if ok:
            return layout.copy()
    return None


def _compose_counts(total_slots: int, n_groups: int, rng: np.random.Generator) -> list[int]:
    base = np.full(n_groups, 3, dtype=int)
    extra = total_slots - 3 * n_groups
    if extra > 0:
        base += rng.multinomial(extra, np.full(n_groups, 1.0 / n_groups))
    return [int(x) for x in base]


def _apply_noise(block: np.ndarray, slot_h: int, slot_w: int, rng: np.random.Generator) -> np.ndarray:
    noisy = block.copy()
    payload = [(r, c) for r in range(slot_h) for c in range(slot_w) if (r, c) != (0, 0)]
    max_flips = max(1, payload.__len__() // 4)
    n_flips = int(rng.integers(0, max_flips + 1))
    if n_flips == 0:
        return noisy
    flip_idx = rng.choice(len(payload), size=n_flips, replace=False)
    for idx in np.atleast_1d(flip_idx):
        r, c = payload[int(idx)]
        noisy[r, c] = PATTERN_COLOR if noisy[r, c] == BACKGROUND else BACKGROUND
    return noisy


def _render_example(
    slot_rows: int,
    slot_cols: int,
    slot_h: int,
    slot_w: int,
    assignment: np.ndarray,
    templates: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[np.ndarray]], int]:
    arr_h = slot_rows * slot_h + (slot_rows - 1)
    arr_w = slot_cols * slot_w + (slot_cols - 1)
    top = int(rng.integers(0, 4))
    left = int(rng.integers(0, 4))
    bottom = int(rng.integers(0, 4))
    right = int(rng.integers(0, 4))
    grid = np.zeros((top + arr_h + bottom, left + arr_w + right), dtype=int)
    out = np.zeros_like(grid)
    noisy_by_color: dict[int, list[np.ndarray]] = {int(color): [] for color in templates}
    total_flips = 0
    for sr in range(slot_rows):
        for sc in range(slot_cols):
            color = int(assignment[sr, sc])
            r0 = top + sr * (slot_h + 1)
            c0 = left + sc * (slot_w + 1)
            clean = templates[color].copy()
            clean[0, 0] = color
            noisy = _apply_noise(clean, slot_h, slot_w, rng)
            total_flips += int(np.sum(noisy != clean))
            noisy_by_color[color].append(noisy.copy())
            grid[r0 : r0 + slot_h, c0 : c0 + slot_w] = noisy
            out[r0 : r0 + slot_h, c0 : c0 + slot_w] = clean
    return grid, out, noisy_by_color, total_flips


def _recover_without_groups(noisy_by_color: dict[int, list[np.ndarray]]) -> np.ndarray | None:
    all_blocks = [block for blocks in noisy_by_color.values() for block in blocks]
    slot_h, slot_w = all_blocks[0].shape
    template = np.zeros((slot_h, slot_w), dtype=int)
    for r in range(slot_h):
        for c in range(slot_w):
            if (r, c) == (0, 0):
                continue
            values = [int(block[r, c]) for block in all_blocks]
            counts = Counter(values)
            zeros = counts.get(BACKGROUND, 0)
            greys = counts.get(PATTERN_COLOR, 0)
            if zeros == greys:
                return None
            template[r, c] = PATTERN_COLOR if greys > zeros else BACKGROUND
    return template


def _verify_semantics(
    inp: np.ndarray,
    expected: np.ndarray,
    assignment: np.ndarray,
    templates: dict[int, np.ndarray],
    noisy_by_color: dict[int, list[np.ndarray]],
    total_flips: int,
) -> bool:
    if total_flips == 0 or np.array_equal(inp, expected):
        return False
    if len(templates) < 2:
        return False

    grey_counts = {color: int(np.sum(template == PATTERN_COLOR)) for color, template in templates.items()}
    if len(set(grey_counts.values())) != 1:
        return False
    if len({tuple(template.ravel()) for template in templates.values()}) != len(templates):
        return False

    for color in templates:
        slots = noisy_by_color[color]
        if len(slots) < 3:
            return False
        try:
            recovered = _recover_template(slots)
        except AssertionError:
            return False
        if not np.array_equal(recovered, templates[color]):
            return False

    if not any(len(set(map(int, row))) >= 2 for row in assignment):
        return False
    if not any(len(set(map(int, assignment[:, c]))) >= 2 for c in range(assignment.shape[1])):
        return False
    for color in templates:
        pos = np.argwhere(assignment == color)
        if len({int(r) for r, _ in pos}) < 2 or len({int(c) for _, c in pos}) < 2:
            return False

    global_template = _recover_without_groups(noisy_by_color)
    if global_template is not None and all(np.array_equal(global_template, t) for t in templates.values()):
        return False

    count_sets = {
        color: {int(np.sum(block[1:, :] == PATTERN_COLOR)) + int(np.sum(block[0, 1:] == PATTERN_COLOR)) for block in blocks}
        for color, blocks in noisy_by_color.items()
    }
    overlap = False
    color_list = list(count_sets)
    for i, color_a in enumerate(color_list):
        for color_b in color_list[i + 1 :]:
            if count_sets[color_a] & count_sets[color_b]:
                overlap = True
    if not overlap:
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    n_groups = int(rng.choice([2, 3], p=[0.35, 0.65]))
    slot_rows, slot_cols = _sample_layout(rng, n_groups)
    slot_h = int(rng.choice([3, 4]))
    slot_w = int(rng.choice([3, 4]))
    colors = list(map(int, rng.choice(CONTROL_COLORS, size=n_groups, replace=False)))
    templates = _sample_templates(slot_h, slot_w, colors, rng)
    if templates is None:
        return None

    counts = _compose_counts(slot_rows * slot_cols, n_groups, rng)
    assignment = _mixed_assignment(slot_rows, slot_cols, colors, counts, rng)
    if assignment is None:
        return None

    inp, expected, noisy_by_color, total_flips = _render_example(
        slot_rows, slot_cols, slot_h, slot_w, assignment, templates, rng
    )
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _verify_semantics(inp, expected, assignment, templates, noisy_by_color, total_flips):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
