import random
from collections import Counter
from itertools import product

import numpy as np


SHAPE_LIBRARY = tuple(
    (
        max(r for r, _ in coords) + 1,
        max(c for _, c in coords) + 1,
        coords,
    )
    for coords in (
        ((0, 1), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (1, 1), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 1), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
    )
)

PERIOD_PAIRS = ((2, 3), (3, 2), (3, 4), (4, 3))
PALETTE = tuple(range(10))
GEN_RETRIES = 400


def _mode_color(values: np.ndarray) -> int:
    flat = [int(v) for v in np.asarray(values).ravel()]
    return Counter(flat).most_common(1)[0][0]


def _contiguous_spans(flags: np.ndarray) -> list[tuple[int, int]]:
    spans = []
    start = None
    for idx, flag in enumerate(flags):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(flags) - 1))
    return spans


def _normalize_mask(mask: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "shape mask must be non-empty"
    r0, c0 = coords.min(axis=0)
    shifted = [(int(r - r0), int(c - c0)) for r, c in coords]
    height = int(coords[:, 0].max() - r0 + 1)
    width = int(coords[:, 1].max() - c0 + 1)
    return height, width, tuple(sorted(shifted))


def _has_smaller_period(pattern: tuple) -> bool:
    size = len(pattern)
    for step in range(1, size):
        if size % step != 0:
            continue
        if all(pattern[i] == pattern[i % step] for i in range(size)):
            return True
    return False


def _infer_full_sequence(observed: list) -> tuple:
    n_slots = len(observed)
    labels = tuple(dict.fromkeys(value for value in observed if value is not None))
    assert labels, "need at least one surviving slot"

    candidates = set()
    for period in range(1, n_slots):
        residue_values = [None] * period
        valid = True
        for index, value in enumerate(observed):
            if value is None:
                continue
            residue = index % period
            if residue_values[residue] is None:
                residue_values[residue] = value
            elif residue_values[residue] != value:
                valid = False
                break
        if not valid:
            continue

        missing_residues = [i for i, value in enumerate(residue_values) if value is None]
        assignments = product(labels, repeat=len(missing_residues)) if missing_residues else [()]
        for assigned in assignments:
            pattern = list(residue_values)
            for residue, value in zip(missing_residues, assigned):
                pattern[residue] = value
            pattern = tuple(pattern)
            if _has_smaller_period(pattern):
                continue
            full = tuple(pattern[i % period] for i in range(n_slots))
            if all(value is None or value == full[i] for i, value in enumerate(observed)):
                candidates.add(full)

    assert len(candidates) == 1, "survivors must determine a unique periodic completion"
    return next(iter(candidates))


def _extract_slots(grid: np.ndarray):
    colors = [int(v) for v in np.unique(grid)]
    separator = None
    row_spans = None
    col_spans = None
    best_score = -1
    for color in colors:
        row_flags = np.all(grid == color, axis=1)
        col_flags = np.all(grid == color, axis=0)
        rows = _contiguous_spans(~row_flags)
        cols = _contiguous_spans(~col_flags)
        if len(rows) != 1 or len(cols) < 2:
            continue
        score = int(row_flags.sum() + col_flags.sum())
        if score > best_score:
            best_score = score
            separator = color
            row_spans = rows
            col_spans = cols
    assert separator is not None, "failed to find the slot separator color"

    slot_rows = row_spans[0]
    slot_cols = col_spans
    non_separator = grid[grid != separator]
    background = _mode_color(non_separator)
    return int(separator), int(background), slot_rows, slot_cols


def _render_mask_in_cell(cell_shape: tuple[int, int], shape_key) -> np.ndarray:
    height, width, coords = shape_key
    cell_h, cell_w = cell_shape
    assert height <= cell_h and width <= cell_w
    top = (cell_h - height) // 2
    left = (cell_w - width) // 2
    mask = np.zeros(cell_shape, dtype=bool)
    for r, c in coords:
        mask[top + r, left + c] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the emptied symbol slots by solving separate shape and color cycles.

    Input structure:
    - The grid contains a single horizontal row of symbol slots separated by one-pixel separator lines.
    - Each occupied slot contains exactly one monochrome symbol on a shared slot background.
    - Across the full row, symbol shapes repeat with one period and symbol colors repeat with a different period.

    Rule:
    - Parse the slots while keeping the separator lines and surviving occupied slots unchanged.
    - Infer the unique periodic shape sequence from the surviving slot masks while ignoring color.
    - Infer the unique periodic color sequence from the surviving slot colors while ignoring shape, then fill each empty slot with the inferred shape and color at that position.

    Key insight:
    - The visible pair pattern is produced by two decoupled cycles, so the emptied slots can only be restored by solving shape periodicity and color periodicity independently.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator, background, (r0, r1), col_spans = _extract_slots(g)
    cell_shape = (r1 - r0 + 1, col_spans[0][1] - col_spans[0][0] + 1)
    assert all((c1 - c0 + 1, r1 - r0 + 1) == (cell_shape[1], cell_shape[0]) for c0, c1 in col_spans)

    observed_shapes = []
    observed_colors = []
    shape_templates: dict[tuple[int, int, tuple[tuple[int, int], ...]], np.ndarray] = {}

    for c0, c1 in col_spans:
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        content = patch[(patch != background) & (patch != separator)]
        if content.size == 0:
            observed_shapes.append(None)
            observed_colors.append(None)
            continue

        colors = np.unique(content)
        assert len(colors) == 1, "each slot must contain a single monochrome symbol"
        color = int(colors[0])
        mask = patch == color
        shape_key = _normalize_mask(mask)
        observed_shapes.append(shape_key)
        observed_colors.append(color)
        shape_templates.setdefault(shape_key, mask.copy())

    full_shapes = _infer_full_sequence(observed_shapes)
    full_colors = _infer_full_sequence(observed_colors)

    out = g.copy()
    for index, (c0, c1) in enumerate(col_spans):
        if observed_shapes[index] is not None:
            continue
        shape_key = full_shapes[index]
        color = full_colors[index]
        mask = shape_templates.get(shape_key)
        if mask is None:
            mask = _render_mask_in_cell(cell_shape, shape_key)
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[:] = background
        patch[mask] = color

    return out


def _build_full_sequences(n_slots: int, shape_period: int, color_period: int):
    shape_cycle = random.sample(SHAPE_LIBRARY, shape_period)
    color_choices = random.sample(PALETTE, color_period + 2)
    color_cycle = tuple(color_choices[:color_period])
    background = color_choices[color_period]
    separator = color_choices[color_period + 1]
    full_shapes = tuple(shape_cycle[i % shape_period] for i in range(n_slots))
    full_colors = tuple(color_cycle[i % color_period] for i in range(n_slots))
    return full_shapes, full_colors, background, separator


def _valid_missing_sets(full_shapes: tuple, full_colors: tuple) -> list[tuple[int, ...]]:
    n_slots = len(full_shapes)
    valid = []
    for bits in range(1, 1 << n_slots):
        missing = tuple(i for i in range(n_slots) if bits & (1 << i))
        if len(missing) > 2:
            continue
        observed_shapes = [None if i in missing else full_shapes[i] for i in range(n_slots)]
        observed_colors = [None if i in missing else full_colors[i] for i in range(n_slots)]
        try:
            inferred_shapes = _infer_full_sequence(observed_shapes)
            inferred_colors = _infer_full_sequence(observed_colors)
        except AssertionError:
            continue
        if inferred_shapes == full_shapes and inferred_colors == full_colors:
            valid.append(missing)
    return valid


def _make_grid(
    full_shapes: tuple,
    full_colors: tuple,
    background: int,
    separator: int,
    cell_h: int,
    cell_w: int,
    use_outer_border: bool,
    missing: tuple[int, ...],
) -> np.ndarray:
    n_slots = len(full_shapes)
    width = n_slots * cell_w + (n_slots - 1)
    if use_outer_border:
        width += 2
    height = cell_h + 2
    grid = np.full((height, width), separator, dtype=int)

    slot_spans = []
    col = 1 if use_outer_border else 0
    for slot_index in range(n_slots):
        slot_spans.append((col, col + cell_w - 1))
        col += cell_w
        if slot_index != n_slots - 1:
            col += 1

    grid[1:-1, :] = background
    if use_outer_border:
        for sep_col in [0, width - 1]:
            grid[:, sep_col] = separator
    for left, right in slot_spans[:-1]:
        sep_col = right + 1
        grid[:, sep_col] = separator

    for slot_index, (c0, c1) in enumerate(slot_spans):
        if slot_index in missing:
            continue
        mask = _render_mask_in_cell((cell_h, cell_w), full_shapes[slot_index])
        patch = grid[1:-1, c0 : c1 + 1]
        patch[mask] = full_colors[slot_index]

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        n_slots = 6
        shape_period, color_period = random.choice(PERIOD_PAIRS)
        full_shapes, full_colors, background, separator = _build_full_sequences(
            n_slots, shape_period, color_period
        )
        valid_missing = _valid_missing_sets(full_shapes, full_colors)
        if not valid_missing:
            continue

        preferred = [missing for missing in valid_missing if len(missing) == 2]
        missing = random.choice(preferred or valid_missing)
        cell_h = random.choice((3, 4))
        cell_w = random.choice((3, 4))
        use_outer_border = False if cell_w == 4 else random.choice((False, True))

        grid = _make_grid(
            full_shapes,
            full_colors,
            background,
            separator,
            cell_h,
            cell_w,
            use_outer_border,
            missing,
        )
        try:
            out = main(grid)
        except AssertionError:
            continue
        expected = _make_grid(
            full_shapes,
            full_colors,
            background,
            separator,
            cell_h,
            cell_w,
            use_outer_border,
            (),
        )
        if np.array_equal(out, expected):
            return grid
    raise RuntimeError("failed to generate a valid dual-cycle slot puzzle")
