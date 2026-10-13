from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, paste


BG = Color.BLACK
TEMPLATE_PALETTE = (Color.BLUE, Color.GREEN)
PREFIX_PALETTE = (Color.RED, Color.YELLOW, Color.PINK, Color.ORANGE)
SEPARATOR = Color.GREY
MARKER = Color.TEAL

PARTIAL_ROW_COUNT = 4
TEMPLATE_WIDTH_RANGE = (8, 12)
MACRO_ROW_RANGE = (3, 5)
GENERATION_RETRIES = 400

ORDERED_PREFIX_PAIRS = [(a, b) for a in PREFIX_PALETTE for b in PREFIX_PALETTE if a != b]
_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


def _signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(tuple(grid.shape)).encode()).hexdigest()


def _partition_spans(length: int, separators: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in [int(v) for v in separators.tolist()]:
        if start <= sep - 1:
            spans.append((start, sep - 1))
        start = sep + 1
    if start <= length - 1:
        spans.append((start, length - 1))
    return spans


def _is_alternating_template_row(row: np.ndarray) -> bool:
    if row.ndim != 1 or row.size < 2 or np.any(row == BG):
        return False
    colors = np.unique(row)
    if len(colors) != 2 or int(row[0]) == int(row[1]):
        return False
    return bool(np.all(row[::2] == row[0]) and np.all(row[1::2] == row[1]))


def _leading_prefix_len(row: np.ndarray) -> int | None:
    zeros = np.flatnonzero(row == BG)
    if len(zeros) == 0:
        return None
    prefix_len = int(zeros[0])
    if prefix_len == 0:
        return None
    if np.any(row[:prefix_len] == BG) or np.any(row[prefix_len:] != BG):
        return None
    return prefix_len


def _is_partial_prefix_row(row: np.ndarray) -> bool:
    if np.any(row == SEPARATOR) or int(row[0]) == BG:
        return False
    prefix_len = _leading_prefix_len(row)
    if prefix_len is None or not (2 <= prefix_len <= 4):
        return False
    return set(map(int, np.unique(row[:prefix_len]))) <= set(PREFIX_PALETTE)


def _decode_completed_row(template_row: np.ndarray, partial_row: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int]:
    prefix_len = _leading_prefix_len(partial_row)
    assert prefix_len is not None and 2 <= prefix_len <= 4, "expected a short leading prefix"

    color_a = int(template_row[0])
    color_b = int(template_row[1])
    prefix = partial_row[:prefix_len]
    a_pixels = prefix[template_row[:prefix_len] == color_a]
    b_pixels = prefix[template_row[:prefix_len] == color_b]

    assert len(a_pixels) > 0 and len(b_pixels) > 0, "prefix must expose both template colors"
    assert len(np.unique(a_pixels)) == 1 and len(np.unique(b_pixels)) == 1, "prefix mapping must be consistent"

    mapped_a = int(a_pixels[0])
    mapped_b = int(b_pixels[0])
    assert mapped_a != mapped_b, "the two template colors must map to different replacements"

    completed = np.where(template_row == color_a, mapped_a, mapped_b).astype(int)
    return completed, (mapped_a, mapped_b), prefix_len


def _parse_instance(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    template_rows = [r for r in range(g.shape[0]) if _is_alternating_template_row(g[r])]
    assert len(template_rows) == 1, "expected exactly one alternating full template row"
    template_r = int(template_rows[0])
    template_row = g[template_r].copy()

    partial_rows: list[int] = []
    completed_rows: list[np.ndarray] = []
    mappings: list[tuple[int, int]] = []
    prefix_lengths: list[int] = []
    r = template_r + 1
    while r < g.shape[0] and _is_partial_prefix_row(g[r]):
        completed, mapping, prefix_len = _decode_completed_row(template_row, g[r])
        partial_rows.append(r)
        completed_rows.append(completed)
        mappings.append(mapping)
        prefix_lengths.append(prefix_len)
        r += 1

    assert len(partial_rows) >= 2, "expected multiple prefix rows below the template row"
    lower_start = r
    assert lower_start < g.shape[0], "expected a lower stamping region"

    stamp = np.vstack(completed_rows).astype(int)
    lower = g[lower_start:].copy()
    sep_rows = np.flatnonzero(np.all(lower == SEPARATOR, axis=1))
    row_spans = _partition_spans(lower.shape[0], sep_rows)
    assert row_spans, "expected at least one macro row"

    sep_cols = np.flatnonzero(np.all(lower == SEPARATOR, axis=0))
    col_spans = _partition_spans(lower.shape[1], sep_cols) if len(sep_cols) else [(0, lower.shape[1] - 1)]
    assert col_spans, "expected at least one macro column"

    for r0, r1 in row_spans:
        assert r1 - r0 + 1 <= stamp.shape[0], "macro row cannot be taller than the decoded stamp"
    for c0, c1 in col_spans:
        assert c1 - c0 + 1 <= stamp.shape[1], "macro column cannot be wider than the decoded stamp"

    marked_cells: list[tuple[int, int]] = []
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            cell = lower[r0 : r1 + 1, c0 : c1 + 1]
            teal = np.argwhere(cell == MARKER)
            assert len(teal) <= 1, "expected at most one marker per macro cell"
            if len(teal) == 1:
                rr, cc = map(int, teal[0])
                assert (rr, cc) == (0, 0), "marker must sit at the macro cell's top-left corner"
                marked_cells.append((ri, ci))
            allowed = {BG, MARKER}
            assert set(map(int, np.unique(cell))) <= allowed, "lower non-separator cells may contain only background or markers"

    return {
        "template_r": template_r,
        "template_row": template_row,
        "partial_rows": partial_rows,
        "completed_rows": completed_rows,
        "mappings": mappings,
        "prefix_lengths": prefix_lengths,
        "lower_start": lower_start,
        "lower": lower,
        "sep_rows": sep_rows,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "stamp": stamp,
        "marked_cells": marked_cells,
    }


def _render_lower(lower_shape: tuple[int, int], row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]], marked_cells: list[tuple[int, int]], stamp: np.ndarray) -> np.ndarray:
    lower_out = np.full(lower_shape, BG, dtype=int)

    for i in range(len(row_spans) - 1):
        sep_r = row_spans[i][1] + 1
        lower_out[sep_r, :] = SEPARATOR
    for j in range(len(col_spans) - 1):
        sep_c = col_spans[j][1] + 1
        lower_out[:, sep_c] = SEPARATOR

    for ri, ci in marked_cells:
        r0, r1 = row_spans[ri]
        c0, c1 = col_spans[ci]
        cell = lower_out[r0 : r1 + 1, c0 : c1 + 1]
        paste(cell, stamp, 0, 0, clip=True)

    return lower_out


def main(grid: np.ndarray) -> np.ndarray:
    """Decode row-wise prefix color maps into a 2D stamp, then paste that stamp into every marked macro cell.

    Input structure:
    - The top region contains one full-width alternating template row using exactly two colors.
    - The next several rows are partial rows: each has a short non-black prefix starting at column 0 and black elsewhere.
    - The lower region is split into macro cells by full grey separator rows and may contain teal marker pixels at macro-cell top-left corners.

    Rule:
    - Decode each partial row by matching its visible prefix against the template row's two color roles, then complete the whole row with the inferred two-color substitution.
    - Stack the completed rows to form a 2D stamp template.
    - Keep the upper region, but replace each partial row with its completed form.
    - In the lower region, preserve grey separators and stamp the decoded 2D template into every marked macro cell, aligned to that cell's top-left corner and clipped to the cell if needed.
    - Leave unmarked macro cells black.

    Key insight:
    - The stamp is not shown directly: the template row defines only color roles, while each prefix row reveals one row-specific two-color mapping. Those decoded rows must be completed and stacked before the copied pattern is even knowable.
    """
    spec = _parse_instance(grid)
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for row_index, completed in zip(spec["partial_rows"], spec["completed_rows"]):
        out[int(row_index)] = completed

    lower_out = _render_lower(
        spec["lower"].shape,
        spec["row_spans"],
        spec["col_spans"],
        spec["marked_cells"],
        spec["stamp"],
    )
    out[int(spec["lower_start"]) :] = lower_out
    return out


def _make_template(width: int) -> np.ndarray:
    first = random.choice(TEMPLATE_PALETTE)
    second = TEMPLATE_PALETTE[1] if first == TEMPLATE_PALETTE[0] else TEMPLATE_PALETTE[0]
    return np.array([first if c % 2 == 0 else second for c in range(width)], dtype=int)


def _choose_row_mappings() -> list[tuple[int, int]]:
    distinct_count = random.randint(2, 3)
    base = random.sample(ORDERED_PREFIX_PAIRS, distinct_count)
    mappings = base + [random.choice(base) for _ in range(PARTIAL_ROW_COUNT - distinct_count)]
    random.shuffle(mappings)
    return mappings


def _build_recipe() -> dict[str, object]:
    width = random.randint(*TEMPLATE_WIDTH_RANGE)
    template = _make_template(width)
    mappings = _choose_row_mappings()
    prefix_lengths = [2, 3, 4, random.choice([2, 3, 4])]
    random.shuffle(prefix_lengths)
    completed_rows = [
        np.where(template == int(template[0]), mapped_a, mapped_b).astype(int)
        for mapped_a, mapped_b in mappings
    ]
    stamp = np.vstack(completed_rows).astype(int)

    macro_rows = random.randint(*MACRO_ROW_RANGE)
    marked_count = random.randint(2, min(4, macro_rows - 1))
    marked_rows = sorted(random.sample(range(macro_rows), marked_count))

    return {
        "template": template,
        "mappings": mappings,
        "prefix_lengths": prefix_lengths,
        "completed_rows": completed_rows,
        "stamp": stamp,
        "macro_rows": macro_rows,
        "marked_rows": marked_rows,
    }


def _render_input_from_recipe(recipe: dict[str, object]) -> np.ndarray:
    template = recipe["template"]
    completed_rows = recipe["completed_rows"]
    prefix_lengths = recipe["prefix_lengths"]
    macro_rows = int(recipe["macro_rows"])
    marked_rows = set(recipe["marked_rows"])
    stamp_h, stamp_w = recipe["stamp"].shape

    lower_h = macro_rows * stamp_h + (macro_rows - 1)
    grid = np.full((1 + len(completed_rows) + lower_h, stamp_w), BG, dtype=int)
    grid[0] = template

    for i, (completed, prefix_len) in enumerate(zip(completed_rows, prefix_lengths), start=1):
        grid[i, :prefix_len] = completed[:prefix_len]

    lower_start = 1 + len(completed_rows)
    for band in range(macro_rows):
        r0 = lower_start + band * (stamp_h + 1)
        if band in marked_rows:
            grid[r0, 0] = MARKER
        if band < macro_rows - 1:
            grid[r0 + stamp_h, :] = SEPARATOR

    return grid


def _expected_output_from_recipe(recipe: dict[str, object]) -> np.ndarray:
    template = recipe["template"]
    completed_rows = recipe["completed_rows"]
    stamp = recipe["stamp"]
    macro_rows = int(recipe["macro_rows"])
    marked_rows = set(recipe["marked_rows"])
    stamp_h, stamp_w = stamp.shape

    lower_h = macro_rows * stamp_h + (macro_rows - 1)
    out = np.full((1 + len(completed_rows) + lower_h, stamp_w), BG, dtype=int)
    out[0] = template
    for i, completed in enumerate(completed_rows, start=1):
        out[i] = completed

    lower_start = 1 + len(completed_rows)
    for band in range(macro_rows):
        r0 = lower_start + band * (stamp_h + 1)
        if band in marked_rows:
            out[r0 : r0 + stamp_h, :] = stamp
        if band < macro_rows - 1:
            out[r0 + stamp_h, :] = SEPARATOR

    return out


def _validate_candidate(inp: np.ndarray, recipe: dict[str, object]) -> bool:
    expected = _expected_output_from_recipe(recipe)
    try:
        parsed = _parse_instance(inp)
        actual = main(inp)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False

    if not np.array_equal(parsed["template_row"], recipe["template"]):
        return False
    if not np.array_equal(parsed["stamp"], recipe["stamp"]):
        return False
    if [tuple(pair) for pair in parsed["mappings"]] != [tuple(pair) for pair in recipe["mappings"]]:
        return False

    prefix_lengths = list(map(int, parsed["prefix_lengths"]))
    if set(prefix_lengths) != {2, 3, 4}:
        return False

    mappings = [tuple(pair) for pair in parsed["mappings"]]
    if len(set(mappings)) < 2:
        return False

    stamp = parsed["stamp"]
    if len({tuple(row.tolist()) for row in stamp}) < 2:
        return False

    if len(parsed["marked_cells"]) != len(recipe["marked_rows"]):
        return False
    if not (2 <= len(parsed["marked_cells"]) <= 4):
        return False
    if len(parsed["marked_cells"]) >= len(parsed["row_spans"]) * len(parsed["col_spans"]):
        return False

    for row_idx in parsed["partial_rows"]:
        if np.array_equal(inp[int(row_idx)], expected[int(row_idx)]):
            return False

    lower_start = int(parsed["lower_start"])
    lower_out = actual[lower_start:]
    for ri, (r0, r1) in enumerate(parsed["row_spans"]):
        for ci, (c0, c1) in enumerate(parsed["col_spans"]):
            cell = lower_out[r0 : r1 + 1, c0 : c1 + 1]
            if (ri, ci) in parsed["marked_cells"]:
                if not np.array_equal(cell, stamp[: cell.shape[0], : cell.shape[1]]):
                    return False
            else:
                if np.any(cell != BG):
                    return False

    for i in range(len(parsed["row_spans"]) - 1):
        if not np.all(lower_out[parsed["row_spans"][i][1] + 1, :] == SEPARATOR):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        recipe = _build_recipe()
        inp = _render_input_from_recipe(recipe)
        out = _expected_output_from_recipe(recipe)
        inp_sig = _signature(inp)
        out_sig = _signature(out)
        if inp_sig in _SEEN_INPUT_SIGS or out_sig in _SEEN_OUTPUT_SIGS:
            continue
        if not _validate_candidate(inp, recipe):
            continue
        _SEEN_INPUT_SIGS.add(inp_sig)
        _SEEN_OUTPUT_SIGS.add(out_sig)
        return inp
    raise RuntimeError("failed to generate a valid input for puzzle 5")
