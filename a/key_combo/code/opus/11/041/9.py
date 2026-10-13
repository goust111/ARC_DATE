from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
SEP = 7
CELL_COLORS = (1, 2, 3, 4)
GEN_MAX_ATTEMPTS = 400


def _bands(size: int, separators: Iterable[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for cut in sorted(int(v) for v in separators):
        if start < cut:
            bands.append((start, cut))
        start = cut + 1
    if start < size:
        bands.append((start, size))
    return bands


def _gravity_cell(cell: np.ndarray) -> np.ndarray:
    h, w = map(int, cell.shape)
    out = np.zeros((h, w), dtype=int)
    for c in range(w):
        values = [int(v) for v in cell[:, c] if int(v) != BG]
        if values:
            out[h - len(values) :, c] = values
    return out


def _forms_bottom_rectangle(cell: np.ndarray) -> tuple[bool, int]:
    h, w = map(int, cell.shape)
    mask = cell != BG
    count = int(mask.sum())
    if count == 0 or count % w != 0:
        return False, 0
    block_h = count // w
    if block_h > h:
        return False, 0
    expected = np.zeros((h, w), dtype=bool)
    expected[h - block_h :, :] = True
    return bool(np.array_equal(mask, expected)), int(block_h)


def _cell_summary(cell: np.ndarray) -> dict:
    dropped = _gravity_cell(cell)
    colors = sorted({int(v) for v in cell.flatten() if int(v) != BG})
    rect_ok, block_h = _forms_bottom_rectangle(dropped)
    single_color = len(colors) == 1
    valid = bool(single_color and rect_ok)
    patch = (
        dropped[dropped.shape[0] - block_h :, :].copy()
        if valid
        else np.zeros((0, 0), dtype=int)
    )
    return {
        "input": np.asarray(cell, dtype=int),
        "gravity": dropped,
        "colors": colors,
        "single_color": single_color,
        "rect_ok": rect_ok,
        "block_h": int(block_h),
        "valid": valid,
        "patch": patch,
        "changed": not np.array_equal(cell, dropped),
        "count": int((cell != BG).sum()),
    }


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_rows = [r for r in range(h) if np.all(g[r, :] == SEP)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == SEP)]
    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)

    cells: list[list[dict]] = []
    for ri, (r0, r1) in enumerate(row_bands):
        row_cells: list[dict] = []
        for ci, (c0, c1) in enumerate(col_bands):
            cell = g[r0:r1, c0:c1]
            summary = _cell_summary(cell)
            summary["row"] = int(ri)
            summary["col"] = int(ci)
            summary["bbox"] = (int(r0), int(c0), int(r1 - 1), int(c1 - 1))
            row_cells.append(summary)
        cells.append(row_cells)
    return {
        "sep_rows": sep_rows,
        "sep_cols": sep_cols,
        "row_bands": row_bands,
        "col_bands": col_bands,
        "cells": cells,
    }


def _pack_valid_rows(valid_rows: list[list[np.ndarray]]) -> np.ndarray:
    strips: list[np.ndarray] = []
    for patches in valid_rows:
        if not patches:
            continue
        row_h = max(int(p.shape[0]) for p in patches)
        row_w = sum(int(p.shape[1]) for p in patches) + (len(patches) - 1)
        strip = np.zeros((row_h, row_w), dtype=int)
        c = 0
        for patch in patches:
            ph, pw = map(int, patch.shape)
            strip[:ph, c : c + pw] = patch
            c += pw + 1
        strips.append(strip)

    if not strips:
        return np.zeros((1, 1), dtype=int)

    out_h = sum(int(s.shape[0]) for s in strips)
    out_w = max(int(s.shape[1]) for s in strips)
    out = np.zeros((out_h, out_w), dtype=int)
    r = 0
    for strip in strips:
        sh, sw = map(int, strip.shape)
        out[r : r + sh, :sw] = strip
        r += sh
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split a separator lattice into cells, let each cell's pixels fall, and pack only the valid monochrome deposits.

    Input structure:
    - The background is black, and three full orange rows plus three full orange columns divide the canvas into a 4x4 lattice of rectangular cells.
    - Each cell contains a small set of non-black, non-orange pixels chosen from a visible color palette, with empty space above and around them.
    - Pixel colors and column distributions vary by cell, so some cells are monochrome, some are mixed-color, and some already hint at different post-gravity shapes.

    Rule:
    - Detect the full orange separator rows and columns and use them to recover the 4x4 cell boundaries.
    - Inside each cell, apply downward gravity independently per column: every non-black pixel falls to the lowest available row in its own column while preserving vertical order within the column.
    - A cell is valid only if, after gravity, all non-black pixels share one color and occupy exactly a solid full-width rectangle anchored to the bottom of the cell.
    - Extract each valid deposited rectangle, preserve the original lattice row order and left-to-right order within each row, compact each non-empty row with one black column between rectangles, then stack the non-empty rows with no blank spacer rows.

    Key insight:
    - Monochrome cells are not sufficient: gravity preserves column counts, so only cells whose dropped pixels become a bottom-anchored full-width rectangle survive into the packed output.
    """
    info = _analyze_grid(np.asarray(grid, dtype=int))
    assert len(info["row_bands"]) == 4 and len(info["col_bands"]) == 4

    valid_rows: list[list[np.ndarray]] = []
    for row in info["cells"]:
        patches = [cell["patch"] for cell in row if cell["valid"]]
        if patches:
            valid_rows.append(patches)
    return _pack_valid_rows(valid_rows)


def _random_distinct_rows(height: int, count: int) -> list[int]:
    rows = sorted(random.sample(range(height), count))
    return [int(r) for r in rows]


def _scatter_columns(column_values: list[list[int]], height: int) -> np.ndarray:
    width = len(column_values)
    cell = np.zeros((height, width), dtype=int)
    for c, values in enumerate(column_values):
        if not values:
            continue
        rows = _random_distinct_rows(height, len(values))
        for r, color in zip(rows, values):
            cell[int(r), int(c)] = int(color)
    return cell


def _needs_motion(cell: np.ndarray) -> bool:
    return not np.array_equal(cell, _gravity_cell(cell))


def _make_valid_cell(height: int, width: int, block_h: int, color: int) -> np.ndarray:
    assert 2 <= width * block_h <= 6
    for _ in range(80):
        cell = _scatter_columns([[int(color)] * block_h for _ in range(width)], height)
        summary = _cell_summary(cell)
        if summary["valid"] and summary["block_h"] == block_h and _needs_motion(cell):
            return cell
    raise RuntimeError("failed to build valid cell")


def _random_nonrect_counts(height: int, width: int, total: int) -> list[int]:
    assert 2 <= total <= min(6, height * width)
    for _ in range(200):
        counts = [0] * width
        remaining = total
        cols = list(range(width))
        random.shuffle(cols)
        while remaining > 0:
            c = random.choice(cols)
            if counts[c] < height:
                counts[c] += 1
                remaining -= 1
        if len({v for v in counts if v > 0}) == 1 and 0 not in counts:
            continue
        cell = _scatter_columns([[1] * n for n in counts], height)
        if not _forms_bottom_rectangle(_gravity_cell(cell))[0]:
            return counts
    raise RuntimeError("failed to sample non-rectangular column counts")


def _make_mono_bad_geom(height: int, width: int) -> np.ndarray:
    color = random.choice(CELL_COLORS)
    max_total = min(6, height * width)
    totals = list(range(max(2, width), max_total + 1))
    random.shuffle(totals)
    for total in totals:
        counts = _random_nonrect_counts(height, width, total)
        for _ in range(60):
            cell = _scatter_columns([[int(color)] * n for n in counts], height)
            summary = _cell_summary(cell)
            if summary["single_color"] and not summary["rect_ok"] and _needs_motion(cell):
                return cell
    raise RuntimeError("failed to build monochrome invalid cell")


def _make_multi_good_geom(height: int, width: int) -> np.ndarray:
    colors = random.sample(CELL_COLORS, 2)
    ks = [1]
    if width == 3 and height >= 2:
        ks.append(2)
    random.shuffle(ks)
    for block_h in ks:
        total = width * block_h
        if not 2 <= total <= 6:
            continue
        for _ in range(80):
            column_values: list[list[int]] = []
            used = set()
            for _c in range(width):
                stack = [random.choice(colors) for _ in range(block_h)]
                used.update(stack)
                column_values.append(stack)
            if len(used) < 2:
                continue
            cell = _scatter_columns(column_values, height)
            summary = _cell_summary(cell)
            if (not summary["single_color"]) and summary["rect_ok"] and _needs_motion(cell):
                return cell
    raise RuntimeError("failed to build mixed-color rectangular cell")


def _make_multi_bad_geom(height: int, width: int) -> np.ndarray:
    colors = random.sample(CELL_COLORS, 2)
    max_total = min(6, height * width)
    totals = list(range(2, max_total + 1))
    random.shuffle(totals)
    for total in totals:
        counts = _random_nonrect_counts(height, width, total)
        for _ in range(80):
            column_values: list[list[int]] = []
            used = set()
            for n in counts:
                stack = [random.choice(colors) for _ in range(n)]
                used.update(stack)
                column_values.append(stack)
            if len(used) < 2:
                continue
            cell = _scatter_columns(column_values, height)
            summary = _cell_summary(cell)
            if (not summary["single_color"]) and (not summary["rect_ok"]) and _needs_motion(cell):
                return cell
    raise RuntimeError("failed to build mixed-color invalid cell")


def _layout_bounds(lengths: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    bands: list[tuple[int, int]] = []
    separators: list[int] = []
    pos = 0
    for i, span in enumerate(lengths):
        bands.append((pos, pos + span))
        pos += span
        if i < len(lengths) - 1:
            separators.append(pos)
            pos += 1
    return bands, separators


def _expected_output(expected_cells: dict[tuple[int, int], tuple[int, int, int]]) -> np.ndarray:
    valid_rows: list[list[np.ndarray]] = []
    for ri in range(4):
        row_patches: list[np.ndarray] = []
        for ci in range(4):
            spec = expected_cells.get((ri, ci))
            if spec is None:
                continue
            color, block_h, width = spec
            row_patches.append(np.full((block_h, width), int(color), dtype=int))
        if row_patches:
            valid_rows.append(row_patches)
    return _pack_valid_rows(valid_rows)


def _sample_col_widths() -> list[int]:
    widths = [random.choice((3, 4, 5)) for _ in range(4)]
    if 3 not in widths:
        widths[random.randrange(4)] = 3
    return [int(w) for w in widths]


def _pick_valid_positions(col_widths: list[int]) -> tuple[set[tuple[int, int]], dict[int, int]]:
    width3_cols = [i for i, w in enumerate(col_widths) if w == 3]
    tall_row = random.randrange(4) if width3_cols and random.random() < 0.7 else None
    valid_height_by_row = {ri: 1 for ri in range(4)}
    if tall_row is not None:
        valid_height_by_row[tall_row] = 2

    candidates: list[tuple[int, int]] = []
    for ri in range(4):
        for ci, width in enumerate(col_widths):
            if valid_height_by_row[ri] == 2 and width != 3:
                continue
            candidates.append((ri, ci))

    target = random.randint(4, min(8, len(candidates)))
    for _ in range(200):
        chosen = set(random.sample(candidates, target))
        if tall_row is not None and not any(pos[0] == tall_row for pos in chosen):
            continue
        row_counts = [sum(1 for r, _c in chosen if r == ri) for ri in range(4)]
        nonempty_rows = [cnt for cnt in row_counts if cnt > 0]
        if len(nonempty_rows) < 2:
            continue
        if max(nonempty_rows) < 2:
            continue
        return chosen, valid_height_by_row
    raise RuntimeError("failed to sample valid positions")


def _ensure_color_diversity(specs: dict[tuple[int, int], tuple[int, int, int]]) -> None:
    valid_colors = {color for color, _h, _w in specs.values()}
    if len(valid_colors) >= 2:
        return
    (r0, c0), (color, block_h, width) = next(iter(specs.items()))
    replacement = next(col for col in CELL_COLORS if col != color)
    specs[(r0, c0)] = (int(replacement), int(block_h), int(width))


def _build_candidate() -> tuple[np.ndarray, np.ndarray]:
    row_heights = [random.choice((4, 5)) for _ in range(4)]
    col_widths = _sample_col_widths()
    row_bands, sep_rows = _layout_bounds(row_heights)
    col_bands, sep_cols = _layout_bounds(col_widths)

    grid = np.zeros((row_bands[-1][1] + 3, col_bands[-1][1] + 3), dtype=int)
    for r in sep_rows:
        grid[int(r), :] = SEP
    for c in sep_cols:
        grid[:, int(c)] = SEP

    valid_positions, valid_height_by_row = _pick_valid_positions(col_widths)
    expected_specs: dict[tuple[int, int], tuple[int, int, int]] = {}

    invalid_positions = [(ri, ci) for ri in range(4) for ci in range(4) if (ri, ci) not in valid_positions]
    random.shuffle(invalid_positions)
    invalid_types = ["mono_bad", "multi_good", "multi_bad"]
    extra_choices = [random.choice(invalid_types) for _ in range(len(invalid_positions) - len(invalid_types))]
    invalid_plan = invalid_types + extra_choices
    random.shuffle(invalid_plan)
    plan_map = {pos: invalid_plan[i] for i, pos in enumerate(invalid_positions)}

    valid_colors = [random.choice(CELL_COLORS) for _ in range(len(valid_positions))]
    if len(set(valid_colors)) == 1 and len(valid_colors) >= 2:
        valid_colors[-1] = next(c for c in CELL_COLORS if c != valid_colors[-1])
    random.shuffle(valid_colors)

    for ri in range(4):
        for ci in range(4):
            r0, r1 = row_bands[ri]
            c0, c1 = col_bands[ci]
            height = r1 - r0
            width = c1 - c0
            if (ri, ci) in valid_positions:
                block_h = valid_height_by_row[ri]
                color = valid_colors.pop()
                cell = _make_valid_cell(height, width, block_h, color)
                expected_specs[(ri, ci)] = (int(color), int(block_h), int(width))
            else:
                kind = plan_map[(ri, ci)]
                if kind == "mono_bad":
                    cell = _make_mono_bad_geom(height, width)
                elif kind == "multi_good":
                    cell = _make_multi_good_geom(height, width)
                else:
                    cell = _make_multi_bad_geom(height, width)
            grid[r0:r1, c0:c1] = cell

    _ensure_color_diversity(expected_specs)
    for (ri, ci), (color, block_h, _width) in expected_specs.items():
        r0, r1 = row_bands[ri]
        c0, c1 = col_bands[ci]
        grid[r0:r1, c0:c1] = _make_valid_cell(r1 - r0, c1 - c0, block_h, color)

    return grid, _expected_output(expected_specs)


def _validate_generated(grid: np.ndarray, expected: np.ndarray) -> None:
    info = _analyze_grid(grid)
    assert len(info["sep_rows"]) == 3 and len(info["sep_cols"]) == 3
    assert len(info["row_bands"]) == 4 and len(info["col_bands"]) == 4

    valid_positions = set()
    counts_by_row = [0, 0, 0, 0]
    saw_mono_bad = False
    saw_multi_good = False
    saw_multi_bad = False
    changed_cells = 0

    for row in info["cells"]:
        for cell in row:
            inner = cell["input"]
            assert int((inner == SEP).sum()) == 0
            assert set(int(v) for v in np.unique(inner)) <= {0, 1, 2, 3, 4}
            assert 2 <= cell["count"] <= 6
            if cell["changed"]:
                changed_cells += 1
            if cell["valid"]:
                valid_positions.add((cell["row"], cell["col"]))
                counts_by_row[cell["row"]] += 1
            elif cell["single_color"] and not cell["rect_ok"]:
                saw_mono_bad = True
            elif (not cell["single_color"]) and cell["rect_ok"]:
                saw_multi_good = True
            elif (not cell["single_color"]) and (not cell["rect_ok"]):
                saw_multi_bad = True

    assert 4 <= len(valid_positions) <= 8
    assert changed_cells == 16
    assert len([cnt for cnt in counts_by_row if cnt > 0]) >= 2
    assert max(counts_by_row) >= 2
    assert saw_mono_bad and saw_multi_good and saw_multi_bad

    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual, _pack_valid_rows([[c["patch"] for c in row if c["valid"]] for row in info["cells"] if any(c["valid"] for c in row)]))
    assert len({int(v) for v in np.unique(actual) if int(v) != 0}) >= 2
    assert max(actual.shape) <= 30


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid, expected = _build_candidate()
        try:
            _validate_generated(grid, expected)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
