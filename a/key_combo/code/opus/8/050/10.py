from __future__ import annotations

from collections import Counter
import random

import numpy as np


RED = 2
SEPARATOR = 5
BG = 0

Span = tuple[int, int]
CellPos = tuple[int, int]

CARDINALS = [(-1, 0), (0, -1), (0, 1), (1, 0)]
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
TEMPLATE_FAMILIES = [
    CARDINALS,
    DIAGONALS,
    CARDINALS + DIAGONALS,
    [(-1, 0), (0, 1), (1, 0)],
    [(0, -1), (-1, 0), (-1, 1), (1, 1)],
    [(-1, -1), (-1, 0), (0, 1), (1, 1)],
]
CELL_SIZES = (3, 5, 7, 9)
MAX_ATTEMPTS = 400


def _groups(indices: list[int]) -> list[Span]:
    assert indices, "expected separator lines"
    groups: list[Span] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
        else:
            groups.append((start, prev))
            start = prev = idx
    groups.append((start, prev))
    return groups


def _spans_between(line_groups: list[Span], limit: int) -> list[Span]:
    spans: list[Span] = []
    start = 0
    for line_start, line_end in line_groups:
        if start < line_start:
            spans.append((start, line_start - 1))
        start = line_end + 1
    if start < limit:
        spans.append((start, limit - 1))
    assert spans, "expected cells between separators"
    sizes = {end - start + 1 for start, end in spans}
    assert len(sizes) == 1, "cell interiors must have a common size"
    return spans


def _separator_color(grid: np.ndarray) -> int:
    row_colors = {
        int(grid[r, 0])
        for r in range(grid.shape[0])
        if np.all(grid[r, :] == grid[r, 0])
    }
    col_colors = {
        int(grid[0, c])
        for c in range(grid.shape[1])
        if np.all(grid[:, c] == grid[0, c])
    }
    candidates = row_colors & col_colors
    assert len(candidates) == 1, "expected one full-row/full-column separator color"
    return int(next(iter(candidates)))


def _compress_cells(grid: np.ndarray) -> tuple[list[Span], list[Span], list[list[np.ndarray]]]:
    sep = _separator_color(grid)
    row_lines = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == sep)]
    col_lines = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == sep)]
    row_spans = _spans_between(_groups(row_lines), grid.shape[0])
    col_spans = _spans_between(_groups(col_lines), grid.shape[1])

    cells: list[list[np.ndarray]] = []
    for r0, r1 in row_spans:
        row: list[np.ndarray] = []
        for c0, c1 in col_spans:
            row.append(grid[r0 : r1 + 1, c0 : c1 + 1].copy())
        cells.append(row)
    return row_spans, col_spans, cells


def _expand_cells(
    row_spans: list[Span],
    col_spans: list[Span],
    cells: list[list[np.ndarray]],
    separator: int,
) -> np.ndarray:
    h = row_spans[-1][1] + 1
    w = col_spans[-1][1] + 1
    out = np.full((h, w), int(separator), dtype=int)
    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            out[r0 : r1 + 1, c0 : c1 + 1] = cells[r][c]
    return out


def _cell_sig(cell: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in cell.ravel())


def _template_anchor(k: int) -> int:
    # For odd 3k this is the exact center; for even 3k it deterministically picks
    # the upper-left pixel of the 2x2 middle block.
    return (3 * int(k) - 1) // 2


def _template_from_exemplar(cell: np.ndarray) -> np.ndarray:
    size = int(cell.shape[0])
    assert cell.shape == (size, size) and size % 2 == 1
    center = size // 2
    assert int(cell[center, center]) == RED
    template = np.zeros((3, 3), dtype=int)
    for dr, dc in NEIGHBORS8:
        color = int(cell[center + dr, center + dc])
        if color != BG:
            template[dr + 1, dc + 1] = color
    assert np.any(template != BG), "exemplar template must be non-empty"
    return template


def _scaled_template(template: np.ndarray, k: int) -> np.ndarray:
    return np.kron(template, np.ones((int(k), int(k)), dtype=int))


def _stamp_mask(cell_size: int, scaled: np.ndarray) -> tuple[np.ndarray, bool]:
    mask = np.zeros((cell_size, cell_size), dtype=bool)
    center = cell_size // 2
    anchor = _template_anchor(scaled.shape[0] // 3)
    start_r = center - anchor
    start_c = center - anchor
    clipped = False
    for rr, cc in np.argwhere(scaled != BG):
        tr = start_r + int(rr)
        tc = start_c + int(cc)
        if 0 <= tr < cell_size and 0 <= tc < cell_size:
            mask[tr, tc] = True
        else:
            clipped = True
    return mask, clipped


def _stamp_into_cell(cell: np.ndarray, scaled: np.ndarray, k: int) -> np.ndarray:
    out = cell.copy()
    size = int(cell.shape[0])
    center = size // 2
    anchor = _template_anchor(k)
    start_r = center - anchor
    start_c = center - anchor
    for rr, cc in np.argwhere(scaled != BG):
        tr = start_r + int(rr)
        tc = start_c + int(cc)
        if 0 <= tr < size and 0 <= tc < size and int(out[tr, tc]) == BG:
            out[tr, tc] = int(scaled[rr, cc])
    return out


def _parse_puzzle(
    grid: np.ndarray,
) -> dict:
    row_spans, col_spans, cells = _compress_cells(grid)
    n_rows = len(row_spans)
    n_cols = len(col_spans)
    assert n_rows == n_cols, "expected a square cell lattice"
    cell_size = row_spans[0][1] - row_spans[0][0] + 1
    assert cell_size % 2 == 1, "cell interiors must have a center pixel"
    center = cell_size // 2

    exemplar: CellPos | None = None
    targets: list[CellPos] = []
    marker_positions: set[CellPos] = set()
    for r in range(n_rows):
        for c in range(n_cols):
            cell = cells[r][c]
            if int(cell[center, center]) != RED:
                continue
            marker_positions.add((r, c))
            ring = [
                int(cell[center + dr, center + dc])
                for dr, dc in NEIGHBORS8
            ]
            if any(color != BG for color in ring):
                assert exemplar is None, "expected exactly one decorated exemplar"
                exemplar = (r, c)
            else:
                targets.append((r, c))
    assert exemplar is not None, "missing exemplar marker cell"
    assert targets, "expected at least one target marker cell"

    counts: Counter[tuple[int, ...]] = Counter()
    sig_to_cell: dict[tuple[int, ...], np.ndarray] = {}
    non_marker_positions: list[CellPos] = []
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) in marker_positions:
                continue
            sig = _cell_sig(cells[r][c])
            counts[sig] += 1
            sig_to_cell.setdefault(sig, cells[r][c].copy())
            non_marker_positions.append((r, c))

    assert counts, "expected non-marker cells"
    majority_sig, majority_count = max(counts.items(), key=lambda kv: kv[1])
    assert sum(v == majority_count for v in counts.values()) == 1, "majority pattern must be unique"
    majority_cell = sig_to_cell[majority_sig]
    minority_positions = [
        pos for pos in non_marker_positions if _cell_sig(cells[pos[0]][pos[1]]) != majority_sig
    ]
    k = len(minority_positions)
    assert 1 <= k <= 3, "minority count must be 1..3"
    assert majority_count > k, "majority pattern must remain the most frequent non-marker pattern"

    template = _template_from_exemplar(cells[exemplar[0]][exemplar[1]])
    return {
        "row_spans": row_spans,
        "col_spans": col_spans,
        "cells": cells,
        "cell_size": cell_size,
        "separator": _separator_color(grid),
        "majority_sig": majority_sig,
        "majority_cell": majority_cell,
        "minority_positions": minority_positions,
        "exemplar": exemplar,
        "targets": targets,
        "template": template,
        "k": k,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count minority-pattern cells to choose a scale factor, then stamp the exemplar neighborhood into matching marker cells.

    Input structure:
    - A square lattice is divided by full grey separator lines into equal square cells.
    - One cell contains a red center pixel plus colored pixels in its 8-neighborhood; other marker cells contain only the same red center.
    - Non-marker cells are mostly copies of one majority interior pattern, with 1-3 minority cells using different interior patterns.

    Rule:
    - Ignore the marker cells when comparing cell interiors; among the remaining cells, identify the majority pattern and count how many cells differ from it.
    - Use that minority-cell count K as the scale factor for the exemplar cell's 3x3 center-neighborhood template, expanding each template pixel into a KxK block.
    - Stamp the scaled template around every target red marker, drawing only onto background pixels and clipping anything that would cross the target cell's separator boundaries.
    - Replace every minority cell with the majority pattern, while leaving the exemplar cell unchanged.

    Key insight:
    - The special minority cells do not contribute a shape to copy; only their count matters, and that hidden global count controls how large the exemplar neighborhood becomes before stamping.
    """
    g = np.asarray(grid, dtype=int)
    parsed = _parse_puzzle(g)
    out_cells = [[cell.copy() for cell in row] for row in parsed["cells"]]

    majority_cell = parsed["majority_cell"]
    for r, c in parsed["minority_positions"]:
        out_cells[r][c] = majority_cell.copy()

    k = int(parsed["k"])
    scaled = _scaled_template(parsed["template"], k)
    for r, c in parsed["targets"]:
        out_cells[r][c] = _stamp_into_cell(out_cells[r][c], scaled, k)

    return _expand_cells(
        parsed["row_spans"],
        parsed["col_spans"],
        out_cells,
        parsed["separator"],
    )


def _layout_candidates(k: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    min_cell = max(3, 2 * k + 1)
    for n in range(3, 6):
        for cell_size in CELL_SIZES:
            if cell_size < min_cell:
                continue
            if n * cell_size + (n - 1) <= 30:
                out.append((n, cell_size))
    return out


def _choose_layout(k: int) -> tuple[int, int]:
    candidates = _layout_candidates(k)
    weighted: list[tuple[int, int]] = []
    for n, cell_size in candidates:
        clip_bias = 3 if cell_size < 3 * k else 1
        spread_bias = 2 if n >= 4 else 1
        weighted.extend([(n, cell_size)] * (clip_bias * spread_bias))
    return random.choice(weighted)


def _rand_colors(count: int, *, exclude: set[int]) -> list[int]:
    pool = [c for c in range(1, 10) if c not in exclude]
    return random.sample(pool, count)


def _make_template() -> np.ndarray:
    shape = random.choice(TEMPLATE_FAMILIES)
    if random.random() < 0.4:
        shape = random.sample(NEIGHBORS8, random.randint(3, 6))
    fill_count = 1 if random.random() < 0.55 else 2
    fills = _rand_colors(fill_count, exclude={BG, RED, SEPARATOR})
    template = np.zeros((3, 3), dtype=int)
    if fill_count == 1:
        for dr, dc in shape:
            template[dr + 1, dc + 1] = fills[0]
    else:
        for dr, dc in shape:
            if (dr, dc) in CARDINALS:
                template[dr + 1, dc + 1] = fills[0]
            else:
                template[dr + 1, dc + 1] = fills[1]
        if len(set(int(template[dr + 1, dc + 1]) for dr, dc in shape)) == 1:
            dr, dc = random.choice(shape)
            template[dr + 1, dc + 1] = fills[1]
    assert np.any(template != BG)
    return template


def _safe_majority_positions(cell_size: int, template: np.ndarray, k: int) -> list[tuple[int, int]]:
    scaled = _scaled_template(template, k)
    blocked, _clipped = _stamp_mask(cell_size, scaled)
    center = cell_size // 2
    for dr, dc in NEIGHBORS8:
        blocked[center + dr, center + dc] = True
    blocked[center, center] = True
    return [tuple(map(int, rc)) for rc in np.argwhere(~blocked)]


def _make_majority(cell_size: int, template: np.ndarray, k: int) -> np.ndarray:
    majority = np.zeros((cell_size, cell_size), dtype=int)
    safe = _safe_majority_positions(cell_size, template, k)
    if not safe or random.random() < 0.45:
        return majority

    accent = _rand_colors(1, exclude={BG, RED, SEPARATOR})[0]
    count = random.randint(1, min(3, len(safe)))
    chosen = random.sample(safe, count)
    for r, c in chosen:
        majority[r, c] = accent
    return majority


def _make_marker_cell(majority: np.ndarray, template: np.ndarray | None) -> np.ndarray:
    cell = majority.copy()
    size = cell.shape[0]
    center = size // 2
    cell[center, center] = RED
    if template is not None:
        for dr, dc in NEIGHBORS8:
            color = int(template[dr + 1, dc + 1])
            if color != BG:
                cell[center + dr, center + dc] = color
    return cell


def _random_pattern(cell_size: int, *, avoid: set[tuple[int, ...]]) -> np.ndarray:
    colors = _rand_colors(3, exclude={BG, RED, SEPARATOR})
    for _ in range(200):
        pat = np.zeros((cell_size, cell_size), dtype=int)
        count = random.randint(1, min(5, cell_size + 1))
        coords = random.sample(
            [(r, c) for r in range(cell_size) for c in range(cell_size)],
            count,
        )
        palette = [colors[0]] if random.random() < 0.55 else colors[:2]
        for r, c in coords:
            pat[r, c] = random.choice(palette)
        center = cell_size // 2
        if int(pat[center, center]) == RED:
            continue
        sig = _cell_sig(pat)
        if sig not in avoid:
            return pat
    raise AssertionError("failed to sample a distinct minority pattern")


def _place_cells(
    n: int,
    target_count: int,
    minority_count: int,
) -> tuple[CellPos, list[CellPos], list[CellPos]]:
    positions = [(r, c) for r in range(n) for c in range(n)]
    exemplar = random.choice(positions)
    remaining = [pos for pos in positions if pos != exemplar]
    random.shuffle(remaining)
    targets = remaining[:target_count]
    minorities = remaining[target_count : target_count + minority_count]
    return exemplar, targets, minorities


def _replace_cell(
    grid: np.ndarray,
    row_spans: list[Span],
    col_spans: list[Span],
    pos: CellPos,
    patch: np.ndarray,
) -> np.ndarray:
    out = grid.copy()
    r0, r1 = row_spans[pos[0]]
    c0, c1 = col_spans[pos[1]]
    out[r0 : r1 + 1, c0 : c1 + 1] = patch
    return out


def _build_grid_from_cells(n: int, cell_size: int, cells: list[list[np.ndarray]]) -> np.ndarray:
    side = n * cell_size + (n - 1)
    grid = np.full((side, side), SEPARATOR, dtype=int)
    for r in range(n):
        r0 = r * (cell_size + 1)
        for c in range(n):
            c0 = c * (cell_size + 1)
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = cells[r][c]
    return grid


def _validate_instance(inp: np.ndarray, expected: np.ndarray, *, require_clip: bool) -> bool:
    try:
        parsed = _parse_puzzle(inp)
        actual = main(inp)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(inp, expected):
        return False

    k = int(parsed["k"])
    template_nonbg = int(np.count_nonzero(parsed["template"]))
    target_count = len(parsed["targets"])
    if k == target_count or k == template_nonbg:
        return False

    minority_pixels = [
        int(np.count_nonzero(parsed["cells"][r][c]))
        for r, c in parsed["minority_positions"]
    ]
    if sum(minority_pixels) in {k, target_count, template_nonbg}:
        return False

    majority_sig = parsed["majority_sig"]

    for r, c in parsed["minority_positions"]:
        cell = parsed["cells"][r][c]
        if _cell_sig(cell) == majority_sig:
            return False
        r0, r1 = parsed["row_spans"][r]
        c0, c1 = parsed["col_spans"][c]
        if not np.array_equal(expected[r0 : r1 + 1, c0 : c1 + 1], parsed["majority_cell"]):
            return False

    changed_targets = 0
    for r, c in parsed["targets"]:
        r0, r1 = parsed["row_spans"][r]
        c0, c1 = parsed["col_spans"][c]
        before = inp[r0 : r1 + 1, c0 : c1 + 1]
        after = expected[r0 : r1 + 1, c0 : c1 + 1]
        if np.count_nonzero(after != before) == 0:
            return False
        changed_targets += 1
    if changed_targets == 0:
        return False

    scaled = _scaled_template(parsed["template"], k)
    _mask, clipped = _stamp_mask(parsed["cell_size"], scaled)
    if require_clip and not clipped:
        return False

    if k > 1:
        test_pos = parsed["minority_positions"][0]
        altered = _replace_cell(
            inp,
            parsed["row_spans"],
            parsed["col_spans"],
            test_pos,
            parsed["majority_cell"],
        )
        try:
            if np.array_equal(main(altered), expected):
                return False
        except Exception:
            return False
    else:
        majority_cells = []
        marker_positions = set(parsed["targets"]) | {parsed["exemplar"]}
        for r in range(len(parsed["cells"])):
            for c in range(len(parsed["cells"])):
                if (r, c) in marker_positions or (r, c) in parsed["minority_positions"]:
                    continue
                if _cell_sig(parsed["cells"][r][c]) == majority_sig:
                    majority_cells.append((r, c))
        if not majority_cells:
            return False
        altered = _replace_cell(
            inp,
            parsed["row_spans"],
            parsed["col_spans"],
            majority_cells[0],
            parsed["cells"][parsed["minority_positions"][0][0]][parsed["minority_positions"][0][1]],
        )
        try:
            if np.array_equal(main(altered), expected):
                return False
        except Exception:
            return False

    return True


def _generate_once() -> tuple[np.ndarray, np.ndarray, bool] | None:
    k = random.choices([1, 2, 3], weights=[3, 2, 3], k=1)[0]
    n, cell_size = _choose_layout(k)

    allowed_targets = [
        t
        for t in range(1, 4)
        if n * n - (1 + t + k) > k
    ]
    if not allowed_targets:
        return None
    target_count = random.choice(allowed_targets)

    template = _make_template()
    scaled = _scaled_template(template, k)
    _blocked, clipped = _stamp_mask(cell_size, scaled)

    majority = _make_majority(cell_size, template, k)
    exemplar_cell = _make_marker_cell(majority, template)
    target_cell = _make_marker_cell(majority, None)
    stamped_target = _stamp_into_cell(target_cell, scaled, k)
    if np.array_equal(stamped_target, target_cell):
        return None

    exemplar, targets, minorities = _place_cells(n, target_count, k)
    cells_in = [[majority.copy() for _ in range(n)] for _ in range(n)]
    cells_out = [[majority.copy() for _ in range(n)] for _ in range(n)]
    cells_in[exemplar[0]][exemplar[1]] = exemplar_cell
    cells_out[exemplar[0]][exemplar[1]] = exemplar_cell.copy()

    for r, c in targets:
        cells_in[r][c] = target_cell.copy()
        cells_out[r][c] = stamped_target.copy()

    minority_types = random.randint(1, k)
    avoid = {
        _cell_sig(majority),
        _cell_sig(exemplar_cell),
        _cell_sig(target_cell),
        _cell_sig(stamped_target),
    }
    pattern_pool: list[np.ndarray] = []
    for _ in range(minority_types):
        pat = _random_pattern(cell_size, avoid=avoid)
        pattern_pool.append(pat)
        avoid.add(_cell_sig(pat))

    for idx, pos in enumerate(minorities):
        pat = pattern_pool[idx % minority_types]
        cells_in[pos[0]][pos[1]] = pat.copy()
        cells_out[pos[0]][pos[1]] = majority.copy()

    inp = _build_grid_from_cells(n, cell_size, cells_in)
    out = _build_grid_from_cells(n, cell_size, cells_out)
    return inp, out, clipped


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _generate_once()
        if built is None:
            continue
        inp, out, clipped = built
        if _validate_instance(inp, out, require_clip=clipped and random.random() < 0.7):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
