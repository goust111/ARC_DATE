from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GENERATION_RETRIES = 400


Rect = tuple[int, int, int, int]  # r0, r1, c0, c1 with exclusive ends


def _blank_indices(grid: np.ndarray, axis: int) -> list[int]:
    return [int(i) for i, is_blank in enumerate(np.all(grid == BG, axis=axis)) if bool(is_blank)]


def _segments(length: int, blanks: list[int]) -> list[tuple[int, int]]:
    cuts = [-1, *sorted(blanks), int(length)]
    parts: list[tuple[int, int]] = []
    for left, right in zip(cuts, cuts[1:]):
        start = left + 1
        end = right
        if start < end:
            parts.append((start, end))
    return parts


def _panel_rects(row_segments: list[tuple[int, int]], col_segments: list[tuple[int, int]]) -> list[tuple[int, int, Rect]]:
    rects: list[tuple[int, int, Rect]] = []
    for ri, (r0, r1) in enumerate(row_segments):
        for ci, (c0, c1) in enumerate(col_segments):
            rects.append((ri, ci, (r0, r1, c0, c1)))
    return rects


def _area(rect: Rect) -> int:
    r0, r1, c0, c1 = rect
    return int(r1 - r0) * int(c1 - c0)


def _coverage_rect(
    panel_rect: Rect,
    panel_idx: tuple[int, int],
    main_rect: Rect,
    main_idx: tuple[int, int],
) -> Rect | None:
    pr0, pr1, pc0, pc1 = panel_rect
    mr0, mr1, mc0, mc1 = main_rect
    drow = int(panel_idx[0] - main_idx[0])
    dcol = int(panel_idx[1] - main_idx[1])
    if abs(drow) > 1 or abs(dcol) > 1 or (drow == 0 and dcol == 0):
        return None

    if drow == -1:
        rr0, rr1 = 0, pr1 - pr0
    elif drow == 0:
        rr0, rr1 = 0, mr1 - mr0
    else:
        rr0, rr1 = (mr1 - mr0) - (pr1 - pr0), mr1 - mr0

    if dcol == -1:
        cc0, cc1 = 0, pc1 - pc0
    elif dcol == 0:
        cc0, cc1 = 0, mc1 - mc0
    else:
        cc0, cc1 = (mc1 - mc0) - (pc1 - pc0), mc1 - mc0

    return rr0, rr1, cc0, cc1


def _source_local_from_target(
    target_r: int,
    target_c: int,
    coverage: Rect,
    panel_shape: tuple[int, int],
    panel_idx: tuple[int, int],
    main_idx: tuple[int, int],
) -> tuple[int, int]:
    tr0, _tr1, tc0, _tc1 = coverage
    ph, pw = panel_shape
    drow = int(panel_idx[0] - main_idx[0])
    dcol = int(panel_idx[1] - main_idx[1])

    local_r = int(target_r - tr0)
    local_c = int(target_c - tc0)
    src_r = ph - 1 - local_r if drow != 0 else local_r
    src_c = pw - 1 - local_c if dcol != 0 else local_c
    return src_r, src_c


def _choose_supported_coords(allowed: np.ndarray) -> list[tuple[int, int]] | None:
    h, w = map(int, allowed.shape)
    if h == 0 or w == 0:
        return None
    if np.any(~allowed.any(axis=1)) or np.any(~allowed.any(axis=0)):
        return None

    for _ in range(50):
        chosen: set[tuple[int, int]] = set()
        col_load = [0] * w
        rows = list(range(h))
        random.shuffle(rows)
        ok = True
        for r in rows:
            cols = np.flatnonzero(allowed[r]).astype(int).tolist()
            if not cols:
                ok = False
                break
            min_load = min(col_load[c] for c in cols)
            options = [c for c in cols if col_load[c] == min_load]
            c = random.choice(options)
            chosen.add((r, c))
            col_load[c] += 1
        if not ok:
            continue

        row_load = [0] * h
        for r, _c in chosen:
            row_load[r] += 1

        cols = list(range(w))
        random.shuffle(cols)
        for c in cols:
            if any(cc == c for _rr, cc in chosen):
                continue
            rows_here = np.flatnonzero(allowed[:, c]).astype(int).tolist()
            if not rows_here:
                ok = False
                break
            min_load = min(row_load[r] for r in rows_here)
            options = [r for r in rows_here if row_load[r] == min_load]
            r = random.choice(options)
            chosen.add((r, c))
            row_load[r] += 1
        if not ok:
            continue

        remaining = [(r, c) for r in range(h) for c in range(w) if allowed[r, c] and (r, c) not in chosen]
        random.shuffle(remaining)
        extra_cap = max(1, (h * w) // 5)
        for r, c in remaining[: random.randint(0, extra_cap)]:
            chosen.add((r, c))

        if len({r for r, _c in chosen}) == h and len({c for _r, c in chosen}) == w:
            return sorted(chosen)
    return None


def _paint_cells(grid: np.ndarray, top: int, left: int, cells: list[tuple[int, int, int]]) -> None:
    for r, c, color in cells:
        grid[top + r, left + c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Fold every side panel across the blank crease lines onto the unique largest panel and crop that panel.

    Input structure:
    - Black background cells fill the canvas, with exactly one or two fully blank rows/columns acting as crease lines.
    - Those blank lines partition the canvas into rectangular panels, and exactly one panel has the largest area.
    - Colored pixels may appear in the largest panel and in neighboring or diagonal side panels whose sizes do not exceed the largest panel.

    Rule:
    - Detect the fully blank rows and columns, split the canvas into panels, and choose the unique largest panel as the base panel.
    - Mirror each panel separated from the base by one blank row and/or one blank column across the corresponding crease line(s), mapping diagonal panels by two mirrors.
    - Stamp folded pixels only onto black cells of the base panel; existing colored cells in the base panel remain unchanged, then output the cropped base panel.

    Key insight:
    - The empty full-length lines are physical fold creases rather than missing guides, so the solution is determined by panel segmentation plus mirror overlays with base-panel priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    blank_rows = _blank_indices(g, axis=1)
    blank_cols = _blank_indices(g, axis=0)
    assert 1 <= len(blank_rows) + len(blank_cols) <= 2, "expected one or two crease lines"

    row_segments = _segments(g.shape[0], blank_rows)
    col_segments = _segments(g.shape[1], blank_cols)
    rects = _panel_rects(row_segments, col_segments)
    areas = [_area(rect) for _ri, _ci, rect in rects]
    max_area = max(areas)
    assert areas.count(max_area) == 1, "expected a unique largest panel"

    main_pos = areas.index(max_area)
    main_ri, main_ci, main_rect = rects[main_pos]
    mr0, mr1, mc0, mc1 = main_rect
    out = g[mr0:mr1, mc0:mc1].copy()

    fold_panels = []
    for ri, ci, rect in rects:
        coverage = _coverage_rect(rect, (ri, ci), main_rect, (main_ri, main_ci))
        if coverage is None:
            continue
        fold_panels.append((int(_area(rect)), ri, ci, rect, coverage))
    fold_panels.sort(key=lambda item: (item[0], item[1], item[2]))

    for _panel_area, ri, ci, rect, _coverage in fold_panels:
        r0, r1, c0, c1 = rect
        for r in range(r0, r1):
            for c in range(c0, c1):
                color = int(g[r, c])
                if color == BG:
                    continue

                rr = r
                cc = c
                if ri != main_ri:
                    crease_r = blank_rows[min(ri, main_ri)]
                    rr = 2 * crease_r - rr
                if ci != main_ci:
                    crease_c = blank_cols[min(ci, main_ci)]
                    cc = 2 * crease_c - cc

                assert mr0 <= rr < mr1 and mc0 <= cc < mc1, "folded cell must land inside the base panel"
                local_r = rr - mr0
                local_c = cc - mc0
                if int(out[local_r, local_c]) == BG:
                    out[local_r, local_c] = color

    return out


def _build_layout() -> tuple[
    str,
    list[tuple[int, int]],
    list[tuple[int, int]],
    tuple[int, int],
]:
    branch = random.choices(["h", "v", "hv"], weights=[3, 3, 4], k=1)[0]

    if branch == "h":
        width = random.randint(8, 15)
        main_h = random.randint(6, 12)
        side_h = random.randint(3, main_h - 1)
        main_row = random.randint(0, 1)
        row_sizes = [(main_h if i == main_row else side_h) for i in range(2)]
        return branch, [(0, row_sizes[0]), (row_sizes[0] + 1, row_sizes[0] + 1 + row_sizes[1])], [(0, width)], (main_row, 0)

    if branch == "v":
        height = random.randint(8, 15)
        main_w = random.randint(6, 12)
        side_w = random.randint(3, main_w - 1)
        main_col = random.randint(0, 1)
        col_sizes = [(main_w if i == main_col else side_w) for i in range(2)]
        return branch, [(0, height)], [(0, col_sizes[0]), (col_sizes[0] + 1, col_sizes[0] + 1 + col_sizes[1])], (0, main_col)

    main_h = random.randint(6, 12)
    main_w = random.randint(6, 12)
    side_h = random.randint(3, main_h)
    side_w = random.randint(3, main_w)
    if side_h == main_h and side_w == main_w:
        side_h -= 1
    main_row = random.randint(0, 1)
    main_col = random.randint(0, 1)
    row_sizes = [(main_h if i == main_row else side_h) for i in range(2)]
    col_sizes = [(main_w if i == main_col else side_w) for i in range(2)]
    return (
        branch,
        [(0, row_sizes[0]), (row_sizes[0] + 1, row_sizes[0] + 1 + row_sizes[1])],
        [(0, col_sizes[0]), (col_sizes[0] + 1, col_sizes[0] + 1 + col_sizes[1])],
        (main_row, main_col),
    )


def _rect_from_segments(
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
    idx: tuple[int, int],
) -> Rect:
    r0, r1 = row_segments[idx[0]]
    c0, c1 = col_segments[idx[1]]
    return r0, r1, c0, c1


def _main_support(h: int, w: int, palette: list[int]) -> list[tuple[int, int, int]]:
    coords = _choose_supported_coords(np.ones((h, w), dtype=bool))
    assert coords is not None
    return [(r, c, random.choice(palette)) for r, c in coords]


def _generate_panel_targets(
    main_input: np.ndarray,
    reserved_visible: np.ndarray,
    coverage: Rect,
    palette: list[int],
) -> list[tuple[int, int, int]] | None:
    r0, r1, c0, c1 = coverage
    allowed = (main_input[r0:r1, c0:c1] == BG) & (~reserved_visible[r0:r1, c0:c1])
    coords = _choose_supported_coords(allowed)
    if coords is None:
        return None
    return [(r0 + r, c0 + c, random.choice(palette)) for r, c in coords]


def _exact_blanks(grid: np.ndarray) -> tuple[list[int], list[int]]:
    return _blank_indices(grid, axis=1), _blank_indices(grid, axis=0)


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    _branch, row_segments, col_segments, main_idx = _build_layout()
    full_h = row_segments[-1][1]
    full_w = col_segments[-1][1]
    if full_h > 30 or full_w > 30:
        return None

    rects = _panel_rects(row_segments, col_segments)
    main_rect = _rect_from_segments(row_segments, col_segments, main_idx)
    mr0, mr1, mc0, mc1 = main_rect
    main_h = mr1 - mr0
    main_w = mc1 - mc0

    palette = random.sample(PALETTE, k=random.randint(4, 7))
    main_input = np.zeros((main_h, main_w), dtype=int)
    expected = np.zeros((main_h, main_w), dtype=int)
    for r, c, color in _main_support(main_h, main_w, palette):
        main_input[r, c] = color
        expected[r, c] = color

    sources = []
    for ri, ci, rect in rects:
        coverage = _coverage_rect(rect, (ri, ci), main_rect, main_idx)
        if coverage is None:
            continue
        sources.append((ri, ci, rect, coverage))
    sources.sort(key=lambda item: _area(item[3]))

    reserved_visible = np.zeros((main_h, main_w), dtype=bool)
    panel_targets: list[dict] = []
    for ri, ci, rect, coverage in sources:
        target_cells = _generate_panel_targets(main_input, reserved_visible, coverage, palette)
        if target_cells is None:
            return None
        for tr, tc, color in target_cells:
            expected[tr, tc] = color
            reserved_visible[tr, tc] = True
        panel_targets.append(
            {
                "idx": (ri, ci),
                "rect": rect,
                "coverage": coverage,
                "cells": target_cells,
            }
        )

    blockable = [panel for panel in panel_targets if len(panel["cells"]) >= 2]
    random.shuffle(blockable)
    block_count = min(len(blockable), random.randint(1, max(1, len(blockable)))) if blockable else 0
    for panel in blockable[:block_count]:
        keep = random.choice(panel["cells"])
        candidates = [cell for cell in panel["cells"] if cell != keep and int(main_input[cell[0], cell[1]]) == BG]
        if not candidates:
            continue
        tr, tc, source_color = random.choice(candidates)
        options = [color for color in palette if color != source_color]
        block_color = random.choice(options) if options else source_color
        main_input[tr, tc] = block_color
        expected[tr, tc] = block_color

    canvas = np.zeros((full_h, full_w), dtype=int)
    canvas[mr0:mr1, mc0:mc1] = main_input

    for panel in panel_targets:
        ri, ci = panel["idx"]
        r0, r1, c0, c1 = panel["rect"]
        coverage = panel["coverage"]
        ph, pw = r1 - r0, c1 - c0
        local_cells: list[tuple[int, int, int]] = []
        for tr, tc, color in panel["cells"]:
            sr, sc = _source_local_from_target(tr, tc, coverage, (ph, pw), (ri, ci), main_idx)
            local_cells.append((sr, sc, color))
        _paint_cells(canvas, r0, c0, local_cells)

    blank_rows, blank_cols = _exact_blanks(canvas)
    intended_blank_rows = sorted({end for _start, end in row_segments[:-1]})
    intended_blank_cols = sorted({end for _start, end in col_segments[:-1]})
    if blank_rows != intended_blank_rows or blank_cols != intended_blank_cols:
        return None

    if np.any(expected == BG):
        nonzero = int(np.count_nonzero(expected))
        if nonzero < max(main_h, main_w):
            return None

    return canvas, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        built = _build_attempt()
        if built is None:
            continue
        grid, expected = built
        try:
            solved = main(grid)
        except AssertionError:
            continue
        if solved.shape != expected.shape:
            continue
        if np.array_equal(solved, expected):
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 7")
