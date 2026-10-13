from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


BG = 0
DIVIDER = 5
BAR_COLORS = (3, 4, 6, 9)
ROW_PALETTE_COLORS = (1, 2, 7, 8)
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _divider_grid(g: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    candidates: list[tuple[int, list[int], list[int]]] = []
    h, w = g.shape
    for color in colors_present(g, exclude={bg}):
        rows = [r for r in range(h) if np.all(g[r, :] == color)]
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1
    return candidates[0]


def _segments(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = 0
    for divider in dividers:
        if start < divider:
            segments.append((start, divider))
        start = divider + 1
    if start < length:
        segments.append((start, length))
    assert segments
    return segments


def _cell_bounds(row: int, col: int, row_segments: list[tuple[int, int]], col_segments: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    r0, r1 = row_segments[row]
    c0, c1 = col_segments[col]
    return r0, r1, c0, c1


def _pick_source_cell(
    g: np.ndarray,
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
    bg: int,
) -> tuple[tuple[int, int], np.ndarray]:
    found: list[tuple[tuple[int, int], np.ndarray]] = []
    for rr in range(len(row_segments)):
        for cc in range(1, len(col_segments)):
            r0, r1, c0, c1 = _cell_bounds(rr, cc, row_segments, col_segments)
            cell = g[r0:r1, c0:c1]
            if np.any(cell != bg):
                found.append(((rr, cc), cell.copy()))
    assert len(found) == 1
    return found[0]


def _extract_bars(cell: np.ndarray, bg: int) -> list[dict[str, int | str]]:
    h, w = cell.shape
    full_rows = [r for r in range(h) if np.all(cell[r, :] != bg)]
    full_cols = [c for c in range(w) if np.all(cell[:, c] != bg)]
    assert full_rows and full_cols
    assert 2 <= len(full_rows) + len(full_cols) <= 3

    bars: list[dict[str, int | str]] = []
    col_set = set(full_cols)
    row_set = set(full_rows)

    for r in full_rows:
        base_colors = {int(cell[r, c]) for c in range(w) if c not in col_set}
        assert len(base_colors) == 1
        bars.append({"id": f"h{r}", "orient": "h", "index": int(r), "color": int(next(iter(base_colors)))})

    for c in full_cols:
        base_colors = {int(cell[r, c]) for r in range(h) if r not in row_set}
        assert len(base_colors) == 1
        bars.append({"id": f"v{c}", "orient": "v", "index": int(c), "color": int(next(iter(base_colors)))})

    colors = [int(bar["color"]) for bar in bars]
    assert len(set(colors)) == len(colors)
    return bars


def _top_bar(cell: np.ndarray, bars: list[dict[str, int | str]]) -> dict[str, int | str]:
    edges: set[tuple[int, int]] = set()
    horizontal = [bar for bar in bars if bar["orient"] == "h"]
    vertical = [bar for bar in bars if bar["orient"] == "v"]
    assert horizontal and vertical

    for hbar in horizontal:
        for vbar in vertical:
            visible = int(cell[int(hbar["index"]), int(vbar["index"])])
            h_color = int(hbar["color"])
            v_color = int(vbar["color"])
            assert visible in {h_color, v_color}
            if visible == h_color:
                edges.add((v_color, h_color))
            else:
                edges.add((h_color, v_color))

    higher: dict[int, set[int]] = {int(bar["color"]): set() for bar in bars}
    for lower, upper in edges:
        higher[lower].add(upper)

    top_colors = [color for color in higher if not higher[color]]
    assert len(top_colors) == 1
    top_color = top_colors[0]
    return next(bar for bar in bars if int(bar["color"]) == top_color)


def _template_mask_from_bar(shape: tuple[int, int], bar: dict[str, int | str]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if bar["orient"] == "h":
        mask[int(bar["index"]), :] = True
    else:
        mask[:, int(bar["index"])] = True
    return mask


def _analyze_input(g: np.ndarray) -> dict[str, object]:
    bg = get_bg(g)
    divider_color, divider_rows, divider_cols = _divider_grid(g, bg)
    row_segments = _segments(g.shape[0], divider_rows)
    col_segments = _segments(g.shape[1], divider_cols)
    assert len(set(r1 - r0 for r0, r1 in row_segments)) == 1
    assert len(set(c1 - c0 for c0, c1 in col_segments)) == 1

    row_colors: list[int] = []
    for rr in range(len(row_segments)):
        r0, r1, c0, c1 = _cell_bounds(rr, 0, row_segments, col_segments)
        cell = g[r0:r1, c0:c1]
        marker_colors = colors_present(cell, exclude={bg})
        assert len(marker_colors) == 1
        row_colors.append(int(marker_colors[0]))

    source_loc, source_cell = _pick_source_cell(g, row_segments, col_segments, bg)
    bars = _extract_bars(source_cell, bg)
    top_bar = _top_bar(source_cell, bars)
    template_mask = _template_mask_from_bar(source_cell.shape, top_bar)
    assert np.all(source_cell[template_mask] == int(top_bar["color"]))

    for rr in range(len(row_segments)):
        for cc in range(1, len(col_segments)):
            if (rr, cc) == source_loc:
                continue
            r0, r1, c0, c1 = _cell_bounds(rr, cc, row_segments, col_segments)
            assert np.all(g[r0:r1, c0:c1] == bg)

    return {
        "bg": int(bg),
        "divider_color": int(divider_color),
        "row_segments": row_segments,
        "col_segments": col_segments,
        "row_colors": row_colors,
        "source_loc": source_loc,
        "source_cell": source_cell,
        "template_mask": template_mask,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the unique top bar inside one crossed-bar cell, then stamp its mask into every empty cell using each row's palette color.

    Input structure:
    - Black background fills a divider grid of equal-sized rectangular cells, with one full-row/full-column divider color separating row bands and column bands.
    - In the first column of cells, each row band contains one single-color marker pixel that provides that row's palette color.
    - Exactly one non-first-column cell contains 2-3 distinct full-span bars: each bar is a single full row or full column inside that cell, and perpendicular crossings show only the higher-layer bar color.

    Rule:
    - Detect the divider grid and read the row palette colors from the first-column marker cells.
    - In the unique crossed-bar cell, identify each full-row and full-column bar, then use every horizontal/vertical crossing to add a lower-to-higher layer constraint toward the visible crossing color.
    - Find the unique bar with nothing above it, take its full row-or-column mask as the template, and stamp that mask into every other non-first-column cell using the corresponding row palette color.
    - Keep the original crossed-bar cell, divider bars, and first-column marker pixels unchanged.

    Key insight:
    - The template is not chosen by size or frequency; the visible colors at bar crossings encode a partial layer order, and only the unique maximal bar becomes the stamp mask.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_input(g)
    bg = int(info["bg"])
    row_segments = info["row_segments"]
    col_segments = info["col_segments"]
    row_colors = info["row_colors"]
    source_loc = info["source_loc"]
    template_mask = info["template_mask"]

    out = g.copy()
    for rr, color in enumerate(row_colors):
        for cc in range(1, len(col_segments)):
            if (rr, cc) == source_loc:
                continue
            r0, r1, c0, c1 = _cell_bounds(rr, cc, row_segments, col_segments)
            out[r0:r1, c0:c1] = bg
            out[r0:r1, c0:c1][template_mask] = int(color)
    return out


def _grid_shape(row_count: int, col_count: int, cell_h: int, cell_w: int) -> tuple[int, int]:
    return row_count * cell_h + (row_count - 1), col_count * cell_w + (col_count - 1)


def _cell_origin(row: int, col: int, cell_h: int, cell_w: int) -> tuple[int, int]:
    return row * (cell_h + 1), col * (cell_w + 1)


def _empty_grid(row_count: int, col_count: int, cell_h: int, cell_w: int) -> np.ndarray:
    h, w = _grid_shape(row_count, col_count, cell_h, cell_w)
    grid = np.full((h, w), BG, dtype=int)
    for r in range(cell_h, h, cell_h + 1):
        grid[r, :] = DIVIDER
    for c in range(cell_w, w, cell_w + 1):
        grid[:, c] = DIVIDER
    return grid


def _pick_index_set(length: int, count: int) -> list[int]:
    indices = list(range(length))
    if count == 1:
        return [random.choice(indices)]
    min_gap = 2 if length >= 5 else 1
    candidates = [combo for combo in itertools.combinations(indices, count) if all(b - a >= min_gap for a, b in zip(combo, combo[1:]))]
    if not candidates:
        candidates = list(itertools.combinations(indices, count))
    return list(random.choice(candidates))


def _choose_geometry(top_orient: str) -> tuple[int, int]:
    if top_orient == "h":
        options = [(5, 4), (5, 5), (4, 4), (4, 5)]
    else:
        options = [(4, 5), (5, 5), (4, 4), (5, 4)]
    return random.choice(options)


def _build_source_cell(mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "two":
        top_orient = random.choice(("h", "v"))
        h_count, v_count = 1, 1
    elif mode == "single_top":
        top_orient = random.choice(("h", "v"))
        h_count, v_count = (1, 2) if top_orient == "h" else (2, 1)
    else:
        top_orient = random.choice(("h", "v"))
        h_count, v_count = (2, 1) if top_orient == "h" else (1, 2)

    cell_h, cell_w = _choose_geometry(top_orient)
    rows = _pick_index_set(cell_h, h_count)
    cols = _pick_index_set(cell_w, v_count)
    bar_count = h_count + v_count
    colors = random.sample(BAR_COLORS, bar_count)

    horizontal_bars = [{"orient": "h", "index": row, "color": colors.pop()} for row in rows]
    vertical_bars = [{"orient": "v", "index": col, "color": colors.pop()} for col in cols]

    if mode == "two":
        if top_orient == "h":
            top_bar = horizontal_bars[0]
            order = [vertical_bars[0], horizontal_bars[0]]
        else:
            top_bar = vertical_bars[0]
            order = [horizontal_bars[0], vertical_bars[0]]
    elif mode == "single_top":
        if top_orient == "h":
            top_bar = horizontal_bars[0]
            lower = vertical_bars[:]
            random.shuffle(lower)
            order = lower + [horizontal_bars[0]]
        else:
            top_bar = vertical_bars[0]
            lower = horizontal_bars[:]
            random.shuffle(lower)
            order = lower + [vertical_bars[0]]
    else:
        if top_orient == "h":
            top_bar = random.choice(horizontal_bars)
            bottom_bar = next(bar for bar in horizontal_bars if bar is not top_bar)
            order = [bottom_bar, vertical_bars[0], top_bar]
        else:
            top_bar = random.choice(vertical_bars)
            bottom_bar = next(bar for bar in vertical_bars if bar is not top_bar)
            order = [bottom_bar, horizontal_bars[0], top_bar]

    cell = np.full((cell_h, cell_w), BG, dtype=int)
    for bar in order:
        if bar["orient"] == "h":
            cell[int(bar["index"]), :] = int(bar["color"])
        else:
            cell[:, int(bar["index"])] = int(bar["color"])

    template_mask = _template_mask_from_bar((cell_h, cell_w), top_bar)
    return cell, template_mask


def _marker_positions(cell_h: int, cell_w: int, row_count: int) -> list[tuple[int, int]]:
    all_positions = [(r, c) for r in range(cell_h) for c in range(cell_w)]
    return random.sample(all_positions, row_count)


def _expected_output(inp: np.ndarray) -> np.ndarray:
    return main(inp)


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    info = _analyze_input(inp)
    row_segments = info["row_segments"]
    col_segments = info["col_segments"]
    row_colors = info["row_colors"]
    source_loc = info["source_loc"]
    template_mask = info["template_mask"]

    if len(set(row_colors)) != len(row_colors):
        return False

    source_cell = info["source_cell"]
    bars = _extract_bars(source_cell, int(info["bg"]))
    if len(bars) not in {2, 3}:
        return False
    top_bar = _top_bar(source_cell, bars)
    if not np.array_equal(template_mask, _template_mask_from_bar(source_cell.shape, top_bar)):
        return False

    stamped_cells = 0
    for rr, color in enumerate(row_colors):
        for cc in range(1, len(col_segments)):
            r0, r1, c0, c1 = _cell_bounds(rr, cc, row_segments, col_segments)
            cell = expected[r0:r1, c0:c1]
            if (rr, cc) == source_loc:
                if not np.array_equal(cell, source_cell):
                    return False
                continue
            stamped_cells += 1
            painted = np.zeros_like(cell)
            painted[template_mask] = color
            if not np.array_equal(cell, painted):
                return False

    return stamped_cells >= 2 and not np.array_equal(inp, expected)


def _gen_attempt() -> np.ndarray | None:
    row_count = random.randint(3, 4)
    col_count = random.randint(3, 5)
    mode = random.choice(("two", "single_top", "same_orient_top"))
    source_cell, template_mask = _build_source_cell(mode)
    cell_h, cell_w = source_cell.shape

    if max(_grid_shape(row_count, col_count, cell_h, cell_w)) > 30:
        return None
    if row_count > len(ROW_PALETTE_COLORS):
        return None

    grid = _empty_grid(row_count, col_count, cell_h, cell_w)
    row_colors = random.sample(ROW_PALETTE_COLORS, row_count)
    marker_positions = _marker_positions(cell_h, cell_w, row_count)
    for rr, color in enumerate(row_colors):
        r0, c0 = _cell_origin(rr, 0, cell_h, cell_w)
        mr, mc = marker_positions[rr]
        grid[r0 + mr, c0 + mc] = int(color)

    source_row = random.randrange(row_count)
    source_col = random.randrange(1, col_count)
    r0, c0 = _cell_origin(source_row, source_col, cell_h, cell_w)
    grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = source_cell

    expected = _expected_output(grid)

    if expected.shape != grid.shape:
        return None
    if not _semantic_checks(grid, expected):
        return None

    other_row = (source_row + 1) % row_count
    other_col = 1 if source_col != 1 else min(2, col_count - 1)
    rr0, cc0 = _cell_origin(other_row, other_col, cell_h, cell_w)
    probe = expected[rr0 : rr0 + cell_h, cc0 : cc0 + cell_w]
    if probe[template_mask].size != int(template_mask.sum()):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
