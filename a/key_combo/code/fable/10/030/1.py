from __future__ import annotations

import random

import numpy as np


Line = dict[str, int | str]


ORIENT_H = "H"
ORIENT_V = "V"
ALL_COLORS = tuple(range(10))

GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (12, 22)
PERIOD_H_RANGE = (2, 4)
PERIOD_W_RANGE = (2, 5)
H_LINE_COUNT_RANGE = (2, 3)
V_LINE_COUNT_RANGE = (2, 3)
BG_COLOR_COUNT_RANGE = (2, 4)
GEN_ATTEMPTS = 500


def _tile_grid(template: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rr = np.arange(shape[0]) % template.shape[0]
    cc = np.arange(shape[1]) % template.shape[1]
    return template[np.ix_(rr, cc)]


def _line_full_mask(shape: tuple[int, int], line: Line) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if line["orient"] == ORIENT_H:
        mask[int(line["index"]), :] = True
    else:
        mask[:, int(line["index"])] = True
    return mask


def _cross_cell(line_a: Line, line_b: Line) -> tuple[int, int]:
    if line_a["orient"] == ORIENT_H:
        return int(line_a["index"]), int(line_b["index"])
    return int(line_b["index"]), int(line_a["index"])


def _infer_lines(grid: np.ndarray) -> dict[int, Line]:
    g = np.asarray(grid, dtype=int)
    lines: dict[int, Line] = {}
    used_h: set[int] = set()
    used_v: set[int] = set()

    for color in sorted(set(map(int, g.flat))):
        coords = np.argwhere(g == color)
        if coords.shape[0] < 2:
            continue
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if len(rows) == 1 and len(cols) >= 2:
            row = int(rows[0])
            assert row not in used_h, "two horizontal lines cannot share a row"
            used_h.add(row)
            lines[int(color)] = {"orient": ORIENT_H, "index": row}
        elif len(cols) == 1 and len(rows) >= 2:
            col = int(cols[0])
            assert col not in used_v, "two vertical lines cannot share a column"
            used_v.add(col)
            lines[int(color)] = {"orient": ORIENT_V, "index": col}

    assert any(line["orient"] == ORIENT_H for line in lines.values()), "expected a horizontal line"
    assert any(line["orient"] == ORIENT_V for line in lines.values()), "expected a vertical line"
    return lines


def _crossing_info(grid: np.ndarray, lines: dict[int, Line]) -> tuple[dict[int, list[tuple[int, int]]], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    colors = list(lines)
    crossings: dict[int, list[tuple[int, int]]] = {color: [] for color in colors}
    below_to_above: set[tuple[int, int]] = set()

    for i, color_a in enumerate(colors):
        line_a = lines[color_a]
        for color_b in colors[i + 1 :]:
            line_b = lines[color_b]
            if line_a["orient"] == line_b["orient"]:
                continue
            r, c = _cross_cell(line_a, line_b)
            visible = int(g[r, c])
            assert visible in {color_a, color_b}, "crossing must show one of the two line colors"
            crossings[color_a].append((r, c))
            crossings[color_b].append((r, c))
            if visible == color_a:
                below_to_above.add((color_b, color_a))
            else:
                below_to_above.add((color_a, color_b))

    for color, cells in crossings.items():
        assert cells, f"line {color} must cross at least one perpendicular line"
    return crossings, below_to_above


def _find_bottom_line(grid: np.ndarray, lines: dict[int, Line]) -> int:
    g = np.asarray(grid, dtype=int)
    crossings, _edges = _crossing_info(g, lines)
    bottoms = [color for color, cells in crossings.items() if all(int(g[r, c]) != color for r, c in cells)]
    assert len(bottoms) == 1, "expected a unique fully-hidden bottom line"
    return bottoms[0]


def _line_union_mask(shape: tuple[int, int], lines: dict[int, Line]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for line in lines.values():
        mask |= _line_full_mask(shape, line)
    return mask


def _infer_periodic_template(grid: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    observed = np.argwhere(observed_mask)
    assert observed.size > 0, "need visible background cells to infer the lattice"
    candidates: list[tuple[int, int, int, np.ndarray]] = []

    for ph in range(1, g.shape[0] + 1):
        for pw in range(1, g.shape[1] + 1):
            template = np.full((ph, pw), -1, dtype=int)
            ok = True
            for r, c in observed:
                rr, cc = int(r % ph), int(c % pw)
                value = int(g[r, c])
                if template[rr, cc] == -1:
                    template[rr, cc] = value
                elif template[rr, cc] != value:
                    ok = False
                    break
            if ok and np.all(template != -1):
                candidates.append((ph * pw, ph, pw, template))

    assert candidates, "no periodic lattice fits the visible background"
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    best_area = candidates[0][0]
    best = [item for item in candidates if item[0] == best_area]
    assert len(best) == 1, "expected a unique smallest periodic template"
    return best[0][3]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete the unique fully-hidden bottom line and restore its non-crossing cells from the periodic background.

    Input structure:
    - The whole grid is a regular two-dimensional periodic color lattice whose palette does not include any line color.
    - Several one-cell-thick colored lines span an entire row or an entire column, and perpendicular lines may cross.
    - At each crossing, only the top line's color is visible, so lower lines may be partially hidden.

    Rule:
    - Reconstruct every full horizontal or vertical line from its visible same-color collinear pixels.
    - Use each crossing cell to read a layer constraint: the visible line color is above the other crossing line, then find the unique line that is invisible at all of its own crossings.
    - Remove that bottom line only: keep crossing cells unchanged, but replace its other cells with the color obtained by inferring and tiling the unique smallest periodic background from the uncovered cells.

    Key insight:
    - The target is not the most obscured visible fragment by itself; the crossings encode which line is always underneath, and the missing background must be recovered from the still-visible residues of the hidden lattice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    lines = _infer_lines(g)
    bottom_color = _find_bottom_line(g, lines)
    bottom_line = lines[bottom_color]

    all_line_mask = _line_union_mask(g.shape, lines)
    template = _infer_periodic_template(g, ~all_line_mask)
    restored = _tile_grid(template, g.shape)

    out = g.copy()
    bottom_mask = _line_full_mask(g.shape, bottom_line)
    crossing_mask = np.zeros_like(bottom_mask)
    for color, line in lines.items():
        if color == bottom_color or line["orient"] == bottom_line["orient"]:
            continue
        r, c = _cross_cell(bottom_line, line)
        crossing_mask[r, c] = True

    restore_mask = bottom_mask & (~crossing_mask)
    out[restore_mask] = restored[restore_mask]
    return out


def _random_tile(ph: int, pw: int, palette: list[int]) -> np.ndarray | None:
    for _ in range(200):
        tile = np.array([[random.choice(palette) for _ in range(pw)] for _ in range(ph)], dtype=int)
        if len(set(map(int, tile.flat))) < 2:
            continue
        if len({tuple(map(int, row)) for row in tile}) < 2:
            continue
        if len({tuple(map(int, tile[:, c])) for c in range(pw)}) < 2:
            continue
        primitive = True
        for sh in range(1, ph + 1):
            for sw in range(1, pw + 1):
                if sh == ph and sw == pw:
                    continue
                if sh * sw >= ph * pw:
                    continue
                if np.array_equal(tile, _tile_grid(tile[:sh, :sw], tile.shape)):
                    primitive = False
                    break
            if not primitive:
                break
        if primitive:
            return tile
    return None


def _sample_positions(length: int, count: int, modulus: int) -> list[int] | None:
    for _ in range(400):
        positions = sorted(random.sample(range(length), count))
        if any(b - a < 2 for a, b in zip(positions, positions[1:])):
            continue
        if not all(any((idx not in positions) and idx % modulus == residue for idx in range(length)) for residue in range(modulus)):
            continue
        return positions
    return None


def _paint_line(grid: np.ndarray, line: Line) -> None:
    color = int(line["color"])
    if line["orient"] == ORIENT_H:
        grid[int(line["index"]), :] = color
    else:
        grid[:, int(line["index"])] = color


def _render_instance(shape: tuple[int, int], tile: np.ndarray, order: list[Line], skip_color: int | None = None) -> np.ndarray:
    grid = _tile_grid(tile, shape)
    for line in order:
        if skip_color is not None and int(line["color"]) == skip_color:
            continue
        _paint_line(grid, line)
    return grid


def _expected_restore_mask(lines: dict[int, Line], bottom_color: int, shape: tuple[int, int]) -> np.ndarray:
    bottom_line = lines[bottom_color]
    mask = _line_full_mask(shape, bottom_line)
    crossings = np.zeros(shape, dtype=bool)
    for color, line in lines.items():
        if color == bottom_color or line["orient"] == bottom_line["orient"]:
            continue
        r, c = _cross_cell(bottom_line, line)
        crossings[r, c] = True
    return mask & (~crossings)


def _visible_counts(grid: np.ndarray, line_colors: set[int]) -> dict[int, int]:
    return {color: int(np.count_nonzero(grid == color)) for color in line_colors}


def _validate_instance(inp: np.ndarray, expected: np.ndarray, line_specs: list[Line], bottom_color: int) -> bool:
    try:
        inferred = _infer_lines(inp)
    except AssertionError:
        return False

    expected_lines = {int(line["color"]): {"orient": line["orient"], "index": int(line["index"])} for line in line_specs}
    if inferred != expected_lines:
        return False

    try:
        inferred_bottom = _find_bottom_line(inp, inferred)
    except AssertionError:
        return False
    if inferred_bottom != bottom_color:
        return False

    restore_mask = _expected_restore_mask(inferred, bottom_color, inp.shape)
    if np.count_nonzero(restore_mask) == 0:
        return False
    if not np.array_equal(inp[~restore_mask], expected[~restore_mask]):
        return False
    if np.array_equal(inp, expected):
        return False

    union_mask = _line_union_mask(inp.shape, inferred)
    try:
        template = _infer_periodic_template(inp, ~union_mask)
    except AssertionError:
        return False
    background = _tile_grid(template, inp.shape)
    if not np.array_equal(background[~union_mask], inp[~union_mask]):
        return False

    counts = _visible_counts(inp, set(expected_lines))
    if all(color == bottom_color or counts[color] > counts[bottom_color] for color in counts):
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _generate_attempt() -> np.ndarray | None:
    ph = random.randint(*PERIOD_H_RANGE)
    pw = random.randint(*PERIOD_W_RANGE)
    h_count = random.randint(*H_LINE_COUNT_RANGE)
    v_count = random.randint(*V_LINE_COUNT_RANGE)
    total_lines = h_count + v_count

    max_bg_colors = min(BG_COLOR_COUNT_RANGE[1], len(ALL_COLORS) - total_lines)
    if max_bg_colors < BG_COLOR_COUNT_RANGE[0]:
        return None
    bg_color_count = random.randint(BG_COLOR_COUNT_RANGE[0], max_bg_colors)

    colors = random.sample(ALL_COLORS, total_lines + bg_color_count)
    bg_palette = colors[:bg_color_count]
    line_palette = colors[bg_color_count:]

    tile = _random_tile(ph, pw, bg_palette)
    if tile is None:
        return None

    min_h = max(GRID_H_RANGE[0], ph * 3, h_count * 2 + 3)
    min_w = max(GRID_W_RANGE[0], pw * 3, v_count * 2 + 3)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    height = random.randint(min_h, GRID_H_RANGE[1])
    width = random.randint(min_w, GRID_W_RANGE[1])
    rows = _sample_positions(height, h_count, ph)
    cols = _sample_positions(width, v_count, pw)
    if rows is None or cols is None:
        return None

    horizontals: list[Line] = []
    verticals: list[Line] = []
    for row, color in zip(rows, line_palette[:h_count]):
        horizontals.append({"orient": ORIENT_H, "index": row, "color": color})
    for col, color in zip(cols, line_palette[h_count:]):
        verticals.append({"orient": ORIENT_V, "index": col, "color": color})

    bottom_group = horizontals if random.random() < 0.5 else verticals
    witness_group = verticals if bottom_group is horizontals else horizontals
    bottom = random.choice(bottom_group)
    witness = random.choice(witness_group)
    others = [line for line in horizontals + verticals if line is not bottom and line is not witness]
    random.shuffle(others)
    order = [bottom, witness] + others

    inp = _render_instance((height, width), tile, order)
    expected = _render_instance((height, width), tile, order, skip_color=int(bottom["color"]))
    if not _validate_instance(inp, expected, horizontals + verticals, int(bottom["color"])):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
