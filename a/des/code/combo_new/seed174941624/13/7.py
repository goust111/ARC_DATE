from __future__ import annotations

import random

import numpy as np


BG = 0
MAROON = 9
PALETTE = tuple(range(1, 9))
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_mapping_table(
    g: np.ndarray,
) -> tuple[tuple[int, int, int, int], dict[int, int]]:
    mask = (g != BG) & (g != MAROON)
    candidates: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for coords in _components(mask):
        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height != 2:
            continue
        if coords.shape[0] != 2 * width:
            continue
        if not (r0 == 0 or r1 == g.shape[0] - 1 or c0 == 0 or c1 == g.shape[1] - 1):
            continue
        block = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(block == BG) or np.any(block == MAROON):
            continue
        candidates.append(((r0, c0, r1, c1), block.copy()))

    assert len(candidates) == 1, f"expected one 2xN edge mapping table, got {len(candidates)}"
    bbox, block = candidates[0]
    mapping: dict[int, int] = {}
    for col in range(block.shape[1]):
        src = int(block[0, col])
        dst = int(block[1, col])
        assert src not in mapping, "lens colors must be unique in the mapping table"
        mapping[src] = dst
    return bbox, mapping


def _find_lenses(g: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
    h, w = g.shape
    row_lenses: dict[int, int] = {}
    col_lenses: dict[int, int] = {}

    for r in range(h):
        row = g[r, :]
        if len(set(map(int, row))) != 1:
            continue
        color = int(row[0])
        if color not in (BG, MAROON):
            row_lenses[r] = color

    for c in range(w):
        col = g[:, c]
        if len(set(map(int, col))) != 1:
            continue
        color = int(col[0])
        if color not in (BG, MAROON):
            col_lenses[c] = color

    if row_lenses and col_lenses:
        for r, row_color in row_lenses.items():
            for c, col_color in col_lenses.items():
                assert row_color == col_color == int(g[r, c]), "mixed lens orientations must agree at intersections"

    assert row_lenses or col_lenses, "expected at least one lens bar"
    return row_lenses, col_lenses


def _find_sources(
    g: np.ndarray,
    table_bbox: tuple[int, int, int, int],
    row_lenses: dict[int, int],
    col_lenses: dict[int, int],
) -> list[tuple[int, int, int, tuple[int, int]]]:
    h, w = g.shape
    r0, c0, r1, c1 = table_bbox
    table_mask = np.zeros_like(g, dtype=bool)
    table_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    sources: list[tuple[int, int, int, tuple[int, int]]] = []
    used_indicators: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (BG, MAROON):
                continue
            if table_mask[r, c]:
                continue
            if r in row_lenses or c in col_lenses:
                continue

            maroons: list[tuple[int, int]] = []
            for dr, dc in DIRS:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == MAROON:
                    maroons.append((nr, nc))

            assert len(maroons) == 1, f"every source must touch exactly one maroon indicator, got {len(maroons)} at {(r, c)}"
            ir, ic = maroons[0]
            direction = (r - ir, c - ic)
            assert direction in DIRS
            sources.append((r, c, color, direction))
            used_indicators.add((ir, ic))

    maroon_cells = {(int(r), int(c)) for r, c in np.argwhere(g == MAROON)}
    assert maroon_cells == used_indicators, "all maroon pixels must serve as source indicators"
    assert sources, "expected at least one signal source"
    return sorted(sources)


def _paint_signal(
    out: np.ndarray,
    g: np.ndarray,
    source: tuple[int, int, int, tuple[int, int]],
    mapping: dict[int, int],
    row_lenses: dict[int, int],
    col_lenses: dict[int, int],
) -> None:
    h, w = g.shape
    r, c, current, direction = source
    dr, dc = direction
    while 0 <= r < h and 0 <= c < w:
        out[r, c] = current

        lens_colors: list[int] = []
        if r in row_lenses:
            lens_colors.append(int(row_lenses[r]))
        if c in col_lenses:
            lens_colors.append(int(col_lenses[c]))
        if lens_colors:
            lens_color = lens_colors[0]
            assert all(color == lens_color for color in lens_colors)
            assert lens_color in mapping, f"missing mapping for lens color {lens_color}"
            current = int(mapping[lens_color])

        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each colored signal source in the direction opposite its maroon marker, recoloring the ray after every full-row or full-column lens it crosses.

    Input structure:
    - The background is black.
    - A small 2-row edge table maps lens colors in its top row to replacement output colors in its bottom row.
    - Each source is a single non-black pixel touching exactly one maroon indicator pixel, and each lens bar is a single-color line spanning a whole row or a whole column.

    Rule:
    - Read the 2xN edge mapping table as lens-color -> next-signal-color.
    - For each source, launch a ray away from the adjacent maroon indicator and paint every traversed cell up to the grid edge.
    - Paint a lens cell with the current signal color, then update the signal color according to that lens's table entry for the cells after the crossing.
    - Erase all maroon indicator pixels to black; the mapping table itself is otherwise unchanged unless a ray would directly pass through it.

    Key insight:
    - The puzzle combines three independent deductions: reverse the marker to recover the launch direction, detect the edge table as a color-substitution lookup, and apply those substitutions in the exact order that each ray crosses the lens bars.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    table_bbox, mapping = _find_mapping_table(g)
    row_lenses, col_lenses = _find_lenses(g)
    lens_colors = set(row_lenses.values()) | set(col_lenses.values())
    assert lens_colors <= set(mapping), "every lens color must appear in the mapping table"
    sources = _find_sources(g, table_bbox, row_lenses, col_lenses)

    out = g.copy()
    out[g == MAROON] = BG
    for source in sources:
        _paint_signal(out, g, source, mapping, row_lenses, col_lenses)
    return out


def _sample_spaced_positions(pool: list[int], count: int, gap: int = 1) -> list[int] | None:
    if len(pool) < count:
        return None
    for _ in range(300):
        picks = sorted(random.sample(pool, count))
        if all(b - a > gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _cross_count(index: int, direction: int, lens_positions: list[int]) -> int:
    if direction < 0:
        return sum(pos < index for pos in lens_positions)
    return sum(pos > index for pos in lens_positions)


def _sample_source_color(lens_colors: list[int]) -> int:
    non_lens = [color for color in PALETTE if color not in lens_colors]
    if non_lens and random.random() < 0.5:
        return random.choice(non_lens)
    return random.choice(lens_colors if random.random() < 0.5 else list(PALETTE))


def _place_table(
    grid: np.ndarray,
    side: str,
    offset: int,
    inputs: list[int],
    outputs: list[int],
) -> tuple[int, int, int, int]:
    width = len(inputs)
    h, w = grid.shape
    if side == "left":
        grid[offset : offset + 2, 0:width] = np.array([inputs, outputs], dtype=int)
        return offset, 0, offset + 1, width - 1
    if side == "right":
        grid[offset : offset + 2, w - width : w] = np.array([inputs, outputs], dtype=int)
        return offset, w - width, offset + 1, w - 1
    if side == "top":
        grid[0:2, offset : offset + width] = np.array([inputs, outputs], dtype=int)
        return 0, offset, 1, offset + width - 1
    if side == "bottom":
        grid[h - 2 : h, offset : offset + width] = np.array([inputs, outputs], dtype=int)
        return h - 2, offset, h - 1, offset + width - 1
    raise AssertionError(f"unknown side {side}")


def _gen_vertical_mode() -> np.ndarray | None:
    h = random.randint(14, 26)
    w = random.randint(14, 26)
    lens_count = random.randint(2, min(4, (h - 6) // 2))
    side = random.choice(["left", "right"])

    table_row0 = random.randint(1, h - 3)
    reserved_rows = {row for row in (table_row0 - 1, table_row0, table_row0 + 1, table_row0 + 2) if 0 <= row < h}
    lens_pool = [r for r in range(1, h - 1) if r not in reserved_rows]
    lens_rows = _sample_spaced_positions(lens_pool, lens_count, gap=1)
    if lens_rows is None:
        return None

    table_width = lens_count
    if side == "left":
        source_col_pool = list(range(table_width + 2, w - 1))
    else:
        source_col_pool = list(range(1, w - table_width - 2))
    max_sources = min(4, max(2, len(source_col_pool) // 2))
    source_count = random.randint(2, max_sources)
    source_cols = _sample_spaced_positions(source_col_pool, source_count, gap=1)
    if source_cols is None:
        return None

    lens_colors = random.sample(PALETTE, lens_count)
    table_inputs = random.sample(lens_colors, len(lens_colors))
    table_outputs = [random.choice([color for color in PALETTE if color != src]) for src in table_inputs]
    input_to_output = {src: dst for src, dst in zip(table_inputs, table_outputs)}

    directions = []
    if source_count >= 2:
        directions.extend(random.sample([-1, 1], 2))
    while len(directions) < source_count:
        directions.append(random.choice([-1, 1]))

    sources: list[tuple[int, int, int, int]] = []
    for i, col in enumerate(source_cols):
        direction = directions[i]
        min_cross = 2 if i == 0 and lens_count >= 2 else 1
        candidates = [
            row
            for row in range(1, h - 1)
            if row not in lens_rows
            and (row - direction) not in lens_rows
            and 0 <= row - direction < h
            and _cross_count(row, direction, lens_rows) >= min_cross
        ]
        if not candidates:
            return None
        row = random.choice(candidates)
        color = _sample_source_color(lens_colors)
        sources.append((row, col, direction, color))

    grid = np.full((h, w), BG, dtype=int)
    for row, color in zip(lens_rows, lens_colors):
        grid[row, :] = color

    _place_table(grid, side, table_row0, table_inputs, table_outputs)

    for row, col, direction, color in sources:
        grid[row, col] = color
        grid[row - direction, col] = MAROON

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    assert any(input_to_output[color] != color for color in table_inputs)
    return grid


def _gen_horizontal_mode() -> np.ndarray | None:
    h = random.randint(14, 26)
    w = random.randint(14, 26)
    lens_count = random.randint(2, min(4, (w - 6) // 2))
    side = random.choice(["top", "bottom"])

    table_col0 = random.randint(1, w - lens_count - 1)
    reserved_cols = {
        col
        for col in range(table_col0 - 1, table_col0 + lens_count + 1)
        if 0 <= col < w
    }
    lens_pool = [c for c in range(1, w - 1) if c not in reserved_cols]
    lens_cols = _sample_spaced_positions(lens_pool, lens_count, gap=1)
    if lens_cols is None:
        return None

    source_row_pool = list(range(3, h - 3))
    max_sources = min(4, max(2, len(source_row_pool) // 2))
    source_count = random.randint(2, max_sources)
    source_rows = _sample_spaced_positions(source_row_pool, source_count, gap=1)
    if source_rows is None:
        return None

    lens_colors = random.sample(PALETTE, lens_count)
    table_inputs = random.sample(lens_colors, len(lens_colors))
    table_outputs = [random.choice([color for color in PALETTE if color != src]) for src in table_inputs]
    input_to_output = {src: dst for src, dst in zip(table_inputs, table_outputs)}

    directions = []
    if source_count >= 2:
        directions.extend(random.sample([-1, 1], 2))
    while len(directions) < source_count:
        directions.append(random.choice([-1, 1]))

    sources: list[tuple[int, int, int, int]] = []
    for i, row in enumerate(source_rows):
        direction = directions[i]
        min_cross = 2 if i == 0 and lens_count >= 2 else 1
        candidates = [
            col
            for col in range(1, w - 1)
            if col not in lens_cols
            and (col - direction) not in lens_cols
            and 0 <= col - direction < w
            and _cross_count(col, direction, lens_cols) >= min_cross
        ]
        if not candidates:
            return None
        col = random.choice(candidates)
        color = _sample_source_color(lens_colors)
        sources.append((row, col, direction, color))

    grid = np.full((h, w), BG, dtype=int)
    for col, color in zip(lens_cols, lens_colors):
        grid[:, col] = color

    row_offset = 0 if side == "top" else h - 2
    _place_table(grid, side, table_col0, table_inputs, table_outputs)
    if side == "top":
        assert row_offset == 0
    else:
        assert row_offset == h - 2

    for row, col, direction, color in sources:
        grid[row, col] = color
        grid[row, col - direction] = MAROON

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    assert any(input_to_output[color] != color for color in table_inputs)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_vertical_mode() if random.random() < 0.5 else _gen_horizontal_mode()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
