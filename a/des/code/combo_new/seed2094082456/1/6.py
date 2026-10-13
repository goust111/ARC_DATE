import random

import numpy as np


BG = 0
GREY = 5
LINE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
MIN_BAND_SIZE = 5
MAX_PATTERN_SIZE = 6


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for cut in separators:
        if start < cut:
            bands.append((start, cut - 1))
        start = cut + 1
    if start < size:
        bands.append((start, size - 1))
    return bands


def _separator_positions(band_sizes: list[int]) -> list[int]:
    positions: list[int] = []
    cursor = band_sizes[0]
    for size in band_sizes[1:]:
        positions.append(cursor)
        cursor += 1 + size
    return positions


def _mode_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values.astype(int), return_counts=True)
    return int(colors[np.argmax(counts)])


def _transform_mask(mask: np.ndarray, do_h: bool, do_v: bool) -> np.ndarray:
    out = mask
    if do_v:
        out = np.flipud(out)
    if do_h:
        out = np.fliplr(out)
    return out


def _find_instruction_block(
    grid: np.ndarray, line_by_color: dict[int, tuple[str, int]]
) -> tuple[str, int, int]:
    h, w = grid.shape
    corners = {
        "tl": (slice(0, 2), slice(0, 2)),
        "tr": (slice(0, 2), slice(w - 2, w)),
        "bl": (slice(h - 2, h), slice(0, 2)),
        "br": (slice(h - 2, h), slice(w - 2, w)),
    }
    candidates: list[tuple[str, int, int]] = []
    for name, (rs, cs) in corners.items():
        block = grid[rs, cs]
        if block.shape != (2, 2):
            continue
        a = int(block[0, 0])
        b = int(block[1, 0])
        if a != int(block[0, 1]) or b != int(block[1, 1]) or a == b:
            continue
        if a not in line_by_color or b not in line_by_color:
            continue
        candidates.append((name, a, b))
    assert len(candidates) == 1, "expected exactly one corner instruction block"
    return candidates[0]


def _region_active(line: tuple[str, int], row_band: int, col_band: int) -> bool:
    orient, cut_index = line
    if orient == "row":
        return row_band in {cut_index, cut_index + 1}
    return col_band in {cut_index, cut_index + 1}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the grey scatter in only those subregions touching the two separator lines named by the corner instruction block.

    Input structure:
    - The background is black.
    - One-cell-thick colored separator lines run horizontally and/or vertically across the whole grid and partition it into rectangular subregions.
    - Each separator line has its own color, while the subregions contain sparse grey scatter pixels on the black background.
    - One grid corner contains a 2x2 instruction block whose top row is one separator color and whose bottom row is another separator color.

    Rule:
    - Decode the two instruction colors and match each one to its separator line.
    - The top instruction color activates a horizontal mirror: every subregion sharing an edge with that line mirrors its grey scatter left-right inside its own rectangle.
    - The bottom instruction color activates a vertical mirror: every subregion sharing an edge with that line mirrors its grey scatter up-down inside its own rectangle.
    - A subregion touching both activated lines applies both mirrors, while all other subregions stay unchanged.
    - Separator lines and the instruction block remain unchanged.

    Key insight:
    - The corner block does not add new geometry; it selects which existing separator lines matter and which mirror direction each selected line triggers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    sep_rows = [r for r in range(h) if np.all((g[r, :] != BG) & (g[r, :] != GREY))]
    sep_cols = [c for c in range(w) if np.all((g[:, c] != BG) & (g[:, c] != GREY))]
    assert len(sep_rows) + len(sep_cols) >= 2, "need at least two separator lines"

    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    assert row_bands and col_bands

    line_by_color: dict[int, tuple[str, int]] = {}
    for idx, r in enumerate(sep_rows):
        color = _mode_color(g[r, :])
        assert color not in {BG, GREY}
        assert color not in line_by_color, "separator colors must be unique"
        line_by_color[color] = ("row", idx)
    for idx, c in enumerate(sep_cols):
        color = _mode_color(g[:, c])
        assert color not in {BG, GREY}
        assert color not in line_by_color, "separator colors must be unique"
        line_by_color[color] = ("col", idx)

    _corner_name, a_color, b_color = _find_instruction_block(g, line_by_color)
    a_line = line_by_color[a_color]
    b_line = line_by_color[b_color]

    out = g.copy()
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            do_h = _region_active(a_line, ri, ci)
            do_v = _region_active(b_line, ri, ci)
            if not (do_h or do_v):
                continue

            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            grey_mask = patch == GREY
            if not bool(grey_mask.any()):
                continue

            moved = _transform_mask(grey_mask, do_h=do_h, do_v=do_v)
            base = patch.copy()
            base[grey_mask] = BG
            base[moved & (base == BG)] = GREY
            out[r0 : r1 + 1, c0 : c1 + 1] = base

    return out


def _sample_band_sizes(line_count: int) -> list[int]:
    band_count = line_count + 1
    sizes = [MIN_BAND_SIZE] * band_count
    budget = 30 - line_count - sum(sizes)
    assert budget >= 0
    cap = 2 if band_count >= 4 else 3
    order = list(range(band_count))
    random.shuffle(order)
    while budget > 0 and order:
        idx = order.pop(0)
        add = random.randint(0, min(cap, budget))
        sizes[idx] += add
        budget -= add
    return sizes


def _choose_layout() -> tuple[str, int, int]:
    mode = random.choice(("cross", "parallel_h", "parallel_v"))
    if mode == "cross":
        return mode, random.choice((2, 3)), random.choice((2, 3))
    if mode == "parallel_h":
        return mode, random.choice((3, 4)), random.choice((1, 2))
    return mode, random.choice((1, 2)), random.choice((3, 4))


def _pick_active_lines(mode: str, h_count: int, v_count: int) -> tuple[tuple[str, int], tuple[str, int]]:
    if mode == "cross":
        a_line: tuple[str, int] = ("row", random.randrange(h_count))
        b_line: tuple[str, int] = ("col", random.randrange(v_count))
        if random.choice((False, True)):
            a_line, b_line = b_line, a_line
        return a_line, b_line

    if mode == "parallel_h":
        start = random.randrange(h_count - 1)
        lines = [("row", start), ("row", start + 1)]
    else:
        start = random.randrange(v_count - 1)
        lines = [("col", start), ("col", start + 1)]

    if random.choice((False, True)):
        lines.reverse()
    return lines[0], lines[1]


def _instruction_region(
    corner: str, row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]]
) -> tuple[int, int]:
    ri = 0 if corner in {"tl", "tr"} else len(row_bands) - 1
    ci = 0 if corner in {"tl", "bl"} else len(col_bands) - 1
    return ri, ci


def _protected_cells(height: int, width: int, corner: str | None) -> set[tuple[int, int]]:
    if corner is None:
        return set()
    if corner == "tl":
        return {(0, 0), (0, 1), (1, 0), (1, 1)}
    if corner == "tr":
        return {(0, width - 2), (0, width - 1), (1, width - 2), (1, width - 1)}
    if corner == "bl":
        return {(height - 2, 0), (height - 2, 1), (height - 1, 0), (height - 1, 1)}
    return {
        (height - 2, width - 2),
        (height - 2, width - 1),
        (height - 1, width - 2),
        (height - 1, width - 1),
    }


def _sample_pattern(
    height: int,
    width: int,
    do_h: bool,
    do_v: bool,
    protected: set[tuple[int, int]],
) -> np.ndarray | None:
    candidates = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if (r, c) not in protected
    ]
    if len(candidates) < 2:
        return None

    max_cells = min(MAX_PATTERN_SIZE, len(candidates))
    for _ in range(200):
        count = random.randint(2, max_cells)
        chosen = random.sample(candidates, count)
        mask = np.zeros((height, width), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        moved = _transform_mask(mask, do_h=do_h, do_v=do_v)
        if any(moved[r, c] for r, c in protected):
            continue
        if (do_h or do_v) and np.array_equal(mask, moved):
            continue
        return mask
    return None


def _paint_lines(
    grid: np.ndarray,
    sep_rows: list[int],
    sep_cols: list[int],
    row_colors: list[int],
    col_colors: list[int],
) -> None:
    if random.choice((False, True)):
        for r, color in zip(sep_rows, row_colors):
            grid[r, :] = color
        for c, color in zip(sep_cols, col_colors):
            grid[:, c] = color
    else:
        for c, color in zip(sep_cols, col_colors):
            grid[:, c] = color
        for r, color in zip(sep_rows, row_colors):
            grid[r, :] = color


def _gen_attempt() -> np.ndarray | None:
    mode, h_count, v_count = _choose_layout()
    row_sizes = _sample_band_sizes(h_count)
    col_sizes = _sample_band_sizes(v_count)
    sep_rows = _separator_positions(row_sizes)
    sep_cols = _separator_positions(col_sizes)
    h = sum(row_sizes) + h_count
    w = sum(col_sizes) + v_count

    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    if len(row_bands) < 2 or len(col_bands) < 2:
        return None

    a_line, b_line = _pick_active_lines(mode, h_count, v_count)
    categories = {
        (_region_active(a_line, ri, ci), _region_active(b_line, ri, ci))
        for ri in range(len(row_bands))
        for ci in range(len(col_bands))
    }
    if categories != {(False, False), (True, False), (False, True), (True, True)}:
        return None

    total_lines = h_count + v_count
    if total_lines > len(LINE_COLORS):
        return None
    line_palette = random.sample(list(LINE_COLORS), total_lines)
    row_colors = line_palette[:h_count]
    col_colors = line_palette[h_count:]

    descriptor_to_color: dict[tuple[str, int], int] = {}
    for idx, color in enumerate(row_colors):
        descriptor_to_color[("row", idx)] = color
    for idx, color in enumerate(col_colors):
        descriptor_to_color[("col", idx)] = color

    grid = np.full((h, w), BG, dtype=int)
    _paint_lines(grid, sep_rows, sep_cols, row_colors, col_colors)

    corner = random.choice(("tl", "tr", "bl", "br"))
    instr_region = _instruction_region(corner, row_bands, col_bands)

    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            corner_name = corner if (ri, ci) == instr_region else None
            protected = _protected_cells(height, width, corner_name)
            do_h = _region_active(a_line, ri, ci)
            do_v = _region_active(b_line, ri, ci)
            mask = _sample_pattern(height, width, do_h, do_v, protected)
            if mask is None:
                return None
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            patch[mask] = GREY

    a_color = descriptor_to_color[a_line]
    b_color = descriptor_to_color[b_line]
    if corner == "tl":
        grid[0:2, 0:2] = np.array([[a_color, a_color], [b_color, b_color]], dtype=int)
    elif corner == "tr":
        grid[0:2, w - 2 : w] = np.array([[a_color, a_color], [b_color, b_color]], dtype=int)
    elif corner == "bl":
        grid[h - 2 : h, 0:2] = np.array([[a_color, a_color], [b_color, b_color]], dtype=int)
    else:
        grid[h - 2 : h, w - 2 : w] = np.array([[a_color, a_color], [b_color, b_color]], dtype=int)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
