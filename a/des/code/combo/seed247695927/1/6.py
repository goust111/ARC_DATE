from __future__ import annotations

import random

import numpy as np


BG = 3       # green
SEP = 9      # maroon
_BRANCH_COUNTER = 0


def _separator_info(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    row_candidates: dict[int, list[int]] = {}
    col_candidates: dict[int, list[int]] = {}

    for r in range(h):
        vals = np.unique(g[r, :])
        if vals.size == 1:
            row_candidates.setdefault(int(vals[0]), []).append(int(r))

    for c in range(w):
        vals = np.unique(g[:, c])
        if vals.size == 1:
            col_candidates.setdefault(int(vals[0]), []).append(int(c))

    colors = sorted(set(row_candidates) & set(col_candidates))
    assert len(colors) == 1, "expected exactly one separator color"
    sep = colors[0]
    rows = row_candidates[sep]
    cols = col_candidates[sep]
    assert rows and cols, "expected both horizontal and vertical separator lines"
    return int(sep), rows, cols


def _background_color(grid: np.ndarray, sep_rows: list[int], sep_cols: list[int]) -> int:
    g = np.asarray(grid, dtype=int)
    mask = np.ones(g.shape, dtype=bool)
    mask[sep_rows, :] = False
    mask[:, sep_cols] = False
    vals, counts = np.unique(g[mask], return_counts=True)
    assert vals.size > 0, "expected non-separator area"
    return int(vals[counts.argmax()])


def _intervals(lines: list[int], length: int) -> list[tuple[int, int]]:
    bounds = [-1, *sorted(lines), int(length)]
    parts = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    assert all(end > start for start, end in parts), "separator lines must leave non-empty cells"
    return [(int(start), int(end)) for start, end in parts]


def _majority_color(block: np.ndarray, bg: int, sep: int) -> int:
    vals = block[(block != int(bg)) & (block != int(sep))]
    assert vals.size > 0, "each cell must contain at least one colored pixel"
    colors, counts = np.unique(vals, return_counts=True)
    best = int(counts.max())
    winners = [int(color) for color, count in zip(colors, counts) if int(count) == best]
    assert len(winners) == 1, "cell majority must be unique"
    return winners[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered colored pixels by separator-defined cells, compress each cell
    to its dominant color, then flip the compressed grid along its longer axis.
    """
    g = np.asarray(grid, dtype=int)
    sep, sep_rows, sep_cols = _separator_info(g)
    bg = _background_color(g, sep_rows, sep_cols)

    row_parts = _intervals(sep_rows, g.shape[0])
    col_parts = _intervals(sep_cols, g.shape[1])
    out = np.zeros((len(row_parts), len(col_parts)), dtype=int)

    for i, (r0, r1) in enumerate(row_parts):
        for j, (c0, c1) in enumerate(col_parts):
            out[i, j] = _majority_color(g[r0:r1, c0:c1], bg, sep)

    if out.shape[0] > out.shape[1]:
        out = np.flipud(out)
    elif out.shape[1] > out.shape[0]:
        out = np.fliplr(out)
    return out


def _branch_shape() -> tuple[int, int]:
    global _BRANCH_COUNTER
    branch = _BRANCH_COUNTER % 3
    _BRANCH_COUNTER += 1

    if branch == 0:  # R > C
        rows = random.randint(3, 5)
        cols = random.randint(2, rows - 1)
    elif branch == 1:  # C > R
        cols = random.randint(3, 5)
        rows = random.randint(2, cols - 1)
    else:  # R == C
        rows = cols = random.randint(2, 4)
    return int(rows), int(cols)


def _paint_cell(
    grid: np.ndarray,
    rs: int,
    re: int,
    cs: int,
    ce: int,
    palette: list[int],
) -> None:
    h = re - rs
    w = ce - cs
    area = h * w
    cells = [(r, c) for r in range(rs, re) for c in range(cs, ce)]
    random.shuffle(cells)

    dominant = int(random.choice(palette))
    others = [color for color in palette if color != dominant]
    random.shuffle(others)
    noise_colors = others[: random.randint(0, min(2, len(others)))]

    total_marks = random.randint(4, min(area, 9))
    dominant_count = random.randint(max(2, (total_marks // 2) + 1), total_marks)
    remaining = total_marks - dominant_count

    counts = {dominant: dominant_count}
    for idx, color in enumerate(noise_colors):
        left_colors = len(noise_colors) - idx - 1
        max_take = remaining - left_colors
        take = 0 if max_take <= 0 else random.randint(1, max_take)
        counts[int(color)] = int(take)
        remaining -= take

    if remaining > 0:
        counts[dominant] += int(remaining)

    offset = 0
    for color, count in counts.items():
        for r, c in cells[offset : offset + count]:
            grid[r, c] = int(color)
        offset += int(count)

    assert counts[dominant] > max([counts.get(color, 0) for color in noise_colors] + [0])


def generate_input() -> np.ndarray:
    for _ in range(200):
        rows, cols = _branch_shape()
        row_sizes = [random.randint(3, 7) for _ in range(rows)]
        col_sizes = [random.randint(3, 7) for _ in range(cols)]

        height = sum(row_sizes) + (rows - 1)
        width = sum(col_sizes) + (cols - 1)
        grid = np.full((height, width), BG, dtype=int)

        sep_rows = []
        pos = row_sizes[0]
        for idx in range(rows - 1):
            sep_rows.append(int(pos))
            if idx + 1 < rows - 1:
                pos += 1 + row_sizes[idx + 1]

        sep_cols = []
        pos = col_sizes[0]
        for idx in range(cols - 1):
            sep_cols.append(int(pos))
            if idx + 1 < cols - 1:
                pos += 1 + col_sizes[idx + 1]

        if sep_rows:
            grid[sep_rows, :] = SEP
        if sep_cols:
            grid[:, sep_cols] = SEP

        palette = [color for color in range(10) if color not in (BG, SEP)]
        random.shuffle(palette)
        palette = palette[: random.randint(3, 6)]

        r0 = 0
        used_colors: set[int] = set()
        for rh in row_sizes:
            c0 = 0
            for cw in col_sizes:
                block_palette = palette.copy()
                if random.random() < 0.5:
                    extra = [color for color in range(10) if color not in (BG, SEP) and color not in block_palette]
                    if extra:
                        block_palette.append(random.choice(extra))
                _paint_cell(grid, r0, r0 + rh, c0, c0 + cw, block_palette)
                block = grid[r0 : r0 + rh, c0 : c0 + cw]
                used_colors.update(int(v) for v in np.unique(block) if int(v) not in (BG, SEP))
                c0 += cw + 1
            r0 += rh + 1

        if len(used_colors) < 2:
            continue

        out = main(grid)
        if out.shape == (rows, cols):
            return grid

    raise RuntimeError("failed to generate a valid input")
