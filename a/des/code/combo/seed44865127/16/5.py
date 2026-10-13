from __future__ import annotations

import random

import numpy as np


COLOR_POOL = list(range(10))
GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _consecutive_groups(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    groups: list[tuple[int, int]] = []
    start = indices[0]
    end = indices[0]
    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
            continue
        groups.append((start, end))
        start = idx
        end = idx
    groups.append((start, end))
    return groups


def _is_solid_non_bg_col(grid: np.ndarray, bg: int, col: int) -> bool:
    value = int(grid[0, col])
    return value != bg and bool(np.all(grid[:, col] == value))


def _is_solid_non_bg_row(grid: np.ndarray, bg: int, row: int) -> bool:
    value = int(grid[row, 0])
    return value != bg and bool(np.all(grid[row, :] == value))


def _horizontal_candidates(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    h, w = grid.shape
    solid_cols = [c for c in range(w) if _is_solid_non_bg_col(grid, bg, c)]
    groups = _consecutive_groups(solid_cols)
    candidates: list[dict[str, int]] = []

    for left, right in zip(groups, groups[1:], strict=False):
        l0, l1 = left
        r0, r1 = right
        if r0 - l1 != 2:
            continue
        if l0 < 2 or r1 > w - 3:
            continue

        sep_color = int(grid[0, l0])
        if not bool(np.all(grid[:, l0 : l1 + 1] == sep_color)):
            continue
        if not bool(np.all(grid[:, r0 : r1 + 1] == sep_color)):
            continue
        if int(grid[0, r0]) != sep_color:
            continue

        src = grid[:, :l0]
        legend = grid[:, l1 + 1]
        tgt = grid[:, r1 + 1 :]
        if src.shape[1] < 2 or tgt.shape[1] < 2:
            continue
        if not bool(np.all(tgt == bg)):
            continue
        if bool(np.any(legend == bg)):
            continue

        counts = np.count_nonzero(src != bg, axis=1)
        if bool(np.any(counts <= 0)) or bool(np.any(counts > legend.size)):
            continue

        candidates.append(
            {
                "kind": 0,
                "src_end": l0,
                "legend_idx": l1 + 1,
                "tgt_start": r1 + 1,
            }
        )

    return candidates


def _vertical_candidates(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    h, w = grid.shape
    solid_rows = [r for r in range(h) if _is_solid_non_bg_row(grid, bg, r)]
    groups = _consecutive_groups(solid_rows)
    candidates: list[dict[str, int]] = []

    for top, bottom in zip(groups, groups[1:], strict=False):
        t0, t1 = top
        b0, b1 = bottom
        if b0 - t1 != 2:
            continue
        if t0 < 2 or b1 > h - 3:
            continue

        sep_color = int(grid[t0, 0])
        if not bool(np.all(grid[t0 : t1 + 1, :] == sep_color)):
            continue
        if not bool(np.all(grid[b0 : b1 + 1, :] == sep_color)):
            continue
        if int(grid[b0, 0]) != sep_color:
            continue

        src = grid[:t0, :]
        legend = grid[t1 + 1, :]
        tgt = grid[b1 + 1 :, :]
        if src.shape[0] < 2 or tgt.shape[0] < 2:
            continue
        if not bool(np.all(tgt == bg)):
            continue
        if bool(np.any(legend == bg)):
            continue

        counts = np.count_nonzero(src != bg, axis=0)
        if bool(np.any(counts <= 0)) or bool(np.any(counts > legend.size)):
            continue

        candidates.append(
            {
                "kind": 1,
                "src_end": t0,
                "legend_idx": t1 + 1,
                "tgt_start": b1 + 1,
            }
        )

    return candidates


def _parse_layout(grid: np.ndarray) -> tuple[int, dict[str, int], int]:
    bg = _get_bg(grid)
    candidates = _horizontal_candidates(grid, bg) + _vertical_candidates(grid, bg)
    assert len(candidates) == 1, "expected exactly one valid panel layout"
    return int(candidates[0]["kind"]), candidates[0], bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use per-line non-background counts in the source panel as indices into a visible color legend.

    Horizontal layout:
    - left source panel, separator columns, one-column legend, separator columns, right target panel
    - count each source row's non-background cells, then fill the matching target row

    Vertical layout:
    - top source panel, separator rows, one-row legend, separator rows, bottom target panel
    - count each source column's non-background cells, then fill the matching target column
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    kind, layout, bg = _parse_layout(g)
    out = g.copy()

    if kind == 0:
        src = g[:, : layout["src_end"]]
        legend = g[:, layout["legend_idx"]]
        counts = np.count_nonzero(src != bg, axis=1)
        fill = legend[counts - 1]
        out[:, layout["tgt_start"] :] = fill[:, None]
    else:
        src = g[: layout["src_end"], :]
        legend = g[layout["legend_idx"], :]
        counts = np.count_nonzero(src != bg, axis=0)
        fill = legend[counts - 1]
        out[layout["tgt_start"] :, :] = fill[None, :]

    return out


def _line_positions(length: int, count: int) -> list[int]:
    assert 1 <= count <= length
    if count == length:
        return list(range(length))
    if count == 1:
        return [random.randrange(length)]

    mode = random.choice(["scatter", "block"])
    if mode == "block":
        start = random.randint(0, length - count)
        return list(range(start, start + count))
    return sorted(random.sample(range(length), count))


def _source_palette(legend_colors: list[int]) -> list[int]:
    size = random.randint(2, min(4, len(legend_colors)))
    return random.sample(legend_colors, size)


def _make_counts(line_count: int, max_count: int) -> list[int]:
    counts = [random.randint(1, max_count) for _ in range(line_count)]
    i, j = random.sample(range(line_count), 2)
    counts[i] = 1
    counts[j] = max_count
    return counts


def _draw_horizontal_source(height: int, width: int, bg: int, palette: list[int], counts: list[int]) -> np.ndarray:
    panel = np.full((height, width), bg, dtype=int)
    for r, count in enumerate(counts):
        positions = _line_positions(width, count)
        row_colors = [random.choice(palette) for _ in positions]
        if count == width and len(set(row_colors)) == 1 and len(palette) > 1:
            row_colors[-1] = random.choice([c for c in palette if c != row_colors[0]])
        for c, color in zip(positions, row_colors, strict=False):
            panel[r, c] = color
    return panel


def _draw_vertical_source(height: int, width: int, bg: int, palette: list[int], counts: list[int]) -> np.ndarray:
    panel = np.full((height, width), bg, dtype=int)
    for c, count in enumerate(counts):
        positions = _line_positions(height, count)
        col_colors = [random.choice(palette) for _ in positions]
        if count == height and len(set(col_colors)) == 1 and len(palette) > 1:
            col_colors[-1] = random.choice([x for x in palette if x != col_colors[0]])
        for r, color in zip(positions, col_colors, strict=False):
            panel[r, c] = color
    return panel


def _has_uniform_non_bg_line(panel: np.ndarray, bg: int) -> bool:
    for row in panel:
        if int(row[0]) != bg and bool(np.all(row == row[0])):
            return True
    for col in panel.T:
        if int(col[0]) != bg and bool(np.all(col == col[0])):
            return True
    return False


def _bg_is_unique_mode(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    best = int(np.max(counts))
    bg_count = int(counts[np.where(values == bg)[0][0]])
    return bg_count == best and int(np.sum(counts == best)) == 1


def _expected_horizontal(
    source: np.ndarray,
    legend_colors: list[int],
    bg: int,
    sep_color: int,
    sep_width: int,
    target_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, _ = source.shape
    legend = np.array(legend_colors, dtype=int)[:, None]
    sep = np.full((height, sep_width), sep_color, dtype=int)
    target = np.full((height, target_width), bg, dtype=int)
    grid = np.hstack([source, sep, legend, sep, target])
    out = grid.copy()
    counts = np.count_nonzero(source != bg, axis=1)
    fill = np.array(legend_colors, dtype=int)[counts - 1]
    out[:, -target_width:] = fill[:, None]
    return grid, out


def _expected_vertical(
    source: np.ndarray,
    legend_colors: list[int],
    bg: int,
    sep_color: int,
    sep_height: int,
    target_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    _, width = source.shape
    legend = np.array(legend_colors, dtype=int)[None, :]
    sep = np.full((sep_height, width), sep_color, dtype=int)
    target = np.full((target_height, width), bg, dtype=int)
    grid = np.vstack([source, sep, legend, sep, target])
    out = grid.copy()
    counts = np.count_nonzero(source != bg, axis=0)
    fill = np.array(legend_colors, dtype=int)[counts - 1]
    out[-target_height:, :] = fill[None, :]
    return grid, out


def _gen_horizontal() -> np.ndarray | None:
    height = random.randint(4, 8)
    source_width = random.randint(2, height)
    target_width = random.randint(2, 7)
    sep_width = random.randint(1, 2)

    picked = random.sample(COLOR_POOL, 2 + height)
    bg = picked[0]
    sep_color = picked[1]
    legend_colors = picked[2:]
    counts = _make_counts(height, source_width)
    source = _draw_horizontal_source(height, source_width, bg, _source_palette(legend_colors), counts)
    if _has_uniform_non_bg_line(source, bg):
        return None
    if len(set(map(int, source[source != bg].tolist()))) < 2:
        return None

    grid, expected = _expected_horizontal(source, legend_colors, bg, sep_color, sep_width, target_width)
    if not _bg_is_unique_mode(grid, bg):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def _gen_vertical() -> np.ndarray | None:
    width = random.randint(4, 8)
    source_height = random.randint(2, width)
    target_height = random.randint(2, 7)
    sep_height = random.randint(1, 2)

    picked = random.sample(COLOR_POOL, 2 + width)
    bg = picked[0]
    sep_color = picked[1]
    legend_colors = picked[2:]
    counts = _make_counts(width, source_height)
    source = _draw_vertical_source(source_height, width, bg, _source_palette(legend_colors), counts)
    if _has_uniform_non_bg_line(source, bg):
        return None
    if len(set(map(int, source[source != bg].tolist()))) < 2:
        return None

    grid, expected = _expected_vertical(source, legend_colors, bg, sep_color, sep_height, target_height)
    if not _bg_is_unique_mode(grid, bg):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        if random.choice([False, True]):
            grid = _gen_horizontal()
        else:
            grid = _gen_vertical()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
