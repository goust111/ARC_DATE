from __future__ import annotations

from collections import Counter
import random

import numpy as np


GRAY = 5
ALL_COLORS = tuple(range(10))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_ATTEMPTS = 400
PANEL_GROUP_SPAN = 4
VERT_GROUP_SHAPE = (3, 5)
HORZ_GROUP_SHAPE = (5, 3)
VERT_GRAY_SLOTS = [(0, 0), (0, 2), (2, 0), (2, 2)]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _runs_from_mask(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(np.asarray(mask_1d, dtype=bool)):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask_1d)))
    return runs


def _bands_between_stops(length: int, stops: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for stop in sorted(int(s) for s in stops):
        if start < stop:
            bands.append((start, stop))
        start = stop + 1
    if start < length:
        bands.append((start, length))
    return bands


def _full_rows(subgrid: np.ndarray, color: int) -> list[int]:
    return [r for r in range(subgrid.shape[0]) if np.all(subgrid[r, :] == color)]


def _full_cols(subgrid: np.ndarray, color: int) -> list[int]:
    return [c for c in range(subgrid.shape[1]) if np.all(subgrid[:, c] == color)]


def _majority_color(values: list[int]) -> int:
    ranked = Counter(map(int, values)).most_common()
    assert ranked, "expected at least one vote"
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1], f"majority tie at values={values}"
    return int(ranked[0][0])


def _find_split(grid: np.ndarray, bg: int) -> tuple[int, str, int]:
    candidates = []
    for color in colors_present(grid, exclude={bg}):
        rows = _full_rows(grid, color)
        cols = _full_cols(grid, color)
        if rows or cols:
            candidates.append((color, rows, cols))
    singles = [item for item in candidates if len(item[1]) + len(item[2]) == 1]
    assert len(singles) == 1, f"expected one single global split line, got {candidates}"
    sep_color, rows, cols = singles[0]
    assert (len(rows) == 1) ^ (len(cols) == 1), "expected exactly one global split line"
    if rows:
        return int(rows[0]), "row", int(sep_color)
    return int(cols[0]), "col", int(sep_color)


def _find_board_separator(board: np.ndarray, bg: int) -> int:
    candidates = []
    for color in colors_present(board, exclude={bg}):
        if _full_rows(board, color) and _full_cols(board, color):
            candidates.append(int(color))
    assert len(candidates) == 1, f"expected one internal separator color, got {candidates}"
    return candidates[0]


def _looks_like_board(subgrid: np.ndarray, bg: int) -> bool:
    return any(_full_rows(subgrid, color) and _full_cols(subgrid, color) for color in colors_present(subgrid, exclude={bg}))


def _split_panels(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray, str, int]:
    split_idx, axis, sep_color = _find_split(grid, bg)
    if axis == "col":
        left = grid[:, :split_idx]
        right = grid[:, split_idx + 1 :]
        if _looks_like_board(left, bg):
            return left, right, "right", sep_color
        assert _looks_like_board(right, bg), "could not locate board next to vertical split"
        return right, left, "left", sep_color

    top = grid[:split_idx, :]
    bottom = grid[split_idx + 1 :, :]
    if _looks_like_board(top, bg):
        return top, bottom, "bottom", sep_color
    assert _looks_like_board(bottom, bg), "could not locate board next to horizontal split"
    return bottom, top, "top", sep_color


def _recover_board(board: np.ndarray, bg: int) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    sep_color = _find_board_separator(board, bg)
    sep_rows = _full_rows(board, sep_color)
    sep_cols = _full_cols(board, sep_color)
    row_bands = _bands_between_stops(board.shape[0], sep_rows)
    col_bands = _bands_between_stops(board.shape[1], sep_cols)
    assert len(row_bands) >= 2 and len(col_bands) >= 2

    blocks: list[np.ndarray] = []
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            block = board[r0:r1, c0:c1]
            assert block.size > 0
            blocks.append(block)
    block_shape = blocks[0].shape
    assert all(block.shape == block_shape for block in blocks)

    motif = np.empty(block_shape, dtype=int)
    for r in range(block_shape[0]):
        for c in range(block_shape[1]):
            motif[r, c] = _majority_color([int(block[r, c]) for block in blocks])
    assert np.any(motif != bg), "recovered motif cannot be empty"

    missing = np.zeros((len(row_bands), len(col_bands)), dtype=int)
    idx = 0
    for rr in range(len(row_bands)):
        for cc in range(len(col_bands)):
            block = blocks[idx]
            idx += 1
            keep_mask = block != bg
            assert np.all(block[keep_mask] == motif[keep_mask]), "corrupted copy changed colors instead of erasing"
            missing[rr, cc] = int(np.sum((motif != bg) & (block == bg)))
    return motif, row_bands, missing


def _legend_mapping(legend: np.ndarray, bg: int, side: str) -> dict[int, int]:
    if side in ("left", "right"):
        bands = _runs_from_mask(np.any(legend != bg, axis=1))
        slices = [legend[r0:r1, :] for r0, r1 in bands]
    else:
        bands = _runs_from_mask(np.any(legend != bg, axis=0))
        slices = [legend[:, c0:c1] for c0, c1 in bands]
    assert slices, "expected legend groups"

    mapping: dict[int, int] = {}
    for group in slices:
        colors = colors_present(group, exclude={bg})
        assert GRAY in colors, "legend must contain gray count blocks"
        value_colors = [color for color in colors if color != GRAY]
        assert len(value_colors) == 1, "each legend group must have one value color"
        count = len(components_mask(group == GRAY))
        assert count > 0
        assert count not in mapping, "legend keys must be unique"
        mapping[count] = int(value_colors[0])
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the hidden motif by majority vote, count each region's missing pixels, and translate those counts through the legend.

    Input structure:
    - One full separator row or column splits the canvas into a legend panel and a main board.
    - Inside the main board, separator rows and columns partition an R x C table of equal rectangular regions.
    - Every region is a corrupted copy of the same motif: some motif pixels were erased to background, and no copy is complete.
    - The legend panel contains several groups; each group has some disconnected gray blocks plus one adjacent colored block.

    Rule:
    - Detect the main-board separators and align all regions by their local coordinates.
    - Reconstruct the complete motif by taking the unique majority color at each local position across all regions.
    - For each region, count how many motif pixels are missing there, then find the legend group with that many gray components.
    - Output an R x C summary grid whose cell color is the value color from the matching legend group.

    Key insight:
    - The counted quantity is not visible directly: "missing pixels" only exists relative to a motif that never appears intact, so the majority-vote reconstruction must happen before either counting or legend decoding can work.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    board, legend, side, _split_color = _split_panels(g, bg)
    _motif, row_bands, missing = _recover_board(board, bg)
    mapping = _legend_mapping(legend, bg, side)
    assert set(map(int, missing.ravel())) <= set(mapping)

    out = np.empty((len(row_bands), missing.shape[1]), dtype=int)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            out[r, c] = mapping[int(missing[r, c])]
    return out


def _render_vertical_group(count: int, color: int, bg: int) -> np.ndarray:
    group = np.full(VERT_GROUP_SHAPE, bg, dtype=int)
    for rr, cc in VERT_GRAY_SLOTS[:count]:
        group[rr, cc] = GRAY
    group[0:2, 3:5] = int(color)
    return group


def _render_horizontal_group(count: int, color: int, bg: int) -> np.ndarray:
    return _render_vertical_group(count, color, bg).T


def _panel_capacity(board_extent: int) -> int:
    return min(4, (board_extent + 1) // PANEL_GROUP_SPAN)


def _make_patchwork_motif(
    height: int, width: int, colors: list[int], bg: int, rng: np.random.Generator
) -> np.ndarray:
    row_band_count = int(rng.integers(2, min(4, height) + 1))
    col_band_count = int(rng.integers(2, min(4, width) + 1))
    row_cuts = sorted(rng.choice(np.arange(1, height), size=row_band_count - 1, replace=False).tolist())
    col_cuts = sorted(rng.choice(np.arange(1, width), size=col_band_count - 1, replace=False).tolist())
    row_edges = [0] + row_cuts + [height]
    col_edges = [0] + col_cuts + [width]

    row_index = np.zeros(height, dtype=int)
    col_index = np.zeros(width, dtype=int)
    for idx, (start, end) in enumerate(zip(row_edges[:-1], row_edges[1:])):
        row_index[start:end] = idx
    for idx, (start, end) in enumerate(zip(col_edges[:-1], col_edges[1:])):
        col_index[start:end] = idx

    motif = np.full((height, width), bg, dtype=int)
    support_offset = int(rng.integers(3))
    color_offset = int(rng.integers(len(colors)))
    for r in range(height):
        for c in range(width):
            if (row_index[r] + col_index[c] + r + c + support_offset) % 3 == 0:
                continue
            color_idx = (row_index[r] + 2 * col_index[c] + (r % 2) + color_offset) % len(colors)
            motif[r, c] = int(colors[color_idx])
    support = motif != bg
    support_area = int(np.sum(support))
    if support_area < max(6, height + width - 2) or support_area > int(np.floor(height * width * 0.75)):
        return _make_patchwork_motif(height, width, colors, bg, rng)
    if set(map(int, motif[support])) != set(map(int, colors)):
        return _make_patchwork_motif(height, width, colors, bg, rng)
    if not np.all(np.any(support, axis=0)) or not np.all(np.any(support, axis=1)):
        return _make_patchwork_motif(height, width, colors, bg, rng)
    return motif


def _assign_missing_sets(
    motif: np.ndarray, bg: int, counts: list[int], capacity: int, rng: np.random.Generator
) -> list[set[tuple[int, int]]] | None:
    cells = [tuple(map(int, coord)) for coord in np.argwhere(motif != bg)]
    if sum(counts) > len(cells) * capacity:
        return None
    order = sorted(range(len(counts)), key=lambda idx: counts[idx], reverse=True)
    for _ in range(120):
        remaining = {cell: capacity for cell in cells}
        chosen_sets: list[set[tuple[int, int]] | None] = [None] * len(counts)
        ok = True
        for idx in order:
            need = int(counts[idx])
            available = [cell for cell in cells if remaining[cell] > 0]
            if len(available) < need:
                ok = False
                break
            weights = np.asarray([remaining[cell] for cell in available], dtype=float)
            picks = rng.choice(len(available), size=need, replace=False, p=weights / weights.sum())
            selected = {available[int(i)] for i in np.asarray(picks).tolist()}
            if len(selected) != need:
                ok = False
                break
            for cell in selected:
                remaining[cell] -= 1
            chosen_sets[idx] = selected
        if ok and all(item is not None for item in chosen_sets):
            return [set(item) for item in chosen_sets if item is not None]
    return None


def _render_board(
    motif: np.ndarray, missing_sets: list[set[tuple[int, int]]], rows: int, cols: int, bg: int, sep_color: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    cell_h, cell_w = motif.shape
    board = np.full((rows * cell_h + rows - 1, cols * cell_w + cols - 1), bg, dtype=int)
    for r in range(1, rows):
        board[r * cell_h + (r - 1), :] = sep_color
    for c in range(1, cols):
        board[:, c * cell_w + (c - 1)] = sep_color

    blocks: list[np.ndarray] = []
    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            block = motif.copy()
            for r, c in missing_sets[idx]:
                block[r, c] = bg
            idx += 1
            r0 = rr * (cell_h + 1)
            c0 = cc * (cell_w + 1)
            board[r0 : r0 + cell_h, c0 : c0 + cell_w] = block
            blocks.append(block)
    return board, blocks


def _render_legend(
    board_shape: tuple[int, int],
    side: str,
    mapping: dict[int, int],
    bg: int,
) -> np.ndarray:
    board_h, board_w = map(int, board_shape)
    keys = list(mapping)
    random.shuffle(keys)
    if side in ("left", "right"):
        legend = np.full((board_h, VERT_GROUP_SHAPE[1]), bg, dtype=int)
        used_h = PANEL_GROUP_SPAN * len(keys) - 1
        margin = random.randint(0, board_h - used_h)
        for idx, key in enumerate(keys):
            r0 = margin + idx * PANEL_GROUP_SPAN
            legend[r0 : r0 + VERT_GROUP_SHAPE[0], :] = _render_vertical_group(key, mapping[key], bg)
        return legend

    legend = np.full((HORZ_GROUP_SHAPE[0], board_w), bg, dtype=int)
    used_w = PANEL_GROUP_SPAN * len(keys) - 1
    margin = random.randint(0, board_w - used_w)
    for idx, key in enumerate(keys):
        c0 = margin + idx * PANEL_GROUP_SPAN
        legend[:, c0 : c0 + HORZ_GROUP_SHAPE[1]] = _render_horizontal_group(key, mapping[key], bg)
    return legend


def _assemble_canvas(board: np.ndarray, legend: np.ndarray, side: str, split_color: int, bg: int) -> np.ndarray:
    if side == "right":
        out = np.full((board.shape[0], board.shape[1] + 1 + legend.shape[1]), bg, dtype=int)
        out[:, : board.shape[1]] = board
        out[:, board.shape[1]] = split_color
        out[:, board.shape[1] + 1 :] = legend
        return out
    if side == "left":
        out = np.full((board.shape[0], legend.shape[1] + 1 + board.shape[1]), bg, dtype=int)
        out[:, : legend.shape[1]] = legend
        out[:, legend.shape[1]] = split_color
        out[:, legend.shape[1] + 1 :] = board
        return out
    if side == "bottom":
        out = np.full((board.shape[0] + 1 + legend.shape[0], board.shape[1]), bg, dtype=int)
        out[: board.shape[0], :] = board
        out[board.shape[0], :] = split_color
        out[board.shape[0] + 1 :, :] = legend
        return out
    out = np.full((legend.shape[0] + 1 + board.shape[0], board.shape[1]), bg, dtype=int)
    out[: legend.shape[0], :] = legend
    out[legend.shape[0], :] = split_color
    out[legend.shape[0] + 1 :, :] = board
    return out


def _validate_example(
    grid: np.ndarray,
    expected: np.ndarray,
    motif: np.ndarray,
    blocks: list[np.ndarray],
    counts: list[int],
    mapping: dict[int, int],
    bg: int,
) -> bool:
    if get_bg(grid) != bg:
        return False
    if not np.array_equal(main(grid), expected):
        return False
    board, _legend, _side, _split_color = _split_panels(grid, bg)
    if not np.array_equal(_recover_board(board, bg)[0], motif):
        return False
    if sorted(set(counts)) != sorted(mapping):
        return False
    for block, count in zip(blocks, counts):
        if count <= 0:
            return False
        if np.array_equal(block, motif):
            return False
        if int(np.sum((motif != bg) & (block == bg))) != count:
            return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    side = random.choice(("left", "right", "top", "bottom"))
    rows = int(rng.integers(3, 5))
    cols = int(rng.integers(3, 5))
    cell_h = int(rng.integers(4, 6))
    cell_w = int(rng.integers(4, 6))
    board_h = rows * cell_h + rows - 1
    board_w = cols * cell_w + cols - 1
    panel_extent = board_h if side in ("left", "right") else board_w
    max_key = min(_panel_capacity(panel_extent), rows * cols, cell_h * cell_w - 1)
    if max_key < 2:
        return None

    key_count = int(rng.integers(2, max_key + 1))
    counts = list(range(1, key_count + 1))
    while len(counts) < rows * cols:
        counts.append(int(rng.integers(1, key_count + 1)))
    rng.shuffle(counts)

    palette = [color for color in ALL_COLORS if color != GRAY]
    rng.shuffle(palette)
    bg = int(palette.pop())
    split_color = int(palette.pop())
    board_sep_color = int(palette.pop())
    legend_colors = [int(palette.pop()) for _ in range(key_count)]
    motif_color_count = min(3, len(palette))
    if motif_color_count < 2:
        return None
    motif_color_count = int(rng.integers(2, motif_color_count + 1))
    motif_colors = [int(palette.pop()) for _ in range(motif_color_count)]

    motif = _make_patchwork_motif(cell_h, cell_w, motif_colors, bg, rng)
    capacity = (rows * cols - 1) // 2
    missing_sets = _assign_missing_sets(motif, bg, counts, capacity, rng)
    if missing_sets is None:
        return None

    board, blocks = _render_board(motif, missing_sets, rows, cols, bg, board_sep_color)
    mapping = {key: legend_colors[key - 1] for key in range(1, key_count + 1)}
    legend = _render_legend(board.shape, side, mapping, bg)
    grid = _assemble_canvas(board, legend, side, split_color, bg)
    expected = np.asarray([[mapping[counts[r * cols + c]] for c in range(cols)] for r in range(rows)], dtype=int)

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _validate_example(grid, expected, motif, blocks, counts, mapping, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid missing-count input")
