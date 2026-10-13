import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 300


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _band_ranges(length: int, separators: list[int]) -> list[tuple[int, int]]:
    sep_set = set(map(int, separators))
    bands: list[tuple[int, int]] = []
    start = 0
    while start < length:
        while start < length and start in sep_set:
            start += 1
        if start >= length:
            break
        end = start
        while end < length and end not in sep_set:
            end += 1
        bands.append((start, end - 1))
        start = end + 1
    return bands


def _separator_structure(grid: np.ndarray, sep_color: int = SEP) -> tuple[list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_rows = [r for r in range(h) if np.all(g[r, :] == sep_color)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == sep_color)]

    claimed = np.zeros_like(g, dtype=bool)
    if sep_rows:
        claimed[sep_rows, :] = True
    if sep_cols:
        claimed[:, sep_cols] = True
    assert np.array_equal(g == sep_color, claimed), "separator pixels must form full rows/columns"

    row_bands = _band_ranges(h, sep_rows)
    col_bands = _band_ranges(w, sep_cols)
    assert row_bands and col_bands, "expected at least one partition block"
    return sep_rows, sep_cols, row_bands, col_bands


def _counts_for_block(
    row_band: tuple[int, int],
    col_band: tuple[int, int],
    shape: tuple[int, int],
    sep_rows: set[int],
    sep_cols: set[int],
) -> tuple[int, int]:
    h, w = map(int, shape)
    r0, r1 = map(int, row_band)
    c0, c1 = map(int, col_band)
    v = int(c0 > 0 and (c0 - 1) in sep_cols) + int(c1 < w - 1 and (c1 + 1) in sep_cols)
    h_count = int(r0 > 0 and (r0 - 1) in sep_rows) + int(r1 < h - 1 and (r1 + 1) in sep_rows)
    return v, h_count


def _apply_transform(patch: np.ndarray, v_count: int, h_count: int) -> np.ndarray:
    if v_count > h_count:
        return np.fliplr(patch)
    if h_count > v_count:
        return np.flipud(patch)
    assert v_count == h_count and v_count > 0, "equal counts must be positive"
    return np.rot90(patch, 2)


def _compress_without_separators(grid: np.ndarray, sep_rows: list[int], sep_cols: list[int]) -> np.ndarray:
    sep_row_set = set(sep_rows)
    sep_col_set = set(sep_cols)
    keep_rows = [r for r in range(grid.shape[0]) if r not in sep_row_set]
    keep_cols = [c for c in range(grid.shape[1]) if c not in sep_col_set]
    return np.asarray(grid, dtype=int)[np.ix_(keep_rows, keep_cols)].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each partition block according to how many horizontal versus vertical separator edges surround it.

    Input structure:
    - The background is black (0), and grey (5) full rows and columns act as one-pixel separators.
    - Removing those separator rows and columns partitions the grid into rectangular blocks.
    - Each block contains a small asymmetric colored motif made of a few non-background pixels.

    Rule:
    - Identify every rectangular block delimited by the grey separator rows/columns.
    - For each block, count V = the number of adjacent vertical separator sides and H = the number of adjacent horizontal separator sides; outer grid borders do not count.
    - If V > H, mirror the whole block left-right; if H > V, mirror it top-bottom; if V == H and V > 0, rotate the whole block by 180 degrees.
    - Remove every separator row and column, then reassemble the transformed blocks in their original relative order.

    Key insight:
    - The separator topology is both the spatial partition and the instruction code: each block's local balance of vertical versus horizontal separator contacts selects its geometric transform.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep_rows, sep_cols, row_bands, col_bands = _separator_structure(g, sep_color=SEP)
    sep_row_set = set(sep_rows)
    sep_col_set = set(sep_cols)

    out_h = sum(r1 - r0 + 1 for r0, r1 in row_bands)
    out_w = sum(c1 - c0 + 1 for c0, c1 in col_bands)
    out = np.full((out_h, out_w), BG, dtype=int)

    out_r = 0
    for row_band in row_bands:
        block_h = row_band[1] - row_band[0] + 1
        out_c = 0
        for col_band in col_bands:
            block_w = col_band[1] - col_band[0] + 1
            patch = crop_bbox(g, (row_band[0], col_band[0], row_band[1], col_band[1]))
            assert not np.any(patch == SEP), "block interiors must exclude separator pixels"
            v_count, h_count = _counts_for_block(row_band, col_band, g.shape, sep_row_set, sep_col_set)
            assert v_count > 0 or h_count > 0, "every block must touch at least one separator"
            transformed = _apply_transform(patch, v_count, h_count)
            out[out_r : out_r + block_h, out_c : out_c + block_w] = transformed
            out_c += block_w
        out_r += block_h

    return out


def _neighbors4(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _connected(mask: np.ndarray) -> bool:
    coords = list(map(tuple, np.argwhere(mask)))
    if not coords:
        return False
    seen = {coords[0]}
    stack = [coords[0]]
    h, w = mask.shape
    while stack:
        cell = stack.pop()
        for nxt in _neighbors4(cell, h, w):
            if not mask[nxt] or nxt in seen:
                continue
            seen.add(nxt)
            stack.append(nxt)
    return len(seen) == len(coords)


def _generate_pattern(block_h: int, block_w: int) -> np.ndarray:
    rows = list(range(1, block_h - 1))
    cols = list(range(1, block_w - 1))
    for _ in range(200):
        patch = np.zeros((block_h, block_w), dtype=int)
        area = int(np.random.randint(2, min(6, len(rows) * len(cols)) + 1))
        start = (int(np.random.choice(rows)), int(np.random.choice(cols)))
        cells = [start]
        used = {start}
        while len(cells) < area:
            frontier = []
            for cell in cells:
                for nxt in _neighbors4(cell, block_h, block_w):
                    if nxt in used or nxt[0] not in rows or nxt[1] not in cols:
                        continue
                    frontier.append(nxt)
            if not frontier:
                break
            nxt = frontier[int(np.random.randint(len(frontier)))]
            used.add(nxt)
            cells.append(nxt)
        if len(cells) != area:
            continue
        color_count = int(np.random.randint(2, min(4, area) + 1))
        colors = np.random.choice(PALETTE, size=color_count, replace=False)
        assigned = np.random.choice(colors, size=area, replace=True)
        if len(set(map(int, assigned))) < 2:
            assigned[0] = int(colors[0])
            assigned[-1] = int(colors[-1])
        for (r, c), color in zip(cells, assigned):
            patch[r, c] = int(color)
        transforms = (np.fliplr(patch), np.flipud(patch), np.rot90(patch, 2))
        if any(np.array_equal(patch, t) for t in transforms):
            continue
        if len({t.tobytes() for t in (patch, *transforms)}) < 4:
            continue
        if not _connected(patch != BG):
            continue
        return patch
    raise RuntimeError("failed to generate a sufficiently asymmetric pattern")


def _build_partition(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    bands: list[tuple[int, int]] = []
    separators: list[int] = []
    cursor = 0
    for idx, size in enumerate(sizes):
        bands.append((cursor, cursor + size - 1))
        cursor += size
        if idx != len(sizes) - 1:
            separators.append(cursor)
            cursor += 1
    return bands, separators, cursor


def _semantic_check(grid: np.ndarray) -> None:
    sep_rows, sep_cols, row_bands, col_bands = _separator_structure(grid, sep_color=SEP)
    assert len(row_bands) >= 3 and len(col_bands) >= 3, "generator must expose all three transform branches"
    raw_compact = _compress_without_separators(grid, sep_rows, sep_cols)
    out = main(grid)
    assert not np.array_equal(out, raw_compact), "output must not collapse to separator removal alone"
    sep_row_set = set(sep_rows)
    sep_col_set = set(sep_cols)

    seen_branches = set()
    out_r = 0
    for row_band in row_bands:
        block_h = row_band[1] - row_band[0] + 1
        out_c = 0
        for col_band in col_bands:
            block_w = col_band[1] - col_band[0] + 1
            patch = crop_bbox(grid, (row_band[0], col_band[0], row_band[1], col_band[1]))
            v_count, h_count = _counts_for_block(row_band, col_band, grid.shape, sep_row_set, sep_col_set)
            branch = "rotate"
            if v_count > h_count:
                branch = "mirror_lr"
            elif h_count > v_count:
                branch = "mirror_ud"
            seen_branches.add(branch)
            expected = _apply_transform(patch, v_count, h_count)
            actual = out[out_r : out_r + block_h, out_c : out_c + block_w]
            assert np.array_equal(actual, expected), "block transform mismatch"
            assert not np.array_equal(actual, patch), "each generated block must visibly change"
            out_c += block_w
        out_r += block_h
    assert seen_branches == {"mirror_lr", "mirror_ud", "rotate"}, "all transform types must appear"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        row_count = int(np.random.randint(3, 5))
        col_count = int(np.random.randint(3, 5))
        row_sizes = [int(np.random.randint(4, 7)) for _ in range(row_count)]
        col_sizes = [int(np.random.randint(4, 7)) for _ in range(col_count)]
        row_bands, sep_rows, total_h = _build_partition(row_sizes)
        col_bands, sep_cols, total_w = _build_partition(col_sizes)
        if total_h > 30 or total_w > 30:
            continue

        grid = np.full((total_h, total_w), BG, dtype=int)
        if sep_rows:
            grid[sep_rows, :] = SEP
        if sep_cols:
            grid[:, sep_cols] = SEP

        ok = True
        for row_band in row_bands:
            for col_band in col_bands:
                block_h = row_band[1] - row_band[0] + 1
                block_w = col_band[1] - col_band[0] + 1
                try:
                    patch = _generate_pattern(block_h, block_w)
                except RuntimeError:
                    ok = False
                    break
                grid[row_band[0] : row_band[1] + 1, col_band[0] : col_band[1] + 1] = patch
            if not ok:
                break
        if not ok:
            continue

        try:
            _semantic_check(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle input")
