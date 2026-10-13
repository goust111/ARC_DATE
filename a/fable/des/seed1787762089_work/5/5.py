import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAROON = 9

HEADER_COLORS = (BLUE, RED, GREEN, YELLOW)
_RNG = np.random.default_rng()


def _mode_color(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).ravel()
    counts = np.bincount(flat, minlength=10)
    return int(np.argmax(counts))


def _rotate_cw(arr: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(arr, -int(turns) % 4)


def _find_separator_color(grid: np.ndarray) -> int:
    body = grid[1:]
    row_counts: dict[int, int] = {}
    col_counts: dict[int, int] = {}

    for row in body:
        if np.all(row == row[0]):
            color = int(row[0])
            row_counts[color] = row_counts.get(color, 0) + 1

    for c in range(body.shape[1]):
        col = body[:, c]
        if np.all(col == col[0]):
            color = int(col[0])
            col_counts[color] = col_counts.get(color, 0) + 1

    candidates = set(row_counts) & set(col_counts)
    assert candidates, "no separator color found"
    return max(candidates, key=lambda color: row_counts[color] + col_counts[color])


def _false_spans(mask: np.ndarray, offset: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for idx, is_sep in enumerate(mask):
        if not is_sep and start is None:
            start = idx
        elif is_sep and start is not None:
            spans.append((offset + start, offset + idx))
            start = None
    if start is not None:
        spans.append((offset + start, offset + len(mask)))
    return spans


def _tile_spans(grid: np.ndarray, separator: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_sep = np.all(grid[1:] == separator, axis=1)
    col_sep = np.all(grid[1:] == separator, axis=0)
    row_spans = _false_spans(row_sep, 1)
    col_spans = _false_spans(col_sep, 0)
    assert row_spans and col_spans, "expected at least one tile row and column"
    heights = {r1 - r0 for r0, r1 in row_spans}
    widths = {c1 - c0 for c0, c1 in col_spans}
    assert len(heights) == 1 and len(widths) == 1, "tiles must be equally sized"
    assert next(iter(heights)) == next(iter(widths)), "tiles must be square"
    return row_spans, col_spans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the maroon-covered tile using the row-major clockwise rotation schedule
    and the header-row column-color legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g[0])
    separator = _find_separator_color(g)
    row_spans, col_spans = _tile_spans(g, separator)
    cols = len(col_spans)

    header_colors: list[int] = []
    for c0, c1 in col_spans:
        segment = g[0, c0:c1]
        colors = np.unique(segment[segment != bg])
        assert len(colors) == 1, "each tile column must have one header color"
        header_colors.append(int(colors[0]))

    blocked = None
    visible_masks: dict[int, np.ndarray] = {}
    for tr, (r0, r1) in enumerate(row_spans):
        for tc, (c0, c1) in enumerate(col_spans):
            idx = tr * cols + tc
            tile = g[r0:r1, c0:c1]
            if np.all(tile == tile[0, 0]) and int(tile[0, 0]) not in (bg, separator):
                assert blocked is None, "expected exactly one covered tile"
                blocked = (idx, tr, tc, r0, r1, c0, c1)
                continue
            mask = tile != bg
            assert mask.any(), "visible tiles must expose a pattern"
            visible_masks[idx] = mask

            tile_colors = set(map(int, np.unique(tile)))
            assert tile_colors.issubset({bg, header_colors[tc]}), "tile uses unexpected colors"

    assert blocked is not None, "no covered tile found"
    assert visible_masks, "expected at least one visible tile"

    ref_idx = min(visible_masks)
    ref_mask = visible_masks[ref_idx]
    for idx, mask in visible_masks.items():
        expected = _rotate_cw(ref_mask, idx - ref_idx)
        assert np.array_equal(mask, expected), "visible tiles must follow the global rotation schedule"

    blocked_idx, _blocked_row, blocked_col, r0, r1, c0, c1 = blocked
    restored_mask = _rotate_cw(ref_mask, blocked_idx - ref_idx)

    out = g.copy()
    tile = out[r0:r1, c0:c1]
    tile[:, :] = bg
    tile[restored_mask] = header_colors[blocked_col]
    return out


# --------------- generator ---------------

TILE_SIZE_CHOICES = (5, 7)
ROW_COL_CHOICES = (
    (2, 3),
    (2, 4),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 2),
    (4, 3),
)
MAX_GENERATION_ATTEMPTS = 300
MASK_ATTEMPTS = 200


def _neighbors4(r: int, c: int, size: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < size and 0 <= nc < size:
            out.append((nr, nc))
    return out


def _random_connected_mask(size: int) -> np.ndarray | None:
    min_cells = max(3, size)
    max_cells = size * size - 2
    for _ in range(MASK_ATTEMPTS):
        target = int(_RNG.integers(min_cells, max_cells + 1))
        mask = np.zeros((size, size), dtype=bool)
        start = (int(_RNG.integers(size)), int(_RNG.integers(size)))
        mask[start] = True
        cells = [start]

        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, size):
                    if not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(_RNG.integers(len(frontier)))]
            mask[nr, nc] = True
            cells.append((nr, nc))

        if len(cells) != target:
            continue
        coords = np.argwhere(mask)
        if len(np.unique(coords[:, 0])) < 2 or len(np.unique(coords[:, 1])) < 2:
            continue
        if len({np.rot90(mask, k).tobytes() for k in range(4)}) != 4:
            continue
        return mask
    return None


def _center_mask(mask: np.ndarray, tile_size: int) -> np.ndarray:
    out = np.zeros((tile_size, tile_size), dtype=bool)
    size = mask.shape[0]
    offset = (tile_size - size) // 2
    out[offset : offset + size, offset : offset + size] = mask
    return out


def _build_layout(rows: int, cols: int, tile_size: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = 1 + rows * tile_size + (rows + 1)
    width = cols * tile_size + (cols + 1)
    grid = np.full((height, width), BLACK, dtype=int)

    row_spans: list[tuple[int, int]] = []
    r = 1
    grid[r, :] = GREY
    r += 1
    for _ in range(rows):
        r0 = r
        r1 = r0 + tile_size
        row_spans.append((r0, r1))
        r = r1
        grid[r, :] = GREY
        r += 1

    col_spans: list[tuple[int, int]] = []
    c = 0
    grid[1:, c] = GREY
    c += 1
    for _ in range(cols):
        c0 = c
        c1 = c0 + tile_size
        col_spans.append((c0, c1))
        c = c1
        grid[1:, c] = GREY
        c += 1

    return grid, row_spans, col_spans


def _expected_grid(
    rows: int,
    cols: int,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    centered_mask: np.ndarray,
    header_colors: list[int],
    base_turns: int,
) -> np.ndarray:
    tile_size = centered_mask.shape[0]
    out, _, _ = _build_layout(rows, cols, tile_size)
    for tc, (c0, c1) in enumerate(col_spans):
        center = c0 + tile_size // 2
        out[0, center] = header_colors[tc]
    for tr, (r0, r1) in enumerate(row_spans):
        for tc, (c0, c1) in enumerate(col_spans):
            idx = tr * cols + tc
            tile_mask = _rotate_cw(centered_mask, base_turns + idx)
            tile = np.full((tile_size, tile_size), BLACK, dtype=int)
            tile[tile_mask] = header_colors[tc]
            out[r0:r1, c0:c1] = tile
    return out


def _gen_attempt() -> np.ndarray | None:
    rows, cols = ROW_COL_CHOICES[int(_RNG.integers(len(ROW_COL_CHOICES)))]
    tile_size = TILE_SIZE_CHOICES[int(_RNG.integers(len(TILE_SIZE_CHOICES)))]
    inner_size_choices = (3,) if tile_size == 5 else (3, 5)
    inner_size = inner_size_choices[int(_RNG.integers(len(inner_size_choices)))]
    inner_mask = _random_connected_mask(inner_size)
    if inner_mask is None:
        return None

    centered_mask = _center_mask(inner_mask, tile_size)
    base_turns = int(_RNG.integers(4))
    header_colors = list(map(int, _RNG.choice(np.array(HEADER_COLORS), size=cols, replace=False)))

    grid, row_spans, col_spans = _build_layout(rows, cols, tile_size)
    solved = _expected_grid(rows, cols, row_spans, col_spans, centered_mask, header_colors, base_turns)
    grid[:, :] = solved

    blocked_idx = int(_RNG.integers(rows * cols))
    blocked_row = blocked_idx // cols
    blocked_col = blocked_idx % cols
    r0, r1 = row_spans[blocked_row]
    c0, c1 = col_spans[blocked_col]
    grid[r0:r1, c0:c1] = MAROON

    if not np.array_equal(main(grid), solved):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
