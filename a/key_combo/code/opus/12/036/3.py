import numpy as np


MASK_FG = 9
BG = 0
SEP = 5
PALETTE_POOL = np.array([1, 2, 3, 4, 6, 7], dtype=int)
GEN_ATTEMPTS = 400


def _group_consecutive(values: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    if not values:
        return runs
    start = prev = int(values[0])
    for value in values[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def _find_separator_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    sep_cols: list[int] = []
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if np.all(col == col[0]) and int(col[0]) == SEP:
            sep_cols.append(int(c))
    runs = _group_consecutive(sep_cols)
    if len(runs) not in (2, 3):
        raise AssertionError(f"expected 2 or 3 separator runs, got {len(runs)}")
    return runs


def _panel_spans(width: int, runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for r0, r1 in runs:
        spans.append((start, int(r0)))
        start = int(r1) + 1
    spans.append((start, int(width)))
    if len(spans) < 3:
        raise AssertionError("need at least three panels")
    return spans


def _extract_mask(panel: np.ndarray) -> np.ndarray:
    if panel.shape[0] < 5 or panel.shape[1] < 5:
        raise AssertionError(f"panel too small for mask: {panel.shape}")
    if panel.shape[0] != panel.shape[1]:
        raise AssertionError(f"mask panel must be square, got {panel.shape}")
    content = panel[1:-1, 1:-1]
    colors = set(map(int, np.unique(content)))
    if not colors <= {BG, MASK_FG}:
        raise AssertionError(f"unexpected mask colors: {colors}")
    mask = content == MASK_FG
    if mask.shape[0] not in (3, 4):
        raise AssertionError(f"unexpected mask size: {mask.shape}")
    return mask


def _extract_palette(panel: np.ndarray, k: int) -> list[int]:
    nz = np.argwhere(panel != BG)
    if nz.size == 0:
        raise AssertionError("palette panel is empty")
    rows, counts = np.unique(nz[:, 0], return_counts=True)
    row = int(rows[np.argmax(counts)])
    cols = sorted(int(c) for c in np.where(panel[row] != BG)[0])
    if len(cols) != k:
        raise AssertionError(f"palette must have {k} cells, got {len(cols)}")
    if cols[-1] - cols[0] + 1 != k:
        raise AssertionError("palette cells must be contiguous")
    colors = [int(panel[row, c]) for c in cols]
    if len(set(colors)) != k:
        raise AssertionError("palette colors must be distinct")
    if any(color in (BG, SEP, MASK_FG) for color in colors):
        raise AssertionError(f"invalid palette colors: {colors}")
    return colors


def _render(mask_inner: np.ndarray, mask_outer: np.ndarray, palette: list[int]) -> np.ndarray:
    n = int(mask_inner.shape[0])
    out = np.zeros((n * n, n * n), dtype=int)
    for outer_r in range(n):
        tile = np.zeros((n, n), dtype=int)
        for inner_c in range(n):
            tile[:, inner_c][mask_inner[:, inner_c]] = int(palette[(inner_c + outer_r) % n])
        for outer_c in range(n):
            if mask_outer[outer_r, outer_c]:
                r0 = outer_r * n
                c0 = outer_c * n
                out[r0 : r0 + n, c0 : c0 + n] = tile
    return out


def _render_fixed_palette(mask_inner: np.ndarray, mask_outer: np.ndarray, palette: list[int]) -> np.ndarray:
    n = int(mask_inner.shape[0])
    out = np.zeros((n * n, n * n), dtype=int)
    tile = np.zeros((n, n), dtype=int)
    for inner_c in range(n):
        tile[:, inner_c][mask_inner[:, inner_c]] = int(palette[inner_c])
    for outer_r in range(n):
        for outer_c in range(n):
            if mask_outer[outer_r, outer_c]:
                r0 = outer_r * n
                c0 = outer_c * n
                out[r0 : r0 + n, c0 : c0 + n] = tile
    return out


def _render_outer_col_shift(mask_inner: np.ndarray, mask_outer: np.ndarray, palette: list[int]) -> np.ndarray:
    n = int(mask_inner.shape[0])
    out = np.zeros((n * n, n * n), dtype=int)
    for outer_r in range(n):
        for outer_c in range(n):
            if not mask_outer[outer_r, outer_c]:
                continue
            tile = np.zeros((n, n), dtype=int)
            for inner_c in range(n):
                tile[:, inner_c][mask_inner[:, inner_c]] = int(palette[(inner_c + outer_c) % n])
            r0 = outer_r * n
            c0 = outer_c * n
            out[r0 : r0 + n, c0 : c0 + n] = tile
    return out


def _render_inner_row_shift(mask_inner: np.ndarray, mask_outer: np.ndarray, palette: list[int]) -> np.ndarray:
    n = int(mask_inner.shape[0])
    out = np.zeros((n * n, n * n), dtype=int)
    for outer_r in range(n):
        tile = np.zeros((n, n), dtype=int)
        for inner_r in range(n):
            for inner_c in range(n):
                if mask_inner[inner_r, inner_c]:
                    tile[inner_r, inner_c] = int(palette[(inner_c + inner_r) % n])
        for outer_c in range(n):
            if mask_outer[outer_r, outer_c]:
                r0 = outer_r * n
                c0 = outer_c * n
                out[r0 : r0 + n, c0 : c0 + n] = tile
    return out


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("grid must be 2D")
    runs = _find_separator_runs(g)
    spans = _panel_spans(g.shape[1], runs)
    mask_a_panel = g[:, spans[0][0] : spans[0][1]]
    mask_b_panel = g[:, spans[1][0] : spans[1][1]]
    palette_panel = g[:, spans[2][0] : spans[2][1]]

    mask_inner = _extract_mask(mask_a_panel)
    mask_outer = _extract_mask(mask_b_panel)
    if mask_inner.shape != mask_outer.shape:
        raise AssertionError("both masks must share the same size")

    palette = _extract_palette(palette_panel, int(mask_inner.shape[1]))
    return mask_inner, mask_outer, palette


def main(grid: np.ndarray) -> np.ndarray:
    """Expand one binary mask by another, then recolor each outer row with a cyclically shifted palette.

    Input structure:
    - The grid contains separator-delimited panels on a black background.
    - The first two panels are same-size 3x3 or 4x4 binary masks, with maroon foreground cells inside a 1-cell black border.
    - The third panel contains a horizontal palette of K adjacent 1x1 color cells, where K equals the mask width; an optional final panel may be empty.

    Rule:
    - Read the first panel as the inner mask M1 and the second panel as the outer mask M2.
    - Form the Kronecker expansion of M2 by M1: each foreground cell of M2 becomes one copy of M1, and each background cell becomes an all-black block of the same size.
    - For every foreground pixel inside an expanded block coming from outer row r of M2 and inner-column c of M1, recolor it with palette[(c + r) mod K].
    - Keep all background pixels black, and output only the expanded colored grid.

    Key insight:
    - Geometry comes from the Kronecker product of the two binary masks, but color depends on the outer-row index, so neither the layout rule nor a fixed palette alone explains the output.
    """
    mask_inner, mask_outer, palette = _parse_input(grid)
    return _render(mask_inner, mask_outer, palette)


def _sample_inner_mask(rng: np.random.Generator, n: int) -> np.ndarray | None:
    total_min = max(n + 1, int(np.ceil(0.4 * n * n)))
    total_max = min(n * n - 1, int(np.floor(0.75 * n * n)))
    if total_min > total_max:
        return None

    mask = np.zeros((n, n), dtype=bool)
    for c in range(n):
        mask[rng.integers(0, n), c] = True

    target = int(rng.integers(total_min, total_max + 1))
    cells = [(r, c) for r in range(n) for c in range(n) if not mask[r, c]]
    rng.shuffle(cells)
    for r, c in cells[: max(0, target - int(mask.sum()))]:
        mask[r, c] = True

    if int(mask.sum()) < 2 or int(mask.sum()) >= n * n:
        return None
    if np.any(mask.sum(axis=0) == 0):
        return None
    if len({tuple(row.tolist()) for row in mask}) < 2:
        return None
    if len({tuple(col.tolist()) for col in mask.T}) < 2:
        return None
    return mask


def _sample_outer_mask(rng: np.random.Generator, n: int) -> np.ndarray | None:
    total_min = max(n + 2, int(np.ceil(0.4 * n * n)))
    total_max = min(n * n - 1, int(np.floor(0.75 * n * n)))
    if total_min > total_max:
        return None

    mask = np.zeros((n, n), dtype=bool)
    for r in range(n):
        mask[r, rng.integers(0, n)] = True

    target = int(rng.integers(total_min, total_max + 1))
    cells = [(r, c) for r in range(n) for c in range(n) if not mask[r, c]]
    rng.shuffle(cells)
    for r, c in cells[: max(0, target - int(mask.sum()))]:
        mask[r, c] = True

    if int(mask.sum()) < 2 or int(mask.sum()) >= n * n:
        return None
    if np.any(mask.sum(axis=1) == 0):
        return None
    if np.count_nonzero(mask.sum(axis=0) > 0) < 2:
        return None
    if np.max(mask.sum(axis=0)) < 2:
        return None
    if len({tuple(row.tolist()) for row in mask}) < 2:
        return None
    return mask


def _is_semantically_good(mask_inner: np.ndarray, mask_outer: np.ndarray, palette: list[int]) -> bool:
    true_out = _render(mask_inner, mask_outer, palette)
    if not np.array_equal(true_out > 0, np.kron(mask_outer, mask_inner)):
        return False
    if np.array_equal(true_out, _render_fixed_palette(mask_inner, mask_outer, palette)):
        return False
    if np.array_equal(true_out, _render_outer_col_shift(mask_inner, mask_outer, palette)):
        return False
    if np.array_equal(true_out, _render_inner_row_shift(mask_inner, mask_outer, palette)):
        return False
    if np.array_equal(np.kron(mask_outer, mask_inner), np.kron(mask_inner, mask_outer)):
        return False
    used_colors = set(map(int, true_out[true_out != BG]))
    if used_colors != set(map(int, palette)):
        return False
    return True


def _build_input(
    mask_inner: np.ndarray,
    mask_outer: np.ndarray,
    palette: list[int],
    include_blank_panel: bool,
    blank_width: int,
    palette_row: int,
) -> np.ndarray:
    n = int(mask_inner.shape[0])
    mask_panel_a = np.zeros((n + 2, n + 2), dtype=int)
    mask_panel_b = np.zeros((n + 2, n + 2), dtype=int)
    palette_panel = np.zeros((n + 2, n + 2), dtype=int)

    mask_panel_a[1:-1, 1:-1][mask_inner] = MASK_FG
    mask_panel_b[1:-1, 1:-1][mask_outer] = MASK_FG
    palette_panel[palette_row, 1 : 1 + n] = np.array(palette, dtype=int)

    panels = [mask_panel_a, mask_panel_b, palette_panel]
    if include_blank_panel:
        panels.append(np.zeros((n + 2, blank_width), dtype=int))

    width = sum(panel.shape[1] for panel in panels) + (len(panels) - 1)
    grid = np.zeros((n + 2, width), dtype=int)

    c0 = 0
    for index, panel in enumerate(panels):
        w = panel.shape[1]
        grid[:, c0 : c0 + w] = panel
        c0 += w
        if index != len(panels) - 1:
            grid[:, c0] = SEP
            c0 += 1
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        n = int(rng.choice([3, 4]))
        mask_inner = _sample_inner_mask(rng, n)
        mask_outer = _sample_outer_mask(rng, n)
        if mask_inner is None or mask_outer is None:
            continue

        palette = list(map(int, rng.choice(PALETTE_POOL, size=n, replace=False)))
        if not _is_semantically_good(mask_inner, mask_outer, palette):
            continue

        include_blank_panel = bool(rng.integers(0, 2))
        blank_width = int(rng.integers(2, n + 3)) if include_blank_panel else 0
        palette_row = int(rng.integers(1, n + 1))
        grid = _build_input(mask_inner, mask_outer, palette, include_blank_panel, blank_width, palette_row)

        try:
            parsed_inner, parsed_outer, parsed_palette = _parse_input(grid)
        except AssertionError:
            continue
        if not np.array_equal(parsed_inner, mask_inner):
            continue
        if not np.array_equal(parsed_outer, mask_outer):
            continue
        if parsed_palette != palette:
            continue

        out = main(grid)
        if out.ndim != 2 or max(out.shape) > 30:
            continue
        if not np.array_equal(out, _render(mask_inner, mask_outer, palette)):
            continue
        if set(map(int, out[out != BG])) != set(palette):
            continue
        return grid

    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
