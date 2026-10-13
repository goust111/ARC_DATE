import random
from collections import Counter

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _group_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _bands_from_separators(size: int, sep_indices: list[int]) -> list[tuple[int, int]]:
    runs = _group_runs(sorted(sep_indices))
    bands: list[tuple[int, int]] = []
    cursor = 0
    for start, end in runs:
        if cursor < start:
            bands.append((cursor, start))
        cursor = end + 1
    if cursor < size:
        bands.append((cursor, size))
    return bands


def _separator_layout(
    grid: np.ndarray, bg: int
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    line_counts: Counter[int] = Counter()

    for r in range(h):
        row = g[r, :]
        color = int(row[0])
        if color != bg and np.all(row == color):
            line_counts[color] += 1
    for c in range(w):
        col = g[:, c]
        color = int(col[0])
        if color != bg and np.all(col == color):
            line_counts[color] += 1

    assert line_counts, "expected full-width/full-height separator lines"
    sep = max(line_counts.items(), key=lambda item: item[1])[0]

    sep_rows = [r for r in range(h) if np.all(g[r, :] == sep)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == sep)]
    row_bands = _bands_from_separators(h, sep_rows)
    col_bands = _bands_from_separators(w, sep_cols)

    assert row_bands and len(col_bands) >= 3, "expected panel grid with at least three columns"
    panel_heights = {r1 - r0 for r0, r1 in row_bands}
    panel_widths = {c1 - c0 for c0, c1 in col_bands}
    assert len(panel_heights) == 1 and len(panel_widths) == 1, "panels must be equal-sized"

    panel_h = panel_heights.pop()
    panel_w = panel_widths.pop()
    assert panel_h > 0 and panel_w > 0
    return sep, row_bands, col_bands, panel_h, panel_w


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "shape must be non-empty"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _shape_signature(panel: np.ndarray, bg: int, sep: int) -> tuple[tuple[int, int], tuple[int, ...]]:
    mask = (panel != bg) & (panel != sep)
    norm = _normalize_mask(mask)
    return tuple(norm.shape), tuple(int(v) for v in norm.astype(np.uint8).ravel())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the odd panel in each row of a separator-divided panel grid and stack those panels vertically.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = 0 if np.any(g == 0) else _dominant_color(g)
    sep, row_bands, col_bands, panel_h, panel_w = _separator_layout(g, bg)

    odd_panels: list[np.ndarray] = []
    for r0, r1 in row_bands:
        panels = [g[r0:r1, c0:c1].copy() for c0, c1 in col_bands]
        signatures = [_shape_signature(panel, bg, sep) for panel in panels]
        counts = Counter(signatures)
        odd_sigs = [sig for sig, count in counts.items() if count == 1]
        common_sigs = [sig for sig, count in counts.items() if count == len(panels) - 1]
        assert len(odd_sigs) == 1 and len(common_sigs) == 1, "each row must contain one unique odd shape"
        odd_panels.append(panels[signatures.index(odd_sigs[0])])

    out_h = len(odd_panels) * panel_h + max(0, len(odd_panels) - 1)
    out = np.full((out_h, panel_w), bg, dtype=int)
    for idx, panel in enumerate(odd_panels):
        start = idx * (panel_h + 1)
        out[start : start + panel_h, :] = panel
        if idx + 1 < len(odd_panels):
            out[start + panel_h, :] = sep
    return out


# --------------- generator ---------------

BG_COLOR = 0
SEP_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

ROW_COUNT_RANGE = (2, 4)
COL_COUNT_RANGE = (3, 5)
PANEL_H_RANGE = (4, 7)
PANEL_W_RANGE = (4, 7)

SHAPE_MODE_NAMES = ("rect", "hollow", "plus", "tee", "el", "stairs", "growth")
GENERATION_ATTEMPTS = 300


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    cells = {tuple(map(int, rc)) for rc in coords}
    stack = [next(iter(cells))]
    seen = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nbr = (r + dr, c + dc)
            if nbr in cells and nbr not in seen:
                stack.append(nbr)
    return len(seen) == len(cells)


def _random_growth_shape(h: int, w: int) -> np.ndarray:
    target = random.randint(max(3, min(h, w)), max(3, min(h * w, h + w + 2)))
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _base_shape(mode: str, h: int, w: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    if mode == "rect":
        fill_h = random.randint(1, h)
        fill_w = random.randint(1, w)
        if fill_h * fill_w < 2:
            return None
        mask[:fill_h, :fill_w] = True
    elif mode == "hollow":
        if h < 3 or w < 3:
            return None
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
    elif mode == "plus":
        if h < 3 or w < 3:
            return None
        mask[h // 2, :] = True
        mask[:, w // 2] = True
    elif mode == "tee":
        if h < 2 or w < 3:
            return None
        mask[0, :] = True
        mask[:, w // 2] = True
    elif mode == "el":
        mask[:, 0] = True
        mask[-1, :] = True
    elif mode == "stairs":
        for r in range(h):
            c = round((w - 1) * r / max(1, h - 1))
            mask[r, : c + 1] = True
    elif mode == "growth":
        mask = _random_growth_shape(h, w)
    else:
        return None
    return mask


def _random_transform(mask: np.ndarray, max_h: int, max_w: int) -> np.ndarray:
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.35 and mask.shape[1] <= max_h and mask.shape[0] <= max_w:
        mask = mask.T
    return np.array(mask, dtype=bool)


def _random_shape(max_h: int, max_w: int) -> np.ndarray:
    h_limit = min(max_h, 6)
    w_limit = min(max_w, 6)
    for _ in range(100):
        h = random.randint(2, h_limit)
        w = random.randint(2, w_limit)
        mode = random.choice(SHAPE_MODE_NAMES)
        mask = _base_shape(mode, h, w)
        if mask is None:
            continue
        mask = _random_transform(mask, max_h, max_w)
        mask = _normalize_mask(mask)
        if mask.shape[0] > max_h or mask.shape[1] > max_w:
            continue
        if int(mask.sum()) < 2:
            continue
        if not _is_connected(mask):
            continue
        return mask
    raise RuntimeError("failed to sample a shape")


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    norm = _normalize_mask(mask)
    return tuple(norm.shape), tuple(int(v) for v in norm.astype(np.uint8).ravel())


def _mask_distance(a: np.ndarray, b: np.ndarray) -> int:
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])
    aa = np.zeros((h, w), dtype=bool)
    bb = np.zeros((h, w), dtype=bool)
    aa[: a.shape[0], : a.shape[1]] = a
    bb[: b.shape[0], : b.shape[1]] = b
    return int(np.count_nonzero(aa != bb))


def _sample_distinct_shapes(panel_h: int, panel_w: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(100):
        repeated = _random_shape(panel_h, panel_w)
        odd = _random_shape(panel_h, panel_w)
        if _mask_signature(repeated) == _mask_signature(odd):
            continue
        if _mask_distance(repeated, odd) < 2:
            continue
        return repeated, odd
    raise RuntimeError("failed to sample distinct row shapes")


def _render_panel(panel_h: int, panel_w: int, mask: np.ndarray, color: int) -> np.ndarray:
    panel = np.full((panel_h, panel_w), BG_COLOR, dtype=int)
    max_r = panel_h - mask.shape[0]
    max_c = panel_w - mask.shape[1]
    top = random.randint(0, max_r)
    left = random.randint(0, max_c)
    region = panel[top : top + mask.shape[0], left : left + mask.shape[1]]
    region[mask] = color
    return panel


def _stack_expected(odd_panels: list[np.ndarray]) -> np.ndarray:
    panel_h, panel_w = odd_panels[0].shape
    out_h = len(odd_panels) * panel_h + len(odd_panels) - 1
    out = np.full((out_h, panel_w), BG_COLOR, dtype=int)
    for idx, panel in enumerate(odd_panels):
        start = idx * (panel_h + 1)
        out[start : start + panel_h, :] = panel
        if idx + 1 < len(odd_panels):
            out[start + panel_h, :] = SEP_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    row_count = random.randint(*ROW_COUNT_RANGE)
    col_count = random.randint(*COL_COUNT_RANGE)
    panel_h = random.randint(*PANEL_H_RANGE)
    panel_w = random.randint(*PANEL_W_RANGE)

    grid_h = row_count * panel_h + row_count - 1
    grid_w = col_count * panel_w + col_count - 1
    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)

    row_starts = [r * (panel_h + 1) for r in range(row_count)]
    col_starts = [c * (panel_w + 1) for c in range(col_count)]
    for r in row_starts[:-1]:
        grid[r + panel_h, :] = SEP_COLOR
    for c in col_starts[:-1]:
        grid[:, c + panel_w] = SEP_COLOR

    odd_panels: list[np.ndarray] = []
    for row_idx, row_start in enumerate(row_starts):
        repeated_mask, odd_mask = _sample_distinct_shapes(panel_h, panel_w)
        odd_col = random.randrange(col_count)
        repeated_color = random.choice(PALETTE)

        if random.random() < 0.5:
            odd_color = repeated_color
        else:
            odd_color = random.choice([c for c in PALETTE if c != repeated_color])

        row_signatures = []
        for col_idx, col_start in enumerate(col_starts):
            if col_idx == odd_col:
                panel = _render_panel(panel_h, panel_w, odd_mask, odd_color)
                odd_panels.append(panel.copy())
            else:
                panel = _render_panel(panel_h, panel_w, repeated_mask, repeated_color)

            grid[row_start : row_start + panel_h, col_start : col_start + panel_w] = panel
            row_signatures.append(
                _shape_signature(panel, BG_COLOR, SEP_COLOR)
            )

        counts = Counter(row_signatures)
        if sorted(counts.values()) != [1, col_count - 1]:
            return None
        assert len(row_signatures) == col_count
        assert odd_panels[row_idx].shape == (panel_h, panel_w)

    expected = _stack_expected(odd_panels)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
