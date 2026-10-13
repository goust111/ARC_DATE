import random

import numpy as np


BG = 0
DIVIDER = 5
PATH = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8)

GEN_RETRIES = 300
LAYER_RANGE = (3, 6)
BLOCK_WIDTH_RANGE = (3, 6)
BAND_HEIGHT_RANGE = (2, 4)
PATH_GAP_RANGE = (1, 3)
CENTER_HEIGHT_RANGE = (1, 4)
CENTER_WIDTH_RANGE = (1, 5)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and int(values[end]) == int(values[start]):
            end += 1
        runs.append((start, end, int(values[start])))
        start = end
    return runs


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    best = None
    start = 0
    while start < len(mask):
        if not bool(mask[start]):
            start += 1
            continue
        end = start + 1
        while end < len(mask) and bool(mask[end]):
            end += 1
        cand = (start, end - 1)
        if best is None or (cand[1] - cand[0]) > (best[1] - best[0]):
            best = cand
        start = end
    assert best is not None
    return best


def _find_divider(grid: np.ndarray) -> tuple[int, int]:
    full_cols = np.all(grid == DIVIDER, axis=0)
    return _longest_true_run(full_cols)


def _parse_left_panel(panel: np.ndarray) -> tuple[list[int], list[bool]]:
    band_mask = (panel != BG) & (panel != DIVIDER) & (panel != PATH)
    r0, c0, r1, c1 = _bbox(band_mask)
    band = panel[r0 : r1 + 1, c0 : c1 + 1]

    col_colors: list[int | None] = []
    for c in range(band.shape[1]):
        vals = np.unique(band[:, c][(band[:, c] != BG) & (band[:, c] != DIVIDER) & (band[:, c] != PATH)])
        assert vals.size <= 1
        col_colors.append(None if vals.size == 0 else int(vals[0]))

    block_colors: list[int] = []
    block_cols: list[list[int]] = []
    current_color = None
    current_cols: list[int] = []
    for idx, color in enumerate(col_colors):
        if color is None:
            continue
        if color != current_color:
            if current_cols:
                block_colors.append(int(current_color))
                block_cols.append(current_cols)
            current_color = int(color)
            current_cols = [idx]
        else:
            current_cols.append(idx)
    if current_cols:
        block_colors.append(int(current_color))
        block_cols.append(current_cols)

    assert block_colors
    assert len(block_colors) == len(set(block_colors))

    path_mask = panel == PATH
    for c in range(c0, c1 + 1):
        assert np.any(path_mask[:, c]), "path must project onto every band column"

    orientations: list[bool] = []
    for cols in block_cols:
        sample = cols[1:-1] if len(cols) >= 3 else cols
        sample_cols = [c0 + c for c in sample]

        votes = 0
        for c in sample_cols:
            rows = np.flatnonzero(path_mask[:, c])
            assert rows.size > 0
            if int(rows.max()) < r0:
                votes -= 1
            elif int(rows.min()) > r1:
                votes += 1
            else:
                has_above = bool(np.any(rows < r0))
                has_below = bool(np.any(rows > r1))
                if has_above and not has_below:
                    votes -= 1
                elif has_below and not has_above:
                    votes += 1
        assert votes != 0, "each block must have a decisive path side"
        orientations.append(votes < 0)

    assert any(orientations) and not all(orientations)
    return block_colors, orientations


def _extract_template(panel: np.ndarray) -> tuple[tuple[int, int, int, int], list[int]]:
    mask = (panel != BG) & (panel != DIVIDER) & (panel != PATH)
    r0, c0, r1, c1 = _bbox(mask)
    sub = panel[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((sub != BG) & (sub != DIVIDER) & (sub != PATH))

    top, left = 0, 0
    bottom, right = sub.shape[0] - 1, sub.shape[1] - 1
    raw_layers: list[int] = []
    while top <= bottom and left <= right:
        border_vals = []
        border_vals.extend(int(x) for x in sub[top, left : right + 1])
        border_vals.extend(int(x) for x in sub[bottom, left : right + 1])
        if top + 1 <= bottom - 1:
            border_vals.extend(int(x) for x in sub[top + 1 : bottom, left])
            border_vals.extend(int(x) for x in sub[top + 1 : bottom, right])
        uniq = sorted(set(border_vals))
        assert len(uniq) == 1
        raw_layers.append(uniq[0])
        top += 1
        left += 1
        bottom -= 1
        right -= 1

    layers: list[int] = []
    for color in raw_layers:
        if not layers or layers[-1] != color:
            layers.append(color)
    assert len(layers) == len(set(layers))
    return (r0, c0, r1, c1), layers


def _paint_template(panel: np.ndarray, bbox: tuple[int, int, int, int], layers: list[int]) -> None:
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    for rr in range(height):
        for cc in range(width):
            depth = min(rr, cc, height - 1 - rr, width - 1 - cc)
            idx = min(depth, len(layers) - 1)
            panel[r0 + rr, c0 + cc] = layers[idx]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the concentric template colors using the path's above/below grouping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    div0, div1 = _find_divider(g)
    assert 0 < div0 <= div1 < g.shape[1] - 1

    left = g[:, :div0]
    right = g[:, div1 + 1 :]
    block_colors, is_up = _parse_left_panel(left)
    bbox, template_layers = _extract_template(right)

    assert len(block_colors) == len(template_layers)
    assert set(block_colors) == set(template_layers)

    ups = [color for color, up in zip(block_colors, is_up) if up]
    downs = [color for color, up in zip(block_colors, is_up) if not up]
    reordered = ups + list(reversed(downs))
    assert len(reordered) == len(template_layers)

    out = g.copy()
    _paint_template(out[:, div1 + 1 :], bbox, reordered)
    return out


def _target_order(colors: list[int], is_up: list[bool]) -> list[int]:
    ups = [color for color, up in zip(colors, is_up) if up]
    downs = [color for color, up in zip(colors, is_up) if not up]
    return ups + list(reversed(downs))


def _random_pattern(rng: random.Random, n: int) -> list[bool]:
    for _ in range(100):
        pattern = [bool(rng.getrandbits(1)) for _ in range(n)]
        if any(pattern) and not all(pattern):
            changes = sum(pattern[i] != pattern[i - 1] for i in range(1, n))
            if changes >= 1:
                return pattern
    raise RuntimeError("failed to sample a valid path pattern")


def _paint_nested_template(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    colors: list[int],
) -> None:
    for rr in range(height):
        for cc in range(width):
            depth = min(rr, cc, height - 1 - rr, width - 1 - cc)
            idx = min(depth, len(colors) - 1)
            grid[top + rr, left + cc] = colors[idx]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_layers = rng.randint(*LAYER_RANGE)
    colors = rng.sample(PALETTE, n_layers)
    block_widths = [rng.randint(*BLOCK_WIDTH_RANGE) for _ in range(n_layers)]
    band_h = rng.randint(*BAND_HEIGHT_RANGE)
    above_gap = rng.randint(*PATH_GAP_RANGE)
    below_gap = rng.randint(*PATH_GAP_RANGE)
    pattern = _random_pattern(rng, n_layers)
    desired_layers = _target_order(colors, pattern)

    template_colors = colors[:]
    for _ in range(50):
        rng.shuffle(template_colors)
        if template_colors != desired_layers:
            break
    else:
        return None

    center_h = rng.randint(*CENTER_HEIGHT_RANGE)
    center_w = rng.randint(*CENTER_WIDTH_RANGE)
    template_h = center_h + 2 * (n_layers - 1)
    template_w = center_w + 2 * (n_layers - 1)

    band_w = sum(block_widths)
    left_panel_w = band_w + rng.randint(2, 6)
    right_panel_w = template_w + rng.randint(2, 6)
    total_w = left_panel_w + 1 + right_panel_w

    left_content_h = above_gap + band_h + below_gap
    total_h = max(left_content_h, template_h) + rng.randint(2, 6)

    grid = np.full((total_h, total_w), BG, dtype=int)
    divider_col = left_panel_w
    grid[:, divider_col] = DIVIDER

    band_left_max = left_panel_w - band_w
    template_left_max = right_panel_w - template_w
    if band_left_max < 1 or template_left_max < 1:
        return None

    band_left = rng.randint(1, band_left_max)
    template_left = divider_col + 1 + rng.randint(1, template_left_max)
    band_top = rng.randint(above_gap, total_h - band_h - below_gap)
    template_top = rng.randint(0, total_h - template_h)

    x = band_left
    for width, color in zip(block_widths, colors):
        grid[band_top : band_top + band_h, x : x + width] = color
        x += width

    above_row = band_top - above_gap
    below_row = band_top + band_h - 1 + below_gap
    x = band_left
    prev_row = above_row if pattern[0] else below_row
    for idx, (width, up) in enumerate(zip(block_widths, pattern)):
        row = above_row if up else below_row
        if idx > 0 and row != prev_row:
            grid[min(prev_row, row) : max(prev_row, row) + 1, x] = PATH
        grid[row, x : x + width] = PATH
        prev_row = row
        x += width

    _paint_nested_template(grid, template_top, template_left, template_h, template_w, template_colors)

    if np.count_nonzero(grid == PATH) == 0:
        return None

    out = main(grid)
    if not np.any(out != grid):
        return None
    if not np.array_equal(out[:, : divider_col + 1], grid[:, : divider_col + 1]):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
