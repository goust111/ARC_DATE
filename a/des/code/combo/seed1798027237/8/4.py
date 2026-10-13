import random
from collections import Counter

import numpy as np


FRAME_COLOR = 5
MAX_GENERATION_ATTEMPTS = 300
PATTERN_MASKS = (
    ((1, 0), (0, 0)),
    ((0, 1), (0, 0)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1)),
    ((1, 1), (0, 0)),
    ((0, 0), (1, 1)),
    ((1, 0), (1, 0)),
    ((0, 1), (0, 1)),
    ((1, 0), (0, 1)),
    ((0, 1), (1, 0)),
    ((1, 1), (1, 0)),
    ((1, 1), (0, 1)),
    ((1, 0), (1, 1)),
    ((0, 1), (1, 1)),
)


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, int, int, int, int]:
    h, w = grid.shape
    scan_h = min(3, h)
    scan_w = min(3, w)
    for r0 in range(scan_h):
        for c0 in range(scan_w):
            frame = int(grid[r0, c0])
            if frame == bg:
                continue
            if r0 > 0 and int(grid[r0 - 1, c0]) == frame:
                continue
            if c0 > 0 and int(grid[r0, c0 - 1]) == frame:
                continue

            c1 = c0
            while c1 < w and int(grid[r0, c1]) == frame:
                c1 += 1
            r1 = r0
            while r1 < h and int(grid[r1, c0]) == frame:
                r1 += 1

            if r1 - r0 < 4 or c1 - c0 < 4:
                continue

            legend = grid[r0:r1, c0:c1]
            if not np.all(legend[0, :] == frame):
                continue
            if not np.all(legend[-1, :] == frame):
                continue
            if not np.all(legend[:, 0] == frame):
                continue
            if not np.all(legend[:, -1] == frame):
                continue
            return r0, r1, c0, c1, frame
    raise ValueError("legend not found")


def _cell_spans(separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for a, b in zip(separators, separators[1:]):
        if b - a == 3:
            spans.append((a + 1, b))
    return spans


def _parse_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, np.ndarray], np.ndarray]:
    r0, r1, c0, c1, frame = _find_legend(grid, bg)
    legend = grid[r0:r1, c0:c1]
    row_seps = [int(r) for r in range(legend.shape[0]) if np.all(legend[r, :] == frame)]
    col_seps = [int(c) for c in range(legend.shape[1]) if np.all(legend[:, c] == frame)]
    row_spans = _cell_spans(row_seps)
    col_spans = _cell_spans(col_seps)

    mapping: dict[int, np.ndarray] = {}
    for rr0, rr1 in row_spans:
        for cc0, cc1 in col_spans:
            patch = legend[rr0:rr1, cc0:cc1]
            if patch.shape != (2, 2):
                continue
            colors = set(map(int, np.unique(patch))) - {bg, frame}
            if len(colors) != 1:
                continue
            color = colors.pop()
            if not np.any(patch == bg):
                continue
            if not np.any(patch == color):
                continue
            mapping[color] = patch.copy()

    if not mapping:
        raise ValueError("legend contains no valid pattern cells")

    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0:r1, c0:c1] = True
    return mapping, legend_mask


def _band_bounds(
    shape: tuple[int, int], p1: tuple[int, int], p2: tuple[int, int]
) -> tuple[int, int, int, int]:
    h, w = shape
    r1, c1 = p1
    r2, c2 = p2
    if r1 == r2:
        left, right = sorted((c1, c2))
        top = max(0, r1 - 1)
        bottom = min(h, r1 + 1)
        return top, bottom, left, right + 1
    if c1 == c2:
        top, bottom = sorted((r1, r2))
        left = max(0, c1 - 1)
        right = min(w, c1 + 1)
        return top, bottom + 1, left, right
    raise ValueError("marker pair must share a row or column")


def _tile_pattern(pattern: np.ndarray, height: int, width: int) -> np.ndarray:
    reps_r = (height + 1) // 2
    reps_c = (width + 1) // 2
    return np.tile(pattern, (reps_r, reps_c))[:height, :width]


def _apply_bands(
    grid: np.ndarray,
    mapping: dict[int, np.ndarray],
    legend_mask: np.ndarray,
    pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    out = grid.copy()
    for color, (p1, p2) in pairs.items():
        pattern = mapping[color]
        r0, r1, c0, c1 = _band_bounds(out.shape, p1, p2)
        tiled = _tile_pattern(pattern, r1 - r0, c1 - c0)
        region = out[r0:r1, c0:c1]
        mask = ~legend_mask[r0:r1, c0:c1]
        region[mask] = tiled[mask]
        out[r0:r1, c0:c1] = region
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    mapping, legend_mask = _parse_legend(g, bg)

    pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for color in mapping:
        coords = np.argwhere((g == color) & (~legend_mask))
        if len(coords) != 2:
            continue
        p1 = (int(coords[0, 0]), int(coords[0, 1]))
        p2 = (int(coords[1, 0]), int(coords[1, 1]))
        if p1[0] != p2[0] and p1[1] != p2[1]:
            continue
        pairs[color] = (p1, p2)

    return _apply_bands(g, mapping, legend_mask, pairs)


def _legend_shapes(n_cells: int) -> list[tuple[int, int]]:
    shapes = [(1, n_cells), (n_cells, 1)]
    if n_cells == 4:
        shapes.append((2, 2))
    return shapes


def _choose_orientations(rng: random.Random, count: int) -> list[str]:
    if count == 1:
        return [rng.choice(("H", "V"))]
    if rng.random() < 0.7:
        horizontal = rng.randint(1, count - 1)
    else:
        horizontal = rng.randint(0, count)
    orientations = ["H"] * horizontal + ["V"] * (count - horizontal)
    rng.shuffle(orientations)
    return orientations


def _reserve(mask: np.ndarray, rect: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = mask.shape
    r0, r1, c0, c1 = rect
    rr0 = max(0, r0 - pad)
    rr1 = min(h, r1 + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(w, c1 + pad)
    mask[rr0:rr1, cc0:cc1] = True


def _place_pair(
    rng: random.Random,
    blocked: np.ndarray,
    legend_mask: np.ndarray,
    orientation: str,
    clipped: bool,
    legend_bounds: tuple[int, int, int, int],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = blocked.shape
    legend_r0, legend_r1, legend_c0, legend_c1 = legend_bounds

    for _ in range(200):
        if orientation == "H":
            row = 0 if clipped else rng.randrange(h)
            min_left = legend_c1 + 1 if row < legend_r1 and row >= legend_r0 else 0
            if clipped:
                min_left = max(min_left, legend_c1 + 1 if legend_r0 == 0 else 0)
            if min_left > w - 3:
                continue
            left = rng.randint(min_left, w - 3)
            right = rng.randint(left + 2, w - 1)
        else:
            col = 0 if clipped else rng.randrange(w)
            min_top = legend_r1 + 1 if col < legend_c1 and col >= legend_c0 else 0
            if clipped:
                min_top = max(min_top, legend_r1 + 1 if legend_c0 == 0 else 0)
            if min_top > h - 3:
                continue
            top = rng.randint(min_top, h - 3)
            bottom = rng.randint(top + 2, h - 1)

        if orientation == "H":
            p1 = (row, left)
            p2 = (row, right)
        else:
            p1 = (top, col)
            p2 = (bottom, col)

        r0, r1, c0, c1 = _band_bounds((h, w), p1, p2)
        if np.any(legend_mask[r0:r1, c0:c1]):
            continue
        if np.any(blocked[r0:r1, c0:c1]):
            continue
        if blocked[p1] or blocked[p2]:
            continue
        return p1, p2
    return None


def _render_pattern(mask: tuple[tuple[int, int], tuple[int, int]], bg: int, color: int) -> np.ndarray:
    arr = np.full((2, 2), bg, dtype=int)
    for r in range(2):
        for c in range(2):
            if mask[r][c]:
                arr[r, c] = color
    return arr


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice([c for c in range(10) if c != FRAME_COLOR])
    n_colors = rng.choices([2, 3, 4], weights=[4, 4, 1], k=1)[0]
    shapes = _legend_shapes(n_colors)
    rng.shuffle(shapes)

    colors = [c for c in range(10) if c not in {bg, FRAME_COLOR}]
    rng.shuffle(colors)
    marker_colors = colors[:n_colors]
    chosen_masks = rng.sample(PATTERN_MASKS, n_colors)

    legend_rows, legend_cols = shapes[0]
    legend_h = 3 * legend_rows + 1
    legend_w = 3 * legend_cols + 1
    legend_r0 = rng.randint(0, 1)
    legend_c0 = rng.randint(0, 1)

    min_h = max(12, legend_r0 + legend_h + 6)
    min_w = max(12, legend_c0 + legend_w + 6)
    if min_h > 24 or min_w > 24:
        return None
    h = rng.randint(min_h, 24)
    w = rng.randint(min_w, 24)

    grid = np.full((h, w), bg, dtype=int)
    legend_mask = np.zeros((h, w), dtype=bool)

    lr1 = legend_r0 + legend_h
    lc1 = legend_c0 + legend_w
    grid[legend_r0:lr1, legend_c0:lc1] = FRAME_COLOR
    legend_mask[legend_r0:lr1, legend_c0:lc1] = True

    mapping: dict[int, np.ndarray] = {}
    idx = 0
    for rr in range(legend_rows):
        for cc in range(legend_cols):
            color = marker_colors[idx]
            patch = _render_pattern(chosen_masks[idx], bg, color)
            sr = legend_r0 + 1 + 3 * rr
            sc = legend_c0 + 1 + 3 * cc
            grid[sr : sr + 2, sc : sc + 2] = patch
            mapping[color] = patch
            idx += 1

    blocked = np.zeros((h, w), dtype=bool)
    _reserve(blocked, (legend_r0, lr1, legend_c0, lc1), pad=1)

    orientations = _choose_orientations(rng, n_colors)
    clipped_index = rng.randrange(n_colors) if rng.random() < 0.45 else None

    pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    legend_bounds = (legend_r0, lr1, legend_c0, lc1)
    for i, color in enumerate(marker_colors):
        pair = _place_pair(
            rng,
            blocked,
            legend_mask,
            orientations[i],
            clipped=(i == clipped_index),
            legend_bounds=legend_bounds,
        )
        if pair is None:
            return None
        p1, p2 = pair
        grid[p1] = color
        grid[p2] = color
        pairs[color] = (p1, p2)
        _reserve(blocked, _band_bounds((h, w), p1, p2), pad=1)

    expected = _apply_bands(grid, mapping, legend_mask, pairs)
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
