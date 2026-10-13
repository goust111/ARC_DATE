from __future__ import annotations

import numpy as np


BG = 0
YELLOW = 4
PATTERN_COLORS = np.array([1, 2, 5, 6, 7, 8, 9], dtype=int)
TRANSFORMS = ("horizontal", "vertical", "transpose")
MAX_GENERATION_ATTEMPTS = 400
MIN_REGION_SIZE = 5
MAX_REGION_SIZE = 8


def _dominant_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)
    assert counts.size > 0
    if counts.size > 1:
        assert counts[order[-1]] > counts[order[-2]], "expected a unique dominant background"
    return int(colors[order[-1]])


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(rows) >= 2 and len(cols) >= 2:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected one full-row/full-column separator color"
    return candidates[0]


def _region_spans(lines: list[int], length: int) -> list[tuple[int, int]]:
    cuts = sorted(set(int(x) for x in lines))
    assert cuts[0] == 0 and cuts[-1] == length - 1, "separator lines must include the outer border"
    spans: list[tuple[int, int]] = []
    for a, b in zip(cuts, cuts[1:]):
        assert a + 1 <= b - 1, "separator lines must leave nonempty regions"
        spans.append((a + 1, b - 1))
    return spans


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a nonempty pattern"
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _marker_transform(markers: np.ndarray) -> str:
    assert markers.shape == (2, 2), "expected exactly two markers"
    if int(markers[0, 0]) == int(markers[1, 0]):
        return "horizontal"
    if int(markers[0, 1]) == int(markers[1, 1]):
        return "vertical"
    return "transpose"


def _paste_centered(
    region: np.ndarray,
    patch: np.ndarray,
    anchor_bbox: tuple[int, int, int, int],
    bg: int,
) -> np.ndarray:
    out = region.copy()
    pattern_mask = (region != bg) & (region != YELLOW)
    out[pattern_mask] = bg
    out[out == YELLOW] = bg

    r0, c0, r1, c1 = anchor_bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    new_h, new_w = patch.shape
    new_r0 = int(round(center_r - (new_h - 1) / 2.0))
    new_c0 = int(round(center_c - (new_w - 1) / 2.0))
    new_r1 = new_r0 + new_h - 1
    new_c1 = new_c0 + new_w - 1
    assert 0 <= new_r0 <= new_r1 < region.shape[0], "transformed pattern leaves region vertically"
    assert 0 <= new_c0 <= new_c1 < region.shape[1], "transformed pattern leaves region horizontally"
    out[new_r0 : new_r1 + 1, new_c0 : new_c1 + 1] = patch
    return out


def _transform_region(region: np.ndarray, bg: int) -> np.ndarray:
    markers = np.argwhere(region == YELLOW)
    assert markers.shape[0] == 2, "each region must contain exactly two yellow markers"

    pattern_mask = (region != bg) & (region != YELLOW)
    assert np.any(pattern_mask), "each region must contain a colored pattern"
    r0, c0, r1, c1 = _bbox(pattern_mask)

    for mr, mc in markers:
        assert not (r0 <= mr <= r1 and c0 <= mc <= c1), "markers must lie outside the pattern bbox"

    patch = region[r0 : r1 + 1, c0 : c1 + 1].copy()
    kind = _marker_transform(markers)
    if kind == "horizontal":
        transformed = np.fliplr(patch)
    elif kind == "vertical":
        transformed = np.flipud(patch)
    else:
        transformed = patch.T
    return _paste_centered(region, transformed, (r0, c0, r1, c1), bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each region's marker pair as a geometric instruction and transform the enclosed colored pattern in place.

    Input structure:
    - A black background is partitioned by one-cell-thick full green separator rows and columns into rectangular regions.
    - Every region contains one colored asymmetric pattern plus exactly two yellow marker pixels.
    - The yellow markers lie inside the same region but outside the pattern's bounding box.

    Rule:
    - Identify the unique separator color that forms complete rows and complete columns, then split the grid into the enclosed rectangular regions.
    - In each region, read the relative layout of the two yellow markers: same row means horizontal flip, same column means vertical flip, otherwise transpose.
    - Apply that transform to the pattern about its bounding-box center, keep the green separators unchanged, and erase the yellow markers to background.

    Key insight:
    - The yellow pixels are not moved objects; only their geometric relationship matters, because that relationship encodes which symmetry operation to apply to the nearby pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = _dominant_color(g)
    _separator_color, sep_rows, sep_cols = _find_separator(g, bg)
    row_spans = _region_spans(sep_rows, g.shape[0])
    col_spans = _region_spans(sep_cols, g.shape[1])

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            out[r0 : r1 + 1, c0 : c1 + 1] = _transform_region(g[r0 : r1 + 1, c0 : c1 + 1], bg)
    return out


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _is_asymmetric(mask: np.ndarray) -> bool:
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(mask, np.flipud(mask)):
        return False
    if mask.shape[0] == mask.shape[1] and np.array_equal(mask, mask.T):
        return False
    return True


def _random_mask(rng: np.random.Generator, height: int, width: int) -> np.ndarray | None:
    area = height * width
    min_cells = max(4, area // 4)
    max_cells = max(min_cells, area - 2)
    for _ in range(200):
        fill_prob = float(rng.uniform(0.30, 0.60))
        mask = rng.random((height, width)) < fill_prob
        cells = int(mask.sum())
        if not (min_cells <= cells <= max_cells):
            continue
        if not mask[0, :].any() or not mask[-1, :].any():
            continue
        if not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if not _connected(mask):
            continue
        if not _is_asymmetric(mask):
            continue
        return mask
    return None


def _sample_bbox(rng: np.random.Generator, region_h: int, region_w: int, kind: str) -> tuple[int, int, int, int] | None:
    if kind == "transpose":
        max_side = min(region_h, region_w) - 2
        if max_side < 3:
            return None
        side = int(rng.integers(3, max_side + 1))
        height = side
        width = side
    else:
        max_h = region_h - 2
        max_w = region_w - 2
        if max_h < 3 or max_w < 3:
            return None
        for _ in range(20):
            height = int(rng.integers(3, max_h + 1))
            width = int(rng.integers(3, max_w + 1))
            if max_h >= 4 and max_w >= 4 and kind in {"horizontal", "vertical"} and height != width:
                break
        else:
            height = int(rng.integers(3, max_h + 1))
            width = int(rng.integers(3, max_w + 1))

    top = int(rng.integers(1, region_h - height))
    left = int(rng.integers(1, region_w - width))
    return top, left, height, width


def _marker_positions(
    rng: np.random.Generator,
    region_h: int,
    region_w: int,
    bbox: tuple[int, int, int, int],
    kind: str,
) -> list[tuple[int, int]]:
    top, left, height, width = bbox
    bottom = top + height - 1
    right = left + width - 1

    if kind == "horizontal":
        rows = [r for r in range(region_h) if r < top or r > bottom]
        row = int(rng.choice(rows))
        cols = rng.choice(region_w, size=2, replace=False)
        return [(row, int(cols[0])), (row, int(cols[1]))]

    if kind == "vertical":
        cols = [c for c in range(region_w) if c < left or c > right]
        col = int(rng.choice(cols))
        rows = rng.choice(region_h, size=2, replace=False)
        return [(int(rows[0]), col), (int(rows[1]), col)]

    outside = [
        (r, c)
        for r in range(region_h)
        for c in range(region_w)
        if not (top <= r <= bottom and left <= c <= right)
    ]
    for _ in range(100):
        idx = rng.choice(len(outside), size=2, replace=False)
        p0 = outside[int(idx[0])]
        p1 = outside[int(idx[1])]
        if p0[0] != p1[0] and p0[1] != p1[1]:
            return [p0, p1]
    raise AssertionError("failed to place transpose markers")


def _region_patch(
    rng: np.random.Generator,
    region_h: int,
    region_w: int,
    kind: str,
) -> np.ndarray | None:
    bbox = _sample_bbox(rng, region_h, region_w, kind)
    if bbox is None:
        return None

    top, left, height, width = bbox
    mask = _random_mask(rng, height, width)
    if mask is None:
        return None

    color = int(rng.choice(PATTERN_COLORS))
    patch = np.full((region_h, region_w), BG, dtype=int)
    window = patch[top : top + height, left : left + width]
    window[mask] = color
    for r, c in _marker_positions(rng, region_h, region_w, bbox, kind):
        patch[r, c] = YELLOW
    return patch


def _balanced_transforms(count: int, rng: np.random.Generator) -> list[str]:
    kinds = [TRANSFORMS[i % len(TRANSFORMS)] for i in range(count)]
    rng.shuffle(kinds)
    return kinds


def _row_col_sizes(rng: np.random.Generator, count: int) -> list[int]:
    return [int(rng.integers(MIN_REGION_SIZE, MAX_REGION_SIZE + 1)) for _ in range(count)]


def _render_grid(
    row_sizes: list[int],
    col_sizes: list[int],
    region_patches: list[np.ndarray],
    separator: int,
) -> np.ndarray:
    height = sum(row_sizes) + len(row_sizes) + 1
    width = sum(col_sizes) + len(col_sizes) + 1
    grid = np.full((height, width), BG, dtype=int)

    sep_rows = [0]
    pos = 0
    for size in row_sizes:
        pos += size + 1
        sep_rows.append(pos)
    sep_cols = [0]
    pos = 0
    for size in col_sizes:
        pos += size + 1
        sep_cols.append(pos)

    grid[sep_rows, :] = separator
    grid[:, sep_cols] = separator

    idx = 0
    r_start = 1
    for region_h in row_sizes:
        c_start = 1
        for region_w in col_sizes:
            grid[r_start : r_start + region_h, c_start : c_start + region_w] = region_patches[idx]
            idx += 1
            c_start += region_w + 1
        r_start += region_h + 1
    return grid


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    row_count = int(rng.integers(2, 4))
    col_count = int(rng.integers(2, 4))
    row_sizes = _row_col_sizes(rng, row_count)
    col_sizes = _row_col_sizes(rng, col_count)

    region_count = row_count * col_count
    kinds = _balanced_transforms(region_count, rng)

    patches: list[np.ndarray] = []
    for idx, kind in enumerate(kinds):
        region_h = row_sizes[idx // col_count]
        region_w = col_sizes[idx % col_count]
        patch = _region_patch(rng, region_h, region_w, kind)
        if patch is None:
            return None
        patches.append(patch)

    grid = _render_grid(row_sizes, col_sizes, patches, separator=3)
    try:
        solved = main(grid)
    except Exception:
        return None

    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
