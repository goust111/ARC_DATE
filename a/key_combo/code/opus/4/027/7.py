from __future__ import annotations

from collections import Counter, deque

import numpy as np


MARKER_COLOR = 7
NOISE_COLOR = 5
PAYLOAD_COLORS = np.array([1, 2, 3, 4, 6, 8, 9], dtype=int)
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == int(color):
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": int(color), "coords": coords, "bbox": _bbox(coords), "area": len(coords)})
    return comps


def _paired_marker_blocks(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    bg = _get_bg(grid)
    candidates: list[list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        comps = _find_color_components(grid, color)
        if not comps or len(comps) % 2 != 0:
            continue

        blocks = []
        for comp in comps:
            r0, c0, r1, c1 = comp["bbox"]
            if comp["area"] != 4 or (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
                blocks = []
                break
            blocks.append(comp["bbox"])
        if not blocks:
            continue

        by_span: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        for block in blocks:
            by_span.setdefault((block[0], block[2]), []).append(block)

        pairs = []
        gap_widths = set()
        valid = True
        for span in sorted(by_span):
            row_blocks = sorted(by_span[span], key=lambda b: b[1])
            if len(row_blocks) != 2:
                valid = False
                break
            left, right = row_blocks
            gap_width = right[1] - left[3] - 1
            if gap_width <= 0:
                valid = False
                break
            pairs.append((left, right))
            gap_widths.add(gap_width)

        if valid and pairs and len(gap_widths) == 1:
            candidates.append(pairs)

    assert len(candidates) == 1, "expected exactly one marker-block pairing color"
    return candidates[0]


def _settle_region(region: np.ndarray, bg: int = 0) -> np.ndarray:
    h, w = region.shape
    out = np.full((h, w), int(bg), dtype=int)
    for c in range(w):
        vals = [int(region[r, c]) for r in range(h) if int(region[r, c]) != int(bg)]
        start = h - len(vals)
        for i, val in enumerate(vals):
            out[start + i, c] = val
    return out


def _extract_regions(grid: np.ndarray) -> list[np.ndarray]:
    pairs = _paired_marker_blocks(grid)
    regions = []
    for left, right in pairs:
        assert left[0] == right[0] and left[2] == right[2]
        region = grid[left[0] : left[2] + 1, left[3] + 1 : right[1]]
        assert region.shape[0] == 2 and region.shape[1] > 0
        regions.append(region.copy())
    widths = {region.shape[1] for region in regions}
    assert regions and len(widths) == 1
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each rectangle between row-aligned 2x2 marker blocks, settle its pixels downward, and stack the settled bands.

    Input structure:
    - The grid has a black-dominant background plus one marker color whose connected components are all 2x2 blocks.
    - These marker blocks form left/right pairs sharing the same two-row span, with a same-width interior strip between each pair.
    - The interior strips contain scattered colored payload pixels, while unrelated noise may appear outside the marker spans.

    Rule:
    - Identify the unique color whose components are all 2x2 blocks and pair those blocks by shared row span.
    - For each pair in top-to-bottom order, crop the rectangle strictly between the two blocks.
    - Within each cropped rectangle, apply independent downward gravity to every column, preserving the top-to-bottom order of non-background pixels in that column.
    - Vertically stack the settled rectangles in extraction order and discard the markers plus everything outside the cropped interiors.

    Key insight:
    - Gravity acts on each extracted strip separately, so cropping must happen before settling; otherwise pixels from different strips would incorrectly mix.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    settled = [_settle_region(region, bg=bg) for region in _extract_regions(g)]
    return np.vstack(settled)


def _sample_row_starts(rng: np.random.Generator, height: int, num_pairs: int) -> list[int] | None:
    gaps = rng.integers(1, 4, size=num_pairs - 1)
    used = 2 * num_pairs + int(gaps.sum())
    extra = height - used
    if extra < 0:
        return None
    row = int(rng.integers(0, extra + 1))
    starts = [row]
    for gap in gaps:
        row += 2 + int(gap)
        starts.append(row)
    return starts


def _make_region(rng: np.random.Generator, width: int, require_float: bool) -> np.ndarray:
    for _ in range(200):
        region = np.zeros((2, width), dtype=int)
        count = int(rng.integers(3, 7))
        cols = rng.choice(width, size=count, replace=False)
        colors = rng.choice(PAYLOAD_COLORS, size=count, replace=True)

        top_only = 0
        for idx, col in enumerate(cols):
            mode = int(rng.choice([0, 1, 2], p=[0.45, 0.4, 0.15]))
            if mode == 0:
                region[0, int(col)] = int(colors[idx])
                top_only += 1
            elif mode == 1:
                region[1, int(col)] = int(colors[idx])
            else:
                top_color = int(colors[idx])
                bottom_choices = [c for c in PAYLOAD_COLORS if int(c) != top_color]
                bottom_color = int(rng.choice(bottom_choices))
                region[0, int(col)] = top_color
                region[1, int(col)] = bottom_color

        if require_float and top_only == 0:
            continue
        if np.count_nonzero(region) < 3:
            continue

        used_colors = sorted(set(map(int, region[region != 0])))
        if len(used_colors) < 2:
            continue

        bad_block = False
        for c in range(width - 1):
            if region[0, c] != 0 and region[1, c] != 0 and region[0, c + 1] != 0 and region[1, c + 1] != 0:
                vals = {int(region[0, c]), int(region[1, c]), int(region[0, c + 1]), int(region[1, c + 1])}
                if len(vals) == 1:
                    bad_block = True
                    break
        if bad_block:
            continue
        return region
    return None


def _stacked_without_local_gravity(regions: list[np.ndarray]) -> np.ndarray:
    return np.vstack(regions)


def _stacked_with_global_gravity(regions: list[np.ndarray]) -> np.ndarray:
    return _settle_region(np.vstack(regions), bg=0)


def _add_noise(rng: np.random.Generator, grid: np.ndarray, protected: np.ndarray) -> None:
    free = np.argwhere(~protected)
    if len(free) == 0:
        return
    count = int(rng.integers(3, min(11, len(free)) + 1))
    chosen = free[rng.choice(len(free), size=count, replace=False)]
    for r, c in chosen:
        grid[int(r), int(c)] = NOISE_COLOR


def _render_layout(
    height: int,
    width: int,
    row_starts: list[int],
    left_cols: list[int],
    gap_width: int,
    regions: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((height, width), dtype=int)
    protected = np.zeros((height, width), dtype=bool)
    for row, left_col, region in zip(row_starts, left_cols, regions, strict=False):
        right_col = left_col + 2 + gap_width
        grid[row : row + 2, left_col : left_col + 2] = MARKER_COLOR
        grid[row : row + 2, left_col + 2 : right_col] = region
        grid[row : row + 2, right_col : right_col + 2] = MARKER_COLOR
        protected[row : row + 2, left_col : right_col + 2] = True
    return grid, protected


def _expected_output(regions: list[np.ndarray]) -> np.ndarray:
    return np.vstack([_settle_region(region, bg=0) for region in regions])


def _is_valid_example(grid: np.ndarray, regions: list[np.ndarray]) -> bool:
    settled_regions = [_settle_region(region, bg=0) for region in regions]
    expected = np.vstack(settled_regions)

    if not any(not np.array_equal(region, settled) for region, settled in zip(regions, settled_regions, strict=False)):
        return False
    settled_signatures = {settled.tobytes() for settled in settled_regions}
    if len(settled_signatures) != len(settled_regions):
        return False
    if np.array_equal(_stacked_without_local_gravity(regions), expected):
        return False
    if np.array_equal(_stacked_with_global_gravity(regions), expected):
        return False

    try:
        extracted = _extract_regions(grid)
        if len(extracted) != len(regions):
            return False
        if any(not np.array_equal(a, b) for a, b in zip(extracted, regions, strict=False)):
            return False
        actual = main(grid)
    except Exception:
        return False

    return np.array_equal(actual, expected)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    num_pairs = int(rng.integers(3, 5))
    gap_width = int(rng.integers(6, 11))

    min_width = gap_width + 8
    max_width = min(24, gap_width + 12)
    if min_width > max_width:
        return None
    width = int(rng.integers(min_width, max_width + 1))

    min_height = 2 * num_pairs + (num_pairs - 1) + 2
    max_height = min(26, min_height + 7)
    if min_height > max_height:
        return None
    height = int(rng.integers(min_height, max_height + 1))

    row_starts = _sample_row_starts(rng, height, num_pairs)
    if row_starts is None:
        return None

    left_limit = width - gap_width - 4
    if left_limit < 0:
        return None
    left_cols = [int(rng.integers(0, left_limit + 1)) for _ in range(num_pairs)]

    regions = []
    must_float = int(rng.integers(num_pairs))
    for idx in range(num_pairs):
        region = _make_region(rng, gap_width, require_float=(idx == must_float))
        if region is None:
            return None
        regions.append(region)

    grid, protected = _render_layout(height, width, row_starts, left_cols, gap_width, regions)
    _add_noise(rng, grid, protected)

    counts = np.bincount(grid.ravel(), minlength=10)
    if counts[0] <= max(counts[1:]):
        return None
    if not _is_valid_example(grid, regions):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-7 input")
