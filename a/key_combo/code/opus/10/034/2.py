from __future__ import annotations

from functools import lru_cache
from itertools import combinations, product
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


BAR_COLORS = (Color.TEAL, Color.RED)
MARKER_POOL = (Color.BLUE, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.MAROON)
HEIGHT_VALUES = tuple(range(2, 8))
GEN_RETRY_LIMIT = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _extract_bars(grid: np.ndarray) -> list[dict[str, int]]:
    h, _w = grid.shape
    bars: list[dict[str, int]] = []
    for color in BAR_COLORS:
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox_coords(coords)
            area = int(coords.shape[0])
            assert c0 == c1, "bars must be one column wide"
            assert r1 == h - 1, "bars must be bottom-aligned"
            assert area == r1 - r0 + 1, "bars must be solid vertical segments"
            bars.append({"color": int(color), "col": int(c0), "height": int(area)})
    bars.sort(key=lambda item: item["col"])
    assert 3 <= len(bars) <= 5, "expected 3 to 5 bars"
    assert len({bar["col"] for bar in bars}) == len(bars), "bars must be in distinct columns"
    assert any(bar["color"] == Color.TEAL for bar in bars), "need at least one teal bar"
    return bars


def _extract_template_mask(grid: np.ndarray) -> np.ndarray:
    comps = _components(grid == Color.GREY)
    assert len(comps) == 1, "expected exactly one grey template"
    bbox = _bbox_coords(comps[0])
    mask = _crop_bbox(grid == Color.GREY, bbox)
    assert 3 <= mask.shape[0] <= 4 and 3 <= mask.shape[1] <= 4, "template bbox must be 3x3 to 4x4"
    assert int(mask.sum()) >= 4, "template must contain several pixels"
    return mask


def _extract_markers(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    reserved = {bg, Color.GREY, Color.TEAL, Color.RED}
    markers: list[tuple[int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in reserved):
        coords = np.argwhere(grid == color)
        assert coords.shape[0] == 1, "each marker color must appear exactly once"
        r, c = map(int, coords[0])
        markers.append((r, c, int(color)))
    assert len(markers) >= 2, "need at least two markers"
    assert len({c for _r, c, _color in markers}) == 1, "markers must lie in one vertical column"
    markers.sort(key=lambda item: item[0])
    return markers


def _render_output(mask: np.ndarray, marker_colors: list[int], bg: int) -> np.ndarray:
    th, tw = mask.shape
    out = np.full((th * len(marker_colors), tw), int(bg), dtype=int)
    for i, color in enumerate(marker_colors):
        patch = np.full(mask.shape, int(bg), dtype=int)
        patch[mask] = int(color)
        out[i * th : (i + 1) * th, :] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use the left bar arithmetic to choose how many top-to-bottom color markers recolor and stack the grey template.

    Input structure:
    - A black background contains a left group of 3 to 5 single-column vertical bars and a separate right group.
    - Teal bars are positive operands and red bars are negative operands; every bar is solid and bottom-aligned.
    - The right group contains one grey multi-pixel template shape plus a vertical column of distinct single-pixel markers.

    Rule:
    - Compute N as the total teal bar height minus the total red bar height.
    - Crop the grey template to its bounding box and use the grey cells as a mask.
    - Sort the marker pixels from top to bottom, keep only the first N markers, and ignore the rest.
    - Recolor one template copy with each kept marker color and concatenate the copies vertically without gaps.

    Key insight:
    - The bar arithmetic does not draw a bar directly; it produces a count that controls how many ordered marker colors drive template stamping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    bars = _extract_bars(g)
    n = sum(bar["height"] if bar["color"] == Color.TEAL else -bar["height"] for bar in bars)
    assert n >= 1, "bar arithmetic must produce a positive count"

    mask = _extract_template_mask(g)
    markers = _extract_markers(g, bg)
    assert len(markers) > n, "there must be redundant markers beyond the selected prefix"
    return _render_output(mask, [color for _r, _c, color in markers[:n]], bg)


@lru_cache(maxsize=None)
def _bar_candidates(bar_count: int, n: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    out: list[tuple[tuple[int, int], ...]] = []
    for heights in combinations(HEIGHT_VALUES, bar_count):
        for colors in product(BAR_COLORS, repeat=bar_count):
            if Color.TEAL not in colors:
                continue
            diff = sum(h if color == Color.TEAL else -h for h, color in zip(heights, colors))
            if diff == n:
                out.append(tuple((int(color), int(height)) for height, color in zip(heights, colors)))
    return tuple(out)


def _sample_bar_specs() -> tuple[list[tuple[int, int]], int]:
    while True:
        bar_count = random.randint(3, 5)
        n = random.randint(1, 4)
        candidates = list(_bar_candidates(bar_count, n))
        if not candidates:
            continue
        with_red = [candidate for candidate in candidates if any(color == Color.RED for color, _h in candidate)]
        all_teal = [candidate for candidate in candidates if all(color == Color.TEAL for color, _h in candidate)]
        if with_red and random.random() < 0.75:
            chosen = list(random.choice(with_red))
        elif all_teal:
            chosen = list(random.choice(all_teal))
        else:
            chosen = list(random.choice(candidates))
        random.shuffle(chosen)
        return chosen, n


def _gen_template_mask(height: int, width: int) -> np.ndarray:
    for _ in range(200):
        target = random.randint(max(4, height + width - 1), height * width - 1)
        cells = {(random.randrange(height), random.randrange(width))}
        while len(cells) < target:
            base_r, base_c = random.choice(tuple(cells))
            nbrs = [
                (nr, nc)
                for nr, nc in _neighbors4(base_r, base_c)
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells
            ]
            if nbrs:
                cells.add(random.choice(nbrs))
        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not mask.all() and mask.sum() >= 4 and np.all(mask.any(axis=0)) and np.all(mask.any(axis=1)):
            return mask
    raise RuntimeError("failed to generate template mask")


def _choose_layout(
    bars: list[tuple[int, int]],
    template_shape: tuple[int, int],
    marker_count: int,
) -> tuple[int, int, list[int], int, int, list[int], int, int]:
    th, tw = template_shape
    max_bar_height = max(height for _color, height in bars)

    left_pad = random.randint(1, 2)
    bar_steps = [random.randint(2, 3) for _ in range(len(bars) - 1)]
    bar_cols = [left_pad]
    for step in bar_steps:
        bar_cols.append(bar_cols[-1] + step)
    left_end = bar_cols[-1]

    zone_gap = random.randint(2, 4)
    local_gap = random.randint(2, 4)
    right_side_markers = random.choice((False, True))
    right_start = left_end + zone_gap + 1
    if right_side_markers:
        template_col = right_start
        marker_col = template_col + tw + local_gap
        right_end = marker_col
    else:
        marker_col = right_start
        template_col = marker_col + 1 + local_gap
        right_end = template_col + tw - 1

    right_pad = random.randint(1, 3)
    width = right_end + right_pad + 1

    marker_steps = [random.randint(2, 3) for _ in range(marker_count - 1)]
    marker_span = 1 + sum(marker_steps)
    min_height = max(max_bar_height + 1, th, marker_span)
    height = random.randint(max(12, min_height), min(22, max(12, min_height + 6)))

    template_top = random.randint(0, height - th)
    marker_rows = [random.randint(0, height - marker_span)]
    for step in marker_steps:
        marker_rows.append(marker_rows[-1] + step)

    assert height <= 30 and width <= 30
    return height, width, bar_cols, template_top, template_col, marker_rows, marker_col, left_end


def _place_bars(grid: np.ndarray, specs: list[tuple[int, int]], cols: list[int]) -> list[dict[str, int]]:
    h, _w = grid.shape
    meta: list[dict[str, int]] = []
    for (color, height), col in zip(specs, cols):
        grid[h - height : h, col] = int(color)
        meta.append({"color": int(color), "height": int(height), "col": int(col)})
    return meta


def _expected_from_construction(mask: np.ndarray, marker_colors: list[int], n: int) -> np.ndarray:
    return _render_output(mask, marker_colors[:n], Color.BLACK)


def _mutate_bar(grid: np.ndarray, bars: list[dict[str, int]], n: int) -> np.ndarray | None:
    out = grid.copy()
    h = grid.shape[0]
    if n > 1:
        for bar in bars:
            if bar["color"] == Color.TEAL and bar["height"] > 2:
                top = h - bar["height"]
                out[top, bar["col"]] = Color.BLACK
                return out
        for bar in bars:
            if bar["color"] == Color.RED and bar["height"] < 7:
                top = h - bar["height"] - 1
                if top >= 0 and out[top, bar["col"]] == Color.BLACK:
                    out[top:h, bar["col"]] = Color.RED
                    return out
    else:
        for bar in bars:
            if bar["color"] == Color.TEAL and bar["height"] < 7:
                top = h - bar["height"] - 1
                if top >= 0 and out[top, bar["col"]] == Color.BLACK:
                    out[top:h, bar["col"]] = Color.TEAL
                    return out
        for bar in bars:
            if bar["color"] == Color.RED and bar["height"] > 2:
                top = h - bar["height"]
                out[top, bar["col"]] = Color.BLACK
                return out
    return None


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    bars: list[dict[str, int]],
    marker_rows: list[int],
    marker_col: int,
    marker_colors: list[int],
    n: int,
) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    spare_colors = [color for color in MARKER_POOL if color not in marker_colors]
    if not spare_colors:
        return False
    spare = spare_colors[0]

    ignored = grid.copy()
    ignored[marker_rows[n], marker_col] = spare
    if not np.array_equal(main(ignored), expected):
        return False

    selected = grid.copy()
    selected[marker_rows[0], marker_col] = spare
    if np.array_equal(main(selected), expected):
        return False

    mutated = _mutate_bar(grid, bars, n)
    if mutated is None:
        return False
    try:
        mutated_out = main(mutated)
    except Exception:
        return False
    if np.array_equal(mutated_out, expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    specs, n = _sample_bar_specs()
    marker_count = random.randint(n + 1, len(MARKER_POOL) - 1)
    marker_colors = random.sample(MARKER_POOL, marker_count)

    template_h = random.randint(3, 4)
    template_w = random.randint(3, 4)
    mask = _gen_template_mask(template_h, template_w)

    try:
        height, width, bar_cols, template_top, template_col, marker_rows, marker_col, left_end = _choose_layout(
            specs, mask.shape, marker_count
        )
    except ValueError:
        return None

    if width > 30 or height > 30:
        return None
    if marker_col <= left_end + 1 or template_col <= left_end + 1:
        return None

    grid = np.full((height, width), Color.BLACK, dtype=int)
    bars = _place_bars(grid, specs, bar_cols)

    patch = np.full(mask.shape, Color.BLACK, dtype=int)
    patch[mask] = Color.GREY
    grid[template_top : template_top + template_h, template_col : template_col + template_w] = patch

    for row, color in zip(marker_rows, marker_colors):
        if grid[row, marker_col] != Color.BLACK:
            return None
        grid[row, marker_col] = int(color)

    if np.any(grid[:, : left_end + 1] == Color.GREY):
        return None

    expected = _expected_from_construction(mask, marker_colors, n)
    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return None
    if not _semantic_checks(grid, expected, bars, marker_rows, marker_col, marker_colors, n):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
