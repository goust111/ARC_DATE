from __future__ import annotations

import numpy as np


BAR_COLORS = (1, 2, 3, 4, 6)
BACKGROUND = 0
BAR_COUNT = 5
MIN_BAR_HEIGHT = 3
MAX_BAR_HEIGHT = 8
MIN_SCATTER = 8
MAX_SCATTER = 15

_RNG = np.random.default_rng()
_GEN_CALLS = 0


def _bottom_run_length(grid: np.ndarray, col: int, color: int) -> int:
    height = 0
    for r in range(grid.shape[0] - 1, -1, -1):
        if int(grid[r, col]) != color:
            break
        height += 1
    return height


def _extract_bars(grid: np.ndarray) -> tuple[list[dict[str, int]], np.ndarray]:
    bottom_cols = [int(c) for c in np.where(grid[-1] != BACKGROUND)[0]]
    assert len(bottom_cols) == BAR_COUNT, "expected exactly five bottom-aligned bars"

    bars: list[dict[str, int]] = []
    bar_mask = np.zeros_like(grid, dtype=bool)
    for col in bottom_cols:
        color = int(grid[-1, col])
        assert color in BAR_COLORS, "bar color must come from the puzzle palette"
        height = _bottom_run_length(grid, col, color)
        assert MIN_BAR_HEIGHT <= height <= grid.shape[0], "invalid bar height"
        top = grid.shape[0] - height
        assert np.all(grid[top:, col] == color), "bar body must be contiguous and monochrome"
        bars.append({"col": col, "color": color, "height": height, "top": top})
        bar_mask[top:, col] = True

    bars.sort(key=lambda bar: bar["col"])
    assert len({bar["col"] for bar in bars}) == BAR_COUNT, "bar columns must be distinct"
    assert len({bar["color"] for bar in bars}) == BAR_COUNT, "bar colors must be distinct"
    return bars, bar_mask


def _extract_scatter(grid: np.ndarray, bar_mask: np.ndarray) -> tuple[np.ndarray, dict[int, int], int]:
    scatter_mask = (grid != BACKGROUND) & ~bar_mask
    scatter_coords = np.argwhere(scatter_mask)
    assert MIN_SCATTER <= len(scatter_coords) <= MAX_SCATTER, "unexpected scatter count"

    counts: dict[int, int] = {}
    for r, c in scatter_coords:
        color = int(grid[r, c])
        counts[color] = counts.get(color, 0) + 1
        r0 = max(0, int(r) - 1)
        r1 = min(grid.shape[0], int(r) + 2)
        c0 = max(0, int(c) - 1)
        c1 = min(grid.shape[1], int(c) + 2)
        neighborhood = scatter_mask[r0:r1, c0:c1]
        assert int(neighborhood.sum()) == 1, "scatter markers must remain isolated single pixels"

    assert set(counts) == set(BAR_COLORS), "scatter must mention every bar color at least once"
    max_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == max_count]
    assert len(winners) == 1, "scatter frequency winner must be unique"
    return scatter_mask, counts, winners[0]


def _analyze(grid: np.ndarray) -> tuple[list[dict[str, int]], np.ndarray, dict[int, int], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "grid must satisfy ARC size limits"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC palette ids"

    bars, bar_mask = _extract_bars(g)
    scatter_mask, counts, anchor_color = _extract_scatter(g, bar_mask)
    anchor_indices = [i for i, bar in enumerate(bars) if bar["color"] == anchor_color]
    assert len(anchor_indices) == 1, "anchor color must belong to exactly one bar"
    return bars, scatter_mask, counts, anchor_color, anchor_indices[0]


def _rotate_non_anchor(bars: list[dict[str, int]], anchor_index: int) -> list[dict[str, int]]:
    out_bars = [dict(bar) for bar in bars]
    slots = [i for i in range(len(bars)) if i != anchor_index]
    colors = [bars[i]["color"] for i in slots]
    heights = [bars[i]["height"] for i in slots]

    rotated_colors = colors[-1:] + colors[:-1]
    rotated_heights = heights[1:] + heights[:1]

    for slot, new_color, new_height in zip(slots, rotated_colors, rotated_heights):
        out_bars[slot]["color"] = int(new_color)
        out_bars[slot]["height"] = int(new_height)
        out_bars[slot]["top"] = int(bars[slot]["top"])
    return out_bars


def _render(shape: tuple[int, int], bars: list[dict[str, int]]) -> np.ndarray:
    height = shape[0]
    out = np.zeros(shape, dtype=int)
    for bar in bars:
        bar_height = int(bar["height"])
        col = int(bar["col"])
        out[height - bar_height :, col] = int(bar["color"])
    return out


def _derive_output(grid: np.ndarray, anchor_index: int | None = None) -> np.ndarray:
    bars, _scatter_mask, _counts, _anchor_color, inferred_anchor = _analyze(grid)
    used_anchor = inferred_anchor if anchor_index is None else int(anchor_index)
    transformed = _rotate_non_anchor(bars, used_anchor)
    return _render(grid.shape, transformed)


def main(grid: np.ndarray) -> np.ndarray:
    """Anchor the bar whose color is most frequent among the scattered pixels, erase the pixels, and rotate the other bars' colors and heights in opposite directions.

    Input structure:
    - Black background with exactly five distinct one-cell-wide vertical bars, all bottom-aligned.
    - Every bar is monochrome, occupies one column, and uses one of the colors {1,2,3,4,6}.
    - Additional isolated single-pixel markers of those same five colors are scattered above the bars.

    Rule:
    - Count only the scattered single pixels, not the bar pixels.
    - The unique most frequent scatter color identifies the anchored bar of the same color.
    - Erase every scatter pixel to black.
    - Keep the anchored bar unchanged, then order the other four bars by column.
    - On those four non-anchor slots, rotate colors one step right and heights one step left, while keeping each slot's original column.

    Key insight:
    - The scattered pixels are a frequency code that selects which bar is excluded from the cyclic attribute swap; changing the anchor changes the whole remaining rotation.
    """
    return _derive_output(np.asarray(grid, dtype=int))


def _variant_transform(
    bars: list[dict[str, int]],
    anchor_index: int,
    *,
    rotate_colors: bool,
    rotate_heights: bool,
) -> list[dict[str, int]]:
    out_bars = [dict(bar) for bar in bars]
    slots = [i for i in range(len(bars)) if i != anchor_index]
    colors = [bars[i]["color"] for i in slots]
    heights = [bars[i]["height"] for i in slots]
    shifted_colors = colors[-1:] + colors[:-1] if rotate_colors else colors
    shifted_heights = heights[1:] + heights[:1] if rotate_heights else heights
    for slot, color, height in zip(slots, shifted_colors, shifted_heights):
        out_bars[slot]["color"] = int(color)
        out_bars[slot]["height"] = int(height)
    return out_bars


def _sample_columns() -> list[int] | None:
    left_margin = int(_RNG.integers(1, 4))
    gaps = _RNG.integers(1, 4, size=BAR_COUNT - 1)
    cols = [left_margin]
    for gap in gaps:
        cols.append(cols[-1] + 1 + int(gap))
    right_margin = int(_RNG.integers(1, 4))
    width = cols[-1] + right_margin + 1
    if width > 30:
        return None
    return cols


def _sample_heights(anchor_index: int) -> list[int] | None:
    heights = _RNG.choice(np.arange(MIN_BAR_HEIGHT, MAX_BAR_HEIGHT + 1), size=BAR_COUNT, replace=False)
    heights = [int(v) for v in heights]
    anchor_height = heights[anchor_index]
    if anchor_height == min(heights) or anchor_height == max(heights):
        return None
    if max(h for i, h in enumerate(heights) if i != anchor_index) - anchor_height < 2:
        return None
    return heights


def _sample_scatter_counts(anchor_index: int, heights: list[int]) -> np.ndarray | None:
    counts = np.ones(BAR_COUNT, dtype=int)
    anchor_count = int(_RNG.integers(3, 5))
    counts[anchor_index] = anchor_count

    competitor_choices = sorted(
        [i for i in range(BAR_COUNT) if i != anchor_index],
        key=lambda idx: heights[idx],
        reverse=True,
    )
    competitor = competitor_choices[0]
    counts[competitor] = anchor_count - 1

    other_slots = [i for i in range(BAR_COUNT) if i not in {anchor_index, competitor}]
    _RNG.shuffle(other_slots)
    for slot in other_slots:
        if _RNG.random() < 0.7:
            counts[slot] += 1
    if _RNG.random() < 0.5 and counts.sum() < MAX_SCATTER:
        slot = int(_RNG.choice(other_slots))
        if counts[slot] < anchor_count - 1:
            counts[slot] += 1

    if not (MIN_SCATTER <= int(counts.sum()) <= MAX_SCATTER):
        return None
    if counts[anchor_index] != counts.max() or int((counts == counts.max()).sum()) != 1:
        return None

    total_pixels = counts + np.array(heights, dtype=int)
    if total_pixels[anchor_index] >= total_pixels.max():
        return None
    return counts


def _scatter_candidates(shape: tuple[int, int], bars: list[dict[str, int]]) -> list[tuple[int, int]]:
    height, width = shape
    min_bar_top = min(bar["top"] for bar in bars)
    max_row = min(min_bar_top - 3, height // 2)
    if max_row < 1:
        return []

    blocked_cols: set[int] = set()
    for bar in bars:
        for dc in (-1, 0, 1):
            cc = bar["col"] + dc
            if 0 <= cc < width:
                blocked_cols.add(cc)

    candidates = []
    for r in range(1, max_row + 1):
        for c in range(width):
            if c in blocked_cols:
                continue
            candidates.append((r, c))
    _RNG.shuffle(candidates)
    return candidates


def _place_scatter(grid: np.ndarray, bars: list[dict[str, int]], counts: np.ndarray) -> bool:
    candidates = _scatter_candidates(grid.shape, bars)
    needed = int(counts.sum())
    if len(candidates) < needed:
        return False

    chosen: list[tuple[int, int]] = []
    occupied = np.zeros_like(grid, dtype=bool)
    for bar in bars:
        occupied[grid.shape[0] - bar["height"] :, bar["col"]] = True

    for r, c in candidates:
        r0 = max(0, r - 1)
        r1 = min(grid.shape[0], r + 2)
        c0 = max(0, c - 1)
        c1 = min(grid.shape[1], c + 2)
        if occupied[r0:r1, c0:c1].any():
            continue
        chosen.append((r, c))
        occupied[r, c] = True
        if len(chosen) == needed:
            break

    if len(chosen) != needed:
        return False

    scatter_colors: list[int] = []
    for color, count in zip(BAR_COLORS, counts.tolist()):
        scatter_colors.extend([int(color)] * int(count))
    _RNG.shuffle(scatter_colors)
    for (r, c), color in zip(chosen, scatter_colors):
        grid[r, c] = int(color)
    return True


def _bars_from_params(shape: tuple[int, int], columns: list[int], colors: list[int], heights: list[int]) -> list[dict[str, int]]:
    height = shape[0]
    bars = []
    for col, color, bar_height in zip(columns, colors, heights):
        bars.append(
            {
                "col": int(col),
                "color": int(color),
                "height": int(bar_height),
                "top": int(height - bar_height),
            }
        )
    return bars


def _build_input(anchor_index: int) -> np.ndarray | None:
    columns = _sample_columns()
    if columns is None:
        return None

    heights = _sample_heights(anchor_index)
    if heights is None:
        return None

    colors = [int(v) for v in _RNG.permutation(BAR_COLORS)]
    width = columns[-1] + int(_RNG.integers(2, 5))
    height = int(_RNG.integers(16, 21))
    if width > 30 or height > 30:
        return None

    bars = _bars_from_params((height, width), columns, colors, heights)
    counts = _sample_scatter_counts(anchor_index, heights)
    if counts is None:
        return None

    grid = _render((height, width), bars)
    if not _place_scatter(grid, bars, counts):
        return None

    return grid


def _validate_sample(grid: np.ndarray) -> bool:
    bars, scatter_mask, counts, _anchor_color, anchor_index = _analyze(grid)
    out = main(grid)
    expected = _render(grid.shape, _rotate_non_anchor(bars, anchor_index))

    if not np.array_equal(out, expected):
        return False
    if np.any(out[scatter_mask] != BACKGROUND):
        return False
    if np.array_equal(grid, out):
        return False

    for wrong_anchor in range(BAR_COUNT):
        if wrong_anchor == anchor_index:
            continue
        wrong = _render(grid.shape, _rotate_non_anchor(bars, wrong_anchor))
        if np.array_equal(wrong, out):
            return False

    no_color_rotation = _render(
        grid.shape,
        _variant_transform(bars, anchor_index, rotate_colors=False, rotate_heights=True),
    )
    no_height_rotation = _render(
        grid.shape,
        _variant_transform(bars, anchor_index, rotate_colors=True, rotate_heights=False),
    )
    if np.array_equal(no_color_rotation, out) or np.array_equal(no_height_rotation, out):
        return False

    anchor_height = bars[anchor_index]["height"]
    if anchor_height == min(bar["height"] for bar in bars):
        return False
    if anchor_height == max(bar["height"] for bar in bars):
        return False

    total_pixels = {color: counts[color] + next(bar["height"] for bar in bars if bar["color"] == color) for color in BAR_COLORS}
    anchor_color = bars[anchor_index]["color"]
    if total_pixels[anchor_color] >= max(total_pixels.values()):
        return False

    return True


def generate_input() -> np.ndarray:
    global _GEN_CALLS

    for _ in range(500):
        anchor_index = _GEN_CALLS % BAR_COUNT
        _GEN_CALLS += 1
        grid = _build_input(anchor_index)
        if grid is None:
            continue
        try:
            if _validate_sample(grid):
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid frequency-anchored bar-rotation input")
