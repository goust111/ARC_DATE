from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG_COLOR = 0
COLOR_POOL = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 300
INNER_COUNT_RANGE = (1, 6)
FENCE_COUNT_RANGE = (2, 5)
PALETTE_SIZE_RANGE = (6, 8)
FENCE_SIDE_RANGE = (5, 9)  # outer side lengths


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_palette_row(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    h, _w = grid.shape
    for r in range(h - 1, -1, -1):
        cols = np.flatnonzero(grid[r] != bg)
        if len(cols) < 2:
            continue
        if not np.all(np.diff(cols) > 1):
            continue
        colors = [int(grid[r, c]) for c in cols]
        if len(set(colors)) != len(colors):
            continue
        return int(r), [int(c) for c in cols], colors
    raise AssertionError("palette row not found")


def _monochrome_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _frame_from_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict | None:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0 = min(rows)
    r1 = max(rows)
    c0 = min(cols)
    c1 = max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    color = int(grid[coords[0][0], coords[0][1]])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border_cells = 2 * (height + width) - 4
    if len(coords) != border_cells:
        return None
    if not np.all(patch[0, :] == color):
        return None
    if not np.all(patch[-1, :] == color):
        return None
    if not np.all(patch[:, 0] == color):
        return None
    if not np.all(patch[:, -1] == color):
        return None
    if height > 2 and width > 2 and np.any(patch[1:-1, 1:-1] == color):
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "color": int(color),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each closed fence according to how many colored pixels it encloses.

    The bottom palette row provides the color lookup: the N-th non-background
    palette pixel from the left gives the recolor target for a fence containing
    N interior colored pixels.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    palette_row, _palette_cols, palette_colors = _find_palette_row(g, bg)

    upper = g[:palette_row, :]
    frames: list[dict] = []
    for comp in _monochrome_components(upper, bg):
        frame = _frame_from_component(upper, comp)
        if frame is not None:
            frames.append(frame)

    assert frames, "no fence frames found"

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        count = int(np.count_nonzero(inner != bg))
        assert 1 <= count <= len(palette_colors), "interior count outside palette range"
        target = int(palette_colors[count - 1])

        out[r0, c0 : c1 + 1] = target
        out[r1, c0 : c1 + 1] = target
        out[r0 : r1 + 1, c0] = target
        out[r0 : r1 + 1, c1] = target
        out[r0 + 1 : r1, c0 + 1 : c1] = bg

    return out


def _compositions(total: int, parts: int) -> list[int]:
    if total <= 0:
        return [0] * parts
    probs = [1.0 / parts] * parts
    return [int(x) for x in np.random.multinomial(int(total), probs)]


def _palette_positions(width: int, count: int) -> list[int] | None:
    min_span = 2 * count + 1
    if width < min_span:
        return None
    extra = width - min_span
    gaps = _compositions(extra, count + 1)
    cols: list[int] = []
    pos = 1 + gaps[0]
    cols.append(int(pos))
    for i in range(1, count):
        pos += 2 + gaps[i]
        cols.append(int(pos))
    return cols


def _inner_capacity(inner_h: int, inner_w: int) -> int:
    best = 0
    for ro in (0, 1):
        for co in (0, 1):
            rows = len(range(ro, inner_h, 2))
            cols = len(range(co, inner_w, 2))
            best = max(best, rows * cols)
    return best


def _frame_size_for_count(count: int) -> tuple[int, int]:
    choices: list[tuple[int, int]] = []
    for h in range(FENCE_SIDE_RANGE[0], FENCE_SIDE_RANGE[1] + 1):
        for w in range(FENCE_SIDE_RANGE[0], FENCE_SIDE_RANGE[1] + 1):
            if _inner_capacity(h - 2, w - 2) >= count:
                choices.append((int(h), int(w)))
    if not choices:
        raise AssertionError("no frame sizes available")
    return random.choice(choices)


def _sample_inner_points(inner_h: int, inner_w: int, count: int) -> list[tuple[int, int]] | None:
    candidate_sets: list[list[tuple[int, int]]] = []
    for ro in (0, 1):
        for co in (0, 1):
            cells = [(r, c) for r in range(ro, inner_h, 2) for c in range(co, inner_w, 2)]
            if len(cells) >= count:
                candidate_sets.append(cells)
    if not candidate_sets:
        return None
    cells = random.choice(candidate_sets)
    return [(int(r), int(c)) for r, c in random.sample(cells, count)]


def _partition_indices(count: int, band_count: int) -> list[list[int]]:
    order = list(range(count))
    random.shuffle(order)
    bands = [[] for _ in range(band_count)]
    for i, idx in enumerate(order):
        bands[i % band_count].append(idx)
    for band in bands:
        random.shuffle(band)
    return bands


def _expected_output(
    grid: np.ndarray,
    frames: list[dict],
    palette_colors: list[int],
    bg: int,
) -> np.ndarray:
    out = grid.copy()
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        target = int(palette_colors[int(frame["count"]) - 1])
        out[r0, c0 : c1 + 1] = target
        out[r1, c0 : c1 + 1] = target
        out[r0 : r1 + 1, c0] = target
        out[r0 : r1 + 1, c1] = target
        out[r0 + 1 : r1, c0 + 1 : c1] = bg
    return out


def _build_attempt() -> tuple[np.ndarray, list[dict], list[int]] | None:
    bg = BG_COLOR
    fence_count = random.randint(*FENCE_COUNT_RANGE)
    used_max = random.randint(2, INNER_COUNT_RANGE[1])
    palette_size = random.randint(max(PALETTE_SIZE_RANGE[0], used_max), PALETTE_SIZE_RANGE[1])

    fence_color = random.choice(COLOR_POOL)
    usable_colors = [c for c in COLOR_POOL if c != fence_color]
    palette_colors = random.sample(usable_colors, palette_size)

    counts = [random.randint(1, used_max) for _ in range(fence_count)]
    counts[random.randrange(fence_count)] = used_max
    if fence_count >= 2 and len(set(counts)) == 1:
        idx = random.randrange(fence_count)
        counts[idx] = 1 if counts[idx] != 1 else min(used_max, 2)

    frames_meta: list[dict] = []
    for count in counts:
        h, w = _frame_size_for_count(count)
        frames_meta.append({"count": int(count), "size": (int(h), int(w))})

    if fence_count <= 2:
        band_count = random.choice((1, 2))
    elif fence_count == 3:
        band_count = random.choice((1, 2))
    else:
        band_count = 2
    band_count = min(band_count, fence_count)

    bands = _partition_indices(fence_count, band_count)
    band_heights: list[int] = []
    band_widths: list[int] = []
    band_gaps: list[list[int]] = []
    for band in bands:
        sizes = [frames_meta[idx]["size"] for idx in band]
        gaps = [random.randint(2, 4) for _ in range(max(0, len(band) - 1))]
        band_gaps.append(gaps)
        band_heights.append(max(h for h, _w in sizes))
        band_widths.append(sum(w for _h, w in sizes) + sum(gaps))

    vertical_gaps = [random.randint(2, 4) for _ in range(max(0, band_count - 1))]
    top_margin = random.randint(1, 2)
    gap_to_palette = random.randint(2, 4)

    min_width = max(max(band_widths), 2 * palette_size + 1)
    width = min_width + random.randint(2, 6)
    height = top_margin + sum(band_heights) + sum(vertical_gaps) + gap_to_palette + 1
    if height < 12:
        height = 12

    palette_cols = _palette_positions(width, palette_size)
    if palette_cols is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    palette_row = height - 1
    for col, color in zip(palette_cols, palette_colors):
        grid[palette_row, int(col)] = int(color)

    band_top = top_margin
    placed_frames: list[dict] = []
    for band_idx, band in enumerate(bands):
        band_width = band_widths[band_idx]
        start_col = random.randint(1, width - band_width - 1)
        cursor = start_col
        band_height = band_heights[band_idx]
        for pos, frame_idx in enumerate(band):
            frame = frames_meta[frame_idx]
            h, w = frame["size"]
            row_offset = random.randint(0, band_height - h)
            r0 = band_top + row_offset
            c0 = cursor
            r1 = r0 + h - 1
            c1 = c0 + w - 1

            grid[r0, c0 : c1 + 1] = fence_color
            grid[r1, c0 : c1 + 1] = fence_color
            grid[r0 : r1 + 1, c0] = fence_color
            grid[r0 : r1 + 1, c1] = fence_color

            inner_h = h - 2
            inner_w = w - 2
            inner_points = _sample_inner_points(inner_h, inner_w, int(frame["count"]))
            if inner_points is None:
                return None
            inner_colors = random.sample(usable_colors, int(frame["count"]))
            for (ir, ic), color in zip(inner_points, inner_colors):
                grid[r0 + 1 + ir, c0 + 1 + ic] = int(color)

            placed_frames.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "count": int(frame["count"]),
                }
            )
            if pos < len(band_gaps[band_idx]):
                cursor += w + band_gaps[band_idx][pos]
        if band_idx < len(vertical_gaps):
            band_top += band_height + vertical_gaps[band_idx]

    return grid, placed_frames, palette_colors


def _validate_generated(grid: np.ndarray, frames: list[dict], palette_colors: list[int]) -> bool:
    expected = _expected_output(grid, frames, palette_colors, BG_COLOR)
    actual = main(grid)
    return bool(np.array_equal(expected, actual))


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        attempt = _build_attempt()
        if attempt is None:
            continue
        grid, frames, palette_colors = attempt
        if _validate_generated(grid, frames, palette_colors):
            return grid
    raise RuntimeError("failed to generate a valid input")
