from __future__ import annotations

import random

import numpy as np


BG = 0
SPRITE = 5
BAND_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GENERATION_ATTEMPTS = 300


def _mask_from_lines(lines: list[str]) -> np.ndarray:
    mask = np.array([[ch == "1" for ch in row] for row in lines], dtype=bool)
    assert mask.ndim == 2 and mask.shape[0] % 2 == 1 and mask.shape[1] % 2 == 1
    assert bool(mask[mask.shape[0] // 2, mask.shape[1] // 2])
    return mask


SHAPE_LIBRARY = [
    _mask_from_lines(["111", "010", "010"]),
    _mask_from_lines(["110", "111", "010"]),
    _mask_from_lines(["010", "111", "011"]),
    _mask_from_lines(["11110", "00100", "00111"]),
    _mask_from_lines(["11111", "00100", "00110"]),
    _mask_from_lines(["111", "100", "111", "001", "001"]),
    _mask_from_lines(["010", "110", "111", "010", "011"]),
    _mask_from_lines(["00100", "01110", "11111", "00110", "00111"]),
    _mask_from_lines(["11111", "00100", "01110", "00100", "00111"]),
    _mask_from_lines(["11100", "00100", "11111", "00100", "00111"]),
]


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc), h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    m = np.asarray(mask, dtype=np.uint8)
    return tuple(map(int, m.shape)), tuple(map(int, m.ravel()))


def _shape_key_from_coords(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return _mask_key(mask)


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.asarray(out, dtype=bool)


def _detect_bands(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    row_bands: list[tuple[int, int]] = []
    col_bands: list[tuple[int, int]] = []

    for r in range(h):
        vals = np.unique(g[r, :])
        if vals.size == 1 and int(vals[0]) != BG:
            row_bands.append((int(r), int(vals[0])))
    for c in range(w):
        vals = np.unique(g[:, c])
        if vals.size == 1 and int(vals[0]) != BG:
            col_bands.append((int(c), int(vals[0])))

    band_mask = np.zeros_like(g, dtype=bool)
    for r, _color in row_bands:
        band_mask[r, :] = True
    for c, _color in col_bands:
        band_mask[:, c] = True
    return row_bands, col_bands, band_mask


def _extract_sprites(grid: np.ndarray) -> list[dict[str, object]]:
    row_bands, col_bands, band_mask = _detect_bands(grid)
    assert row_bands or col_bands, "expected at least one full-row or full-column band"

    sprite_mask = (grid != BG) & (~band_mask)
    sprite_colors = sorted(set(map(int, grid[sprite_mask])))
    assert len(sprite_colors) == 1, "all sprites must share one color"
    assert sprite_colors[0] == SPRITE, "sprites are expected to be gray"

    sprites: list[dict[str, object]] = []
    for coords in _components(sprite_mask):
        r0, c0, r1, c1 = _bbox(coords)
        sprites.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "center": ((r0 + r1) // 2, (c0 + c1) // 2),
                "key": _shape_key_from_coords(coords),
            }
        )
    assert sprites, "expected at least one sprite"
    return sprites


def _strip_exit_t(a: int, b: int, index: int) -> float | None:
    delta = float(b - a)
    if abs(delta) < 1e-9:
        return None

    lo = (index - 0.5 - a) / delta
    hi = (index + 0.5 - a) / delta
    enter = max(0.0, min(lo, hi))
    leave = min(1.0, max(lo, hi))
    eps = 1e-9
    if leave < enter - eps:
        return None
    if leave <= eps or enter >= 1.0 - eps:
        return None
    return float(leave)


def _band_color_for_pair(
    first: dict[str, object],
    second: dict[str, object],
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
) -> int:
    start = tuple(map(int, first["center"]))
    end = tuple(map(int, second["center"]))
    events: list[tuple[float, int, int]] = []

    for r, color in row_bands:
        exit_t = _strip_exit_t(start[0], end[0], r)
        if exit_t is not None:
            events.append((exit_t, 0, int(color)))
    for c, color in col_bands:
        exit_t = _strip_exit_t(start[1], end[1], c)
        if exit_t is not None:
            events.append((exit_t, 1, int(color)))

    assert events, "each pair must have at least one crossed band"
    events.sort()
    return int(events[-1][2])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair identical gray sprites by exact translated shape, then recolor each pair
    using the color of the last full-row/full-column band crossed by the line
    between their bbox-center pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_bands, col_bands, _band_mask = _detect_bands(g)
    sprites = _extract_sprites(g)

    by_shape: dict[tuple[tuple[int, int], tuple[int, ...]], list[dict[str, object]]] = {}
    for sprite in sprites:
        by_shape.setdefault(sprite["key"], []).append(sprite)

    assert all(len(group) == 2 for group in by_shape.values()), "each shape must appear exactly twice"

    out = g.copy()
    for group in by_shape.values():
        first, second = sorted(group, key=lambda sprite: tuple(map(int, sprite["center"])))
        color = _band_color_for_pair(first, second, row_bands, col_bands)
        for sprite in group:
            coords = sprite["coords"]
            out[coords[:, 0], coords[:, 1]] = color

    return out


def _segments_from_sizes(
    sizes: list[int],
    margin_before: int,
    margin_after: int,
) -> tuple[list[tuple[int, int]], list[int], int]:
    segments: list[tuple[int, int]] = []
    bands: list[int] = []
    cur = margin_before
    for i, size in enumerate(sizes):
        segments.append((cur, cur + size - 1))
        cur += size
        if i < len(sizes) - 1:
            bands.append(cur)
            cur += 1
    total = cur + margin_after
    return segments, bands, total


def _choose_shapes(pair_count: int) -> list[np.ndarray] | None:
    for _ in range(200):
        shapes: list[np.ndarray] = []
        seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
        for _pair in range(pair_count):
            for _attempt in range(100):
                mask = _transform_mask(random.choice(SHAPE_LIBRARY))
                key = _mask_key(mask)
                if key in seen:
                    continue
                seen.add(key)
                shapes.append(mask)
                break
            else:
                break
        if len(shapes) == pair_count:
            return shapes
    return None


def _slot_starts(lengths: list[int], margin_before: int, margin_after: int, gap_range: tuple[int, int]) -> tuple[list[int], int]:
    gaps = [random.randint(*gap_range) for _ in range(max(0, len(lengths) - 1))]
    starts: list[int] = []
    cur = margin_before
    for i, length in enumerate(lengths):
        starts.append(cur)
        cur += length
        if i < len(lengths) - 1:
            cur += gaps[i]
    total = cur + margin_after
    return starts, total


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = color


def _random_top(seg: tuple[int, int], size: int) -> int | None:
    lo, hi = seg
    max_top = hi - size + 1
    if max_top < lo:
        return None
    return random.randint(lo, max_top)


def _choose_cross_counts(band_count: int, pair_count: int) -> list[int]:
    if band_count == 1:
        return [1] * pair_count
    counts = [1, 2]
    while len(counts) < pair_count:
        counts.append(random.choice((1, 2)))
    random.shuffle(counts)
    return counts


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        row_bands, col_bands, band_mask = _detect_bands(grid)
        if bool(row_bands) == bool(col_bands):
            return False
        if any(abs(row_bands[i][0] - row_bands[i - 1][0]) != 1 for i in range(1, len(row_bands))):
            pass
        sprite_mask = (grid != BG) & (~band_mask)
        if np.count_nonzero(sprite_mask) == 0:
            return False
        comps = _components(sprite_mask)
        groups: dict[tuple[tuple[int, int], tuple[int, ...]], list[dict[str, object]]] = {}
        for coords in comps:
            r0, c0, r1, c1 = _bbox(coords)
            sprite = {
                "coords": coords,
                "center": ((r0 + r1) // 2, (c0 + c1) // 2),
                "key": _shape_key_from_coords(coords),
            }
            groups.setdefault(sprite["key"], []).append(sprite)
        if not groups or any(len(group) != 2 for group in groups.values()):
            return False
        crossed_counts = []
        for group in groups.values():
            first, second = sorted(group, key=lambda sprite: sprite["center"])
            events = []
            for r, color in row_bands:
                exit_t = _strip_exit_t(first["center"][0], second["center"][0], r)
                if exit_t is not None:
                    events.append((exit_t, color))
            for c, color in col_bands:
                exit_t = _strip_exit_t(first["center"][1], second["center"][1], c)
                if exit_t is not None:
                    events.append((exit_t, color))
            if not events:
                return False
            crossed_counts.append(len(events))
        out = main(grid)
        if np.array_equal(out, grid):
            return False
        if len(row_bands) + len(col_bands) >= 2 and max(crossed_counts) < 2:
            return False
        return True
    except AssertionError:
        return False


def _build_horizontal_case(pair_count: int, band_count: int, shapes: list[np.ndarray]) -> np.ndarray | None:
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    row_sizes = [random.randint(6, 8) for _ in range(band_count + 1)]
    row_segments, band_rows, height = _segments_from_sizes(row_sizes, top_margin, bottom_margin)

    slot_widths = [mask.shape[1] + random.randint(1, 2) for mask in shapes]
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    slot_lefts, width = _slot_starts(slot_widths, left_margin, right_margin, (2, 3))
    if height > 30 or width > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    band_colors = random.sample(BAND_COLORS, band_count)
    for r, color in zip(band_rows, band_colors):
        grid[r, :] = int(color)

    cross_counts = _choose_cross_counts(band_count, pair_count)
    for i, (mask, slot_left, slot_width, cross_count) in enumerate(zip(shapes, slot_lefts, slot_widths, cross_counts)):
        h, w = mask.shape
        start_channel = random.randint(0, band_count - cross_count)
        end_channel = start_channel + cross_count

        top_a = _random_top(row_segments[start_channel], h)
        top_b = _random_top(row_segments[end_channel], h)
        if top_a is None or top_b is None:
            return None

        left_lo = slot_left
        left_hi = slot_left + slot_width - w
        if left_hi < left_lo:
            return None

        if random.random() < 0.7:
            left_a = left_b = random.randint(left_lo, left_hi)
        else:
            left_a = random.randint(left_lo, left_hi)
            left_b = random.randint(left_lo, left_hi)

        _paint_mask(grid, mask, top_a, left_a, SPRITE)
        _paint_mask(grid, mask, top_b, left_b, SPRITE)

    return grid


def _build_vertical_case(pair_count: int, band_count: int, shapes: list[np.ndarray]) -> np.ndarray | None:
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    col_sizes = [random.randint(6, 8) for _ in range(band_count + 1)]
    col_segments, band_cols, width = _segments_from_sizes(col_sizes, left_margin, right_margin)

    slot_heights = [mask.shape[0] + random.randint(1, 2) for mask in shapes]
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    slot_tops, height = _slot_starts(slot_heights, top_margin, bottom_margin, (2, 3))
    if height > 30 or width > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    band_colors = random.sample(BAND_COLORS, band_count)
    for c, color in zip(band_cols, band_colors):
        grid[:, c] = int(color)

    cross_counts = _choose_cross_counts(band_count, pair_count)
    for mask, slot_top, slot_height, cross_count in zip(shapes, slot_tops, slot_heights, cross_counts):
        h, w = mask.shape
        start_channel = random.randint(0, band_count - cross_count)
        end_channel = start_channel + cross_count

        left_a = _random_top(col_segments[start_channel], w)
        left_b = _random_top(col_segments[end_channel], w)
        if left_a is None or left_b is None:
            return None

        top_lo = slot_top
        top_hi = slot_top + slot_height - h
        if top_hi < top_lo:
            return None

        if random.random() < 0.7:
            top_a = top_b = random.randint(top_lo, top_hi)
        else:
            top_a = random.randint(top_lo, top_hi)
            top_b = random.randint(top_lo, top_hi)

        _paint_mask(grid, mask, top_a, left_a, SPRITE)
        _paint_mask(grid, mask, top_b, left_b, SPRITE)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        pair_count = random.randint(2, 3)
        band_count = random.choice((1, 2))
        shapes = _choose_shapes(pair_count)
        if shapes is None:
            continue

        orientation = random.choice(("horizontal", "vertical"))
        if orientation == "horizontal":
            grid = _build_horizontal_case(pair_count, band_count, shapes)
        else:
            grid = _build_vertical_case(pair_count, band_count, shapes)
        if grid is None:
            continue
        if _validate_generated(grid):
            return grid

    raise RuntimeError("failed to generate a valid puzzle input")
