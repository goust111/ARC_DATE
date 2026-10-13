from __future__ import annotations

from collections import deque
import random

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _legend_rows(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, _w = grid.shape

    top_has = bool(np.any(grid[0] != bg))
    bottom_has = bool(np.any(grid[h - 1] != bg))
    assert top_has ^ bottom_has, "legend must appear on exactly one horizontal edge"

    if top_has:
        end = 0
        while end < h and np.any(grid[end] != bg):
            end += 1
        assert end >= 2, "legend band must be at least two rows tall"
        if end < h:
            assert np.all(grid[end] == bg), "legend must be separated from frames by a blank row"
        return 0, int(end)

    start = h - 1
    while start >= 0 and np.any(grid[start] != bg):
        start -= 1
    start += 1
    assert h - start >= 2, "legend band must be at least two rows tall"
    if start > 0:
        assert np.all(grid[start - 1] == bg), "legend must be separated from frames by a blank row"
    return int(start), int(h)


def _find_legend(
    grid: np.ndarray, bg: int
) -> tuple[dict[int, tuple[np.ndarray, int]], np.ndarray]:
    r0, r1 = _legend_rows(grid, bg)
    band = grid[r0:r1, :]
    active_cols = np.flatnonzero(np.any(band != bg, axis=0))
    assert active_cols.size > 0, "legend must contain colored cells"

    c0 = int(active_cols[0])
    c1 = int(active_cols[-1]) + 1

    key_cols: list[int] = []
    for c in range(c0, c1):
        col = band[:, c]
        if int(col[0]) != bg and bool(np.all(col == col[0])):
            key_cols.append(int(c))

    assert len(key_cols) >= 2, "legend needs at least two entries"
    widths = [b - a for a, b in zip(key_cols, key_cols[1:])]
    assert len(set(widths)) == 1, "legend entries must have equal width"
    entry_w = int(widths[0])
    assert entry_w >= 2, "legend entry width must leave space for a pattern"
    assert key_cols[0] == c0, "legend must begin with a key column"
    assert key_cols[-1] + entry_w == c1, "legend columns must be fully partitioned into entries"

    mapping: dict[int, tuple[np.ndarray, int]] = {}
    for kc in key_cols:
        key_color = int(band[0, kc])
        pattern_patch = band[:, kc + 1 : kc + entry_w]
        assert pattern_patch.size > 0

        colors = {int(v) for v in np.unique(pattern_patch)}
        colors.discard(bg)
        assert len(colors) == 1, "each legend pattern must use exactly one foreground color"
        pattern_color = colors.pop()
        pattern_mask = pattern_patch == pattern_color

        assert pattern_mask.any() and (~pattern_mask).any(), "legend patterns must be binary and nontrivial"
        assert np.all(pattern_mask.any(axis=0)), "every pattern column must show foreground"
        assert np.all((~pattern_mask).any(axis=0)), "every pattern column must keep some background"
        mapping[key_color] = (pattern_mask, pattern_color)

    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0:r1, c0:c1] = True
    return mapping, legend_mask


def _find_frames(
    grid: np.ndarray, bg: int, legend_mask: np.ndarray
) -> list[tuple[int, int, int, int, int]]:
    mask = (grid != bg) & (~legend_mask)
    frames: list[tuple[int, int, int, int, int]] = []

    for comp in _components(mask):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        color = int(grid[comp[0][0], comp[0][1]])

        if any(int(grid[r, c]) != color for r, c in comp):
            continue

        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue

        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[h - 1, :] = True
        border[:, 0] = True
        border[:, w - 1] = True

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if len(comp) != int(border.sum()):
            continue
        if not np.array_equal(sub == color, border):
            continue
        if not np.all(sub[1:-1, 1:-1] == bg):
            continue
        frames.append((int(r0), int(c0), int(r1), int(c1), color))

    assert frames, "expected at least one hollow frame"
    return frames


def _tile_pattern(
    pattern_mask: np.ndarray, pattern_color: int, shape: tuple[int, int], bg: int
) -> np.ndarray:
    ih, iw = shape
    rr, cc = np.indices((ih, iw))
    tiled_mask = pattern_mask[rr % pattern_mask.shape[0], cc % pattern_mask.shape[1]]
    out = np.full((ih, iw), bg, dtype=int)
    out[tiled_mask] = int(pattern_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow frame interior with the tiled pattern selected by its border color from the legend band.

    Input structure:
    - The background is black, and a horizontal legend band sits at the top or bottom edge.
    - Each legend entry has a one-column solid color key on the left and a binary pattern on the right made from black plus one foreground color.
    - Away from the legend, several one-pixel-thick hollow rectangular frames of various sizes sit on the background.

    Rule:
    - Read the legend as a mapping from frame-border color to a pattern mask plus its foreground color.
    - For every hollow frame, find the legend entry whose key color matches the frame border color.
    - Tile that entry's binary pattern across the frame interior, aligned to the interior's top-left corner, while leaving the frame border and the legend unchanged.

    Key insight:
    - The frame color is only an index into the legend; the actual fill uses the legend pattern's own foreground color and repeats spatially by tiling.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    legend, legend_mask = _find_legend(g, bg)
    frames = _find_frames(g, bg, legend_mask)

    out = g.copy()
    for r0, c0, r1, c1, border_color in frames:
        assert border_color in legend, "every frame border color must appear in the legend"
        pattern_mask, pattern_color = legend[border_color]
        tiled = _tile_pattern(pattern_mask, pattern_color, (r1 - r0 - 1, c1 - c0 - 1), bg)
        out[r0 + 1 : r1, c0 + 1 : c1] = tiled
    return out


# --------------- generator ---------------

BG = 0
ALL_COLORS = tuple(range(1, 10))
ENTRY_COUNT_RANGE = (2, 4)
BLOCK_H_RANGE = (3, 5)
FRAME_OUTER_RANGE = (4, 9)
GRID_LIMIT = 30
MIN_GRID_ROWS = 14
MIN_GRID_COLS = 16
MAX_FRAME_COUNT = 5
PLACEMENT_TRIES = 150
GEN_ATTEMPTS = 400


def _pattern_valid(mask: np.ndarray) -> bool:
    return bool(
        mask.any()
        and (~mask).any()
        and np.all(mask.any(axis=0))
        and np.all((~mask).any(axis=0))
        and len({tuple(row.tolist()) for row in mask}) >= 2
    )


def _structured_patterns(h: int, w: int) -> list[np.ndarray]:
    rr, cc = np.indices((h, w))
    candidates = [
        (rr + cc) % 2 == 0,
        (rr + cc) % 2 == 1,
        (rr + 2 * cc) % 3 == 0,
        (2 * rr + cc) % 3 == 0,
        (rr - cc) % 3 == 0,
        (rr + cc + 1) % 3 == 0,
        (rr + 2 * cc) % 4 <= 1,
        (2 * rr + cc) % 4 <= 1,
    ]

    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    for mask in candidates:
        if not _pattern_valid(mask):
            continue
        sig = mask.astype(np.uint8).tobytes()
        if sig in seen:
            continue
        seen.add(sig)
        out.append(mask.copy())
    return out


def _sample_patterns(count: int, h: int, w: int, rng: random.Random) -> list[np.ndarray]:
    patterns = _structured_patterns(h, w)
    seen = {p.astype(np.uint8).tobytes() for p in patterns}

    for _ in range(300):
        if len(patterns) >= count:
            break
        density = rng.uniform(0.35, 0.65)
        mask = np.array(
            [[rng.random() < density for _ in range(w)] for _ in range(h)],
            dtype=bool,
        )
        if not _pattern_valid(mask):
            continue
        sig = mask.astype(np.uint8).tobytes()
        if sig in seen:
            continue
        seen.add(sig)
        patterns.append(mask)

    assert len(patterns) >= count, "failed to create enough distinct legend patterns"
    rng.shuffle(patterns)
    return [patterns[i].copy() for i in range(count)]


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)


def _try_place_frame(
    occupied: np.ndarray,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    fh: int,
    fw: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for r0 in range(row_lo, row_hi - fh + 2):
        for c0 in range(col_lo, col_hi - fw + 2):
            r1 = r0 + fh - 1
            c1 = c0 + fw - 1
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(occupied.shape[0] - 1, r1 + 1)
            cc1 = min(occupied.shape[1] - 1, c1 + 1)
            if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            candidates.append((int(r0), int(c0)))
    if not candidates:
        return None
    return rng.choice(candidates)


def _frame_size_cap(frame_count: int, avail_h: int, avail_w: int) -> tuple[int, int]:
    size_cap = 11 - frame_count
    max_h = min(FRAME_OUTER_RANGE[1], avail_h, size_cap)
    max_w = min(FRAME_OUTER_RANGE[1], avail_w, size_cap + 1)
    return max(int(FRAME_OUTER_RANGE[0]), int(max_h)), max(int(FRAME_OUTER_RANGE[0]), int(max_w))


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    entry_count = rng.randint(ENTRY_COUNT_RANGE[0], ENTRY_COUNT_RANGE[1])
    block_h = rng.randint(BLOCK_H_RANGE[0], BLOCK_H_RANGE[1])
    block_w = block_h
    pattern_w = block_w - 1
    legend_w = entry_count * block_w

    min_cols = max(MIN_GRID_COLS, legend_w + 1)
    cols = rng.randint(min_cols, GRID_LIMIT)
    rows = rng.randint(max(MIN_GRID_ROWS, block_h + 7), GRID_LIMIT)

    legend_side_top = rng.random() < 0.5
    legend_c0 = rng.randint(0, cols - legend_w)
    legend_r0 = 0 if legend_side_top else rows - block_h

    frame_count = entry_count if rng.random() < 0.5 else rng.randint(
        entry_count + 1, min(MAX_FRAME_COUNT, entry_count + 2)
    )

    color_pool = list(ALL_COLORS)
    rng.shuffle(color_pool)
    key_colors = color_pool[:entry_count]
    pattern_colors = color_pool[entry_count : 2 * entry_count]
    patterns = _sample_patterns(entry_count, block_h, pattern_w, rng)

    grid = np.full((rows, cols), BG, dtype=int)
    expected = np.full((rows, cols), BG, dtype=int)

    legend_items: dict[int, tuple[np.ndarray, int]] = {}
    for i in range(entry_count):
        c0 = legend_c0 + i * block_w
        key_color = key_colors[i]
        pattern_color = pattern_colors[i]
        pattern_mask = patterns[i]

        grid[legend_r0 : legend_r0 + block_h, c0] = key_color
        expected[legend_r0 : legend_r0 + block_h, c0] = key_color
        grid[legend_r0 : legend_r0 + block_h, c0 + 1 : c0 + block_w][pattern_mask] = pattern_color
        expected[legend_r0 : legend_r0 + block_h, c0 + 1 : c0 + block_w][pattern_mask] = pattern_color
        legend_items[key_color] = (pattern_mask, pattern_color)

    if legend_side_top:
        row_lo = block_h + 1
        row_hi = rows - 2
    else:
        row_lo = 1
        row_hi = rows - block_h - 2
    col_lo = 1
    col_hi = cols - 2

    avail_h = row_hi - row_lo + 1
    avail_w = col_hi - col_lo + 1
    if avail_h < FRAME_OUTER_RANGE[0] or avail_w < FRAME_OUTER_RANGE[0]:
        return None

    max_h, max_w = _frame_size_cap(frame_count, avail_h, avail_w)
    if max_h < FRAME_OUTER_RANGE[0] or max_w < FRAME_OUTER_RANGE[0]:
        return None

    occupied = np.zeros((rows, cols), dtype=bool)
    occupied[legend_r0 : legend_r0 + block_h, legend_c0 : legend_c0 + legend_w] = True

    frame_colors = key_colors.copy()
    while len(frame_colors) < frame_count:
        frame_colors.append(rng.choice(key_colors))
    rng.shuffle(frame_colors)

    placed_any_repeated = len(frame_colors) > len(set(frame_colors))
    sizes: list[tuple[int, int, int]] = []
    for color in frame_colors:
        fh = rng.randint(FRAME_OUTER_RANGE[0], max_h)
        fw = rng.randint(FRAME_OUTER_RANGE[0], max_w)
        sizes.append((fh * fw, fh, fw))
    sizes.sort(reverse=True)

    ordered_colors = frame_colors[:]
    rng.shuffle(ordered_colors)
    if placed_any_repeated:
        ordered_colors.sort(key=lambda col: frame_colors.count(col), reverse=True)

    for idx, (_area, fh, fw) in enumerate(sizes):
        border_color = ordered_colors[idx]
        placed = None
        for _ in range(PLACEMENT_TRIES):
            placed = _try_place_frame(occupied, row_lo, row_hi, col_lo, col_hi, fh, fw, rng)
            if placed is not None:
                break
        if placed is None:
            return None

        r0, c0 = placed
        r1 = r0 + fh - 1
        c1 = c0 + fw - 1
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(rows - 1, r1 + 1)
        cc1 = min(cols - 1, c1 + 1)
        occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

        _draw_frame(grid, r0, c0, r1, c1, border_color)
        _draw_frame(expected, r0, c0, r1, c1, border_color)
        pattern_mask, pattern_color = legend_items[border_color]
        expected[r0 + 1 : r1, c0 + 1 : c1] = _tile_pattern(
            pattern_mask, pattern_color, (fh - 2, fw - 2), BG
        )

    vals, counts = np.unique(grid, return_counts=True)
    if int(vals[counts.argmax()]) != BG:
        return None

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        built = _build_case(rng)
        if built is not None:
            grid, _expected = built
            return grid
    raise RuntimeError("failed to generate a valid legend-to-pattern tiling task")
