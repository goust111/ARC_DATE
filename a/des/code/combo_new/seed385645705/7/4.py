import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    winners = values[counts == counts.max()]
    assert winners.size == 1, "background color must be uniquely most common"
    return int(winners[0])


def _row_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    cols = np.flatnonzero(mask)
    if cols.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(cols) > 1)
    starts = np.r_[cols[0], cols[breaks + 1]]
    ends = np.r_[cols[breaks], cols[-1]]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _selector_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    height, _width = map(int, grid.shape)
    candidates = []
    for raw_color in np.unique(grid):
        color = int(raw_color)
        if color == bg:
            continue

        rows = []
        starts = []
        ends = []
        ok = True
        for r in range(height):
            runs = _row_runs(grid[r] == color)
            if len(runs) > 1:
                ok = False
                break
            if runs:
                rows.append(r)
                starts.append(runs[0][0])
                ends.append(runs[0][1])
        if not ok or len(rows) < 2:
            continue

        gaps = [b - a for a, b in zip(rows, rows[1:])]
        if len(set(gaps)) != 1:
            continue
        band_h = gaps[0]
        if band_h < 2 or height % band_h != 0:
            continue
        n_bands = height // band_h
        mid = rows[0]
        if mid >= band_h or mid != band_h // 2:
            continue
        expected_rows = [mid + i * band_h for i in range(n_bands)]
        if rows != expected_rows:
            continue

        right_side_marker = True
        for band_i, start in enumerate(starts):
            r0 = band_i * band_h
            r1 = r0 + band_h
            other = (grid[r0:r1] != bg) & (grid[r0:r1] != color)
            if not np.any(other):
                right_side_marker = False
                break
            other_cols = np.argwhere(other)[:, 1]
            if int(other_cols.max()) >= int(start):
                right_side_marker = False
                break
        if not right_side_marker:
            continue

        candidates.append(
            {
                "color": color,
                "band_h": int(band_h),
                "mid": int(mid),
                "rows": rows,
                "starts": starts,
                "ends": ends,
                "score": float(np.mean(starts)),
            }
        )
    return candidates


def _find_selector(grid: np.ndarray, bg: int) -> dict:
    candidates = _selector_candidates(grid, bg)
    assert candidates, "expected a uniform right-side selector-bar color"
    best_score = max(item["score"] for item in candidates)
    best = [item for item in candidates if item["score"] == best_score]
    assert len(best) == 1, "selector-bar structure must be unique"
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack right-edge crops from row-band sprites using each band's selector length.

    Input structure:
    - The dominant color is the background.
    - The grid is split into equal-height horizontal bands.
    - In every band, a left-side foreground sprite spans the band height, and a
      uniform selector color forms one horizontal run on the band's middle row to
      the right of that sprite.

    Rule:
    - Measure each selector run length L.
    - For the sprite in the same band, take the height-of-band by L rectangle
      ending at the sprite's rightmost foreground column.
    - Vertically stack these crops in top-to-bottom band order on a background
      canvas whose width is the maximum L, right-aligning shorter crops.
    - The selector bars are discarded.

    Key insight:
    - The right-side bar is a crop-width indicator, not part of the output; the
      hidden direction is to crop leftward from each sprite's right edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, _width = map(int, g.shape)
    bg = _most_common_color(g)
    selector = _find_selector(g, bg)
    selector_color = int(selector["color"])
    band_h = int(selector["band_h"])
    assert height % band_h == 0
    n_bands = height // band_h

    pieces = []
    lengths = []
    for band_i in range(n_bands):
        r0 = band_i * band_h
        r1 = r0 + band_h
        strip = g[r0:r1]
        bar_r = int(selector["rows"][band_i])
        runs = _row_runs(g[bar_r] == selector_color)
        assert len(runs) == 1, "expected one selector run per band"
        start, end = runs[0]
        length = end - start + 1

        shape_mask = (strip != bg) & (strip != selector_color)
        coords = np.argwhere(shape_mask)
        assert coords.size, "expected a foreground sprite in every band"
        c_min = int(coords[:, 1].min())
        c_max = int(coords[:, 1].max())
        assert c_max < start, "selector run must sit to the right of the sprite"
        assert length <= c_max - c_min + 1, "selector length must fit inside sprite width"

        crop_left = c_max - length + 1
        piece = strip[:, crop_left : c_max + 1].copy()
        pieces.append(piece)
        lengths.append(length)

    out_w = max(lengths)
    out = np.full((n_bands * band_h, out_w), bg, dtype=int)
    for band_i, piece in enumerate(pieces):
        length = int(piece.shape[1])
        r0 = band_i * band_h
        out[r0 : r0 + band_h, out_w - length : out_w] = piece
    return out


# --------------- generator ---------------

ARC_GRID_MAX = 30
COLOR_VALUES = tuple(range(10))
BAND_HEIGHT_OPTIONS = (3, 5, 7)
BAND_COUNT_RANGE = (2, 6)
SHAPE_WIDTH_RANGE = (3, 10)
LEFT_PAD_RANGE = (0, 2)
SELECTOR_GAP_RANGE = (1, 4)
RIGHT_PAD_RANGE = (0, 4)
FOREGROUND_PALETTE_MAX = 4
GENERATION_ATTEMPTS = 300


def _make_shape(height: int, width: int) -> np.ndarray:
    kind = random.choice(("solid", "frame", "rails", "notched", "stair"))
    mask = np.zeros((height, width), dtype=bool)

    if kind == "solid":
        mask[:, :] = True
    elif kind == "frame" and height >= 3 and width >= 3:
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
    elif kind == "notched" and height >= 4 and width >= 5:
        mask[:, :] = True
        notch_h = random.randint(1, max(1, height - 2))
        notch_w = random.randint(1, max(1, width - 3))
        nr = random.randint(1, height - notch_h - 1)
        nc = random.randint(1, width - notch_w - 1)
        mask[nr : nr + notch_h, nc : nc + notch_w] = False
    elif kind == "stair":
        mask[:, 0] = True
        mask[:, -1] = True
        for c in range(width):
            r = round(c * (height - 1) / max(1, width - 1))
            mask[int(r), c] = True
            if r + 1 < height and random.random() < 0.45:
                mask[int(r) + 1, c] = True
    else:
        mask[:, 0] = True
        mask[:, -1] = True
        rows = list(range(height))
        random.shuffle(rows)
        for r in rows[: random.randint(1, height)]:
            a = random.randint(0, width - 2)
            b = random.randint(a + 1, width - 1)
            mask[r, a : b + 1] = True

    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _bg_is_unique_most_common(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    bg_count = int(counts[values == bg][0])
    other_counts = [int(n) for v, n in zip(values, counts) if int(v) != int(bg)]
    return bool(other_counts) and bg_count > max(other_counts)


def _gen_attempt() -> np.ndarray | None:
    band_h = random.choice(BAND_HEIGHT_OPTIONS)
    max_bands = min(BAND_COUNT_RANGE[1], ARC_GRID_MAX // band_h)
    if max_bands < BAND_COUNT_RANGE[0]:
        return None
    n_bands = random.randint(BAND_COUNT_RANGE[0], max_bands)
    height = band_h * n_bands
    mid = band_h // 2

    bg, selector = random.sample(COLOR_VALUES, 2)
    color_pool = [c for c in COLOR_VALUES if c not in (bg, selector)]
    fg_count = random.randint(1, min(FOREGROUND_PALETTE_MAX, n_bands, len(color_pool)))
    foregrounds = random.sample(color_pool, fg_count)

    specs = []
    min_width = 0
    for band_i in range(n_bands):
        shape_w = random.randint(*SHAPE_WIDTH_RANGE)
        length = random.randint(1, shape_w)
        left = random.randint(*LEFT_PAD_RANGE)
        gap = random.randint(*SELECTOR_GAP_RANGE)
        right = random.randint(*RIGHT_PAD_RANGE)
        base_start = left + shape_w + gap
        needed_w = base_start + length + right
        if needed_w > ARC_GRID_MAX:
            return None
        specs.append((band_i, shape_w, length, left, base_start))
        min_width = max(min_width, needed_w)

    if min_width > ARC_GRID_MAX:
        return None
    width = random.randint(min_width, ARC_GRID_MAX)
    grid = np.full((height, width), bg, dtype=int)

    for band_i, shape_w, length, left, base_start in specs:
        r0 = band_i * band_h
        fg = random.choice(foregrounds)
        shape = _make_shape(band_h, shape_w)
        grid[r0 : r0 + band_h, left : left + shape_w][shape] = fg

        max_start = width - length
        if base_start > max_start:
            return None
        bar_start = random.randint(base_start, max_start)
        grid[r0 + mid, bar_start : bar_start + length] = selector

    if not _bg_is_unique_most_common(grid, bg):
        return None
    try:
        out = main(grid)
    except Exception:
        return None
    if out.ndim != 2 or max(out.shape) > ARC_GRID_MAX:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
