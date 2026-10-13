import random
from collections import Counter

import numpy as np


PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    boundary = list(map(int, grid[0]))
    boundary.extend(map(int, grid[h - 1]))
    boundary.extend(map(int, grid[1 : h - 1, 0]))
    boundary.extend(map(int, grid[1 : h - 1, w - 1]))
    return int(Counter(boundary).most_common(1)[0][0])


def _row_color(row: np.ndarray, bg: int) -> int | None:
    colors = [int(color) for color in np.unique(row) if int(color) != bg]
    assert len(colors) <= 1, "each row must contain at most one foreground layer color"
    return colors[0] if colors else None


def _extract_bands(grid: np.ndarray, bg: int) -> list[dict]:
    bands: list[dict] = []
    h = grid.shape[0]
    r = 0
    while r < h:
        color = _row_color(grid[r], bg)
        if color is None:
            r += 1
            continue

        r0 = int(r)
        while r + 1 < h and _row_color(grid[r + 1], bg) == color:
            r += 1
        r1 = int(r)

        rows = grid[r0 : r1 + 1]
        present = np.any(rows == int(color), axis=0)
        cols = np.flatnonzero(present)
        assert cols.size > 0
        left = int(cols[0])
        right = int(cols[-1])

        gaps: list[tuple[int, int]] = []
        c = left
        while c <= right:
            if bool(present[c]):
                c += 1
                continue
            g0 = int(c)
            while c <= right and not bool(present[c]):
                c += 1
            gaps.append((int(g0), int(c - 1)))

        bands.append({"color": int(color), "r0": int(r0), "r1": int(r1), "gaps": gaps})
        r += 1

    return bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify stacked horizontal sediment bands and let upper same-color strata seep
    straight downward through background gaps in lower bands.

    A gap is filled column-by-column only when the nearest visible pixel above that
    column has the same color as the current band. Filling continues downward until
    another color or the grid bottom, so higher layers keep covering lower ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    bands = _extract_bands(g, bg)

    out = g.copy()
    h, w = out.shape

    for band in bands:
        color = int(band["color"])
        top = int(band["r0"])
        for gap_left, gap_right in band["gaps"]:
            for col in range(int(gap_left), int(gap_right) + 1):
                row = int(top - 1)
                while row >= 0 and int(out[row, col]) == bg:
                    row -= 1
                if row < 0 or int(out[row, col]) != color:
                    continue

                fill_row = int(row + 1)
                while fill_row < h and int(out[fill_row, col]) == bg:
                    out[fill_row, col] = color
                    fill_row += 1

    assert out.shape == (h, w)
    return out


def _pick_interval(left: int, right: int, width_min: int, width_max: int) -> tuple[int, int] | None:
    if right - left + 1 < width_min:
        return None
    width = random.randint(width_min, min(width_max, right - left + 1))
    start = random.randint(left, right - width + 1)
    return int(start), int(start + width - 1)


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _apply_gap(mask: np.ndarray, gap: tuple[int, int]) -> None:
    mask[int(gap[0]) : int(gap[1]) + 1] = False


def _extra_gap(
    span: tuple[int, int],
    protected_open: list[tuple[int, int]],
    protected_gap: list[tuple[int, int]],
    min_width: int = 1,
    max_width: int = 3,
) -> tuple[int, int] | None:
    left, right = map(int, span)
    inner_left = int(left + 1)
    inner_right = int(right - 1)
    if inner_right < inner_left:
        return None
    for _ in range(30):
        gap = _pick_interval(inner_left, inner_right, min_width, max_width)
        if gap is None:
            return None
        if any(_overlaps(gap, interval) for interval in protected_open):
            continue
        if any(_overlaps(gap, interval) for interval in protected_gap):
            continue
        return gap
    return None


def _span_walk(width: int, count: int) -> list[tuple[int, int]]:
    left = random.randint(1, 2)
    right = random.randint(max(left + 6, width - 5), width - 2)
    spans = []
    for _ in range(count):
        if spans:
            left += random.choice((-1, 0, 1))
            right += random.choice((-1, 0, 1))
        left = max(1, min(left, width - 7))
        right = min(width - 2, max(right, left + 6))
        spans.append((int(left), int(right)))
    return spans


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    width = random.randint(12, 18)
    band_count = random.randint(5, 7)

    while True:
        thicknesses = [random.randint(1, 3 if band_count <= 5 else 2) for _ in range(band_count)]
        top_margin = random.randint(1, 4)
        total_height = sum(thicknesses) + top_margin
        if 11 <= total_height <= 18:
            break

    height = total_height
    bg = random.choice(PALETTE)
    available = [color for color in PALETTE if color != bg]

    seep_to_bottom = bool(random.getrandbits(1))
    upper_hi = band_count - 3 if seep_to_bottom else band_count - 4
    upper_idx = random.randint(0, upper_hi)
    lower_hi = band_count - 1 if seep_to_bottom else band_count - 2
    lower_idx = random.randint(upper_idx + 2, lower_hi)

    seep_color = random.choice(available)
    other_colors = random.sample([color for color in available if color != seep_color], band_count - 2)
    colors: list[int | None] = [None] * band_count
    colors[upper_idx] = seep_color
    colors[lower_idx] = seep_color

    other_iter = iter(other_colors)
    for idx in range(band_count):
        if colors[idx] is None:
            colors[idx] = next(other_iter)
    colors = [int(color) for color in colors]

    candidate_dead = [idx for idx in range(band_count) if idx not in (upper_idx, lower_idx)]
    dead_idx = random.choice(candidate_dead)
    dead_color = int(colors[dead_idx])

    spans = _span_walk(width, band_count)

    shaft_left_bound = max(spans[idx][0] for idx in range(upper_idx, lower_idx + 1)) + 1
    shaft_right_bound = min(spans[idx][1] for idx in range(upper_idx, lower_idx + 1)) - 1
    shaft = _pick_interval(shaft_left_bound, shaft_right_bound, 1, 3)
    if shaft is None:
        return None

    dead_left = int(spans[dead_idx][0] + 1)
    dead_right = int(spans[dead_idx][1] - 1)
    if dead_right < dead_left:
        return None

    dead_gap = None
    for _ in range(30):
        candidate = _pick_interval(dead_left, dead_right, 1, 3)
        if candidate is not None and not _overlaps(candidate, shaft):
            dead_gap = candidate
            break
    if dead_gap is None:
        return None

    masks = []
    for idx, span in enumerate(spans):
        mask = np.zeros(width, dtype=bool)
        left, right = map(int, span)
        mask[left : right + 1] = True

        protected_open: list[tuple[int, int]] = []
        protected_gap: list[tuple[int, int]] = []

        if idx == upper_idx:
            protected_open.append(shaft)
        if upper_idx < idx <= lower_idx:
            protected_gap.append(shaft)
        if seep_to_bottom and idx > lower_idx:
            protected_gap.append(shaft)
        if idx == dead_idx:
            protected_gap.append(dead_gap)

        if upper_idx < idx <= lower_idx:
            _apply_gap(mask, shaft)
        if seep_to_bottom and idx > lower_idx:
            _apply_gap(mask, shaft)
        if idx == dead_idx:
            _apply_gap(mask, dead_gap)

        extra_gap_count = random.randint(0, 2)
        if idx == upper_idx:
            extra_gap_count = random.randint(0, 1)
        for _ in range(extra_gap_count):
            gap = _extra_gap(span, protected_open, protected_gap)
            if gap is None:
                continue
            _apply_gap(mask, gap)
            protected_gap.append(gap)

        if not np.any(mask):
            return None
        cols = np.flatnonzero(mask)
        if cols.size < 3:
            return None
        masks.append(mask)

    grid = np.full((height, width), bg, dtype=int)
    band_rows = []
    row = height - 1
    for idx in range(band_count - 1, -1, -1):
        thickness = int(thicknesses[idx])
        r0 = int(row - thickness + 1)
        if r0 < top_margin:
            return None
        mask = masks[idx]
        color = int(colors[idx])
        for r in range(r0, row + 1):
            grid[r, mask] = color
        band_rows.append((idx, r0, row))
        row = r0 - 1
    band_rows.reverse()

    metadata = {
        "bg": int(bg),
        "seep_color": int(seep_color),
        "dead_color": int(dead_color),
        "upper_idx": int(upper_idx),
        "lower_idx": int(lower_idx),
        "dead_idx": int(dead_idx),
        "shaft": tuple(map(int, shaft)),
        "dead_gap": tuple(map(int, dead_gap)),
        "rows": {int(idx): (int(r0), int(r1)) for idx, r0, r1 in band_rows},
        "seep_to_bottom": bool(seep_to_bottom),
    }
    return grid, metadata


def _validate_generated(grid: np.ndarray, meta: dict) -> bool:
    bg = int(meta["bg"])
    for row in grid:
        colors = {int(color) for color in np.unique(row) if int(color) != bg}
        if len(colors) > 1:
            return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    shaft_left, shaft_right = map(int, meta["shaft"])
    lower_r0, lower_r1 = meta["rows"][int(meta["lower_idx"])]
    dead_r0, dead_r1 = meta["rows"][int(meta["dead_idx"])]
    seep_color = int(meta["seep_color"])

    filled = False
    for col in range(shaft_left, shaft_right + 1):
        for row in range(lower_r0, lower_r1 + 1):
            if int(grid[row, col]) != bg:
                return False
            if int(out[row, col]) == seep_color:
                filled = True
    if not filled:
        return False

    dead_left, dead_right = map(int, meta["dead_gap"])
    for col in range(dead_left, dead_right + 1):
        for row in range(dead_r0, dead_r1 + 1):
            if int(grid[row, col]) != bg or int(out[row, col]) != bg:
                return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is None:
            continue
        grid, meta = candidate
        if _validate_generated(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid sediment-layer input")
