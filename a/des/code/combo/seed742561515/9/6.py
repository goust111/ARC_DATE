import random
from typing import Iterable

import numpy as np


BG = 0
MAX_GENERATION_ATTEMPTS = 400
COLOR_POOL = tuple(range(1, 10))
SEPARATOR_ROWS_MIN = 3

BAND_COUNT_RANGE = (2, 4)
BAND_HEIGHT_RANGE = (3, 5)
SPRITE_WIDTH_RANGE = (3, 6)
COPY_COUNT_RANGE = (3, 5)
GAP_RANGE = (1, 3)
SIDE_MARGIN_RANGE = (1, 4)
RIGHT_SLACK_RANGE = (1, 4)
TEMPLATE_FILL_FRACTION = (0.55, 0.85)
MAX_TEMPLATE_ATTEMPTS = 200
MAX_DAMAGE_ATTEMPTS = 200


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for value in map(int, idx[1:]):
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev + 1))
        start = prev = value
    runs.append((start, prev + 1))
    return runs


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_separator_rows(grid: np.ndarray) -> tuple[int, list[int]]:
    h, w = grid.shape
    threshold = max(1, w - 1)
    candidates: list[tuple[int, list[int]]] = []
    for color in map(int, np.unique(grid)):
        rows = [r for r in range(h) if int(np.count_nonzero(grid[r] == color)) >= threshold]
        if len(rows) < SEPARATOR_ROWS_MIN:
            continue
        if rows[0] != 0 or rows[-1] != h - 1:
            continue
        gaps = [rows[i + 1] - rows[i] - 1 for i in range(len(rows) - 1)]
        if min(gaps) <= 0 or len(set(gaps)) != 1:
            continue
        candidates.append((color, rows))
    assert len(candidates) == 1, "expected a unique family of equally spaced separator rows"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify the incomplete repeated sprite in each horizontal band.

    Complete copies are erased. The missing pixels of the damaged copy are
    restored with the band's marker color taken from the adjacent separator row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator, sep_rows = _find_separator_rows(g)
    out = g.copy()

    for upper, lower in zip(sep_rows, sep_rows[1:]):
        band = g[upper + 1 : lower]
        assert band.size > 0, "bands must be non-empty"

        colors = [int(c) for c in np.unique(band) if int(c) not in {BG, separator}]
        assert len(colors) == 1, "each band should contain one foreground color"
        fg = colors[0]

        row_mask = np.any(band == fg, axis=1)
        col_mask = np.any(band == fg, axis=0)
        row_runs = _runs(row_mask)
        col_runs = _runs(col_mask)
        assert len(row_runs) == 1, "all copies should share one vertical window"
        assert len(col_runs) >= 2, "need multiple repeated copies in each band"

        top, bottom = row_runs[0]
        widths = {end - start for start, end in col_runs}
        assert len(widths) == 1, "copies should share one width"

        windows = [(band[top:bottom, start:end] == fg) for start, end in col_runs]
        template = np.logical_or.reduce(windows)
        broken = [i for i, window in enumerate(windows) if not np.array_equal(window, template)]
        assert len(broken) == 1, "expected exactly one incomplete copy"
        broken_idx = broken[0]
        missing = template & ~windows[broken_idx]
        assert np.any(missing), "the incomplete copy must miss at least one pixel"

        start, end = col_runs[broken_idx]
        marker_candidates = []
        for sep_row in (upper, lower):
            for col in range(start, end):
                value = int(g[sep_row, col])
                if value not in {separator, BG}:
                    marker_candidates.append(value)
        assert len(marker_candidates) == 1, "expected one marker aligned with the broken copy"
        marker_color = marker_candidates[0]

        band_out = out[upper + 1 : lower]
        band_out[band_out == fg] = BG
        patch = band_out[top:bottom, start:end]
        patch[template] = fg
        patch[missing] = marker_color

    return out


def _sample_template(height: int, width: int) -> np.ndarray | None:
    area = height * width
    lo = max(height + width - 1, int(np.ceil(area * TEMPLATE_FILL_FRACTION[0])))
    hi = min(area - 1, int(np.floor(area * TEMPLATE_FILL_FRACTION[1])))
    if hi < lo:
        hi = lo

    for _ in range(MAX_TEMPLATE_ATTEMPTS):
        target = random.randint(lo, hi)
        seed = (random.randrange(height), random.randrange(width))
        cells = {seed}
        frontier = set(_neighbors4(seed[0], seed[1], height, width))

        while frontier and len(cells) < target:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            if cell in cells:
                continue
            cells.add(cell)
            for nb in _neighbors4(cell[0], cell[1], height, width):
                if nb not in cells:
                    frontier.add(nb)

        if len(cells) != target:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue

        row_counts = mask.sum(axis=1)
        col_counts = mask.sum(axis=0)
        removable = [(r, c) for r, c in cells if row_counts[r] > 1 and col_counts[c] > 1]
        if removable:
            if random.random() < 0.5:
                mask = np.fliplr(mask)
            if random.random() < 0.5:
                mask = np.flipud(mask)
            return mask
    return None


def _sample_damage(mask: np.ndarray) -> np.ndarray | None:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    row_counts = mask.sum(axis=1).astype(int)
    col_counts = mask.sum(axis=0).astype(int)
    removable = [(r, c) for r, c in coords if row_counts[r] > 1 and col_counts[c] > 1]
    if not removable:
        return None

    max_remove = min(3, len(removable), len(coords) - 1)
    if max_remove <= 0:
        return None

    for _ in range(MAX_DAMAGE_ATTEMPTS):
        count = random.randint(1, max_remove)
        chosen = random.sample(removable, count)
        kept = mask.copy()
        for r, c in chosen:
            kept[r, c] = False
        if np.any(kept) and np.all(kept.any(axis=0)) and np.all(kept.any(axis=1)):
            damage = np.zeros_like(mask, dtype=bool)
            for r, c in chosen:
                damage[r, c] = True
            return damage
    return None


def _sample_band(height: int, fg: int, marker: int) -> dict | None:
    width = random.randint(*SPRITE_WIDTH_RANGE)
    template = _sample_template(height, width)
    if template is None:
        return None

    damage = _sample_damage(template)
    if damage is None:
        return None

    copies = random.randint(*COPY_COUNT_RANGE)
    gap = random.randint(*GAP_RANGE)
    left = random.randint(*SIDE_MARGIN_RANGE)
    right = random.randint(*RIGHT_SLACK_RANGE)
    broken_idx = random.randrange(copies)
    starts = [left + i * (width + gap) for i in range(copies)]

    damage_cols = [int(c) for c in np.flatnonzero(np.any(damage, axis=0))]
    marker_col = starts[broken_idx] + random.choice(damage_cols)

    return {
        "fg": fg,
        "marker": marker,
        "width": width,
        "template": template,
        "damage": damage,
        "copies": copies,
        "gap": gap,
        "left": left,
        "right": right,
        "starts": starts,
        "broken_idx": broken_idx,
        "marker_col": marker_col,
        "used_width": starts[-1] + width + right,
    }


def _render_input(bands: list[dict], band_height: int, separator: int, marker_side: str, width: int) -> np.ndarray:
    total_rows = len(bands) * band_height + len(bands) + 1
    grid = np.full((total_rows, width), BG, dtype=int)

    for row in range(0, total_rows, band_height + 1):
        grid[row, :] = separator

    for band_idx, spec in enumerate(bands):
        top = 1 + band_idx * (band_height + 1)
        for copy_idx, start in enumerate(spec["starts"]):
            sprite = spec["template"].copy()
            if copy_idx == spec["broken_idx"]:
                sprite &= ~spec["damage"]
            window = grid[top : top + band_height, start : start + spec["width"]]
            window[sprite] = spec["fg"]

        marker_row = top - 1 if marker_side == "top" else top + band_height
        grid[marker_row, spec["marker_col"]] = spec["marker"]

    return grid


def _render_output(inp: np.ndarray, bands: list[dict], band_height: int) -> np.ndarray:
    out = inp.copy()
    for band_idx, spec in enumerate(bands):
        top = 1 + band_idx * (band_height + 1)
        band = out[top : top + band_height]
        band[band == spec["fg"]] = BG

        start = spec["starts"][spec["broken_idx"]]
        patch = band[:, start : start + spec["width"]]
        patch[spec["template"]] = spec["fg"]
        patch[spec["damage"]] = spec["marker"]

    return out


def _gen_attempt() -> np.ndarray | None:
    band_count = random.randint(*BAND_COUNT_RANGE)
    band_height = random.randint(*BAND_HEIGHT_RANGE)
    marker_side = random.choice(("top", "bottom"))

    needed_colors = 1 + 2 * band_count
    palette = random.sample(COLOR_POOL, needed_colors)
    separator = palette[0]
    fg_colors = palette[1 : 1 + band_count]
    marker_colors = palette[1 + band_count :]

    bands = []
    for fg, marker in zip(fg_colors, marker_colors):
        spec = _sample_band(band_height, fg, marker)
        if spec is None:
            return None
        bands.append(spec)

    width = max(spec["used_width"] for spec in bands)
    inp = _render_input(bands, band_height, separator, marker_side, width)
    expected = _render_output(inp, bands, band_height)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
