import random

import numpy as np


BG = 0
YELLOW = 4
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)

MIN_STRIPES = 3
MAX_STRIPES = 6
MIN_SEGMENT_WIDTH = 3
MAX_SEGMENT_WIDTH = 9
MASK_ATTEMPTS = 120
MASK_DENSITY_RANGE = (0.35, 0.78)
ROW_GAP_CHOICES = (1, 2)
TOP_PAD_RANGE = (0, 2)
BOTTOM_PAD_RANGE = (0, 2)
MAX_WIDTH_EXTRA = 5
MAX_GENERATION_ATTEMPTS = 300


def _stripe_rows(grid: np.ndarray) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if np.any(grid[r] != BG)]


def _fold_band(band: np.ndarray, segment_count: int, color: int) -> np.ndarray:
    restored = np.where(band == YELLOW, color, band)
    assert restored.size % segment_count == 0, "stripe span must split into equal segments"
    segment_width = restored.size // segment_count
    segments = restored.reshape(segment_count, segment_width)
    assert set(map(int, np.unique(segments))) <= {BG, color}, "stripe must contain only background and one main color"
    return np.where(np.any(segments == color, axis=0), color, BG)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each marked horizontal strip into multiple aligned segments and merge them into one row.

    Input structure:
    - The background is Black.
    - Several one-pixel-high horizontal strips appear on separate rows, with at least
      one fully Black row between neighboring strips.
    - Each strip uses one main color plus one to three Yellow marker pixels, and Black
      gaps may appear inside the strip.

    Rule:
    - For each strip, count its Yellow marker pixels; this count N is the number of fold segments.
    - Trim the strip to its non-Black span, replace Yellow back with the strip's main color,
      and split the span into N equal left-to-right segments.
    - Stack the N segments on top of each other: each output position keeps the main color
      if any segment has it there, otherwise it stays Black.
    - Output the folded strips in top-to-bottom order with no blank rows between them, and
      right-pad shorter folded rows with Black to the maximum folded width.

    Key insight:
    - The Yellow pixels do not survive as markers; they only encode how many equal horizontal
      parts the strip must be folded into before taking the color union.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rows = _stripe_rows(g)
    assert rows, "expected at least one colored strip"

    folded_rows: list[np.ndarray] = []
    max_width = 0
    for row_index in rows:
        row = g[row_index]
        non_bg = np.flatnonzero(row != BG)
        assert len(non_bg) > 0, "strip row must contain non-background pixels"
        c0, c1 = int(non_bg[0]), int(non_bg[-1])
        band = row[c0 : c1 + 1]

        segment_count = int(np.count_nonzero(band == YELLOW))
        assert 1 <= segment_count <= 3, "marker count must be between 1 and 3"

        colors = {int(v) for v in band if v not in (BG, YELLOW)}
        assert len(colors) == 1, "each strip must have exactly one main color"
        color = next(iter(colors))

        folded = _fold_band(band, segment_count, color)
        folded_rows.append(folded.astype(int))
        max_width = max(max_width, int(folded.size))

    out = np.full((len(folded_rows), max_width), BG, dtype=int)
    for r, folded in enumerate(folded_rows):
        out[r, : folded.size] = folded
    return out


def _balanced_segment_counts(n_stripes: int) -> list[int]:
    counts = [1, 2, 3]
    while len(counts) < n_stripes:
        counts.append(random.choice((1, 2, 3)))
    random.shuffle(counts)
    return counts


def _sample_segment_mask(segment_count: int, segment_width: int, require_gap: bool) -> np.ndarray | None:
    for _ in range(MASK_ATTEMPTS):
        density = random.uniform(*MASK_DENSITY_RANGE)
        mask = np.array(
            [
                [random.random() < density for _ in range(segment_width)]
                for _ in range(segment_count)
            ],
            dtype=bool,
        )

        if require_gap and segment_width >= 3:
            gap_col = random.randint(1, segment_width - 2)
            mask[:, gap_col] = False

        mask[0, 0] = True
        mask[-1, -1] = True

        for seg_idx in range(segment_count):
            if not mask[seg_idx].any():
                mask[seg_idx, random.randrange(segment_width)] = True

        if int(mask.sum()) <= segment_count:
            empty_cells = np.argwhere(~mask)
            if len(empty_cells) == 0:
                continue
            rr, cc = map(int, random.choice(empty_cells.tolist()))
            mask[rr, cc] = True

        union = np.any(mask, axis=0)
        if union.sum() < 2:
            continue
        if require_gap and not np.any(~union):
            continue
        if segment_count > 1 and all(np.array_equal(mask[0], mask[i]) for i in range(1, segment_count)):
            continue
        return mask
    return None


def _render_stripe(segment_count: int, segment_width: int, color: int, require_gap: bool) -> tuple[np.ndarray, np.ndarray] | None:
    mask = _sample_segment_mask(segment_count, segment_width, require_gap)
    if mask is None:
        return None

    band = np.full(segment_count * segment_width, BG, dtype=int)
    for seg_idx in range(segment_count):
        start = seg_idx * segment_width
        band[start : start + segment_width] = np.where(mask[seg_idx], color, BG)
        marker_candidates = np.flatnonzero(mask[seg_idx])
        marker_col = int(random.choice(marker_candidates.tolist()))
        band[start + marker_col] = YELLOW

    folded = np.where(np.any(mask, axis=0), color, BG).astype(int)
    return band, folded


def _row_positions(n_stripes: int) -> tuple[list[int], int]:
    top_pad = random.randint(*TOP_PAD_RANGE)
    bottom_pad = random.randint(*BOTTOM_PAD_RANGE)
    gaps = [random.choice(ROW_GAP_CHOICES) for _ in range(n_stripes - 1)]

    rows: list[int] = []
    r = top_pad
    for idx in range(n_stripes):
        rows.append(r)
        r += 1
        if idx + 1 < n_stripes:
            r += gaps[idx]
    height = r + bottom_pad
    return rows, height


def _gen_attempt() -> np.ndarray | None:
    n_stripes = random.randint(MIN_STRIPES, MAX_STRIPES)
    segment_counts = _balanced_segment_counts(n_stripes)
    colors = random.sample(PALETTE, n_stripes)
    gap_index = random.randrange(n_stripes)

    stripes: list[np.ndarray] = []
    folded_rows: list[np.ndarray] = []
    for idx, (segment_count, color) in enumerate(zip(segment_counts, colors)):
        segment_width = random.randint(MIN_SEGMENT_WIDTH, MAX_SEGMENT_WIDTH)
        rendered = _render_stripe(
            segment_count=segment_count,
            segment_width=segment_width,
            color=color,
            require_gap=(idx == gap_index),
        )
        if rendered is None:
            return None
        band, folded = rendered
        stripes.append(band)
        folded_rows.append(folded)

    rows, height = _row_positions(n_stripes)
    spans = [int(band.size) for band in stripes]
    base_width = max(spans)
    width = base_width + random.randint(0, min(MAX_WIDTH_EXTRA, 30 - base_width))
    if height > 30 or width > 30:
        return None

    inp = np.full((height, width), BG, dtype=int)
    out_width = max(int(row.size) for row in folded_rows)
    out = np.full((n_stripes, out_width), BG, dtype=int)

    for out_r, (grid_r, band, folded) in enumerate(zip(rows, stripes, folded_rows)):
        left = random.randint(0, width - band.size)
        inp[grid_r, left : left + band.size] = band
        out[out_r, : folded.size] = folded

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
