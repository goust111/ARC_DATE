from __future__ import annotations

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 400


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _band_spans(grid: np.ndarray) -> list[dict[str, int]]:
    h, w = grid.shape
    spans: list[dict[str, int]] = []
    c = 0
    while c < w:
        if grid[h - 1, c] == BG:
            c += 1
            continue
        c0 = c
        while c + 1 < w and grid[h - 1, c + 1] != BG:
            c += 1
        c1 = c
        region = grid[:, c0 : c1 + 1]
        colors = [int(v) for v in np.unique(region) if int(v) != BG]
        assert len(colors) == 1, "each band span must contain exactly one foreground color"
        spans.append(
            {
                "c0": int(c0),
                "c1": int(c1),
                "color": int(colors[0]),
                "count": int(np.count_nonzero(region != BG)),
            }
        )
        c += 1
    assert len(spans) >= 2, "expected multiple vertical bands"
    return spans


def _legend_entries(grid: np.ndarray, cutoff_col: int) -> list[dict[str, int]]:
    legend_mask = grid[:, :cutoff_col] != BG
    comps = _components4(legend_mask)
    entries: list[dict[str, int]] = []
    for comp in comps:
        rows = comp[:, 0]
        cols = comp[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "legend blocks must be monochrome"
        assert c0 == c1, "legend blocks must be one pixel wide"
        entries.append({"color": int(next(iter(colors))), "height": int(r1 - r0 + 1)})
    assert entries, "legend must be present to the left of the bands"
    return entries


def _parse_problem(grid: np.ndarray) -> tuple[list[dict[str, int]], list[int], dict[int, int]]:
    spans = _band_spans(grid)
    counts = [band["count"] for band in spans]
    assert len(set(counts)) == len(counts), "band pixel counts must be unique"

    rank_order = sorted(
        range(len(spans)),
        key=lambda idx: (-spans[idx]["count"], spans[idx]["c0"]),
    )

    entries = _legend_entries(grid, spans[0]["c0"])
    legend_by_rank = {entry["height"]: entry["color"] for entry in entries}
    assert len(legend_by_rank) == len(entries), "legend heights must be unique"
    assert set(legend_by_rank) == set(range(1, len(spans) + 1)), "legend heights must index all ranks"

    band_colors = {band["color"] for band in spans}
    assert {entry["color"] for entry in entries} == band_colors, "legend colors must match the band colors"
    return spans, rank_order, legend_by_rank


def _transform(grid: np.ndarray, *, fill_mode: str, recolor: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    spans, rank_order, legend_by_rank = _parse_problem(g)

    rank_color_by_band: dict[int, int] = {}
    for rank_pos, band_idx in enumerate(rank_order, start=1):
        rank_color_by_band[band_idx] = int(legend_by_rank[rank_pos])

    band_color_by_band = {
        idx: (rank_color_by_band[idx] if recolor else spans[idx]["color"])
        for idx in range(len(spans))
    }

    out = g.copy()
    for idx, band in enumerate(spans):
        c0, c1 = band["c0"], band["c1"]
        region = out[:, c0 : c1 + 1]
        if recolor:
            region[region != BG] = band_color_by_band[idx]

    if fill_mode == "rank":
        next_band = {
            rank_order[pos]: rank_order[(pos + 1) % len(rank_order)]
            for pos in range(len(rank_order))
        }
    elif fill_mode == "space":
        next_band = {
            idx: (idx + 1) % len(spans)
            for idx in range(len(spans))
        }
    else:
        raise ValueError(f"unsupported fill_mode: {fill_mode}")

    for idx, band in enumerate(spans):
        c0, c1 = band["c0"], band["c1"]
        region = out[:, c0 : c1 + 1]
        region[region == BG] = band_color_by_band[next_band[idx]]

    out[:, : spans[0]["c0"]][g[:, : spans[0]["c0"]] != BG] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint ranked vertical bands from a height-coded legend, then fill each band's gaps with the next rank's color.

    Input structure:
    - Black background with 4-5 separated vertical color bands on the right.
    - Each band occupies its own contiguous column span, reaches the bottom row, and contains only its own color or black gaps.
    - A left-side legend contains one-pixel-wide monochrome vertical blocks; each block's color matches one band color and its height is a rank index.

    Rule:
    - Count each band's non-black pixels and rank the bands from most pixels to fewest.
    - Read the legend as height -> rank, and recolor every non-black pixel in the rank-h band to the legend block's color.
    - After all recoloring, fill the black gaps inside each band with the recolored color of the next band in rank order, wrapping from the last rank back to rank 1.
    - Erase the legend pixels to black and leave separator/background cells outside the band spans unchanged.

    Key insight:
    - The legend's heights encode rank positions, and the gap-filling cycle follows that hidden rank order rather than the bands' spatial left-to-right order.
    """
    return _transform(grid, fill_mode="rank", recolor=True)


def _cyclic_next(order: list[int]) -> dict[int, int]:
    return {
        order[idx]: order[(idx + 1) % len(order)]
        for idx in range(len(order))
    }


def _random_rank_order(n_bands: int, rng: np.random.Generator) -> list[int]:
    space_order = list(range(n_bands))
    space_next = _cyclic_next(space_order)
    while True:
        order = list(map(int, rng.permutation(n_bands)))
        if _cyclic_next(order) != space_next:
            return order


def _random_derangement(order: list[int], rng: np.random.Generator) -> list[int]:
    while True:
        perm = list(order)
        rng.shuffle(perm)
        if all(a != b for a, b in zip(order, perm)):
            return perm


def _assign_counts(
    widths: list[int],
    height: int,
    rank_order: list[int],
    rng: np.random.Generator,
) -> list[int] | None:
    n_bands = len(widths)
    mins = [max(2 * w + 1, int(np.ceil(0.35 * height * w))) for w in widths]
    maxs = [max(mins[i], int(np.floor(0.85 * height * widths[i]))) for i in range(n_bands)]
    maxs = [min(maxs[i], height * widths[i] - 1) for i in range(n_bands)]
    if any(mins[i] > maxs[i] for i in range(n_bands)):
        return None

    counts = [0] * n_bands

    def rec(pos: int, prev_limit: int | None) -> bool:
        if pos == n_bands:
            return True
        idx = rank_order[pos]
        lo = mins[idx]
        hi = maxs[idx] if prev_limit is None else min(maxs[idx], prev_limit - 2)
        if hi < lo:
            return False
        candidates = list(range(lo, hi + 1))
        rng.shuffle(candidates)
        candidates.sort(reverse=True)
        for value in candidates:
            counts[idx] = int(value)
            if rec(pos + 1, int(value)):
                return True
        counts[idx] = 0
        return False

    if rec(0, None):
        return counts
    return None


def _band_mask(height: int, width: int, count: int, rng: np.random.Generator) -> np.ndarray:
    area = height * width
    assert 2 * width <= count < area
    mask = np.zeros((height, width), dtype=bool)
    required = {(0, c) for c in range(width)} | {(height - 1, c) for c in range(width)}
    cells = [(r, c) for r in range(height) for c in range(width)]
    remaining = [cell for cell in cells if cell not in required]
    chosen = list(required)
    extra_needed = count - len(required)
    if extra_needed > 0:
        picks = rng.choice(len(remaining), size=extra_needed, replace=False)
        chosen.extend(remaining[int(i)] for i in np.atleast_1d(picks))
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _build_grid(
    height: int,
    widths: list[int],
    colors: list[int],
    counts: list[int],
    legend_by_rank: dict[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    n_bands = len(widths)
    legend_area_w = 3
    legend_col = int(rng.integers(0, legend_area_w))
    left_gap = int(rng.integers(1, 3))
    separators = [int(rng.integers(1, 3)) for _ in range(n_bands - 1)]
    right_margin = int(rng.integers(0, 4))
    width = legend_area_w + left_gap + sum(widths) + sum(separators) + right_margin
    assert width <= 30

    grid = np.zeros((height, width), dtype=int)

    blocks = [(rank, int(legend_by_rank[rank])) for rank in range(1, n_bands + 1)]
    rng.shuffle(blocks)
    row = 0
    for idx, (rank, color) in enumerate(blocks):
        block_h = int(rank)
        grid[row : row + block_h, legend_col] = int(color)
        row += block_h
        if idx != len(blocks) - 1:
            row += 1

    col = legend_area_w + left_gap
    for band_idx, band_w in enumerate(widths):
        c0, c1 = col, col + band_w
        mask = _band_mask(height, band_w, counts[band_idx], rng)
        grid[:, c0:c1][mask] = int(colors[band_idx])
        col = c1
        if band_idx != n_bands - 1:
            col += separators[band_idx]

    return grid


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        spans, rank_order, legend_by_rank = _parse_problem(grid)
        n_bands = len(spans)
        assert n_bands in (4, 5)
        assert set(legend_by_rank) == set(range(1, n_bands + 1))
        assert all(np.any(grid[:, band["c0"] : band["c1"] + 1] == BG) for band in spans)

        rank_colors = [legend_by_rank[pos] for pos in range(1, n_bands + 1)]
        original_rank_colors = [spans[idx]["color"] for idx in rank_order]
        assert all(a != b for a, b in zip(rank_colors, original_rank_colors))

        correct = main(grid)
        wrong_space = _transform(grid, fill_mode="space", recolor=True)
        wrong_skip_recolor = _transform(grid, fill_mode="rank", recolor=False)
        wrong_both = _transform(grid, fill_mode="space", recolor=False)

        assert not np.array_equal(correct, grid)
        assert not np.array_equal(correct, wrong_space)
        assert not np.array_equal(correct, wrong_skip_recolor)
        assert not np.array_equal(correct, wrong_both)

        # Every band should still show both its own recolored cells and borrowed gap cells.
        for pos, band_idx in enumerate(rank_order):
            band = spans[band_idx]
            region = correct[:, band["c0"] : band["c1"] + 1]
            own = legend_by_rank[pos + 1]
            borrow = legend_by_rank[(pos + 1) % n_bands + 1]
            colors = {int(v) for v in np.unique(region)}
            assert colors == {own, borrow}

        return True
    except Exception:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_bands = int(rng.integers(4, 6))
    legend_height = n_bands * (n_bands + 1) // 2 + (n_bands - 1)
    min_h = max(legend_height + 1, 16)
    max_h = min(26, legend_height + 6)
    if min_h > max_h:
        return None
    height = int(rng.integers(min_h, max_h + 1))

    widths = [int(v) for v in rng.integers(1, 4, size=n_bands)]
    rank_order = _random_rank_order(n_bands, rng)
    counts = _assign_counts(widths, height, rank_order, rng)
    if counts is None:
        return None

    colors = [int(v) for v in rng.choice(np.asarray(PALETTE), size=n_bands, replace=False)]
    color_sources = _random_derangement(rank_order, rng)
    legend_by_rank = {
        rank_pos + 1: int(colors[color_sources[rank_pos]])
        for rank_pos in range(n_bands)
    }

    grid = _build_grid(height, widths, colors, counts, legend_by_rank, rng)
    if not _validate_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ranking-driven cyclic band puzzle")
