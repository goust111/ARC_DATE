from __future__ import annotations

import numpy as np


SEP = 0
GENERATION_RETRIES = 400
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background_color(grid: np.ndarray) -> int:
    values = grid[grid != SEP]
    assert values.size > 0
    colors, counts = np.unique(values, return_counts=True)
    best = int(np.argmax(counts))
    return int(colors[best])


def _rectangular_regions(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    regions: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            r0 = r1 = int(r)
            c0 = c1 = int(c)
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                r0 = min(r0, cr)
                r1 = max(r1, cr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            patch = mask[r0 : r1 + 1, c0 : c1 + 1]
            assert patch.all(), "each region must be a solid rectangle of non-black cells"
            assert len(cells) == patch.size
            regions.append((r0, c0, r1, c1))
    regions.sort()
    return regions


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert rows.size > 0 and cols.size > 0
    return mask[int(rows[0]) : int(rows[-1]) + 1, int(cols[0]) : int(cols[-1]) + 1]


def _extract_template(region: np.ndarray, bg: int) -> tuple[int, np.ndarray]:
    colors = [int(color) for color in np.unique(region) if int(color) != int(bg)]
    assert len(colors) == 1, "template region must contain exactly one foreground color"
    color = int(colors[0])
    mask = _crop_mask(region == color)
    assert mask.any()
    return color, mask


def _analyze_grid(grid: np.ndarray) -> tuple[int, list[tuple[int, np.ndarray]], list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _background_color(g)
    regions = _rectangular_regions(g != SEP)
    assert len(regions) >= 3, "expected two template regions and at least one vote region"

    top = min(r0 for r0, _c0, _r1, _c1 in regions)
    template_regions = [rect for rect in regions if rect[0] == top]
    assert len(template_regions) == 2, "expected exactly two template regions in the top band"
    template_regions.sort(key=lambda rect: rect[1])
    template_region_set = set(template_regions)

    templates: list[tuple[int, np.ndarray]] = []
    for r0, c0, r1, c1 in template_regions:
        color, mask = _extract_template(g[r0 : r1 + 1, c0 : c1 + 1], bg)
        assert color not in [known for known, _ in templates], "template colors must differ"
        templates.append((color, mask))

    vote_regions = [rect for rect in regions if rect not in template_region_set]
    assert vote_regions, "expected at least one non-template region"

    template_colors = [color for color, _ in templates]
    for r0, c0, r1, c1 in vote_regions:
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        counts = [int(np.count_nonzero(patch == color)) for color in template_colors]
        assert sum(counts) > 0, "every vote region must contain at least one vote pixel"
        assert counts[0] != counts[1], "vote counts must not tie"
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        for _color, mask in templates:
            mh, mw = mask.shape
            assert mh <= height and mw <= width, "templates must fit inside every vote region"

    return bg, templates, template_regions, vote_regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a black-line partitioned grid, use the majority vote color in each non-template
    region to choose one of the two top-row template shapes, then redraw that template
    in the winning color aligned to the region's top-left corner.
    """
    g = np.asarray(grid, dtype=int)
    bg, templates, _template_regions, vote_regions = _analyze_grid(g)

    template_map = {color: mask for color, mask in templates}
    template_colors = [color for color, _mask in templates]
    out = g.copy()

    for r0, c0, r1, c1 in vote_regions:
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        counts = [int(np.count_nonzero(patch == color)) for color in template_colors]
        winner = template_colors[0] if counts[0] > counts[1] else template_colors[1]

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        mask = template_map[winner]
        mh, mw = mask.shape
        window = out[r0 : r0 + mh, c0 : c0 + mw]
        window[mask] = int(winner)

    return out


def _split_total(total: int, parts: int, min_value: int, rng: np.random.Generator) -> list[int]:
    assert parts >= 1
    assert total >= parts * min_value
    values = np.full(parts, int(min_value), dtype=int)
    slack = int(total - parts * min_value)
    for _ in range(slack):
        values[int(rng.integers(0, parts))] += 1
    rng.shuffle(values)
    return [int(v) for v in values]


def _random_connected_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        target = int(rng.integers(3, min(h * w, 7) + 1))
        start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        cells = {start}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            pick = frontier[int(rng.integers(0, len(frontier)))]
            cells.add(pick)

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) < 3 or int(mask.sum()) == mask.size:
            continue
        return mask
    raise RuntimeError("failed to generate a connected template mask")


def _same_mask(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _paint_template(region: np.ndarray, mask: np.ndarray, color: int, rng: np.random.Generator) -> None:
    h, w = region.shape
    mh, mw = mask.shape
    off_r = int(rng.integers(0, h - mh + 1))
    off_c = int(rng.integers(0, w - mw + 1))
    window = region[off_r : off_r + mh, off_c : off_c + mw]
    window[mask] = int(color)


def _paint_votes(region: np.ndarray, winner: int, loser: int, rng: np.random.Generator) -> None:
    area = int(region.size)
    h, w = region.shape
    max_votes = max(3, min(area // 3, 9))
    total_votes = int(rng.integers(3, max_votes + 1))

    if bool(rng.random() < 0.2):
        loser_votes = 0
    else:
        loser_votes = int(rng.integers(1, (total_votes // 2) + 1))
    winner_votes = int(total_votes - loser_votes)
    if winner_votes <= loser_votes:
        winner_votes = loser_votes + 1
        total_votes = winner_votes + loser_votes
    if total_votes > area:
        total_votes = area
        loser_votes = min(loser_votes, (total_votes - 1) // 2)
        winner_votes = total_votes - loser_votes
    assert winner_votes > loser_votes >= 0

    picks = rng.choice(area, size=total_votes, replace=False)
    win_rows = picks[:winner_votes] // w
    win_cols = picks[:winner_votes] % w
    lose_rows = picks[winner_votes:] // w
    lose_cols = picks[winner_votes:] % w
    region[win_rows, win_cols] = int(winner)
    region[lose_rows, lose_cols] = int(loser)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    palette = [int(v) for v in rng.choice(np.arange(1, 10), size=3, replace=False)]
    bg, color_a, color_b = palette

    mask_a = _random_connected_mask(rng)
    mask_b = _random_connected_mask(rng)
    if _same_mask(mask_a, mask_b):
        return None

    if bool(rng.integers(0, 2)):
        mask_a, mask_b = mask_b, mask_a

    template_h = max(mask_a.shape[0], mask_b.shape[0]) + 1
    vote_h_min = max(mask_a.shape[0], mask_b.shape[0])
    vote_w_min = max(mask_a.shape[1], mask_b.shape[1])

    band_count = int(rng.integers(2, 5))
    cols_per_band = [2] + [int(rng.integers(2, 5)) for _ in range(band_count - 1)]
    band_heights = [template_h] + [int(rng.integers(vote_h_min, vote_h_min + 4)) for _ in range(band_count - 1)]

    min_region_w = vote_w_min + 1
    min_total_w = max(k * min_region_w + k + 1 for k in cols_per_band)
    full_w = int(rng.integers(min_total_w, min_total_w + 9))
    full_h = int(sum(band_heights) + band_count + 1)

    grid = np.zeros((full_h, full_w), dtype=int)
    region_specs: list[tuple[int, int, int, int, int, int]] = []

    r = 1
    for band_idx, (band_h, band_cols) in enumerate(zip(band_heights, cols_per_band)):
        usable_w = full_w - band_cols - 1
        widths = _split_total(usable_w, band_cols, min_region_w, rng)
        c = 1
        for col_idx, width in enumerate(widths):
            grid[r : r + band_h, c : c + width] = int(bg)
            region_specs.append((band_idx, col_idx, r, c, band_h, width))
            c += width + 1
        r += band_h + 1

    top_regions = [spec for spec in region_specs if spec[0] == 0]
    if len(top_regions) != 2:
        return None

    lower_regions = [spec for spec in region_specs if spec[0] != 0]
    if len(lower_regions) < 2:
        return None

    for spec, mask, color in zip(sorted(top_regions, key=lambda item: item[1]), [mask_a, mask_b], [color_a, color_b]):
        _band, _col, r0, c0, h, w = spec
        if mask.shape[0] > h or mask.shape[1] > w:
            return None
        _paint_template(grid[r0 : r0 + h, c0 : c0 + w], mask, color, rng)

    winners = [color_a if bool(rng.integers(0, 2)) else color_b for _ in lower_regions]
    if len(set(winners)) == 1:
        winners[int(rng.integers(0, len(winners)))] = color_b if winners[0] == color_a else color_a

    for spec, winner in zip(lower_regions, winners):
        _band, _col, r0, c0, h, w = spec
        loser = color_b if winner == color_a else color_a
        region = grid[r0 : r0 + h, c0 : c0 + w]
        _paint_votes(region, winner, loser, rng)

    if _background_color(grid) != bg:
        return None

    try:
        _analyze_grid(grid)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
