from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _canonicalize_band(band_grid: np.ndarray) -> tuple[np.ndarray, bool]:
    coords = np.argwhere(band_grid != BG)
    assert coords.size > 0, "expected a non-empty band"
    diag_offsets = coords[:, 1] - coords[:, 0]
    anti_offsets = coords[:, 1] + coords[:, 0]
    diag_span = int(diag_offsets.max() - diag_offsets.min())
    anti_span = int(anti_offsets.max() - anti_offsets.min())
    assert diag_span != anti_span, "expected one dominant diagonal direction"
    if diag_span < anti_span:
        return band_grid.copy(), False
    return np.fliplr(band_grid), True


def _learn_band(canonical_band: np.ndarray) -> tuple[list[int], int, dict[tuple[int, int], int]]:
    h, w = canonical_band.shape
    coords = np.argwhere(canonical_band != BG)
    offsets = sorted({int(c - r) for r, c in coords})
    assert 1 <= len(offsets) <= 2, "expected a 1-2 pixel diagonal band"
    if len(offsets) == 2:
        assert offsets[1] - offsets[0] == 1, "band offsets must be consecutive"

    observations: list[tuple[int, int, int]] = []
    for r in range(h):
        for off in offsets:
            c = r + off
            if 0 <= c < w:
                observations.append((r, off, int(canonical_band[r, c])))
    assert observations, "expected visible band observations"

    for period in range(1, h + 1):
        if not all(any(rr % period == phase and 0 <= rr + off < w for rr in range(h)) for off in offsets for phase in range(period)):
            continue
        mapping: dict[tuple[int, int], int] = {}
        ok = True
        for r, off, color in observations:
            key = (off, r % period)
            prev = mapping.get(key)
            if prev is None:
                mapping[key] = color
            elif prev != color:
                ok = False
                break
        if ok and len(mapping) == len(offsets) * period:
            return offsets, period, mapping
    raise AssertionError("failed to learn a consistent row-periodic band")


def _paint_band(
    height: int,
    width: int,
    offsets: list[int],
    period: int,
    pattern: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.full((2 * height, 2 * width), BG, dtype=int)
    for r in range(2 * height):
        for off in offsets:
            c = r + off
            if 0 <= c < 2 * width:
                color = int(pattern[(off, r % period)])
                if color != BG:
                    out[r, c] = color
    return out


def _extract_band_and_markers(grid: np.ndarray) -> tuple[np.ndarray, Counter[int]]:
    fg_mask = grid != BG
    comps = _components(fg_mask, connectivity=8)
    assert comps, "expected a foreground band and markers"
    comps.sort(key=len, reverse=True)
    assert len(comps[0]) >= 2, "band must have at least two cells"
    if len(comps) > 1:
        assert len(comps[0]) > len(comps[1]), "band must be the unique largest 8-connected component"

    band_mask = np.zeros_like(fg_mask, dtype=bool)
    band_mask[comps[0][:, 0], comps[0][:, 1]] = True

    marker_counts: Counter[int] = Counter()
    for comp in comps[1:]:
        assert len(comp) == 1, "markers must be isolated single pixels"
        r, c = map(int, comp[0])
        marker_counts[int(grid[r, c])] += 1
    assert marker_counts, "expected at least one scattered marker color"
    return band_mask, marker_counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the diagonal periodic band to a doubled canvas, then replace unsupported band colors using the rarest unused marker colors.

    Input structure:
    - The background is black.
    - One narrow diagonal band is the unique largest 8-connected foreground component; it slopes either down-right or down-left, uses 1-2 adjacent diagonal offsets, and follows a row-periodic color pattern that may contain background gaps inside the band.
    - Away from the band, isolated single-pixel markers use several colors with distinct frequencies.

    Rule:
    - Separate the band from the scattered markers, canonicalize the band to the down-right direction, and learn its occupied diagonal offsets plus the smallest row period of the visible colors and gaps.
    - Paint that same periodic band onto a 2H x 2W black canvas, undoing the horizontal flip when the original band sloped down-left.
    - Let B be the non-black colors used by the extended band and M the marker colors. Unsupported colors U = B - M are sorted by their frequency inside the extended band from rare to common, while replacement colors R = M - B are sorted by their marker counts from rare to common; recolor each U color to the correspondingly ranked R color.

    Key insight:
    The scattered markers do not move; their color set and multiplicities define a replacement ranking that is applied only after the diagonal band has first been reconstructed on the doubled canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(map(int, g.shape)) <= 30, "ARC grids must fit within 30x30"

    band_mask, marker_counts = _extract_band_and_markers(g)
    band_grid = np.where(band_mask, g, BG)
    canonical_band, flipped = _canonicalize_band(band_grid)
    offsets, period, pattern = _learn_band(canonical_band)
    extended = _paint_band(g.shape[0], g.shape[1], offsets, period, pattern)
    assert np.array_equal(extended[: g.shape[0], : g.shape[1]], canonical_band), "learned band must reproduce the crop"
    if flipped:
        extended = np.fliplr(extended)

    band_colors = {int(x) for x in np.unique(extended) if int(x) != BG}
    marker_colors = set(marker_counts)
    unsupported = [color for color in band_colors if color not in marker_colors]
    replacements = [color for color in marker_colors if color not in band_colors]
    assert len(unsupported) == len(replacements) > 0, "band-only and marker-only color counts must match"

    band_freq = Counter(int(x) for x in extended.ravel() if int(x) != BG)
    assert len({marker_counts[c] for c in marker_counts}) == len(marker_counts), "marker color frequencies must be distinct"
    assert len({band_freq[c] for c in unsupported}) == len(unsupported), "unsupported band colors must have distinct frequencies"

    unsupported_sorted = sorted(unsupported, key=lambda c: band_freq[c])
    replacements_sorted = sorted(replacements, key=lambda c: marker_counts[c])
    mapping = dict(zip(unsupported_sorted, replacements_sorted))

    out = extended.copy()
    for src, dst in mapping.items():
        out[extended == src] = dst
    return out


def _minimal_period(tile: np.ndarray) -> int:
    rows = tile.shape[0]
    for period in range(1, rows + 1):
        if all(np.array_equal(tile[r], tile[r % period]) for r in range(rows)):
            return period
    return rows


def _paint_canonical_from_tile(size: int, offsets: tuple[int, ...], tile: np.ndarray) -> np.ndarray:
    out = np.full((2 * size, 2 * size), BG, dtype=int)
    period = tile.shape[0]
    for r in range(2 * size):
        for idx, off in enumerate(offsets):
            c = r + off
            if 0 <= c < 2 * size:
                color = int(tile[r % period, idx])
                if color != BG:
                    out[r, c] = color
    return out


def _band_design() -> tuple[int, int, tuple[int, ...], list[int], list[int], list[int]]:
    size = random.randint(8, 13)
    valid_periods = [p for p in (2, 3, 4) if size >= 2 * p + 1]
    period = random.choice(valid_periods)
    offsets = random.choice([(0,), (-1, 0), (0, 1)])
    max_band_colors = min(3, period * len(offsets))

    unsupported_choices = [1]
    if max_band_colors >= 3:
        unsupported_choices.append(2)
    unsupported_count = random.choice(unsupported_choices)
    max_supported = min(2, max_band_colors - unsupported_count)
    if unsupported_count == 2:
        supported_count = 1
    else:
        supported_count = random.randint(1, max_supported)
    band_color_count = supported_count + unsupported_count
    band_colors = random.sample(PALETTE, band_color_count)
    supported_colors = band_colors[:supported_count]
    unsupported_colors = band_colors[supported_count:]
    reserve_colors = random.sample([c for c in PALETTE if c not in band_colors], unsupported_count)
    return size, period, offsets, supported_colors, unsupported_colors, reserve_colors


def _random_tile(
    period: int,
    offsets: tuple[int, ...],
    colors: list[int],
) -> np.ndarray:
    width = len(offsets)
    use_gaps = width == 2 and random.random() < 0.6
    tile = np.full((period, width), BG, dtype=int)
    for _ in range(200):
        for r in range(period):
            if width == 1:
                tile[r, 0] = random.choice(colors)
                continue
            row = [random.choice(colors), random.choice(colors)]
            if use_gaps and random.random() < 0.7:
                row[random.randrange(2)] = BG
            if row[0] == BG and row[1] == BG:
                row[random.randrange(2)] = random.choice(colors)
            tile[r] = row

        if use_gaps and BG not in tile:
            continue
        if any(np.all(tile[:, idx] == BG) for idx in range(width)):
            continue
        if set(colors) - {int(x) for x in tile.ravel() if int(x) != BG}:
            continue
        if _minimal_period(tile) != period:
            continue
        return tile.copy()
    raise ValueError("failed to sample a valid tile")


def _marker_candidates(band_crop: np.ndarray) -> list[tuple[int, int]]:
    h, w = band_crop.shape
    band_cells = [tuple(map(int, p)) for p in np.argwhere(band_crop != BG)]
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if band_crop[r, c] != BG:
                continue
            if all(max(abs(r - br), abs(c - bc)) >= 2 for br, bc in band_cells):
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _choose_isolated_positions(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) >= 2 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == count:
                return chosen
    return None


def _semantic_checks(
    inp: np.ndarray,
    extended_before_recolor: np.ndarray,
    expected: np.ndarray,
    unsupported_colors: list[int],
    reserve_colors: list[int],
) -> None:
    band_mask, marker_counts = _extract_band_and_markers(inp)
    assert np.any(band_mask)
    assert len(marker_counts) >= 2
    assert len({marker_counts[c] for c in marker_counts}) == len(marker_counts)
    assert np.array_equal(main(inp), expected)
    assert not np.array_equal(extended_before_recolor, expected), "replacement step must matter"
    if len(unsupported_colors) == 2:
        swapped = expected.copy()
        swapped[expected == reserve_colors[0]] = 99
        swapped[expected == reserve_colors[1]] = reserve_colors[0]
        swapped[swapped == 99] = reserve_colors[1]
        assert not np.array_equal(swapped, expected), "frequency ranking must affect the output"


def _gen_attempt() -> np.ndarray | None:
    size, period, offsets, supported_colors, unsupported_colors, reserve_colors = _band_design()
    band_colors = supported_colors + unsupported_colors

    for _ in range(80):
        try:
            tile = _random_tile(period, offsets, band_colors)
        except ValueError:
            continue
        canonical_full = _paint_canonical_from_tile(size, offsets, tile)
        canonical_crop = canonical_full[:size, :size].copy()
        if len(_components(canonical_crop != BG, connectivity=8)) != 1:
            continue
        if len(_components(canonical_full != BG, connectivity=8)) != 1:
            continue

        direction_is_slash = random.random() < 0.5
        base_crop = np.fliplr(canonical_crop) if direction_is_slash else canonical_crop
        extended_before_recolor = np.fliplr(canonical_full) if direction_is_slash else canonical_full

        band_freq = Counter(int(x) for x in extended_before_recolor.ravel() if int(x) != BG)
        if len({band_freq[c] for c in unsupported_colors}) != len(unsupported_colors):
            continue

        marker_colors = supported_colors + reserve_colors
        max_marker_count = 5 if len(marker_colors) == 2 else 4
        count_pool = list(range(1, max_marker_count + 1))
        if len(marker_colors) > len(count_pool):
            continue
        marker_color_counts = random.sample(count_pool, len(marker_colors))
        total_markers = sum(marker_color_counts)

        candidates = _marker_candidates(base_crop)
        positions = _choose_isolated_positions(candidates, total_markers)
        if positions is None:
            continue

        inp = base_crop.copy()
        idx = 0
        for color, count in zip(marker_colors, marker_color_counts):
            for _ in range(count):
                r, c = positions[idx]
                idx += 1
                inp[r, c] = color

        replacement_map = {
            src: dst
            for src, dst in zip(
                sorted(unsupported_colors, key=lambda c: band_freq[c]),
                sorted(reserve_colors, key=lambda c: marker_color_counts[marker_colors.index(c)]),
            )
        }
        expected = extended_before_recolor.copy()
        for src, dst in replacement_map.items():
            expected[extended_before_recolor == src] = dst

        try:
            _semantic_checks(inp, extended_before_recolor, expected, unsupported_colors, reserve_colors)
        except AssertionError:
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
