from __future__ import annotations

import numpy as np


BG = 0
MARKER = 8
TOP = "top"
BOTTOM = "bottom"
LEFT = "left"
RIGHT = "right"
OPPOSITE = {TOP: BOTTOM, BOTTOM: TOP, LEFT: RIGHT, RIGHT: LEFT}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MAX_GEN_ATTEMPTS = 500
MAX_BAND_ATTEMPTS = 2000


def _edge_from_marker(r: int, c: int, shape: tuple[int, int]) -> str:
    h, w = shape
    on = []
    if r == 0:
        on.append(TOP)
    if r == h - 1:
        on.append(BOTTOM)
    if c == 0:
        on.append(LEFT)
    if c == w - 1:
        on.append(RIGHT)
    assert len(on) == 1, "marker must lie on exactly one non-corner edge"
    return on[0]


def _count_same_neighbors(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    total = 0
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
            total += 1
    return total


def _supported_mask_bool(mask: np.ndarray) -> np.ndarray:
    supported = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        supported[r, c] = _count_same_neighbors(mask, int(r), int(c)) >= 2
    return supported


def _clean_bool(mask: np.ndarray) -> np.ndarray:
    supported = _supported_mask_bool(mask)
    return mask & supported


def _clean_band(band: np.ndarray, fg: int) -> np.ndarray:
    mask = band == fg
    keep = _supported_mask_bool(mask)
    out = band.copy()
    out[mask & ~keep] = BG
    return out


def _extract_band(grid: np.ndarray, fg: int, edge: str) -> np.ndarray:
    coords = np.argwhere(grid == fg)
    assert coords.size > 0, "expected a non-marker foreground band"
    rows = coords[:, 0]
    cols = coords[:, 1]
    if edge == TOP:
        assert int(rows.min()) == 0
        thickness = int(rows.max()) + 1
        return grid[:thickness, :].copy()
    if edge == BOTTOM:
        assert int(rows.max()) == grid.shape[0] - 1
        thickness = grid.shape[0] - int(rows.min())
        return grid[-thickness:, :].copy()
    if edge == LEFT:
        assert int(cols.min()) == 0
        thickness = int(cols.max()) + 1
        return grid[:, :thickness].copy()
    assert edge == RIGHT
    assert int(cols.max()) == grid.shape[1] - 1
    thickness = grid.shape[1] - int(cols.min())
    return grid[:, -thickness:].copy()


def _tile_seed(seed: np.ndarray, band_edge: str, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.zeros(shape, dtype=int)
    if band_edge == TOP:
        step = seed.shape[0]
        for r in range(h):
            out[r, :] = seed[r % step, :]
        return out
    if band_edge == BOTTOM:
        step = seed.shape[0]
        start = h - step
        for r in range(h):
            out[r, :] = seed[(r - start) % step, :]
        return out
    if band_edge == LEFT:
        step = seed.shape[1]
        for c in range(w):
            out[:, c] = seed[:, c % step]
        return out

    step = seed.shape[1]
    start = w - step
    for c in range(w):
        out[:, c] = seed[:, (c - start) % step]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Remove weakly supported noise from the edge band, then tile the cleaned band toward the teal-marked opposite edge.

    Input structure:
    - A black grid contains one non-black edge band touching exactly one side of the grid.
    - The band is 3-5 cells thick, includes black gaps, and mixes true pattern pixels with same-color noise pixels.
    - A single teal marker lies on the opposite edge and indicates the tiling target side.

    Rule:
    - In the band, keep only foreground pixels that have at least two same-color 4-neighbors; turn weaker pixels black.
    - Treat the cleaned band, including its internal black gaps, as the seed motif.
    - Repeat that seed away from the band along the perpendicular axis until the whole grid is filled.
    - Finally erase the teal marker cell back to black.

    Key insight:
    - Noise cannot be identified by color because it matches the pattern color, so local 4-neighbor support must be used before tiling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    marker_cells = np.argwhere(g == MARKER)
    assert marker_cells.shape == (1, 2), "expected exactly one teal marker"
    marker_r, marker_c = map(int, marker_cells[0])
    marker_edge = _edge_from_marker(marker_r, marker_c, g.shape)
    band_edge = OPPOSITE[marker_edge]

    colors = sorted(int(v) for v in np.unique(g) if int(v) not in (BG, MARKER))
    assert len(colors) == 1, "expected exactly one non-marker foreground color"
    fg = colors[0]

    band = _extract_band(g, fg, band_edge)
    seed = _clean_band(band, fg)
    out = _tile_seed(seed, band_edge, g.shape)
    out[marker_r, marker_c] = BG
    return out


def _axis_period(seed: np.ndarray, band_edge: str) -> int:
    if band_edge in (TOP, BOTTOM):
        lines = [tuple(int(v) for v in row) for row in seed]
    else:
        lines = [tuple(int(v) for v in seed[:, c]) for c in range(seed.shape[1])]
    n = len(lines)
    for p in range(1, n + 1):
        if n % p == 0 and all(lines[i] == lines[i % p] for i in range(n)):
            return p
    return n


def _noise_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c]:
                continue
            if _count_same_neighbors(mask, r, c) <= 1:
                candidates.append((r, c))
    return candidates


def _line_variety(seed: np.ndarray, band_edge: str) -> tuple[int, int]:
    if band_edge in (TOP, BOTTOM):
        short_axis = len({tuple(int(v) for v in row) for row in seed})
        long_axis = len({tuple(int(v) for v in seed[:, c]) for c in range(seed.shape[1])})
        return short_axis, long_axis
    short_axis = len({tuple(int(v) for v in seed[:, c]) for c in range(seed.shape[1])})
    long_axis = len({tuple(int(v) for v in row) for row in seed})
    return short_axis, long_axis


def _generate_clean_mask(
    rng: np.random.Generator,
    band_edge: str,
    short: int,
    long: int,
    period: int,
    repeats: int,
) -> np.ndarray:
    unit_shape = (short, period) if band_edge in (TOP, BOTTOM) else (period, short)
    for _ in range(MAX_BAND_ATTEMPTS):
        density = float(rng.uniform(0.35, 0.60))
        unit = rng.random(unit_shape) < density
        if band_edge in (TOP, BOTTOM):
            mask = np.tile(unit, (1, repeats))
            if np.any(~mask.any(axis=1)):
                continue
        else:
            mask = np.tile(unit, (repeats, 1))
            if np.any(~mask.any(axis=0)):
                continue

        if not mask.any() or mask.all():
            continue
        if not np.all(_supported_mask_bool(mask)[mask]):
            continue
        if _axis_period(mask.astype(int), band_edge) != short:
            continue
        short_variety, long_variety = _line_variety(mask.astype(int), band_edge)
        if short_variety < 2 or long_variety < 2:
            continue
        if len(_noise_candidates(mask)) < 2:
            continue
        return mask
    raise RuntimeError("failed to generate a non-degenerate clean band")


def _inject_noise(
    rng: np.random.Generator, clean_mask: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    raw = clean_mask.copy()
    candidates = _noise_candidates(raw)
    rng.shuffle(candidates)
    target = int(rng.integers(1, min(4, len(candidates)) + 1))
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in chosen):
            continue
        raw[r, c] = True
        if np.array_equal(_clean_bool(raw), clean_mask):
            chosen.append((r, c))
            if len(chosen) == target:
                break
        else:
            raw[r, c] = False

    if not chosen:
        return None
    for r, c in chosen:
        if _count_same_neighbors(raw, r, c) > 1:
            return None
    if not np.array_equal(_clean_bool(raw), clean_mask):
        return None
    return raw, chosen


def _paint_band(mask: np.ndarray, fg: int) -> np.ndarray:
    band = np.zeros(mask.shape, dtype=int)
    band[mask] = fg
    return band


def _place_band(grid: np.ndarray, band: np.ndarray, band_edge: str) -> None:
    if band_edge == TOP:
        grid[: band.shape[0], :] = band
    elif band_edge == BOTTOM:
        grid[-band.shape[0] :, :] = band
    elif band_edge == LEFT:
        grid[:, : band.shape[1]] = band
    else:
        grid[:, -band.shape[1] :] = band


def _edge_positions(shape: tuple[int, int], edge: str) -> list[tuple[int, int]]:
    h, w = shape
    if edge == TOP:
        return [(0, c) for c in range(1, w - 1)]
    if edge == BOTTOM:
        return [(h - 1, c) for c in range(1, w - 1)]
    if edge == LEFT:
        return [(r, 0) for r in range(1, h - 1)]
    return [(r, w - 1) for r in range(1, h - 1)]


def _render_expected(
    shape: tuple[int, int], clean_band: np.ndarray, band_edge: str, marker_pos: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    tiled = _tile_seed(clean_band, band_edge, shape)
    out = tiled.copy()
    out[marker_pos] = BG
    return tiled, out


def _validate_example(
    inp: np.ndarray,
    clean_band: np.ndarray,
    raw_band: np.ndarray,
    band_edge: str,
    marker_pos: tuple[int, int],
    fg: int,
) -> None:
    marker_edge = OPPOSITE[band_edge]
    assert _edge_from_marker(marker_pos[0], marker_pos[1], inp.shape) == marker_edge
    assert np.count_nonzero(inp == MARKER) == 1
    assert np.array_equal(_clean_band(raw_band, fg), clean_band)

    tiled, expected = _render_expected(inp.shape, clean_band, band_edge, marker_pos)
    assert tiled[marker_pos] == fg, "marker erasure must change the tiled result"
    assert np.array_equal(main(inp), expected)

    raw_tiled = _tile_seed(raw_band, band_edge, inp.shape)
    raw_tiled[marker_pos] = BG
    assert not np.array_equal(raw_tiled, expected), "noise-cleaning must affect the output"

    assert not np.array_equal(tiled, expected), "marker removal must affect the output"
    assert not np.array_equal(inp, expected), "tiling must change the grid beyond the input band"

    recovered = _extract_band(inp, fg, band_edge)
    assert np.array_equal(recovered, raw_band)
    assert np.any(raw_band != clean_band), "input must actually contain removable noise"
    assert _axis_period(clean_band, band_edge) == (clean_band.shape[0] if band_edge in (TOP, BOTTOM) else clean_band.shape[1])


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    band_edges = [TOP, BOTTOM, LEFT, RIGHT]
    fg_choices = [1, 2, 3, 4, 5, 6, 7, 9]

    for _ in range(MAX_GEN_ATTEMPTS):
        band_edge = band_edges[int(rng.integers(0, len(band_edges)))]
        marker_edge = OPPOSITE[band_edge]
        thickness = int(rng.integers(3, 6))
        tile_count = int(rng.integers(3, 6))
        period = int(rng.integers(3, 6))
        repeats = int(rng.integers(3, 6))
        fg = int(fg_choices[int(rng.integers(0, len(fg_choices)))])

        if band_edge in (TOP, BOTTOM):
            h = thickness * tile_count
            w = period * repeats
            short = thickness
            long = w
        else:
            h = period * repeats
            w = thickness * tile_count
            short = thickness
            long = h

        if max(h, w) > 30:
            continue

        try:
            clean_mask = _generate_clean_mask(rng, band_edge, short, long, period, repeats)
        except RuntimeError:
            continue
        injected = _inject_noise(rng, clean_mask)
        if injected is None:
            continue
        raw_mask, _noise = injected

        clean_band = _paint_band(clean_mask, fg)
        raw_band = _paint_band(raw_mask, fg)
        tiled = _tile_seed(clean_band, band_edge, (h, w))
        marker_candidates = [pos for pos in _edge_positions((h, w), marker_edge) if tiled[pos] == fg]
        if not marker_candidates:
            continue
        marker_pos = marker_candidates[int(rng.integers(0, len(marker_candidates)))]

        inp = np.zeros((h, w), dtype=int)
        _place_band(inp, raw_band, band_edge)
        inp[marker_pos] = MARKER

        try:
            _validate_example(inp, clean_band, raw_band, band_edge, marker_pos, fg)
        except AssertionError:
            continue
        return inp

    raise RuntimeError("failed to generate a valid input for the new ARC task")
