from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
MOTIF_TOP = 0
MOTIF_LEFT = 0
MOTIF_SHAPE = (3, 3)
MOTIF_COLORS = (1, 2)
MARKER_PALETTE = (3, 4, 5, 6, 7, 8, 9)
HEIGHT_RANGE = (18, 27)
MARKER_COUNT_CHOICES = (3, 4)
MAX_GENERATION_ATTEMPTS = 600


def get_bg(grid: np.ndarray) -> int:
    """Background color = most common value in the grid."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    """Scale a sprite by integer replication, matching arc_utils semantics."""
    k = int(factor)
    if k <= 0:
        raise ValueError("scale factor must be >= 1")
    return np.kron(np.asarray(sprite, dtype=int), np.ones((k, k), dtype=int))


def paste_clip(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> np.ndarray:
    """Paste with clipping, using the same (r, c) top-left convention as arc_utils."""
    out = np.asarray(dst)
    patch = np.asarray(patch)
    h, w = patch.shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(out.shape[0], int(r0) + h)
    cc1 = min(out.shape[1], int(c0) + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return out
    pr0 = rr0 - int(r0)
    pc0 = cc0 - int(c0)
    out[rr0:rr1, cc0:cc1] = patch[pr0 : pr0 + (rr1 - rr0), pc0 : pc0 + (cc1 - cc0)]
    return out


def _extract_motif(grid: np.ndarray) -> np.ndarray:
    h, w = map(int, grid.shape)
    assert h >= MOTIF_SHAPE[0] and w >= MOTIF_SHAPE[1], "grid must contain the 3x3 motif"
    motif = np.asarray(grid, dtype=int)[MOTIF_TOP : MOTIF_TOP + 3, MOTIF_LEFT : MOTIF_LEFT + 3].copy()
    assert motif.shape == MOTIF_SHAPE
    assert np.all(motif != BACKGROUND), "base motif must fully occupy its 3x3 slot"
    return motif


def _extract_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    marker_mask = g != int(bg)
    marker_mask[MOTIF_TOP : MOTIF_TOP + 3, MOTIF_LEFT : MOTIF_LEFT + 3] = False
    coords = np.argwhere(marker_mask)
    assert 3 <= int(coords.shape[0]) <= 4, "expected 3 or 4 singleton markers"
    cols = {int(c) for _r, c in coords}
    assert len(cols) == 1, "all markers must be in one fixed column"
    markers = [(int(r), int(c), int(g[r, c])) for r, c in coords]
    rows = [r for r, _c, _color in markers]
    colors = [color for _r, _c, color in markers]
    assert len(set(rows)) == len(rows), "marker rows must be distinct"
    assert len(set(colors)) == len(colors), "marker colors must be distinct"
    markers.sort(key=lambda item: item[0])
    return markers


def _band_bounds(height: int, marker_rows: list[int], midpoint_mode: str = "floor") -> list[tuple[int, int]]:
    mids: list[int] = []
    for a, b in zip(marker_rows, marker_rows[1:]):
        if midpoint_mode == "floor":
            mid = (int(a) + int(b)) // 2
        elif midpoint_mode == "ceil":
            mid = (int(a) + int(b) + 1) // 2
        else:
            raise ValueError(f"unknown midpoint_mode: {midpoint_mode}")
        mids.append(int(mid))
    tops = [0] + [mid + 1 for mid in mids]
    bottoms = mids + [int(height) - 1]
    bands = [(int(top), int(bottom)) for top, bottom in zip(tops, bottoms)]
    assert len(bands) == len(marker_rows)
    return bands


def _render_output(
    shape: tuple[int, int],
    motif: np.ndarray,
    markers: list[tuple[int, int, int]],
    *,
    midpoint_mode: str = "floor",
    scale_mode: str = "band",
) -> np.ndarray:
    h, w = map(int, shape)
    motif = np.asarray(motif, dtype=int)
    rows = [int(r) for r, _c, _color in markers]
    bands = _band_bounds(h, rows, midpoint_mode=midpoint_mode)
    out = np.full((h, w), BACKGROUND, dtype=int)
    motif_h, motif_w = map(int, motif.shape)
    gaps = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]

    for idx, ((top, bottom), (_r, _c, color)) in enumerate(zip(bands, markers)):
        band_h = int(bottom - top + 1)
        if scale_mode == "band":
            scale = band_h // motif_h
        elif scale_mode == "constant1":
            scale = 1 if band_h >= motif_h else 0
        elif scale_mode == "gap":
            if not gaps:
                gap = band_h
            elif idx == 0:
                gap = gaps[0]
            elif idx == len(rows) - 1:
                gap = gaps[-1]
            else:
                gap = min(gaps[idx - 1], gaps[idx])
            scale = gap // motif_h
        else:
            raise ValueError(f"unknown scale_mode: {scale_mode}")

        if scale >= 1:
            stamp = scale_sprite(motif, scale)
            left = (w - int(stamp.shape[1])) // 2
            paste_clip(out, stamp, top, left)

        out[top, :] = int(color)
        out[bottom, :] = int(color)
        out[top : bottom + 1, 0] = int(color)
        out[top : bottom + 1, w - 1] = int(color)

    for r, c, color in markers:
        out[int(r), int(c)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition the grid into marker-owned horizontal bands and place band-scaled motif copies.

    Input structure:
    - The background is black, and a 3x3 two-color motif sits in the upper-left corner.
    - Three or four single-pixel markers of distinct colors appear in one fixed middle column at distinct, non-consecutive rows.
    - The marker rows implicitly divide the canvas into horizontal bands by midpoint ownership.

    Rule:
    - Sort the markers by row and split the grid at the floor midpoints between adjacent marker rows.
    - For each band of height H, compute the scale factor s = floor(H / 3); if s >= 1, place the 3x3 motif enlarged to 3s x 3s, top-aligned in the band and horizontally centered, clipping only if needed.
    - Draw the full border of each band in that band's marker color, then restore the original marker pixels on top.

    Key insight:
    - The irregular vertical spacing of the markers is not just a partition cue: through band height, it encodes the scale of the motif copy that appears in each band.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    motif = _extract_motif(g)
    markers = _extract_markers(g)
    return _render_output(g.shape, motif, markers)


def _random_motif() -> np.ndarray:
    for _ in range(200):
        motif = np.random.choice(MOTIF_COLORS, size=MOTIF_SHAPE, replace=True).astype(int)
        counts = Counter(map(int, motif.ravel()))
        row_patterns = {tuple(map(int, row)) for row in motif}
        col_patterns = {tuple(map(int, col)) for col in motif.T}
        if min(counts.values()) < 2:
            continue
        if len(row_patterns) < 2 or len(col_patterns) < 2:
            continue
        if np.array_equal(motif, motif[:, ::-1]) and np.array_equal(motif, motif[::-1, :]):
            continue
        return motif
    return np.array([[1, 2, 1], [2, 1, 2], [1, 1, 2]], dtype=int)


def _sample_marker_rows(height: int, count: int, want_zero_band: bool) -> list[int] | None:
    assert count in MARKER_COUNT_CHOICES
    for _ in range(400):
        gaps = [random.randint(3, 9) for _ in range(count - 1)]
        if want_zero_band:
            if random.random() < 0.5:
                gaps[0] = 3
                force_top = True
            else:
                gaps[-1] = 3
                force_top = False
        else:
            force_top = None
        if len(set(gaps)) == 1:
            continue
        total_gap = sum(gaps)
        slack = int(height) - 1 - total_gap
        if slack < 0:
            continue

        if want_zero_band:
            start = 0 if force_top else slack
        else:
            start = random.randint(0, slack)

        rows = [int(start)]
        cur = int(start)
        for gap in gaps:
            cur += int(gap)
            rows.append(cur)

        bands = _band_bounds(height, rows, midpoint_mode="floor")
        heights = [bottom - top + 1 for top, bottom in bands]
        scales = [band_h // 3 for band_h in heights]
        if want_zero_band and not any(scale == 0 for scale in scales):
            continue
        if len(set(scales)) < 2:
            continue
        if sum(scale >= 1 for scale in scales) < 2:
            continue
        if not any(scale != 1 for scale in scales):
            continue
        if max(scales) > 4:
            continue
        if not any((rows[i] + rows[i + 1]) % 2 == 1 for i in range(len(rows) - 1)):
            continue
        return rows
    return None


def _validate_sample(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    motif = _extract_motif(g)
    markers = _extract_markers(g)
    h, w = map(int, g.shape)
    rows = [int(r) for r, _c, _color in markers]
    cols = [int(c) for _r, c, _color in markers]
    colors = [int(color) for _r, _c, color in markers]
    gaps = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
    bands = _band_bounds(h, rows, midpoint_mode="floor")
    heights = [bottom - top + 1 for top, bottom in bands]
    scales = [band_h // 3 for band_h in heights]

    assert len(markers) in (3, 4)
    assert len(set(cols)) == 1
    assert len(set(colors)) == len(colors)
    assert all(3 <= gap <= 9 for gap in gaps)
    assert len(set(gaps)) >= 2
    assert len(set(scales)) >= 2
    assert sum(scale >= 1 for scale in scales) >= 2
    assert any(scale != 1 for scale in scales)
    assert max(scales) <= w // 3, "generator should keep full-width stamps visible"
    assert any((rows[i] + rows[i + 1]) % 2 == 1 for i in range(len(rows) - 1))

    for (top, bottom), row in zip(bands, rows):
        assert int(top) <= int(row) <= int(bottom), "each marker must own exactly one band"

    actual = main(g)
    assert actual.shape == g.shape
    assert not np.array_equal(actual, g), "puzzle should not collapse to identity"

    wrong_mid = _render_output(g.shape, motif, markers, midpoint_mode="ceil", scale_mode="band")
    wrong_scale = _render_output(g.shape, motif, markers, midpoint_mode="floor", scale_mode="constant1")
    wrong_gap = _render_output(g.shape, motif, markers, midpoint_mode="floor", scale_mode="gap")
    assert not np.array_equal(actual, wrong_mid), "floor midpoint ownership must matter"
    assert not np.array_equal(actual, wrong_scale), "band-height scaling must matter"
    assert not np.array_equal(actual, wrong_gap), "raw marker gaps must not replace band-height scaling"


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    marker_count = random.choice(MARKER_COUNT_CHOICES)
    want_zero_band = random.random() < 0.45
    rows = _sample_marker_rows(height, marker_count, want_zero_band=want_zero_band)
    if rows is None:
        return None

    bands = _band_bounds(height, rows, midpoint_mode="floor")
    scales = [(bottom - top + 1) // 3 for top, bottom in bands]
    max_scale = max(scales)
    min_width = max(8, 3 * max_scale)
    max_width = min(14, min_width + 3)
    if min_width > max_width:
        return None
    width = random.randint(min_width, max_width)
    marker_col = width // 2
    if marker_col <= 2:
        return None

    motif = _random_motif()
    marker_colors = random.sample(MARKER_PALETTE, marker_count)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[:3, :3] = motif
    for row, color in zip(rows, marker_colors):
        grid[int(row), int(marker_col)] = int(color)

    _validate_sample(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _gen_attempt()
        except AssertionError:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
