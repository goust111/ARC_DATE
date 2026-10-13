from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.exists() and str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

try:
    from arc_utils import bbox, get_bg
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
        return int(vals[np.argmax(counts)])

    def bbox(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)


BG = 0
MARKER = 5
MOTIF_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
MAX_GENERATION_ATTEMPTS = 400


def _stretch(patch: np.ndarray, scale_r: int, scale_c: int) -> np.ndarray:
    if scale_r <= 0 or scale_c <= 0:
        raise AssertionError("scale factors must be positive")
    return np.repeat(np.repeat(patch, int(scale_r), axis=0), int(scale_c), axis=1)


def _parse_input(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    assert bg == BG, "background must be black"

    motif_mask = (g != bg) & (g != MARKER)
    box = bbox(motif_mask)
    assert box is not None, "expected a top-left motif"
    r0, c0, r1, c1 = map(int, box)
    assert (r0, c0) == (0, 0), "motif must touch the top-left corner"

    motif = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert np.any(motif != bg), "motif crop must contain foreground"

    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == MARKER)]
    assert markers, "expected at least one gray marker"
    marker_set = set(markers)
    for r, c in markers:
        assert r >= r1 + 2 and c >= c1 + 2, "markers must lie strictly below-right of the motif"
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) != (r, c):
                    assert (rr, cc) not in marker_set, "markers must be isolated single pixels"

    scales = []
    for r, c in sorted(markers):
        dr = r - r1 - 1
        dc = c - c1 - 1
        assert dr >= 1 and dc >= 1
        scales.append((r, c, dr, dc))

    return g, motif, (r0, c0, r1, c1), scales


def _render(grid: np.ndarray, scale_mode) -> np.ndarray:
    g, motif, box, markers = _parse_input(grid)
    _r0, _c0, r1, c1 = box
    motif_h, motif_w = motif.shape

    out = g.copy()
    used = np.zeros(g.shape, dtype=bool)
    used[: r1 + 1, : c1 + 1] = True

    for r, c, dr, dc in markers:
        scale_r, scale_c = scale_mode(dr, dc)
        stamp = _stretch(motif, scale_r, scale_c)
        h, w = stamp.shape
        assert r + h <= g.shape[0] and c + w <= g.shape[1], "scaled copy must fit"

        region = g[r : r + h, c : c + w]
        assert region[0, 0] == MARKER, "marker must anchor the copy"
        assert np.count_nonzero(region != BG) == 1, "stamp regions must be empty except for their marker"
        assert not used[r : r + h, c : c + w].any(), "scaled copies must not overlap"

        out[r : r + h, c : c + w] = stamp
        used[r : r + h, c : c + w] = True

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp anisotropically stretched copies of the top-left motif onto every gray marker.

    Input structure:
    - The grid has a black background and a multicolor motif cropped against the top-left corner.
    - Every other foreground cell is a gray single-pixel marker located strictly below and to the right of the motif.
    - For each marker, the row gap below the motif bbox and the column gap to the right of the motif bbox are both at least one cell.

    Rule:
    - Keep the original motif unchanged.
    - For each gray marker, compute `dr` as the number of background rows between the motif bbox bottom edge and the marker row, and `dc` as the analogous background-column gap from the motif bbox right edge.
    - Replace that marker with a copy of the motif where each motif pixel becomes a `dr`-by-`dc` block, with the stretched copy's top-left cell placed on the marker.

    Key insight:
    - The gray marker encodes two independent scale factors at once: vertical scale comes only from the vertical gap, and horizontal scale comes only from the horizontal gap.
    """
    return _render(grid, lambda dr, dc: (dr, dc))


def _random_motif(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    hole_mask = np.zeros((h, w), dtype=bool)
    if h >= 3 and w >= 3 and rng.random() < 0.5:
        interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        hole_count = int(rng.integers(1, min(2, len(interior)) + 1))
        picks = rng.choice(len(interior), size=hole_count, replace=False)
        for idx in np.atleast_1d(picks):
            r, c = interior[int(idx)]
            hole_mask[r, c] = True

    coords = np.argwhere(~hole_mask)
    color_count = int(rng.integers(2, min(5, len(coords)) + 1))
    palette = rng.choice(MOTIF_COLORS, size=color_count, replace=False)
    vals = np.empty(len(coords), dtype=int)
    vals[:color_count] = palette
    if len(coords) > color_count:
        vals[color_count:] = rng.choice(palette, size=len(coords) - color_count, replace=True)
    rng.shuffle(vals)

    motif = np.full((h, w), BG, dtype=int)
    for (r, c), color in zip(coords, vals):
        motif[int(r), int(c)] = int(color)
    return motif


def _sample_layout(rng: np.random.Generator):
    motif_h = int(rng.integers(2, 5))
    motif_w = int(rng.integers(2, 5))

    max_large_dr = (30 - motif_h) // (motif_h + 1)
    max_large_dc = (30 - motif_w) // (motif_w + 1)
    s_max = min(max_large_dr // (motif_h + 1), max_large_dc // (motif_w + 1))
    if s_max < 1:
        return None

    base = int(rng.integers(1, s_max + 1))
    large_dr_min = base * (motif_h + 1)
    large_dc_min = base * (motif_w + 1)
    large_dr = int(rng.integers(large_dr_min, max_large_dr + 1))
    large_dc = int(rng.integers(large_dc_min, max_large_dc + 1))

    markers = [(base, base), (base, large_dc), (large_dr, base)]
    if rng.random() < 0.6:
        markers.append((large_dr, large_dc))

    height_needed = motif_h
    width_needed = motif_w
    for dr, dc in markers:
        marker_r = motif_h + dr
        marker_c = motif_w + dc
        height_needed = max(height_needed, marker_r + motif_h * dr)
        width_needed = max(width_needed, marker_c + motif_w * dc)

    if height_needed > 30 or width_needed > 30:
        return None

    extra_rows = int(rng.integers(0, min(3, 31 - height_needed)))
    extra_cols = int(rng.integers(0, min(3, 31 - width_needed)))
    motif = _random_motif(rng, motif_h, motif_w)
    return motif, markers, height_needed + extra_rows, width_needed + extra_cols


def _wrong_outputs(inp: np.ndarray):
    return [
        lambda: _render(inp, lambda dr, dc: (dr, dr)),
        lambda: _render(inp, lambda dr, dc: (dc, dc)),
        lambda: _render(inp, lambda dr, dc: (dc, dr)),
        lambda: _render(inp, lambda dr, dc: (max(dr, dc), max(dr, dc))),
        lambda: _render(inp, lambda dr, dc: (min(dr, dc), min(dr, dc))),
    ]


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        g, motif, box, parsed = _parse_input(inp)
        out = main(inp)
    except Exception:
        return False

    _r0, _c0, r1, c1 = box
    pairs = [(dr, dc) for _r, _c, dr, dc in parsed]
    if not any(dr == dc for dr, dc in pairs):
        return False
    if not any(dr != dc for dr, dc in pairs):
        return False
    if not any(dr0 == dr1 and dc0 != dc1 for dr0, dc0 in pairs for dr1, dc1 in pairs if (dr0, dc0) != (dr1, dc1)):
        return False
    if not any(dc0 == dc1 and dr0 != dr1 for dr0, dc0 in pairs for dr1, dc1 in pairs if (dr0, dc0) != (dr1, dc1)):
        return False

    used = np.zeros(g.shape, dtype=bool)
    used[: r1 + 1, : c1 + 1] = True
    for r, c, dr, dc in parsed:
        stamp = _stretch(motif, dr, dc)
        h, w = stamp.shape
        if not np.array_equal(out[r : r + h, c : c + w], stamp):
            return False
        if used[r : r + h, c : c + w].any():
            return False
        used[r : r + h, c : c + w] = True

    for build_wrong in _wrong_outputs(inp):
        try:
            wrong = build_wrong()
        except Exception:
            continue
        if np.array_equal(wrong, out):
            return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    layout = _sample_layout(rng)
    if layout is None:
        return None

    motif, markers, height, width = layout
    grid = np.full((height, width), BG, dtype=int)
    grid[: motif.shape[0], : motif.shape[1]] = motif

    for dr, dc in markers:
        grid[motif.shape[0] + dr, motif.shape[1] + dc] = MARKER

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-axis stamp puzzle input")
