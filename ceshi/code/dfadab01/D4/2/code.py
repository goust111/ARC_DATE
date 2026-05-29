# Variant of dfadab01
# Dimension: D4 — 样本字形展示与消失 (symbol_instructions)
# Change: 输入中可能包含一个样本字形（完整4x4图案）及其右下角一格处的单像素标签（标签颜色=该字形对应的标记色），样本和标签在输出中消失 → 样本字形编码为调色板条

from __future__ import annotations

import random

import numpy as np


# Design analysis
# ---------------
# spec:
#   unchanged_core:
#     - Single-pixel markers still expand into fixed 4x4 glyph SHAPES.
#     - Glyphs are still anchored at the marker pixel and clipped by the grid edge.
#     - The scene is still a sparse marker lattice on a black background.
#   D4 change concretized:
#     - The old 4x4 sample glyph + below-right label is removed entirely.
#     - Instead, the input always contains one vertical N x 2 palette strip on the left or right edge.
#     - Each strip row is [marker_color, glyph_color], so the strip teaches color mapping only.
#     - The glyph shape must therefore remain fixed per marker color; only the output color is instance-specific.
#   behavior trace:
#     - Original: seeing marker 2 was enough to paint the yellow frame because color and shape were both hardcoded.
#     - Variant: seeing marker 2 still selects the frame SHAPE, but the output COLOR is read from the edge strip row [2, ?].
#     - Difference: the solver must now detect and erase the palette strip before expanding scene markers.
#   complexity calibration:
#     - Output colors are chosen as a permutation/subset of {1,4,6,7}, not fixed to the original per-marker colors.
#     - This makes the palette strip necessary while keeping the alphabet small and visually learnable.
#     - Each used marker type appears at least once in the scene, and 2-4 marker types are used, so the strip has repeated support.
#   hidden assumptions invalidated from the original main():
#     - Bundling marker->(output_color, mask) no longer works because output colors are not fixed.
#     - Looking for a 4x4 sample glyph near a marker no longer works because the legend is now a 1-pixel-thick strip.
#     - Any marker-colored pixel used to be a scene marker unless it matched the old legend-label pattern; now edge-strip cells must be masked out.
#   placement legality:
#     - The strip occupies exactly two edge columns and generator keeps the adjacent inner column empty as padding.
#     - Scene markers are placed on a lattice with row/col spacing 4-6, so expanded glyphs never overlap each other.
#     - When the strip is on the left, scene columns start at col 3; when on the right, the last scene column is at most w-4.
#   visual observability:
#     - The strip is contiguous and isolated, so it reads clearly as an instruction panel rather than scene content.
#     - Because at least one used marker gets a non-canonical output color every time, the variant never degenerates back to the old fixed-color rule.
#   cleanliness:
#     - Input uses only marker dots plus a compact edge strip; there is no stacked overlap between legend pixels and 4x4 glyph samples.
#   new solver assumptions and generator guarantees:
#     - Assumption: there is exactly one contiguous left/right edge strip whose first column contains unique marker colors.
#     - Guarantee: generator always creates exactly one such strip, keeps its rows contiguous, and leaves one empty inner padding column beside it.


MARKER_GLYPHS: dict[int, np.ndarray] = {
    2: np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    3: np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=bool,
    ),
    5: np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    ),
    8: np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ],
        dtype=bool,
    ),
}

MARKER_COLORS = tuple(MARKER_GLYPHS.keys())
GLYPH_COLORS = (1, 4, 6, 7)
CANONICAL_COLORS = {2: 4, 3: 1, 5: 6, 8: 7}

GRID_SIZES = (10, 20)
SPACING_RANGE = (4, 6)
MAX_RETRIES = 400


def _paint_clipped(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    h, w = out.shape
    for dr in range(4):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(4):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                out[rr, cc] = color


def _edge_runs(valid_rows: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    r0: int | None = None
    for r, ok in enumerate(valid_rows):
        if ok and r0 is None:
            r0 = r
        elif not ok and r0 is not None:
            runs.append((r0, r - 1))
            r0 = None
    if r0 is not None:
        runs.append((r0, len(valid_rows) - 1))
    return runs


def _extract_palette_mapping(g: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    h, w = g.shape
    assert w >= 3, "grid must be wide enough for a 2-column strip plus padding"

    best: tuple[int, str, int, int] | None = None
    for side, c0 in (("left", 0), ("right", w - 2)):
        c1 = c0 + 1
        pad_col = 2 if side == "left" else w - 3
        valid_rows = [
            int(g[r, c0]) in MARKER_GLYPHS and int(g[r, c1]) in GLYPH_COLORS for r in range(h)
        ]
        for r0, r1 in _edge_runs(valid_rows):
            keys = [int(g[r, c0]) for r in range(r0, r1 + 1)]
            vals = [int(g[r, c1]) for r in range(r0, r1 + 1)]
            if len(set(keys)) != len(keys) or len(set(vals)) != len(vals):
                continue
            if any(int(g[r, pad_col]) != 0 for r in range(r0, r1 + 1)):
                continue
            if r0 > 0 and np.any(g[r0 - 1, c0 : c1 + 1] != 0):
                continue
            if r1 + 1 < h and np.any(g[r1 + 1, c0 : c1 + 1] != 0):
                continue
            score = (r1 - r0 + 1, side, r0, r1)
            if best is None or score > best:
                best = score

    assert best is not None, "palette strip not found"
    _, side, r0, r1 = best
    c0 = 0 if side == "left" else w - 2

    pal_mask = np.zeros_like(g, dtype=bool)
    pal_mask[r0 : r1 + 1, c0 : c0 + 2] = True
    mapping = {int(g[r, c0]): int(g[r, c0 + 1]) for r in range(r0, r1 + 1)}
    return mapping, pal_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each scene marker into a fixed 4x4 glyph, using an edge palette strip to choose the glyph color.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8.
    - One left-edge or right-edge N×2 palette strip, where each row is [marker_color, glyph_color].

    Rule:
    - Read the edge strip as a marker→glyph-color mapping, then erase the strip itself.
    - Replace every remaining marker pixel by its fixed 4x4 glyph shape, anchored at that pixel and clipped by the grid boundary.
    - Marker color still selects the glyph shape; only the glyph color comes from the strip.

    Key insight:
    - The edge strip is a compressed legend: it no longer shows a 4x4 sample glyph, so the solver must combine strip-read colors with a fixed glyph dictionary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, *MARKER_COLORS, *GLYPH_COLORS})

    mapping, pal_mask = _extract_palette_mapping(g)
    assert mapping, "palette mapping must be non-empty"

    out = np.zeros_like(g)
    seen_marker = False
    for marker_color, mask in MARKER_GLYPHS.items():
        coords = np.argwhere((g == marker_color) & (~pal_mask))
        if coords.size:
            seen_marker = True
            assert marker_color in mapping, f"missing palette row for marker {marker_color}"
        for r, c in coords:
            _paint_clipped(out, int(r), int(c), int(mapping[marker_color]), mask)

    assert seen_marker
    return out


def _sample_positions(rng: random.Random, lo: int, hi: int, count: int) -> list[int]:
    assert lo <= hi and count >= 1
    if count == 1:
        return [rng.randint(lo, hi)]

    min_gap, max_gap = SPACING_RANGE
    for _ in range(200):
        gaps = [rng.randint(min_gap, max_gap) for _ in range(count - 1)]
        span = sum(gaps)
        if span <= hi - lo:
            start = rng.randint(lo, hi - span)
            vals = [start]
            for gap in gaps:
                vals.append(vals[-1] + gap)
            return vals

    span = min_gap * (count - 1)
    assert span <= hi - lo
    start = rng.randint(lo, hi - span)
    return [start + i * min_gap for i in range(count)]


def _feasible_count(span: int) -> int:
    return 1 + span // SPACING_RANGE[0]


def _choose_marker_mapping(rng: random.Random, used_markers: list[int]) -> dict[int, int]:
    for _ in range(100):
        colors = rng.sample(GLYPH_COLORS, len(used_markers))
        mapping = {marker: color for marker, color in zip(used_markers, colors, strict=False)}
        if any(mapping[m] != CANONICAL_COLORS[m] for m in used_markers):
            return mapping
    raise RuntimeError("failed to sample a non-canonical palette mapping")


def _render_scene(h: int, w: int, placed: list[tuple[int, int, int]], mapping: dict[int, int]) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for r, c, marker in placed:
        _paint_clipped(out, r, c, mapping[marker], MARKER_GLYPHS[marker])
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.choice(GRID_SIZES)
    w = h
    side = rng.choice(("left", "right"))

    scene_c_lo = 3 if side == "left" else 0
    scene_c_hi = w - 1 if side == "left" else w - 4
    row_span = h - 1
    col_span = scene_c_hi - scene_c_lo
    max_rows = min(5, _feasible_count(row_span))
    max_cols = min(5, _feasible_count(col_span))
    if max_rows < 2 or max_cols < 2:
        return None

    rows = rng.randint(2, max_rows)
    cols = rng.randint(2, max_cols)
    row_pos = _sample_positions(rng, 0, h - 1, rows)
    col_pos = _sample_positions(rng, scene_c_lo, scene_c_hi, cols)

    all_cells = [(r, c) for r in row_pos for c in col_pos]
    min_fill = max(2, (6 * len(all_cells) + 9) // 10)
    fill_count = rng.randint(min_fill, len(all_cells))
    used_count = rng.randint(2, min(4, fill_count, len(MARKER_COLORS)))
    used_markers = rng.sample(list(MARKER_COLORS), used_count)
    mapping = _choose_marker_mapping(rng, used_markers)

    chosen_cells = rng.sample(all_cells, fill_count)
    assigned_markers = used_markers.copy()
    assigned_markers.extend(rng.choice(used_markers) for _ in range(fill_count - used_count))
    rng.shuffle(assigned_markers)

    inp = np.zeros((h, w), dtype=int)
    placed: list[tuple[int, int, int]] = []
    for (r, c), marker in zip(chosen_cells, assigned_markers, strict=False):
        inp[r, c] = marker
        placed.append((r, c, marker))

    pal_rows = used_markers.copy()
    rng.shuffle(pal_rows)
    pal_r0 = rng.randint(0, h - len(pal_rows))
    pal_c0 = 0 if side == "left" else w - 2
    for i, marker in enumerate(pal_rows):
        inp[pal_r0 + i, pal_c0] = marker
        inp[pal_r0 + i, pal_c0 + 1] = mapping[marker]

    expected = _render_scene(h, w, placed, mapping)

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {MAX_RETRIES} attempts")
