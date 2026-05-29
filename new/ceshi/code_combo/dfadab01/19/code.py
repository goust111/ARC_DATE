# Combo variant of dfadab01 (index 19)
# combo_spec: D4/2+D11/1
# Name: 调色板条+边缘隐藏样本

from __future__ import annotations

import random

import numpy as np


# Design analysis
# ---------------
# spec:
#   unchanged_core:
#     - Single-pixel scene markers still expand into 4x4 glyphs whose top-left corner is the marker pixel.
#     - Expansion is still clipped by the grid boundary.
#     - The visible scene is still a sparse lattice of marker dots on a black background.
#   fusion concretized:
#     - The old in-grid 4x4 sample glyph and its below-right label disappear completely.
#     - Each used marker type now has one hidden edge band: either a 2x20 strip stacked on the top edge,
#       or a 20x2 strip stacked on the left edge.
#     - Every fifth slot is a black separator, so the remaining 16 slots are read as 4 groups of 4 and
#       therefore as a row-major flattening of a 4x4 glyph.
#     - The outer lane of a band is constant marker color on data slots; the inner lane uses marker color
#       for 0-bits and glyph color for 1-bits. Decoding one band therefore yields both the marker key and
#       the glyph's output color/pattern.
#   behavior trace:
#     - Original: a marker color directly implied one fixed mask and one fixed output color, and an optional
#       interior sample merely reminded the solver of that fact.
#     - Fusion: a marker color is only a key. Example: if a top-edge band has outer row = all 2 on data slots
#       and inner row grouped as [2 4 4 2 | 4 2 2 4 | 4 2 2 4 | 2 4 4 2], marker 2 now means "paint the
#       blue-ring mask in color 4", even though that is not its canonical assignment.
#     - Difference: the solver must first decode compressed edge legends, then use the recovered per-instance
#       shape/color dictionary on the interior markers.
#   complexity calibration:
#     - Each instance uses 2-3 marker types, and every used type appears at least once in the scene.
#     - Shapes are sampled injectively from the original 4-glyph alphabet and colors are sampled injectively
#       from {1,4,6,7}; this keeps the legend small and human-trackable.
#     - At least one marker gets a non-canonical shape or non-canonical color, so the variant cannot collapse
#       back to the original fixed dictionary.
#   hidden assumptions invalidated from the original main():
#     - Hardcoding marker->(glyph_color, mask) is no longer valid.
#     - Searching for a visible 4x4 sample near a marker is no longer valid because the sample is edge-encoded.
#     - Marker-colored pixels on the top/left edge are no longer necessarily scene markers; whole legend bands
#       must be detected and masked out before expansion.
#   placement legality:
#     - A hidden sample needs 16 data slots plus 4 separators, so the generator fixes the grid to 20x20.
#     - Top legends occupy rows [0, 2k) and reserve row 2k as blank padding; left legends occupy cols
#       [0, 2k) and reserve col 2k as blank padding.
#     - Scene markers are placed on a 4-6 spaced lattice inside the remaining rectangle, so expanded glyphs
#       never overlap each other or the hidden-sample area.
#   visual observability:
#     - The 4+1+4+1+4+1+4 grouping makes the flattened 4x4 structure visible to a human instead of hiding it
#       in arithmetic or ordering tricks.
#     - The constant outer lane announces the controlling marker color, while the inner lane makes the output
#       color and filled cells visible at the same time.
#     - Repeated use of every decoded marker in the scene gives direct support for the inferred legend.
#   cleanliness:
#     - Legends live on one dedicated edge with one empty separator row/column; the interior remains a clean
#       sparse marker field.
#   generator guarantees for the new solver assumptions:
#     - Exactly one orientation is used per instance: stacked top bands or stacked left bands.
#     - Bands use unique marker keys, unique glyph colors, and unique glyph masks.
#     - The decoded mask always belongs to the original 4-glyph alphabet, and every legend band has at least
#       one matching scene marker.


CANONICAL_GLYPHS: dict[int, np.ndarray] = {
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

MARKER_COLORS = tuple(CANONICAL_GLYPHS.keys())
GLYPH_COLORS = (1, 4, 6, 7)
CANONICAL_COLORS = {2: 4, 3: 1, 5: 6, 8: 7}

GRID_SIZE = 20
DATA_IDXS = tuple(i for i in range(GRID_SIZE) if i % 5 != 4)
SEP_IDXS = tuple(i for i in range(GRID_SIZE) if i % 5 == 4)
SPACING_RANGE = (4, 6)
MAX_RETRIES = 400


def _shape_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in mask.astype(int).reshape(-1).tolist())


VALID_SHAPE_KEYS = {_shape_key(mask) for mask in CANONICAL_GLYPHS.values()}


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


def _decode_band(outer: np.ndarray, inner: np.ndarray) -> tuple[int, int, np.ndarray] | None:
    if outer.shape != (GRID_SIZE,) or inner.shape != (GRID_SIZE,):
        return None
    if any(int(outer[i]) != 0 or int(inner[i]) != 0 for i in SEP_IDXS):
        return None

    outer_vals = [int(outer[i]) for i in DATA_IDXS]
    marker_set = set(outer_vals)
    if len(marker_set) != 1:
        return None
    marker = outer_vals[0]
    if marker not in MARKER_COLORS:
        return None

    inner_vals = [int(inner[i]) for i in DATA_IDXS]
    if any(v == 0 for v in inner_vals):
        return None
    glyph_candidates = set(inner_vals) - {marker}
    if len(glyph_candidates) != 1:
        return None
    glyph_color = next(iter(glyph_candidates))
    if glyph_color not in GLYPH_COLORS:
        return None
    if any(v not in {marker, glyph_color} for v in inner_vals):
        return None

    mask = np.array([v == glyph_color for v in inner_vals], dtype=bool).reshape(4, 4)
    if _shape_key(mask) not in VALID_SHAPE_KEYS:
        return None
    return marker, glyph_color, mask


def _try_decode_top(g: np.ndarray) -> tuple[dict[int, tuple[int, np.ndarray]], np.ndarray] | None:
    h, w = g.shape
    if w != GRID_SIZE:
        return None

    legend_mask = np.zeros_like(g, dtype=bool)
    mapping: dict[int, tuple[int, np.ndarray]] = {}
    seen_shapes: set[tuple[int, ...]] = set()
    seen_colors: set[int] = set()
    r = 0
    while r + 1 < h:
        decoded = _decode_band(g[r, :], g[r + 1, :])
        if decoded is None:
            break
        marker, glyph_color, mask = decoded
        shape_key = _shape_key(mask)
        if marker in mapping or glyph_color in seen_colors or shape_key in seen_shapes:
            return None
        mapping[marker] = (glyph_color, mask)
        seen_colors.add(glyph_color)
        seen_shapes.add(shape_key)
        legend_mask[r : r + 2, :] = True
        r += 2

    if not mapping or r >= h or np.any(g[r, :] != 0):
        return None
    legend_mask[r, :] = True
    return mapping, legend_mask


def _try_decode_left(g: np.ndarray) -> tuple[dict[int, tuple[int, np.ndarray]], np.ndarray] | None:
    h, w = g.shape
    if h != GRID_SIZE:
        return None

    legend_mask = np.zeros_like(g, dtype=bool)
    mapping: dict[int, tuple[int, np.ndarray]] = {}
    seen_shapes: set[tuple[int, ...]] = set()
    seen_colors: set[int] = set()
    c = 0
    while c + 1 < w:
        decoded = _decode_band(g[:, c], g[:, c + 1])
        if decoded is None:
            break
        marker, glyph_color, mask = decoded
        shape_key = _shape_key(mask)
        if marker in mapping or glyph_color in seen_colors or shape_key in seen_shapes:
            return None
        mapping[marker] = (glyph_color, mask)
        seen_colors.add(glyph_color)
        seen_shapes.add(shape_key)
        legend_mask[:, c : c + 2] = True
        c += 2

    if not mapping or c >= w or np.any(g[:, c] != 0):
        return None
    legend_mask[:, c] = True
    return mapping, legend_mask


def _extract_hidden_legend(
    g: np.ndarray,
) -> tuple[dict[int, tuple[int, np.ndarray]], np.ndarray]:
    candidates = []
    top = _try_decode_top(g)
    if top is not None:
        candidates.append(top)
    left = _try_decode_left(g)
    if left is not None:
        candidates.append(left)
    assert len(candidates) == 1, "hidden edge legend not found uniquely"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each scene marker using the glyph pattern and glyph color decoded from hidden edge bands.

    Input structure:
    - A 20x20 black-background grid contains scene markers in colors 2, 3, 5, and/or 8.
    - On exactly one edge, stacked 2x20 top bands or 20x2 left bands hide the sample legends.
    - In each band, the outer lane is the marker color and the inner lane is a separator-grouped 16-cell flattening of the 4x4 glyph, using marker color for empty cells and glyph color for filled cells.

    Rule:
    - Detect the hidden edge bands, decode each one into a marker -> (glyph color, 4x4 mask) entry, and ignore the legend area itself.
    - Replace every remaining marker dot by its decoded 4x4 glyph, anchored at the marker and clipped at the grid border.
    - The output contains only the painted scene glyphs; the hidden legend disappears.

    Key insight:
    - The edge strip is both a palette and a compressed sample: the constant outer lane tells which marker is being defined, while the grouped inner lane reveals the glyph shape and its paint color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE)
    assert set(np.unique(g)).issubset({0, *MARKER_COLORS, *GLYPH_COLORS})

    mapping, legend_mask = _extract_hidden_legend(g)
    scene = g.copy()
    scene[legend_mask] = 0

    present_markers = {int(v) for v in np.unique(scene) if int(v) in MARKER_COLORS}
    assert present_markers
    assert present_markers == set(mapping.keys())

    out = np.zeros_like(g)
    for marker in sorted(present_markers):
        glyph_color, mask = mapping[marker]
        for r, c in np.argwhere(scene == marker):
            _paint_clipped(out, int(r), int(c), int(glyph_color), mask)

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


def _choose_mapping(
    rng: random.Random, used_markers: list[int]
) -> dict[int, tuple[int, np.ndarray]]:
    shape_sources_all = list(MARKER_COLORS)
    for _ in range(200):
        shape_sources = rng.sample(shape_sources_all, len(used_markers))
        glyph_colors = rng.sample(list(GLYPH_COLORS), len(used_markers))
        mapping = {
            marker: (glyph_color, CANONICAL_GLYPHS[src])
            for marker, glyph_color, src in zip(
                used_markers, glyph_colors, shape_sources, strict=False
            )
        }
        if any(
            glyph_color != CANONICAL_COLORS[marker]
            or _shape_key(mask) != _shape_key(CANONICAL_GLYPHS[marker])
            for marker, (glyph_color, mask) in mapping.items()
        ):
            return mapping
    raise RuntimeError("failed to sample a non-canonical hidden legend")


def _build_top_band(marker: int, glyph_color: int, mask: np.ndarray) -> np.ndarray:
    band = np.zeros((2, GRID_SIZE), dtype=int)
    flat = mask.reshape(-1)
    for idx, bit in zip(DATA_IDXS, flat.tolist(), strict=False):
        band[0, idx] = marker
        band[1, idx] = glyph_color if bool(bit) else marker
    return band


def _build_left_band(marker: int, glyph_color: int, mask: np.ndarray) -> np.ndarray:
    band = np.zeros((GRID_SIZE, 2), dtype=int)
    flat = mask.reshape(-1)
    for idx, bit in zip(DATA_IDXS, flat.tolist(), strict=False):
        band[idx, 0] = marker
        band[idx, 1] = glyph_color if bool(bit) else marker
    return band


def _render_scene(
    placed: list[tuple[int, int, int]], mapping: dict[int, tuple[int, np.ndarray]]
) -> np.ndarray:
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for r, c, marker in placed:
        glyph_color, mask = mapping[marker]
        _paint_clipped(out, r, c, glyph_color, mask)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    side = rng.choice(("top", "left"))
    used_count = rng.randint(2, 3)
    used_markers = rng.sample(list(MARKER_COLORS), used_count)
    mapping = _choose_mapping(rng, used_markers)

    scene_r_lo = 2 * used_count + 1 if side == "top" else 0
    scene_c_lo = 2 * used_count + 1 if side == "left" else 0
    scene_r_hi = GRID_SIZE - 1
    scene_c_hi = GRID_SIZE - 1

    row_span = scene_r_hi - scene_r_lo
    col_span = scene_c_hi - scene_c_lo
    max_rows = min(4, _feasible_count(row_span))
    max_cols = min(4, _feasible_count(col_span))
    if max_rows < 2 or max_cols < 2:
        return None

    rows = rng.randint(2, max_rows)
    cols = rng.randint(2, max_cols)
    row_pos = _sample_positions(rng, scene_r_lo, scene_r_hi, rows)
    col_pos = _sample_positions(rng, scene_c_lo, scene_c_hi, cols)

    all_cells = [(r, c) for r in row_pos for c in col_pos]
    min_fill = max(used_count, (6 * len(all_cells) + 9) // 10)
    fill_count = rng.randint(min_fill, len(all_cells))

    chosen_cells = rng.sample(all_cells, fill_count)
    assigned_markers = used_markers.copy()
    assigned_markers.extend(rng.choice(used_markers) for _ in range(fill_count - used_count))
    rng.shuffle(assigned_markers)

    inp = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    placed: list[tuple[int, int, int]] = []
    for (r, c), marker in zip(chosen_cells, assigned_markers, strict=False):
        inp[r, c] = marker
        placed.append((r, c, marker))

    legend_order = used_markers.copy()
    rng.shuffle(legend_order)
    if side == "top":
        for i, marker in enumerate(legend_order):
            glyph_color, mask = mapping[marker]
            inp[2 * i : 2 * i + 2, :] = _build_top_band(marker, glyph_color, mask)
    else:
        for i, marker in enumerate(legend_order):
            glyph_color, mask = mapping[marker]
            inp[:, 2 * i : 2 * i + 2] = _build_left_band(marker, glyph_color, mask)

    expected = _render_scene(placed, mapping)
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
