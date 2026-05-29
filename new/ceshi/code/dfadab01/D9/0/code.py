# Variant of dfadab01
# Dimension: D9 — 字形放置的多步处理顺序 (multi_step)
# Change: 单步处理：所有标记同时被替换为字形，无先后顺序依赖... → 按颜色分批次处理

from __future__ import annotations

import random

import numpy as np


GLYPHS: dict[int, tuple[int, np.ndarray]] = {
    2: (
        4,
        np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    3: (
        1,
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
            ],
            dtype=bool,
        ),
    ),
    5: (
        6,
        np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    8: (
        7,
        np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ],
            dtype=bool,
        ),
    ),
}

PROCESS_ORDER: tuple[int, ...] = (2, 3, 5, 8)


DESIGN_ANALYSIS = """
variant_spec:
  core_change:
    original: marker spacing prevented glyph overlap, so paint order was implementation detail only.
    variant: real markers are packed with spacing 2 on one axis, so 4x4 glyph footprints overlap and
      the explicit batch order 2 -> 3 -> 5 -> 8 determines which color survives on shared pixels.
  concrete_visual_form:
    input:
      - same single-pixel markers in colors 2/3/5/8
      - optional legend keeps the old format: one 4x4 sample glyph plus its marker at (+4,+4)
      - dense rows or columns of markers create intended future overlap in the output
    output:
      - same glyph dictionary and same clipping rule
      - overlapping regions show only the later batch's glyph color
  behavior_trace:
    example:
      markers: color-8 at (0,0), color-2 at (0,2)
      original generator: this never happened because spacing >= 4 removed interaction
      variant:
        1. paint color-2 glyph at (0,2)
        2. paint color-8 glyph at (0,0) later in the global order
        3. cells where both masks are true now end up orange (7), not yellow (4)
      difference: the same geometric overlap is no longer ambiguous or ignorable; order is now visible.
  complexity_calibration:
    - only one axis overlaps; the orthogonal axis keeps spacing >= 4
    - this keeps the output readable while still creating multiple cross-color collisions
    - each example uses 2-4 marker colors with repeated rows/columns so the priority is inferable
    - first two lines are opposite marker sequences, preventing a false left-to-right interpretation
  implicit_assumption_audit:
    - iterating GLYPHS.items() was harmless before because outputs never overlapped; now order matters,
      so main() must use an explicit PROCESS_ORDER constant instead of relying on dict insertion order.
    - _paint_clipped always overwrites; before this was benign, now it is the intended last-wins rule.
    - legend skipping via _has_source_glyph still works, but only if generator keeps each legend's 5x5
      bbox clear of real markers and other legends; the generator enforces that.
  placement_legality:
    - marker top-left positions are fully inside the grid so no input object overlaps another object
    - overlap happens only in the derived output footprints
    - legends are placed only when a clean 5x5 area exists
  observability_and_tidiness:
    - cross-color overlap is mandatory
    - order sensitivity is checked by comparing forward-order and reverse-order outputs
    - horizontal or vertical banding keeps the scene structured instead of chaotic
  new_solver_assumptions_and_generator_guards:
    - solver assumes marker colors belong to PROCESS_ORDER; generator samples only from that set
    - solver assumes at least one real marker exists; generator always places a full marker lattice
    - solver assumes any legend label has its exact source glyph at (-4,-4); generator draws legends in that format
"""


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    patch = g[r0 : r0 + 4, c0 : c0 + 4]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


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


def _ordered_output(g: np.ndarray, order: tuple[int, ...]) -> np.ndarray:
    out = np.zeros_like(g)
    seen_marker = False
    for marker_color in order:
        glyph_color, mask = GLYPHS[marker_color]
        coords = np.argwhere(g == marker_color)
        if coords.size:
            seen_marker = True
        for r, c in coords:
            if _has_source_glyph(g, int(r) - 4, int(c) - 4, glyph_color, mask):
                continue
            _paint_clipped(out, int(r), int(c), glyph_color, mask)
    assert seen_marker
    return out


def _repeat_to_length(seq: list[int], length: int) -> list[int]:
    out: list[int] = []
    while len(out) < length:
        out.extend(seq)
    return out[:length]


def _real_markers(g: np.ndarray) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    for marker_color in PROCESS_ORDER:
        glyph_color, mask = GLYPHS[marker_color]
        for r, c in np.argwhere(g == marker_color):
            rr, cc = int(r), int(c)
            if _has_source_glyph(g, rr - 4, cc - 4, glyph_color, mask):
                continue
            markers.append((marker_color, rr, cc))
    return markers


def _variant_metrics(g: np.ndarray) -> dict[str, int]:
    H, W = g.shape
    coverage: dict[tuple[int, int], set[int]] = {}
    for marker_color, r0, c0 in _real_markers(g):
        _, mask = GLYPHS[marker_color]
        for dr in range(4):
            rr = r0 + dr
            if rr < 0 or rr >= H:
                continue
            for dc in range(4):
                cc = c0 + dc
                if 0 <= cc < W and bool(mask[dr, dc]):
                    coverage.setdefault((rr, cc), set()).add(marker_color)

    overlap_pixels = sum(1 for colors in coverage.values() if len(colors) >= 2)
    forward = _ordered_output(g, PROCESS_ORDER)
    reverse = _ordered_output(g, tuple(reversed(PROCESS_ORDER)))
    order_diff_pixels = int(np.count_nonzero(forward != reverse))
    return {
        "overlap_pixels": int(overlap_pixels),
        "order_diff_pixels": order_diff_pixels,
        "real_markers": len(_real_markers(g)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored control pixel into its fixed 4x4 glyph, resolving overlaps by color batches.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8.
    - Some inputs also include one sample 4x4 glyph in its output color, plus a matching marker at the sample's lower-right corner.
    - Unlike the original non-overlapping layouts, marker spacing may be tight enough that expanded glyphs collide.

    Rule:
    - Use the same glyph dictionary as before: 2->yellow frame, 3->blue ring, 5->pink diagonal blocks, 8->orange X.
    - Paint all color-2 markers first, then all color-3 markers, then color-5, then color-8; later batches overwrite earlier painted cells wherever glyph masks overlap.
    - Clip each 4x4 glyph to the grid bounds, and remove the sample glyph/label pair by skipping label markers.

    Key insight:
    - The alphabet itself is unchanged; the new information is a global marker-color priority, visible only where dense marker placements make glyph footprints overlap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, *GLYPHS.keys(), 1, 4, 6, 7})
    return _ordered_output(g, PROCESS_ORDER)


def _place_legend(g: np.ndarray, used_colors: list[int], rng: random.Random) -> bool:
    H, W = g.shape
    candidates = [
        (0, 0),
        (0, W - 5),
        (H - 5, 0),
        (H - 5, W - 5),
        (0, max(0, (W - 5) // 2)),
        (max(0, (H - 5) // 2), 0),
    ]
    rng.shuffle(candidates)

    for r0, c0 in candidates:
        if r0 < 0 or c0 < 0 or r0 + 4 >= H or c0 + 4 >= W:
            continue
        if np.any(g[r0 : r0 + 5, c0 : c0 + 5] != 0):
            continue
        marker_color = rng.choice(used_colors)
        glyph_color, mask = GLYPHS[marker_color]
        for dr in range(4):
            for dc in range(4):
                if bool(mask[dr, dc]):
                    g[r0 + dr, c0 + dc] = glyph_color
        g[r0 + 4, c0 + 4] = marker_color
        return True
    return False


def _build_marker_grid(rng: random.Random) -> np.ndarray:
    orientation = rng.choice(["horizontal", "vertical"])
    size = rng.choice([10, 20])
    H = W = size
    g = np.zeros((H, W), dtype=int)

    used_count = rng.choices([2, 3, 4], weights=[1, 3, 3], k=1)[0]
    used_colors = rng.sample(list(PROCESS_ORDER), used_count)
    palette = rng.sample(used_colors, len(used_colors))
    reverse_palette = list(reversed(palette))

    if orientation == "horizontal":
        row_spacing = rng.choice([4, 5])
        col_spacing = 2
        rows_max = 1 + (H - 4) // row_spacing
        cols_max = 1 + (W - 4) // col_spacing
        rows = min(rows_max, rng.choice([2, 3, 4]))
        cols = min(cols_max, rng.choice([4, 5, 6]))
        rows = max(rows, 2)
        cols = max(cols, 4)
        height = 4 + (rows - 1) * row_spacing
        width = 4 + (cols - 1) * col_spacing
        r_start = rng.randint(0, H - height)
        c_start = rng.randint(0, W - width)
        row_positions = [r_start + i * row_spacing for i in range(rows)]
        col_positions = [c_start + i * col_spacing for i in range(cols)]

        for ridx, r in enumerate(row_positions):
            if ridx == 0:
                base = palette
            elif ridx == 1:
                base = reverse_palette
            else:
                shift = ridx % len(palette)
                base = palette[shift:] + palette[:shift]
            seq = _repeat_to_length(base, cols)
            for cidx, c in enumerate(col_positions):
                g[r, c] = seq[cidx]
    else:
        row_spacing = 2
        col_spacing = rng.choice([4, 5])
        rows_max = 1 + (H - 4) // row_spacing
        cols_max = 1 + (W - 4) // col_spacing
        rows = min(rows_max, rng.choice([4, 5, 6]))
        cols = min(cols_max, rng.choice([2, 3, 4]))
        rows = max(rows, 4)
        cols = max(cols, 2)
        height = 4 + (rows - 1) * row_spacing
        width = 4 + (cols - 1) * col_spacing
        r_start = rng.randint(0, H - height)
        c_start = rng.randint(0, W - width)
        row_positions = [r_start + i * row_spacing for i in range(rows)]
        col_positions = [c_start + i * col_spacing for i in range(cols)]

        for cidx, c in enumerate(col_positions):
            if cidx == 0:
                base = palette
            elif cidx == 1:
                base = reverse_palette
            else:
                shift = cidx % len(palette)
                base = palette[shift:] + palette[:shift]
            seq = _repeat_to_length(base, rows)
            for ridx, r in enumerate(row_positions):
                g[r, c] = seq[ridx]

    if rng.random() < 0.5:
        _place_legend(g, used_colors, rng)

    return g


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(200):
        g = _build_marker_grid(rng)
        out = main(g)
        metrics = _variant_metrics(g)
        if metrics["real_markers"] < 4:
            continue
        if metrics["overlap_pixels"] < 4:
            continue
        if metrics["order_diff_pixels"] < 2:
            continue
        assert out.shape == g.shape
        return g

    raise RuntimeError("failed to generate a D9-visible instance")
