# Combo variant of dfadab01 (index 3)
# combo_spec: D1/4+D9/0
# Name: 字形叠加+按色批处理

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
  combo: D1/4 + D9/0
  concrete_visual_form:
    input:
      - still uses single-cell markers in {2,3,5,8}
      - still may contain one legend glyph drawn in output color, with its label marker at (+4,+4)
      - real markers are arranged on a tidy lattice, but one axis now uses spacing 2 so the derived 4x4 glyphs overlap
    output:
      - glyph dictionary and clipping behavior are unchanged
      - markers are processed in global batches 2 -> 3 -> 5 -> 8
      - when a later glyph lands on an already painted cell, the cell stores max(existing_color, glyph_color) instead of blind overwrite
  behavior_trace:
    example:
      markers:
        - color 2 at (0,0) -> yellow frame (color 4)
        - color 3 at (0,2) -> blue ring (color 1)
      original task:
        - such dense placement was absent, so both glyphs never interacted
      D9-only last-wins:
        - batch 2 paints first
        - batch 3 paints later and would replace shared cells by color 1
      fused combo:
        - batch 2 still paints first and batch 3 still arrives later
        - on shared mask cells the merge rule is max(4,1)=4, so part of the earlier yellow frame remains visible
      difference:
        - the same overlap now exposes both the batch schedule and the compositing rule; later arrival does not guarantee visual dominance
  complexity_calibration:
    - only one axis is compressed to spacing 2; the orthogonal axis stays at 4 or 5 so rows/columns remain readable
    - generator always includes marker colors 2 and 3, because their mapped glyph colors 4 and 1 create visible max-vs-overwrite disagreements
    - optional extra colors 5 and/or 8 add additional collisions where the later batch does dominate, preventing the puzzle from collapsing into a pure 2/3 special case
    - repeated band motifs and reversed second row/column give multiple corroborating overlaps instead of a single accident
  implicit_assumption_audit:
    - original main() could iterate GLYPHS.items() because overlap never mattered; combo main() must use explicit PROCESS_ORDER
    - original _paint_clipped overwrote cells; combo logic must merge with max() on occupied cells
    - legend detection via _has_source_glyph still assumes each label marker has an exact 4x4 sample glyph at (-4,-4); generator preserves a clean 5x5 legend area
    - solver still assumes inputs only use marker colors plus legend output colors; generator samples only from that palette
  placement_legality:
    - input objects never overlap: only single-pixel markers are placed on the lattice, and legends are inserted only in empty 5x5 windows
    - overlap occurs only after expansion in the output space
    - all marker coordinates stay inside the board; clipping is allowed only for derived glyph footprints near edges
  observability_and_tidy_layout:
    - every accepted sample has real overlap, at least one 2-vs-3 overlap, and a nonzero pixel difference from the D9-only overwrite output
    - the lattice stays striped rather than chaotic, so humans can compare repeated collisions across rows/columns
    - overlap depth is limited by using a single dense axis; this keeps the output decipherable
  generator_metrics:
    - overlap_pixels: cells influenced by at least two real markers
    - pair23_overlap_pixels: cells touched by both marker colors 2 and 3
    - overwrite_diff_pixels: cells where combo output differs from the D9-only last-wins output
    - pair_count: number of distinct cross-color overlap pairs present in one instance
  new_solver_assumptions_and_generator_guards:
    - solver assumes at least one non-legend marker exists; generator always places a populated lattice
    - solver assumes sample legends never intersect real markers; _place_legend enforces that
    - solver assumes the only meaningful merge is max(existing_color, glyph_color); generator guarantees overlaps are frequent enough for that rule to be observable
"""


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    patch = g[r0 : r0 + 4, c0 : c0 + 4]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _paint_clipped_overwrite(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    h, w = out.shape
    for dr in range(4):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(4):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                out[rr, cc] = color


def _paint_clipped_max_merge(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    h, w = out.shape
    for dr in range(4):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(4):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                out[rr, cc] = max(int(out[rr, cc]), color)


def _ordered_output(g: np.ndarray, use_max_merge: bool) -> np.ndarray:
    out = np.zeros_like(g)
    seen_marker = False
    for marker_color in PROCESS_ORDER:
        glyph_color, mask = GLYPHS[marker_color]
        coords = np.argwhere(g == marker_color)
        if coords.size:
            seen_marker = True
        for r, c in coords:
            rr, cc = int(r), int(c)
            if _has_source_glyph(g, rr - 4, cc - 4, glyph_color, mask):
                continue
            if use_max_merge:
                _paint_clipped_max_merge(out, rr, cc, glyph_color, mask)
            else:
                _paint_clipped_overwrite(out, rr, cc, glyph_color, mask)
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


def _coverage_sets(g: np.ndarray) -> dict[tuple[int, int], set[int]]:
    h, w = g.shape
    coverage: dict[tuple[int, int], set[int]] = {}
    for marker_color, r0, c0 in _real_markers(g):
        _, mask = GLYPHS[marker_color]
        for dr in range(4):
            rr = r0 + dr
            if rr < 0 or rr >= h:
                continue
            for dc in range(4):
                cc = c0 + dc
                if 0 <= cc < w and bool(mask[dr, dc]):
                    coverage.setdefault((rr, cc), set()).add(marker_color)
    return coverage


def _variant_metrics(g: np.ndarray) -> dict[str, int]:
    coverage = _coverage_sets(g)
    overlap_pixels = 0
    pair23_overlap_pixels = 0
    pair_set: set[tuple[int, int]] = set()
    for colors in coverage.values():
        if len(colors) >= 2:
            overlap_pixels += 1
            ordered = sorted(colors)
            for idx, a in enumerate(ordered):
                for b in ordered[idx + 1 :]:
                    pair_set.add((a, b))
            if 2 in colors and 3 in colors:
                pair23_overlap_pixels += 1

    combo = _ordered_output(g, use_max_merge=True)
    overwrite = _ordered_output(g, use_max_merge=False)
    overwrite_diff_pixels = int(np.count_nonzero(combo != overwrite))
    early_survivor_pixels = int(np.count_nonzero((combo == 4) & (overwrite == 1)))
    return {
        "real_markers": len(_real_markers(g)),
        "overlap_pixels": overlap_pixels,
        "pair23_overlap_pixels": pair23_overlap_pixels,
        "pair_count": len(pair_set),
        "overwrite_diff_pixels": overwrite_diff_pixels,
        "early_survivor_pixels": early_survivor_pixels,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each marker into its fixed 4x4 glyph, but resolve dense overlaps by batched max-compositing.

    Input structure:
    - The board is still mostly black, with single-cell markers in colors 2, 3, 5, and/or 8.
    - Some inputs also contain one legend entry: a 4x4 sample glyph already drawn in its output color, plus a matching marker at the sample's lower-right corner.
    - Real markers may now be packed closely enough that their 4x4 output footprints overlap after expansion.

    Rule:
    - Use the same glyph dictionary as the base task: 2->yellow frame, 3->blue ring, 5->pink diagonal blocks, 8->orange X.
    - Process marker colors in the fixed batch order 2, then 3, then 5, then 8; for each real marker, place its glyph with the marker as the glyph's top-left corner and clip at the grid edge.
    - If a glyph paints onto an already nonzero output cell, keep the larger of the existing output color and the incoming glyph color instead of blindly overwriting; skip legend labels and erase the legend sample itself as before.

    Key insight:
    - The alphabet and legend format are unchanged, but dense placement turns the task into layered compositing: you must infer both the global color-batch schedule and that overlap pixels follow a max-color merge rather than simple last-wins painting.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, *GLYPHS.keys(), 1, 4, 6, 7})
    return _ordered_output(g, use_max_merge=True)


def _place_legend(g: np.ndarray, used_colors: list[int], rng: random.Random) -> bool:
    h, w = g.shape
    candidates = [
        (0, 0),
        (0, w - 5),
        (h - 5, 0),
        (h - 5, w - 5),
        (0, max(0, (w - 5) // 2)),
        (max(0, (h - 5) // 2), 0),
    ]
    rng.shuffle(candidates)

    for r0, c0 in candidates:
        if r0 < 0 or c0 < 0 or r0 + 4 >= h or c0 + 4 >= w:
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


def _build_palette(rng: random.Random) -> list[int]:
    extra_count = rng.choice([0, 1, 2])
    extras = rng.sample([5, 8], k=extra_count)
    return [2, 3, *extras]


def _build_marker_grid(rng: random.Random) -> np.ndarray:
    orientation = rng.choice(["horizontal", "vertical"])
    size = rng.choices([10, 20], weights=[1, 4], k=1)[0]
    h = w = size
    g = np.zeros((h, w), dtype=int)

    palette = _build_palette(rng)
    reverse_palette = list(reversed(palette))

    if orientation == "horizontal":
        row_spacing = rng.choice([4, 5])
        col_spacing = 2
        rows_max = 1 + (h - 4) // row_spacing
        cols_max = 1 + (w - 4) // col_spacing
        rows = min(rows_max, rng.choice([2, 3, 4]))
        cols = min(cols_max, rng.choice([5, 6, 7]))
        rows = max(rows, 2)
        cols = max(cols, 5 if size == 20 else 4)
        height = 4 + (rows - 1) * row_spacing
        width = 4 + (cols - 1) * col_spacing
        r_start = rng.randint(0, h - height)
        c_start = rng.randint(0, w - width)
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
        rows_max = 1 + (h - 4) // row_spacing
        cols_max = 1 + (w - 4) // col_spacing
        rows = min(rows_max, rng.choice([5, 6, 7]))
        cols = min(cols_max, rng.choice([2, 3, 4]))
        rows = max(rows, 5 if size == 20 else 4)
        cols = max(cols, 2)
        height = 4 + (rows - 1) * row_spacing
        width = 4 + (cols - 1) * col_spacing
        r_start = rng.randint(0, h - height)
        c_start = rng.randint(0, w - width)
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

    if rng.random() < 0.4:
        _place_legend(g, palette, rng)

    return g


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(400):
        g = _build_marker_grid(rng)
        out = main(g)
        metrics = _variant_metrics(g)
        if metrics["real_markers"] < 6:
            continue
        if metrics["overlap_pixels"] < 8:
            continue
        if metrics["pair23_overlap_pixels"] < 4:
            continue
        if metrics["overwrite_diff_pixels"] < 2:
            continue
        if metrics["early_survivor_pixels"] < 2:
            continue
        if metrics["pair_count"] < 1:
            continue
        assert out.shape == g.shape
        return g

    raise RuntimeError("failed to generate a visible D1/4+D9/0 instance")
