# Combo variant of dfadab01 (index 17)
# combo_spec: D2/2+D9/0
# Name: 颜色循环映射+按色批处理

from __future__ import annotations

import random

import numpy as np


DESIGN_ANALYSIS = """
combo_design:
  concrete_behavior_and_trace:
    - Keep the original 4x4 glyph alphabet as a fixed palette:
      yellow frame, blue ring, magenta diagonal blocks, orange X.
    - Real markers are first read in row-major scan order, left-to-right then top-to-bottom.
      Scan index 0 gets palette item 0, index 1 gets item 1, index 2 gets item 2, index 3 gets
      item 3, then the cycle repeats. This assignment ignores marker color.
    - Painting does not follow that scan order. After every real marker has already received its
      cycle phase, glyphs are stamped in marker-color batches 2 -> 3 -> 5 -> 8, so later colors
      overwrite earlier ones on overlaps.
    - Example: markers at (0,0)=8 and (0,2)=2 overlap horizontally.
      Original task:
        1. color 8 always means orange X
        2. color 2 always means yellow frame
        3. with batch order, the color-8 X paints after the color-2 frame
      Fused variant:
        1. scan order sees (0,0) first, so marker 8 receives the yellow-frame glyph
        2. scan order sees (0,2) second, so marker 2 receives the blue-ring glyph
        3. painting still happens by marker color, so the color-2 blue ring paints before the
           color-8 yellow frame, and the overlapping cells finish as the later frame color
      Difference:
        both "which glyph a marker gets" and "which overlap survives" changed, and they depend on
        two different visible orderings.
  complexity_calibration:
    - Only one axis is dense: columns are 2 cells apart while rows stay 4 or 5 cells apart.
      This guarantees overlap without turning the whole board into an unreadable pileup.
    - Each sample uses 15-24 real markers arranged in aligned rows. Repetition is strong enough for
      people to spot both the 4-step palette cycle and the fixed color-batch overwrite rule.
    - The generator favors 3 or 4 marker colors so several batch levels are visible. It also makes
      the row width 5 or 6, not a multiple of the 4-step cycle, so a single marker color appears
      at multiple cycle phases instead of collapsing into one fixed remapping.
  implicit_assumption_audit:
    - Original assumption: marker color directly selects one glyph entry from GLYPHS.
      Broken here: glyph choice now comes from row-major scan position.
    - Original assumption: iterating colors and painting immediately is equivalent to any other
      order because glyphs never overlap.
      Broken here: overlaps are intentional, so main() must separate "assign glyph by scan order"
      from "paint by marker-color batch order".
    - Original assumption: legend labels can be recognized using the same marker->glyph mapping.
      Broken here: a marker color no longer implies one glyph, so label detection must match
      against any source glyph drawn at (-4,-4), independent of marker color.
    - New conflict checked: if same-color markers overlapped each other, within-batch iteration
      order would matter even though the spec only defines inter-color batch order.
      The generator avoids that by repeating colors only at spacing >= 4 along the dense axis.
  placement_legality:
    - Input objects are still single pixels; only derived output glyphs overlap.
    - Marker top-left coordinates always stay inside the grid, so input markers never collide or go
      out of bounds. Clipping, when present, happens only to the 4x4 expanded glyphs near the edge.
    - The generator never places legends, but main() still supports skipping an old-style legend
      label if one is present.
  observability_and_tidiness:
    - The marker lattice is orderly, not random noise, so scan order is visually legible.
    - Overlap happens in horizontal bands only. Different rows stay separated, which keeps the
      output readable even though later batches overwrite earlier paint.
    - The generator rejects cases where the fused output is too close either to the fixed-mapping
      baseline or to scan-order painting, ensuring that both fused dimensions are actually visible.
  inferability_of_changes:
    - The cycle uses only the four original glyph types/colors, so there is no extra color burden.
    - Human-solvable clue for D2/2: the same marker color appears in multiple scan phases and thus
      receives different glyphs within one board; the generator measures that directly.
    - Human-solvable clue for D9/0: overlap outcomes disagree with what a pure scan-order painter
      would produce; the generator keeps only examples with a visible batch-order effect.
  new_main_assumptions_and_generator_guards:
    - main() assumes every real marker color is one of {2,3,5,8}; generator samples only those.
    - main() assumes row-major order over real markers is well defined; generator uses one marker
      per lattice site, so there are no ties.
    - main() assumes any legend label, if present, has a complete 4x4 palette glyph exactly at
      (-4,-4); that is the only pattern _is_legend_label() recognizes.
    - main() assumes same-color overlaps are irrelevant; generator guarantees same-color markers are
      separated by at least 4 cells along the dense axis and by at least 4 rows vertically.
"""


PROCESS_ORDER: tuple[int, ...] = (2, 3, 5, 8)

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

PALETTE: tuple[tuple[int, np.ndarray], ...] = tuple(GLYPHS[color] for color in PROCESS_ORDER)
ALLOWED_COLORS = {0, *PROCESS_ORDER, *(glyph_color for glyph_color, _mask in PALETTE)}


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


def _is_legend_label(g: np.ndarray, r: int, c: int) -> bool:
    for glyph_color, mask in PALETTE:
        if _has_source_glyph(g, r - 4, c - 4, glyph_color, mask):
            return True
    return False


def _collect_real_markers(g: np.ndarray) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    for marker_color in PROCESS_ORDER:
        for r, c in np.argwhere(g == marker_color):
            rr = int(r)
            cc = int(c)
            if _is_legend_label(g, rr, cc):
                continue
            markers.append((marker_color, rr, cc))
    markers.sort(key=lambda item: (item[1], item[2], item[0]))
    return markers


def _assign_cycle(markers: list[tuple[int, int, int]]) -> list[tuple[int, int, int, int]]:
    assigned: list[tuple[int, int, int, int]] = []
    for idx, (marker_color, r, c) in enumerate(markers):
        assigned.append((marker_color, r, c, idx % len(PALETTE)))
    return assigned


def _render_batched(
    shape: tuple[int, int],
    assigned: list[tuple[int, int, int, int]],
    order: tuple[int, ...],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for marker_color in order:
        for color, r, c, phase in assigned:
            if color != marker_color:
                continue
            glyph_color, mask = PALETTE[phase]
            _paint_clipped(out, r, c, glyph_color, mask)
    return out


def _render_scan_order(shape: tuple[int, int], assigned: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for _marker_color, r, c, phase in assigned:
        glyph_color, mask = PALETTE[phase]
        _paint_clipped(out, r, c, glyph_color, mask)
    return out


def _render_fixed_mapping(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for marker_color in PROCESS_ORDER:
        glyph_color, mask = GLYPHS[marker_color]
        for color, r, c in markers:
            if color == marker_color:
                _paint_clipped(out, r, c, glyph_color, mask)
    return out


def _overlap_pixels(shape: tuple[int, int], assigned: list[tuple[int, int, int, int]]) -> int:
    h, w = shape
    cover = np.zeros(shape, dtype=int)
    for _marker_color, r0, c0, phase in assigned:
        _glyph_color, mask = PALETTE[phase]
        for dr in range(4):
            rr = r0 + dr
            if rr < 0 or rr >= h:
                continue
            for dc in range(4):
                cc = c0 + dc
                if 0 <= cc < w and bool(mask[dr, dc]):
                    cover[rr, cc] += 1
    return int(np.count_nonzero(cover >= 2))


def _variant_metrics(g: np.ndarray) -> dict[str, int]:
    markers = _collect_real_markers(g)
    assigned = _assign_cycle(markers)
    fused = _render_batched(g.shape, assigned, PROCESS_ORDER)
    scan = _render_scan_order(g.shape, assigned)
    fixed = _render_fixed_mapping(g.shape, markers)

    phase_sets: dict[int, set[int]] = {color: set() for color in PROCESS_ORDER}
    for marker_color, _r, _c, phase in assigned:
        phase_sets[marker_color].add(int(phase))

    used_colors = sum(bool(phase_sets[color]) for color in PROCESS_ORDER)
    multi_phase_colors = sum(len(phase_sets[color]) >= 2 for color in PROCESS_ORDER)
    return {
        "real_markers": len(markers),
        "used_colors": int(used_colors),
        "multi_phase_colors": int(multi_phase_colors),
        "overlap_pixels": _overlap_pixels(g.shape, assigned),
        "batch_vs_scan_pixels": int(np.count_nonzero(fused != scan)),
        "cycle_vs_fixed_pixels": int(np.count_nonzero(fused != fixed)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand real marker pixels into a 4-step glyph cycle, then resolve overlaps by marker-color batches.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8.
    - Markers usually form aligned rows with tight horizontal spacing, so their future 4x4 glyphs can overlap.
    - Some inputs may still contain an old-style sample glyph plus a label marker at its lower-right; those are removable legend artifacts, not real markers.

    Rule:
    - Ignore any legend label whose upper-left 4x4 area already shows one of the palette glyphs.
    - Read the remaining real markers in row-major scan order and assign glyphs cyclically in the fixed palette order: yellow frame, blue ring, magenta diagonal blocks, orange X, then repeat.
    - After those assignments are fixed, paint all color-2 markers first, then color-3, then color-5, then color-8; later batches overwrite earlier painted cells, and every glyph is clipped to the grid.

    Key insight:
    - Marker color no longer tells you which glyph to draw; position in the global scan order does that, while marker color only decides when that preassigned glyph gets to overwrite others.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))).issubset(ALLOWED_COLORS)

    markers = _collect_real_markers(g)
    assert markers, "expected at least one real marker"
    assigned = _assign_cycle(markers)
    return _render_batched(g.shape, assigned, PROCESS_ORDER)


def _repeat_to_length(seq: list[int], length: int) -> list[int]:
    out: list[int] = []
    while len(out) < length:
        out.extend(seq)
    return out[:length]


def _build_marker_grid(rng: random.Random) -> np.ndarray:
    h = w = 20
    g = np.zeros((h, w), dtype=int)

    used_count = rng.choices([3, 4, 2], weights=[4, 3, 1], k=1)[0]
    used_colors = rng.sample(list(PROCESS_ORDER), used_count)
    base_seq = rng.sample(used_colors, len(used_colors))

    rows = rng.choice([3, 4])
    cols = rng.choice([5, 6])
    row_spacing = rng.choice([4, 5])
    col_spacing = 2

    row_start_max = h - 1 - (rows - 1) * row_spacing
    col_start_max = w - 1 - (cols - 1) * col_spacing
    r_start = rng.randint(0, row_start_max)
    c_start = rng.randint(0, col_start_max)
    row_positions = [r_start + i * row_spacing for i in range(rows)]
    col_positions = [c_start + i * col_spacing for i in range(cols)]

    shift0 = rng.randrange(len(base_seq))
    shift_step = 1 if len(base_seq) <= 2 else rng.choice([1, len(base_seq) - 1])
    for ridx, r in enumerate(row_positions):
        shift = (shift0 + ridx * shift_step) % len(base_seq)
        row_colors = _repeat_to_length(base_seq[shift:] + base_seq[:shift], cols)
        for cidx, c in enumerate(col_positions):
            g[r, c] = int(row_colors[cidx])

    return g


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(300):
        g = _build_marker_grid(rng)
        out = main(g)
        metrics = _variant_metrics(g)
        if metrics["real_markers"] < 12:
            continue
        if metrics["used_colors"] < 3:
            continue
        if metrics["multi_phase_colors"] < 2:
            continue
        if metrics["overlap_pixels"] < 8:
            continue
        if metrics["batch_vs_scan_pixels"] < 6:
            continue
        if metrics["cycle_vs_fixed_pixels"] < 10:
            continue
        assert out.shape == g.shape
        return g

    raise RuntimeError("failed to generate a D2/2+D9/0-visible instance")
