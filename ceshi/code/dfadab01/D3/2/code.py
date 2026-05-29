# Variant of dfadab01
# Dimension: D3 — 标记来源对象 (src)
# Change: 散布在网格中的单像素标记，使用字形字典中的颜色（2、3、5、8），标记位置即字形放置的左上角... → 对角线像素对标记

from __future__ import annotations

import random

import numpy as np


# Design analysis
# a) Concrete behavior and trace:
#    - Real control markers are no longer single pixels. Each one is a same-color diagonal pair
#      sitting on opposite corners of a 5x5 box: either NW/SE or NE/SW.
#    - The midpoint of that 5x5 box is the old anchor coordinate. That recovered midpoint is used
#      as the top-left corner for the 4x4 glyph, so the main transformation after decoding stays
#      the same as the original task.
#    - Example: pixels 2 at (4, 9) and (8, 13) decode to center (6, 11). The original task would
#      have used a single 2 already at (6, 11); the variant forces the solver to pair first, then
#      average, then paint the yellow frame from that midpoint.
# b) Complexity calibration:
#    - Using a fixed 5x5 marker box keeps the rule visible and avoids overfitting to arbitrary
#      rectangle sizes, but it still creates a real pairing problem because identical colors may
#      repeat across the grid.
#    - Two diagonal orientations are used so the inputs do not collapse to a single visual motif.
# c) Hidden-assumption audit of the original solver:
#    - `np.argwhere(g == marker_color)` originally returned one logical marker per pixel; that is no
#      longer true, so the variant must reconstruct markers from diagonal pairs first.
#    - `_has_source_glyph(g, r - 4, c - 4, ...)` depended on `r, c` being the old anchor. That
#      remains valid after decoding pair midpoints, so the legend-skip rule can stay unchanged.
#    - `_paint_clipped(...)` and the glyph dictionary do not depend on marker shape, so they are
#      preserved.
# d) Placement legality:
#    - Generator centers live on a coarse grid with spacing 5 or 6. That keeps 4x4 output glyphs
#      non-overlapping and prevents accidental cross-pairings between different markers of the same
#      color, because valid pair endpoints differ by exactly 4 in row and column.
#    - Legend labels are also diagonal pairs; their decoded center is still exactly (+4, +4) from
#      the sample glyph top-left, so the solver can detect and skip them.
# e) Visual observability:
#    - From a few examples, a human can see that no single colored pixel directly expands anymore;
#      instead, two same-color corner pixels imply one glyph centered between them.
#    - The output glyph shapes and colors are unchanged, so the only conceptual novelty is marker
#      sourcing, which matches the requested D3 change.
# f) Visual cleanliness:
#    - Marker pairs are sparse and occupy only two cells. Legend sample glyphs use output colors
#      {1,4,6,7}, which are disjoint from marker colors {2,3,5,8}, so role confusion is limited.
# g) Color inference:
#    - No new color logic is introduced. The fixed mapping from marker color to glyph color/pattern
#      is preserved, and the optional legend still gives a direct visible exemplar.
# New solver assumptions:
#    - Every marker-colored pixel belongs to exactly one same-color diagonal pair at offset (±4, ±4).
#    - Generator enforces those constraints and validates them with both pair-consistency checks and
#      an oracle `main(input) == expected_output` test.


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

PAIR_RADIUS = 2
GRID_SHAPES = ((10, 10), (20, 20))
LEGEND_PROB = 0.65
GEN_RETRIES = 500

PAIR_DIAGONALS: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((-PAIR_RADIUS, -PAIR_RADIUS), (PAIR_RADIUS, PAIR_RADIUS)),
    ((-PAIR_RADIUS, PAIR_RADIUS), (PAIR_RADIUS, -PAIR_RADIUS)),
)


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


def _pair_cells(center_r: int, center_c: int, diag_index: int) -> tuple[tuple[int, int], tuple[int, int]]:
    (dr1, dc1), (dr2, dc2) = PAIR_DIAGONALS[diag_index]
    return (center_r + dr1, center_c + dc1), (center_r + dr2, center_c + dc2)


def _decode_centers(g: np.ndarray, marker_color: int) -> list[tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(g == marker_color)]
    if not coords:
        return []
    assert len(coords) % 2 == 0, "marker pixels must come in pairs"

    coord_set = set(coords)
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in coords:
        found: list[tuple[int, int]] = []
        for dr in (-2 * PAIR_RADIUS, 2 * PAIR_RADIUS):
            for dc in (-2 * PAIR_RADIUS, 2 * PAIR_RADIUS):
                q = (r + dr, c + dc)
                if q in coord_set:
                    found.append(q)
        uniq = sorted(set(found))
        assert len(uniq) <= 1, "marker pixel participates in multiple diagonal pairs"
        neighbors[(r, c)] = uniq

    centers: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for pt in sorted(coords):
        if pt in used:
            continue
        partner_list = neighbors[pt]
        assert len(partner_list) == 1, "marker pixel is missing its diagonal partner"
        partner = partner_list[0]
        assert neighbors[partner] == [pt], "pairing must be mutual"
        used.add(pt)
        used.add(partner)
        centers.append(((pt[0] + partner[0]) // 2, (pt[1] + partner[1]) // 2))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each same-color diagonal marker pair into a fixed 4x4 glyph.

    Input structure:
    - Black background with control colors 2, 3, 5, and/or 8, but each logical marker is now two
      same-color pixels placed on opposite corners of a 5x5 box.
    - Some inputs also include one sample 4x4 glyph in the output color, with a matching diagonal
      label pair whose decoded center sits one cell-block below-right of that sample.

    Rule:
    - Pair same-color pixels that are separated by four rows and four columns; their midpoint is the
      glyph anchor coordinate.
    - Replace each decoded marker by its learned 4x4 glyph: 2->yellow frame, 3->blue ring,
      5->pink diagonal blocks, 8->orange X.
    - Paint each glyph with its top-left corner at that midpoint, clipping at the grid boundary.
    - Remove the sample glyphs and skip the label pairs that belong to them.

    Key insight:
    - The diagonal pixel pairs are only encoded anchors; once their midpoint is recovered, the task
      reduces to the original fixed glyph-substitution rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, *GLYPHS.keys(), 1, 4, 6, 7})

    out = np.zeros_like(g)
    painted_any = False
    for marker_color, (glyph_color, mask) in GLYPHS.items():
        centers = _decode_centers(g, marker_color)
        for r, c in centers:
            if _has_source_glyph(g, int(r) - 4, int(c) - 4, glyph_color, mask):
                continue
            _paint_clipped(out, int(r), int(c), glyph_color, mask)
            painted_any = True

    assert painted_any
    return out


def _pick_axis_positions(size: int) -> list[int]:
    choices: list[tuple[int, int, int, int]] = []
    for step in (5, 6):
        max_count = ((size - 3) - 2) // step + 1
        for count in range(2, max_count + 1):
            max_start = (size - 3) - step * (count - 1)
            if 2 <= max_start:
                choices.append((step, count, 2, max_start))
    step, count, lo, hi = random.choice(choices)
    start = hi if random.random() < 0.5 else random.randint(lo, hi)
    return [start + step * i for i in range(count)]


def _build_output(shape: tuple[int, int], placements: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for marker_color, r0, c0 in placements:
        glyph_color, mask = GLYPHS[marker_color]
        _paint_clipped(out, r0, c0, glyph_color, mask)
    return out


def _pairing_is_unique(g: np.ndarray) -> bool:
    try:
        for marker_color in GLYPHS:
            coords = np.argwhere(g == marker_color)
            centers = _decode_centers(g, marker_color)
            if len(coords) != 2 * len(centers):
                return False
        return True
    except AssertionError:
        return False


def _legend_candidates(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h - 6) for c in range(w - 6)]


def _try_add_legend(inp: np.ndarray, used_markers: list[int]) -> bool:
    if not used_markers or random.random() >= LEGEND_PROB:
        return True

    h, w = inp.shape
    marker_color = random.choice(used_markers)
    glyph_color, mask = GLYPHS[marker_color]
    candidates = _legend_candidates(h, w)
    random.shuffle(candidates)

    for r0, c0 in candidates:
        center_r, center_c = r0 + 4, c0 + 4
        label_cells = _pair_cells(center_r, center_c, 1)

        ok = True
        for rr, cc in label_cells:
            if not (0 <= rr < h and 0 <= cc < w) or inp[rr, cc] != 0:
                ok = False
                break
        if not ok:
            continue

        glyph_cells = [(r0 + dr, c0 + dc) for dr in range(4) for dc in range(4) if bool(mask[dr, dc])]
        for rr, cc in glyph_cells:
            if inp[rr, cc] != 0:
                ok = False
                break
        if not ok:
            continue

        trial = inp.copy()
        for rr, cc in glyph_cells:
            trial[rr, cc] = glyph_color
        for rr, cc in label_cells:
            trial[rr, cc] = marker_color
        if _pairing_is_unique(trial):
            inp[:, :] = trial
            return True

    return False


def _generate_once() -> tuple[np.ndarray, np.ndarray] | None:
    h, w = random.choice(GRID_SHAPES)
    row_positions = _pick_axis_positions(h)
    col_positions = _pick_axis_positions(w)
    slots = [(r, c) for r in row_positions for c in col_positions]

    min_filled = max(1, int(np.ceil(0.6 * len(slots))))
    n_filled = random.randint(min_filled, len(slots))
    chosen_slots = random.sample(slots, n_filled)

    used_count = random.randint(1, min(len(GLYPHS), n_filled))
    used_markers = random.sample(list(GLYPHS.keys()), used_count)
    assignments = used_markers.copy()
    while len(assignments) < n_filled:
        assignments.append(random.choice(used_markers))
    random.shuffle(assignments)

    inp = np.zeros((h, w), dtype=int)
    placements: list[tuple[int, int, int]] = []
    occupied: set[tuple[int, int]] = set()
    for (r0, c0), marker_color in zip(chosen_slots, assignments):
        diag_index = random.randint(0, 1)
        pair = _pair_cells(r0, c0, diag_index)
        if pair[0] in occupied or pair[1] in occupied:
            return None
        for rr, cc in pair:
            if not (0 <= rr < h and 0 <= cc < w):
                return None
            inp[rr, cc] = marker_color
            occupied.add((rr, cc))
        placements.append((marker_color, r0, c0))

    if not _pairing_is_unique(inp):
        return None

    if not _try_add_legend(inp, used_markers):
        return None

    out = _build_output((h, w), placements)
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _generate_once()
        if result is not None:
            inp, _ = result
            return inp
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
