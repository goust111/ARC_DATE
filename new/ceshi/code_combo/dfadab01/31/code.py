# Combo variant of dfadab01 (index 31)
# combo_spec: D3/2+D7/1
# Name: 对角像素对+对角线排列

from __future__ import annotations

import random

import numpy as np


# Design analysis
# a) Concrete behavior and trace:
#    - In the base task, one marker-colored pixel directly gives the top-left corner of one 4x4
#      glyph. In D3/2, that anchor is encoded as a same-color diagonal pair whose midpoint is the
#      old anchor. In this combo, those decoded anchors are no longer scattered on a grid; they are
#      arranged as a staircase along one main diagonal or one anti-diagonal.
#    - Example: suppose the real anchors are (2, 12), (7, 7), and (12, 2), which lie on one
#      anti-diagonal with step 5. If the color at (7, 7) is 5, the input does not contain a single
#      pixel at (7, 7); instead it contains a same-color pair such as (5, 5) and (9, 9). The
#      solver must first pair those two pixels, average them back to (7, 7), then paint the fixed
#      4x4 magenta glyph from that decoded anchor. Relative to the original task, both the marker
#      representation and the global placement pattern have changed.
# b) Complexity calibration:
#    - The marker encoding stays simple: every logical marker is still exactly two same-color pixels
#      at offset (±2, ±2), so the pairing rule is learnable from a few examples.
#    - The diagonal-layout dimension is kept visible by generating one whole staircase per sample,
#      with 3-4 logical markers on a 20x20 board and a uniform step of 5 or 6. That is enough
#      repetition for humans to notice the diagonal arrangement, while still leaving room for the
#      optional legend and avoiding crowded layouts.
#    - Marker colors are drawn from only 1-3 types per sample, with at least one repeated color
#      once there are 3+ markers. This prevents the task from degenerating into "every color is
#      unique so pairing is trivial", but keeps the search space small enough to stay readable.
# c) Hidden-assumption audit of the original solver:
#    - `np.argwhere(g == marker_color)` no longer returns one logical object per coordinate; it
#      returns two endpoints per logical marker. The combo solver therefore decodes centers before
#      any glyph painting happens.
#    - `_has_source_glyph(g, r - 4, c - 4, ...)` still works after decoding because the legend rule
#      is unchanged semantically: the label marker's decoded center remains exactly one 4x4 block
#      below-right of the sample glyph's top-left corner.
#    - `_paint_clipped(...)` and the glyph dictionary do not depend on how anchors are encoded or
#      arranged globally, so they remain valid without modification.
#    - The new `_decode_centers(...)` introduces a fresh assumption: each marker-colored pixel must
#      have exactly one same-color partner at delta (±4, ±4), and no pixel may simultaneously look
#      compatible with multiple partners. The generator explicitly enforces and validates this.
# d) Placement legality:
#    - Real decoded anchors are placed on a single diagonal line with uniform step 5 or 6. Since
#      the glyph size is 4, adjacent 4x4 output boxes differ by at least 5 rows and 5 columns and
#      therefore cannot overlap.
#    - Marker endpoints stay inside the board by keeping decoded centers in the closed range
#      `[2, h-3] x [2, w-3]`. This leaves room for the ±2 endpoint offsets while still allowing the
#      painted 4x4 glyph itself to be clipped near the lower or right boundary, matching the
#      original task's clipping behavior.
#    - The optional legend is added only if its glyph cells and its diagonal label pair do not hit
#      existing nonzero cells and do not break pair uniqueness.
# e) Visual observability:
#    - Across 2-4 examples, a human can see two simultaneous regularities: every control object is
#      a two-pixel diagonal marker, and those marker objects themselves march along one diagonal
#      staircase. The output removes the input pairs and replaces them with the same fixed 4x4 glyph
#      alphabet as before, so the input/output comparison highlights both changes cleanly.
# f) Visual cleanliness:
#    - Using a single staircase instead of a dense grid avoids clutter. Keeping only one legend
#      entry and limiting the board to one real diagonal chain prevents the input from becoming a
#      mixture of overlapping local and global patterns.
#    - Marker colors {2, 3, 5, 8} remain disjoint from glyph colors {1, 4, 6, 7}, so legend cells
#      are visually distinct from control markers.
# g) Change inferability:
#    - No new color remapping is introduced. The only new inference burden is structural: pair the
#      same-color diagonals, decode their centers, and notice that those decoded centers follow one
#      diagonal line. Because the mapping from marker color to glyph shape/color is unchanged and
#      optionally demonstrated by the legend, the fusion remains recoverable from a few examples.
# New solver assumptions:
#    - Every marker-colored pixel belongs to exactly one same-color pair at offset (±4, ±4).
#    - All real decoded anchors in generated samples lie on one main diagonal or one anti-diagonal,
#      although the solver itself does not need to rely on that extra regularity.
#    - The generator checks both pair uniqueness and the oracle condition `main(input) == output`.


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
GRID_SHAPES = ((20, 20),)
DIAGONAL_STEPS = (5, 6)
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
    Decode each diagonal two-pixel marker into a fixed 4x4 glyph.

    Input structure:
    - Black background with control colors 2, 3, 5, and/or 8, but each logical marker is encoded
      as two same-color pixels on opposite corners of a 5x5 box rather than a single anchor pixel.
    - The decoded marker centers form one staircase along a main diagonal or one anti-diagonal.
    - Some inputs also include one sample 4x4 glyph in the output color, plus a matching diagonal
      label pair whose decoded center sits one glyph-block below-right of that sample.

    Rule:
    - Pair same-color pixels that differ by four rows and four columns; the midpoint of each pair is
      the glyph anchor.
    - Replace each decoded anchor by its learned 4x4 glyph: 2->yellow frame, 3->blue ring,
      5->magenta diagonal blocks, 8->orange X.
    - Paint each glyph with its top-left corner at that decoded anchor, clipping at the grid edge.
    - Ignore legend sample glyphs and skip the diagonal label pairs that belong to them.

    Key insight:
    - The colored input pixels are not direct paint locations anymore; they are encoded endpoints of
      anchors. Once each same-color diagonal pair is decoded to its midpoint, the rest is the same
      fixed glyph-substitution rule as in the original task.
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


def _placements_follow_one_diagonal(placements: list[tuple[int, int, int]]) -> bool:
    centers = sorted((r, c) for _, r, c in placements)
    if len(centers) < 2:
        return False
    deltas = [(r1 - r0, c1 - c0) for (r0, c0), (r1, c1) in zip(centers, centers[1:])]
    step_r, step_c = deltas[0]
    if step_r not in DIAGONAL_STEPS or abs(step_c) != step_r:
        return False
    return all(dr == step_r and dc == step_c for dr, dc in deltas)


def _choose_diagonal_centers(h: int, w: int) -> tuple[str, int, list[tuple[int, int]]]:
    direction = random.choice(("main", "anti"))
    step = random.choice(DIAGONAL_STEPS)
    max_len = 1 + min((h - 5) // step, (w - 5) // step)
    assert max_len >= 3
    length = random.randint(3, max_len)

    max_start_r = h - 3 - step * (length - 1)
    start_r = random.randint(2, max_start_r)
    centers: list[tuple[int, int]] = []

    if direction == "main":
        max_start_c = w - 3 - step * (length - 1)
        start_c = random.randint(2, max_start_c)
        centers = [(start_r + step * i, start_c + step * i) for i in range(length)]
    else:
        min_start_c = 2 + step * (length - 1)
        start_c = random.randint(min_start_c, w - 3)
        centers = [(start_r + step * i, start_c - step * i) for i in range(length)]

    return direction, step, centers


def _assign_marker_colors(n_markers: int) -> list[int]:
    max_used = min(3, max(1, n_markers - 1))
    used_count = random.randint(1, max_used)
    used_markers = random.sample(list(GLYPHS.keys()), used_count)

    assignments = used_markers.copy()
    while len(assignments) < n_markers:
        assignments.append(random.choice(used_markers))
    random.shuffle(assignments)
    return assignments


def _legend_candidates(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h - 6) for c in range(w - 6)]


def _try_add_legend(inp: np.ndarray, used_markers: list[int], diag_index: int) -> bool:
    if not used_markers or random.random() >= LEGEND_PROB:
        return True

    h, w = inp.shape
    marker_color = random.choice(used_markers)
    glyph_color, mask = GLYPHS[marker_color]
    candidates = _legend_candidates(h, w)
    random.shuffle(candidates)

    for r0, c0 in candidates:
        center_r, center_c = r0 + 4, c0 + 4
        label_cells = _pair_cells(center_r, center_c, diag_index)

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
    line_kind, _, centers = _choose_diagonal_centers(h, w)
    pair_diag_index = 1 if line_kind == "main" else 0
    assignments = _assign_marker_colors(len(centers))
    used_markers = sorted(set(assignments))

    inp = np.zeros((h, w), dtype=int)
    placements: list[tuple[int, int, int]] = []
    occupied: set[tuple[int, int]] = set()

    for (r0, c0), marker_color in zip(centers, assignments):
        pair = _pair_cells(r0, c0, pair_diag_index)
        if pair[0] in occupied or pair[1] in occupied:
            return None
        for rr, cc in pair:
            if not (0 <= rr < h and 0 <= cc < w):
                return None
            inp[rr, cc] = marker_color
            occupied.add((rr, cc))
        placements.append((marker_color, r0, c0))

    if not _placements_follow_one_diagonal(placements):
        return None
    if not _pairing_is_unique(inp):
        return None
    if not _try_add_legend(inp, used_markers, pair_diag_index):
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
