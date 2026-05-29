# Combo variant of dfadab01 (index 35)
# combo_spec: D3/0+D1/0
# Name: 2x2色块标记+字形放大

from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


# Design analysis for combo D3/0 + D1/0
# --------------------------------------
# a) Concrete behavior and trace
#    - Original task:
#      one marker-colored pixel at (r, c) directly anchors one fixed 4x4 glyph.
#      A legend, when present, is a 4x4 sample glyph plus a same-color label pixel at (r+4, c+4).
#    - Fused task:
#      each logical marker becomes an isolated 2x2 same-color square whose top-left is still the
#      anchor. The substituted glyph is no longer 4x4; it is the canonical 4x4 mask enlarged by
#      factor 2, so the output glyph is 8x8. The legend scales in the same way: an 8x8 sample
#      glyph in the output color plus a 2x2 label block at (r+8, c+8).
#    - Small trace:
#      original: red marker cell at (6, 10) -> yellow 4x4 frame at top-left (6, 10).
#      fused: red 2x2 block covering rows 6..7, cols 10..11 -> yellow 8x8 frame at top-left
#      (6, 10). If an 8x8 yellow sample already exists at (6, 10) and a red 2x2 label block sits
#      at (14, 18), that block is legend metadata and must be skipped.
#    - The two changes are coupled, not independent: the solver must first recognize that the
#      source object is a 2x2 marker square, and then replace that larger marker with an 8x8 glyph
#      aligned to the marker square's top-left corner.
#
# b) Complexity calibration
#    - The scale choice is fixed to exactly x2. This keeps the rule visible and learnable from a
#      few examples while still making the output clearly different from the base task.
#    - Using 8x8 glyphs instead of 6x6 preserves the original 4x4 dictionary exactly via pixel
#      enlargement, so the transformed shapes remain recognizable.
#    - Marker blocks are always perfect 2x2 squares, not arbitrary rectangles, which avoids
#      unnecessary branching in the recognition rule.
#
# c) Hidden-assumption audit against the original solver
#    - `np.argwhere(g == marker_color)` no longer returns one logical marker per cell. In the fused
#      task, four marker-colored cells belong to one logical object, so main() must decode 2x2
#      connected components first.
#    - `_has_source_glyph(g, r-4, c-4, ...)` depended on legend labels being single pixels offset
#      by the old glyph size. That assumption breaks twice: label size becomes 2x2 and glyph size
#      becomes 8. The new skip test is "does a full 8x8 matching glyph exist exactly 8 rows and
#      8 columns above-left of this 2x2 block?".
#    - `_paint_clipped()` still works conceptually, but its loop bounds must follow the enlarged
#      mask shape rather than the old hard-coded 4x4.
#
# d) Placement legality
#    - Generator places marker anchors on a coarse lattice with spacing 8 or 9, so enlarged glyph
#      footprints do not overlap even before clipping.
#    - Every marker-colored connected component is exactly one isolated 2x2 square.
#    - Optional legends are only placed where the full 8x8 glyph and its 2x2 label fit without
#      overlapping markers or any nonzero output cells.
#
# e) Visual observability
#    - Humans can directly see that marker-colored objects are now 2x2 blocks rather than pixels.
#    - The output glyphs are visibly doubled in scale, so the mapping is "large marker -> large
#      glyph", not merely a relabeling of the original task.
#    - The generator includes repeated examples of the same source colors and often includes a
#      legend, making the enlarged correspondence easy to infer.
#
# f) Visual cleanliness
#    - Only three element types appear: sparse 2x2 marker blocks, at most one enlarged legend, and
#      black background. There is no multi-layer overlap inside the same local region.
#    - Wide spacing keeps neighboring 8x8 glyphs visually separated in the output.
#
# g) Color-rule inferability
#    - The color mapping itself is unchanged from the source task. Marker colors {2,3,5,8} still
#      select output colors {4,1,6,7}; the new difficulty comes from source-object scale and glyph
#      scale, both visible in the grid.
#
# h) New solver assumptions and how generate_input() guarantees them
#    - Every marker-colored connected component is a 2x2 filled square with area 4.
#      generate_input() only writes markers in that format and never lets same-color blocks touch.
#    - Any pre-drawn glyph-colored sample is a full, unclipped 8x8 enlarged glyph.
#      generate_input() searches only full-fit legend locations.
#    - A legend label block, if present, sits exactly at (+8, +8) from its sample glyph top-left.
#      generate_input() uses that fixed convention only.
#    - Real marker anchors are separated by at least 8 cells in both row and column.
#      generate_input() uses an 8/9-spaced lattice and verifies `main(input) == expected_output`.


SCALE = 2
MARKER_SIZE = 2

BASE_GLYPHS: dict[int, tuple[int, np.ndarray]] = {
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


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(bool), np.ones((factor, factor), dtype=bool))


GLYPHS: dict[int, tuple[int, np.ndarray]] = {
    marker_color: (glyph_color, _scale_mask(mask, SCALE))
    for marker_color, (glyph_color, mask) in BASE_GLYPHS.items()
}

GLYPH_SIZE = next(iter(GLYPHS.values()))[1].shape[0]
MARKER_COLORS = tuple(sorted(GLYPHS))
GLYPH_COLORS = tuple(sorted({glyph_color for glyph_color, _mask in GLYPHS.values()}))
ALLOWED_INPUT_COLORS = {0, *MARKER_COLORS, *GLYPH_COLORS}
GRID_SIZES = (20, 24)
LEGEND_PROB = 0.7
MAX_GENERATION_ATTEMPTS = 400


def _component_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            out.append((min(rows), min(cols), max(rows), max(cols)))
    return out


def _marker_components(g: np.ndarray, marker_color: int) -> list[tuple[int, int]]:
    comps: list[tuple[int, int]] = []
    for r0, c0, r1, c1 in _component_bboxes(g == marker_color):
        area = int(np.sum(g[r0 : r1 + 1, c0 : c1 + 1] == marker_color))
        assert area == 4, "marker component must have area 4"
        assert r1 - r0 + 1 == MARKER_SIZE and c1 - c0 + 1 == MARKER_SIZE, "marker component must be 2x2"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == marker_color), "marker component must be solid"
        comps.append((int(r0), int(c0)))
    return sorted(comps)


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    gh, gw = mask.shape
    if r0 < 0 or c0 < 0 or r0 + gh > h or c0 + gw > w:
        return False
    patch = g[r0 : r0 + gh, c0 : c0 + gw]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _paint_clipped(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    h, w = out.shape
    gh, gw = mask.shape
    for dr in range(gh):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(gw):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                out[rr, cc] = color


def _draw_full_glyph(inp: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    gh, gw = mask.shape
    for dr in range(gh):
        for dc in range(gw):
            if bool(mask[dr, dc]):
                inp[r0 + dr, c0 + dc] = color


def _render_markers(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r0, c0, marker_color in markers:
        glyph_color, mask = GLYPHS[int(marker_color)]
        _paint_clipped(out, int(r0), int(c0), int(glyph_color), mask)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each 2x2 marker block by its enlarged 8x8 glyph.

    Input structure:
    - Black background with marker colors 2, 3, 5, and/or 8, but each logical marker is now an
      isolated 2x2 same-color square.
    - Some inputs also include one legend: an already-drawn 8x8 sample glyph in the output color,
      together with a matching 2x2 label block positioned one enlarged glyph-width below-right.

    Rule:
    - Read each 2x2 marker square by its top-left corner.
    - Replace it with the corresponding x2-enlarged glyph: 2->yellow frame, 3->blue ring,
      5->pink diagonal blocks, 8->orange X.
    - Paint each glyph with its top-left corner aligned to the marker block's top-left corner,
      clipping any part that extends beyond the grid.
    - Remove legend glyphs and skip the 2x2 label blocks that belong to them.

    Key insight:
    - The color dictionary is unchanged, but both the source marker object and the substituted glyph
      have been scaled up together, so the solver must detect 2x2 control blocks and map them to
      8x8 enlarged versions of the original glyph alphabet.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset(ALLOWED_INPUT_COLORS)

    out = np.zeros_like(g)
    seen_marker = False
    for marker_color, (glyph_color, mask) in GLYPHS.items():
        anchors = _marker_components(g, marker_color)
        if anchors:
            seen_marker = True
        for r0, c0 in anchors:
            if _has_source_glyph(g, r0 - GLYPH_SIZE, c0 - GLYPH_SIZE, glyph_color, mask):
                continue
            _paint_clipped(out, r0, c0, glyph_color, mask)

    assert seen_marker
    return out


def _sample_axis_positions(rng: random.Random, size: int, count: int) -> list[int] | None:
    steps = [8, 9]
    rng.shuffle(steps)
    for step in steps:
        span = step * (count - 1)
        max_start = size - MARKER_SIZE - span
        if max_start < 0:
            continue
        start = rng.randint(0, max_start)
        return [start + step * i for i in range(count)]
    return None


def _build_occupancy(rng: random.Random, rows: int, cols: int) -> np.ndarray:
    occ = np.zeros((rows, cols), dtype=bool)

    for r in range(rows):
        occ[r, rng.randrange(cols)] = True
    for c in range(cols):
        occ[rng.randrange(rows), c] = True

    min_fill = max(rows + cols - 1, math.ceil(0.6 * rows * cols))
    target = rng.randint(min_fill, rows * cols)
    vacant = [(r, c) for r in range(rows) for c in range(cols) if not bool(occ[r, c])]
    rng.shuffle(vacant)
    for r, c in vacant[: max(0, target - int(occ.sum()))]:
        occ[r, c] = True
    return occ


def _assign_marker_colors(
    rng: random.Random,
    anchors: list[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    type_count = rng.randint(1, min(4, len(anchors)))
    active = rng.sample(list(MARKER_COLORS), type_count)
    order = anchors[:]
    rng.shuffle(order)

    colored: list[tuple[int, int, int]] = []
    for idx, (r0, c0) in enumerate(order):
        if idx < len(active):
            marker_color = active[idx]
        else:
            marker_color = rng.choice(active)
        colored.append((r0, c0, marker_color))
    colored.sort()
    return colored


def _paint_marker_block(inp: np.ndarray, r0: int, c0: int, marker_color: int) -> None:
    inp[r0 : r0 + MARKER_SIZE, c0 : c0 + MARKER_SIZE] = int(marker_color)


def _legend_cells(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    gh, gw = mask.shape
    for dr in range(gh):
        for dc in range(gw):
            if bool(mask[dr, dc]):
                cells.add((r0 + dr, c0 + dc))
    for dr in range(MARKER_SIZE):
        for dc in range(MARKER_SIZE):
            cells.add((r0 + gh + dr, c0 + gw + dc))
    return cells


def _block_has_same_color_neighbor(inp: np.ndarray, r0: int, c0: int, size: int, color: int) -> bool:
    h, w = inp.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r0 + size + 1)
    cc1 = min(w, c0 + size + 1)
    patch = inp[rr0:rr1, cc0:cc1]
    if not np.any(patch == color):
        return False
    mask = np.ones_like(patch, dtype=bool)
    mask[r0 - rr0 : r0 - rr0 + size, c0 - cc0 : c0 - cc0 + size] = False
    return bool(np.any(patch[mask] == color))


def _try_add_legend(
    rng: random.Random,
    inp: np.ndarray,
    expected: np.ndarray,
    used_marker_colors: set[int],
) -> bool:
    if not used_marker_colors or rng.random() >= LEGEND_PROB:
        return False

    h, w = inp.shape
    candidates: list[tuple[int, int, int]] = []
    for marker_color in sorted(used_marker_colors):
        glyph_color, mask = GLYPHS[marker_color]
        gh, gw = mask.shape
        max_r0 = h - gh - MARKER_SIZE
        max_c0 = w - gw - MARKER_SIZE
        if max_r0 < 0 or max_c0 < 0:
            continue
        for r0 in range(max_r0 + 1):
            for c0 in range(max_c0 + 1):
                cells = _legend_cells(mask, r0, c0)
                if any(inp[r, c] != 0 for r, c in cells):
                    continue
                if any(expected[r, c] != 0 for r, c in cells):
                    continue
                label_r = r0 + gh
                label_c = c0 + gw
                if _block_has_same_color_neighbor(inp, label_r, label_c, MARKER_SIZE, marker_color):
                    continue
                candidates.append((marker_color, r0, c0))

    if not candidates:
        return False

    marker_color, r0, c0 = rng.choice(candidates)
    glyph_color, mask = GLYPHS[marker_color]
    _draw_full_glyph(inp, r0, c0, glyph_color, mask)
    _paint_marker_block(inp, r0 + GLYPH_SIZE, c0 + GLYPH_SIZE, marker_color)
    return True


def _legacy_like_output(inp: np.ndarray) -> np.ndarray:
    out = np.zeros_like(inp)
    for marker_color, (glyph_color, base_mask) in BASE_GLYPHS.items():
        for r, c in np.argwhere(inp == marker_color):
            _paint_clipped(out, int(r), int(c), glyph_color, base_mask)
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(MAX_GENERATION_ATTEMPTS):
        size = rng.choice(GRID_SIZES)
        max_axis_count = 3
        row_count = rng.randint(2, max_axis_count)
        col_count = rng.randint(2, max_axis_count)

        row_positions = _sample_axis_positions(rng, size, row_count)
        col_positions = _sample_axis_positions(rng, size, col_count)
        if row_positions is None or col_positions is None:
            continue

        occ = _build_occupancy(rng, row_count, col_count)
        anchors = [(row_positions[r], col_positions[c]) for r in range(row_count) for c in range(col_count) if bool(occ[r, c])]
        markers = _assign_marker_colors(rng, anchors)
        expected = _render_markers((size, size), markers)

        inp = np.zeros((size, size), dtype=int)
        for r0, c0, marker_color in markers:
            _paint_marker_block(inp, r0, c0, marker_color)

        _try_add_legend(rng, inp, expected, {marker_color for _, _, marker_color in markers})

        if not np.array_equal(main(inp), expected):
            continue

        legacy = _legacy_like_output(inp)
        if np.array_equal(legacy, expected):
            continue

        return inp

    raise RuntimeError("failed to generate a valid fused dfadab01 combo instance")
