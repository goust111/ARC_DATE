# Combo variant of dfadab01 (index 25)
# combo_spec: D6/2+D9/1
# Name: 紧凑裁剪+先消除样本

from __future__ import annotations

import math

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


# Design analysis for combo D6/2 + D9/1
# --------------------------------------
# a) Concrete behavior trace
#    - Original dfadab01 treats each marker pixel as the top-left of a fixed 4x4 glyph, clips that glyph
#      to the input frame, and simply ignores a legend label when it sees the matching 4x4 sample above-left.
#    - The fused rule changes two parts together:
#        1. D9/1 makes the process explicitly sequential. First find every legend (sample glyph in output
#           color + its label marker at +4,+4) and erase both from a working copy. Only after that do we
#           collect the remaining real markers and expand them.
#        2. D6/2 removes the fixed-frame output. Glyphs are placed on an unconstrained canvas and the final
#           answer is tightly cropped to the occupied glyph rectangle instead of staying the input size.
#    - Small example:
#        input has a blue sample ring at rows 0..3, cols 0..3, its green label at (4,4), and real green
#        markers at (7,2) and (11,6).
#      Original task would output a 20x20-style full frame with rings clipped to the original grid.
#      Fused task first erases the sample and label, then stamps the two real rings, and finally returns
#      only rows 7..14 and cols 2..9. The legend completely disappears and the outer blank border is gone.
#
# b) Complexity calibration
#    - The input vocabulary is unchanged: the same four marker colors and the same fixed 4x4 glyph alphabet
#      appear, so the task still has strong repeated evidence.
#    - The generator always includes exactly one legend and places it outside the real-marker bounding box.
#      That makes the serial "erase samples first" step observable, because a solver that skips it would
#      keep extra content and a larger crop.
#    - Real markers occupy a regular 2D lattice with spacing >= 4, so glyphs remain non-overlapping and the
#      tight crop is easy to read as "remove the empty margin", not as a noisy packing effect.
#
# c) Implicit-assumption audit against the original main()
#    - `_paint_clipped(...)` depended on the original input frame as the final canvas. D6/2 breaks that:
#      rendering must no longer crop to `grid.shape`; instead the solver paints full glyphs and crops once
#      at the end.
#    - The original `if _has_source_glyph(...): continue` only suppressed the label marker on the fly. D9/1
#      is stronger: the sample glyph pixels themselves must be removed before marker collection, so the new
#      solver performs an explicit cleanup pass on a working grid.
#    - The original solver could render directly onto an all-black output. The fused solver instead reasons
#      in serial phases: detect legends on the original grid, erase them from a working copy, clear marker
#      anchor pixels, then stamp glyphs and crop.
#
# d) Placement legality
#    - Generator markers stay on a lattice with row/column spacing 4-6, so 4x4 glyph boxes never overlap.
#    - The legend is restricted to a top corner band while real markers start several rows lower, so legend
#      pixels and label markers never collide with real marker anchors.
#    - Real-marker bounding boxes are kept strictly inside the input frame. This is not required by the
#      fused solver, but it keeps the scenes clean and makes the crop effect about margin removal rather than
#      about recovering off-frame fragments.
#
# e) Visual observability
#    - Humans can still infer the marker->glyph dictionary from repeated placements across the lattice.
#    - The crop change is obvious because the output hugs the marker cluster instead of preserving the large
#      empty border from the input.
#    - The sample-erasure step is also visible because every training input contains a legend that is absent
#      from the output and positioned outside the real cluster; if it were not erased first, it would alter
#      both the output contents and the crop.
#
# f) Visual cleanliness
#    - Only three visual ingredients remain: black background, sparse marker lattice, and one isolated
#      legend entry. There are no overlapping glyphs or multi-layer collisions.
#    - The top-band legend and lower marker cluster are separated by at least one empty row, so the input
#      is easy to segment by eye.
#    - Tight cropping removes the unused border but preserves the relative layout of the real glyph cluster.
#
# g) New assumptions introduced by main(), and generator support
#    - `main()` now assumes any legend label has the exact matching 4x4 sample at (-4,-4). The generator
#      constructs legends in that canonical form.
#    - `main()` now assumes the crop should be driven by real rendered content after legend erasure. The
#      generator always creates at least four real markers and keeps the legend disjoint from that cluster.
#    - `main()` allows full-glyph rendering beyond the input frame, but the generator keeps real markers
#      safely inside the frame so the variation remains visually tidy and easy to infer from few examples.


GRID_SHAPE = (20, 20)
ROW_COUNT_CHOICES = [2, 3]
COL_COUNT_CHOICES = [2, 3, 4]
SPACING_CHOICES = [4, 5, 6]
MIN_FILL_RATE = 0.6
MAX_RETRIES = 500
ALLOWED_COLORS = {0, *GLYPHS.keys(), 1, 4, 6, 7}


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    patch = g[r0 : r0 + 4, c0 : c0 + 4]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _paint_full(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    for dr in range(4):
        for dc in range(4):
            if bool(mask[dr, dc]):
                out[r0 + dr, c0 + dc] = color


def _find_legend_labels(g: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
    legends: list[tuple[int, int, int, int, np.ndarray]] = []
    for marker_color, (glyph_color, mask) in GLYPHS.items():
        for r, c in np.argwhere(g == marker_color):
            rr = int(r)
            cc = int(c)
            if _has_source_glyph(g, rr - 4, cc - 4, glyph_color, mask):
                legends.append((rr, cc, marker_color, glyph_color, mask))
    return legends


def _prepare_scene(g: np.ndarray, erase_legend: bool) -> tuple[np.ndarray, list[tuple[int, int, int, np.ndarray]]]:
    work = g.copy()
    if erase_legend:
        for r, c, _marker_color, _glyph_color, _mask in _find_legend_labels(g):
            work[r - 4 : r, c - 4 : c] = 0
            work[r, c] = 0

    placements: list[tuple[int, int, int, np.ndarray]] = []
    for marker_color, (glyph_color, mask) in GLYPHS.items():
        for r, c in np.argwhere(work == marker_color):
            rr = int(r)
            cc = int(c)
            placements.append((rr, cc, glyph_color, mask))
            work[rr, cc] = 0
    return work, placements


def _render_tight(work: np.ndarray, placements: list[tuple[int, int, int, np.ndarray]]) -> np.ndarray:
    assert placements, "no real markers remain after legend removal"

    h, w = work.shape
    max_r = max(h, max(r + 4 for r, _c, _color, _mask in placements))
    max_c = max(w, max(c + 4 for _r, c, _color, _mask in placements))
    canvas = np.zeros((max_r, max_c), dtype=int)
    canvas[:h, :w] = work

    for r, c, glyph_color, mask in placements:
        _paint_full(canvas, r, c, glyph_color, mask)

    pts = np.argwhere(canvas != 0)
    assert pts.size, "rendering produced an empty canvas"
    r0 = int(pts[:, 0].min())
    r1 = int(pts[:, 0].max())
    c0 = int(pts[:, 1].min())
    c1 = int(pts[:, 1].max())
    return canvas[r0 : r1 + 1, c0 : c1 + 1]


def _solve_variant(g: np.ndarray, erase_legend: bool) -> np.ndarray:
    work, placements = _prepare_scene(g, erase_legend=erase_legend)
    return _render_tight(work, placements)


def _shape_diff_score(a: np.ndarray, b: np.ndarray) -> int:
    ah, aw = a.shape
    bh, bw = b.shape
    h = max(ah, bh)
    w = max(aw, bw)
    pa = np.zeros((h, w), dtype=int)
    pb = np.zeros((h, w), dtype=int)
    pa[:ah, :aw] = a
    pb[:bh, :bw] = b
    return int(np.count_nonzero(pa != pb))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove any legend entry first, then expand the remaining markers and crop the result to the real glyph cluster.

    Input structure:
    - Black background with single-cell markers in colors 2, 3, 5, and/or 8 arranged on a sparse lattice.
    - Exactly the same fixed 4x4 glyph alphabet as the base task is used.
    - A legend may appear as one sample 4x4 glyph in its output color plus a matching label marker at the
      sample's lower-right offset (+4,+4).

    Rule:
    - Step 1: detect each legend by pairing a marker with the matching 4x4 sample above-left, and erase both
      the sample region and the label marker from a working grid.
    - Step 2: expand every remaining marker into its fixed 4x4 glyph at the same top-left anchor, with no
      clipping to the original input frame.
    - Return only the tight crop around the rendered real glyphs, so empty border cells from the input do
      not survive in the output.

    Key insight:
    - The legend is a temporary decoding aid, not part of the final scene. The output is therefore built
      from the cleaned marker set, and its size is determined by the true glyph cluster rather than by the
      original input frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset(ALLOWED_COLORS)
    return _solve_variant(g, erase_legend=True)


def _sample_positions(
    rng: np.random.Generator,
    count_choices: list[int],
    start_min: int,
    start_max_extra_margin: int,
    total_len: int,
) -> list[int] | None:
    count = int(rng.choice(count_choices))
    spacing = int(rng.choice(SPACING_CHOICES))
    span = (count - 1) * spacing
    start_max = total_len - 4 - span - start_max_extra_margin
    if start_max < start_min:
        return None
    start = int(rng.integers(start_min, start_max + 1))
    return [start + i * spacing for i in range(count)]


def _choose_active_cells(rng: np.random.Generator, rows: int, cols: int) -> list[tuple[int, int]]:
    total = rows * cols
    target = int(rng.integers(max(4, math.ceil(MIN_FILL_RATE * total)), total + 1))

    active = {
        (0, 0),
        (0, cols - 1),
        (rows - 1, 0),
        (rows - 1, cols - 1),
    }
    rest = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in active]
    rng.shuffle(rest)
    for cell in rest[: max(0, target - len(active))]:
        active.add(cell)
    return sorted(active)


def _draw_legend(
    grid: np.ndarray,
    top: int,
    left: int,
    marker_color: int,
) -> None:
    glyph_color, mask = GLYPHS[marker_color]
    for dr in range(4):
        for dc in range(4):
            if bool(mask[dr, dc]):
                grid[top + dr, left + dc] = glyph_color
    grid[top + 4, left + 4] = marker_color


def _original_style_output(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    out = np.zeros_like(g)
    for marker_color, (glyph_color, mask) in GLYPHS.items():
        for r, c in np.argwhere(g == marker_color):
            rr = int(r)
            cc = int(c)
            if _has_source_glyph(g, rr - 4, cc - 4, glyph_color, mask):
                continue
            for dr in range(4):
                for dc in range(4):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[dr, dc]):
                        out[nr, nc] = glyph_color
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h, w = GRID_SHAPE
    grid = np.zeros((h, w), dtype=int)

    row_positions = _sample_positions(
        rng,
        count_choices=ROW_COUNT_CHOICES,
        start_min=6,
        start_max_extra_margin=1,
        total_len=h,
    )
    col_positions = _sample_positions(
        rng,
        count_choices=COL_COUNT_CHOICES,
        start_min=1,
        start_max_extra_margin=1,
        total_len=w,
    )
    if row_positions is None or col_positions is None:
        return None

    active_cells = _choose_active_cells(rng, len(row_positions), len(col_positions))
    n_types = int(rng.integers(2, len(GLYPHS) + 1))
    marker_types = [int(x) for x in rng.choice(list(GLYPHS.keys()), size=n_types, replace=False)]

    for rr, cc in active_cells:
        grid[row_positions[rr], col_positions[cc]] = int(rng.choice(marker_types))

    legend_marker = int(rng.choice(marker_types))
    legend_top = int(rng.integers(0, 2))
    legend_left = int(rng.integers(0, 2))
    _draw_legend(grid, legend_top, legend_left, legend_marker)

    out = main(grid)
    if out.size == 0:
        return None

    # D6/2 must matter: the fused output should be smaller than the original frame.
    if out.shape == grid.shape:
        return None

    # D9/1 must matter: skipping the legend-erasure phase should change the cropped result.
    no_erase = _solve_variant(grid, erase_legend=False)
    if _shape_diff_score(out, no_erase) == 0:
        return None

    # The fused behavior should not collapse to the base task.
    if np.array_equal(out, _original_style_output(grid)):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
