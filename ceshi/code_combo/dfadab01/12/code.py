# Combo variant of dfadab01 (index 12)
# combo_spec: D3/1+D6/3
# Name: 边缘控制像素+禁止裁剪

from __future__ import annotations

import random

import numpy as np


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Original dfadab01 treated every interior marker pixel as a local placement anchor and
#      expanded it into a clipped 4x4 glyph at that same coordinate.
#    - In this combo, a placement is reconstructed from two same-color border controls:
#        * "tl": top edge cell (0, c) and left edge cell (r, 0) decode top-left (r, c)
#        * "br": bottom edge cell (H-1, c) and right edge cell (r, W-1) decode top-left (r, c)
#    - After decoding, D6/3 changes the rendering policy: if the resulting 4x4 glyph would
#      cross the grid boundary, that decoded placement is ignored entirely instead of clipped.
#    - Small trace on a 10x10 grid:
#        source task: a red marker at (7, 2) would paint the visible part of the yellow frame.
#        combo task: red at top (0, 2) and left (7, 0) decodes the same (7, 2), but 7+4>10,
#        so the pair is a valid coordinate code yet produces no output at all.
#        difference: placement is now non-local and boundary-near decodings disappear completely.
#
# b) Complexity calibration
#    - Pure border pairing becomes ambiguous if one color appears multiple times on the same side,
#      because top/left or bottom/right controls would combine combinatorially. To keep the task
#      learnable, each color uses each side at most once, so each color can decode at most one
#      "tl" placement and one "br" placement.
#    - Generator examples always contain both at least one fully fitting decoded placement and at
#      least one overflowing decoded placement. That makes the "decode first, then filter by fit"
#      branch visible instead of degenerating to the border-only variant.
#
# c) Hidden-assumption audit against the original solver
#    - Original `np.argwhere(g == marker_color)` assumed every marker cell independently meant
#      "paint here". That is false now: a single border pixel is only half of a coordinate code.
#    - Original legend test `_has_source_glyph(g, r-4, c-4, ...)` depended on a label marker being
#      one block down-right from the sample glyph. That convention no longer exists, so legend
#      detection must run after border decoding by checking whether the decoded position already
#      contains the full sample glyph in the input.
#    - Original clipped painting from every decoded point is also invalid. Rendering is now gated by
#      `_fits_glyph(...)`, so overflow positions are skipped instead of partially drawn.
#
# d) Placement legality
#    - Real decoded placements use distinct border control cells and their visible clipped footprints
#      are kept non-overlapping, so both the new output and the counterfactual clipped output stay
#      readable.
#    - Legend glyphs are always fully inside the interior strip `1..size-5`, which prevents them
#      from overwriting border controls and guarantees `_has_source_glyph(...)` can detect them.
#    - Corner border cells are avoided by construction because decoded rows/cols are restricted to
#      `1..size-2`.
#
# e) Visual observability
#    - Human solvers can see that marker colors live only on the border while output glyphs appear
#      inside the canvas, so row and column must be cross-read from different edges.
#    - Each sample also includes at least one decoded pair near the lower and/or right boundary that
#      fails to produce any glyph. That visible absence is the clue that clipping is forbidden.
#
# f) Visual cleanliness
#    - The border contains only isolated control pixels; the interior contains only optional legend
#      samples plus the blank background. There is no multi-layer pileup in the input.
#    - Real output glyphs never overlap each other, and skipped overflow placements are represented
#      only by border controls, which keeps the result sparse and easy to parse.
#
# g) Color inference
#    - The glyph/color dictionary is unchanged from the source task. The only new inference burden is
#      structural: first decode coordinates from edge pairs, then keep only placements whose 4x4 box
#      fits completely inside the grid.
#
# New implicit assumptions introduced by main()
#    - For any marker color, each of the four sides contributes at most one coordinate cell.
#    - A decoded placement exists only when both halves of the chosen channel are present.
#    - Any pre-drawn glyph in the input exactly matches one decoded full-fit placement and serves as
#      legend metadata rather than a scene object.
#    - generate_input() enforces all of these conditions.

GLYPH_SIZE = 4
MARKER_COLORS = (2, 3, 5, 8)
GLYPH_COLORS = (1, 4, 6, 7)
ALLOWED_COLORS = {0, *MARKER_COLORS, *GLYPH_COLORS}
CHANNELS = ("tl", "br")
MAX_RETRIES = 400

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


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + GLYPH_SIZE > h or c0 + GLYPH_SIZE > w:
        return False
    patch = g[r0 : r0 + GLYPH_SIZE, c0 : c0 + GLYPH_SIZE]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _fits_glyph(shape: tuple[int, int], r0: int, c0: int) -> bool:
    h, w = shape
    return 0 <= r0 and 0 <= c0 and r0 + GLYPH_SIZE <= h and c0 + GLYPH_SIZE <= w


def _paint_full(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    out[r0 : r0 + GLYPH_SIZE, c0 : c0 + GLYPH_SIZE][mask] = color


def _paint_clipped(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    h, w = out.shape
    for dr in range(GLYPH_SIZE):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(GLYPH_SIZE):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                out[rr, cc] = color


def _draw_full_glyph(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    grid[r0 : r0 + GLYPH_SIZE, c0 : c0 + GLYPH_SIZE][mask] = color


def _edge_cells(channel: str, r0: int, c0: int, h: int, w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if channel == "tl":
        return (0, c0), (r0, 0)
    if channel == "br":
        return (h - 1, c0), (r0, w - 1)
    raise ValueError(channel)


def _decoded_positions(g: np.ndarray, marker_color: int) -> list[tuple[str, int, int]]:
    h, w = g.shape
    top = [int(c) for c in np.flatnonzero(g[0, :] == marker_color)]
    bottom = [int(c) for c in np.flatnonzero(g[h - 1, :] == marker_color)]
    left = [int(r) for r in np.flatnonzero(g[:, 0] == marker_color)]
    right = [int(r) for r in np.flatnonzero(g[:, w - 1] == marker_color)]

    assert len(top) <= 1 and len(bottom) <= 1 and len(left) <= 1 and len(right) <= 1

    out: list[tuple[str, int, int]] = []
    if top or left:
        assert len(top) == 1 and len(left) == 1
        out.append(("tl", left[0], top[0]))
    if bottom or right:
        assert len(bottom) == 1 and len(right) == 1
        out.append(("br", right[0], bottom[0]))
    return out


def _decode_records(grid: np.ndarray) -> list[tuple[int, int, np.ndarray, str, int, int, bool, bool]]:
    g = np.asarray(grid, dtype=int)
    records: list[tuple[int, int, np.ndarray, str, int, int, bool, bool]] = []
    for marker_color, (glyph_color, mask) in GLYPHS.items():
        for channel, r0, c0 in _decoded_positions(g, marker_color):
            is_legend = _has_source_glyph(g, r0, c0, glyph_color, mask)
            fits = _fits_glyph(g.shape, r0, c0)
            records.append((marker_color, glyph_color, mask, channel, r0, c0, is_legend, fits))
    return records


def _behavior_stats(grid: np.ndarray) -> dict[str, object]:
    records = _decode_records(grid)
    legend_pairs = 0
    real_pairs = 0
    fitting_real = 0
    skipped_overflow = 0
    real_channels: set[str] = set()
    for _marker_color, _glyph_color, _mask, channel, _r0, _c0, is_legend, fits in records:
        if is_legend:
            legend_pairs += 1
            continue
        real_pairs += 1
        real_channels.add(channel)
        if fits:
            fitting_real += 1
        else:
            skipped_overflow += 1
    return {
        "decoded_pairs": len(records),
        "legend_pairs": legend_pairs,
        "real_pairs": real_pairs,
        "fitting_real": fitting_real,
        "skipped_overflow": skipped_overflow,
        "real_channels": real_channels,
    }


def _render(grid: np.ndarray, *, allow_clipping: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))).issubset(ALLOWED_COLORS)

    marker_mask = np.isin(g, MARKER_COLORS)
    if np.any(marker_mask):
        h, w = g.shape
        rr, cc = np.argwhere(marker_mask).T
        assert np.all((rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1))

    out = np.zeros_like(g)
    records = _decode_records(g)
    assert records
    for _marker_color, glyph_color, mask, _channel, r0, c0, is_legend, fits in records:
        if is_legend:
            continue
        if allow_clipping:
            _paint_clipped(out, r0, c0, glyph_color, mask)
        elif fits:
            _paint_full(out, r0, c0, glyph_color, mask)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode same-color edge controls into fixed glyphs, but discard decoded placements that would need clipping.

    Input structure:
    - Black background with marker colors 2, 3, 5, and/or 8 appearing only on the outer border.
    - A top+left pair or a bottom+right pair of the same marker color encodes one glyph top-left coordinate.
    - Some inputs also include one pre-drawn 4x4 sample glyph in the output color; its own edge pair acts as a legend.

    Rule:
    - Reconstruct each placement from its edge pair, then map the marker color to the same fixed glyph alphabet as the
      source task: 2->yellow frame, 3->blue ring, 5->pink blocks, 8->orange X.
    - Remove any legend sample and skip its matching decoded pair.
    - Paint a decoded glyph only when the full 4x4 footprint stays inside the grid; if the decoded top-left would force
      clipping on the bottom or right edge, ignore that pair completely.

    Key insight:
    - The border pixels encode row and column separately, so decoding the coordinate is only the first step; the second
      step is to filter out edge-near decodings whose glyph would overflow.
    """
    return _render(grid, allow_clipping=False)


def _bbox_overlaps(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> bool:
    r10, c10, r11, c11 = b1
    r20, c20, r21, c21 = b2
    return not (r11 < r20 or r21 < r10 or c11 < c20 or c21 < c10)


def _visible_bbox(shape: tuple[int, int], r0: int, c0: int) -> tuple[int, int, int, int] | None:
    h, w = shape
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(h - 1, r0 + GLYPH_SIZE - 1)
    cc1 = min(w - 1, c0 + GLYPH_SIZE - 1)
    if rr0 > rr1 or cc0 > cc1:
        return None
    return (rr0, cc0, rr1, cc1)


def _candidate_positions(
    size: int,
    *,
    full_fit: bool,
    keep_off_border: bool,
    rng: random.Random,
) -> list[tuple[int, int]]:
    if keep_off_border:
        rows = list(range(1, size - GLYPH_SIZE))
        cols = list(range(1, size - GLYPH_SIZE))
    else:
        rows = list(range(1, size - 1))
        cols = list(range(1, size - 1))

    fit_limit = size - GLYPH_SIZE
    positions = [(r, c) for r in rows for c in cols if _fits_glyph((size, size), r, c) == full_fit]

    if full_fit:
        frontier = [(r, c) for r, c in positions if r == fit_limit or c == fit_limit]
        frontier_set = set(frontier)
        interior = [(r, c) for r, c in positions if (r, c) not in frontier_set]
        rng.shuffle(frontier)
        rng.shuffle(interior)
        return frontier + interior if rng.random() < 0.5 else interior + frontier

    row_over = [(r, c) for r, c in positions if r > fit_limit and c <= fit_limit]
    col_over = [(r, c) for r, c in positions if c > fit_limit and r <= fit_limit]
    both_over = [(r, c) for r, c in positions if r > fit_limit and c > fit_limit]
    rng.shuffle(row_over)
    rng.shuffle(col_over)
    rng.shuffle(both_over)
    blocks = [row_over, col_over, both_over]
    rng.shuffle(blocks)
    ordered: list[tuple[int, int]] = []
    for block in blocks:
        ordered.extend(block)
    return ordered


def _choose_real_specs(size: int, rng: random.Random) -> list[tuple[int, str, bool]]:
    colors = rng.sample(sorted(GLYPHS), rng.randint(2, 4))
    first_channel = rng.choice(CHANNELS)
    second_channel = "br" if first_channel == "tl" else "tl"

    specs: list[tuple[int, str, bool]] = [
        (colors[0], first_channel, True),
        (colors[1], second_channel, False),
    ]

    slots = [(color, channel) for color in colors for channel in CHANNELS]
    remaining = [slot for slot in slots if slot not in {(colors[0], first_channel), (colors[1], second_channel)}]
    rng.shuffle(remaining)

    max_real = min(4 if size == 10 else 6, len(slots))
    target = rng.randint(len(specs), max_real)
    for color, channel in remaining[: target - len(specs)]:
        specs.append((color, channel, rng.random() < 0.65))

    rng.shuffle(specs)
    return specs


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _attempt in range(MAX_RETRIES):
        size = rng.choice((10, 20))
        real_specs = _choose_real_specs(size, rng)

        edge_occ: set[tuple[int, int]] = set()
        visible_boxes: list[tuple[int, int, int, int]] = []
        placements: list[tuple[int, str, int, int, bool]] = []
        used_channels: dict[int, set[str]] = {color: set() for color in GLYPHS}

        ok = True
        for color, channel, full_fit in real_specs:
            placed = False
            for r0, c0 in _candidate_positions(size, full_fit=full_fit, keep_off_border=False, rng=rng):
                ctrl_cells = _edge_cells(channel, r0, c0, size, size)
                if any(cell in edge_occ for cell in ctrl_cells):
                    continue
                vis_box = _visible_bbox((size, size), r0, c0)
                assert vis_box is not None
                if any(_bbox_overlaps(vis_box, other) for other in visible_boxes):
                    continue

                edge_occ.update(ctrl_cells)
                visible_boxes.append(vis_box)
                placements.append((color, channel, r0, c0, False))
                used_channels[color].add(channel)
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        real_colors = sorted({color for color, _channel, _r0, _c0, is_legend in placements if not is_legend})
        legend_options = [
            (color, channel)
            for color in real_colors
            for channel in CHANNELS
            if channel not in used_channels[color]
        ]
        if legend_options and rng.random() < 0.7:
            rng.shuffle(legend_options)
            legend_added = False
            for color, channel in legend_options:
                for r0, c0 in _candidate_positions(size, full_fit=True, keep_off_border=True, rng=rng):
                    ctrl_cells = _edge_cells(channel, r0, c0, size, size)
                    if any(cell in edge_occ for cell in ctrl_cells):
                        continue
                    legend_box = (r0, c0, r0 + GLYPH_SIZE - 1, c0 + GLYPH_SIZE - 1)
                    if any(_bbox_overlaps(legend_box, other) for other in visible_boxes):
                        continue

                    edge_occ.update(ctrl_cells)
                    visible_boxes.append(legend_box)
                    placements.append((color, channel, r0, c0, True))
                    used_channels[color].add(channel)
                    legend_added = True
                    break
                if legend_added:
                    break

        inp = np.zeros((size, size), dtype=int)
        for color, channel, r0, c0, is_legend in placements:
            glyph_color, mask = GLYPHS[color]
            if is_legend:
                _draw_full_glyph(inp, r0, c0, glyph_color, mask)
            for rr, cc in _edge_cells(channel, r0, c0, size, size):
                inp[rr, cc] = color

        stats = _behavior_stats(inp)
        if stats["fitting_real"] < 1 or stats["skipped_overflow"] < 1:
            continue
        if len(stats["real_channels"]) < 2:
            continue

        new_out = main(inp)
        old_out = _render(inp, allow_clipping=True)
        if np.array_equal(new_out, old_out):
            continue
        if not np.any(new_out != 0):
            continue

        assert stats["real_pairs"] == sum(1 for _color, _channel, _r0, _c0, is_legend in placements if not is_legend)
        assert np.array_equal(new_out, _render(inp, allow_clipping=False))
        return inp

    raise RuntimeError("failed to generate a non-degenerate combo sample")
