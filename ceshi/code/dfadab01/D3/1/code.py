# Variant of dfadab01
# Dimension: D3 — 标记来源对象 (src)
# Change: 散布在网格中的单像素标记，使用字形字典中的颜色（2、3、5、8），标记位置即字形放置的左上角... → 网格边缘控制像素

from __future__ import annotations

import random

import numpy as np


# Design analysis for the D3 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - The original task used one interior marker pixel per glyph placement. Here that source object is replaced
#      by two same-color control pixels on the border:
#        * channel "tl": (0, c) on the top edge and (r, 0) on the left edge encode glyph top-left (r, c)
#        * channel "br": (H-1, c) on the bottom edge and (r, W-1) on the right edge encode glyph top-left (r, c)
#    - The glyph dictionary itself is unchanged: marker color still chooses one fixed 4x4 boolean mask and one
#      output color. The output is still just the painted glyphs on black, clipped by the grid edge.
#    - Legend behavior stays structurally close to the original task: an optional sample glyph may appear already
#      drawn in the input, and the matching control pair points at its own top-left. Solver must treat that pair
#      as legend metadata and skip painting it into the output.
#    - Small trace:
#        original: input has red pixel at (6, 12) -> solver paints yellow frame with top-left (6, 12)
#        variant: input has red at top (0, 12) and red at left (6, 0) -> solver crosses the edge coordinates and
#        recovers the same top-left (6, 12), then paints the same yellow frame.
#        difference: the placement information is no longer local; it must be reconstructed from border geometry.
#
# b) Complexity calibration
#    - Pure same-color border pairs would become ambiguous if one color had many top controls and many left controls,
#      because the Cartesian product would over-generate glyphs. To keep the variant readable and deterministic,
#      each color gets at most one control on each side, which yields at most one "tl" placement and one "br"
#      placement per color. This still permits repeated support for the same color, but caps it at two placements.
#    - Examples therefore usually contain 2-6 real glyph placements across 2-4 colors, enough to make the border
#      mechanism visually obvious without introducing pairing ambiguity.
#
# c) Hidden-assumption audit against the original solver
#    - The original loop `coords = np.argwhere(g == marker_color)` assumed every marker pixel independently encoded
#      a placement. That is no longer valid: a single edge pixel is incomplete and must be paired with a same-color
#      pixel on the orthogonal edge.
#    - The original legend test `_has_source_glyph(g, r-4, c-4, ...)` relied on the label pixel being exactly one
#      cell below-right of the legend bbox. That positional convention disappears. The new safe test is:
#      "after decoding a border pair to (r, c), does a full matching glyph already exist at (r, c)?"
#    - Unconditional painting from every marker pixel would now overproduce glyphs, so decoding is split into side
#      collection -> channel reconstruction -> optional legend skip -> paint.
#
# d) Placement legality
#    - Generator guarantees that 4x4 glyph bboxes of real placements do not overlap each other.
#    - Legend glyph, when present, also gets a non-overlapping full 4x4 area so the sample stays legible.
#    - Border control cells are unique as input pixels. The generator never reuses a border cell for two different
#      controls, and it avoids corner controls to keep side membership visually clean.
#    - Solver assumptions are matched by construction: for each marker color there is at most one top, bottom,
#      left, and right control pixel, so decoding "tl" and "br" is unambiguous.
#
# e) Visual observability
#    - Across examples, marker colors are absent from the interior and appear only on the border. Human solvers can
#      see that row and column must be read from two different edges because the output glyphs sit away from the
#      controls instead of on top of them.
#    - Both channels are sampled by the generator, so the transformation is not mistaken for "always read from top
#      and left only".
#
# f) Visual cleanliness
#    - Only one abstraction layer occupies the border: isolated control pixels. The interior contains either empty
#      space or one optional legend glyph. This avoids multi-object overlap and keeps the encoded coordinates easy
#      to scan.
#
# g) Color inference
#    - The color mapping itself is unchanged and remains supported by repeated glyph/color usage plus occasional
#      legend entries. No ordering or branching depends on invisible properties such as numeric color ids.
#
# New implicit assumptions introduced by main()
#    - A decoded placement exists only when both halves of a channel are present.
#    - For a given color, each side contributes at most one coordinate.
#    - Any pre-drawn glyph in the input is a legend glyph and exactly matches one decoded placement.
#    - generate_input() enforces all three assumptions.


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


def _draw_full_glyph(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    for dr in range(4):
        for dc in range(4):
            if bool(mask[dr, dc]):
                grid[r0 + dr, c0 + dc] = color


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


def _bbox_overlaps(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> bool:
    r10, c10, r11, c11 = b1
    r20, c20, r21, c21 = b2
    return not (r11 < r20 or r21 < r10 or c11 < c20 or c21 < c10)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode same-color border controls into fixed 4x4 glyph placements.

    Input structure:
    - Black background with marker colors 2, 3, 5, and/or 8 appearing only on the outer border.
    - A top+left pair or bottom+right pair of the same marker color encodes one glyph top-left coordinate.
    - Some inputs also include one pre-drawn 4x4 sample glyph in the output color; its own border pair serves as a legend.

    Rule:
    - Read each border pair as one placement: top+left gives (row from left edge, col from top edge), bottom+right gives
      (row from right edge, col from bottom edge).
    - Replace that decoded placement by the corresponding fixed 4x4 glyph: 2->yellow frame, 3->blue ring, 5->pink blocks,
      8->orange X.
    - Clip glyphs at the grid boundary, and skip any decoded placement whose glyph is already fully drawn in the input
      as a legend sample.

    Key insight:
    - A single colored border pixel is not a complete object anymore; the row and column must be cross-located from a
      same-color pair on perpendicular edges.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, *GLYPHS.keys(), 1, 4, 6, 7})

    marker_mask = np.isin(g, list(GLYPHS.keys()))
    if np.any(marker_mask):
        h, w = g.shape
        rr, cc = np.argwhere(marker_mask).T
        assert np.all((rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1))

    out = np.zeros_like(g)
    seen_marker = False
    for marker_color, (glyph_color, mask) in GLYPHS.items():
        placements = _decoded_positions(g, marker_color)
        if placements:
            seen_marker = True
        for _channel, r0, c0 in placements:
            if _has_source_glyph(g, int(r0), int(c0), glyph_color, mask):
                continue
            _paint_clipped(out, int(r0), int(c0), glyph_color, mask)

    assert seen_marker
    return out


def generate_input() -> np.ndarray:
    def choose_real_specs() -> list[tuple[int, str]]:
        colors = random.sample(sorted(GLYPHS), random.randint(2, 4))
        specs: list[tuple[int, str]] = []

        # Seed both channels when possible so the border mechanism is visibly two-sided.
        if len(colors) >= 2:
            specs.append((colors[0], "tl"))
            specs.append((colors[1], "br"))
            rest = colors[2:]
        else:
            specs.append((colors[0], random.choice(["tl", "br"])))
            rest = []

        for color in rest:
            specs.append((color, random.choice(["tl", "br"])))

        max_real = min(6 if size == 20 else 4, 2 * len(colors))
        target = random.randint(len(specs), max_real)
        extras = [(color, channel) for color in colors for channel in ("tl", "br") if (color, channel) not in specs]
        random.shuffle(extras)
        while len(specs) < target and extras:
            specs.append(extras.pop())
        random.shuffle(specs)
        return specs

    def candidate_positions(full_fit: bool) -> list[tuple[int, int]]:
        if full_fit:
            # Keep the legend fully away from the border so it cannot overwrite edge controls.
            rows = list(range(1, size - 4))
            cols = list(range(1, size - 4))
        else:
            rows = list(range(1, size - 1))
            cols = list(range(1, size - 1))
        clip_bias = [(r, c) for r in rows for c in cols if (r >= size - 3) or (c >= size - 3)]
        clip_set = set(clip_bias)
        safe = [(r, c) for r in rows for c in cols if (r, c) not in clip_set]
        random.shuffle(clip_bias)
        random.shuffle(safe)
        if full_fit or not clip_bias or random.random() < 0.45:
            return clip_bias + safe
        return safe + clip_bias

    for _attempt in range(400):
        size = random.choice([10, 20])
        real_specs = choose_real_specs()
        bboxes: list[tuple[int, int, int, int]] = []
        edge_occ: set[tuple[int, int]] = set()
        placements: list[tuple[int, str, int, int, bool]] = []
        used_channels: dict[int, set[str]] = {color: set() for color in GLYPHS}

        ok = True
        for color, channel in real_specs:
            placed = False
            for r0, c0 in candidate_positions(full_fit=False):
                bbox = (r0, c0, r0 + 3, c0 + 3)
                if any(_bbox_overlaps(bbox, other) for other in bboxes):
                    continue
                ctrl_cells = _edge_cells(channel, r0, c0, size, size)
                if any(cell in edge_occ for cell in ctrl_cells):
                    continue
                bboxes.append(bbox)
                edge_occ.update(ctrl_cells)
                placements.append((color, channel, r0, c0, False))
                used_channels[color].add(channel)
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        legend_options = [(color, channel) for color, channels in used_channels.items() for channel in ("tl", "br") if channel not in channels]
        if legend_options and random.random() < 0.7:
            random.shuffle(legend_options)
            legend_added = False
            for color, channel in legend_options:
                for r0, c0 in candidate_positions(full_fit=True):
                    bbox = (r0, c0, r0 + 3, c0 + 3)
                    if any(_bbox_overlaps(bbox, other) for other in bboxes):
                        continue
                    ctrl_cells = _edge_cells(channel, r0, c0, size, size)
                    if any(cell in edge_occ for cell in ctrl_cells):
                        continue
                    bboxes.append(bbox)
                    edge_occ.update(ctrl_cells)
                    placements.append((color, channel, r0, c0, True))
                    used_channels[color].add(channel)
                    legend_added = True
                    break
                if legend_added:
                    break

        inp = np.zeros((size, size), dtype=int)
        out = np.zeros((size, size), dtype=int)

        for color, channel, r0, c0, is_legend in placements:
            glyph_color, mask = GLYPHS[color]
            if is_legend:
                _draw_full_glyph(inp, r0, c0, glyph_color, mask)
            else:
                _paint_clipped(out, r0, c0, glyph_color, mask)
            for rr, cc in _edge_cells(channel, r0, c0, size, size):
                inp[rr, cc] = color

        if np.array_equal(main(inp), out):
            return inp

    raise RuntimeError("failed to generate a valid D3 variant input")
