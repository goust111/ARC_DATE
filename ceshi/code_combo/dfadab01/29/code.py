# Combo variant of dfadab01 (index 29)
# combo_spec: D6/3+D7/0
# Name: 禁止裁剪+随机散布

from __future__ import annotations

import random

import numpy as np


# Design analysis for the combo variant
# a) Concretization and behavior trace
#    - The fixed alphabet is unchanged: marker colors {2, 3, 5, 8} still point to the
#      same 4x4 output masks/colors, and an optional legend is still a painted sample
#      glyph with its label marker at (+4, +4).
#    - The two changes interact through position. Example on a 10x10 grid:
#      markers at (1, 1), (2, 6), (6, 0), (8, 5), scattered without row/col spacing.
#      Old task: every real marker expands, so (8, 5) contributes a cropped fragment.
#      New combo: each marker is checked independently; (1, 1), (2, 6), (6, 0) survive
#      only if their full 4x4 boxes fit, while (8, 5) is ignored because 8 + 4 > 10.
#      Random scattering matters because edge-near markers can now appear anywhere,
#      not only on the last row/column of a regular lattice.
# b) Complexity calibration
#    - The rule change stays single-branch: color chooses glyph, then position decides
#      whether that glyph is active. There is no extra remapping or shape mutation.
#    - The generator always includes both at least one fully fitting marker and at
#      least one overflowing marker, so the "skip instead of crop" behavior is visible.
#    - Scatter is constrained by 4x4 non-overlap boxes, keeping scenes readable while
#      still breaking the original equal-spacing cue.
# c) Implicit-assumption audit for main()
#    - Original logic assumed every non-legend marker should paint something via
#      clipping. That assumption is invalid now; painting must be gated by _fits_glyph().
#    - Legend detection via _has_source_glyph() still works because the legend sample
#      is always fully inside the grid and isolated from real markers by construction.
#    - Iterating markers color-by-color remains safe because surviving glyphs never
#      overlap, so paint order does not affect the output.
# d) Placement legality
#    - The generator treats every real marker as owning a 4x4 intended glyph box,
#      even for overflowing markers. Those boxes are sampled without overlap, so the
#      input is unambiguous and the clipped baseline would also stay non-overlapping.
#    - The optional legend uses a disjoint 5x5 reservation box (4x4 sample + label),
#      preventing collisions with marker pixels or intended glyph regions.
# e) Visual observability
#    - Each accepted sample has real scattered markers in several unrelated rows/cols,
#      plus at least one edge-near marker that disappears entirely in the output.
#    - The generator rejects any case where the new output matches the old clipped
#      output, so the boundary-rule change is guaranteed to manifest.
# f) Visual cleanliness
#    - Non-overlapping 4x4 boxes keep glyphs separated even though marker pixels are
#      irregularly placed. The legend is isolated, so it reads as a removable key.
#    - Random scatter is limited to modest marker counts, avoiding dense clutter.
# g) Color inference
#    - Color semantics are unchanged from the source puzzle. The only new inference is
#      that marker location relative to the bottom/right border can deactivate it.

GLYPH_SIZE = 4
LEGEND_SIZE = GLYPH_SIZE + 1
MARKER_COLORS = (2, 3, 5, 8)
GLYPH_COLORS = (1, 4, 6, 7)
ALLOWED_COLORS = {0, *MARKER_COLORS, *GLYPH_COLORS}
MAX_RETRIES = 500

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
    patch = out[r0 : r0 + GLYPH_SIZE, c0 : c0 + GLYPH_SIZE]
    patch[mask] = color


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


def _render(grid: np.ndarray, *, allow_clipping: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))).issubset(ALLOWED_COLORS)

    out = np.zeros_like(g)
    seen_marker = False
    for marker_color, (glyph_color, mask) in GLYPHS.items():
        coords = np.argwhere(g == marker_color)
        if coords.size:
            seen_marker = True
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if _has_source_glyph(g, rr - GLYPH_SIZE, cc - GLYPH_SIZE, glyph_color, mask):
                continue
            if allow_clipping:
                _paint_clipped(out, rr, cc, glyph_color, mask)
            elif _fits_glyph(g.shape, rr, cc):
                _paint_full(out, rr, cc, glyph_color, mask)

    assert seen_marker
    return out


def _iter_real_markers(g: np.ndarray) -> list[tuple[int, int, int]]:
    real: list[tuple[int, int, int]] = []
    for marker_color, (glyph_color, mask) in GLYPHS.items():
        for r, c in np.argwhere(g == marker_color):
            rr = int(r)
            cc = int(c)
            if _has_source_glyph(g, rr - GLYPH_SIZE, cc - GLYPH_SIZE, glyph_color, mask):
                continue
            real.append((rr, cc, marker_color))
    return real


def _box_for_marker(r0: int, c0: int) -> tuple[int, int, int, int]:
    return (r0, c0, r0 + GLYPH_SIZE, c0 + GLYPH_SIZE)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _axis_has_uniform_spacing(values: list[int]) -> bool:
    if len(values) < 3:
        return False
    diffs = [b - a for a, b in zip(values, values[1:])]
    return len(set(diffs)) == 1


def _looks_regular_scatter(positions: list[tuple[int, int]]) -> bool:
    rows = sorted({r for r, _ in positions})
    cols = sorted({c for _, c in positions})
    return _axis_has_uniform_spacing(rows) and _axis_has_uniform_spacing(cols)


def _try_sample_positions(
    shape: tuple[int, int],
    fit_target: int,
    overflow_target: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    h, w = shape
    fit_candidates = [(r, c) for r in range(h) for c in range(w) if _fits_glyph(shape, r, c)]
    overflow_candidates = [(r, c) for r in range(h) for c in range(w) if not _fits_glyph(shape, r, c)]
    rng.shuffle(fit_candidates)
    rng.shuffle(overflow_candidates)

    chosen: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []

    for pool, target in ((fit_candidates, fit_target), (overflow_candidates, overflow_target)):
        added = 0
        for r, c in pool:
            box = _box_for_marker(r, c)
            if any(_boxes_overlap(box, other) for other in boxes):
                continue
            chosen.append((r, c))
            boxes.append(box)
            added += 1
            if added == target:
                break
        if added != target:
            return None

    return chosen


def _choose_scattered_positions(shape: tuple[int, int], rng: random.Random) -> list[tuple[int, int]]:
    h, w = shape
    fit_range = (2, 3) if h == 10 and w == 10 else (4, 7)
    overflow_range = (1, 2) if h == 10 and w == 10 else (2, 4)

    for _ in range(200):
        fit_target = rng.randint(*fit_range)
        overflow_target = rng.randint(*overflow_range)
        positions = _try_sample_positions(shape, fit_target, overflow_target, rng)
        if positions is None:
            continue
        rows = {r for r, _ in positions}
        cols = {c for _, c in positions}
        if len(rows) < 3 or len(cols) < 3:
            continue
        if _looks_regular_scatter(positions):
            continue
        return positions

    raise ValueError("could not choose irregular scattered marker positions")


def _place_legend(
    grid: np.ndarray,
    marker_boxes: list[tuple[int, int, int, int]],
    used_colors: tuple[int, ...],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for marker_color in used_colors:
        for r0 in range(h - GLYPH_SIZE):
            for c0 in range(w - GLYPH_SIZE):
                legend_box = (r0, c0, r0 + LEGEND_SIZE, c0 + LEGEND_SIZE)
                if any(_boxes_overlap(legend_box, box) for box in marker_boxes):
                    continue
                if np.any(grid[r0 : r0 + LEGEND_SIZE, c0 : c0 + LEGEND_SIZE] != 0):
                    continue
                candidates.append((marker_color, r0, c0))

    if not candidates:
        return False

    marker_color, r0, c0 = rng.choice(candidates)
    glyph_color, mask = GLYPHS[marker_color]
    patch = grid[r0 : r0 + GLYPH_SIZE, c0 : c0 + GLYPH_SIZE]
    patch[mask] = glyph_color
    grid[r0 + GLYPH_SIZE, c0 + GLYPH_SIZE] = marker_color
    return True


def _sample_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    g = np.asarray(grid, dtype=int)
    real_markers = _iter_real_markers(g)
    skipped = sum(1 for r, c, _ in real_markers if not _fits_glyph(g.shape, r, c))
    old_out = _render(g, allow_clipping=True)
    new_out = _render(g, allow_clipping=False)
    return {
        "real_marker_count": len(real_markers),
        "skipped_marker_count": skipped,
        "diff_pixels": int(np.count_nonzero(old_out != new_out)),
        "regular_scatter": _looks_regular_scatter([(r, c) for r, c, _ in real_markers]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand scattered control pixels into fixed 4x4 glyphs, but only when the whole glyph fits.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8, now scattered irregularly instead of sitting on a regular lattice.
    - Some inputs also contain one legend entry: a 4x4 sample glyph in its output color plus a matching label marker one glyph-step down-right.

    Rule:
    - Use the same fixed glyph dictionary as the source task: marker color chooses one 4x4 output mask/color pair.
    - Remove the legend sample and ignore its label marker.
    - For every real marker, paint the glyph only if the entire 4x4 footprint stays inside the grid; markers whose glyph would cross the border contribute nothing at all.

    Key insight:
    - The mapping is still determined by marker color, but scattered placement means you must inspect each marker separately and filter out the edge-near ones instead of expecting cropped fragments.
    """
    return _render(grid, allow_clipping=False)


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(MAX_RETRIES):
        h = rng.choice((10, 20))
        w = rng.choice((10, 20))
        shape = (h, w)

        try:
            positions = _choose_scattered_positions(shape, rng)
        except ValueError:
            continue

        grid = np.zeros(shape, dtype=int)
        marker_boxes = [_box_for_marker(r, c) for r, c in positions]

        used_count = min(rng.randint(1, len(MARKER_COLORS)), len(positions))
        used_colors = tuple(sorted(rng.sample(MARKER_COLORS, used_count)))

        color_bag = list(used_colors)
        while len(color_bag) < len(positions):
            color_bag.append(rng.choice(used_colors))
        rng.shuffle(color_bag)

        for (r, c), marker_color in zip(positions, color_bag):
            grid[r, c] = marker_color

        if rng.random() < 0.6:
            _place_legend(grid, marker_boxes, used_colors, rng)

        metrics = _sample_metrics(grid)
        if int(metrics["skipped_marker_count"]) < 1:
            continue
        if int(metrics["diff_pixels"]) < 1:
            continue
        if bool(metrics["regular_scatter"]):
            continue
        if int(metrics["real_marker_count"]) - int(metrics["skipped_marker_count"]) < 1:
            continue

        out = main(grid)
        if not np.any(out != 0):
            continue

        # New assumptions after the combo change:
        # - generate_input() guarantees at least one surviving and one skipped real marker.
        # - All intended 4x4 marker boxes are pairwise disjoint, so _render() remains
        #   order-independent even though markers are no longer arranged on a lattice.
        # - Any legend label always has a fully visible 4x4 sample up-left, matching
        #   the _has_source_glyph() predicate used by main().
        assert np.array_equal(out, _render(grid, allow_clipping=False))
        return grid

    raise RuntimeError("failed to generate a non-degenerate combo sample")
