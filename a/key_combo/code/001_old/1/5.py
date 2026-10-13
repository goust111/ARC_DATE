from __future__ import annotations

import random

import numpy as np


def _mirror_coord(r: int, c: int, h: int, w: int, divider_axis: str, divider_index: int) -> tuple[int, int]:
    if divider_axis == "row":
        return 2 * divider_index - r, c
    return r, 2 * divider_index - c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Output the unmarked half of a divided mirror pair after repairing it from the marked complete half.

    Input structure:
    - A full grey divider row or column splits the grid into two equal halves.
    - One half contains a complete colored pattern plus exactly two marker pixels of a color not used elsewhere.
    - The other half contains the mirror image of the pattern with some colored cells erased to black.

    Rule:
    - The half containing the two markers is the reference half; the opposite half is the output base.
    - For each black cell in the output half, copy the non-black, non-marker color at its mirror position in the reference half.
    - The divider and marker pixels are omitted from the output.

    Key insight:
    The markers identify the complete source half, but the answer is the opposite damaged half repaired by mirror lookup.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    grey = 5
    divider_rows = [r for r in range(h) if np.all(g[r, :] == grey)]
    divider_cols = [c for c in range(w) if np.all(g[:, c] == grey)]
    if len(divider_rows) == 1:
        axis = "row"
        d = divider_rows[0]
        halves = [(0, d), (d + 1, h)]
    elif len(divider_cols) == 1:
        axis = "col"
        d = divider_cols[0]
        halves = [(0, d), (d + 1, w)]
    else:
        raise ValueError("expected one full grey divider")

    marker_color = None
    marker_positions: list[tuple[int, int]] = []
    for color in [int(v) for v in np.unique(g) if int(v) not in (0, grey)]:
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == color)]
        if len(pts) != 2:
            continue
        if axis == "row":
            same_half = all(r < d for r, _c in pts) or all(r > d for r, _c in pts)
        else:
            same_half = all(c < d for _r, c in pts) or all(c > d for _r, c in pts)
        mirrors_are_background = same_half and all(
            int(g[_mirror_coord(r, c, h, w, axis, d)]) == 0 for r, c in pts
        )
        if mirrors_are_background:
            marker_color = color
            marker_positions = pts
            break
    if marker_color is None:
        raise ValueError("no marker pair found")

    if axis == "row":
        source_top = marker_positions[0][0] < d
        source_slice = (slice(0, d), slice(None)) if source_top else (slice(d + 1, h), slice(None))
        target_slice = (slice(d + 1, h), slice(None)) if source_top else (slice(0, d), slice(None))
        row_offset = d + 1 if not source_top else 0
    else:
        source_left = marker_positions[0][1] < d
        source_slice = (slice(None), slice(0, d)) if source_left else (slice(None), slice(d + 1, w))
        target_slice = (slice(None), slice(d + 1, w)) if source_left else (slice(None), slice(0, d))
        row_offset = 0

    target = g[target_slice].copy()
    out = target.copy()
    for rr in range(out.shape[0]):
        for cc in range(out.shape[1]):
            if int(out[rr, cc]) != 0:
                continue
            if axis == "row":
                global_r = (d + 1 + rr) if marker_positions[0][0] < d else rr
                global_c = cc
            else:
                global_r = rr
                global_c = (d + 1 + cc) if marker_positions[0][1] < d else cc
            mr, mc = _mirror_coord(global_r, global_c, h, w, axis, d)
            value = int(g[mr, mc])
            if value not in (0, marker_color, grey):
                out[rr, cc] = value
    return out


def generate_input() -> np.ndarray:
    for _attempt in range(200):
        axis = random.choice(["row", "col"])
        if axis == "row":
            half_h = random.randint(5, 8)
            half_w = random.randint(9, 12)
            h, w = half_h * 2 + 1, half_w
            d = half_h
            source_first = random.choice([True, False])
            g = np.zeros((h, w), dtype=int)
            g[d, :] = 5
            shape = (half_h, half_w)
        else:
            half_h = random.randint(5, 8)
            half_w = random.randint(9, 12)
            h, w = half_h, half_w * 2 + 1
            d = half_w
            source_first = random.choice([True, False])
            g = np.zeros((h, w), dtype=int)
            g[:, d] = 5
            shape = (half_h, half_w)

        pattern = np.zeros(shape, dtype=int)
        colors = random.sample([c for c in range(1, 10) if c != 5], random.randint(2, 4))
        cells = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
        fill_count = random.randint((len(cells) * 3) // 10, (len(cells) * 6) // 10)
        for r, c in random.sample(cells, fill_count):
            pattern[r, c] = random.choice(colors)
        marker_color = next(c for c in range(1, 10) if c not in colors and c != 5)
        background_cells = [(r, c) for r, c in cells if pattern[r, c] == 0]
        if len(background_cells) < 2:
            continue
        markers = random.sample(background_cells, 2)
        damaged = pattern.copy()
        nonzero = [(r, c) for r, c in cells if damaged[r, c] != 0]
        for r, c in random.sample(nonzero, min(random.randint(3, 8), len(nonzero))):
            damaged[r, c] = 0

        if axis == "row":
            top = pattern if source_first else np.flipud(damaged)
            bottom = np.flipud(damaged) if source_first else pattern
            g[:d, :] = top
            g[d + 1 :, :] = bottom
            for r, c in markers:
                rr = r if source_first else d + 1 + r
                g[rr, c] = marker_color
        else:
            left = pattern if source_first else np.fliplr(damaged)
            right = np.fliplr(damaged) if source_first else pattern
            g[:, :d] = left
            g[:, d + 1 :] = right
            for r, c in markers:
                cc = c if source_first else d + 1 + c
                g[r, cc] = marker_color
        return g
    raise RuntimeError("failed to generate puzzle input")
