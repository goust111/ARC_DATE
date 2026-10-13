from __future__ import annotations

import random
from collections import Counter

import numpy as np


ColorPair = tuple[int, int]
Segment = dict[str, int | str]


def _background_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _find_separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(row == row[0]) and int(row[0]) != bg:
            candidates.append((r, int(row[0])))
    assert len(candidates) == 1, "expected exactly one full horizontal separator row"
    return candidates[0]


def _find_palette_triplet(grid: np.ndarray, bg: int, sep_r: int) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for r in range(sep_r + 1, grid.shape[0]):
        cols = np.flatnonzero(grid[r] != bg)
        if len(cols) != 3:
            continue
        if not np.all(np.diff(cols) == 1):
            continue
        a, b, c = (int(grid[r, col]) for col in cols)
        candidates.append((a, b, c))
    assert len(candidates) == 1, "expected exactly one three-pixel palette row"
    return candidates[0]


def _extract_pairs(grid: np.ndarray, bg: int, sep_r: int) -> list[Segment]:
    top = grid[:sep_r]
    counts = Counter(map(int, top.ravel()))
    segments: list[Segment] = []
    for color, count in counts.items():
        if color == bg:
            continue
        assert count == 2, "every top-region color must appear exactly twice"
        coords = np.argwhere(top == color)
        (r1, c1), (r2, c2) = coords.tolist()
        if r1 == r2:
            c0, c1_ = sorted((int(c1), int(c2)))
            segments.append(
                {
                    "orient": "h",
                    "color": int(color),
                    "row": int(r1),
                    "c0": int(c0),
                    "c1": int(c1_),
                }
            )
        else:
            assert c1 == c2, "paired pixels must share a row or a column"
            r0, r1_ = sorted((int(r1), int(r2)))
            segments.append(
                {
                    "orient": "v",
                    "color": int(color),
                    "col": int(c1),
                    "r0": int(r0),
                    "r1": int(r1_),
                }
            )
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color pixel pair and recolor line crossings with the palette's third color.

    Input structure:
    - The background is the dominant color.
    - A single full-width horizontal separator row splits the grid into a top drawing area
      and a bottom palette area.
    - In the top area, every non-background color appears exactly twice, and each pair lies
      either on one row or on one column.
    - In the bottom area, exactly one row contains three horizontally adjacent non-background
      pixels; its first two colors identify pair colors from the top area, and its third color
      is the special crossing color.

    Rule:
    - For every same-color pair in the top area, draw the inclusive straight horizontal or
      vertical segment between the two endpoints using that pair's own color.
    - Whenever a horizontal segment and a vertical segment of different colors occupy the same
      pixel, replace that pixel with the palette's third color.
    - Leave the separator row and the entire bottom palette area unchanged.

    Key insight:
    - The top area is not a scatter of isolated markers: each repeated color defines one axis-
      aligned segment, and the lower three-pixel palette contributes the new color used only at
      cross-conflict cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    sep_r, _sep_color = _find_separator_row(g, bg)
    palette_a, palette_b, cross_color = _find_palette_triplet(g, bg, sep_r)
    segments = _extract_pairs(g, bg, sep_r)

    pair_colors = {int(seg["color"]) for seg in segments}
    assert palette_a in pair_colors and palette_b in pair_colors and palette_a != palette_b

    out = g.copy()
    horizontals: list[Segment] = []
    verticals: list[Segment] = []
    for seg in segments:
        if seg["orient"] == "h":
            r = int(seg["row"])
            c0 = int(seg["c0"])
            c1 = int(seg["c1"])
            out[r, c0 : c1 + 1] = int(seg["color"])
            horizontals.append(seg)
        else:
            c = int(seg["col"])
            r0 = int(seg["r0"])
            r1 = int(seg["r1"])
            out[r0 : r1 + 1, c] = int(seg["color"])
            verticals.append(seg)

    for hseg in horizontals:
        hr = int(hseg["row"])
        hc0 = int(hseg["c0"])
        hc1 = int(hseg["c1"])
        hcolor = int(hseg["color"])
        for vseg in verticals:
            vcolor = int(vseg["color"])
            if hcolor == vcolor:
                continue
            vc = int(vseg["col"])
            vr0 = int(vseg["r0"])
            vr1 = int(vseg["r1"])
            if hc0 <= vc <= hc1 and vr0 <= hr <= vr1:
                out[hr, vc] = cross_color

    return out


def _segment_cells(seg: Segment) -> set[tuple[int, int]]:
    if seg["orient"] == "h":
        row = int(seg["row"])
        return {(row, c) for c in range(int(seg["c0"]), int(seg["c1"]) + 1)}
    col = int(seg["col"])
    return {(r, col) for r in range(int(seg["r0"]), int(seg["r1"]) + 1)}


def _horizontal_segment(row: int, c0: int, c1: int, color: int) -> Segment:
    return {"orient": "h", "row": row, "c0": c0, "c1": c1, "color": color}


def _vertical_segment(col: int, r0: int, r1: int, color: int) -> Segment:
    return {"orient": "v", "col": col, "r0": r0, "r1": r1, "color": color}


def _place_crossing_pair(rng: random.Random, top_h: int, width: int, colors: list[int]) -> list[Segment] | None:
    if top_h < 7 or width < 8:
        return None
    for _ in range(80):
        cross_r = rng.randint(3, top_h - 4)
        cross_c = rng.randint(3, width - 4)
        h_left = rng.randint(1, cross_c - 2)
        h_right = rng.randint(cross_c + 2, width - 2)
        v_top = rng.randint(1, cross_r - 2)
        v_bottom = rng.randint(cross_r + 2, top_h - 2)
        hseg = _horizontal_segment(cross_r, h_left, h_right, colors[0])
        vseg = _vertical_segment(cross_c, v_top, v_bottom, colors[1])
        return [hseg, vseg]
    return None


def _place_noncrossing_segment(
    rng: random.Random,
    top_h: int,
    width: int,
    orient: str,
    color: int,
    occupied: set[tuple[int, int]],
    used_h_rows: set[int],
    used_v_cols: set[int],
) -> Segment | None:
    for _ in range(200):
        if orient == "h":
            rows = [r for r in range(1, top_h - 1) if r not in used_h_rows]
            if not rows:
                return None
            row = rng.choice(rows)
            c0 = rng.randint(1, width - 4)
            c1 = rng.randint(c0 + 2, width - 2)
            seg = _horizontal_segment(row, c0, c1, color)
            if _segment_cells(seg) & occupied:
                continue
            return seg

        cols = [c for c in range(1, width - 1) if c not in used_v_cols]
        if not cols:
            return None
        col = rng.choice(cols)
        r0 = rng.randint(1, top_h - 4)
        r1 = rng.randint(r0 + 2, top_h - 2)
        seg = _vertical_segment(col, r0, r1, color)
        if _segment_cells(seg) & occupied:
            continue
        return seg
    return None


def _planned_crossings(segments: list[Segment]) -> list[ColorPair]:
    horizontals = [seg for seg in segments if seg["orient"] == "h"]
    verticals = [seg for seg in segments if seg["orient"] == "v"]
    crossings: list[ColorPair] = []
    for hseg in horizontals:
        for vseg in verticals:
            if int(hseg["c0"]) <= int(vseg["col"]) <= int(hseg["c1"]) and int(vseg["r0"]) <= int(hseg["row"]) <= int(vseg["r1"]):
                crossings.append((int(hseg["color"]), int(vseg["color"])))
    return crossings


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(14, 24)
    width = rng.randint(10, 24)
    sep_r = rng.randint(7, height - 4)
    top_h = sep_r

    pair_count = rng.randint(2, 5)
    h_count = rng.randint(1, pair_count - 1)
    v_count = pair_count - h_count
    crossing_mode = bool(rng.randint(0, 1))

    total_needed_colors = pair_count + 3
    if total_needed_colors > 10:
        return None

    palette = list(range(10))
    rng.shuffle(palette)
    bg = palette[0]
    divider = palette[1]
    cross_color = palette[2]
    pair_colors = palette[3 : 3 + pair_count]

    segments: list[Segment] = []
    occupied: set[tuple[int, int]] = set()
    used_h_rows: set[int] = set()
    used_v_cols: set[int] = set()

    crossing_h_color: int | None = None
    crossing_v_color: int | None = None
    next_color = 0

    if crossing_mode:
        crossing = _place_crossing_pair(rng, top_h, width, pair_colors[:2])
        if crossing is None:
            return None
        for seg in crossing:
            segments.append(seg)
            occupied |= _segment_cells(seg)
            if seg["orient"] == "h":
                used_h_rows.add(int(seg["row"]))
                crossing_h_color = int(seg["color"])
            else:
                used_v_cols.add(int(seg["col"]))
                crossing_v_color = int(seg["color"])
        next_color = 2

    while len([seg for seg in segments if seg["orient"] == "h"]) < h_count:
        if next_color >= len(pair_colors):
            return None
        seg = _place_noncrossing_segment(
            rng,
            top_h,
            width,
            "h",
            pair_colors[next_color],
            occupied,
            used_h_rows,
            used_v_cols,
        )
        if seg is None:
            return None
        segments.append(seg)
        occupied |= _segment_cells(seg)
        used_h_rows.add(int(seg["row"]))
        next_color += 1

    while len([seg for seg in segments if seg["orient"] == "v"]) < v_count:
        if next_color >= len(pair_colors):
            return None
        seg = _place_noncrossing_segment(
            rng,
            top_h,
            width,
            "v",
            pair_colors[next_color],
            occupied,
            used_h_rows,
            used_v_cols,
        )
        if seg is None:
            return None
        segments.append(seg)
        occupied |= _segment_cells(seg)
        used_v_cols.add(int(seg["col"]))
        next_color += 1

    if len(segments) != pair_count:
        return None

    crossings = _planned_crossings(segments)
    if crossing_mode:
        if len(crossings) != 1:
            return None
        colors_at_cross = set(crossings[0])
        if colors_at_cross != {crossing_h_color, crossing_v_color}:
            return None
        palette_left = crossing_h_color
        palette_mid = crossing_v_color
    else:
        if crossings:
            return None
        h_colors = [int(seg["color"]) for seg in segments if seg["orient"] == "h"]
        v_colors = [int(seg["color"]) for seg in segments if seg["orient"] == "v"]
        palette_left = rng.choice(h_colors)
        palette_mid = rng.choice(v_colors)

    if rng.randint(0, 1):
        palette_left, palette_mid = palette_mid, palette_left

    grid = np.full((height, width), bg, dtype=int)
    grid[sep_r, :] = divider

    for seg in segments:
        color = int(seg["color"])
        if seg["orient"] == "h":
            row = int(seg["row"])
            grid[row, int(seg["c0"])] = color
            grid[row, int(seg["c1"])] = color
        else:
            col = int(seg["col"])
            grid[int(seg["r0"]), col] = color
            grid[int(seg["r1"]), col] = color

    palette_r = rng.randint(sep_r + 1, height - 1)
    palette_c0 = rng.randint(1, width - 4)
    grid[palette_r, palette_c0 : palette_c0 + 3] = [palette_left, palette_mid, cross_color]

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
