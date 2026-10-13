from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rectangular_frame(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))
    return set(points) == expected


def _segment_lengths(row: np.ndarray) -> dict[int, int]:
    w = row.shape[0]
    out: dict[int, int] = {}
    c = 0
    while c < w:
        color = int(row[c])
        if color in (BG, FRAME):
            c += 1
            continue
        start = c
        while c < w and int(row[c]) == color:
            c += 1
        if color in out:
            raise AssertionError("each strip color must appear in one contiguous segment")
        out[color] = c - start
    assert out, "bottom strip must contain at least one colored segment"
    return out


def _paint_wrapped(
    grid: np.ndarray,
    bottom: int,
    right: int,
    start_r: int,
    start_c: int,
    color: int,
    length: int,
) -> None:
    row_width = right - start_c + 1
    assert row_width >= 1
    remaining = int(length)
    r = int(start_r)
    while remaining > 0:
        assert r <= bottom, "fill length exceeds the region capacity"
        take = min(remaining, row_width)
        grid[r, start_c : start_c + take] = int(color)
        remaining -= take
        r += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed region with the strip length that matches its marker color.

    Input structure:
    - The background is Black.
    - Several Grey 1-pixel rectangular frames enclose empty interiors; each interior contains exactly one single colored marker pixel.
    - The bottom row contains a horizontal color strip made of contiguous segments, one segment per marker color, and the segment lengths are all different.

    Rule:
    - Read the marker color C inside each Grey-framed region.
    - Find the contiguous segment of color C in the bottom strip and measure its length L.
    - Starting from the marker pixel as the leftmost filled cell, paint L cells of color C inside that same region, moving rightward.
    - When the fill reaches the Grey border before using all L cells, continue on the next row down from the same starting column, again moving rightward.
    - Keep the Grey frames and the bottom strip unchanged.

    Key insight:
    - The marker color links a framed region to exactly one strip segment, while the segment length determines how many cells must be painted with row-wise wrapping inside that region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 4 and g.shape[1] >= 4

    strip_lengths = _segment_lengths(g[-1])
    frame_mask = g.copy() == FRAME
    frame_mask[-1, :] = False

    out = g.copy()
    frames = _components4(frame_mask)
    assert frames, "expected at least one framed region"

    for frame in frames:
        assert _is_rectangular_frame(frame), "grey components must be rectangular frames"
        r0, c0, r1, c1 = _bbox(frame)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_coords = np.argwhere((inner != BG) & (inner != FRAME))
        assert marker_coords.shape[0] == 1, "each frame interior must have one marker"
        mr_rel, mc_rel = map(int, marker_coords[0])
        mr = r0 + 1 + mr_rel
        mc = c0 + 1 + mc_rel
        color = int(g[mr, mc])
        assert color in strip_lengths, "marker color must exist in the bottom strip"
        _paint_wrapped(out, r1 - 1, c1 - 1, mr, mc, color, strip_lengths[color])

    return out


def _choose_lengths(count: int) -> list[int]:
    choices = list(range(2, 9))
    return random.sample(choices, count)


def _choose_branches(count: int) -> list[bool]:
    wraps = [bool(random.getrandbits(1)) for _ in range(count)]
    if count >= 2 and (all(wraps) or not any(wraps)):
        wraps[0] = False
        wraps[1] = True
        random.shuffle(wraps)
    return wraps


def _sample_region_geometry(length: int, wrap: bool) -> dict[str, int] | None:
    for _ in range(200):
        inner_h = random.randint(3, 6)
        inner_w = random.randint(4, 8)
        marker_r = random.randint(0, inner_h - 1 if not wrap else inner_h - 2)
        marker_c = random.randint(0, inner_w - 1)
        row_width = inner_w - marker_c
        rows_available = inner_h - marker_r
        capacity = row_width * rows_available
        if wrap:
            if row_width < length <= capacity:
                return {
                    "inner_h": inner_h,
                    "inner_w": inner_w,
                    "marker_r": marker_r,
                    "marker_c": marker_c,
                    "frame_h": inner_h + 2,
                    "frame_w": inner_w + 2,
                }
        else:
            if length <= row_width:
                return {
                    "inner_h": inner_h,
                    "inner_w": inner_w,
                    "marker_r": marker_r,
                    "marker_c": marker_c,
                    "frame_h": inner_h + 2,
                    "frame_w": inner_w + 2,
                }
    return None


def _split_shelves(indices: list[int]) -> list[list[int]]:
    n = len(indices)
    if n == 1:
        return [indices]
    if n == 4:
        cut = random.randint(1, 3)
        return [indices[:cut], indices[cut:]]
    if n == 2:
        if random.choice([False, True]):
            return [indices]
        return [[indices[0]], [indices[1]]]
    if random.choice([False, True]):
        return [indices]
    cut = random.randint(1, n - 1)
    return [indices[:cut], indices[cut:]]


def _layout_regions(specs: list[dict[str, int]], band_len: int) -> tuple[int, int, list[dict[str, int]]] | None:
    order = list(range(len(specs)))
    random.shuffle(order)
    shelves = _split_shelves(order)
    shelf_gaps: list[list[int]] = []
    shelf_widths: list[int] = []
    shelf_heights: list[int] = []

    for shelf in shelves:
        gaps = [random.randint(1, 3) for _ in range(len(shelf) - 1)]
        width = sum(specs[idx]["frame_w"] for idx in shelf) + sum(gaps)
        height = max(specs[idx]["frame_h"] for idx in shelf)
        shelf_gaps.append(gaps)
        shelf_widths.append(width)
        shelf_heights.append(height)

    min_w = max([band_len] + shelf_widths)
    if min_w > 30:
        return None
    width = random.randint(min_w, 30)

    v_gaps = [random.randint(1, 3) for _ in range(len(shelves) - 1)]
    strip_gap = random.randint(1, 3)
    min_h = sum(shelf_heights) + sum(v_gaps) + strip_gap + 1
    if min_h > 30:
        return None
    height = random.randint(min_h, 30)
    extra_top = random.randint(0, height - min_h)

    placed = [dict(spec) for spec in specs]
    current_row = extra_top
    for shelf_idx, shelf in enumerate(shelves):
        shelf_height = shelf_heights[shelf_idx]
        used_width = shelf_widths[shelf_idx]
        start_c = random.randint(0, width - used_width)
        c = start_c
        for item_idx, spec_idx in enumerate(shelf):
            spec = placed[spec_idx]
            r0 = current_row + random.randint(0, shelf_height - spec["frame_h"])
            spec["r0"] = r0
            spec["c0"] = c
            c += spec["frame_w"]
            if item_idx < len(shelf) - 1:
                c += shelf_gaps[shelf_idx][item_idx]
        current_row += shelf_height
        if shelf_idx < len(shelves) - 1:
            current_row += v_gaps[shelf_idx]

    return height, width, placed


def _render_case(specs: list[dict[str, int]], height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), BG, dtype=int)
    expected = np.full((height, width), BG, dtype=int)

    segments = specs[:]
    random.shuffle(segments)
    band_len = sum(spec["length"] for spec in segments)
    band_start = random.randint(0, width - band_len)
    c = band_start
    for spec in segments:
        inp[-1, c : c + spec["length"]] = spec["color"]
        expected[-1, c : c + spec["length"]] = spec["color"]
        c += spec["length"]

    for spec in specs:
        r0 = spec["r0"]
        c0 = spec["c0"]
        r1 = r0 + spec["frame_h"] - 1
        c1 = c0 + spec["frame_w"] - 1
        inp[r0, c0 : c1 + 1] = FRAME
        inp[r1, c0 : c1 + 1] = FRAME
        inp[r0 : r1 + 1, c0] = FRAME
        inp[r0 : r1 + 1, c1] = FRAME
        expected[r0, c0 : c1 + 1] = FRAME
        expected[r1, c0 : c1 + 1] = FRAME
        expected[r0 : r1 + 1, c0] = FRAME
        expected[r0 : r1 + 1, c1] = FRAME

        mr = r0 + 1 + spec["marker_r"]
        mc = c0 + 1 + spec["marker_c"]
        inp[mr, mc] = spec["color"]
        expected[mr, mc] = spec["color"]
        _paint_wrapped(expected, r1 - 1, c1 - 1, mr, mc, spec["color"], spec["length"])

    return inp, expected


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    count = random.randint(2, 4)
    colors = random.sample(PALETTE, count)
    lengths = _choose_lengths(count)
    wraps = _choose_branches(count)

    specs: list[dict[str, int]] = []
    for color, length, wrap in zip(colors, lengths, wraps):
        geom = _sample_region_geometry(length, wrap)
        if geom is None:
            return None
        geom["color"] = color
        geom["length"] = length
        geom["wrap"] = int(wrap)
        specs.append(geom)

    laid_out = _layout_regions(specs, sum(lengths))
    if laid_out is None:
        return None
    height, width, placed = laid_out
    return _render_case(placed, height, width)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_case()
        if built is None:
            continue
        inp, expected = built
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid puzzle input")
