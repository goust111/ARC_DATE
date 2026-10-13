from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
SURROUND = 5
MARKER_COLORS = (1, 2, 3, 7)
PAIR_COUNT_RANGE = (2, 4)
GRID_SIDE_RANGE = (18, 24)
EDGE_MARGIN = 2
MIN_CENTER_DISTANCE = 7
MAX_GENERATION_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 250


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _glyph_bbox(center: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = map(int, center)
    h, w = map(int, shape)
    return max(0, r - 1), max(0, c - 1), min(h - 1, r + 1), min(w - 1, c + 1)


def _place_glyph(out: np.ndarray, center: tuple[int, int], color: int) -> None:
    r, c = map(int, center)
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if _in_bounds(out.shape, rr, cc):
                out[rr, cc] = SURROUND
    out[r, c] = int(color)


def _bridge_cells(
    shape: tuple[int, int], a: tuple[int, int], b: tuple[int, int]
) -> list[tuple[int, int, int]]:
    ar, ac = map(int, a)
    br, bc = map(int, b)
    if ar == br:
        if ac > bc:
            ac, bc = bc, ac
        left_box = _glyph_bbox((ar, ac), shape)
        right_box = _glyph_bbox((br, bc), shape)
        start = int(left_box[3] + 1)
        end = int(right_box[1] - 1)
        if start > end:
            return []
        gap_len = end - start + 1
        return [(ar, c, min(c - start, end - c)) for c in range(start, end + 1)]
    if ac == bc:
        if ar > br:
            ar, br = br, ar
        top_box = _glyph_bbox((ar, ac), shape)
        bottom_box = _glyph_bbox((br, bc), shape)
        start = int(top_box[2] + 1)
        end = int(bottom_box[0] - 1)
        if start > end:
            return []
        gap_len = end - start + 1
        return [(r, ac, min(r - start, end - r)) for r in range(start, end + 1)]
    raise AssertionError("same-color markers must share a row or a column")


def _paint_bridge(out: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int, bg: int) -> None:
    for r, c, end_dist in _bridge_cells(out.shape, a, b):
        if end_dist % 2 == 0 and int(out[r, c]) == int(bg):
            out[r, c] = int(color)


def _pair_records(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    records = []
    for color in _colors_present(g, exclude={bg}):
        coords = np.argwhere(g == int(color))
        assert coords.shape == (2, 2), "each foreground color must appear exactly twice"
        p1 = tuple(map(int, coords[0]))
        p2 = tuple(map(int, coords[1]))
        if p2 < p1:
            p1, p2 = p2, p1
        same_row = int(p1[0]) == int(p2[0])
        same_col = int(p1[1]) == int(p2[1])
        assert same_row ^ same_col, "same-color markers must align horizontally or vertically"
        records.append(
            {
                "color": int(color),
                "p1": p1,
                "p2": p2,
                "axis": "h" if same_row else "v",
                "distance": abs(int(p1[0]) - int(p2[0])) + abs(int(p1[1]) - int(p2[1])),
            }
        )
    records.sort(key=lambda rec: (rec["p1"][0], rec["p1"][1], rec["p2"][0], rec["p2"][1], rec["color"]))
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each paired marker into a grey-ringed glyph and bridge matching glyphs with symmetric color dashes.

    Input structure:
    - The background is black and the only foreground objects are isolated single-pixel color markers.
    - Each foreground color appears exactly twice, and the two markers of one color share either one row or one column.
    - Markers are spaced so their 3x3 glyphs do not overlap and each color pair has an empty straight gap between glyph boxes.

    Rule:
    - Replace every marker by a centered 3x3 glyph whose center keeps the marker color and whose eight neighbors become grey.
    - For each color pair, measure the horizontal or vertical gap from the outside of one glyph bbox to the outside of the other glyph bbox, along the line through the marker centers.
    - In that gap, color a bridge pixel with the pair color iff its distance to the nearer glyph-edge endpoint is even; odd-distance gap cells stay black.

    Key insight:
    - The dash pattern is anchored to the two glyph boxes, not to the original marker pixels or to absolute row/column parity, so both bridge ends always start with a colored dash.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    assert int(bg) == BACKGROUND, "expected black background"

    colors = _colors_present(g, exclude={bg})
    assert 2 <= len(colors) <= 4, "expected 2-4 marker colors"
    assert SURROUND not in colors, "grey is reserved for the surrounding glyph ring"

    marker_coords = np.argwhere(g != int(bg))
    assert marker_coords.shape[0] == 2 * len(colors), "expected exactly two markers per color"

    out = np.full_like(g, int(bg))
    records = _pair_records(g, bg)

    for rec in records:
        _place_glyph(out, rec["p1"], rec["color"])
        _place_glyph(out, rec["p2"], rec["color"])

    for rec in records:
        _paint_bridge(out, rec["p1"], rec["p2"], rec["color"], bg)

    return out


def _mask_with_halo(mask: np.ndarray) -> np.ndarray:
    halo = mask.copy()
    rows, cols = np.nonzero(mask)
    for r, c in zip(rows, cols):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if _in_bounds(mask.shape, rr, cc):
                    halo[rr, cc] = True
    return halo


def _record_mask(shape: tuple[int, int], record: dict) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for center in (record["p1"], record["p2"]):
        r, c = map(int, center)
        r0, c0, r1, c1 = _glyph_bbox((r, c), shape)
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    for r, c, _dist in _bridge_cells(shape, record["p1"], record["p2"]):
        mask[r, c] = True
    return mask


def _orientation_plan(pair_count: int) -> list[str]:
    axes = ["h", "v"]
    while len(axes) < int(pair_count):
        axes.append(random.choice(["h", "v"]))
    random.shuffle(axes)
    return axes[:pair_count]


def _gap_parity_plan(pair_count: int) -> list[int]:
    parities = [0, 1]
    while len(parities) < int(pair_count):
        parities.append(random.choice([0, 1]))
    random.shuffle(parities)
    return parities[:pair_count]


def _sample_gap(axis_span: int, parity: int) -> int | None:
    max_gap = int(axis_span - 8)
    options = [gap for gap in range(4, max_gap + 1) if gap % 2 == int(parity)]
    return random.choice(options) if options else None


def _sample_record(shape: tuple[int, int], color: int, axis: str, parity: int) -> dict | None:
    h, w = map(int, shape)
    if axis == "h":
        gap = _sample_gap(w, parity)
        if gap is None:
            return None
        dist = gap + 3
        row = random.randint(EDGE_MARGIN, h - EDGE_MARGIN - 1)
        left = random.randint(EDGE_MARGIN, w - EDGE_MARGIN - 1 - dist)
        return {"color": int(color), "p1": (row, left), "p2": (row, left + dist), "axis": "h", "gap": gap}
    gap = _sample_gap(h, parity)
    if gap is None:
        return None
    dist = gap + 3
    col = random.randint(EDGE_MARGIN, w - EDGE_MARGIN - 1)
    top = random.randint(EDGE_MARGIN, h - EDGE_MARGIN - 1 - dist)
    return {"color": int(color), "p1": (top, col), "p2": (top + dist, col), "axis": "v", "gap": gap}


def _validate_input(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    assert int(bg) == BACKGROUND
    records = _pair_records(grid, bg)
    assert 2 <= len(records) <= 4

    axes = {rec["axis"] for rec in records}
    parities = set()
    coverage = np.zeros(grid.shape, dtype=bool)
    for rec in records:
        assert rec["distance"] >= MIN_CENTER_DISTANCE
        for center in (rec["p1"], rec["p2"]):
            r, c = map(int, center)
            assert EDGE_MARGIN <= r < grid.shape[0] - EDGE_MARGIN
            assert EDGE_MARGIN <= c < grid.shape[1] - EDGE_MARGIN
        bridge = _bridge_cells(grid.shape, rec["p1"], rec["p2"])
        assert bridge, "each pair must leave a non-empty gap after glyph expansion"
        parities.add(len(bridge) % 2)

        mask = _record_mask(grid.shape, rec)
        assert not bool((coverage & mask).any()), "pair footprints must be disjoint"
        coverage |= mask

    assert axes == {"h", "v"} or len(records) == 1
    assert parities == {0, 1} or len(records) == 1
    return records


def _validate_output(inp: np.ndarray, out: np.ndarray, records: list[dict]) -> None:
    bg = _get_bg(inp)
    assert out.shape == inp.shape
    expected = np.full_like(inp, int(bg))
    for rec in records:
        _place_glyph(expected, rec["p1"], rec["color"])
        _place_glyph(expected, rec["p2"], rec["color"])
    for rec in sorted(records, key=lambda rec: (rec["p1"][0], rec["p1"][1], rec["color"])):
        for r, c, end_dist in _bridge_cells(out.shape, rec["p1"], rec["p2"]):
            if end_dist % 2 == 0:
                assert int(expected[r, c]) == int(bg)
                expected[r, c] = int(rec["color"])
            else:
                assert int(expected[r, c]) == int(bg)
    assert np.array_equal(out, expected), "output does not match glyph-plus-dashed-bridge semantics"


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(MARKER_COLORS, pair_count)
    axes = _orientation_plan(pair_count)
    parities = _gap_parity_plan(pair_count)

    occupied = np.zeros((h, w), dtype=bool)
    blocked = np.zeros((h, w), dtype=bool)
    records: list[dict] = []

    for color, axis, parity in zip(colors, axes, parities):
        placed = None
        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            record = _sample_record((h, w), color, axis, parity)
            if record is None:
                continue
            mask = _record_mask((h, w), record)
            if bool((blocked & mask).any()):
                continue
            placed = record
            occupied |= mask
            blocked |= _mask_with_halo(mask)
            records.append(record)
            break
        if placed is None:
            return None

    inp = np.zeros((h, w), dtype=int)
    for rec in records:
        r1, c1 = map(int, rec["p1"])
        r2, c2 = map(int, rec["p2"])
        inp[r1, c1] = int(rec["color"])
        inp[r2, c2] = int(rec["color"])

    try:
        parsed_records = _validate_input(inp)
        out = main(inp)
        _validate_output(inp, out, parsed_records)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
