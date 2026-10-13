from __future__ import annotations

import numpy as np


BG = 0
YELLOW = 4
PAIR_COLORS = (1, 2, 3)
MAIN = "main"
ANTI = "anti"
MAX_GENERATE_ATTEMPTS = 400


def _segment_from_cells(color: int, orientation: str, cells: list[tuple[int, int]]) -> dict:
    arr = np.array(sorted(cells), dtype=int)
    if orientation == MAIN:
        diag_values = arr[:, 0] - arr[:, 1]
        span_values = arr[:, 0] + arr[:, 1]
    else:
        diag_values = arr[:, 0] + arr[:, 1]
        span_values = arr[:, 0] - arr[:, 1]
    assert len(set(map(int, diag_values))) == 1
    return {
        "color": int(color),
        "orientation": orientation,
        "cells": [(int(r), int(c)) for r, c in arr],
        "diag_id": int(diag_values[0]),
        "span_min": int(span_values.min()),
        "span_max": int(span_values.max()),
        "length": int(arr.shape[0]),
    }


def _extract_segments(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    segments: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        mask = g == color
        for r in range(h):
            for c in range(w):
                if not mask[r, c]:
                    continue
                if not (r > 0 and c > 0 and mask[r - 1, c - 1]):
                    rr, cc = r, c
                    cells: list[tuple[int, int]] = []
                    while rr < h and cc < w and mask[rr, cc]:
                        cells.append((rr, cc))
                        rr += 1
                        cc += 1
                    if len(cells) >= 2:
                        segments.append(_segment_from_cells(color, MAIN, cells))
                if not (r > 0 and c + 1 < w and mask[r - 1, c + 1]):
                    rr, cc = r, c
                    cells = []
                    while rr < h and cc >= 0 and mask[rr, cc]:
                        cells.append((rr, cc))
                        rr += 1
                        cc -= 1
                    if len(cells) >= 2:
                        segments.append(_segment_from_cells(color, ANTI, cells))
        color_cell_count = int(mask.sum())
        used_cells = sum(seg["length"] for seg in segments if seg["color"] == color)
        assert used_cells == color_cell_count, "every colored pixel must belong to exactly one diagonal segment"
    return segments


def _pair_segments(segments: list[dict]) -> list[dict]:
    by_color: dict[int, list[dict]] = {}
    for seg in segments:
        by_color.setdefault(seg["color"], []).append(seg)

    pairs: list[dict] = []
    for color, color_segments in sorted(by_color.items()):
        orientations = {seg["orientation"] for seg in color_segments}
        assert len(orientations) == 1, "same-color segments must share one diagonal orientation"
        orientation = next(iter(orientations))
        ordered = sorted(color_segments, key=lambda seg: seg["diag_id"])
        assert len(ordered) % 2 == 0 and len(ordered) >= 2
        diffs = [ordered[i + 1]["diag_id"] - ordered[i]["diag_id"] for i in range(len(ordered) - 1)]
        assert all(diff > 0 for diff in diffs)
        i = 0
        while i < len(ordered):
            a = ordered[i]
            b = ordered[i + 1]
            assert a["length"] == b["length"], "paired segments must have equal length"
            pairs.append({"color": int(color), "orientation": orientation, "segments": (a, b)})
            i += 2
    return pairs


def _pair_region(shape: tuple[int, int], pair: dict) -> np.ndarray:
    rr, cc = np.indices(shape, dtype=int)
    seg_a, seg_b = pair["segments"]
    if pair["orientation"] == ANTI:
        k_lo, k_hi = sorted((seg_a["diag_id"], seg_b["diag_id"]))
        b_lo = max(seg_a["span_min"], seg_b["span_min"])
        b_hi = min(seg_a["span_max"], seg_b["span_max"])
        return (k_lo < rr + cc) & (rr + cc < k_hi) & (b_lo <= rr - cc) & (rr - cc <= b_hi)
    b_lo, b_hi = sorted((seg_a["diag_id"], seg_b["diag_id"]))
    k_lo = max(seg_a["span_min"], seg_b["span_min"])
    k_hi = min(seg_a["span_max"], seg_b["span_max"])
    return (b_lo < rr - cc) & (rr - cc < b_hi) & (k_lo <= rr + cc) & (rr + cc <= k_hi)


def _paint_pair(out: np.ndarray, pair: dict, axis_override: str | None = None) -> np.ndarray:
    region = _pair_region(out.shape, pair)
    fill_mask = region & (out == BG)
    coords = np.argwhere(fill_mask)
    assert coords.size > 0

    axis = axis_override
    if axis is None:
        axis = "rows" if pair["orientation"] == ANTI else "cols"

    if axis == "rows":
        start = int(coords[:, 0].min())
        primary = (np.indices(out.shape)[0] - start) % 2 == 0
    else:
        start = int(coords[:, 1].min())
        primary = (np.indices(out.shape)[1] - start) % 2 == 0

    color = int(pair["color"])
    out[fill_mask & primary] = color
    out[fill_mask & ~primary] = YELLOW
    return out


def _render_from_pairs(shape: tuple[int, int], pairs: list[dict], wrong_axis_for: int | None = None) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for pair in pairs:
        for seg in pair["segments"]:
            for r, c in seg["cells"]:
                out[r, c] = pair["color"]
    for idx, pair in enumerate(pairs):
        axis_override = None
        if wrong_axis_for == idx:
            axis_override = "cols" if pair["orientation"] == ANTI else "rows"
        _paint_pair(out, pair, axis_override=axis_override)
    return out


def _render_input_from_pairs(shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for pair in pairs:
        for seg in pair["segments"]:
            for r, c in seg["cells"]:
                out[r, c] = pair["color"]
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    coords = np.argwhere(mask)
    h, w = mask.shape
    for r, c in coords:
        r0 = max(0, int(r) - radius)
        r1 = min(h, int(r) + radius + 1)
        c0 = max(0, int(c) - radius)
        c1 = min(w, int(c) + radius + 1)
        out[r0:r1, c0:c1] = True
    return out


def _make_pair(shape: tuple[int, int], color: int, orientation: str, length: int, shift: int, r0: int, c0: int) -> dict:
    if orientation == ANTI:
        seg_a = [(r0 + i, c0 - i) for i in range(length)]
        seg_b = [(r0 + shift + i, c0 + shift - i) for i in range(length)]
    else:
        seg_a = [(r0 + i, c0 + i) for i in range(length)]
        seg_b = [(r0 + shift + i, c0 - shift + i) for i in range(length)]
    pair = {
        "color": int(color),
        "orientation": orientation,
        "segments": (
            _segment_from_cells(color, orientation, seg_a),
            _segment_from_cells(color, orientation, seg_b),
        ),
    }
    region = _pair_region(shape, pair)
    assert region.any()
    return pair


def _sample_orientations(pair_count: int, rng: np.random.Generator) -> list[str]:
    if pair_count == 1:
        return [rng.choice([MAIN, ANTI]).item()]
    if rng.random() < 0.7:
        if pair_count == 2:
            orientations = [MAIN, ANTI]
        else:
            main_count = int(rng.integers(1, pair_count))
            orientations = [MAIN] * main_count + [ANTI] * (pair_count - main_count)
        rng.shuffle(orientations)
        return orientations
    orientation = rng.choice([MAIN, ANTI]).item()
    return [orientation] * pair_count


def _apply_symmetry(grid: np.ndarray, code: int) -> np.ndarray:
    g = np.asarray(grid)
    if code == 0:
        return g.copy()
    if code == 1:
        return np.rot90(g, 1).copy()
    if code == 2:
        return np.rot90(g, 2).copy()
    if code == 3:
        return np.rot90(g, 3).copy()
    if code == 4:
        return np.fliplr(g).copy()
    if code == 5:
        return np.flipud(g).copy()
    if code == 6:
        return np.transpose(g).copy()
    return np.fliplr(np.transpose(g)).copy()


def _validate_instance(inp: np.ndarray, expected: np.ndarray) -> None:
    assert inp.ndim == 2 and expected.ndim == 2
    assert max(inp.shape) <= 30 and max(expected.shape) <= 30
    assert set(np.unique(inp)).issubset({BG, *PAIR_COLORS})
    pairs = _pair_segments(_extract_segments(inp))
    assert len(pairs) in (2, 3)
    assert len({pair["color"] for pair in pairs}) == len(pairs)

    solved = main(inp)
    assert np.array_equal(solved, expected)
    assert not np.array_equal(inp, expected), "output must change the grid"

    for idx, pair in enumerate(pairs):
        pair_only = np.full(inp.shape, BG, dtype=int)
        for seg in pair["segments"]:
            for r, c in seg["cells"]:
                pair_only[r, c] = pair["color"]
        correct = _render_from_pairs(inp.shape, [pair])
        wrong = _render_from_pairs(inp.shape, [pair], wrong_axis_for=0)
        assert not np.array_equal(correct, wrong), "stripe axis must affect the result"

        region = _pair_region(inp.shape, pair)
        fill_mask = region & (pair_only == BG)
        fill_vals = set(map(int, correct[fill_mask]))
        assert pair["color"] in fill_vals and YELLOW in fill_vals, "alternation must use both colors"
        assert correct.shape == wrong.shape

        region_box = _bbox(fill_mask)
        assert region_box is not None
        r0, c0, r1, c1 = region_box
        assert r1 > r0 and c1 > c0, "region must have both row and column extent"
        assert np.any(expected != inp)
        if idx > 0:
            assert not np.array_equal(_render_from_pairs(inp.shape, pairs[:idx]), expected)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each region enclosed by a parallel diagonal-segment pair with alternating stripes whose axis is encoded by the segment direction.

    Input structure:
    - The grid is a black background containing 2-3 colored pairs of short diagonal segments.
    - Each pair uses one color, and the two segments in that pair are parallel and equally long.
    - A segment lies entirely on one main diagonal (r-c constant) or one anti-diagonal (r+c constant), and the enclosed interior is background only.

    Rule:
    - Group non-black pixels by color and extract maximal same-color diagonal segments.
    - Pair same-color segments with the same orientation by nearest parallel-diagonal intercept.
    - For each enclosed parallelogram region, fill only its background pixels with alternating stripes: anti-diagonal segments imply horizontal row stripes, while main-diagonal segments imply vertical column stripes; the first filled row or column uses the pair color and the next uses yellow, alternating thereafter.
    - Keep all original segment pixels unchanged.

    Key insight:
    - The segment pair determines the filled region, but the segment orientation independently encodes whether the alternating fill should advance by rows or by columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    segments = _extract_segments(g)
    pairs = _pair_segments(segments)

    out = g.copy()
    for pair in pairs:
        _paint_pair(out, pair)
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        pair_count = int(rng.choice([2, 3]))
        base_h = int(rng.integers(13 + (pair_count == 3), 19))
        base_w = int(rng.integers(13 + (pair_count == 3), 19))
        shape = (base_h, base_w)

        colors = [int(c) for c in rng.choice(np.array(PAIR_COLORS), size=pair_count, replace=False)]
        orientations = _sample_orientations(pair_count, rng)
        order = list(range(pair_count))
        rng.shuffle(order)

        pairs: list[dict] = []
        forbidden = np.zeros(shape, dtype=bool)
        success = True
        for idx in order:
            color = colors[idx]
            orientation = orientations[idx]
            placed = False
            for _try in range(140):
                length = int(rng.integers(3, 6))
                shift = int(rng.integers(2, 5))
                if orientation == ANTI:
                    if base_h - length - shift < 0 or base_w - length - shift < 0:
                        continue
                    r0 = int(rng.integers(0, base_h - length - shift + 1))
                    c0 = int(rng.integers(length - 1, base_w - shift))
                else:
                    if base_h - length - shift < 0 or base_w - length - shift < 0:
                        continue
                    r0 = int(rng.integers(0, base_h - length - shift + 1))
                    c0 = int(rng.integers(shift, base_w - length + 1))

                pair = _make_pair(shape, color, orientation, length, shift, r0, c0)
                pair_input = _render_input_from_pairs(shape, [pair])
                pair_output = _render_from_pairs(shape, [pair])
                footprint = pair_output != BG
                region = _pair_region(shape, pair) & (pair_input == BG)
                if not region.any():
                    continue
                if np.any(footprint & forbidden):
                    continue
                region_box = _bbox(region)
                assert region_box is not None
                rr0, cc0, rr1, cc1 = region_box
                if rr1 == rr0 or cc1 == cc0:
                    continue
                wrong_axis = _render_from_pairs(shape, [pair], wrong_axis_for=0)
                if np.array_equal(pair_output, wrong_axis):
                    continue
                fill_vals = set(map(int, pair_output[region]))
                if fill_vals != {color, YELLOW}:
                    continue

                pairs.append(pair)
                forbidden |= _dilate(footprint, radius=2)
                placed = True
                break
            if not placed:
                success = False
                break
        if not success:
            continue

        inp = _render_input_from_pairs(shape, pairs)
        expected = _render_from_pairs(shape, pairs)

        symmetry_code = int(rng.integers(0, 8))
        inp = _apply_symmetry(inp, symmetry_code)
        expected = _apply_symmetry(expected, symmetry_code)

        try:
            _validate_instance(inp, expected)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("generate_input: failed to build a valid diagonal-pair stripe instance")
