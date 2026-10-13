from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7

SEGMENT_COLORS = [BLUE, GREEN, YELLOW, PINK, ORANGE]
GEN_RETRIES = 600


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _components_of_color(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_result_zone_start(grid: np.ndarray) -> int:
    grey_cols = np.flatnonzero(grid[-1] == GREY)
    assert grey_cols.size > 0, "expected a grey baseline suffix on the last row"
    start = int(grey_cols[0])
    assert np.all(grid[-1, start:] == GREY), "baseline must be a contiguous suffix"
    return start


def _segment_signature(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = sorted((int(r), int(c)) for r, c in cells)
    ks = {r + c for r, c in coords}
    assert len(ks) == 1, "each segment must lie on one anti-diagonal"
    k = int(next(iter(ks)))
    bs = sorted(r - c for r, c in coords)
    assert all(bs[i + 1] - bs[i] == 2 for i in range(len(bs) - 1)), "segment must be contiguous"
    return k, int(bs[0]), int(bs[-1]), int(len(bs))


def _fill_cells_between_segments(
    seg_a: tuple[int, int, int, int], seg_b: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    k1, bmin1, bmax1, _len1 = seg_a
    k2, bmin2, bmax2, _len2 = seg_b
    if k1 > k2:
        k1, k2 = k2, k1
        bmin1, bmin2 = bmin2, bmin1
        bmax1, bmax2 = bmax2, bmax1
    assert (k1 % 2) == (k2 % 2), "paired anti-diagonals must share parity"

    bs = sorted(set(range(bmin1, bmax1 + 1, 2)) & set(range(bmin2, bmax2 + 1, 2)))
    fill: list[tuple[int, int]] = []
    for b in bs:
        for k in range(k1 + 2, k2, 2):
            fill.append(((k + b) // 2, (k - b) // 2))
    return fill


def _extract_pairs(grid: np.ndarray, left_width: int) -> list[dict[str, object]]:
    left = grid[:, :left_width]
    colors = [int(color) for color in np.unique(left) if int(color) not in (BLACK, GREY)]
    pairs: list[dict[str, object]] = []
    for color in colors:
        comps = _components_of_color(left == color)
        assert len(comps) == 2, "each segment color must form exactly two components"
        seg_a = _segment_signature(comps[0])
        seg_b = _segment_signature(comps[1])
        fill_cells = _fill_cells_between_segments(seg_a, seg_b)
        area = sum(1 for r, c in fill_cells if grid[r, c] == BLACK)
        assert area > 0, "each pair must enclose a non-empty fill region"
        pairs.append(
            {
                "color": color,
                "segments": (seg_a, seg_b),
                "fill_cells": fill_cells,
                "area": int(area),
            }
        )
    assert len(pairs) >= 1
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each parallel anti-diagonal segment pair, count the red area, then sort those areas into colored bars.

    Input structure:
    - The grid is split vertically into a left construction zone and a right result zone.
    - The left zone has a black background and 3 to 4 colors, each color forming exactly two short anti-diagonal segments.
    - The right zone is empty black space above a grey baseline on the last row.

    Rule:
    - Group left-zone cells by color and recover the two same-color anti-diagonal segments for each group.
    - For every main-diagonal axis covered by both segments of a pair, fill the black cells strictly between the two segments with red.
    - Count the filled red cells for each pair as its area, sort the pairs by increasing area, and draw one-cell-wide bars in the right zone on odd columns above the grey baseline using the pair's original color and area as height.
    - Keep the original segments, the red fill, and the baseline.

    Key insight:
    - The same diagonal fill serves two roles at once: it is the visible left-side transformation and also the quantitative measurement that controls right-side bar ordering and heights.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    result_start = _find_result_zone_start(g)
    pairs = _extract_pairs(g, result_start)

    out = g.copy()
    for pair in pairs:
        for r, c in pair["fill_cells"]:
            if 0 <= r < h and 0 <= c < result_start and out[r, c] == BLACK:
                out[r, c] = RED

    ordered = sorted(pairs, key=lambda pair: (int(pair["area"]), int(pair["color"])))
    result_width = w - result_start
    assert result_width >= 2 * len(ordered) + 1, "result zone too narrow for all bars"

    for i, pair in enumerate(ordered):
        height = int(pair["area"])
        col = result_start + 1 + 2 * i
        assert 1 <= height <= h - 1, "bar height must fit above the baseline"
        out[h - 1 - height : h - 1, col] = int(pair["color"])

    return out


def _make_pair_spec(r0: int, c0: int, length: int, shift: int) -> dict[str, object]:
    seg1 = [(r0 + i, c0 - i) for i in range(length)]
    seg2 = [(r0 + shift + i, c0 + shift - i) for i in range(length)]
    fill = [(r0 + t + i, c0 + t - i) for i in range(length) for t in range(1, shift)]
    region = set(seg1) | set(seg2) | set(fill)
    padded = {
        (r + dr, c + dc)
        for r, c in region
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
    }
    return {
        "r0": int(r0),
        "c0": int(c0),
        "length": int(length),
        "shift": int(shift),
        "area": int(length * (shift - 1)),
        "seg1": seg1,
        "seg2": seg2,
        "fill": fill,
        "region": region,
        "padded": padded,
    }


def _all_param_combos(max_height: int) -> list[tuple[int, int]]:
    combos = []
    for length in range(3, 6):
        for shift in range(2, 5):
            area = length * (shift - 1)
            if area <= max_height:
                combos.append((length, shift))
    return combos


def _param_set_is_good(params: list[tuple[int, int]]) -> bool:
    lengths = [length for length, _shift in params]
    shifts = [shift for _length, shift in params]
    areas = [length * (shift - 1) for length, shift in params]
    if len(set(areas)) != len(areas):
        return False

    same_length_diff_area = any(
        lengths[i] == lengths[j] and areas[i] != areas[j]
        for i in range(len(params))
        for j in range(i + 1, len(params))
    )
    same_shift_diff_area = any(
        shifts[i] == shifts[j] and areas[i] != areas[j]
        for i in range(len(params))
        for j in range(i + 1, len(params))
    )
    length_short_bigger_area = any(
        lengths[i] < lengths[j] and areas[i] > areas[j]
        for i in range(len(params))
        for j in range(len(params))
        if i != j
    )
    shift_smaller_bigger_area = any(
        shifts[i] < shifts[j] and areas[i] > areas[j]
        for i in range(len(params))
        for j in range(len(params))
        if i != j
    )
    return same_length_diff_area and same_shift_diff_area and length_short_bigger_area and shift_smaller_bigger_area


def _sample_param_set(n_pairs: int, max_height: int) -> list[tuple[int, int]] | None:
    combos = _all_param_combos(max_height)
    for _ in range(300):
        params = random.sample(combos, n_pairs)
        if _param_set_is_good(params):
            return params
    return None


def _fits_left_zone(spec: dict[str, object], h: int, left_w: int) -> bool:
    for r, c in spec["region"]:
        if not (0 <= r < h - 1 and 0 <= c < left_w):
            return False
    return True


def _place_specs(h: int, left_w: int, params: list[tuple[int, int]]) -> list[dict[str, object]] | None:
    placed: list[dict[str, object]] = []
    padded_union: set[tuple[int, int]] = set()

    for length, shift in sorted(params, key=lambda item: item[0] * item[1], reverse=True):
        candidates: list[dict[str, object]] = []
        max_r0 = h - 2 - shift - (length - 1)
        min_c0 = length - 1
        max_c0 = left_w - 1 - shift
        if max_r0 < 0 or max_c0 < min_c0:
            return None
        for r0 in range(max_r0 + 1):
            for c0 in range(min_c0, max_c0 + 1):
                spec = _make_pair_spec(r0, c0, length, shift)
                if not _fits_left_zone(spec, h, left_w):
                    continue
                if spec["padded"] & padded_union:
                    continue
                candidates.append(spec)
        if not candidates:
            return None
        choice = random.choice(candidates)
        placed.append(choice)
        padded_union |= set(choice["padded"])

    random.shuffle(placed)
    return placed


def _build_input_from_specs(
    h: int, left_w: int, right_w: int, specs: list[dict[str, object]], colors: list[int]
) -> tuple[np.ndarray, list[dict[str, object]]]:
    grid = np.zeros((h, left_w + right_w), dtype=int)
    grid[-1, left_w:] = GREY

    colored_specs: list[dict[str, object]] = []
    for spec, color in zip(specs, colors):
        enriched = dict(spec)
        enriched["color"] = int(color)
        for r, c in enriched["seg1"]:
            grid[r, c] = int(color)
        for r, c in enriched["seg2"]:
            grid[r, c] = int(color)
        colored_specs.append(enriched)
    return grid, colored_specs


def _expected_output_from_specs(
    inp: np.ndarray, left_w: int, specs: list[dict[str, object]]
) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        for r, c in spec["fill"]:
            if out[r, c] == BLACK:
                out[r, c] = RED
    ordered = sorted(specs, key=lambda spec: (int(spec["area"]), int(spec["color"])))
    h = inp.shape[0]
    for i, spec in enumerate(ordered):
        col = left_w + 1 + 2 * i
        height = int(spec["area"])
        out[h - 1 - height : h - 1, col] = int(spec["color"])
    return out


def _validate_semantics(inp: np.ndarray, specs: list[dict[str, object]], left_w: int) -> bool:
    h, w = inp.shape
    if max(h, w) > 30:
        return False
    if not np.all(inp[-1, :left_w] == BLACK):
        return False
    if not np.all(inp[-1, left_w:] == GREY):
        return False

    areas = [int(spec["area"]) for spec in specs]
    if len(set(areas)) != len(areas):
        return False
    if any(area <= 0 or area > h - 1 for area in areas):
        return False

    expected = _expected_output_from_specs(inp, left_w, specs)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    ordered = sorted(specs, key=lambda spec: int(spec["area"]))
    for i, spec in enumerate(ordered):
        col = left_w + 1 + 2 * i
        height = int(spec["area"])
        if np.sum(actual[: h - 1, col] == int(spec["color"])) != height:
            return False
        if height < h - 1 and np.any(actual[: h - 1 - height, col] != BLACK):
            return False

    lengths = [int(spec["length"]) for spec in specs]
    shifts = [int(spec["shift"]) for spec in specs]
    if not any(lengths[i] == lengths[j] and areas[i] != areas[j] for i in range(len(specs)) for j in range(i + 1, len(specs))):
        return False
    if not any(shifts[i] == shifts[j] and areas[i] != areas[j] for i in range(len(specs)) for j in range(i + 1, len(specs))):
        return False
    if not any(lengths[i] < lengths[j] and areas[i] > areas[j] for i in range(len(specs)) for j in range(len(specs)) if i != j):
        return False
    if not any(shifts[i] < shifts[j] and areas[i] > areas[j] for i in range(len(specs)) for j in range(len(specs)) if i != j):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.choice([3, 4, 4])
    h = random.randint(16, 20)
    left_w = random.randint(13, 17)
    right_w = 2 * n_pairs + 1
    if left_w + right_w > 30:
        return None

    params = _sample_param_set(n_pairs, h - 1)
    if params is None:
        return None
    specs = _place_specs(h, left_w, params)
    if specs is None:
        return None

    colors = random.sample(SEGMENT_COLORS, n_pairs)
    inp, colored_specs = _build_input_from_specs(h, left_w, right_w, specs, colors)
    if not _validate_semantics(inp, colored_specs, left_w):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
