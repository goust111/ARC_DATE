from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox, find_components, get_bg


BG = 0
GEN_ATTEMPTS = 400
GRID_MIN = 20
GRID_MAX = 30
OUTER_SIZE_MIN = 12
OUTER_SIZE_MAX = 18
INNER_SIZE_MIN = 6
NOTCH_LEN_MIN = 2
NOTCH_LEN_MAX = 6
OUTER_MARGIN_MIN = 2
FRAME_GAP_MIN = 2
FRAME_GAP_MAX = 4
BAR_GAP = 1


def _unique_most_common_non_bg(grid: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, grid.ravel()))
    colors = [color for color in counts if color != int(bg)]
    assert colors, "expected at least one foreground color"
    best = max(counts[color] for color in colors)
    winners = [color for color in colors if counts[color] == best]
    assert len(winners) == 1, "frame color must be uniquely most common among non-background colors"
    return int(winners[0])


def _border_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _draw_border(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    for r, c in _border_cells(box):
        grid[r, c] = int(color)


def _find_frame_boxes(grid: np.ndarray, frame_color: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    frame_mask = grid == int(frame_color)
    outer = bbox(frame_mask)
    assert outer is not None, "frame pixels must exist"
    outer = tuple(map(int, outer))

    remaining = frame_mask.copy()
    for r, c in _border_cells(outer):
        remaining[r, c] = False
    inner = bbox(remaining)
    assert inner is not None, "expected a nested inner frame"
    inner = tuple(map(int, inner))

    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    assert or0 < ir0 < ir1 < or1 and oc0 < ic0 < ic1 < oc1, "inner frame must be strictly nested"

    inner_border = np.zeros_like(frame_mask, dtype=bool)
    for r, c in _border_cells(inner):
        inner_border[r, c] = True
    assert np.all(~remaining | inner_border), "remaining frame pixels after removing the outer border must form the inner border"
    return outer, inner


def _scan_notches_on_side(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    frame_color: int,
    bg: int,
    orientation: str,
    frame_name: str,
    side: str,
) -> list[dict]:
    notches: list[dict] = []
    i = 0
    while i < len(coords):
        r, c = coords[i]
        value = int(grid[r, c])
        assert value in {int(bg), int(frame_color)}, "frame border may only contain frame-color cells or notch gaps"
        if value == int(frame_color):
            i += 1
            continue
        j = i
        while j < len(coords) and int(grid[coords[j]]) == int(bg):
            j += 1
        cells = coords[i:j]
        notches.append(
            {
                "cells": cells,
                "orientation": orientation,
                "length": len(cells),
                "signature": (orientation, len(cells)),
                "frame": frame_name,
                "side": side,
            }
        )
        i = j
    return notches


def _extract_notches(
    grid: np.ndarray, box: tuple[int, int, int, int], frame_color: int, bg: int, frame_name: str
) -> list[dict]:
    r0, c0, r1, c1 = box
    assert int(grid[r0, c0]) == int(frame_color)
    assert int(grid[r0, c1]) == int(frame_color)
    assert int(grid[r1, c0]) == int(frame_color)
    assert int(grid[r1, c1]) == int(frame_color)

    notches: list[dict] = []
    notches.extend(
        _scan_notches_on_side(
            grid,
            [(r0, c) for c in range(c0 + 1, c1)],
            frame_color,
            bg,
            "h",
            frame_name,
            "top",
        )
    )
    notches.extend(
        _scan_notches_on_side(
            grid,
            [(r1, c) for c in range(c0 + 1, c1)],
            frame_color,
            bg,
            "h",
            frame_name,
            "bottom",
        )
    )
    notches.extend(
        _scan_notches_on_side(
            grid,
            [(r, c0) for r in range(r0 + 1, r1)],
            frame_color,
            bg,
            "v",
            frame_name,
            "left",
        )
    )
    notches.extend(
        _scan_notches_on_side(
            grid,
            [(r, c1) for r in range(r0 + 1, r1)],
            frame_color,
            bg,
            "v",
            frame_name,
            "right",
        )
    )
    return notches


def _extract_bars(grid: np.ndarray, bg: int, frame_color: int) -> list[dict]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    bars: list[dict] = []
    seen_colors: set[int] = set()
    for comp in comps:
        color = int(comp["color"])
        if color == int(frame_color):
            continue
        assert color != int(bg)
        assert color not in seen_colors, "every scattered bar must have a distinct color"
        seen_colors.add(color)

        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = int(comp["area"])
        assert (h == 1) ^ (w == 1), "every fragment must be a 1-cell-thick straight bar"
        assert area == max(h, w), "bar component must fill its straight bounding box"
        orientation = "h" if h == 1 else "v"
        length = max(h, w)
        bars.append({"color": color, "orientation": orientation, "length": length, "signature": (orientation, length)})
    return bars


def _annulus_seed(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> tuple[int, int]:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    for r in range(or0 + 1, or1):
        for c in range(oc0 + 1, oc1):
            if not (ir0 <= r <= ir1 and ic0 <= c <= ic1):
                return r, c
    raise AssertionError("annulus must contain at least one background cell")


def _flood_fill(grid: np.ndarray, seed: tuple[int, int], target: int, color: int) -> None:
    if int(grid[seed]) != int(target):
        return
    h, w = grid.shape
    q = deque([seed])
    grid[seed] = int(color)
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == int(target):
                grid[nr, nc] = int(color)
                q.append((nr, nc))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Plug matching colored bars into the same-shaped gaps of two nested frames, then fill only the annulus with the leftover bar's color.

    Input structure:
    - A black background contains two nested one-cell-thick hollow rectangular frames that share one frame color.
    - Each frame has one or more straight border gaps, and every gap is a contiguous run on a single side.
    - Several scattered monochrome straight bars lie outside the frames; each bar is one cell thick and has a distinct color.

    Rule:
    - Treat the uniquely most common non-background color as the common frame color and recover the outer and inner frame bboxes.
    - Read every border gap as a notch, match it to the unique scattered bar with the same (orientation, length), and place that bar into the notch.
    - Remove the one unmatched scattered bar and use its color to flood-fill exactly the background region inside the outer frame but outside the closed inner frame.
    - Keep the frame color on every non-notch border cell, keep the inner interior black, and keep everything outside the outer frame black.

    Key insight:
    - The bar matching must check both length and orientation, and the inner frame must also be closed before the annulus can be isolated for filling.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame_color = _unique_most_common_non_bg(g, bg)

    outer, inner = _find_frame_boxes(g, frame_color)
    notches = _extract_notches(g, outer, frame_color, bg, "outer") + _extract_notches(g, inner, frame_color, bg, "inner")
    assert notches, "expected at least one border notch"
    assert {n["frame"] for n in notches} == {"outer", "inner"}, "both frames must contain notches"

    notch_sigs = [n["signature"] for n in notches]
    assert len(notch_sigs) == len(set(notch_sigs)), "all notch (orientation, length) pairs must be unique"

    bars = _extract_bars(g, bg, frame_color)
    assert len(bars) == len(notches) + 1, "there must be exactly one extra bar"
    bar_sigs = [b["signature"] for b in bars]
    assert len(bar_sigs) == len(set(bar_sigs)), "bars must also be unique by (orientation, length)"

    pair_to_bar = {bar["signature"]: bar for bar in bars}
    unmatched = [bar for bar in bars if bar["signature"] not in set(notch_sigs)]
    assert len(unmatched) == 1, "there must be exactly one unmatched leftover bar"
    leftover_color = int(unmatched[0]["color"])
    for notch in notches:
        assert notch["signature"] in pair_to_bar, "every notch must have exactly one matching bar"

    out = np.full_like(g, int(bg))
    _draw_border(out, outer, frame_color)
    _draw_border(out, inner, frame_color)
    for notch in notches:
        color = int(pair_to_bar[notch["signature"]]["color"])
        for r, c in notch["cells"]:
            out[r, c] = color

    seed = _annulus_seed(outer, inner)
    _flood_fill(out, seed, bg, leftover_color)
    return out


def _rects_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _choose_geometry(
    rng: np.random.Generator,
) -> tuple[int, int, tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    for _ in range(80):
        h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
        w = int(rng.integers(GRID_MIN, GRID_MAX + 1))

        max_outer_h = min(OUTER_SIZE_MAX, h - 2 * OUTER_MARGIN_MIN)
        max_outer_w = min(OUTER_SIZE_MAX, w - 2 * OUTER_MARGIN_MIN)
        if max_outer_h < OUTER_SIZE_MIN or max_outer_w < OUTER_SIZE_MIN:
            continue

        outer_h = int(rng.integers(OUTER_SIZE_MIN, max_outer_h + 1))
        outer_w = int(rng.integers(OUTER_SIZE_MIN, max_outer_w + 1))
        max_r0 = h - outer_h - OUTER_MARGIN_MIN
        max_c0 = w - outer_w - OUTER_MARGIN_MIN
        if max_r0 < OUTER_MARGIN_MIN or max_c0 < OUTER_MARGIN_MIN:
            continue
        r0 = int(rng.integers(OUTER_MARGIN_MIN, max_r0 + 1))
        c0 = int(rng.integers(OUTER_MARGIN_MIN, max_c0 + 1))
        outer = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)

        for _ in range(80):
            gt = int(rng.integers(FRAME_GAP_MIN, FRAME_GAP_MAX + 1))
            gb = int(rng.integers(FRAME_GAP_MIN, FRAME_GAP_MAX + 1))
            gl = int(rng.integers(FRAME_GAP_MIN, FRAME_GAP_MAX + 1))
            gr = int(rng.integers(FRAME_GAP_MIN, FRAME_GAP_MAX + 1))
            inner_h = outer_h - gt - gb
            inner_w = outer_w - gl - gr
            if inner_h < INNER_SIZE_MIN or inner_w < INNER_SIZE_MIN:
                continue
            inner = (r0 + gt, c0 + gl, r0 + gt + inner_h - 1, c0 + gl + inner_w - 1)
            return h, w, outer, inner
    return None


def _choose_sides(rng: np.random.Generator) -> list[tuple[str, str, str]]:
    horizontal = [
        ("outer", "top", "h"),
        ("outer", "bottom", "h"),
        ("inner", "top", "h"),
        ("inner", "bottom", "h"),
    ]
    vertical = [
        ("outer", "left", "v"),
        ("outer", "right", "v"),
        ("inner", "left", "v"),
        ("inner", "right", "v"),
    ]

    hv_choices = [(1, 2), (2, 1), (2, 2)]
    for _ in range(80):
        h_count, v_count = hv_choices[int(rng.integers(0, len(hv_choices)))]
        h_idx = rng.choice(len(horizontal), size=h_count, replace=False)
        v_idx = rng.choice(len(vertical), size=v_count, replace=False)
        chosen = [horizontal[int(i)] for i in np.atleast_1d(h_idx)] + [vertical[int(i)] for i in np.atleast_1d(v_idx)]
        if {frame for frame, _side, _orient in chosen} == {"outer", "inner"}:
            return chosen
    raise AssertionError("failed to choose a valid side set")


def _length_limit(frame: str, orientation: str, outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> int:
    box = outer if frame == "outer" else inner
    r0, c0, r1, c1 = box
    side_span = (c1 - c0 - 1) if orientation == "h" else (r1 - r0 - 1)
    return min(NOTCH_LEN_MAX, side_span)


def _choose_lengths(
    rng: np.random.Generator,
    sides: list[tuple[str, str, str]],
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> tuple[dict[tuple[str, str, str], int], tuple[str, int]] | None:
    for _ in range(120):
        lengths: dict[tuple[str, str, str], int] = {}
        used_by_orientation = {"h": set(), "v": set()}
        ok = True
        order = rng.permutation(len(sides))
        for idx in np.atleast_1d(order):
            side = sides[int(idx)]
            frame, _which, orient = side
            max_len = _length_limit(frame, orient, outer, inner)
            candidates = [
                length
                for length in range(NOTCH_LEN_MIN, max_len + 1)
                if length not in used_by_orientation[orient]
            ]
            if not candidates:
                ok = False
                break
            length = int(candidates[int(rng.integers(0, len(candidates)))])
            lengths[side] = length
            used_by_orientation[orient].add(length)
        if not ok:
            continue

        h_only = sorted(used_by_orientation["h"] - used_by_orientation["v"])
        v_only = sorted(used_by_orientation["v"] - used_by_orientation["h"])
        leftover_candidates = [("v", length) for length in h_only] + [("h", length) for length in v_only]
        if not leftover_candidates:
            continue
        leftover = leftover_candidates[int(rng.integers(0, len(leftover_candidates)))]
        return lengths, leftover
    return None


def _notch_cells_for_side(box: tuple[int, int, int, int], side: str, length: int, start: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    if side == "top":
        return [(r0, c) for c in range(start, start + length)]
    if side == "bottom":
        return [(r1, c) for c in range(start, start + length)]
    if side == "left":
        return [(r, c0) for r in range(start, start + length)]
    if side == "right":
        return [(r, c1) for r in range(start, start + length)]
    raise AssertionError(f"unknown side: {side}")


def _choose_notches(
    rng: np.random.Generator,
    sides: list[tuple[str, str, str]],
    lengths: dict[tuple[str, str, str], int],
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> list[dict]:
    notches: list[dict] = []
    for spec in sides:
        frame, side, orient = spec
        box = outer if frame == "outer" else inner
        r0, c0, r1, c1 = box
        length = lengths[spec]
        if orient == "h":
            lo = c0 + 1
            hi = c1 - length
        else:
            lo = r0 + 1
            hi = r1 - length
        start = int(rng.integers(lo, hi + 1))
        cells = _notch_cells_for_side(box, side, length, start)
        notches.append(
            {
                "frame": frame,
                "side": side,
                "orientation": orient,
                "length": length,
                "signature": (orient, length),
                "cells": cells,
            }
        )
    assert len({n["signature"] for n in notches}) == len(notches)
    return notches


def _render_input_base(
    shape: tuple[int, int],
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
    frame_color: int,
    notches: list[dict],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    _draw_border(grid, outer, frame_color)
    _draw_border(grid, inner, frame_color)
    for notch in notches:
        for r, c in notch["cells"]:
            grid[r, c] = BG
    return grid


def _render_expected(
    shape: tuple[int, int],
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
    frame_color: int,
    notches: list[dict],
    matched_colors: dict[tuple[str, int], int],
    leftover_color: int,
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    _draw_border(out, outer, frame_color)
    _draw_border(out, inner, frame_color)
    for notch in notches:
        color = int(matched_colors[notch["signature"]])
        for r, c in notch["cells"]:
            out[r, c] = color
    _flood_fill(out, _annulus_seed(outer, inner), BG, leftover_color)
    return out


def _place_bars(
    rng: np.random.Generator,
    grid: np.ndarray,
    outer: tuple[int, int, int, int],
    bars: list[dict],
) -> bool:
    h, w = grid.shape
    placed_rects: list[tuple[int, int, int, int]] = []
    order = rng.permutation(len(bars))
    for idx in np.atleast_1d(order):
        bar = bars[int(idx)]
        patch_h = 1 if bar["orientation"] == "h" else bar["length"]
        patch_w = bar["length"] if bar["orientation"] == "h" else 1
        candidates = []
        for r0 in range(0, h - patch_h + 1):
            for c0 in range(0, w - patch_w + 1):
                rect = (r0, c0, r0 + patch_h - 1, c0 + patch_w - 1)
                if _rects_too_close(rect, outer, BAR_GAP):
                    continue
                if any(_rects_too_close(rect, prev, BAR_GAP) for prev in placed_rects):
                    continue
                candidates.append(rect)
        if not candidates:
            return False
        rect = candidates[int(rng.integers(0, len(candidates)))]
        placed_rects.append(rect)
        r0, c0, r1, c1 = rect
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(bar["color"])
    return True


def _semantic_ok(
    inp: np.ndarray,
    expected: np.ndarray,
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
    frame_color: int,
    notches: list[dict],
    bars: list[dict],
    leftover_color: int,
) -> bool:
    bg = get_bg(inp)
    if int(bg) != BG:
        return False

    try:
        parsed_frame_color = _unique_most_common_non_bg(inp, bg)
        if parsed_frame_color != int(frame_color):
            return False
        parsed_outer, parsed_inner = _find_frame_boxes(inp, parsed_frame_color)
        if parsed_outer != outer or parsed_inner != inner:
            return False
        parsed_notches = _extract_notches(inp, parsed_outer, parsed_frame_color, bg, "outer") + _extract_notches(
            inp, parsed_inner, parsed_frame_color, bg, "inner"
        )
        if {n["signature"] for n in parsed_notches} != {n["signature"] for n in notches}:
            return False
        parsed_bars = _extract_bars(inp, bg, parsed_frame_color)
        if {b["signature"] for b in parsed_bars} != {b["signature"] for b in bars}:
            return False
        solved = main(inp)
    except Exception:
        return False

    if not np.array_equal(solved, expected):
        return False

    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    annulus_mask = np.zeros(inp.shape, dtype=bool)
    annulus_mask[or0 + 1 : or1, oc0 + 1 : oc1] = True
    annulus_mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = False
    if not annulus_mask.any():
        return False
    if np.any(expected[annulus_mask] != int(leftover_color)):
        return False
    if np.any(expected[ir0 + 1 : ir1, ic0 + 1 : ic1] != BG):
        return False
    outside = np.ones(inp.shape, dtype=bool)
    outside[or0 : or1 + 1, oc0 : oc1 + 1] = False
    if np.any(expected[outside] != BG):
        return False
    return True


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    geom = _choose_geometry(rng)
    if geom is None:
        return None
    h, w, outer, inner = geom

    sides = _choose_sides(rng)
    length_plan = _choose_lengths(rng, sides, outer, inner)
    if length_plan is None:
        return None
    lengths, leftover_signature = length_plan
    notches = _choose_notches(rng, sides, lengths, outer, inner)

    n_bars = len(notches) + 1
    palette = rng.choice(np.arange(1, 10), size=n_bars + 1, replace=False).astype(int).tolist()
    frame_color = int(palette[0])
    bar_colors = list(map(int, palette[1:]))
    rng.shuffle(bar_colors)

    matched_colors = {notch["signature"]: color for notch, color in zip(notches, bar_colors[:-1])}
    leftover_color = int(bar_colors[-1])
    bars = [
        {"orientation": notch["orientation"], "length": notch["length"], "signature": notch["signature"], "color": matched_colors[notch["signature"]]}
        for notch in notches
    ]
    bars.append(
        {
            "orientation": leftover_signature[0],
            "length": int(leftover_signature[1]),
            "signature": (leftover_signature[0], int(leftover_signature[1])),
            "color": leftover_color,
        }
    )

    inp = _render_input_base((h, w), outer, inner, frame_color, notches)
    if not _place_bars(rng, inp, outer, bars):
        return None

    expected = _render_expected((h, w), outer, inner, frame_color, notches, matched_colors, leftover_color)
    if not _semantic_ok(inp, expected, outer, inner, frame_color, notches, bars, leftover_color):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-seal ARC input")
