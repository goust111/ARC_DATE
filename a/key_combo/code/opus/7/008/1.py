from __future__ import annotations

import functools
import random
from collections import Counter

import numpy as np


Box = tuple[int, int, int, int]

BG = 0
COLORS = list(range(1, 10))
RECT_SIDES = (2, 3)
GRID_SIDE_RANGE = (18, 25)
ALIGNED_GAP_RANGE = (3, 8)
DIAG_GAP_RANGE = (2, 7)
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(v) for v in np.unique(grid) if int(v) != bg)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _component_boxes(grid: np.ndarray, color: int) -> list[Box]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    boxes: list[Box] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            box = (min(rs), min(cs), max(rs), max(cs))
            r0, c0, r1, c1 = box
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            assert len(cells) == area, "each object must be a solid rectangle"
            assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be filled"
            boxes.append(box)
    return boxes


def _box_cells(box: Box) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _expanded_cells(cells: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _aligned_kind(a: Box, b: Box) -> str | None:
    same_rows = (a[0], a[2]) == (b[0], b[2])
    same_cols = (a[1], a[3]) == (b[1], b[3])
    if same_rows ^ same_cols:
        return "H" if same_rows else "V"
    return None


def _diag_endpoints(a: Box, b: Box) -> tuple[tuple[int, int], tuple[int, int]]:
    upper, lower = sorted((a, b), key=lambda box: (box[0], box[1]))
    ur0, uc0, ur1, uc1 = upper
    lr0, lc0, _lr1, lc1 = lower
    assert ur1 < lr0, "diagonal pair must be vertically separated"
    if uc1 < lc0:
        start = (ur1, uc1)
        end = (lr0, lc0)
    elif lc1 < uc0:
        start = (ur1, uc0)
        end = (lr0, lc1)
    else:
        raise AssertionError("diagonal pair must be horizontally separated")
    dr = end[0] - start[0]
    dc = end[1] - start[1]
    assert dr > 0 and abs(dr) == abs(dc), "facing corners must lie on a 45-degree diagonal"
    return start, end


def _diag_line_cells(a: Box, b: Box) -> list[tuple[int, int]]:
    start, end = _diag_endpoints(a, b)
    dc = 1 if end[1] > start[1] else -1
    out = []
    r = start[0] + 1
    c = start[1] + dc
    while (r, c) != end:
        out.append((r, c))
        r += 1
        c += dc
    assert out, "diagonal connector must contain at least one pixel"
    return out


def _gap_cells(a: Box, b: Box, orient: str) -> list[tuple[int, int]]:
    if orient == "H":
        left, right = sorted((a, b), key=lambda box: box[1])
        r0, _c0, r1, lc1 = left
        _rr0, rc0, _rr1, _rc1 = right
        return [(r, c) for r in range(r0, r1 + 1) for c in range(lc1 + 1, rc0)]
    top, bottom = sorted((a, b), key=lambda box: box[0])
    _tr0, c0, tr1, c1 = top
    br0, _bc0, _br1, _bc1 = bottom
    return [(r, c) for r in range(tr1 + 1, br0) for c in range(c0, c1 + 1)]


def _half_fill_cells(a: Box, b: Box, orient: str) -> list[tuple[int, int]]:
    if orient == "H":
        left, right = sorted((a, b), key=lambda box: box[1])
        r0, _c0, r1, lc1 = left
        _rr0, rc0, _rr1, _rc1 = right
        gap = rc0 - lc1 - 1
        span = gap // 2
        out = []
        for r in range(r0, r1 + 1):
            out.extend((r, lc1 + 1 + i) for i in range(span))
            out.extend((r, rc0 - span + i) for i in range(span))
        return out
    top, bottom = sorted((a, b), key=lambda box: box[0])
    _tr0, c0, tr1, c1 = top
    br0, _bc0, _br1, _bc1 = bottom
    gap = br0 - tr1 - 1
    span = gap // 2
    out = []
    for c in range(c0, c1 + 1):
        out.extend((tr1 + 1 + i, c) for i in range(span))
        out.extend((br0 - span + i, c) for i in range(span))
    return out


def _paint_diagonal(out: np.ndarray, color: int, a: Box, b: Box) -> list[tuple[int, int]]:
    line = _diag_line_cells(a, b)
    for r, c in line:
        assert int(out[r, c]) == BG
        out[r, c] = color
    return line


def _fill_aligned(out: np.ndarray, wall_mask: np.ndarray, color: int, a: Box, b: Box, orient: str) -> None:
    if orient == "H":
        left, right = sorted((a, b), key=lambda box: box[1])
        r0, _c0, r1, lc1 = left
        _rr0, rc0, _rr1, _rc1 = right
        gap = rc0 - lc1 - 1
        span = gap // 2
        for r in range(r0, r1 + 1):
            for i in range(span):
                c = lc1 + 1 + i
                if wall_mask[r, c]:
                    break
                out[r, c] = color
            for i in range(span):
                c = rc0 - 1 - i
                if wall_mask[r, c]:
                    break
                out[r, c] = color
        return

    top, bottom = sorted((a, b), key=lambda box: box[0])
    _tr0, c0, tr1, c1 = top
    br0, _bc0, _br1, _bc1 = bottom
    gap = br0 - tr1 - 1
    span = gap // 2
    for c in range(c0, c1 + 1):
        for i in range(span):
            r = tr1 + 1 + i
            if wall_mask[r, c]:
                break
            out[r, c] = color
        for i in range(span):
            r = br0 - 1 - i
            if wall_mask[r, c]:
                break
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw diagonal connectors for diagonally offset rectangle pairs, then half-fill aligned gaps while those connectors act as walls.

    Input structure:
    - A black background contains 3-4 foreground colors.
    - Each foreground color appears as exactly two separated solid rectangles of size 2-3 by 2-3.
    - Same-color rectangle pairs are either axis-aligned by sharing a row span or column span, or diagonally offset with facing corners on a 45-degree line.

    Rule:
    - For each color, detect its two rectangles and classify the pair as axis-aligned or diagonal.
    - First draw the one-pixel diagonal segment between the facing corners of every diagonal pair, excluding the rectangle endpoint pixels.
    - Then, for every axis-aligned pair, fill each half of the gap from the two rectangles toward the center across the pair's shared span; if the gap length is odd, leave the exact middle line black.
    - During that gap fill, stop each per-row or per-column ray immediately when it reaches a previously drawn diagonal connector, and never overwrite the connector pixels.

    Key insight:
    - The same-color pairing is only the first step: diagonal pairs and aligned pairs follow different rules, and the diagonal result must be computed first because it changes where another color's half-gap fill is allowed to continue.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    assert bg == BG, "expected black background"
    out = g.copy()

    aligned: list[tuple[int, Box, Box, str]] = []
    wall_mask = np.zeros(g.shape, dtype=bool)

    for color in _colors_present(g, bg):
        boxes = _component_boxes(g, color)
        assert len(boxes) == 2, "each foreground color must have exactly two rectangles"
        a, b = boxes
        orient = _aligned_kind(a, b)
        if orient is not None:
            aligned.append((color, a, b, orient))
            continue
        line = _paint_diagonal(out, color, a, b)
        for r, c in line:
            wall_mask[r, c] = True

    for color, a, b, orient in aligned:
        _fill_aligned(out, wall_mask, color, a, b, orient)
    return out


def _empty_state(shape: tuple[int, int]) -> dict:
    h, w = shape
    return {
        "actual": np.zeros((h, w), dtype=bool),
        "touch": np.zeros((h, w), dtype=bool),
        "gap": np.zeros((h, w), dtype=bool),
        "aligned": [],
        "diagonals": [],
        "pairs": [],
    }


def _mark(mask: np.ndarray, cells: list[tuple[int, int]] | set[tuple[int, int]]) -> None:
    for r, c in cells:
        mask[r, c] = True


def _cells_hit(mask: np.ndarray, cells: list[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    return any(mask[r, c] for r, c in cells)


def _valid_aligned_candidate(state: dict, cand: dict) -> bool:
    shape = state["actual"].shape
    expanded = _expanded_cells(cand["rect_cells"], shape)
    if _cells_hit(state["touch"], cand["rect_cells"]):
        return False
    if _cells_hit(state["gap"], cand["rect_cells"]) or _cells_hit(state["gap"], expanded):
        return False
    if _cells_hit(state["actual"], cand["gap_cells"]) or _cells_hit(state["gap"], cand["gap_cells"]):
        return False
    return True


def _valid_diagonal_candidate(state: dict, cand: dict, cross_gap_idx: int | None, require_cross: bool) -> bool:
    shape = state["actual"].shape
    rect_expanded = _expanded_cells(cand["rect_cells"], shape)
    if _cells_hit(state["touch"], cand["rect_cells"] + cand["line_cells"]):
        return False
    if _cells_hit(state["gap"], cand["rect_cells"]) or _cells_hit(state["gap"], rect_expanded):
        return False

    hit_gaps = []
    effective_hit = False
    line_set = set(cand["line_cells"])
    for idx, aligned in enumerate(state["aligned"]):
        gap_hit = bool(line_set & aligned["gap_set"])
        fill_hit = bool(line_set & aligned["fill_set"])
        if gap_hit:
            hit_gaps.append(idx)
        if cross_gap_idx is not None and idx == cross_gap_idx and fill_hit:
            effective_hit = True

    if require_cross:
        return hit_gaps == [cross_gap_idx] and effective_hit
    return not hit_gaps


def _accept_aligned(state: dict, color: int, cand: dict) -> None:
    shape = state["actual"].shape
    expanded = _expanded_cells(cand["rect_cells"], shape)
    _mark(state["actual"], cand["rect_cells"])
    _mark(state["touch"], expanded)
    _mark(state["gap"], cand["gap_cells"])
    meta = {
        "kind": "aligned",
        "color": color,
        "boxes": cand["boxes"],
        "orient": cand["orient"],
        "gap_set": set(cand["gap_cells"]),
        "fill_set": set(cand["fill_cells"]),
    }
    state["aligned"].append(meta)
    state["pairs"].append(meta)


def _accept_diagonal(state: dict, color: int, cand: dict) -> None:
    shape = state["actual"].shape
    all_cells = cand["rect_cells"] + cand["line_cells"]
    expanded = _expanded_cells(all_cells, shape)
    _mark(state["actual"], all_cells)
    _mark(state["touch"], expanded)
    meta = {
        "kind": "diag",
        "color": color,
        "boxes": cand["boxes"],
        "line_cells": list(cand["line_cells"]),
    }
    state["diagonals"].append(meta)
    state["pairs"].append(meta)


def _aligned_candidates(shape: tuple[int, int], orient: str) -> list[dict]:
    return list(_aligned_candidates_cached(shape[0], shape[1], orient))


@functools.lru_cache(maxsize=None)
def _aligned_candidates_cached(h: int, w: int, orient: str) -> tuple[dict, ...]:
    return tuple()


def _diagonal_candidates(shape: tuple[int, int]) -> list[dict]:
    return list(_diagonal_candidates_cached(shape[0], shape[1]))


@functools.lru_cache(maxsize=None)
def _diagonal_candidates_cached(h: int, w: int) -> tuple[dict, ...]:
    return tuple()


def _random_aligned_candidate(shape: tuple[int, int], orient: str) -> dict | None:
    h, w = shape
    if orient == "H":
        height = random.choice(RECT_SIDES)
        width_a = random.choice(RECT_SIDES)
        width_b = random.choice(RECT_SIDES)
        gap = random.randint(*ALIGNED_GAP_RANGE)
        max_c0 = w - width_a - width_b - gap - 2
        if max_c0 < 1 or h - height - 1 < 1:
            return None
        r0 = random.randint(1, h - height - 1)
        c0 = random.randint(1, max_c0)
        a = (r0, c0, r0 + height - 1, c0 + width_a - 1)
        b = (r0, a[3] + gap + 1, r0 + height - 1, a[3] + gap + width_b)
        return {
            "orient": "H",
            "boxes": (a, b),
            "rect_cells": _box_cells(a) + _box_cells(b),
            "gap_cells": _gap_cells(a, b, "H"),
            "fill_cells": _half_fill_cells(a, b, "H"),
        }

    width = random.choice(RECT_SIDES)
    height_a = random.choice(RECT_SIDES)
    height_b = random.choice(RECT_SIDES)
    gap = random.randint(*ALIGNED_GAP_RANGE)
    max_r0 = h - height_a - height_b - gap - 2
    if max_r0 < 1 or w - width - 1 < 1:
        return None
    c0 = random.randint(1, w - width - 1)
    r0 = random.randint(1, max_r0)
    a = (r0, c0, r0 + height_a - 1, c0 + width - 1)
    b = (a[2] + gap + 1, c0, a[2] + gap + height_b, c0 + width - 1)
    return {
        "orient": "V",
        "boxes": (a, b),
        "rect_cells": _box_cells(a) + _box_cells(b),
        "gap_cells": _gap_cells(a, b, "V"),
        "fill_cells": _half_fill_cells(a, b, "V"),
    }


def _random_diagonal_candidate(shape: tuple[int, int]) -> dict | None:
    h, w = shape
    h1 = random.choice(RECT_SIDES)
    w1 = random.choice(RECT_SIDES)
    h2 = random.choice(RECT_SIDES)
    w2 = random.choice(RECT_SIDES)
    gap = random.randint(*DIAG_GAP_RANGE)
    delta = gap + 1
    direction = random.choice((-1, 1))
    max_ur0 = h - h1 - delta - h2 - 1
    if max_ur0 < 1:
        return None
    ur0 = random.randint(1, max_ur0)
    ur1 = ur0 + h1 - 1

    if direction == 1:
        max_uc0 = w - w1 - delta - w2 - 1
        if max_uc0 < 1:
            return None
        uc0 = random.randint(1, max_uc0)
        uc1 = uc0 + w1 - 1
        lr0 = ur1 + delta
        lc0 = uc1 + delta
        lower = (lr0, lc0, lr0 + h2 - 1, lc0 + w2 - 1)
    else:
        min_uc0 = delta + 1
        max_uc0 = w - w1 - 1
        if min_uc0 > max_uc0:
            return None
        uc0 = random.randint(min_uc0, max_uc0)
        uc1 = uc0 + w1 - 1
        lr0 = ur1 + delta
        end_c = uc0 - delta
        lower = (lr0, end_c - w2 + 1, lr0 + h2 - 1, end_c)

    upper = (ur0, uc0, ur1, uc1)
    if min(*lower) < 1 or lower[2] >= h - 1 or lower[3] >= w - 1:
        return None
    return {
        "boxes": (upper, lower),
        "rect_cells": _box_cells(upper) + _box_cells(lower),
        "line_cells": _diag_line_cells(upper, lower),
    }


def _random_crossing_diagonal_candidate(state: dict, gap_idx: int) -> dict | None:
    h, w = state["actual"].shape
    aligned = state["aligned"][gap_idx]
    fill_cells = list(aligned["fill_set"])
    if not fill_cells:
        return None
    xr, xc = random.choice(fill_cells)
    gap = random.randint(*DIAG_GAP_RANGE)
    pos = random.randint(0, gap - 1)
    direction = random.choice((-1, 1))
    h1 = random.choice(RECT_SIDES)
    w1 = random.choice(RECT_SIDES)
    h2 = random.choice(RECT_SIDES)
    w2 = random.choice(RECT_SIDES)

    if direction == 1:
        start = (xr - pos - 1, xc - pos - 1)
        end = (xr + (gap - pos), xc + (gap - pos))
        upper = (start[0] - h1 + 1, start[1] - w1 + 1, start[0], start[1])
        lower = (end[0], end[1], end[0] + h2 - 1, end[1] + w2 - 1)
    else:
        start = (xr - pos - 1, xc + pos + 1)
        end = (xr + (gap - pos), xc - (gap - pos))
        upper = (start[0] - h1 + 1, start[1], start[0], start[1] + w1 - 1)
        lower = (end[0], end[1] - w2 + 1, end[0] + h2 - 1, end[1])

    if min(*upper, *lower) < 1:
        return None
    if upper[2] >= h - 1 or upper[3] >= w - 1:
        return None
    if lower[2] >= h - 1 or lower[3] >= w - 1:
        return None
    return {
        "boxes": (upper, lower),
        "rect_cells": _box_cells(upper) + _box_cells(lower),
        "line_cells": _diag_line_cells(upper, lower),
    }


def _render_input_from_plan(shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for pair in pairs:
        for box in pair["boxes"]:
            r0, c0, r1, c1 = box
            grid[r0 : r1 + 1, c0 : c1 + 1] = pair["color"]
    return grid


def _render_output_from_plan(shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    out = _render_input_from_plan(shape, pairs)
    wall_mask = np.zeros(shape, dtype=bool)
    for pair in pairs:
        if pair["kind"] != "diag":
            continue
        for r, c in pair["line_cells"]:
            out[r, c] = pair["color"]
            wall_mask[r, c] = True
    for pair in pairs:
        if pair["kind"] != "aligned":
            continue
        _fill_aligned(out, wall_mask, pair["color"], pair["boxes"][0], pair["boxes"][1], pair["orient"])
    return out


def _render_unblocked_fill(shape: tuple[int, int], pair: dict) -> np.ndarray:
    out = _render_input_from_plan(shape, [pair])
    wall_mask = np.zeros(shape, dtype=bool)
    _fill_aligned(out, wall_mask, pair["color"], pair["boxes"][0], pair["boxes"][1], pair["orient"])
    return out


def _semantic_checks(shape: tuple[int, int], state: dict, inp: np.ndarray, expected: np.ndarray) -> bool:
    aligned = state["aligned"]
    diagonals = state["diagonals"]
    if len(aligned) < 2 or len(diagonals) < 1:
        return False

    crossed = aligned[0]
    plain = aligned[1]
    line_union = set()
    for pair in diagonals:
        line_union.update(pair["line_cells"])

    if not (crossed["fill_set"] & line_union):
        return False
    if plain["fill_set"] & line_union:
        return False

    crossed_only = _render_unblocked_fill(shape, crossed)
    crossed_blocked = expected.copy()
    mask = np.zeros(shape, dtype=bool)
    for r, c in crossed["fill_set"]:
        mask[r, c] = True
    if np.array_equal(crossed_only[mask], crossed_blocked[mask]):
        return False

    plain_only = _render_unblocked_fill(shape, plain)
    plain_mask = np.zeros(shape, dtype=bool)
    for r, c in plain["fill_set"]:
        plain_mask[r, c] = True
    if not np.array_equal(plain_only[plain_mask], expected[plain_mask]):
        return False

    if np.array_equal(inp, expected):
        return False
    if len(np.unique(inp)) < 4:
        return False
    return True


def _pick_aligned(state: dict, color: int, orient: str) -> bool:
    for _ in range(600):
        cand = _random_aligned_candidate(state["actual"].shape, orient)
        if cand is None:
            continue
        if _valid_aligned_candidate(state, cand):
            _accept_aligned(state, color, cand)
            return True
    return False


def _pick_diagonal(state: dict, color: int, cross_gap_idx: int | None, require_cross: bool) -> bool:
    budget = 1200 if require_cross else 800
    for _ in range(budget):
        cand = _random_crossing_diagonal_candidate(state, cross_gap_idx) if require_cross else _random_diagonal_candidate(state["actual"].shape)
        if cand is None:
            continue
        if _valid_diagonal_candidate(state, cand, cross_gap_idx, require_cross):
            _accept_diagonal(state, color, cand)
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    n_pairs = random.choice((3, 4))
    colors = random.sample(COLORS, n_pairs)
    state = _empty_state((h, w))

    cross_orient = random.choice(("H", "V"))
    plain_orient = random.choice(("H", "V"))
    if not _pick_aligned(state, colors[0], cross_orient):
        return None
    if not _pick_diagonal(state, colors[2], cross_gap_idx=0, require_cross=True):
        return None
    if not _pick_aligned(state, colors[1], plain_orient):
        return None
    if n_pairs == 4 and not _pick_diagonal(state, colors[3], cross_gap_idx=None, require_cross=False):
        return None

    inp = _render_input_from_plan((h, w), state["pairs"])
    expected = _render_output_from_plan((h, w), state["pairs"])
    if not _semantic_checks((h, w), state, inp, expected):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
