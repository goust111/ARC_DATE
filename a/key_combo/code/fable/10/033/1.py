from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
ORANGE = 7
BACKGROUND_CHOICES = [1, 2, 3, 4, 5, 6, 8, 9]

GRID_MIN = 10
GRID_MAX = 18
MARGIN_MIN = 1
MARGIN_MAX = 3
MAX_EXTRA_RECTS = 3
GEN_ATTEMPTS = 600


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touching_side(box: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    r0, c0, r1, c1 = box
    h, w = shape
    sides = []
    if r0 == 0:
        sides.append("top")
    if r1 == h - 1:
        sides.append("bottom")
    if c0 == 0:
        sides.append("left")
    if c1 == w - 1:
        sides.append("right")
    assert len(sides) == 1, "each rectangle must touch exactly one outer edge"
    return sides[0]


def _parse_rectangles(grid: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    colors = sorted(set(map(int, g.ravel())))
    assert BLACK in colors
    bg_colors = [c for c in colors if c != BLACK]
    assert len(bg_colors) == 1, "input must use one uniform non-black background"
    bg = bg_colors[0]

    rects: list[dict[str, int]] = []
    for comp in _components4(g == BLACK):
        r0, c0, r1, c1 = _bbox(comp)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert len(comp) == area, "black component must be a solid rectangle"
        side = _touching_side((r0, c0, r1, c1), g.shape)
        rects.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "side": side,
            }
        )
    assert rects, "expected at least one edge rectangle"
    return bg, rects


def _destination(rect: dict[str, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
    rect_h = r1 - r0 + 1
    rect_w = c1 - c0 + 1
    side = rect["side"]
    if side == "top":
        return h - rect_h, c0, h - 1, c1
    if side == "bottom":
        return 0, c0, rect_h - 1, c1
    if side == "left":
        return r0, w - rect_w, r1, w - 1
    return r0, 0, r1, rect_w - 1


def _render_from_rectangles(
    shape: tuple[int, int], bg: int, rects: list[dict[str, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    coverage = np.zeros((h, w), dtype=int)
    dest_mask = np.zeros((h, w), dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
        if rect["side"] in ("top", "bottom"):
            coverage[:, c0 : c1 + 1] += 1
        else:
            coverage[r0 : r1 + 1, :] += 1
        dr0, dc0, dr1, dc1 = _destination(rect, shape)
        dest_mask[dr0 : dr1 + 1, dc0 : dc1 + 1] = True

    out = np.full((h, w), bg, dtype=int)
    out[coverage % 2 == 1] = ORANGE
    out[dest_mask] = BLACK
    return out, coverage, dest_mask


def main(grid: np.ndarray) -> np.ndarray:
    """Slide all edge black rectangles to the opposite side and color only the cells swept an odd number of times.

    Input structure:
    - The grid has a single uniform background color plus one or more black solid rectangles.
    - Every black rectangle is 4-connected and touches exactly one outer edge.
    - Top/bottom rectangles define vertical sweeps in their own columns, and left/right rectangles define horizontal sweeps in their own rows.

    Rule:
    - Move every rectangle straight to the opposite edge while keeping its size and in-band position.
    - Count, for each cell, how many swept row-stripes or column-stripes cover it.
    - Color cells orange iff that coverage count is odd; leave even-covered cells as background.
    - Finally stamp every destination rectangle in black on top of the parity trail.

    Key insight:
    - The trail is a symmetric-difference composition of all sweep stripes, not a union, so overlap zones cancel when they are covered an even number of times.
    """
    bg, rects = _parse_rectangles(grid)
    out, _coverage, _dest_mask = _render_from_rectangles(grid.shape, bg, rects)
    return out


def _make_rect(side: str, band0: int, band1: int, depth: int, shape: tuple[int, int]) -> dict[str, int]:
    h, w = shape
    if side == "top":
        return {"side": side, "r0": 0, "c0": band0, "r1": depth - 1, "c1": band1}
    if side == "bottom":
        return {"side": side, "r0": h - depth, "c0": band0, "r1": h - 1, "c1": band1}
    if side == "left":
        return {"side": side, "r0": band0, "c0": 0, "r1": band1, "c1": depth - 1}
    return {"side": side, "r0": band0, "c0": w - depth, "r1": band1, "c1": w - 1}


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _rect_box(rect: dict[str, int]) -> tuple[int, int, int, int]:
    return rect["r0"], rect["c0"], rect["r1"], rect["c1"]


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _choose_partial_overlap(lo: int, hi: int, min_len: int, max_len: int) -> tuple[tuple[int, int], tuple[int, int]]:
    size = hi - lo + 1
    while True:
        w1 = random.randint(min_len, min(max_len, size - 1))
        w2 = random.randint(min_len, min(max_len, size - 1))
        ov = random.randint(1, min(w1, w2) - 1)
        union = w1 + w2 - ov
        if union > size:
            continue
        start = random.randint(lo, hi - union + 1)
        a0, a1 = start, start + w1 - 1
        b0, b1 = start + w1 - ov, start + union - 1
        if random.random() < 0.5:
            return (a0, a1), (b0, b1)
        return (b0, b1), (a0, a1)


def _can_add_rect(
    rect: dict[str, int], shape: tuple[int, int], rects: list[dict[str, int]]
) -> bool:
    box = _rect_box(rect)
    dst = _destination(rect, shape)
    rect_h = rect["r1"] - rect["r0"] + 1
    rect_w = rect["c1"] - rect["c0"] + 1
    if rect_h * rect_w < 2:
        return False

    for other in rects:
        other_box = _rect_box(other)
        other_dst = _destination(other, shape)
        if _boxes_overlap(box, other_box) or _boxes_overlap(dst, other_dst):
            return False
        if rect["side"] in ("top", "bottom") and other["side"] in ("top", "bottom"):
            if rect["side"] == other["side"] and _intervals_overlap(rect["c0"], rect["c1"], other["c0"], other["c1"]):
                return False
        if rect["side"] in ("left", "right") and other["side"] in ("left", "right"):
            if rect["side"] == other["side"] and _intervals_overlap(rect["r0"], rect["r1"], other["r0"], other["r1"]):
                return False
    return True


def _build_input_grid(shape: tuple[int, int], bg: int, rects: list[dict[str, int]]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for rect in rects:
        grid[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = BLACK
    return grid


def _union_render(shape: tuple[int, int], bg: int, rects: list[dict[str, int]]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    for rect in rects:
        if rect["side"] in ("top", "bottom"):
            out[:, rect["c0"] : rect["c1"] + 1] = ORANGE
        else:
            out[rect["r0"] : rect["r1"] + 1, :] = ORANGE
    for rect in rects:
        dr0, dc0, dr1, dc1 = _destination(rect, shape)
        out[dr0 : dr1 + 1, dc0 : dc1 + 1] = BLACK
    return out


def _remove_one(rects: list[dict[str, int]], idx: int) -> list[dict[str, int]]:
    return [rect for j, rect in enumerate(rects) if j != idx]


def _validate_semantics(grid: np.ndarray, bg: int, rects: list[dict[str, int]]) -> bool:
    try:
        parsed_bg, parsed_rects = _parse_rectangles(grid)
    except AssertionError:
        return False
    if parsed_bg != bg or len(parsed_rects) != len(rects):
        return False

    out, coverage, dest_mask = _render_from_rectangles(grid.shape, bg, rects)
    if not np.array_equal(main(grid), out):
        return False
    if np.array_equal(grid, out):
        return False
    if not np.any(out == ORANGE):
        return False
    if not np.any(out == bg):
        return False
    if np.array_equal(out, _union_render(grid.shape, bg, rects)):
        return False

    visible_even = (coverage > 0) & (coverage % 2 == 0) & (~dest_mask)
    visible_odd = (coverage % 2 == 1) & (~dest_mask)
    if not np.any(visible_even):
        return False
    if not np.any(visible_odd):
        return False

    sides = {rect["side"] for rect in rects}
    if not {"top", "bottom", "left", "right"}.issubset(sides):
        return False

    for idx in range(len(rects)):
        alt, _alt_cov, _alt_mask = _render_from_rectangles(grid.shape, bg, _remove_one(rects, idx))
        if np.array_equal(alt, out):
            return False
    return True


def _add_random_extra(
    shape: tuple[int, int],
    margins: tuple[int, int, int, int],
    rects: list[dict[str, int]],
) -> None:
    h, w = shape
    top_margin, bottom_margin, left_margin, right_margin = margins
    row_lo, row_hi = top_margin, h - bottom_margin - 1
    col_lo, col_hi = left_margin, w - right_margin - 1

    for _ in range(40):
        side = random.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            max_depth = min(3, top_margin if side == "top" else bottom_margin)
            if max_depth <= 0:
                continue
            depth = random.randint(1, max_depth)
            width_lo = 2 if depth == 1 else 1
            width_hi = min(5, col_hi - col_lo + 1)
            if width_hi < width_lo:
                continue
            width = random.randint(width_lo, width_hi)
            start = random.randint(col_lo, col_hi - width + 1)
            rect = _make_rect(side, start, start + width - 1, depth, shape)
        else:
            max_depth = min(3, left_margin if side == "left" else right_margin)
            if max_depth <= 0:
                continue
            depth = random.randint(1, max_depth)
            height_lo = 2 if depth == 1 else 1
            height_hi = min(5, row_hi - row_lo + 1)
            if height_hi < height_lo:
                continue
            height = random.randint(height_lo, height_hi)
            start = random.randint(row_lo, row_hi - height + 1)
            rect = _make_rect(side, start, start + height - 1, depth, shape)

        if _can_add_rect(rect, shape, rects):
            rects.append(rect)
            return


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    bg = random.choice(BACKGROUND_CHOICES)

    top_margin = random.randint(MARGIN_MIN, MARGIN_MAX)
    bottom_margin = random.randint(MARGIN_MIN, MARGIN_MAX)
    left_margin = random.randint(MARGIN_MIN, MARGIN_MAX)
    right_margin = random.randint(MARGIN_MIN, MARGIN_MAX)

    if h - top_margin - bottom_margin < 4 or w - left_margin - right_margin < 4:
        return None

    row_lo, row_hi = top_margin, h - bottom_margin - 1
    col_lo, col_hi = left_margin, w - right_margin - 1

    try:
        (vc0, vc1), (vd0, vd1) = _choose_partial_overlap(col_lo, col_hi, 2, 5)
        (hr0, hr1), (hs0, hs1) = _choose_partial_overlap(row_lo, row_hi, 2, 5)
    except ValueError:
        return None

    top_depth = random.randint(1, min(3, top_margin))
    bottom_depth = random.randint(1, min(3, bottom_margin))
    left_depth = random.randint(1, min(3, left_margin))
    right_depth = random.randint(1, min(3, right_margin))

    rects = [
        _make_rect("top", vc0, vc1, top_depth, (h, w)),
        _make_rect("bottom", vd0, vd1, bottom_depth, (h, w)),
        _make_rect("left", hr0, hr1, left_depth, (h, w)),
        _make_rect("right", hs0, hs1, right_depth, (h, w)),
    ]

    if any(not _can_add_rect(rects[i], (h, w), rects[:i]) for i in range(len(rects))):
        return None

    for _ in range(random.randint(0, MAX_EXTRA_RECTS)):
        _add_random_extra((h, w), (top_margin, bottom_margin, left_margin, right_margin), rects)

    grid = _build_input_grid((h, w), bg, rects)
    if not _validate_semantics(grid, bg, rects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sweep-parity input")
