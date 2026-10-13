from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

RECT_COLORS = np.array([BLUE, RED, GREEN, YELLOW, PINK, TEAL, MAROON], dtype=int)
VISIBLE_HEIGHTS = np.array([1, 2, 4], dtype=int)
RNG = np.random.default_rng()
MAX_ATTEMPTS = 1200


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _find_axes(grid: np.ndarray) -> tuple[int, int]:
    grey = grid == GREY
    row_counts = grey.sum(axis=1)
    col_counts = grey.sum(axis=0)
    axis_rows = np.flatnonzero(row_counts == row_counts.max())
    axis_cols = np.flatnonzero(col_counts == col_counts.max())
    assert row_counts.max() >= 2 and len(axis_rows) == 1, "expected one horizontal grey axis"
    assert col_counts.max() >= 2 and len(axis_cols) == 1, "expected one vertical grey axis"
    axis_r = int(axis_rows[0])
    axis_c = int(axis_cols[0])
    assert grey[axis_r, axis_c], "expected axes to intersect on a grey cell"
    return axis_r, axis_c


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    mask = ~np.isin(grid, [BLACK, GREY, ORANGE])
    rects: list[dict] = []
    for comp in _components(mask):
        box = _bbox(comp)
        patch = _crop(grid, box)
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "expected monochrome rectangles"
        color = int(next(iter(colors)))
        assert np.all(patch == color), "expected solid filled rectangles"
        rects.append(
            {
                "bbox": box,
                "color": color,
                "arr": patch,
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
            }
        )
    assert 3 <= len(rects) <= 5, "expected 3-5 rectangles"
    return rects


def _extract_bars(grid: np.ndarray) -> list[dict]:
    h, _ = grid.shape
    bars: list[dict] = []
    for comp in _components(grid == ORANGE):
        box = _bbox(comp)
        r0, c0, r1, c1 = box
        assert c0 == c1, "expected one-cell-wide bars"
        assert r1 == h - 1, "expected bottom-aligned bars"
        height = r1 - r0 + 1
        assert height in {1, 2, 3, 4}, "bar height must encode a rotation"
        assert len(comp) == height, "expected contiguous vertical bars"
        bars.append({"bbox": box, "col": c0, "height": height})
    bars.sort(key=lambda bar: bar["col"])
    assert 3 <= len(bars) <= 5, "expected 3-5 bars"
    return bars


def _distance_key(rect: dict, axis_r: int, axis_c: int) -> int:
    r0, c0, r1, c1 = rect["bbox"]
    return abs((r0 + r1) - 2 * axis_r) + abs((c0 + c1) - 2 * axis_c)


def _stack_rectangles(rectangles: list[dict], heights: list[int]) -> np.ndarray:
    patches = [np.rot90(rect["arr"], k=int(height) - 1) for rect, height in zip(rectangles, heights)]
    out_w = max(int(p.shape[1]) for p in patches)
    out_h = sum(int(p.shape[0]) for p in patches) + len(patches) - 1
    out = np.full((out_h, out_w), BLACK, dtype=int)

    cur_r = 0
    for patch in patches:
        ph, pw = patch.shape
        cur_c = (out_w - pw) // 2
        out[cur_r : cur_r + ph, cur_c : cur_c + pw] = patch
        cur_r += ph + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Sort rectangles by distance to the grey-axis intersection, decode each paired bar height as a rotation, and stack the rotated rectangles.

    Input structure:
    - Black background.
    - A grey horizontal axis and a grey vertical axis meet at one intersection in the upper region.
    - Several non-black, non-grey, non-orange solid monochrome rectangles lie in the upper region without touching the axes.
    - A bottom-aligned row of one-cell-wide orange bars lies in the lower region and is read from left to right.

    Rule:
    - Find every solid rectangle and sort the rectangles by the Manhattan distance from each rectangle center to the grey-axis intersection.
    - Find the orange bars from left to right; the k-th bar pairs with the k-th rectangle in that distance order.
    - Decode bar heights 1/2/3/4 as 0/90/180/270 degrees counterclockwise, rotate the paired rectangle, then stack the rotated rectangles from top to bottom with one blank row between neighbors and horizontal centering inside the maximum output width.

    Key insight:
    - The axis-based distance order does double duty: it determines the final stack order and also creates the hidden rectangle-to-bar pairing that reveals each rectangle's rotation.
    """
    g = np.asarray(grid, dtype=int)
    axis_r, axis_c = _find_axes(g)
    rectangles = _extract_rectangles(g)
    bars = _extract_bars(g)
    assert len(rectangles) == len(bars), "rectangle count must match bar count"

    distances = [_distance_key(rect, axis_r, axis_c) for rect in rectangles]
    assert len(set(distances)) == len(distances), "rectangle distances must be unique"
    rectangles.sort(key=lambda rect: _distance_key(rect, axis_r, axis_c))
    heights = [int(bar["height"]) for bar in bars]
    return _stack_rectangles(rectangles, heights)


def _candidate_rectangles(top_h: int, width: int, axis_r: int, axis_c: int) -> list[dict]:
    cands: list[dict] = []
    for rect_h in range(2, 6):
        for rect_w in range(2, 6):
            for r0 in range(top_h - rect_h + 1):
                r1 = r0 + rect_h - 1
                if r0 <= axis_r <= r1:
                    continue
                for c0 in range(width - rect_w + 1):
                    c1 = c0 + rect_w - 1
                    if c0 <= axis_c <= c1:
                        continue
                    box = (r0, c0, r1, c1)
                    cands.append(
                        {
                            "bbox": box,
                            "h": rect_h,
                            "w": rect_w,
                            "d2": abs((r0 + r1) - 2 * axis_r) + abs((c0 + c1) - 2 * axis_c),
                        }
                    )
    return cands


def _pick_rectangles(cands: list[dict], count: int) -> list[dict] | None:
    for _ in range(300):
        chosen: list[dict] = []
        used_d2: set[int] = set()
        for idx in RNG.permutation(len(cands)):
            cand = cands[int(idx)]
            if cand["d2"] in used_d2:
                continue
            if any(not _boxes_separated(cand["bbox"], other["bbox"], gap=1) for other in chosen):
                continue
            chosen.append(dict(cand))
            used_d2.add(int(cand["d2"]))
            if len(chosen) == count:
                return chosen
    return None


def _assign_colors(rects: list[dict]) -> None:
    colors = RNG.choice(RECT_COLORS, size=len(rects), replace=False)
    for rect, color in zip(rects, colors):
        rect["color"] = int(color)
        rect["arr"] = np.full((rect["h"], rect["w"]), int(color), dtype=int)


def _assign_bar_heights(sorted_rects: list[dict]) -> np.ndarray | None:
    non_square = [i for i, rect in enumerate(sorted_rects) if rect["h"] != rect["w"]]
    if len(non_square) < 2:
        return None

    heights = RNG.choice(VISIBLE_HEIGHTS, size=len(sorted_rects), replace=True)
    keep_idx = int(RNG.choice(non_square))
    rotate_idx_choices = [idx for idx in non_square if idx != keep_idx]
    rotate_idx = int(RNG.choice(rotate_idx_choices))
    heights[keep_idx] = 1
    heights[rotate_idx] = int(RNG.choice([2, 4]))
    if len(set(map(int, heights))) == 1:
        swap_idx = int(RNG.integers(len(heights)))
        heights[swap_idx] = int(RNG.choice([1, 2, 4]))
    if not any(int(h) == 1 for h in heights):
        return None
    if not any(int(h) in {2, 4} for h in heights):
        return None
    if not any(
        sorted_rects[i]["h"] != sorted_rects[i]["w"] and int(heights[i]) in {2, 4}
        for i in range(len(sorted_rects))
    ):
        return None
    return heights.astype(int)


def _orders_are_nontrivial(rects: list[dict], axis_r: int, axis_c: int) -> bool:
    dist_order = [
        rect["color"]
        for rect in sorted(rects, key=lambda rect: _distance_key(rect, axis_r, axis_c))
    ]
    left_order = [
        rect["color"]
        for rect in sorted(
            rects,
            key=lambda rect: (rect["bbox"][1] + rect["bbox"][3], rect["bbox"][0] + rect["bbox"][2]),
        )
    ]
    top_order = [
        rect["color"]
        for rect in sorted(
            rects,
            key=lambda rect: (rect["bbox"][0] + rect["bbox"][2], rect["bbox"][1] + rect["bbox"][3]),
        )
    ]
    row_sides = {1 if rect["bbox"][0] + rect["bbox"][2] > 2 * axis_r else -1 for rect in rects}
    col_sides = {1 if rect["bbox"][1] + rect["bbox"][3] > 2 * axis_c else -1 for rect in rects}
    return row_sides == {-1, 1} and col_sides == {-1, 1} and dist_order != left_order and dist_order != top_order


def _render_input(
    width: int,
    top_h: int,
    axis_r: int,
    axis_c: int,
    rects: list[dict],
    bar_start: int,
    bar_heights: np.ndarray,
) -> np.ndarray:
    total_h = top_h + 5
    grid = np.full((total_h, width), BLACK, dtype=int)
    grid[axis_r, :] = GREY
    grid[:top_h, axis_c] = GREY
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = rect["color"]
    for idx, height in enumerate(bar_heights):
        col = bar_start + 2 * idx
        grid[total_h - int(height) :, col] = ORANGE
    return grid


def _validate_generated(grid: np.ndarray, rects_sorted: list[dict], bar_heights: np.ndarray, expected: np.ndarray) -> bool:
    try:
        axis_r, axis_c = _find_axes(grid)
        parsed_rects = _extract_rectangles(grid)
        parsed_bars = _extract_bars(grid)
    except AssertionError:
        return False

    if len(parsed_rects) != len(rects_sorted) or len(parsed_bars) != len(bar_heights):
        return False

    parsed_dists = [_distance_key(rect, axis_r, axis_c) for rect in parsed_rects]
    if len(set(parsed_dists)) != len(parsed_dists):
        return False

    parsed_sorted = sorted(parsed_rects, key=lambda rect: _distance_key(rect, axis_r, axis_c))
    if [rect["color"] for rect in parsed_sorted] != [rect["color"] for rect in rects_sorted]:
        return False
    if [int(bar["height"]) for bar in parsed_bars] != [int(h) for h in bar_heights]:
        return False
    if not _orders_are_nontrivial(parsed_sorted, axis_r, axis_c):
        return False
    if not any(rect["h"] != rect["w"] and int(bar_heights[i]) in {2, 4} for i, rect in enumerate(parsed_sorted)):
        return False

    solved = main(grid)
    return np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    count = int(RNG.integers(3, 6))
    width = int(RNG.integers(max(18, 2 * count + 8), 25))
    top_h = int(RNG.integers(11, 17))
    axis_r = int(RNG.integers(2, top_h - 2))
    axis_c = int(RNG.integers(2, width - 2))

    cands = _candidate_rectangles(top_h, width, axis_r, axis_c)
    rects = _pick_rectangles(cands, count)
    if rects is None:
        return None
    _assign_colors(rects)
    if not _orders_are_nontrivial(rects, axis_r, axis_c):
        return None

    rects_sorted = sorted(rects, key=lambda rect: rect["d2"])
    bar_heights = _assign_bar_heights(rects_sorted)
    if bar_heights is None:
        return None

    bar_start_max = width - (2 * count - 1)
    bar_start = int(RNG.integers(0, bar_start_max + 1))
    grid = _render_input(width, top_h, axis_r, axis_c, rects, bar_start, bar_heights)
    expected = _stack_rectangles(rects_sorted, [int(h) for h in bar_heights])

    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return None
    if not _validate_generated(grid, rects_sorted, bar_heights, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
