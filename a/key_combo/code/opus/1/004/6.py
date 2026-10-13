from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

RECT_COLORS = (YELLOW, GRAY, MAGENTA, CYAN, MAROON)
GEN_ATTEMPTS = 800


def _half_spans(length: int) -> tuple[int, int]:
    before = (length - 1) // 2
    after = length - 1 - before
    return before, after


def _center_from_box(top: int, left: int, height: int, width: int) -> tuple[int, int]:
    return top + (height - 1) // 2, left + (width - 1) // 2


def _component_rectangles(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    blocked = {BLACK, BLUE, RED, GREEN, ORANGE}
    seen = np.zeros((h, w), dtype=bool)
    rects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in blocked or seen[r, c]:
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            top = min(rows)
            bottom = max(rows)
            left = min(cols)
            right = max(cols)
            height = bottom - top + 1
            width = right - left + 1
            sub = grid[top : bottom + 1, left : right + 1]
            assert np.all(sub == color), "foreground component must be a solid rectangle"

            center_r, center_c = _center_from_box(top, left, height, width)
            rects.append(
                {
                    "color": color,
                    "top": top,
                    "left": left,
                    "height": height,
                    "width": width,
                    "center": (center_r, center_c),
                    "arr": np.full((height, width), color, dtype=int),
                }
            )

    return rects


def _parse(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30
    assert h >= 8 and w >= 8

    assert int(g[0, 0]) == ORANGE
    assert np.all(g[0, 1:] == RED)
    assert np.all(g[1:, 0] == GREEN)

    rects = _component_rectangles(g)
    assert 4 <= len(rects) <= 6

    rect_mask = np.zeros((h, w), dtype=bool)
    for rect in rects:
        top = rect["top"]
        left = rect["left"]
        height = rect["height"]
        width = rect["width"]
        assert top >= 1 and left >= 1
        rect_mask[top : top + height, left : left + width] = True

    for r in range(1, h):
        for c in range(1, w):
            if rect_mask[r, c]:
                continue
            expected = BLUE if (r + c) % 2 == 0 else BLACK
            assert int(g[r, c]) == expected, "background must be a checkerboard outside the rectangles"

    return rects


def _stack_height(rects: list[np.ndarray]) -> int:
    if not rects:
        return 0
    return sum(int(rect.shape[0]) for rect in rects) + len(rects) - 1


def _stack(out: np.ndarray, rects: list[np.ndarray], start_col: int) -> None:
    row = 0
    for rect in rects:
        h, w = rect.shape
        out[row : row + h, start_col : start_col + w] = rect
        row += h + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split rectangles by checkerboard-center parity, rotate one group, and stack the two groups side by side.

    Input structure:
    - The background is a blue/black checkerboard whose top-left corner is replaced by an L-shaped reference: row 0 is red, column 0 is green, and (0,0) is orange.
    - Four to six solid interior rectangles use foreground colors different from the checkerboard and the edge markers.
    - Each rectangle is non-square and does not touch the reference lines.

    Rule:
    - For each rectangle, compute the center cell using the top-left choice for even side lengths.
    - If that center lies on an even-parity checkerboard cell, keep the rectangle unrotated and sort it by distance to the horizontal red edge line, breaking ties by center column.
    - If that center lies on an odd-parity checkerboard cell, sort it by distance to the vertical green edge line, breaking ties by center row, then rotate it 90 degrees counterclockwise.
    - Stack the even-parity group top-to-bottom on the left and the odd-parity group top-to-bottom on the right, with one blank column between groups and one blank row between consecutive rectangles.

    Key insight:
    - The checkerboard parity at each rectangle's center is the hidden classifier that decides both the sorting axis and whether the rectangle must be rotated before stacking.
    """
    rects = _parse(np.asarray(grid, dtype=int))

    group_a: list[dict] = []
    group_b: list[dict] = []
    for rect in rects:
        center_r, center_c = rect["center"]
        if (center_r + center_c) % 2 == 0:
            group_a.append(rect)
        else:
            group_b.append(rect)

    group_a.sort(key=lambda rect: (rect["center"][0], rect["center"][1]))
    group_b.sort(key=lambda rect: (rect["center"][1], rect["center"][0]))

    a_rects = [rect["arr"] for rect in group_a]
    b_rects = [np.rot90(rect["arr"], 1) for rect in group_b]

    a_width = max((int(rect.shape[1]) for rect in a_rects), default=0)
    b_width = max((int(rect.shape[1]) for rect in b_rects), default=0)
    gap = 1 if a_rects and b_rects else 0
    out_h = max(_stack_height(a_rects), _stack_height(b_rects))
    out_w = a_width + gap + b_width
    out = np.full((out_h, out_w), BLACK, dtype=int)

    _stack(out, a_rects, 0)
    _stack(out, b_rects, a_width + gap)
    return out


def _sample_shape() -> tuple[int, int]:
    while True:
        short = random.randint(2, 4)
        long = random.randint(3, 6)
        if short == long:
            continue
        if random.random() < 0.5:
            return short, long
        return long, short


def _candidate_centers(height: int, width: int, rect_h: int, rect_w: int, parity: int) -> list[tuple[int, int]]:
    up, down = _half_spans(rect_h)
    left, right = _half_spans(rect_w)
    centers: list[tuple[int, int]] = []
    for r in range(1 + up, height - 1 - down):
        for c in range(1 + left, width - 1 - right):
            if (r + c) % 2 == parity:
                centers.append((r, c))
    random.shuffle(centers)
    return centers


def _fits_center(
    occupied: np.ndarray,
    center_r: int,
    center_c: int,
    rect_h: int,
    rect_w: int,
) -> tuple[int, int] | None:
    top = center_r - (rect_h - 1) // 2
    left = center_c - (rect_w - 1) // 2
    bottom = top + rect_h - 1
    right = left + rect_w - 1
    h, w = occupied.shape

    if top < 1 or left < 1 or bottom > h - 2 or right > w - 2:
        return None

    rr0 = max(0, top - 1)
    rr1 = min(h, bottom + 2)
    cc0 = max(0, left - 1)
    cc1 = min(w, right + 2)
    if np.any(occupied[rr0:rr1, cc0:cc1]):
        return None
    return top, left


def _mark_rect(occupied: np.ndarray, top: int, left: int, rect_h: int, rect_w: int) -> None:
    occupied[top : top + rect_h, left : left + rect_w] = True


def _place_at_center(occupied: np.ndarray, spec: dict, center: tuple[int, int]) -> bool:
    rect_h, rect_w = spec["shape"]
    fit = _fits_center(occupied, center[0], center[1], rect_h, rect_w)
    if fit is None:
        return False
    top, left = fit
    spec["top"] = top
    spec["left"] = left
    spec["center"] = center
    _mark_rect(occupied, top, left, rect_h, rect_w)
    return True


def _place_tie_pair(
    occupied: np.ndarray,
    first: dict,
    second: dict,
    parity: int,
    mode: str,
) -> bool:
    h, w = occupied.shape
    centers1 = _candidate_centers(h, w, first["shape"][0], first["shape"][1], parity)
    centers2 = _candidate_centers(h, w, second["shape"][0], second["shape"][1], parity)

    keyed1: dict[int, list[tuple[int, int]]] = {}
    keyed2: dict[int, list[tuple[int, int]]] = {}
    axis = 0 if mode == "row" else 1
    for center in centers1:
        keyed1.setdefault(center[axis], []).append(center)
    for center in centers2:
        keyed2.setdefault(center[axis], []).append(center)

    common = list(set(keyed1) & set(keyed2))
    random.shuffle(common)
    for key in common:
        random.shuffle(keyed1[key])
        random.shuffle(keyed2[key])
        for center1 in keyed1[key]:
            for center2 in keyed2[key]:
                if center1 == center2:
                    continue
                temp = occupied.copy()
                spec1 = dict(first)
                spec2 = dict(second)
                if not _place_at_center(temp, spec1, center1):
                    continue
                if not _place_at_center(temp, spec2, center2):
                    continue
                occupied[:, :] = temp
                first.update(spec1)
                second.update(spec2)
                return True
    return False


def _place_single(occupied: np.ndarray, spec: dict, parity: int) -> bool:
    h, w = occupied.shape
    centers = _candidate_centers(h, w, spec["shape"][0], spec["shape"][1], parity)
    for center in centers:
        if _place_at_center(occupied, spec, center):
            return True
    return False


def _groups_interleave(specs: list[dict]) -> bool:
    a_rows = [spec["center"][0] for spec in specs if spec["group"] == "A"]
    b_rows = [spec["center"][0] for spec in specs if spec["group"] == "B"]
    a_cols = [spec["center"][1] for spec in specs if spec["group"] == "A"]
    b_cols = [spec["center"][1] for spec in specs if spec["group"] == "B"]
    if not a_rows or not b_rows:
        return False
    rows_overlap = min(a_rows) < max(b_rows) and min(b_rows) < max(a_rows)
    cols_overlap = min(a_cols) < max(b_cols) and min(b_cols) < max(a_cols)
    return rows_overlap and cols_overlap


def _has_tie(specs: list[dict], group: str, axis: int) -> bool:
    values = [spec["center"][axis] for spec in specs if spec["group"] == group]
    return len(values) != len(set(values))


def _build_checkerboard(height: int, width: int) -> np.ndarray:
    rr, cc = np.indices((height, width))
    grid = ((rr + cc) % 2 == 0).astype(int)
    grid[0, :] = RED
    grid[:, 0] = GREEN
    grid[0, 0] = ORANGE
    return grid


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(18, 24)
    width = random.randint(18, 24)
    rect_count = random.randint(4, 5)
    a_count = random.randint(2, rect_count - 2)

    groups = ["A"] * a_count + ["B"] * (rect_count - a_count)
    random.shuffle(groups)
    colors = random.sample(list(RECT_COLORS), rect_count)

    specs: list[dict] = []
    for idx in range(rect_count):
        specs.append(
            {
                "group": groups[idx],
                "shape": _sample_shape(),
                "color": colors[idx],
            }
        )

    if not any(h > w for h, w in (spec["shape"] for spec in specs)):
        return None
    if not any(w > h for h, w in (spec["shape"] for spec in specs)):
        return None

    occupied = np.zeros((height, width), dtype=bool)
    a_specs = [spec for spec in specs if spec["group"] == "A"]
    b_specs = [spec for spec in specs if spec["group"] == "B"]

    force_a_tie = len(a_specs) >= 2 and random.random() < 0.7
    force_b_tie = len(b_specs) >= 2 and random.random() < 0.7

    pair_jobs: list[tuple[str, dict, dict, int, str]] = []
    if force_a_tie:
        pair_jobs.append(("A", a_specs[0], a_specs[1], 0, "row"))
    if force_b_tie:
        pair_jobs.append(("B", b_specs[0], b_specs[1], 1, "col"))
    pair_jobs.sort(key=lambda item: item[1]["shape"][0] * item[1]["shape"][1] + item[2]["shape"][0] * item[2]["shape"][1], reverse=True)

    for _group, first, second, parity, mode in pair_jobs:
        if not _place_tie_pair(occupied, first, second, parity, mode):
            return None

    remaining = [spec for spec in specs if "top" not in spec]
    remaining.sort(key=lambda spec: spec["shape"][0] * spec["shape"][1], reverse=True)
    for spec in remaining:
        parity = 0 if spec["group"] == "A" else 1
        if not _place_single(occupied, spec, parity):
            return None

    if not _groups_interleave(specs):
        return None

    if force_a_tie and not _has_tie(specs, "A", 0):
        return None
    if force_b_tie and not _has_tie(specs, "B", 1):
        return None

    grid = _build_checkerboard(height, width)
    for spec in specs:
        top = spec["top"]
        left = spec["left"]
        rect_h, rect_w = spec["shape"]
        grid[top : top + rect_h, left : left + rect_w] = spec["color"]

    parsed = _parse(grid)
    even_count = sum((rect["center"][0] + rect["center"][1]) % 2 == 0 for rect in parsed)
    odd_count = len(parsed) - even_count
    if even_count < 2 or odd_count < 2:
        return None

    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid checkerboard parity stacking task after {GEN_ATTEMPTS} attempts")
