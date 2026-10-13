import random
from itertools import product

import numpy as np


_CROSS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
_DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

_GRID_MIN = 14
_GRID_MAX = 26
_OBJECT_COUNT_CHOICES = (1, 2, 3, 4)
_RECT_H_MIN = 5
_RECT_H_MAX = 9
_RECT_W_MIN = 5
_RECT_W_MAX = 9
_OBJECT_PADDING = 3
_FRAME_MARGIN = 3
_MAX_NOTCHES_TOTAL = 8
_MAX_RETRIES = 400


def _uniform_side(side: np.ndarray) -> int:
    vals = np.unique(side)
    assert len(vals) == 1
    return int(vals[0])


def _interior_colors(interior: np.ndarray) -> tuple[int, int]:
    vals, counts = np.unique(interior, return_counts=True)
    bg = int(vals[int(np.argmax(counts))])
    others = [int(v) for v in vals if int(v) != bg]
    assert len(others) == 1
    return bg, others[0]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _direction_for_notch(grid: np.ndarray, r: int, c: int, obj: int) -> list[str]:
    found = []
    for name, (dr, dc) in _DIRS.items():
        pr, pc = -dc, dr
        support = (
            (r - 2 * dr, c - 2 * dc),
            (r - dr + pr, c - dc + pc),
            (r - dr - pr, c - dc - pc),
            (r + 2 * pr, c + 2 * pc),
            (r - 2 * pr, c - 2 * pc),
        )
        open_side = (
            (r + 2 * dr, c + 2 * dc),
            (r + dr + pr, c + dc + pc),
            (r + dr - pr, c + dc - pc),
        )
        if not all(_in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == obj for rr, cc in support):
            continue
        if not all((not _in_bounds(grid.shape, rr, cc)) or int(grid[rr, cc]) != obj for rr, cc in open_side):
            continue
        found.append(name)
    return found


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 7 and w >= 7

    side_color = {
        "top": _uniform_side(g[0, 1:-1]),
        "bottom": _uniform_side(g[-1, 1:-1]),
        "left": _uniform_side(g[1:-1, 0]),
        "right": _uniform_side(g[1:-1, -1]),
    }
    bg, obj = _interior_colors(g[1:-1, 1:-1])

    out = g.copy()
    painted: set[tuple[int, int]] = set()
    found_any = False
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            cross = tuple((r + dr, c + dc) for dr, dc in _CROSS)
            if not all(int(g[rr, cc]) == bg for rr, cc in cross):
                continue
            dirs = _direction_for_notch(g, r, c, obj)
            assert len(dirs) <= 1
            if not dirs:
                continue
            found_any = True
            fill = side_color[dirs[0]]
            for cell in cross:
                assert cell not in painted
                painted.add(cell)
                out[cell] = fill

    assert found_any
    return out


def _notch_record(r: int, c: int, direction: str) -> dict[str, object]:
    dr, dc = _DIRS[direction]
    pr, pc = -dc, dr
    cross = tuple((r + rr, c + cc) for rr, cc in _CROSS)
    support = (
        (r - 2 * dr, c - 2 * dc),
        (r - dr + pr, c - dc + pc),
        (r - dr - pr, c - dc - pc),
        (r + 2 * pr, c + 2 * pc),
        (r - 2 * pr, c - 2 * pc),
    )
    open_side = (
        (r + 2 * dr, c + 2 * dc),
        (r + dr + pr, c + dc + pc),
        (r + dr - pr, c + dc - pc),
    )
    return {
        "direction": direction,
        "cross": cross,
        "support": support,
        "open_side": open_side,
    }


def _candidate_notches(r0: int, c0: int, h: int, w: int, direction: str) -> list[dict[str, object]]:
    if direction == "top":
        return [_notch_record(r0, c, direction) for c in range(c0 + 2, c0 + w - 2)]
    if direction == "bottom":
        return [_notch_record(r0 + h - 1, c, direction) for c in range(c0 + 2, c0 + w - 2)]
    if direction == "left":
        return [_notch_record(r, c0, direction) for r in range(r0 + 2, r0 + h - 2)]
    return [_notch_record(r, c0 + w - 1, direction) for r in range(r0 + 2, r0 + h - 2)]


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _place_rectangles(h: int, w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        placed = False
        for _ in range(100):
            rh = random.randint(_RECT_H_MIN, min(_RECT_H_MAX, h - 2 * _FRAME_MARGIN))
            rw = random.randint(_RECT_W_MIN, min(_RECT_W_MAX, w - 2 * _FRAME_MARGIN))
            r0 = random.randint(_FRAME_MARGIN, h - _FRAME_MARGIN - rh)
            c0 = random.randint(_FRAME_MARGIN, w - _FRAME_MARGIN - rw)
            rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if all(_separated(rect, other, _OBJECT_PADDING) for other in rects):
                rects.append(rect)
                placed = True
                break
        if not placed:
            return None
    return rects


def _choose_notches(rects: list[tuple[int, int, int, int]]) -> list[dict[str, object]] | None:
    total = random.randint(len(rects), min(_MAX_NOTCHES_TOTAL, len(rects) * 3))
    plans = [plan for plan in product((1, 2, 3), repeat=len(rects)) if sum(plan) == total]
    if not plans:
        return None
    counts = list(random.choice(plans))
    random.shuffle(counts)

    result: list[dict[str, object]] = []
    used_cross: set[tuple[int, int]] = set()
    used_support: set[tuple[int, int]] = set()

    for (r0, c0, r1, c1), count in zip(rects, counts):
        rh = r1 - r0 + 1
        rw = c1 - c0 + 1
        dirs = ["top", "bottom", "left", "right"]
        random.shuffle(dirs)
        dirs = dirs[:count]
        options = {d: _candidate_notches(r0, c0, rh, rw, d) for d in dirs}

        chosen: list[dict[str, object]] | None = None

        def search(i: int, acc: list[dict[str, object]], local_cross: set[tuple[int, int]], local_support: set[tuple[int, int]]) -> list[dict[str, object]] | None:
            if i == len(dirs):
                return acc
            current = options[dirs[i]][:]
            random.shuffle(current)
            for notch in current:
                cross = set(notch["cross"])
                support = set(notch["support"])
                if cross & used_cross or cross & used_support or support & used_cross:
                    continue
                if cross & local_cross or cross & local_support or support & local_cross:
                    continue
                found = search(i + 1, acc + [notch], local_cross | cross, local_support | support)
                if found is not None:
                    return found
            return None

        chosen = search(0, [], set(), set())
        if chosen is None:
            return None
        for notch in chosen:
            result.append(notch)
            used_cross.update(notch["cross"])
            used_support.update(notch["support"])

    return result


def _oracle_ok(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        return np.array_equal(main(inp), out)
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(_GRID_MIN, _GRID_MAX)
    w = random.randint(_GRID_MIN, _GRID_MAX)
    colors = list(range(10))
    random.shuffle(colors)
    corner, top, bottom, left, right, bg, obj = colors[:7]

    grid_out = np.full((h, w), bg, dtype=int)
    grid_out[0, 0] = corner
    grid_out[0, -1] = corner
    grid_out[-1, 0] = corner
    grid_out[-1, -1] = corner
    grid_out[0, 1:-1] = top
    grid_out[-1, 1:-1] = bottom
    grid_out[1:-1, 0] = left
    grid_out[1:-1, -1] = right

    rect_count = random.choice(_OBJECT_COUNT_CHOICES)
    rects = _place_rectangles(h, w, rect_count)
    if rects is None:
        return None

    total_area = 0
    for r0, c0, r1, c1 in rects:
        total_area += (r1 - r0 + 1) * (c1 - c0 + 1)
        grid_out[r0:r1 + 1, c0:c1 + 1] = obj
    if total_area >= (h - 2) * (w - 2) // 2:
        return None

    notches = _choose_notches(rects)
    if notches is None or not notches:
        return None

    side_color = {"top": top, "bottom": bottom, "left": left, "right": right}
    for notch in notches:
        for rr, cc in notch["cross"]:
            grid_out[rr, cc] = side_color[notch["direction"]]

    grid_in = grid_out.copy()
    for notch in notches:
        for rr, cc in notch["cross"]:
            grid_in[rr, cc] = bg

    interior = grid_in[1:-1, 1:-1]
    vals, counts = np.unique(interior, return_counts=True)
    if int(vals[int(np.argmax(counts))]) != bg:
        return None
    if [int(v) for v in vals if int(v) not in (bg, obj)]:
        return None
    if not _oracle_ok(grid_in, grid_out):
        return None
    return grid_in


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {_MAX_RETRIES} attempts")
