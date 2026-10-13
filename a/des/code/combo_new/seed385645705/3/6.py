from __future__ import annotations

import random

import numpy as np


BG = 0
AZURE = 8
LINE_LEN = 3
RECT_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
DIRS = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}
MAX_GENERATION_ATTEMPTS = 500


def _centered_offsets(n: int) -> list[int]:
    if n <= 0:
        return []
    if n % 2:
        k = n // 2
        return [2 * i for i in range(-k, k + 1)]
    k = n // 2
    return list(range(-(2 * k - 1), 0, 2)) + list(range(1, 2 * k, 2))


def _offsets_fit(offsets: list[int], index: int, length: int) -> bool:
    return all(0 <= index + off < length for off in offsets)


def _fallback_offsets(n: int, index: int, length: int) -> list[int]:
    offsets: list[int] = []
    step = 0
    while len(offsets) < n and step <= length:
        candidates = [0] if step == 0 else [-step, step]
        for off in candidates:
            pos = index + off
            if 0 <= pos < length and off not in offsets:
                offsets.append(off)
                if len(offsets) == n:
                    break
        step += 1
    assert len(offsets) == n, "growth edge must contain enough start cells"
    return offsets


def _line_offsets(n: int, index: int, length: int) -> list[int]:
    centered = _centered_offsets(n)
    if _offsets_fit(centered, index, length):
        return centered
    return _fallback_offsets(n, index, length)


def _overlap(a0: int, a1: int, b0: int, b1: int) -> range:
    lo = max(int(a0), int(b0))
    hi = min(int(a1), int(b1))
    return range(lo, hi + 1) if lo <= hi else range(0)


def _side_length(rect: dict, side: int) -> int:
    if side in (0, 2):
        return int(rect["c1"] - rect["c0"] + 1)
    return int(rect["r1"] - rect["r0"] + 1)


def _edge_cell(rect: dict, side: int, index: int) -> tuple[int, int]:
    if side == 0:
        return int(rect["r0"]), int(rect["c0"] + index)
    if side == 2:
        return int(rect["r1"]), int(rect["c0"] + index)
    if side == 3:
        return int(rect["r0"] + index), int(rect["c0"])
    return int(rect["r0"] + index), int(rect["c1"])


def _edge_index(rect: dict) -> tuple[int, int]:
    r, c = rect["marker"]
    side = int(rect["side"])
    if side in (0, 2):
        return side, int(c - rect["c0"])
    return side, int(r - rect["r0"])


def _line_cells(rect: dict, side: int, index: int, n: int) -> list[tuple[int, int]]:
    length = _side_length(rect, side)
    dr, dc = DIRS[side]
    cells: list[tuple[int, int]] = []
    for off in _line_offsets(n, index, length):
        er, ec = _edge_cell(rect, side, index + off)
        for step in range(1, LINE_LEN + 1):
            cells.append((er + dr * step, ec + dc * step))
    return cells


def _find_rectangles(g: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color in (BG, AZURE):
            continue
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        patch = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        marker_local = np.argwhere(patch == AZURE)
        assert len(marker_local) == 1, "each rectangle color must have one azure edge marker"
        assert np.all((patch == color) | (patch == AZURE)), "rectangle bbox must be solid except marker"
        mr = int(r0 + marker_local[0, 0])
        mc = int(c0 + marker_local[0, 1])
        sides = []
        if mr == int(r0):
            sides.append(0)
        if mc == int(c1):
            sides.append(1)
        if mr == int(r1):
            sides.append(2)
        if mc == int(c0):
            sides.append(3)
        assert len(sides) == 1, "marker must identify exactly one growth edge"
        rects.append(
            {
                "color": color,
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "marker": (mr, mc),
                "side": sides[0],
            }
        )
    rects.sort(key=lambda x: (x["r0"], x["c0"], x["r1"], x["c1"]))
    assert rects, "expected at least one marked rectangle"
    return rects


def _visible(a: dict, b: dict, rects: list[dict]) -> bool:
    if a["c1"] < b["c0"]:
        rows = _overlap(a["r0"], a["r1"], b["r0"], b["r1"])
        for r in rows:
            blocked = False
            for m in rects:
                if m is a or m is b:
                    continue
                if a["c1"] < m["c0"] and m["c1"] < b["c0"] and m["r0"] <= r <= m["r1"]:
                    blocked = True
                    break
            if not blocked:
                return True
    elif b["c1"] < a["c0"]:
        rows = _overlap(a["r0"], a["r1"], b["r0"], b["r1"])
        for r in rows:
            blocked = False
            for m in rects:
                if m is a or m is b:
                    continue
                if b["c1"] < m["c0"] and m["c1"] < a["c0"] and m["r0"] <= r <= m["r1"]:
                    blocked = True
                    break
            if not blocked:
                return True

    if a["r1"] < b["r0"]:
        cols = _overlap(a["c0"], a["c1"], b["c0"], b["c1"])
        for c in cols:
            blocked = False
            for m in rects:
                if m is a or m is b:
                    continue
                if a["r1"] < m["r0"] and m["r1"] < b["r0"] and m["c0"] <= c <= m["c1"]:
                    blocked = True
                    break
            if not blocked:
                return True
    elif b["r1"] < a["r0"]:
        cols = _overlap(a["c0"], a["c1"], b["c0"], b["c1"])
        for c in cols:
            blocked = False
            for m in rects:
                if m is a or m is b:
                    continue
                if b["r1"] < m["r0"] and m["r1"] < a["r0"] and m["c0"] <= c <= m["c1"]:
                    blocked = True
                    break
            if not blocked:
                return True
    return False


def _visible_counts(rects: list[dict]) -> list[int]:
    counts = [0 for _ in rects]
    for i, a in enumerate(rects):
        for j, b in enumerate(rects):
            if i != j and _visible(a, b, rects):
                counts[i] += 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored feeler segments from each marked rectangle according to its visible-neighbor count.

    Input structure:
    - The background is black.
    - The grid contains several separated solid monochrome rectangles with distinct non-black, non-azure colors.
    - Each rectangle has exactly one azure pixel replacing a non-corner cell on one of its four sides; that side is the growth edge.

    Rule:
    - For every rectangle, count the other rectangles visible from its four axis-aligned sides: their row or column spans overlap, they lie on the viewed side, and no third rectangle blocks every straight sightline between them.
    - Restore the azure marker to the rectangle's original color.
    - From the growth edge, draw that many parallel outward segments of length three in the rectangle color. Segment starts are centered on the marker with one-pixel gaps when that layout fits; otherwise they are packed outward from the marker toward alternating side ends.

    Key insight:
    The local marker only gives the growth direction; the number of tentacles comes from a separate global line-of-sight count between rectangles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rects = _find_rectangles(g)
    counts = _visible_counts(rects)
    out = g.copy()

    for rect in rects:
        mr, mc = rect["marker"]
        out[mr, mc] = int(rect["color"])

    h, w = out.shape
    for rect, n in zip(rects, counts):
        side, index = _edge_index(rect)
        for rr, cc in _line_cells(rect, side, index, int(n)):
            assert 0 <= rr < h and 0 <= cc < w, "growth segment must stay in the grid"
            out[rr, cc] = int(rect["color"])
    return out


# --------------- generator ---------------


def _rect(r0: int, c0: int, h: int, w: int, color: int | None = None) -> dict:
    return {
        "color": color,
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r0 + h - 1),
        "c1": int(c0 + w - 1),
    }


def _spaced(a: dict, b: dict, gap: int = 2) -> bool:
    return (
        a["r1"] + gap < b["r0"]
        or b["r1"] + gap < a["r0"]
        or a["c1"] + gap < b["c0"]
        or b["c1"] + gap < a["c0"]
    )


def _fits_rect(rect: dict, h: int, w: int, rects: list[dict]) -> bool:
    if rect["r0"] < 0 or rect["c0"] < 0 or rect["r1"] >= h or rect["c1"] >= w:
        return False
    return all(_spaced(rect, other) for other in rects)


def _sample_overlap_start(a0: int, a1: int, size: int, limit: int) -> int | None:
    min_overlap = min(2, size, int(a1 - a0 + 1))
    low = int(a0 + min_overlap - size)
    high = int(a1 - min_overlap + 1)
    low = max(0, low)
    high = min(limit - size, high)
    if low > high:
        return None
    return random.randint(low, high)


def _place_neighbor(h: int, w: int, hub: dict, side: int, rects: list[dict]) -> bool:
    for _ in range(80):
        rh = random.randint(3, 6)
        rw = random.randint(3, 6)
        gap = random.randint(3, 6)
        if side == 0:
            c0 = _sample_overlap_start(hub["c0"], hub["c1"], rw, w)
            if c0 is None:
                continue
            r1 = hub["r0"] - gap - 1
            cand = _rect(r1 - rh + 1, c0, rh, rw)
        elif side == 2:
            c0 = _sample_overlap_start(hub["c0"], hub["c1"], rw, w)
            if c0 is None:
                continue
            cand = _rect(hub["r1"] + gap + 1, c0, rh, rw)
        elif side == 3:
            r0 = _sample_overlap_start(hub["r0"], hub["r1"], rh, h)
            if r0 is None:
                continue
            c1 = hub["c0"] - gap - 1
            cand = _rect(r0, c1 - rw + 1, rh, rw)
        else:
            r0 = _sample_overlap_start(hub["r0"], hub["r1"], rh, h)
            if r0 is None:
                continue
            cand = _rect(r0, hub["c1"] + gap + 1, rh, rw)
        if _fits_rect(cand, h, w, rects):
            rects.append(cand)
            return True
    return False


def _place_extra(h: int, w: int, rects: list[dict]) -> None:
    for _ in range(random.randint(0, 2)):
        for _attempt in range(100):
            rh = random.randint(3, 6)
            rw = random.randint(3, 6)
            cand = _rect(random.randint(0, h - rh), random.randint(0, w - rw), rh, rw)
            if _fits_rect(cand, h, w, rects):
                rects.append(cand)
                break


def _build_rect_layout() -> tuple[int, int, list[dict]] | None:
    h = random.randint(22, 30)
    w = random.randint(22, 30)
    hub_h = random.randint(4, 7)
    hub_w = random.randint(4, 7)
    if h - hub_h < 14 or w - hub_w < 14:
        return None
    hub = _rect(
        random.randint(7, h - hub_h - 7),
        random.randint(7, w - hub_w - 7),
        hub_h,
        hub_w,
    )
    rects = [hub]
    sides = random.sample([0, 1, 2, 3], random.choice([3, 4]))
    for side in sides:
        if not _place_neighbor(h, w, hub, side, rects):
            return None
    _place_extra(h, w, rects)
    colors = random.sample(RECT_COLORS, len(rects))
    for rect, color in zip(rects, colors):
        rect["color"] = int(color)
    return h, w, rects


def _render_rectangles(h: int, w: int, rects: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for rect in rects:
        grid[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = int(rect["color"])
    return grid


def _candidate_markers(
    rect: dict,
    n: int,
    branch: str,
    h: int,
    w: int,
    reserved: np.ndarray,
) -> list[tuple[int, int, int, list[tuple[int, int]]]]:
    candidates = []
    sides = [0, 1, 2, 3]
    random.shuffle(sides)
    for side in sides:
        length = _side_length(rect, side)
        if length < 3 or n > length:
            continue
        indexes = list(range(1, length - 1))
        random.shuffle(indexes)
        for index in indexes:
            centered = _offsets_fit(_centered_offsets(n), index, length)
            if branch == "center" and not centered:
                continue
            if branch == "fallback" and (n == 0 or centered):
                continue
            cells = _line_cells(rect, side, index, n)
            ok = True
            for rr, cc in cells:
                if not (0 <= rr < h and 0 <= cc < w) or reserved[rr, cc]:
                    ok = False
                    break
            if ok:
                mr, mc = _edge_cell(rect, side, index)
                candidates.append((side, index, mr, mc, cells))
    return candidates


def _assign_markers(h: int, w: int, rects: list[dict], counts: list[int]) -> bool:
    fallback_order = [i for i, n in enumerate(counts) if n >= 3]
    random.shuffle(fallback_order)
    for fallback_index in fallback_order:
        reserved = _render_rectangles(h, w, rects) != BG
        assigned: list[tuple[int, int, int, int, list[tuple[int, int]]] | None] = [None] * len(rects)
        order = [fallback_index] + [i for i in range(len(rects)) if i != fallback_index]
        random.shuffle(order[1:])
        ok = True
        used_center = False
        for i in order:
            branch = "fallback" if i == fallback_index else "center"
            candidates = _candidate_markers(rects[i], counts[i], branch, h, w, reserved)
            if not candidates and i != fallback_index:
                candidates = _candidate_markers(rects[i], counts[i], "any", h, w, reserved)
            if not candidates:
                ok = False
                break
            choice = random.choice(candidates)
            assigned[i] = choice
            side, _index, mr, mc, cells = choice
            rects[i]["side"] = side
            rects[i]["marker"] = (mr, mc)
            for rr, cc in cells:
                reserved[rr, cc] = True
            if i != fallback_index and counts[i] > 0:
                used_center = True
        if ok and used_center:
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    layout = _build_rect_layout()
    if layout is None:
        return None
    h, w, rects = layout
    counts = _visible_counts(rects)
    if max(counts, default=0) < 3 or sum(n > 0 for n in counts) < 3:
        return None
    if any(n > max(_side_length(rect, side) for side in (0, 1, 2, 3)) for rect, n in zip(rects, counts)):
        return None
    if not _assign_markers(h, w, rects, counts):
        return None

    grid = _render_rectangles(h, w, rects)
    for rect in rects:
        mr, mc = rect["marker"]
        grid[mr, mc] = AZURE

    try:
        solved = main(grid)
    except Exception:
        return None
    if solved.shape != grid.shape or max(solved.shape) > 30:
        return None
    if int(np.count_nonzero(grid == AZURE)) != len(rects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
