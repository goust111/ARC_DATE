from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
ANCHOR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
RECT_COUNT_RANGE = (2, 5)
RECT_OUTER_H_RANGE = (5, 8)
RECT_OUTER_W_RANGE = (5, 8)
RECT_MARGIN = 2
PAIR_COUNT_RANGE = (2, 4)
PAIR_SAMPLE_ATTEMPTS = 300
RECT_PLACE_ATTEMPTS = 200
GEN_ATTEMPTS = 400


Coord = tuple[int, int]
Rect = tuple[int, int, int, int]


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[Coord] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _rectangles(grid: np.ndarray) -> list[Rect]:
    rects: list[Rect] = []
    for comp in _components(grid == GREY):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        expected = set()
        for c in range(c0, c1 + 1):
            expected.add((r0, c))
            expected.add((r1, c))
        for r in range(r0 + 1, r1):
            expected.add((r, c0))
            expected.add((r, c1))
        if set(comp) != expected:
            continue
        rects.append((r0, c0, r1, c1))
    rects.sort()
    return rects


def _segment(a: Coord, b: Coord) -> list[Coord]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise ValueError("segment must be axis-aligned")


def _underlying_path(a: Coord, b: Coord, mode: str) -> list[Coord]:
    if mode == "hv":
        bend = (a[0], b[1])
    else:
        bend = (b[0], a[1])
    first = _segment(a, bend)
    second = _segment(bend, b)
    return first + second[1:]


def _visible_path(grid: np.ndarray, a: Coord, b: Coord, mode: str) -> list[Coord] | None:
    visible: list[Coord] = []
    for cell in _underlying_path(a, b, mode):
        r, c = cell
        v = int(grid[r, c])
        if cell == a or cell == b:
            visible.append(cell)
            continue
        if v == GREY:
            continue
        if v != BLACK:
            return None
        visible.append(cell)
    return visible


def _choose_path(grid: np.ndarray, a: Coord, b: Coord) -> tuple[str, list[Coord], tuple[frozenset[Coord], frozenset[Coord]]]:
    hv = _visible_path(grid, a, b, "hv")
    vh = _visible_path(grid, a, b, "vh")
    if hv is None and vh is None:
        raise AssertionError("expected at least one legal path")
    hv_set = frozenset(hv or [])
    vh_set = frozenset(vh or [])
    if hv is None:
        return "vh", vh, (hv_set, vh_set)
    if vh is None:
        return "hv", hv, (hv_set, vh_set)
    if len(hv) < len(vh):
        return "hv", hv, (hv_set, vh_set)
    if len(vh) < len(hv):
        return "vh", vh, (hv_set, vh_set)
    if hv_set == vh_set:
        return "hv", hv, (hv_set, vh_set)
    # Fallback for unexpected equal-length inputs.
    if tuple(hv) <= tuple(vh):
        return "hv", hv, (hv_set, vh_set)
    return "vh", vh, (hv_set, vh_set)


def _anchor_pairs(grid: np.ndarray) -> list[tuple[int, Coord, Coord]]:
    pairs: list[tuple[int, Coord, Coord]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BLACK, GREY):
            continue
        pts = [tuple(map(int, p)) for p in np.argwhere(grid == color)]
        assert len(pts) == 2, "every anchor color must appear exactly twice"
        pairs.append((color, pts[0], pts[1]))
    pairs.sort(key=lambda item: item[0])
    return pairs


def _inside_cells(path: Iterable[Coord], rect: Rect) -> list[Coord]:
    r0, c0, r1, c1 = rect
    return [(r, c) for r, c in path if r0 < r < r1 and c0 < c < c1]


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[Rect], list[dict]]:
    g = np.asarray(grid, dtype=int)
    rects = _rectangles(g)
    out = g.copy()
    path_info: list[dict] = []

    for color, a, b in _anchor_pairs(g):
        mode, path, _alt_sets = _choose_path(g, a, b)
        for r, c in path:
            if out[r, c] != GREY:
                out[r, c] = color
        path_info.append({"color": color, "mode": mode, "path": path, "a": a, "b": b})

    for rect in rects:
        entrants: list[tuple[Coord, int]] = []
        for info in path_info:
            inside = _inside_cells(info["path"], rect)
            if inside:
                entrants.append((min(inside), int(info["color"])))
        if not entrants:
            continue
        _entry, fill = min(entrants)
        r0, c0, r1, c1 = rect
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if out[r, c] == BLACK:
                    out[r, c] = fill

    return out, rects, path_info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color anchor pair with the shorter visible L-path.

    Grey rectangle borders are left unchanged, so a path can cross a region by
    painting the black cells in its interior while skipping the grey frame
    cells it passes behind. A crossed region is then filled using the color of
    the line whose first interior pixel is earliest in row-major order.
    """
    out, _rects, _paths = _solve(grid)
    return out


def _draw_rectangles(grid: np.ndarray, rects: list[Rect]) -> None:
    for r0, c0, r1, c1 in rects:
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY


def _place_rectangles(h: int, w: int) -> list[Rect] | None:
    target = random.randint(*RECT_COUNT_RANGE)
    rects: list[Rect] = []
    blocked = np.zeros((h, w), dtype=bool)
    for _ in range(RECT_PLACE_ATTEMPTS):
        if len(rects) == target:
            break
        oh = random.randint(*RECT_OUTER_H_RANGE)
        ow = random.randint(*RECT_OUTER_W_RANGE)
        if oh >= h - 2 or ow >= w - 2:
            continue
        r0 = random.randint(1, h - oh - 1)
        c0 = random.randint(1, w - ow - 1)
        r1 = r0 + oh - 1
        c1 = c0 + ow - 1
        rr0 = max(0, r0 - RECT_MARGIN)
        cc0 = max(0, c0 - RECT_MARGIN)
        rr1 = min(h - 1, r1 + RECT_MARGIN)
        cc1 = min(w - 1, c1 + RECT_MARGIN)
        if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue
        rects.append((r0, c0, r1, c1))
        blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    if len(rects) < 2:
        return None
    rects.sort()
    return rects


def _anchor_cells(h: int, w: int, rects: list[Rect], forbidden: set[Coord]) -> list[Coord]:
    cells: list[Coord] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in forbidden:
                continue
            inside_box = False
            for r0, c0, r1, c1 in rects:
                if r0 <= r <= r1 and c0 <= c <= c1:
                    inside_box = True
                    break
            if not inside_box:
                cells.append((r, c))
    return cells


def _pair_candidates(
    grid: np.ndarray,
    rects: list[Rect],
    occupied_visible: set[Coord],
    forbidden_anchor_cells: set[Coord],
) -> list[dict]:
    h, w = grid.shape
    free = _anchor_cells(h, w, rects, forbidden_anchor_cells)
    if len(free) < 2:
        return []

    candidates: list[dict] = []
    seen: set[tuple[Coord, Coord, str]] = set()
    for _ in range(PAIR_SAMPLE_ATTEMPTS):
        a, b = random.sample(free, 2)
        if a[0] == b[0] or a[1] == b[1]:
            continue
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 8:
            continue
        hv = _visible_path(grid, a, b, "hv")
        vh = _visible_path(grid, a, b, "vh")
        if hv is None or vh is None:
            continue
        if len(hv) == len(vh):
            if frozenset(hv) != frozenset(vh):
                continue
        mode, path, alt_sets = _choose_path(grid, a, b)
        path_set = set(path)
        if path_set & occupied_visible:
            continue
        crossed = tuple(
            idx for idx, rect in enumerate(rects) if _inside_cells(path, rect)
        )
        sig = tuple(sorted((a, b))) + (mode,)
        if sig in seen:
            continue
        seen.add(sig)
        candidates.append(
            {
                "a": a,
                "b": b,
                "mode": mode,
                "path": path,
                "path_set": path_set,
                "crossed": crossed,
                "sensitive": set(alt_sets[0] | alt_sets[1]),
            }
        )
    return candidates


def _pick_candidate(
    candidates: list[dict],
    crossed_counts: list[int],
    missing_modes: set[str],
    need_shared: bool,
    first_pair: bool,
) -> dict | None:
    if not candidates:
        return None

    pool = candidates
    if first_pair:
        hit = [c for c in pool if c["crossed"]]
        if hit:
            pool = hit
    if need_shared:
        shared = [
            c
            for c in pool
            if c["crossed"] and any(crossed_counts[idx] > 0 for idx in c["crossed"])
        ]
        if shared:
            pool = shared
    if missing_modes:
        mode_hit = [c for c in pool if c["mode"] in missing_modes]
        if mode_hit:
            pool = mode_hit

    scored = sorted(
        pool,
        key=lambda c: (
            0 if c["crossed"] else 1,
            -sum(crossed_counts[idx] > 0 for idx in c["crossed"]),
            -len(c["crossed"]),
            len(c["path"]),
        ),
    )
    best = scored[: max(1, min(12, len(scored)))]
    return random.choice(best)


def _render_expected(inp: np.ndarray, rects: list[Rect], pairs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for pair in pairs:
        color = int(pair["color"])
        for r, c in pair["path"]:
            if out[r, c] != GREY:
                out[r, c] = color

    for rect in rects:
        entrants: list[tuple[Coord, int]] = []
        for pair in pairs:
            inside = _inside_cells(pair["path"], rect)
            if inside:
                entrants.append((min(inside), int(pair["color"])))
        if not entrants:
            continue
        _entry, fill = min(entrants)
        r0, c0, r1, c1 = rect
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if out[r, c] == BLACK:
                    out[r, c] = fill
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    rects = _place_rectangles(h, w)
    if rects is None:
        return None

    inp = np.full((h, w), BLACK, dtype=int)
    _draw_rectangles(inp, rects)

    target_pairs = random.randint(*PAIR_COUNT_RANGE)
    want_shared = len(rects) >= 2 and target_pairs >= 3 and random.random() < 0.55
    want_idle_rect = len(rects) >= 3 and random.random() < 0.55

    pairs: list[dict] = []
    occupied_visible: set[Coord] = set()
    forbidden_anchor_cells: set[Coord] = set()
    crossed_counts = [0 for _ in rects]

    for idx in range(target_pairs):
        candidates = _pair_candidates(inp, rects, occupied_visible, forbidden_anchor_cells)
        missing_modes = {"hv", "vh"} - {pair["mode"] for pair in pairs}
        pick = _pick_candidate(
            candidates,
            crossed_counts,
            missing_modes if idx < target_pairs - 1 or len(pairs) == 0 else set(),
            need_shared=want_shared and not any(count >= 2 for count in crossed_counts),
            first_pair=(idx == 0),
        )
        if pick is None:
            return None
        color = ANCHOR_COLORS[idx]
        inp[pick["a"]] = color
        inp[pick["b"]] = color
        pick["color"] = color
        pairs.append(pick)
        occupied_visible |= pick["path_set"]
        forbidden_anchor_cells |= pick["sensitive"]
        for rect_idx in pick["crossed"]:
            crossed_counts[rect_idx] += 1

    if not any(crossed_counts):
        return None
    if want_shared and not any(count >= 2 for count in crossed_counts):
        return None
    if want_idle_rect and not any(count == 0 for count in crossed_counts):
        return None
    if len({pair["mode"] for pair in pairs}) < 2:
        return None

    expected = _render_expected(inp, rects, pairs)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
