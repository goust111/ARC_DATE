from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 7
RECT_COLOR = 1
PIXEL_COLORS = (2, 3, 4, 6, 8)
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return Counter(map(int, grid.flatten())).most_common(1)[0][0]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == int(bg):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rs = [p[0] for p in coords]
            cs = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "size": len(coords),
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                }
            )
    return comps


def _is_filled_rectangle(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    expected = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    return set(coords) == expected


def _find_rectangle(grid: np.ndarray, bg: int) -> dict:
    comps = _components_by_color(grid, bg)
    assert comps, "expected foreground objects"
    rect_candidates = [
        comp
        for comp in comps
        if _is_filled_rectangle(comp["coords"], comp["bbox"])
        and ((comp["bbox"][1] == 0) ^ (comp["bbox"][3] == grid.shape[1] - 1))
    ]
    assert rect_candidates, "expected one side-attached filled rectangle"
    rect = max(rect_candidates, key=lambda comp: (comp["size"], comp["bbox"]))
    return rect


def _classify_singletons(
    grid: np.ndarray, rect: dict, bg: int
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    r0, c0, r1, c1 = rect["bbox"]
    comps = _components_by_color(grid, bg)
    top: list[dict] = []
    bottom: list[dict] = []
    side: list[dict] = []
    unaligned: list[dict] = []
    singles: list[dict] = []
    rect_coords = set(rect["coords"])
    for comp in comps:
        if set(comp["coords"]) == rect_coords:
            continue
        assert comp["size"] == 1, "all non-rectangle objects must be single pixels"
        (r, c) = comp["coords"][0]
        info = {"color": int(comp["color"]), "r": int(r), "c": int(c)}
        singles.append(info)
        if c0 <= c <= c1 and r < r0:
            top.append(info)
        elif c0 <= c <= c1 and r > r1:
            bottom.append(info)
        elif r0 <= r <= r1 and (c < c0 or c > c1):
            side.append(info)
        else:
            unaligned.append(info)
    return top, bottom, side, unaligned, singles


def _draw_spec_lines(
    out: np.ndarray, rect_bbox: tuple[int, int, int, int], top: list[dict], bottom: list[dict], side: list[dict]
) -> None:
    r0, c0, r1, c1 = rect_bbox
    for pix in top:
        out[pix["r"] : r0, pix["c"]] = pix["color"]
    for pix in bottom:
        out[r1 + 1 : pix["r"] + 1, pix["c"]] = pix["color"]
    for pix in side:
        if pix["c"] < c0:
            out[pix["r"], pix["c"] : c0] = pix["color"]
        else:
            out[pix["r"], c1 + 1 : pix["c"] + 1] = pix["color"]


def _spec_output(grid: np.ndarray) -> np.ndarray:
    bg = _get_bg(grid)
    rect = _find_rectangle(grid, bg)
    top, bottom, side, _unaligned, _singles = _classify_singletons(grid, rect, bg)
    r0, c0, r1, c1 = rect["bbox"]
    rect_h = r1 - r0 + 1
    rect_w = c1 - c0 + 1
    h, w = grid.shape

    out = np.array(grid, dtype=int, copy=True)
    _draw_spec_lines(out, rect["bbox"], top, bottom, side)
    out[r0 : r1 + 1, c0 : c1 + 1] = bg

    dst_left = c0 != 0
    base_c = 0 if dst_left else w - rect_w
    shear_right = len(top) > len(bottom)
    for rel_r in range(rect_h):
        shift = rel_r if shear_right else -rel_r
        start = base_c + shift
        for rel_c in range(rect_w):
            cc = start + rel_c
            if 0 <= cc < w:
                out[r0 + rel_r, cc] = rect["color"]
    return out


def _inspect_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    assert max(g.shape) <= 30, "grid must fit ARC size limits"
    bg = _get_bg(g)
    rect = _find_rectangle(g, bg)
    r0, c0, r1, c1 = rect["bbox"]
    rect_h = r1 - r0 + 1
    rect_w = c1 - c0 + 1
    assert rect["color"] == RECT_COLOR, "rectangle color must be blue"
    assert 3 <= rect_h <= 4 and 4 <= rect_w <= 5, "rectangle size out of target range"
    assert rect["size"] == rect_h * rect_w, "rectangle must be filled"
    assert rect["bbox"][1] == 0 or rect["bbox"][3] == g.shape[1] - 1, "rectangle must touch a side wall"
    assert rect["bbox"][1] != 0 or rect["bbox"][3] != g.shape[1] - 1, "rectangle cannot touch both side walls"
    assert not (r0 <= g.shape[0] // 2 <= r1), "rectangle must not cross the grid midline"

    top, bottom, side, unaligned, singles = _classify_singletons(g, rect, bg)
    assert 4 <= len(singles) <= 5, "generator should keep 4-5 colored singletons"
    assert len(top) >= 1 and len(bottom) >= 1 and len(side) >= 1, "missing required alignment case"
    assert len(top) != len(bottom), "top and bottom votes must differ"
    assert all(p["color"] in PIXEL_COLORS for p in singles), "singleton color must come from visible palette"
    assert len({p["color"] for p in singles}) == len(singles), "singletons should use distinct colors"

    vertical_cols = [p["c"] for p in top + bottom]
    assert len(set(vertical_cols)) == len(vertical_cols), "vertical vote lines should occupy distinct columns"
    side_rows = [p["r"] for p in side]
    assert len(set(side_rows)) == len(side_rows), "side lines should occupy distinct rows"
    assert len(unaligned) <= 1, "only a small distractor budget is allowed"

    line_cells: set[tuple[int, int]] = set()
    for pix in top:
        assert pix["r"] <= r0 - 2, "top pixel must be separated from rectangle"
        cells = {(r, pix["c"]) for r in range(pix["r"], r0)}
        assert not (cells & line_cells), "lines should not overlap each other"
        line_cells |= cells
    for pix in bottom:
        assert pix["r"] >= r1 + 2, "bottom pixel must be separated from rectangle"
        cells = {(r, pix["c"]) for r in range(r1 + 1, pix["r"] + 1)}
        assert not (cells & line_cells), "lines should not overlap each other"
        line_cells |= cells
    for pix in side:
        if pix["c"] < c0:
            assert pix["c"] <= c0 - 2, "side pixel must be separated from rectangle"
            cells = {(pix["r"], c) for c in range(pix["c"], c0)}
        else:
            assert pix["c"] >= c1 + 2, "side pixel must be separated from rectangle"
            cells = {(pix["r"], c) for c in range(c1 + 1, pix["c"] + 1)}
        assert not (cells & line_cells), "lines should not overlap each other"
        line_cells |= cells

    for pix in unaligned:
        assert pix["r"] < r0 or pix["r"] > r1, "unaligned pixel cannot share a rectangle row"
        assert pix["c"] < c0 or pix["c"] > c1, "unaligned pixel cannot share a rectangle column"
        assert (pix["r"], pix["c"]) not in line_cells, "unaligned distractor must stay off future lines"

    return {
        "bg": bg,
        "rect": rect,
        "top": top,
        "bottom": bottom,
        "side": side,
        "unaligned": unaligned,
        "singles": singles,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect aligned single pixels to the side-attached rectangle, then move and shear that rectangle by the line-vote imbalance.

    Input structure:
    - The background is the most common color and is typically orange.
    - One large filled blue rectangle touches exactly one vertical grid edge.
    - Several other colored objects are isolated single pixels; some share a row or a column with the rectangle while others do not.

    Rule:
    - Identify the largest filled foreground rectangle and every other foreground singleton.
    - For each singleton that shares a column with the rectangle from above or below, draw a same-colored vertical line that stops one cell before the rectangle; for each singleton that shares a row from the side, draw the analogous horizontal line; unaligned singletons stay unchanged.
    - Count top-arriving lines U and bottom-arriving lines D, erase the source rectangle, move it to the opposite side wall at the same rows, then shear its rows by one more cell per row: rightward if U > D, leftward if D > U, clipping any cells that leave the grid.

    Key insight:
    - The visible connector lines are also votes: their above-versus-below count determines the later shear direction, so the geometry depends on the line distribution rather than on the original wall side.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    rect = _find_rectangle(g, bg)
    top, bottom, side, _unaligned, _singles = _classify_singletons(g, rect, bg)
    r0, c0, r1, c1 = rect["bbox"]
    rect_h = r1 - r0 + 1
    rect_w = c1 - c0 + 1
    h, w = g.shape

    out = np.array(g, dtype=int, copy=True)
    _draw_spec_lines(out, rect["bbox"], top, bottom, side)
    out[r0 : r1 + 1, c0 : c1 + 1] = bg

    dst_left = c0 != 0
    base_c = 0 if dst_left else w - rect_w
    shear_sign = 1 if len(top) > len(bottom) else -1
    for rel_r in range(rect_h):
        start = base_c + shear_sign * rel_r
        for rel_c in range(rect_w):
            cc = start + rel_c
            if 0 <= cc < w:
                out[r0 + rel_r, cc] = rect["color"]
    return out


def _reserve_blocked(blocked: set[tuple[int, int]], h: int, w: int, cells: set[tuple[int, int]]) -> None:
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    blocked.add((rr, cc))


def _sample_rect_top(h: int, rect_h: int) -> int | None:
    choices = []
    mid = h // 2
    for top in range(2, h - rect_h - 1):
        bottom = top + rect_h - 1
        if bottom < mid or top > mid:
            if top >= 2 and h - bottom - 1 >= 2:
                choices.append(top)
    return random.choice(choices) if choices else None


def _pick_free(candidates: list[tuple[int, int]], blocked: set[tuple[int, int]]) -> tuple[int, int] | None:
    free = [cell for cell in candidates if cell not in blocked]
    return random.choice(free) if free else None


def _build_input() -> np.ndarray | None:
    h = random.randint(13, 18)
    w = random.randint(12, 17)
    rect_h = random.randint(3, 4)
    rect_w = random.randint(4, 5)
    rect_top = _sample_rect_top(h, rect_h)
    if rect_top is None:
        return None

    left_wall = random.choice([False, True])
    top_votes, bottom_votes = random.choice([(1, 2), (2, 1)])
    side_votes = random.choice([1, min(2, rect_h)])
    unaligned_count = 1 if side_votes == 1 and random.random() < 0.7 else 0
    total_pixels = top_votes + bottom_votes + side_votes + unaligned_count
    if total_pixels > len(PIXEL_COLORS):
        return None
    colors = random.sample(list(PIXEL_COLORS), total_pixels)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    rect_left = 0 if left_wall else w - rect_w
    rect_right = rect_left + rect_w - 1
    rect_bottom = rect_top + rect_h - 1
    grid[rect_top : rect_bottom + 1, rect_left : rect_right + 1] = RECT_COLOR

    blocked: set[tuple[int, int]] = set()
    rect_cells = {(r, c) for r in range(rect_top, rect_bottom + 1) for c in range(rect_left, rect_right + 1)}
    _reserve_blocked(blocked, h, w, rect_cells)

    line_only: set[tuple[int, int]] = set()
    placements: list[tuple[int, int, int]] = []

    vertical_cols = random.sample(list(range(rect_left, rect_right + 1)), top_votes + bottom_votes)
    top_cols = vertical_cols[:top_votes]
    bottom_cols = vertical_cols[top_votes:]

    for col in top_cols:
        candidates = [(r, col) for r in range(0, rect_top - 1)]
        choice = _pick_free(candidates, blocked)
        if choice is None:
            return None
        r, c = choice
        placements.append((r, c, colors[len(placements)]))
        segment = {(rr, c) for rr in range(r, rect_top)}
        line_only |= segment
        _reserve_blocked(blocked, h, w, {(r, c)})

    for col in bottom_cols:
        candidates = [(r, col) for r in range(rect_bottom + 2, h)]
        choice = _pick_free(candidates, blocked)
        if choice is None:
            return None
        r, c = choice
        placements.append((r, c, colors[len(placements)]))
        segment = {(rr, c) for rr in range(rect_bottom + 1, r + 1)}
        line_only |= segment
        _reserve_blocked(blocked, h, w, {(r, c)})

    side_rows = random.sample(list(range(rect_top, rect_bottom + 1)), side_votes)
    for row in side_rows:
        if left_wall:
            candidates = [(row, c) for c in range(rect_right + 2, w - 1)]
        else:
            candidates = [(row, c) for c in range(1, rect_left - 1)]
        choice = _pick_free(candidates, blocked)
        if choice is None:
            return None
        r, c = choice
        placements.append((r, c, colors[len(placements)]))
        if c < rect_left:
            segment = {(r, cc) for cc in range(c, rect_left)}
        else:
            segment = {(r, cc) for cc in range(rect_right + 1, c + 1)}
        line_only |= segment
        _reserve_blocked(blocked, h, w, {(r, c)})

    for _ in range(unaligned_count):
        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if (r, c) not in blocked
            and (r, c) not in line_only
            and not (rect_top <= r <= rect_bottom)
            and not (rect_left <= c <= rect_right)
        ]
        choice = _pick_free(candidates, blocked)
        if choice is None:
            return None
        r, c = choice
        placements.append((r, c, colors[len(placements)]))
        _reserve_blocked(blocked, h, w, {(r, c)})

    for r, c, color in placements:
        grid[r, c] = color

    info = _inspect_input(grid)
    expected = _spec_output(grid)
    actual = main(grid)
    assert np.array_equal(actual, expected), "solver and independent specification must agree"

    sr0, sc0, _sr1, _sc1 = info["rect"]["bbox"]
    top_row_blue = np.where(actual[sr0] == RECT_COLOR)[0]
    assert len(top_row_blue) == rect_w, "moved rectangle top row must preserve its width"
    if sc0 == 0:
        assert int(top_row_blue.min()) == w - rect_w, "left-wall source must move to the right wall"
    else:
        assert int(top_row_blue.min()) == 0, "right-wall source must move to the left wall"
    assert len(info["top"]) != len(info["bottom"]), "vote difference must survive validation"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
