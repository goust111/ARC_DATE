from __future__ import annotations

import random
from collections import Counter

import numpy as np


PLUS_REL = {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}
MOVE_DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PALETTE = [1, 2, 3, 4, 7, 8, 9]
BG = 0
GREY = 5
MARK = 6


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return comps


def _is_plus(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    rel = {(r - r0, c - c0) for r, c in comp["coords"]}
    return comp["area"] == 5 and (r1 - r0 + 1) == 3 and (c1 - c0 + 1) == 3 and rel == PLUS_REL


def _paint_rect(grid: np.ndarray, top: int, left: int, side: int, color: int) -> None:
    grid[top : top + side, left : left + side] = int(color)


def _paint_plus(grid: np.ndarray, top: int, left: int, color: int) -> None:
    for dr, dc in PLUS_REL:
        grid[top + dr, left + dc] = int(color)


def _rect(top: int, left: int, side: int) -> tuple[int, int, int, int]:
    return top, left, top + side - 1, left + side - 1


def _point_in_rect(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = rect
    return r0 <= r <= r1 and c0 <= c <= c1


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _rect_gap_ok(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    aa = (a[0] - pad, a[1] - pad, a[2] + pad, a[3] + pad)
    return not _rects_overlap(aa, b)


def _sweep_rect(initial_top: int, initial_left: int, final_top: int, final_left: int, side: int) -> tuple[int, int, int, int]:
    r0 = min(initial_top, final_top)
    c0 = min(initial_left, final_left)
    r1 = max(initial_top, final_top) + side - 1
    c1 = max(initial_left, final_left) + side - 1
    return r0, c0, r1, c1


def _sort_key(square: dict) -> tuple[int, int]:
    top = int(square["top"])
    left = int(square["left"])
    side = int(square["side"])
    direction = square["direction"]
    if direction == "up":
        return 0, top
    if direction == "down":
        return 1, -(top + side - 1)
    if direction == "left":
        return 2, left
    return 3, -(left + side - 1)


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)

    colors = sorted(set(map(int, g.reshape(-1))) - {bg})
    comps: list[dict] = []
    for color in colors:
        comps.extend(_find_components(g, color))

    markers = [comp for comp in comps if _is_plus(comp)]
    assert len(markers) == 1, "expected exactly one 3x3 plus marker"
    marker = markers[0]
    marker_color = int(marker["color"])

    squares: list[dict] = []
    grey_used: set[tuple[int, int]] = set()
    grey_cells = {(int(r), int(c)) for r, c in np.argwhere(g == GREY)}

    for comp in comps:
        if comp is marker or int(comp["color"]) == GREY:
            continue
        color = int(comp["color"])
        assert color not in (bg, marker_color), "unexpected foreground color"
        r0, c0, r1, c1 = comp["bbox"]
        side = r1 - r0 + 1
        assert side == c1 - c0 + 1, "foreground objects must be square"
        assert comp["area"] == side * side, "foreground squares must be solid"

        side_greys: list[tuple[str, tuple[int, int]]] = []
        for rr, cc in grey_cells:
            if r0 - 1 == rr and c0 <= cc <= c1:
                side_greys.append(("up", (rr, cc)))
            if r1 + 1 == rr and c0 <= cc <= c1:
                side_greys.append(("down", (rr, cc)))
            if c0 - 1 == cc and r0 <= rr <= r1:
                side_greys.append(("left", (rr, cc)))
            if c1 + 1 == cc and r0 <= rr <= r1:
                side_greys.append(("right", (rr, cc)))

        if side % 2 == 0:
            assert len(side_greys) == 1, "each even square needs one grey direction pixel"
            direction, grey_rc = side_greys[0]
            grey_used.add(grey_rc)
        else:
            assert not side_greys, "odd squares must not have grey indicators"
            direction = None

        squares.append(
            {
                "color": color,
                "top": r0,
                "left": c0,
                "side": side,
                "direction": direction,
            }
        )

    assert 4 <= len(squares) <= 6, "expected 4-6 squares"
    assert len({sq["color"] for sq in squares}) == len(squares), "square colors must be distinct"
    odd_count = sum(sq["side"] % 2 == 1 for sq in squares)
    even_count = len(squares) - odd_count
    assert odd_count >= 1 and even_count >= 1, "expected both odd and even squares"
    assert grey_used == grey_cells, "every grey pixel must belong to exactly one even square"

    mr0, mc0, _mr1, _mc1 = marker["bbox"]
    return {
        "grid": g,
        "h": h,
        "w": w,
        "bg": bg,
        "marker_color": marker_color,
        "marker_top_left": (mr0, mc0),
        "marker_center": (mr0 + 1, mc0 + 1),
        "squares": squares,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gate solid squares by side parity: mark odd-square centers and slide even squares by grey-arrow direction until they meet an odd anchor or the border.

    Input structure:
    - A black background contains 4-6 separated solid monochrome squares of distinct colors, plus one magenta 3x3 plus marker.
    - Odd-sided squares have no extra pixels; every even-sided square has exactly one adjacent grey pixel on one side indicating its slide direction.
    - The plus marker is only a color source for center marking and a later keep/erase test.

    Rule:
    - Detect the unique plus marker and every solid square, using square side parity to split them into anchors and sliders.
    - Odd-sided squares stay fixed and recolor their single center pixel to the plus-marker color.
    - Even-sided squares erase their grey indicator and translate step by step in the indicated direction, stopping at the last in-bounds position before overlapping any fixed odd square or any already-settled even square.
    - Preserve the input plus marker only when its center lies inside the axis-aligned field spanned by the odd-square centers; otherwise erase it.

    Key insight:
    - The same shape class splits into two roles purely by side parity: odd squares become stationary blockers with magenta centers, and even squares become rigid sliders whose stopping points depend on those blockers.
    """
    parsed = _parse_input(grid)
    g = parsed["grid"]
    h = parsed["h"]
    w = parsed["w"]
    bg = parsed["bg"]
    marker_color = parsed["marker_color"]

    odd_squares = [sq for sq in parsed["squares"] if sq["side"] % 2 == 1]
    even_squares = [sq for sq in parsed["squares"] if sq["side"] % 2 == 0]

    out = np.full_like(g, bg)
    blockers = np.full_like(g, bg)
    odd_centers: list[tuple[int, int]] = []

    for sq in odd_squares:
        top = int(sq["top"])
        left = int(sq["left"])
        side = int(sq["side"])
        color = int(sq["color"])
        _paint_rect(out, top, left, side, color)
        _paint_rect(blockers, top, left, side, color)
        center = (top + side // 2, left + side // 2)
        odd_centers.append(center)
        out[center] = marker_color

    for sq in sorted(even_squares, key=_sort_key):
        top = int(sq["top"])
        left = int(sq["left"])
        side = int(sq["side"])
        color = int(sq["color"])
        dr, dc = MOVE_DIRS[str(sq["direction"])]

        cur_top = top
        cur_left = left
        while True:
            next_top = cur_top + dr
            next_left = cur_left + dc
            if next_top < 0 or next_left < 0 or next_top + side > h or next_left + side > w:
                break
            if np.any(blockers[next_top : next_top + side, next_left : next_left + side] != bg):
                break
            cur_top = next_top
            cur_left = next_left

        _paint_rect(out, cur_top, cur_left, side, color)
        _paint_rect(blockers, cur_top, cur_left, side, color)

    rows = [r for r, _c in odd_centers]
    cols = [c for _r, c in odd_centers]
    center_r, center_c = parsed["marker_center"]
    keep_marker = min(rows) <= center_r <= max(rows) and min(cols) <= center_c <= max(cols)
    if keep_marker:
        plus_top, plus_left = parsed["marker_top_left"]
        _paint_plus(out, plus_top, plus_left, marker_color)

    return out


def _plus_top_left(corner: str, h: int, w: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - 3
    if corner == "bl":
        return h - 3, 0
    return h - 3, w - 3


def _center_in_bbox(center: tuple[int, int], centers: list[tuple[int, int]]) -> bool:
    rows = [r for r, _c in centers]
    cols = [c for _r, c in centers]
    return min(rows) <= center[0] <= max(rows) and min(cols) <= center[1] <= max(cols)


def _available_positions_for_forced_odd(corner: str, h: int, w: int, used: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    if corner in {"tl", "tr"}:
        row = 0
        col_range = range(0, w - 2)
        for col in col_range:
            rect = _rect(row, col, 3)
            if all(_rect_gap_ok(rect, other, 1) for other in used):
                positions.append((row, col))
    else:
        row = h - 3
        col_range = range(0, w - 2)
        for col in col_range:
            rect = _rect(row, col, 3)
            if all(_rect_gap_ok(rect, other, 1) for other in used):
                positions.append((row, col))
    return positions


def _available_positions_for_forced_col_odd(corner: str, h: int, w: int, used: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    if corner in {"tl", "bl"}:
        col = 0
        row_range = range(0, h - 2)
        for row in row_range:
            rect = _rect(row, col, 3)
            if all(_rect_gap_ok(rect, other, 1) for other in used):
                positions.append((row, col))
    else:
        col = w - 3
        row_range = range(0, h - 2)
        for row in row_range:
            rect = _rect(row, col, 3)
            if all(_rect_gap_ok(rect, other, 1) for other in used):
                positions.append((row, col))
    return positions


def _random_square_position(side: int, h: int, w: int, used: list[tuple[int, int, int, int]], tries: int = 150) -> tuple[int, int] | None:
    for _ in range(tries):
        top = random.randint(0, h - side)
        left = random.randint(0, w - side)
        rect = _rect(top, left, side)
        if all(_rect_gap_ok(rect, other, 1) for other in used):
            return top, left
    return None


def _random_output_position(side: int, h: int, w: int, used: list[tuple[int, int, int, int]], tries: int = 150) -> tuple[int, int] | None:
    return _random_square_position(side, h, w, used, tries=tries)


def _grey_cell_for_square(top: int, left: int, side: int, direction: str) -> tuple[int, int]:
    mid = side // 2 - 1 if side == 2 else side // 2
    if direction == "up":
        return top - 1, left + mid
    if direction == "down":
        return top + side, left + mid
    if direction == "left":
        return top + mid, left - 1
    return top + mid, left + side


def _candidate_anchor_finals(anchor: dict, side: int, h: int, w: int) -> list[tuple[str, int, int]]:
    r0, c0 = int(anchor["top"]), int(anchor["left"])
    a_side = int(anchor["side"])
    r1 = r0 + a_side - 1
    c1 = c0 + a_side - 1
    candidates: list[tuple[str, int, int]] = []

    final_top = r1 + 1
    if final_top + side <= h:
        lo = max(0, c0 - side + 1)
        hi = min(w - side, c1)
        for final_left in range(lo, hi + 1):
            candidates.append(("up", final_top, final_left))

    final_top = r0 - side
    if final_top >= 0:
        lo = max(0, c0 - side + 1)
        hi = min(w - side, c1)
        for final_left in range(lo, hi + 1):
            candidates.append(("down", final_top, final_left))

    final_left = c1 + 1
    if final_left + side <= w:
        lo = max(0, r0 - side + 1)
        hi = min(h - side, r1)
        for final_top in range(lo, hi + 1):
            candidates.append(("left", final_top, final_left))

    final_left = c0 - side
    if final_left >= 0:
        lo = max(0, r0 - side + 1)
        hi = min(h - side, r1)
        for final_top in range(lo, hi + 1):
            candidates.append(("right", final_top, final_left))

    random.shuffle(candidates)
    return candidates


def _boundary_finals(side: int, h: int, w: int) -> list[tuple[str, int, int]]:
    candidates: list[tuple[str, int, int]] = []
    for left in range(0, w - side + 1):
        candidates.append(("up", 0, left))
        candidates.append(("down", h - side, left))
    for top in range(0, h - side + 1):
        candidates.append(("left", top, 0))
        candidates.append(("right", top, w - side))
    random.shuffle(candidates)
    return candidates


def _output_rect_ok(
    rect: tuple[int, int, int, int],
    used: list[tuple[int, int, int, int]],
    allow_touch: tuple[int, int, int, int] | None = None,
) -> bool:
    for other in used:
        if allow_touch is not None and other == allow_touch:
            if _rects_overlap(rect, other):
                return False
            continue
        if not _rect_gap_ok(rect, other, 1):
            return False
    return True


def _point_clear(point: tuple[int, int], h: int, w: int, forbidden_rects: list[tuple[int, int, int, int]]) -> bool:
    r, c = point
    if not (0 <= r < h and 0 <= c < w):
        return False
    for rect in forbidden_rects:
        if _point_in_rect(point, rect):
            return False
    return True


def _grey_adjacent_to_rect(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = rect
    return (
        (r == r0 - 1 and c0 <= c <= c1)
        or (r == r1 + 1 and c0 <= c <= c1)
        or (c == c0 - 1 and r0 <= r <= r1)
        or (c == c1 + 1 and r0 <= r <= r1)
    )


def _grey_indicator_ok(
    point: tuple[int, int],
    owner_rect: tuple[int, int, int, int],
    other_rects: list[tuple[int, int, int, int]],
    h: int,
    w: int,
) -> bool:
    if not _point_clear(point, h, w, other_rects + [owner_rect]):
        return False
    if not _grey_adjacent_to_rect(point, owner_rect):
        return False
    return all(not _grey_adjacent_to_rect(point, rect) for rect in other_rects)


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(13, 18)
    w = random.randint(13, 18)
    count = random.randint(4, 6)
    odd_count = random.randint(2, count - 2)
    even_count = count - odd_count
    preserve_marker = random.choice([True, False])
    corner = random.choice(["tl", "tr", "bl", "br"])
    plus_top, plus_left = _plus_top_left(corner, h, w)
    plus_center = (plus_top + 1, plus_left + 1)
    plus_rect = _rect(plus_top, plus_left, 3)

    colors = random.sample(PALETTE, count)
    odd_colors = colors[:odd_count]
    even_colors = colors[odd_count:]
    if preserve_marker:
        odd_sides = [3, 3] + [random.choice([3, 5]) for _ in range(odd_count - 2)]
    else:
        odd_sides = [random.choice([3, 5]) for _ in range(odd_count)]
    even_sides = [random.choice([2, 4]) for _ in range(even_count)]
    random.shuffle(odd_sides)
    random.shuffle(even_sides)

    odds: list[dict] = []
    input_used = [plus_rect]
    output_used = [plus_rect]

    if preserve_marker:
        if odd_sides.count(3) < 2:
            return None
        first = odd_sides.index(3)
        second = odd_sides.index(3, first + 1)
        odd_sides[0], odd_sides[first] = odd_sides[first], odd_sides[0]
        odd_sides[1], odd_sides[second] = odd_sides[second], odd_sides[1]

        row_positions = _available_positions_for_forced_odd(corner, h, w, input_used)
        col_positions = _available_positions_for_forced_col_odd(corner, h, w, input_used)
        row_positions = [pos for pos in row_positions if pos[1] >= 4 or corner in {"tr", "br"}]
        row_positions = [pos for pos in row_positions if pos[1] <= w - 7 or corner in {"tl", "bl"}]
        col_positions = [pos for pos in col_positions if pos[0] >= 4 or corner in {"bl", "br"}]
        col_positions = [pos for pos in col_positions if pos[0] <= h - 7 or corner in {"tl", "tr"}]
        if not row_positions or not col_positions:
            return None

        row_top, row_left = random.choice(row_positions)
        row_rect = _rect(row_top, row_left, 3)
        input_used.append(row_rect)
        output_used.append(row_rect)
        odds.append({"color": odd_colors[0], "side": 3, "top": row_top, "left": row_left})

        col_candidates = [pos for pos in col_positions if _rect_gap_ok(_rect(pos[0], pos[1], 3), row_rect, 1)]
        if not col_candidates:
            return None
        col_top, col_left = random.choice(col_candidates)
        col_rect = _rect(col_top, col_left, 3)
        input_used.append(col_rect)
        output_used.append(col_rect)
        odds.append({"color": odd_colors[1], "side": 3, "top": col_top, "left": col_left})

        start_idx = 2
    else:
        start_idx = 0

    for idx in range(start_idx, odd_count):
        side = odd_sides[idx]
        pos = _random_square_position(side, h, w, input_used)
        if pos is None:
            return None
        top, left = pos
        rect = _rect(top, left, side)
        input_used.append(rect)
        output_used.append(rect)
        odds.append({"color": odd_colors[idx], "side": side, "top": top, "left": left})

    odd_centers = [(odd["top"] + odd["side"] // 2, odd["left"] + odd["side"] // 2) for odd in odds]
    if _center_in_bbox(plus_center, odd_centers) != preserve_marker:
        return None

    anchor_stop_types = ["anchor"] * even_count
    if even_count >= 2:
        anchor_stop_types[0] = "boundary"
        anchor_stop_types[1] = "anchor"
        for i in range(2, even_count):
            anchor_stop_types[i] = random.choice(["anchor", "boundary"])
        random.shuffle(anchor_stop_types)

    evens: list[dict] = []
    sweep_rects: list[tuple[int, int, int, int]] = []
    input_grey_rects: list[tuple[int, int, int, int]] = []

    for color, side, stop_type in zip(even_colors, even_sides, anchor_stop_types):
        placed = False
        for _ in range(200):
            if stop_type == "anchor":
                anchor = random.choice(odds)
                options = _candidate_anchor_finals(anchor, side, h, w)
                allow_touch = _rect(anchor["top"], anchor["left"], anchor["side"])
            else:
                anchor = None
                options = _boundary_finals(side, h, w)
                allow_touch = None

            random.shuffle(options)
            for direction, final_top, final_left in options:
                final_rect = _rect(final_top, final_left, side)
                if not _output_rect_ok(final_rect, output_used, allow_touch=allow_touch):
                    continue

                dr, dc = MOVE_DIRS[direction]
                if direction == "up":
                    max_dist = h - side - final_top
                elif direction == "down":
                    max_dist = final_top
                elif direction == "left":
                    max_dist = w - side - final_left
                else:
                    max_dist = final_left
                if max_dist < 1:
                    continue

                distances = list(range(1, max_dist + 1))
                random.shuffle(distances)
                for dist in distances:
                    init_top = final_top - dr * dist
                    init_left = final_left - dc * dist
                    init_rect = _rect(init_top, init_left, side)
                    if not all(_rect_gap_ok(init_rect, other, 1) for other in input_used + input_grey_rects):
                        continue
                    if any(not _rect_gap_ok(init_rect, other, 0) for other in sweep_rects):
                        continue

                    sweep = _sweep_rect(init_top, init_left, final_top, final_left, side)
                    if any(_rects_overlap(sweep, other) for other in sweep_rects):
                        continue
                    if any(_rects_overlap(sweep, _rect(odd["top"], odd["left"], odd["side"])) for odd in odds):
                        continue
                    if _rects_overlap(sweep, plus_rect):
                        continue

                    grey_cell = _grey_cell_for_square(init_top, init_left, side, direction)
                    grey_rect = (grey_cell[0], grey_cell[1], grey_cell[0], grey_cell[1])
                    if not _grey_indicator_ok(grey_cell, init_rect, input_used + input_grey_rects, h, w):
                        continue
                    if any(not _rect_gap_ok(grey_rect, other, 1) for other in input_used):
                        continue
                    if any(_rects_overlap(grey_rect, other) for other in sweep_rects):
                        continue
                    if any(not _rect_gap_ok(grey_rect, other, 0) for other in output_used):
                        continue

                    evens.append(
                        {
                            "color": color,
                            "side": side,
                            "direction": direction,
                            "input_top": init_top,
                            "input_left": init_left,
                            "final_top": final_top,
                            "final_left": final_left,
                            "stop_type": stop_type,
                        }
                    )
                    input_used.append(init_rect)
                    output_used.append(final_rect)
                    input_grey_rects.append(grey_rect)
                    sweep_rects.append(sweep)
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    _paint_plus(inp, plus_top, plus_left, MARK)
    if preserve_marker:
        _paint_plus(out, plus_top, plus_left, MARK)

    for odd in odds:
        _paint_rect(inp, odd["top"], odd["left"], odd["side"], odd["color"])
        _paint_rect(out, odd["top"], odd["left"], odd["side"], odd["color"])
        center = (odd["top"] + odd["side"] // 2, odd["left"] + odd["side"] // 2)
        out[center] = MARK

    for even in evens:
        _paint_rect(inp, even["input_top"], even["input_left"], even["side"], even["color"])
        grey_r, grey_c = _grey_cell_for_square(even["input_top"], even["input_left"], even["side"], even["direction"])
        inp[grey_r, grey_c] = GREY
        _paint_rect(out, even["final_top"], even["final_left"], even["side"], even["color"])

    if np.array_equal(inp, out):
        return None

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None

    parsed = _parse_input(inp)
    parsed_odd = [sq for sq in parsed["squares"] if sq["side"] % 2 == 1]
    parsed_even = [sq for sq in parsed["squares"] if sq["side"] % 2 == 0]
    if len(parsed_odd) != odd_count or len(parsed_even) != even_count:
        return None
    if _center_in_bbox(parsed["marker_center"], [(sq["top"] + sq["side"] // 2, sq["left"] + sq["side"] // 2) for sq in parsed_odd]) != preserve_marker:
        return None
    if even_count >= 2 and {even["stop_type"] for even in evens} != {"anchor", "boundary"}:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _build_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid parity-gated sliding-square input")
