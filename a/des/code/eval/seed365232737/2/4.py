from __future__ import annotations

import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
SIDES = ("top", "bottom", "left", "right")

# Generator configuration.
GRID_H_RANGE = (10, 16)
GRID_W_RANGE = (10, 16)
PIECE_COUNT_CHOICES = (2, 3, 4)
PIECE_COUNT_WEIGHTS = (3, 4, 3)
INTERACTION_PROB = 0.6
GEN_ATTEMPTS = 600


def _inside_work_area(r: int, c: int, shape: tuple[int, int]) -> bool:
    h, w = shape
    return 1 <= r < h - 1 and 1 <= c < w - 1


def _centered_border_arrows(g: np.ndarray) -> list[dict[str, object]]:
    """
    Border-only interpretation: each arrow is a 3-cell marker on one border side,
    with the odd-colored middle cell as the tip aligned to the controlled row/col.
    """
    h, w = g.shape
    arrows: list[dict[str, object]] = []
    used: set[tuple[int, int]] = set()

    def add_arrow(
        side: str,
        anchor: tuple[int, int],
        lane: int,
        shaft_color: int,
        tip_color: int,
        cells: list[tuple[int, int]],
        dr: int,
        dc: int,
    ) -> None:
        arrows.append(
            {
                "side": side,
                "anchor": anchor,
                "lane": lane,
                "shaft_color": shaft_color,
                "tip_color": tip_color,
                "cells": tuple(cells),
                "dr": dr,
                "dc": dc,
            }
        )
        used.update(cells)

    for c in range(1, w - 1):
        cells = [(0, c - 1), (0, c), (0, c + 1)]
        if any(cell in used for cell in cells):
            continue
        left, mid, right = [int(g[r, cc]) for r, cc in cells]
        if left == right != BLACK and mid not in (BLACK, left):
            add_arrow("top", (0, c), c, left, mid, cells, 1, 0)

    for c in range(1, w - 1):
        cells = [(h - 1, c - 1), (h - 1, c), (h - 1, c + 1)]
        if any(cell in used for cell in cells):
            continue
        left, mid, right = [int(g[r, cc]) for r, cc in cells]
        if left == right != BLACK and mid not in (BLACK, left):
            add_arrow("bottom", (h - 1, c), c, left, mid, cells, -1, 0)

    for r in range(1, h - 1):
        cells = [(r - 1, 0), (r, 0), (r + 1, 0)]
        if any(cell in used for cell in cells):
            continue
        top, mid, bottom = [int(g[rr, c]) for rr, c in cells]
        if top == bottom != BLACK and mid not in (BLACK, top):
            add_arrow("left", (r, 0), r, top, mid, cells, 0, 1)

    for r in range(1, h - 1):
        cells = [(r - 1, w - 1), (r, w - 1), (r + 1, w - 1)]
        if any(cell in used for cell in cells):
            continue
        top, mid, bottom = [int(g[rr, c]) for rr, c in cells]
        if top == bottom != BLACK and mid not in (BLACK, top):
            add_arrow("right", (r, w - 1), r, top, mid, cells, 0, -1)

    return arrows


def _inward_strip_arrows(g: np.ndarray) -> list[dict[str, object]]:
    """
    Fallback interpretation for inputs that encode the 3-cell arrow as an inward strip.
    This keeps `main()` tolerant to the more literal reading of the description.
    """
    h, w = g.shape
    arrows: list[dict[str, object]] = []
    used: set[tuple[int, int]] = set()

    def add_arrow(
        side: str,
        anchor: tuple[int, int],
        lane: int,
        shaft_color: int,
        tip_color: int,
        cells: list[tuple[int, int]],
        dr: int,
        dc: int,
    ) -> None:
        arrows.append(
            {
                "side": side,
                "anchor": anchor,
                "lane": lane,
                "shaft_color": shaft_color,
                "tip_color": tip_color,
                "cells": tuple(cells),
                "dr": dr,
                "dc": dc,
            }
        )
        used.update(cells)

    if h >= 3:
        for c in range(1, w - 1):
            cells = [(0, c), (1, c), (2, c)]
            if any(cell in used for cell in cells):
                continue
            outer, mid, tip = [int(g[r, cc]) for r, cc in cells]
            if outer == mid != BLACK and tip not in (BLACK, outer):
                add_arrow("top", (0, c), c, outer, tip, cells, 1, 0)

        for c in range(1, w - 1):
            cells = [(h - 1, c), (h - 2, c), (h - 3, c)]
            if any(cell in used for cell in cells):
                continue
            outer, mid, tip = [int(g[r, cc]) for r, cc in cells]
            if outer == mid != BLACK and tip not in (BLACK, outer):
                add_arrow("bottom", (h - 1, c), c, outer, tip, cells, -1, 0)

    if w >= 3:
        for r in range(1, h - 1):
            cells = [(r, 0), (r, 1), (r, 2)]
            if any(cell in used for cell in cells):
                continue
            outer, mid, tip = [int(g[rr, c]) for rr, c in cells]
            if outer == mid != BLACK and tip not in (BLACK, outer):
                add_arrow("left", (r, 0), r, outer, tip, cells, 0, 1)

        for r in range(1, h - 1):
            cells = [(r, w - 1), (r, w - 2), (r, w - 3)]
            if any(cell in used for cell in cells):
                continue
            outer, mid, tip = [int(g[rr, c]) for rr, c in cells]
            if outer == mid != BLACK and tip not in (BLACK, outer):
                add_arrow("right", (r, w - 1), r, outer, tip, cells, 0, -1)

    return arrows


def _detect_arrows(g: np.ndarray) -> list[dict[str, object]]:
    arrows = _centered_border_arrows(g)
    if arrows:
        return arrows
    return _inward_strip_arrows(g)


def _collect_pieces(
    g: np.ndarray, arrow_cells: set[tuple[int, int]]
) -> dict[int, tuple[int, int]]:
    h, w = g.shape
    pieces: dict[int, tuple[int, int]] = {}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (r, c) in arrow_cells:
                continue
            color = int(g[r, c])
            if color == BLACK:
                continue
            assert color not in pieces, "piece colors must be unique"
            pieces[color] = (r, c)
    return pieces


def _slide_until_blocked(
    start: tuple[int, int],
    dr: int,
    dc: int,
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[tuple[int, int], bool]:
    r, c = start
    while True:
        nr = r + dr
        nc = c + dc
        if not _inside_work_area(nr, nc, shape):
            return (r, c), False
        if (nr, nc) in occupied:
            return (r, c), True
        r, c = nr, nc


def _solve_scene(
    g: np.ndarray, *, return_debug: bool = False
) -> tuple[np.ndarray, list[dict[str, object]]]:
    h, w = g.shape
    arrows = _detect_arrows(g)
    assert arrows, "expected at least one border arrow"

    arrow_cells = {cell for arrow in arrows for cell in arrow["cells"]}
    pieces = _collect_pieces(g, arrow_cells)
    shaft_colors = [int(arrow["shaft_color"]) for arrow in arrows]
    assert len(set(shaft_colors)) == len(shaft_colors), "shaft colors must be unique"
    assert set(shaft_colors) == set(pieces), "every arrow must match exactly one piece"

    ordered = sorted(
        arrows,
        key=lambda arrow: (int(arrow["anchor"][1]), int(arrow["anchor"][0])),
    )

    out = np.zeros((h - 2, w - 2), dtype=int)
    occupied: set[tuple[int, int]] = set()
    debug: list[dict[str, object]] = []

    for arrow in ordered:
        shaft_color = int(arrow["shaft_color"])
        start = pieces[shaft_color]
        end, blocked_by_piece = _slide_until_blocked(
            start,
            int(arrow["dr"]),
            int(arrow["dc"]),
            occupied,
            g.shape,
        )
        assert end not in occupied, "final piece positions must be distinct"
        occupied.add(end)
        out[end[0] - 1, end[1] - 1] = int(arrow["tip_color"])
        debug.append(
            {
                "side": str(arrow["side"]),
                "start": start,
                "end": end,
                "steps": abs(end[0] - start[0]) + abs(end[1] - start[1]),
                "blocked_by_piece": blocked_by_piece,
                "anchor": tuple(arrow["anchor"]),
                "shaft_color": shaft_color,
                "tip_color": int(arrow["tip_color"]),
            }
        )

    return out, debug if return_debug else []


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each border arrow to the same-colored interior piece, slide the piece in
    the arrow's inward direction, then recolor it with the arrow-tip color.

    The output is only the interior work area after removing the outer border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out, _debug = _solve_scene(g, return_debug=False)
    return out


def _paint_arrow(
    grid: np.ndarray,
    side: str,
    lane: int,
    shaft_color: int,
    tip_color: int,
) -> None:
    h, w = grid.shape
    if side == "top":
        grid[0, lane - 1] = shaft_color
        grid[0, lane] = tip_color
        grid[0, lane + 1] = shaft_color
    elif side == "bottom":
        grid[h - 1, lane - 1] = shaft_color
        grid[h - 1, lane] = tip_color
        grid[h - 1, lane + 1] = shaft_color
    elif side == "left":
        grid[lane - 1, 0] = shaft_color
        grid[lane, 0] = tip_color
        grid[lane + 1, 0] = shaft_color
    else:
        grid[lane - 1, w - 1] = shaft_color
        grid[lane, w - 1] = tip_color
        grid[lane + 1, w - 1] = shaft_color


def _far_enough(
    pos: tuple[int, int], taken: list[tuple[int, int]]
) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in taken)


def _lane_values_for_side(side: str, h: int, w: int) -> list[int]:
    if side in ("top", "bottom"):
        return list(range(2, w - 2))
    return list(range(2, h - 2))


def _pick_lane(
    side: str,
    h: int,
    w: int,
    side_to_lane: dict[str, int],
    pair_kind: str | None,
) -> int | None:
    candidates = _lane_values_for_side(side, h, w)
    random.shuffle(candidates)
    for lane in candidates:
        if side == "top" and "bottom" in side_to_lane and pair_kind != "vertical":
            if lane == side_to_lane["bottom"]:
                continue
        if side == "bottom" and "top" in side_to_lane and pair_kind != "vertical":
            if lane == side_to_lane["top"]:
                continue
        if side == "left" and "right" in side_to_lane and pair_kind != "horizontal":
            if lane == side_to_lane["right"]:
                continue
        if side == "right" and "left" in side_to_lane and pair_kind != "horizontal":
            if lane == side_to_lane["left"]:
                continue
        return lane
    return None


def _candidate_starts(
    side: str,
    lane: int,
    h: int,
    w: int,
    pattern: str | None,
    side_to_lane: dict[str, int],
) -> list[tuple[int, int]]:
    if pattern == "left_top":
        if side == "left":
            return [(lane, c) for c in range(2, max(3, w // 2))]
        if side == "top":
            stop_row = side_to_lane["left"]
            return [(r, lane) for r in range(2, stop_row)]
    elif pattern == "left_bottom":
        if side == "left":
            return [(lane, c) for c in range(2, max(3, w // 2))]
        if side == "bottom":
            stop_row = side_to_lane["left"]
            return [(r, lane) for r in range(stop_row + 1, h - 2)]
    elif pattern == "top_right":
        if side == "top":
            return [(r, lane) for r in range(2, max(3, h // 2))]
        if side == "right":
            stop_col = side_to_lane["top"]
            return [(lane, c) for c in range(stop_col + 1, w - 2)]
    elif pattern == "bottom_right":
        if side == "bottom":
            return [(r, lane) for r in range(max(2, h // 2), h - 2)]
        if side == "right":
            stop_col = side_to_lane["bottom"]
            return [(lane, c) for c in range(stop_col + 1, w - 2)]

    if side == "top":
        return [(r, lane) for r in range(2, h - 2)]
    if side == "bottom":
        return [(r, lane) for r in range(2, h - 2)]
    if side == "left":
        return [(lane, c) for c in range(2, w - 2)]
    return [(lane, c) for c in range(2, w - 2)]


def _build_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    n_pieces = random.choices(PIECE_COUNT_CHOICES, weights=PIECE_COUNT_WEIGHTS, k=1)[0]
    need_interaction = random.random() < INTERACTION_PROB

    if need_interaction:
        pattern = random.choice(
            ("left_top", "left_bottom", "top_right", "bottom_right")
        )
        if pattern in ("left_top", "left_bottom"):
            required = ["left", "top"] if pattern == "left_top" else ["left", "bottom"]
        else:
            required = ["top", "right"] if pattern == "top_right" else ["bottom", "right"]
        extras = [side for side in SIDES if side not in required]
        sides = required + random.sample(extras, n_pieces - 2)
    else:
        pattern = None
        sides = random.sample(SIDES, n_pieces)
    random.shuffle(sides)

    side_to_lane: dict[str, int] = {}
    if pattern == "left_top":
        side_to_lane["left"] = random.randint(3, h - 3)
        side_to_lane["top"] = w - 2
    elif pattern == "left_bottom":
        side_to_lane["left"] = random.randint(2, h - 4)
        side_to_lane["bottom"] = w - 2
    elif pattern == "top_right":
        side_to_lane["top"] = random.randint(2, w - 4)
        side_to_lane["right"] = h - 2
    elif pattern == "bottom_right":
        side_to_lane["bottom"] = random.randint(2, w - 4)
        side_to_lane["right"] = 1

    for side in sides:
        if side in side_to_lane:
            continue
        lane = _pick_lane(side, h, w, side_to_lane, None)
        if lane is None:
            return None
        side_to_lane[side] = lane

    shaft_colors = random.sample(PALETTE, n_pieces)
    tip_pool = [color for color in PALETTE if color not in shaft_colors]
    tip_colors = random.sample(tip_pool, n_pieces)

    side_specs: list[dict[str, int | str]] = []
    for i, side in enumerate(sides):
        side_specs.append(
            {
                "side": side,
                "lane": side_to_lane[side],
                "shaft_color": shaft_colors[i],
                "tip_color": tip_colors[i],
            }
        )

    starts: dict[int, tuple[int, int]] = {}
    taken_starts: list[tuple[int, int]] = []
    for spec in side_specs:
        candidates = _candidate_starts(
            str(spec["side"]),
            int(spec["lane"]),
            h,
            w,
            pattern,
            side_to_lane,
        )
        random.shuffle(candidates)
        placed = False
        for pos in candidates:
            if pos in taken_starts:
                continue
            if not _far_enough(pos, taken_starts):
                continue
            starts[int(spec["shaft_color"])] = pos
            taken_starts.append(pos)
            placed = True
            break
        if not placed:
            return None

    grid = np.zeros((h, w), dtype=int)
    for spec in side_specs:
        _paint_arrow(
            grid,
            str(spec["side"]),
            int(spec["lane"]),
            int(spec["shaft_color"]),
            int(spec["tip_color"]),
        )
    for color, pos in starts.items():
        grid[pos] = int(color)

    try:
        out, debug = _solve_scene(grid, return_debug=True)
    except AssertionError:
        return None

    if out.shape != (h - 2, w - 2):
        return None
    if len(debug) != n_pieces:
        return None

    start_set = {tuple(item["start"]) for item in debug}
    if len(start_set) != n_pieces:
        return None
    if any(tuple(item["end"]) in (start_set - {tuple(item["start"])}) for item in debug):
        return None
    if any(int(item["steps"]) <= 0 for item in debug):
        return None

    blocked_count = sum(bool(item["blocked_by_piece"]) for item in debug)
    if need_interaction and blocked_count == 0:
        return None
    if not need_interaction and blocked_count != 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
