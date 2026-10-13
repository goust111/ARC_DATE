from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIRS = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}
FRONT_SIDE = {"right": "left", "left": "right", "down": "top", "up": "bottom"}
REAR_SIDE = {"right": "right", "left": "left", "down": "bottom", "up": "top"}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _parse_side(vals: np.ndarray, color: int) -> tuple[int, int] | None | bool:
    """Return one local opening span, None for closed, or False for invalid."""
    vals = np.asarray(vals, dtype=int)
    if vals.size < 3 or int(vals[0]) != color or int(vals[-1]) != color:
        return False
    if not bool(np.all((vals == color) | (vals == BG))):
        return False

    bg_pos = [i for i in range(1, vals.size - 1) if int(vals[i]) == BG]
    if not bg_pos:
        return None
    if bg_pos != list(range(bg_pos[0], bg_pos[-1] + 1)):
        return False
    return int(bg_pos[0]), int(bg_pos[-1])


def _frame_candidate(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int):
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None
    if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BG):
        return None

    local = {
        "left": _parse_side(grid[r0 : r1 + 1, c0], color),
        "right": _parse_side(grid[r0 : r1 + 1, c1], color),
        "top": _parse_side(grid[r0, c0 : c1 + 1], color),
        "bottom": _parse_side(grid[r1, c0 : c1 + 1], color),
    }
    if any(v is False for v in local.values()):
        return None

    openings: dict[str, tuple[int, int] | None] = {}
    for side, span in local.items():
        if span is None:
            openings[side] = None
        elif side in ("left", "right"):
            openings[side] = (r0 + span[0], r0 + span[1])
        else:
            openings[side] = (c0 + span[0], c0 + span[1])

    open_sides = [side for side, span in openings.items() if span is not None]
    if len(open_sides) not in (1, 2):
        return None
    if len(open_sides) == 2:
        pair = set(open_sides)
        if pair not in ({"left", "right"}, {"top", "bottom"}):
            return None
        a, b = open_sides
        if openings[a] != openings[b]:
            return None

    wall_cells = []
    for c in range(c0, c1 + 1):
        if int(grid[r0, c]) == color:
            wall_cells.append((r0, c))
        if int(grid[r1, c]) == color:
            wall_cells.append((r1, c))
    for r in range(r0 + 1, r1):
        if int(grid[r, c0]) == color:
            wall_cells.append((r, c0))
        if int(grid[r, c1]) == color:
            wall_cells.append((r, c1))

    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "openings": openings,
        "wall_cells": tuple(sorted(set(wall_cells))),
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    candidates = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != BG):
        m = grid == color
        corners = np.argwhere(m)
        for r0, c0 in corners:
            r0 = int(r0)
            c0 = int(c0)
            for r1 in range(r0 + 2, h):
                if not m[r1, c0]:
                    continue
                for c1 in range(c0 + 2, w):
                    if not (m[r0, c1] and m[r1, c1]):
                        continue
                    cand = _frame_candidate(grid, r0, c0, r1, c1, color)
                    if cand is not None:
                        candidates.append(cand)

    candidates.sort(
        key=lambda fr: (
            -((fr["bbox"][2] - fr["bbox"][0] + 1) * (fr["bbox"][3] - fr["bbox"][1] + 1)),
            fr["bbox"],
        )
    )
    accepted = []
    used: set[tuple[int, int]] = set()
    for fr in candidates:
        cells = set(fr["wall_cells"])
        if cells and cells.isdisjoint(used):
            accepted.append(fr)
            used.update(cells)
    accepted.sort(key=lambda fr: fr["bbox"])
    return accepted


def _find_squares(grid: np.ndarray, frames: list[dict]) -> list[dict]:
    frame_mask = np.zeros(grid.shape, dtype=bool)
    for fr in frames:
        for r, c in fr["wall_cells"]:
            frame_mask[r, c] = True

    squares = []
    for coords in _components((grid != BG) & ~frame_mask):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        color = int(grid[r0, c0])
        assert h == w, "solid block must be square"
        assert coords.shape[0] == h * w, "solid square must fill its bbox"
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == color), "solid square must be monochrome"
        squares.append({"bbox": (r0, c0, r1, c1), "size": h, "color": color, "coords": coords})
    return squares


def _direction_candidates(square: dict, frames: list[dict]) -> list[str]:
    r0, c0, r1, c1 = square["bbox"]
    found = []
    for direction in ("right", "left", "down", "up"):
        for fr in frames:
            fr0, fc0, fr1, fc1 = fr["bbox"]
            if direction == "right" and fc0 > c1 and _overlap(r0, r1, fr0, fr1):
                found.append(direction)
                break
            if direction == "left" and fc1 < c0 and _overlap(r0, r1, fr0, fr1):
                found.append(direction)
                break
            if direction == "down" and fr0 > r1 and _overlap(c0, c1, fc0, fc1):
                found.append(direction)
                break
            if direction == "up" and fr1 < r0 and _overlap(c0, c1, fc0, fc1):
                found.append(direction)
                break
    return found


def _next_frame(r: int, c: int, size: int, direction: str, frames: list[dict]):
    best = None
    best_key = None
    r0, c0, r1, c1 = r, c, r + size - 1, c + size - 1
    for fr in frames:
        fr0, fc0, fr1, fc1 = fr["bbox"]
        if direction == "right":
            if fc0 <= c1 or not _overlap(r0, r1, fr0, fr1):
                continue
            key = fc0
        elif direction == "left":
            if fc1 >= c0 or not _overlap(r0, r1, fr0, fr1):
                continue
            key = -fc1
        elif direction == "down":
            if fr0 <= r1 or not _overlap(c0, c1, fc0, fc1):
                continue
            key = fr0
        else:
            if fr1 >= r0 or not _overlap(c0, c1, fc0, fc1):
                continue
            key = -fr1
        if best is None or key < best_key:
            best = fr
            best_key = key
    return best


def _opening_covers(fr: dict, side: str, start: int, end: int) -> bool:
    span = fr["openings"].get(side)
    return span is not None and int(span[0]) <= int(start) and int(end) <= int(span[1])


def _outside_stop(fr: dict, r: int, c: int, size: int, direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = fr["bbox"]
    if direction == "right":
        return int(r), int(c0 - size)
    if direction == "left":
        return int(r), int(c1 + 1)
    if direction == "down":
        return int(r0 - size), int(c)
    return int(r1 + 1), int(c)


def _center_in_frame(fr: dict, size: int) -> tuple[int, int]:
    r0, c0, r1, c1 = fr["bbox"]
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    assert inner_h >= size and inner_w >= size
    assert (inner_h - size) % 2 == 0 and (inner_w - size) % 2 == 0
    return r0 + 1 + (inner_h - size) // 2, c0 + 1 + (inner_w - size) // 2


def _route_square(square: dict, frames: list[dict], direction: str, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    size = int(square["size"])
    r, c, _, _ = square["bbox"]
    while True:
        fr = _next_frame(r, c, size, direction, frames)
        if fr is None:
            if direction == "right":
                return int(r), int(w - size)
            if direction == "left":
                return int(r), 0
            if direction == "down":
                return int(h - size), int(c)
            return 0, int(c)

        fr0, fc0, fr1, fc1 = fr["bbox"]
        inner_h = fr1 - fr0 - 1
        inner_w = fc1 - fc0 - 1
        fits = inner_h >= size and inner_w >= size
        front = FRONT_SIDE[direction]
        rear = REAR_SIDE[direction]
        if direction in ("right", "left"):
            band_start, band_end = int(r), int(r + size - 1)
        else:
            band_start, band_end = int(c), int(c + size - 1)

        if not fits or not _opening_covers(fr, front, band_start, band_end):
            return _outside_stop(fr, r, c, size, direction)

        if _opening_covers(fr, rear, band_start, band_end):
            if direction == "right":
                c = fc1 + 1
            elif direction == "left":
                c = fc0 - size
            elif direction == "down":
                r = fr1 + 1
            else:
                r = fr0 - size
            continue

        return _center_in_frame(fr, size)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each solid square through open hollow rectangles until the first passable dead-end room captures it.

    Input structure:
    - A black background contains separated one-cell-thick hollow rectangle frames.
    - Each frame has exactly one opening or two opposite aligned openings in its wall.
    - Solid monochrome square blocks sit on visible row or column lanes aimed at a chain of frames.

    Rule:
    - Remove the original square blocks, then slide each block in its uniquely indicated straight direction.
    - At each frame, a block stops flush outside the near wall if that wall has no matching opening or the inner cavity is too small.
    - If the near opening admits the block and the opposite wall also opens, the block exits and continues; otherwise it settles centered in that cavity.
    - Frames that do not capture a block remain unchanged.

    Key insight:
    - A fitting frame with a rear opening is only a corridor; the destination is the first frame on the route that can be entered but cannot be exited.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert np.count_nonzero(g == BG) > g.size // 2, "black background must dominate"

    frames = _find_frames(g)
    assert frames, "expected at least one hollow rectangle"
    squares = _find_squares(g, frames)
    assert squares, "expected at least one solid square"

    out = g.copy()
    for sq in squares:
        coords = sq["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    placements = []
    for sq in squares:
        dirs = _direction_candidates(sq, frames)
        assert len(dirs) == 1, "each square must have one visible route direction"
        tr, tc = _route_square(sq, frames, dirs[0], g.shape)
        size = int(sq["size"])
        assert 0 <= tr <= g.shape[0] - size and 0 <= tc <= g.shape[1] - size
        placements.append((sq, int(tr), int(tc)))

    occupied = out != BG
    for sq, tr, tc in placements:
        size = int(sq["size"])
        target = occupied[tr : tr + size, tc : tc + size]
        assert not bool(target.any()), "settled squares must not overlap walls or each other"
        out[tr : tr + size, tc : tc + size] = int(sq["color"])
        occupied[tr : tr + size, tc : tc + size] = True

    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 500
FRAME_COLORS = [5, 6, 7, 8, 9]
SQUARE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _make_local_route(kind: str, size: int, corridor_count: int) -> dict:
    extra = random.choice([0, 2, 4])
    frame_h = size + 2 + extra
    top_extra = random.randint(0, extra)
    open_p0 = 1 + top_extra
    open_p1 = open_p0 + size - 1

    current_a = 0
    frames = []
    for _ in range(corridor_count):
        inner_w = size + random.choice([0, 2, 4])
        outer_w = inner_w + 2
        gap = random.randint(1, 3)
        a0 = current_a + size + gap
        frames.append(
            {
                "a0": a0,
                "a1": a0 + outer_w - 1,
                "p0": 0,
                "p1": frame_h - 1,
                "open_minus": True,
                "open_plus": True,
                "kind": "corridor",
            }
        )
        current_a = a0 + outer_w

    gap = random.randint(1, 3)
    a0 = current_a + size + gap
    if kind == "small":
        inner_w = max(1, size - 1)
        open_minus, open_plus = True, False
    elif kind == "nofront":
        inner_w = size + random.choice([0, 2])
        open_minus, open_plus = False, True
    else:
        inner_w = size + random.choice([0, 2, 4])
        open_minus, open_plus = True, False
    outer_w = inner_w + 2
    frames.append(
        {
            "a0": a0,
            "a1": a0 + outer_w - 1,
            "p0": 0,
            "p1": frame_h - 1,
            "open_minus": open_minus,
            "open_plus": open_plus,
            "kind": kind,
        }
    )

    return {
        "size": int(size),
        "height": int(frame_h),
        "width": int(a0 + outer_w),
        "sq_a": 0,
        "sq_p": int(open_p0),
        "open_span": (int(open_p0), int(open_p1)),
        "frames": frames,
        "kind": kind,
    }


def _map_square(route: dict, orientation: str, sign: int, axis0: int, perp0: int) -> tuple[int, int]:
    size = route["size"]
    a = route["sq_a"]
    p = route["sq_p"]
    route_w = route["width"]
    if orientation == "h":
        r = perp0 + p
        c = axis0 + a if sign > 0 else axis0 + route_w - 1 - (a + size - 1)
    else:
        r = axis0 + a if sign > 0 else axis0 + route_w - 1 - (a + size - 1)
        c = perp0 + p
    return int(r), int(c)


def _map_frame(route: dict, frame: dict, orientation: str, sign: int, axis0: int, perp0: int) -> dict:
    route_w = route["width"]
    sp0, sp1 = route["open_span"]
    openings = {"left": None, "right": None, "top": None, "bottom": None}

    if orientation == "h":
        r0 = perp0 + frame["p0"]
        r1 = perp0 + frame["p1"]
        if sign > 0:
            c0 = axis0 + frame["a0"]
            c1 = axis0 + frame["a1"]
            minus_side, plus_side = "left", "right"
        else:
            c0 = axis0 + route_w - 1 - frame["a1"]
            c1 = axis0 + route_w - 1 - frame["a0"]
            minus_side, plus_side = "right", "left"
        span = (perp0 + sp0, perp0 + sp1)
    else:
        c0 = perp0 + frame["p0"]
        c1 = perp0 + frame["p1"]
        if sign > 0:
            r0 = axis0 + frame["a0"]
            r1 = axis0 + frame["a1"]
            minus_side, plus_side = "top", "bottom"
        else:
            r0 = axis0 + route_w - 1 - frame["a1"]
            r1 = axis0 + route_w - 1 - frame["a0"]
            minus_side, plus_side = "bottom", "top"
        span = (perp0 + sp0, perp0 + sp1)

    if frame["open_minus"]:
        openings[minus_side] = span
    if frame["open_plus"]:
        openings[plus_side] = span

    return {"bbox": (int(r0), int(c0), int(r1), int(c1)), "openings": openings, "kind": frame["kind"]}


def _paint_frame(grid: np.ndarray, fr: dict, color: int) -> None:
    r0, c0, r1, c1 = fr["bbox"]
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    for side, span in fr["openings"].items():
        if span is None:
            continue
        if side == "left":
            grid[span[0] : span[1] + 1, c0] = BG
        elif side == "right":
            grid[span[0] : span[1] + 1, c1] = BG
        elif side == "top":
            grid[r0, span[0] : span[1] + 1] = BG
        else:
            grid[r1, span[0] : span[1] + 1] = BG


def _place_scene(routes: list[dict], orientation: str, sign: int, frame_color: int, square_colors: list[int]):
    if orientation == "h":
        needed_h = sum(r["height"] for r in routes) + 2 * (len(routes) + 1)
        needed_w = max(r["width"] for r in routes) + 4
        if needed_h > 30 or needed_w > 30:
            return None
        h = random.randint(needed_h, 30)
        w = random.randint(needed_w, 30)
        grid = np.full((h, w), BG, dtype=int)
        expected = grid.copy()
        p = random.randint(1, h - (needed_h - 1))
        axis_limit = w
    else:
        needed_w = sum(r["height"] for r in routes) + 2 * (len(routes) + 1)
        needed_h = max(r["width"] for r in routes) + 4
        if needed_h > 30 or needed_w > 30:
            return None
        h = random.randint(needed_h, 30)
        w = random.randint(needed_w, 30)
        grid = np.full((h, w), BG, dtype=int)
        expected = grid.copy()
        p = random.randint(1, w - (needed_w - 1))
        axis_limit = h

    route_records = []
    for idx, route in enumerate(routes):
        axis0 = random.randint(1, axis_limit - route["width"] - 1)
        perp0 = p
        p += route["height"] + random.randint(2, 3)
        sq_r, sq_c = _map_square(route, orientation, sign, axis0, perp0)
        global_frames = [_map_frame(route, fr, orientation, sign, axis0, perp0) for fr in route["frames"]]
        for fr in global_frames:
            _paint_frame(grid, fr, frame_color)
            _paint_frame(expected, fr, frame_color)

        size = route["size"]
        color = square_colors[idx]
        if np.any(grid[sq_r : sq_r + size, sq_c : sq_c + size] != BG):
            return None
        grid[sq_r : sq_r + size, sq_c : sq_c + size] = color

        direction = (
            "right"
            if orientation == "h" and sign > 0
            else "left"
            if orientation == "h"
            else "down"
            if sign > 0
            else "up"
        )
        if route["kind"] == "deadend":
            tr, tc = _center_in_frame(global_frames[-1], size)
        else:
            tr, tc = _outside_stop(global_frames[-1], sq_r, sq_c, size, direction)
        if np.any(expected[tr : tr + size, tc : tc + size] != BG):
            return None
        expected[tr : tr + size, tc : tc + size] = color
        route_records.append((route, global_frames, (sq_r, sq_c), (tr, tc), direction))

    if not _semantic_scene_check(grid, expected, route_records):
        return None
    return grid, expected


def _semantic_scene_check(grid: np.ndarray, expected: np.ndarray, records: list[tuple]) -> bool:
    if grid.shape != expected.shape or max(grid.shape) > 30:
        return False
    if len(records) < 2:
        return False
    kinds = {rec[0]["kind"] for rec in records}
    if "deadend" not in kinds or not ({"nofront", "small"} & kinds):
        return False

    has_corridor_before_deadend = False
    for route, frames, start, target, direction in records:
        if route["kind"] == "deadend":
            has_corridor_before_deadend |= any(fr["kind"] == "corridor" for fr in frames[:-1])
            if target != _center_in_frame(frames[-1], route["size"]):
                return False
        else:
            if target != _outside_stop(frames[-1], start[0], start[1], route["size"], direction):
                return False
    if not has_corridor_before_deadend:
        return False
    if np.array_equal(grid, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(["h", "v"])
    sign = random.choice([-1, 1])
    n_routes = random.choice([2, 3])
    kinds = ["deadend", random.choice(["nofront", "small"])]
    if n_routes == 3:
        kinds.append(random.choice(["deadend", "nofront", "small"]))
    random.shuffle(kinds)

    routes = []
    for kind in kinds:
        size = random.choice([2, 3])
        corridors = random.choice([1, 2]) if kind == "deadend" else random.choice([0, 1])
        routes.append(_make_local_route(kind, size, corridors))

    frame_color = random.choice(FRAME_COLORS)
    colors = [c for c in SQUARE_COLORS if c != frame_color]
    if len(colors) < len(routes):
        return None
    square_colors = random.sample(colors, len(routes))

    placed = _place_scene(routes, orientation, sign, frame_color, square_colors)
    if placed is None:
        return None
    grid, expected = placed

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fable/3/006/4 input")
