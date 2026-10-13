from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ARROW_SHAPES = {
    (-1, 0): ((0, 1), (1, 0), (1, 2)),
    (1, 0): ((0, 0), (0, 2), (1, 1)),
    (0, -1): ((1, 0), (0, 1), (2, 1)),
    (0, 1): ((0, 0), (2, 0), (1, 1)),
}

# Generator configuration.
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 28)
ROOM_COUNT_RANGE = (2, 6)
ROOM_H_RANGE = (3, 8)
ROOM_W_RANGE = (3, 8)
ROOM_PLACE_TRIES = 200
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    if conn8:
        steps = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        )
    else:
        steps = ((-1, 0), (1, 0), (0, -1), (0, 1))

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in steps:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _room_component(g: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = g.shape
    sr, sc = start
    q = deque([(sr, sc)])
    seen = {start}
    comp: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        comp.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if (
                0 <= nr < h
                and 0 <= nc < w
                and (nr, nc) not in seen
                and int(g[nr, nc]) != GREY
            ):
                seen.add((nr, nc))
                q.append((nr, nc))
    return comp


def _decode_arrow(g: np.ndarray, comp: list[tuple[int, int]]) -> dict[str, object] | None:
    if len(comp) != 3:
        return None
    colors = {int(g[r, c]) for r, c in comp}
    if len(colors) != 1:
        return None
    color = next(iter(colors))
    if color in (BLACK, GREY):
        return None

    degs: dict[tuple[int, int], int] = {}
    for r, c in comp:
        deg = 0
        for rr, cc in comp:
            if (rr, cc) == (r, c):
                continue
            if max(abs(rr - r), abs(cc - c)) == 1:
                deg += 1
        degs[(r, c)] = deg
    if sorted(degs.values()) != [1, 1, 2]:
        return None

    shaft = next(pt for pt, deg in degs.items() if deg == 2)
    wings = [pt for pt, deg in degs.items() if deg == 1]
    mean_r = (wings[0][0] + wings[1][0]) / 2.0
    mean_c = (wings[0][1] + wings[1][1]) / 2.0
    dr_val = shaft[0] - mean_r
    dc_val = shaft[1] - mean_c

    if abs(dr_val) > 0.1 and abs(dc_val) < 0.1:
        direction = (-1, 0) if dr_val < 0 else (1, 0)
    elif abs(dc_val) > 0.1 and abs(dr_val) < 0.1:
        direction = (0, -1) if dc_val < 0 else (0, 1)
    else:
        return None

    return {
        "coords": tuple(sorted(comp)),
        "shaft": shaft,
        "color": color,
        "direction": direction,
    }


def _find_arrows(g: np.ndarray) -> list[dict[str, object]]:
    mask = (g != BLACK) & (g != GREY)
    arrows: list[dict[str, object]] = []
    for comp in _components(mask, conn8=True):
        arrow = _decode_arrow(g, comp)
        if arrow is None:
            continue
        room = _room_component(g, arrow["shaft"])
        arrow["sort_key"] = _bbox(room)[:2]
        arrows.append(arrow)
    arrows.sort(key=lambda item: tuple(item["sort_key"]))
    return arrows


def _expand_room(
    out: np.ndarray,
    room_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    color: int,
) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = room_bbox
    dr, dc = direction

    if (dr, dc) == (-1, 0):
        assert r0 >= 2
        out[r0 - 1, c0 : c1 + 1] = color
        out[r0 - 2, c0 - 1 : c1 + 2] = GREY
    elif (dr, dc) == (1, 0):
        assert r1 + 2 < h
        out[r1 + 1, c0 : c1 + 1] = color
        out[r1 + 2, c0 - 1 : c1 + 2] = GREY
    elif (dr, dc) == (0, -1):
        assert c0 >= 2
        out[r0 : r1 + 1, c0 - 1] = color
        out[r0 - 1 : r1 + 2, c0 - 2] = GREY
    else:
        assert c1 + 2 < w
        out[r0 : r1 + 1, c1 + 1] = color
        out[r0 - 1 : r1 + 2, c1 + 2] = GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each arrow-marked room by pushing the wall in the arrow direction.

    The room interior is the enclosed non-grey component containing the arrow.
    Moving the wall by one cell turns the old wall segment into a colored strip,
    keeps the arrow removed, and redraws the moved wall one cell farther out.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for arrow in _find_arrows(g):
        shaft = tuple(arrow["shaft"])
        color = int(arrow["color"])
        if int(out[shaft]) != color:
            continue

        room = _room_component(out, shaft)
        room_bbox = _bbox(room)

        for r, c in arrow["coords"]:
            out[r, c] = BLACK
        _expand_room(out, room_bbox, tuple(arrow["direction"]), color)

    return out


def _draw_room(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 - 1, c0 - 1 : c1 + 2] = GREY
    grid[r1 + 1, c0 - 1 : c1 + 2] = GREY
    grid[r0 - 1 : r1 + 2, c0 - 1] = GREY
    grid[r0 - 1 : r1 + 2, c1 + 1] = GREY


def _place_arrow(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    color: int,
    rng: random.Random,
) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = bbox
    shape = ARROW_SHAPES[direction]
    height = max(r for r, _ in shape) + 1
    width = max(c for _, c in shape) + 1
    top = rng.randint(r0, r1 - height + 1)
    left = rng.randint(c0, c1 - width + 1)
    coords = []
    for dr, dc in shape:
        rr = top + dr
        cc = left + dc
        grid[rr, cc] = color
        coords.append((rr, cc))
    return tuple(sorted(coords))


def _rooms_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _simulate_expected(
    inp: np.ndarray,
    arrow_specs: list[dict[str, object]],
) -> np.ndarray:
    out = inp.copy()
    for spec in sorted(arrow_specs, key=lambda item: tuple(item["sort_key"])):
        for r, c in spec["coords"]:
            out[r, c] = BLACK
        _expand_room(out, tuple(spec["bbox"]), tuple(spec["direction"]), int(spec["color"]))
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)

    target_rooms = rng.randint(*ROOM_COUNT_RANGE)
    rooms: list[tuple[int, int, int, int]] = []
    reserved: list[tuple[int, int, int, int]] = []

    for _ in range(ROOM_PLACE_TRIES):
        if len(rooms) >= target_rooms:
            break
        room_h = rng.randint(*ROOM_H_RANGE)
        room_w = rng.randint(*ROOM_W_RANGE)
        if room_h + 6 > h or room_w + 6 > w:
            continue

        r0 = rng.randint(2, h - room_h - 3)
        c0 = rng.randint(2, w - room_w - 3)
        r1 = r0 + room_h - 1
        c1 = c0 + room_w - 1
        bbox = (r0, c0, r1, c1)
        reserve = (r0 - 2, c0 - 2, r1 + 2, c1 + 2)
        if any(_rooms_overlap(reserve, prev) for prev in reserved):
            continue
        rooms.append(bbox)
        reserved.append(reserve)
        _draw_room(grid, bbox)

    if len(rooms) < 2:
        return None

    max_arrows = min(len(rooms), 5)
    arrow_count = rng.randint(1, max_arrows)
    arrow_room_ids = set(rng.sample(range(len(rooms)), arrow_count))

    directions = list(DIRS)
    rng.shuffle(directions)
    while len(directions) < arrow_count:
        directions.append(rng.choice(DIRS))
    rng.shuffle(directions)

    colors = rng.sample(COLOR_POOL, arrow_count)
    arrow_specs: list[dict[str, object]] = []

    for room_index, bbox in enumerate(rooms):
        if room_index not in arrow_room_ids:
            continue
        direction = directions.pop()
        color = colors.pop()
        coords = _place_arrow(grid, bbox, direction, color, rng)
        arrow_specs.append(
            {
                "bbox": bbox,
                "coords": coords,
                "direction": direction,
                "color": color,
                "sort_key": bbox[:2],
            }
        )

    if not arrow_specs:
        return None

    expected = _simulate_expected(grid, arrow_specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
