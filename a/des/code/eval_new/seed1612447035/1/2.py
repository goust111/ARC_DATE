import random
from collections import Counter, deque

import numpy as np


_GEN_RETRIES = 400
_BG = 0
_WALL = 5
_ARROW_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_MIN_ROOM_INNER = 5
_MAX_ROOM_INNER = 8
_MIN_SIGNAL_LEN = 2
_MAX_SIGNAL_LEN = 6


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_rooms(grid: np.ndarray, bg: int) -> list[dict[str, tuple[int, int, int, int]]]:
    h, w = grid.shape
    rooms: list[dict[str, tuple[int, int, int, int]]] = []
    for comp in _components(grid == bg):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        ir0, ir1 = min(rows), max(rows)
        ic0, ic1 = min(cols), max(cols)
        if len(comp) != (ir1 - ir0 + 1) * (ic1 - ic0 + 1):
            continue
        if not np.all(grid[ir0 : ir1 + 1, ic0 : ic1 + 1] == bg):
            continue
        br0, bc0, br1, bc1 = ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1
        if br0 < 0 or bc0 < 0 or br1 >= h or bc1 >= w:
            continue
        if np.any(grid[br0, bc0 : bc1 + 1] == bg):
            continue
        if np.any(grid[br1, bc0 : bc1 + 1] == bg):
            continue
        if np.any(grid[br0 : br1 + 1, bc0] == bg):
            continue
        if np.any(grid[br0 : br1 + 1, bc1] == bg):
            continue
        rooms.append({"inner": (ir0, ic0, ir1, ic1), "outer": (br0, bc0, br1, bc1)})
    rooms.sort(key=lambda room: room["outer"])
    return rooms


def _same_color(grid: np.ndarray, cells: list[tuple[int, int]], banned: set[int]) -> int | None:
    h, w = grid.shape
    values: list[int] = []
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return None
        values.append(int(grid[r, c]))
    color = values[0]
    if color in banned:
        return None
    if any(v != color for v in values[1:]):
        return None
    return color


def _room_arrows(grid: np.ndarray, room: dict[str, tuple[int, int, int, int]], bg: int, wall: int) -> list[dict[str, object]]:
    br0, bc0, br1, bc1 = room["outer"]
    arrows: list[dict[str, object]] = []
    seen_colors: set[int] = set()
    banned = {bg, wall}

    for r in range(br0 + 2, br1 - 1):
        color = _same_color(grid, [(r, bc0 - 1), (r - 1, bc0), (r + 1, bc0)], banned)
        if color is not None and color not in seen_colors:
            seen_colors.add(color)
            arrows.append(
                {
                    "color": color,
                    "direction": "right",
                    "sort_key": (r, bc0),
                    "rows": (r - 1, r, r + 1),
                    "cols": None,
                }
            )

        color = _same_color(grid, [(r, bc1 + 1), (r - 1, bc1), (r + 1, bc1)], banned)
        if color is not None and color not in seen_colors:
            seen_colors.add(color)
            arrows.append(
                {
                    "color": color,
                    "direction": "left",
                    "sort_key": (r, bc1),
                    "rows": (r - 1, r, r + 1),
                    "cols": None,
                }
            )

    for c in range(bc0 + 2, bc1 - 1):
        color = _same_color(grid, [(br0 - 1, c), (br0, c - 1), (br0, c + 1)], banned)
        if color is not None and color not in seen_colors:
            seen_colors.add(color)
            arrows.append(
                {
                    "color": color,
                    "direction": "down",
                    "sort_key": (br0, c),
                    "rows": None,
                    "cols": (c - 1, c, c + 1),
                }
            )

        color = _same_color(grid, [(br1 + 1, c), (br1, c - 1), (br1, c + 1)], banned)
        if color is not None and color not in seen_colors:
            seen_colors.add(color)
            arrows.append(
                {
                    "color": color,
                    "direction": "up",
                    "sort_key": (br1, c),
                    "rows": None,
                    "cols": (c - 1, c, c + 1),
                }
            )

    arrows.sort(key=lambda arrow: arrow["sort_key"])
    return arrows


def _paint_border(out: np.ndarray, outer: tuple[int, int, int, int], wall: int) -> None:
    br0, bc0, br1, bc1 = outer
    out[br0, bc0 : bc1 + 1] = wall
    out[br1, bc0 : bc1 + 1] = wall
    out[br0 : br1 + 1, bc0] = wall
    out[br0 : br1 + 1, bc1] = wall


def _fill_from_arrow(
    out: np.ndarray,
    room: dict[str, tuple[int, int, int, int]],
    arrow: dict[str, object],
    bg: int,
) -> None:
    ir0, ic0, ir1, ic1 = room["inner"]
    color = int(arrow["color"])
    direction = str(arrow["direction"])

    if direction == "right":
        for r in arrow["rows"]:
            rr = int(r)
            for c in range(ic0, ic1 + 1):
                if int(out[rr, c]) != bg:
                    break
                out[rr, c] = color
        return

    if direction == "left":
        for r in arrow["rows"]:
            rr = int(r)
            for c in range(ic1, ic0 - 1, -1):
                if int(out[rr, c]) != bg:
                    break
                out[rr, c] = color
        return

    if direction == "down":
        for c in arrow["cols"]:
            cc = int(c)
            for r in range(ir0, ir1 + 1):
                if int(out[r, cc]) != bg:
                    break
                out[r, cc] = color
        return

    if direction == "up":
        for c in arrow["cols"]:
            cc = int(c)
            for r in range(ir1, ir0 - 1, -1):
                if int(out[r, cc]) != bg:
                    break
                out[r, cc] = color
        return

    raise ValueError(f"unknown arrow direction: {direction}")


def _solve_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    counts = Counter(int(v) for v in g.ravel() if int(v) != bg)
    wall = counts.most_common(1)[0][0]
    rooms = _find_rooms(g, bg)

    out = np.full_like(g, bg)
    for room in rooms:
        _paint_border(out, room["outer"], wall)
        for arrow in _room_arrows(g, room, bg, wall):
            _fill_from_arrow(out, room, arrow, bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use same-color signal lines to drive directional stripe fills inside framed rooms.

    Input structure:
    - bg is the most common color.
    - Several rectangular rooms are enclosed by a shared wall color, with empty bg interiors.
    - Each room wall may carry one or two 3-pixel chevrons: two pixels stay on the wall and one matching pixel sits just outside the room, so the chevron orientation points inward.
    - Every chevron color is unique and continues outward as a straight 1-cell-wide signal line ending at a same-color seed pixel.

    Rule:
    - Detect each enclosed rectangular bg room and rebuild its wall as a solid frame in the output.
    - Match every seed/line to its room arrow by color, then read the arrow direction from the chevron geometry.
    - A left/right chevron fills the three interior rows aligned with that chevron from the touched wall inward; an up/down chevron similarly fills the three aligned interior columns.
    - Within one room, process multiple arrows in top-to-bottom then left-to-right order, and stop each stripe as soon as it would hit earlier paint.

    Key insight:
    - The outside line only identifies which color belongs to which wall marker; the chevron geometry on the frame tells you which inward stripe sweep to perform and earlier sweeps claim overlapping cells first.
    """
    return _solve_grid(np.asarray(grid, dtype=int))


def _room_dict(br0: int, bc0: int, inner_h: int, inner_w: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        "outer": (br0, bc0, br0 + inner_h + 1, bc0 + inner_w + 1),
        "inner": (br0 + 1, bc0 + 1, br0 + inner_h, bc0 + inner_w),
    }


def _draw_room(grid: np.ndarray, room: dict[str, tuple[int, int, int, int]]) -> None:
    _paint_border(grid, room["outer"], _WALL)


def _layout_rooms() -> list[dict[str, tuple[int, int, int, int]]] | None:
    num_rooms = 1 if random.random() < 0.45 else 2
    sizes = [
        (
            random.randint(_MIN_ROOM_INNER, _MAX_ROOM_INNER),
            random.randint(_MIN_ROOM_INNER, _MAX_ROOM_INNER),
        )
        for _ in range(num_rooms)
    ]

    if num_rooms == 1:
        inner_h, inner_w = sizes[0]
        frame_h, frame_w = inner_h + 2, inner_w + 2
        min_h = frame_h + 7
        min_w = frame_w + 7
        max_h = min(30, min_h + 6)
        max_w = min(30, min_w + 6)
        if min_h > 30 or min_w > 30:
            return None
        h = random.randint(min_h, max_h)
        w = random.randint(min_w, max_w)
        br0 = random.randint(3, h - frame_h - 4)
        bc0 = random.randint(3, w - frame_w - 4)
        return [_room_dict(br0, bc0, inner_h, inner_w)]

    (ih1, iw1), (ih2, iw2) = sizes
    fh1, fw1 = ih1 + 2, iw1 + 2
    fh2, fw2 = ih2 + 2, iw2 + 2
    orient = random.choice(("horizontal", "vertical"))
    gap = random.randint(4, 7)
    top = random.randint(3, 5)
    bottom = random.randint(3, 5)
    left = random.randint(3, 5)
    right = random.randint(3, 5)

    if orient == "horizontal":
        min_h = top + max(fh1, fh2) + bottom
        min_w = left + fw1 + gap + fw2 + right
        if min_h > 30 or min_w > 30:
            return None
        h = random.randint(min_h, min(30, min_h + 4))
        w = random.randint(min_w, min(30, min_w + 4))
        extra_h1 = h - top - bottom - fh1
        extra_h2 = h - top - bottom - fh2
        br0_1 = top + random.randint(0, extra_h1)
        br0_2 = top + random.randint(0, extra_h2)
        extra_w = w - left - right - fw1 - gap - fw2
        pre = random.randint(0, extra_w)
        between = random.randint(0, extra_w - pre)
        bc0_1 = left + pre
        bc0_2 = bc0_1 + fw1 + gap + between
        return [_room_dict(br0_1, bc0_1, ih1, iw1), _room_dict(br0_2, bc0_2, ih2, iw2)]

    min_h = top + fh1 + gap + fh2 + bottom
    min_w = left + max(fw1, fw2) + right
    if min_h > 30 or min_w > 30:
        return None
    h = random.randint(min_h, min(30, min_h + 4))
    w = random.randint(min_w, min(30, min_w + 4))
    extra_w1 = w - left - right - fw1
    extra_w2 = w - left - right - fw2
    bc0_1 = left + random.randint(0, extra_w1)
    bc0_2 = left + random.randint(0, extra_w2)
    extra_h = h - top - bottom - fh1 - gap - fh2
    pre = random.randint(0, extra_h)
    between = random.randint(0, extra_h - pre)
    br0_1 = top + pre
    br0_2 = br0_1 + fh1 + gap + between
    return [_room_dict(br0_1, bc0_1, ih1, iw1), _room_dict(br0_2, bc0_2, ih2, iw2)]


def _signal_capacity(grid: np.ndarray, start: tuple[int, int], step: tuple[int, int]) -> int:
    h, w = grid.shape
    r, c = start
    dr, dc = step
    cap = 0
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == _BG:
        cap += 1
        r += dr
        c += dc
    return cap


def _arrow_candidates(grid: np.ndarray, room: dict[str, tuple[int, int, int, int]]) -> list[dict[str, object]]:
    br0, bc0, br1, bc1 = room["outer"]
    candidates: list[dict[str, object]] = []

    rows = list(range(br0 + 2, br1 - 1))
    cols = list(range(bc0 + 2, bc1 - 1))
    random.shuffle(rows)
    random.shuffle(cols)

    for r in rows:
        wing_a = (r - 1, bc0)
        wing_b = (r + 1, bc0)
        outer = (r, bc0 - 1)
        if int(grid[outer]) == _BG and int(grid[wing_a]) == _WALL and int(grid[wing_b]) == _WALL:
            cap = _signal_capacity(grid, (r, bc0 - 2), (0, -1))
            if cap >= _MIN_SIGNAL_LEN:
                candidates.append(
                    {
                        "side": "left",
                        "direction": "right",
                        "axis": "rows",
                        "rows": (r - 1, r, r + 1),
                        "cols": None,
                        "arrow_cells": (outer, wing_a, wing_b),
                        "signal_start": (r, bc0 - 2),
                        "signal_step": (0, -1),
                        "capacity": cap,
                    }
                )

        wing_a = (r - 1, bc1)
        wing_b = (r + 1, bc1)
        outer = (r, bc1 + 1)
        if int(grid[outer]) == _BG and int(grid[wing_a]) == _WALL and int(grid[wing_b]) == _WALL:
            cap = _signal_capacity(grid, (r, bc1 + 2), (0, 1))
            if cap >= _MIN_SIGNAL_LEN:
                candidates.append(
                    {
                        "side": "right",
                        "direction": "left",
                        "axis": "rows",
                        "rows": (r - 1, r, r + 1),
                        "cols": None,
                        "arrow_cells": (outer, wing_a, wing_b),
                        "signal_start": (r, bc1 + 2),
                        "signal_step": (0, 1),
                        "capacity": cap,
                    }
                )

    for c in cols:
        wing_a = (br0, c - 1)
        wing_b = (br0, c + 1)
        outer = (br0 - 1, c)
        if int(grid[outer]) == _BG and int(grid[wing_a]) == _WALL and int(grid[wing_b]) == _WALL:
            cap = _signal_capacity(grid, (br0 - 2, c), (-1, 0))
            if cap >= _MIN_SIGNAL_LEN:
                candidates.append(
                    {
                        "side": "top",
                        "direction": "down",
                        "axis": "cols",
                        "rows": None,
                        "cols": (c - 1, c, c + 1),
                        "arrow_cells": (outer, wing_a, wing_b),
                        "signal_start": (br0 - 2, c),
                        "signal_step": (-1, 0),
                        "capacity": cap,
                    }
                )

        wing_a = (br1, c - 1)
        wing_b = (br1, c + 1)
        outer = (br1 + 1, c)
        if int(grid[outer]) == _BG and int(grid[wing_a]) == _WALL and int(grid[wing_b]) == _WALL:
            cap = _signal_capacity(grid, (br1 + 2, c), (1, 0))
            if cap >= _MIN_SIGNAL_LEN:
                candidates.append(
                    {
                        "side": "bottom",
                        "direction": "up",
                        "axis": "cols",
                        "rows": None,
                        "cols": (c - 1, c, c + 1),
                        "arrow_cells": (outer, wing_a, wing_b),
                        "signal_start": (br1 + 2, c),
                        "signal_step": (1, 0),
                        "capacity": cap,
                    }
                )

    return candidates


def _compatible_second(first: dict[str, object], second: dict[str, object]) -> bool:
    if str(first["side"]) == str(second["side"]):
        return False
    if str(first["axis"]) != str(second["axis"]):
        return True
    if str(first["axis"]) == "rows":
        return set(first["rows"]).isdisjoint(second["rows"])
    return set(first["cols"]).isdisjoint(second["cols"])


def _paint_arrow(grid: np.ndarray, candidate: dict[str, object], color: int) -> dict[str, object]:
    for r, c in candidate["arrow_cells"]:
        grid[r, c] = color
    sig_len = random.randint(_MIN_SIGNAL_LEN, min(_MAX_SIGNAL_LEN, int(candidate["capacity"])))
    sr, sc = candidate["signal_start"]
    dr, dc = candidate["signal_step"]
    for k in range(sig_len):
        grid[sr + dr * k, sc + dc * k] = color
    meta = dict(candidate)
    meta["color"] = color
    meta["signal_length"] = sig_len
    return meta


def _place_room_arrows(
    grid: np.ndarray,
    room: dict[str, tuple[int, int, int, int]],
    colors: list[int],
    arrow_count: int,
) -> list[dict[str, object]] | None:
    for _ in range(40):
        snapshot = grid.copy()
        placed: list[dict[str, object]] = []
        ok = True
        for idx in range(arrow_count):
            candidates = _arrow_candidates(snapshot, room)
            if idx == 1:
                candidates = [cand for cand in candidates if _compatible_second(placed[0], cand)]
                if not candidates:
                    ok = False
                    break
                orth = [cand for cand in candidates if str(cand["axis"]) != str(placed[0]["axis"])]
                if orth and random.random() < 0.75:
                    candidates = orth
            if not candidates:
                ok = False
                break
            choice = random.choice(candidates)
            placed.append(_paint_arrow(snapshot, choice, colors[idx]))
        if ok:
            grid[:, :] = snapshot
            return placed
    return None


def _gen_attempt() -> np.ndarray | None:
    rooms = _layout_rooms()
    if rooms is None:
        return None

    brs = [room["outer"] for room in rooms]
    h = max(br1 for _, _, br1, _ in brs) + 4
    w = max(bc1 for _, _, _, bc1 in brs) + 4
    grid = np.full((h, w), _BG, dtype=int)
    for room in rooms:
        _draw_room(grid, room)

    arrow_counts = [1 if random.random() < 0.5 else 2 for _ in rooms]
    total_arrows = sum(arrow_counts)
    if total_arrows > len(_ARROW_COLORS):
        return None
    color_pool = list(_ARROW_COLORS)
    random.shuffle(color_pool)

    metas: list[dict[str, object]] = []
    for room, arrow_count in zip(rooms, arrow_counts):
        room_colors = [color_pool.pop() for _ in range(arrow_count)]
        placed = _place_room_arrows(grid, room, room_colors, arrow_count)
        if placed is None:
            return None
        metas.extend(placed)

    out = _solve_grid(grid)
    for meta in metas:
        if int(np.count_nonzero(out == int(meta["color"]))) < 3:
            return None
    if len(np.unique(out)) <= 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-signal input")
