from __future__ import annotations

import random

import numpy as np


BG = 0
ROOM = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _runs(arr: np.ndarray, value: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(arr):
        if int(arr[i]) != value:
            i += 1
            continue
        j = i
        while j < len(arr) and int(arr[j]) == value:
            j += 1
        out.append((i, j))
        i = j
    return out


def _nonzero_runs(arr: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(arr):
        if int(arr[i]) == BG:
            i += 1
            continue
        j = i
        while j < len(arr) and int(arr[j]) != BG:
            j += 1
        out.append((i, j))
        i = j
    return out


def _find_template(grid: np.ndarray) -> tuple[str, int, list[int]]:
    top_runs = _nonzero_runs(grid[0, :])
    left_runs = _nonzero_runs(grid[:, 0])
    best_top = max(top_runs, key=lambda run: run[1] - run[0], default=None)
    best_left = max(left_runs, key=lambda run: run[1] - run[0], default=None)
    top_len = 0 if best_top is None else best_top[1] - best_top[0]
    left_len = 0 if best_left is None else best_left[1] - best_left[0]

    if top_len >= max(2, left_len):
        assert best_top is not None
        start, end = best_top
        colors = [int(x) for x in grid[0, start:end]]
        assert len(colors) >= 4
        return "top", int(start), colors

    assert best_left is not None and left_len >= 2
    start, end = best_left
    colors = [int(x) for x in grid[start:end, 0]]
    assert len(colors) >= 4
    return "left", int(start), colors


def _extract_gap(side: np.ndarray) -> tuple[int, int] | None:
    if len(side) == 0:
        return None
    missing = np.where(side == BG)[0]
    if len(missing) == 0:
        return None
    if not np.all(np.isin(side, (BG, ROOM))):
        return None
    if missing[-1] - missing[0] + 1 != len(missing):
        return None
    if not np.all(side[: missing[0]] == ROOM):
        return None
    if not np.all(side[missing[-1] + 1 :] == ROOM):
        return None
    return int(missing[0]), int(missing[-1] + 1)


def _analyze_room(grid: np.ndarray, pts: list[tuple[int, int]]) -> dict[str, int] | None:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]

    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return None
    if not np.all(np.isin(patch, (BG, ROOM))):
        return None
    if not np.all(patch[1:-1, 1:-1] == BG):
        return None
    if not np.all(patch[[0, 0, -1, -1], [0, -1, 0, -1]] == ROOM):
        return None

    sides = {
        "top": patch[0, 1:-1],
        "bottom": patch[-1, 1:-1],
        "left": patch[1:-1, 0],
        "right": patch[1:-1, -1],
    }

    gap_side = None
    gap_span = None
    for name, side in sides.items():
        gap = _extract_gap(side)
        if gap is None:
            if not np.all(side == ROOM):
                return None
            continue
        if gap_side is not None:
            return None
        gap_side = name
        gap_span = gap

    if gap_side is None or gap_span is None:
        return None

    start, end = gap_span
    width = end - start
    if width < 1 or width > 4:
        return None

    return {
        "r0": int(r0),
        "r1": int(r1),
        "c0": int(c0),
        "c1": int(c1),
        "side": gap_side,
        "start": int(start),
        "end": int(end),
        "width": int(width),
    }


def _paint_beam(out: np.ndarray, room: dict[str, int], color: int) -> None:
    r0 = room["r0"]
    r1 = room["r1"]
    c0 = room["c0"]
    c1 = room["c1"]
    start = room["start"]
    end = room["end"]
    side = room["side"]

    if side in ("top", "bottom"):
        cols = slice(c0 + 1 + start, c0 + 1 + end)
        out[r0 + 1 : r1, cols] = color
    else:
        rows = slice(r0 + 1 + start, r0 + 1 + end)
        out[rows, c0 + 1 : c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Index template colors by room-gap width, then shoot that color through the room.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    _template_side, _template_start, palette = _find_template(g)
    rooms = []
    for comp in _components(g == ROOM):
        room = _analyze_room(g, comp)
        if room is not None:
            rooms.append(room)

    assert rooms, "expected at least one valid grey room"
    for room in rooms:
        width = room["width"]
        color = int(palette[width - 1])
        _paint_beam(out, room, color)

    return out


def _make_template(grid: np.ndarray, side: str) -> list[int]:
    colors = random.sample(PALETTE_COLORS, 4)
    if side == "top":
        start = random.randint(0, grid.shape[1] - 4)
        grid[0, start : start + 4] = colors
    else:
        start = random.randint(0, grid.shape[0] - 4)
        grid[start : start + 4, 0] = colors
    return [int(c) for c in colors]


def _choose_widths(count: int) -> list[int]:
    widths: list[int] = []
    bag: list[int] = []
    while len(widths) < count:
        if not bag:
            bag = [1, 2, 3, 4]
            random.shuffle(bag)
        widths.append(int(bag.pop()))
    random.shuffle(widths)
    return widths


def _place_room(
    grid: np.ndarray,
    blocked: np.ndarray,
    gap_side: str,
    gap_width: int,
) -> dict[str, int] | None:
    h, w = grid.shape

    if gap_side in ("top", "bottom"):
        min_h = 5
        min_w = max(6, gap_width + 4)
    else:
        min_h = max(6, gap_width + 4)
        min_w = 5

    max_h = min(11, h - 2)
    max_w = min(11, w - 2)
    if min_h > max_h or min_w > max_w:
        return None

    room_h = random.randint(min_h, max_h)
    room_w = random.randint(min_w, max_w)

    for _ in range(80):
        r0 = random.randint(0, h - room_h)
        c0 = random.randint(0, w - room_w)
        r1 = r0 + room_h - 1
        c1 = c0 + room_w - 1

        pr0 = max(0, r0 - 1)
        pc0 = max(0, c0 - 1)
        pr1 = min(h - 1, r1 + 1)
        pc1 = min(w - 1, c1 + 1)
        if np.any(blocked[pr0 : pr1 + 1, pc0 : pc1 + 1]):
            continue

        if gap_side in ("top", "bottom"):
            offset = random.randint(1, room_w - 3 - gap_width)
            gap_r = r0 if gap_side == "top" else r1
            gap_c0 = c0 + 1 + offset
            gap_c1 = gap_c0 + gap_width - 1
        else:
            offset = random.randint(1, room_h - 3 - gap_width)
            gap_c = c0 if gap_side == "left" else c1
            gap_r0 = r0 + 1 + offset
            gap_r1 = gap_r0 + gap_width - 1

        grid[r0, c0 : c1 + 1] = ROOM
        grid[r1, c0 : c1 + 1] = ROOM
        grid[r0 : r1 + 1, c0] = ROOM
        grid[r0 : r1 + 1, c1] = ROOM
        if gap_side in ("top", "bottom"):
            grid[gap_r, gap_c0 : gap_c1 + 1] = BG
        else:
            grid[gap_r0 : gap_r1 + 1, gap_c] = BG

        blocked[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
        return {
            "r0": int(r0),
            "r1": int(r1),
            "c0": int(c0),
            "c1": int(c1),
            "side": gap_side,
            "start": int(offset),
            "end": int(offset + gap_width),
            "width": int(gap_width),
        }

    return None


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(16, 28)
    w = random.randint(16, 28)
    grid = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)

    template_side = random.choice(("top", "left"))
    palette = _make_template(grid, template_side)
    out[:] = grid

    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True
    if template_side == "top":
        blocked[0:2, :] = True
    else:
        blocked[:, 0:2] = True

    room_count = random.randint(2, 5)
    widths = _choose_widths(room_count)
    rooms: list[dict[str, int]] = []
    for width in widths:
        side = random.choice(SIDES)
        room = _place_room(grid, blocked, side, width)
        if room is None:
            return None
        rooms.append(room)

    out = grid.copy()
    for room in rooms:
        _paint_beam(out, room, palette[room["width"] - 1])

    try:
        if not np.array_equal(main(grid), out):
            return None
    except Exception:
        return None
    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            grid, _out = sample
            return grid
    raise RuntimeError("failed to generate input")
