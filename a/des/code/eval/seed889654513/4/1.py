from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
RED = 2
GREY = 5
EXIT = 9

PALETTE_POOL = (1, 3, 4, 6, 7, 8)
SIDES = ("top", "bottom", "left", "right")
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_ATTEMPTS = 400
MIN_ROOM_SIZE = 2


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _room_sort_key(cells: list[tuple[int, int]]) -> tuple[int, int, int]:
    anchor_r, anchor_c = min(cells)
    return (-len(cells), anchor_r, anchor_c)


def _find_palette(grid: np.ndarray) -> list[int]:
    palette: list[tuple[int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, RED, GREY):
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            if len(comp) == 2 and (r1 - r0 + 1) == 2 and (c1 - c0 + 1) == 1:
                palette.append((r0, c0, color))
    palette.sort()
    return [color for _r, _c, color in palette]


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == GREY):
            continue
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not np.any(inner == BG):
            continue
        rooms: list[list[tuple[int, int]]] = []
        room_lookup: dict[tuple[int, int], int] = {}
        for room_idx, local_comp in enumerate(_components(inner == BG)):
            global_cells = [(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in local_comp]
            rooms.append(global_cells)
            for cell in global_cells:
                room_lookup[cell] = room_idx
        if not rooms:
            continue
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "rooms": rooms,
                "room_lookup": room_lookup,
            }
        )
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _find_arrows(grid: np.ndarray) -> list[dict]:
    arrows: list[dict] = []
    for comp in _components(grid == RED):
        if len(comp) != 3:
            continue
        arrows.append({"cells": comp, "bbox": _bbox(comp)})
    arrows.sort(key=lambda arrow: arrow["bbox"])
    return arrows


def _arrow_relation(arrow: dict, frame_bbox: tuple[int, int, int, int]) -> tuple[int, tuple[int, int], tuple[int, int]] | None:
    r0, c0, r1, c1 = frame_bbox
    ar0, ac0, ar1, ac1 = arrow["bbox"]
    cells = arrow["cells"]

    if ac1 < c0 and not (ar1 < r0 or ar0 > r1):
        tip = max(cells, key=lambda p: (p[1], p[0]))
        return c0 - ac1, (0, 1), tip
    if ac0 > c1 and not (ar1 < r0 or ar0 > r1):
        tip = min(cells, key=lambda p: (p[1], p[0]))
        return ac0 - c1, (0, -1), tip
    if ar1 < r0 and not (ac1 < c0 or ac0 > c1):
        tip = max(cells, key=lambda p: (p[0], p[1]))
        return r0 - ar1, (1, 0), tip
    if ar0 > r1 and not (ac1 < c0 or ac0 > c1):
        tip = min(cells, key=lambda p: (p[0], p[1]))
        return ar0 - r1, (-1, 0), tip
    return None


def _associate_arrows(frames: list[dict], arrows: list[dict]) -> dict[int, dict]:
    mapping: dict[int, dict] = {}
    for arrow in arrows:
        best: tuple[int, int, tuple[int, int], tuple[int, int]] | None = None
        for frame_idx, frame in enumerate(frames):
            relation = _arrow_relation(arrow, frame["bbox"])
            if relation is None:
                continue
            dist, direction, tip = relation
            cand = (dist, frame_idx, direction, tip)
            if best is None or cand < best:
                best = cand
        assert best is not None, "each arrow must align with a frame"
        _dist, frame_idx, direction, tip = best
        assert frame_idx not in mapping, "each frame must have exactly one arrow"
        mapping[frame_idx] = {"direction": direction, "tip": tip}
    assert len(mapping) == len(frames), "each frame must receive exactly one arrow"
    return mapping


def _ray_hit_room(
    grid: np.ndarray,
    frame: dict,
    tip: tuple[int, int],
    direction: tuple[int, int],
) -> int:
    h, w = grid.shape
    r0, c0, r1, c1 = frame["bbox"]
    room_lookup = frame["room_lookup"]
    rr, cc = tip
    entered = False
    while True:
        rr += direction[0]
        cc += direction[1]
        if not (0 <= rr < h and 0 <= cc < w):
            break
        inside = r0 <= rr <= r1 and c0 <= cc <= c1
        if inside:
            entered = True
            room_idx = room_lookup.get((rr, cc))
            if room_idx is not None:
                return int(room_idx)
        elif entered:
            break
    raise AssertionError("arrow ray did not reach any room")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill partitioned rectangular rooms by area-ranked palette order, then mark the arrow-hit room as the exit.
    """
    g = np.asarray(grid, dtype=int)
    out = np.array(g, copy=True)

    palette = _find_palette(g)
    frames = _find_frames(g)
    arrows = _find_arrows(g)

    assert palette, "expected a palette"
    assert frames, "expected at least one frame"
    assert len(arrows) == len(frames), "expected one arrow per frame"

    arrow_by_frame = _associate_arrows(frames, arrows)

    out[g == RED] = BG
    for frame_idx, frame in enumerate(frames):
        rooms = frame["rooms"]
        assert len(rooms) <= len(palette), "palette too short for room count"
        order = sorted(range(len(rooms)), key=lambda idx: _room_sort_key(rooms[idx]))
        for rank, room_idx in enumerate(order):
            color = palette[rank]
            for r, c in rooms[room_idx]:
                out[r, c] = color
        exit_room_idx = _ray_hit_room(
            g,
            frame,
            arrow_by_frame[frame_idx]["tip"],
            arrow_by_frame[frame_idx]["direction"],
        )
        for r, c in rooms[exit_room_idx]:
            out[r, c] = EXIT
    return out


def _touches_side(room: tuple[int, int, int, int], side: str, inner_h: int, inner_w: int) -> bool:
    r0, c0, r1, c1 = room
    if side == "top":
        return r0 == 0
    if side == "bottom":
        return r1 == inner_h - 1
    if side == "left":
        return c0 == 0
    return c1 == inner_w - 1


def _partition_attempt(inner_h: int, inner_w: int, room_count: int, want_tie: bool) -> tuple[list[tuple[int, int, int, int]], list[tuple[str, int, int, int]]] | None:
    rooms = [(0, 0, inner_h - 1, inner_w - 1)]
    walls: list[tuple[str, int, int, int]] = []
    tie_done = room_count == 1

    for _ in range(room_count - 1):
        options: list[tuple[str, int, int, bool]] = []
        for room_idx, (r0, c0, r1, c1) in enumerate(rooms):
            for cut in range(r0 + MIN_ROOM_SIZE, r1 - MIN_ROOM_SIZE + 1):
                top_h = cut - r0
                bottom_h = r1 - cut
                options.append(("h", room_idx, cut, top_h == bottom_h))
            for cut in range(c0 + MIN_ROOM_SIZE, c1 - MIN_ROOM_SIZE + 1):
                left_w = cut - c0
                right_w = c1 - cut
                options.append(("v", room_idx, cut, left_w == right_w))
        if want_tie and not tie_done:
            tie_options = [opt for opt in options if opt[3]]
            if tie_options:
                options = tie_options
        if not options:
            return None
        axis, room_idx, cut, is_tie = random.choice(options)
        r0, c0, r1, c1 = rooms.pop(room_idx)
        if axis == "h":
            walls.append(("h", cut, c0, c1))
            children = [(r0, c0, cut - 1, c1), (cut + 1, c0, r1, c1)]
        else:
            walls.append(("v", cut, r0, r1))
            children = [(r0, c0, r1, cut - 1), (r0, cut + 1, r1, c1)]
        rooms.extend(children)
        tie_done = tie_done or is_tie

    if want_tie and not tie_done:
        return None
    if len(rooms) != room_count:
        return None
    return rooms, walls


def _build_frame_spec(room_count: int, want_tie: bool) -> dict | None:
    for _ in range(120):
        base = max(4, room_count + 2)
        inner_h = random.randint(base, base + 4)
        inner_w = random.randint(base, base + 4)
        partition = _partition_attempt(inner_h, inner_w, room_count, want_tie)
        if partition is None:
            continue
        rooms, walls = partition
        frame = np.full((inner_h + 2, inner_w + 2), GREY, dtype=int)
        frame[1:-1, 1:-1] = BG
        for axis, cut, span0, span1 in walls:
            if axis == "h":
                frame[1 + cut, 1 + span0 : 2 + span1] = GREY
            else:
                frame[1 + span0 : 2 + span1, 1 + cut] = GREY

        side = random.choice(SIDES)
        candidates = [idx for idx, room in enumerate(rooms) if _touches_side(room, side, inner_h, inner_w)]
        if not candidates:
            continue
        exit_room_idx = random.choice(candidates)
        bend = random.choice((-1, 1))
        rr0, cc0, rr1, cc1 = rooms[exit_room_idx]
        if side in ("left", "right"):
            line = random.randint(rr0, rr1)
        else:
            line = random.randint(cc0, cc1)

        fh, fw = frame.shape
        if side == "left":
            arrow = [(1 + line, -1), (1 + line, -2), (1 + line + bend, -2)]
        elif side == "right":
            arrow = [(1 + line, fw), (1 + line, fw + 1), (1 + line + bend, fw + 1)]
        elif side == "top":
            arrow = [(-1, 1 + line), (-2, 1 + line), (-2, 1 + line + bend)]
        else:
            arrow = [(fh, 1 + line), (fh + 1, 1 + line), (fh + 1, 1 + line + bend)]

        return {
            "frame": frame,
            "rooms": rooms,
            "side": side,
            "exit_room_idx": exit_room_idx,
            "arrow_local": arrow,
        }
    return None


def _paint_room(out: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        out[r, c] = color


def _global_room_cells(top: int, left: int, room: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room
    return [
        (top + 1 + rr, left + 1 + cc)
        for rr in range(r0, r1 + 1)
        for cc in range(c0, c1 + 1)
    ]


def _expected_output(grid: np.ndarray, palette: list[int], placed_specs: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    out[out == RED] = BG
    for spec in placed_specs:
        room_cells = spec["room_cells"]
        order = sorted(range(len(room_cells)), key=lambda idx: _room_sort_key(room_cells[idx]))
        for rank, room_idx in enumerate(order):
            _paint_room(out, room_cells[room_idx], palette[rank])
        _paint_room(out, room_cells[spec["exit_room_idx"]], EXIT)
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(28, 38)
    width = random.randint(28, 38)
    palette_len = random.randint(3, len(PALETTE_POOL))
    palette = random.sample(PALETTE_POOL, palette_len)
    n_frames = random.randint(1, 3)

    room_counts = [random.randint(1, palette_len) for _ in range(n_frames)]
    if random.choice((False, True)) and all(count == 1 for count in room_counts):
        room_counts[random.randrange(n_frames)] = random.randint(2, palette_len)

    force_tie = random.choice((False, True))
    if force_tie and all(count < 2 for count in room_counts):
        room_counts[random.randrange(n_frames)] = random.randint(2, palette_len)
    tie_frame = random.randrange(n_frames) if force_tie else -1

    specs: list[dict] = []
    for idx, room_count in enumerate(room_counts):
        want_tie = idx == tie_frame and room_count >= 2
        spec = _build_frame_spec(room_count, want_tie)
        if spec is None:
            return None
        specs.append(spec)

    grid = np.full((height, width), BG, dtype=int)

    palette_h = 2 * palette_len + (palette_len - 1)
    if palette_h + 2 >= height:
        return None
    for idx, color in enumerate(palette):
        start_r = 3 * idx
        grid[start_r : start_r + 2, 0] = color

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[: palette_h + 2, :4] = True

    order = sorted(
        range(len(specs)),
        key=lambda idx: specs[idx]["frame"].shape[0] * specs[idx]["frame"].shape[1],
        reverse=True,
    )
    placed: list[dict | None] = [None] * len(specs)
    for idx in order:
        frame = specs[idx]["frame"]
        fh, fw = frame.shape
        if height - fh - 4 < 0 or width - fw - 4 < 0:
            return None
        candidates: list[tuple[int, int]] = []
        for _ in range(160):
            top = random.randint(2, height - fh - 2)
            left = random.randint(2, width - fw - 2)
            reserve = occupied[top - 2 : top + fh + 2, left - 2 : left + fw + 2]
            if reserve.any():
                continue
            candidates.append((top, left))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        occupied[top - 2 : top + fh + 2, left - 2 : left + fw + 2] = True
        grid[top : top + fh, left : left + fw] = frame

        arrow_cells: list[tuple[int, int]] = []
        for lr, lc in specs[idx]["arrow_local"]:
            gr = top + lr
            gc = left + lc
            if not (0 <= gr < height and 0 <= gc < width):
                return None
            if grid[gr, gc] != BG:
                return None
            arrow_cells.append((gr, gc))
        for gr, gc in arrow_cells:
            grid[gr, gc] = RED

        room_cells = [_global_room_cells(top, left, room) for room in specs[idx]["rooms"]]
        placed[idx] = {
            "room_cells": room_cells,
            "exit_room_idx": specs[idx]["exit_room_idx"],
        }

    placed_specs = [spec for spec in placed if spec is not None]
    expected = _expected_output(grid, palette, placed_specs)
    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate valid input")
