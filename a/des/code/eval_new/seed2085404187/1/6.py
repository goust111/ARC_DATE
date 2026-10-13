from __future__ import annotations

import random
from typing import Iterable

import numpy as np


WALL = 9
DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}
GEN_RETRIES = 500


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[counts.argmax()])


def _wall_color(grid: np.ndarray, bg: int) -> int:
    non_bg = np.asarray(grid, dtype=int)
    non_bg = non_bg[non_bg != int(bg)]
    if non_bg.size == 0:
        return WALL
    vals, counts = np.unique(non_bg, return_counts=True)
    if int(WALL) in set(map(int, vals.tolist())):
        return WALL
    return int(vals[counts.argmax()])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _room_gap(bbox: tuple[int, int, int, int], side: str, offset: int) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return (int(r0), int(c0 + offset))
    if side == "bottom":
        return (int(r1), int(c0 + offset))
    if side == "left":
        return (int(r0 + offset), int(c0))
    return (int(r0 + offset), int(c1))


def _extract_rooms(grid: np.ndarray, bg: int, wall: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    rooms: list[dict] = []
    for pts in _components(g == int(wall)):
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if any(r not in (r0, r1) and c not in (c0, c1) for r, c in pts):
            continue

        gap_positions: list[tuple[int, int]] = []
        for c in range(c0, c1 + 1):
            if int(g[r0, c]) != int(wall):
                gap_positions.append((int(r0), int(c)))
            if int(g[r1, c]) != int(wall):
                gap_positions.append((int(r1), int(c)))
        for r in range(r0 + 1, r1):
            if int(g[r, c0]) != int(wall):
                gap_positions.append((int(r), int(c0)))
            if int(g[r, c1]) != int(wall):
                gap_positions.append((int(r), int(c1)))
        if len(gap_positions) != 1:
            continue

        expected_wall = 2 * (r1 - r0 + 1) + 2 * (c1 - c0 + 1) - 5
        if len(pts) != expected_wall:
            continue

        gap = gap_positions[0]
        if gap[0] == r0:
            side = "top"
        elif gap[0] == r1:
            side = "bottom"
        elif gap[1] == c0:
            side = "left"
        else:
            side = "right"

        payload: list[tuple[int, int, int]] = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                v = int(g[r, c])
                if v != int(bg) and v != int(wall):
                    payload.append((int(r), int(c), v))

        rooms.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "gap": (int(gap[0]), int(gap[1])),
                "side": side,
                "dir": DIRS[side],
                "payload": payload,
            }
        )

    rooms.sort(key=lambda room: (room["bbox"][0], room["bbox"][1]))
    return rooms


def _find_target(room: dict, gap_map: dict[tuple[int, int], dict], grid: np.ndarray, wall: int) -> dict | None:
    h, w = grid.shape
    dr, dc = room["dir"]
    r, c = room["gap"]
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return None
        candidate = gap_map.get((int(r), int(c)))
        if candidate is not None and candidate is not room and candidate["dir"] == (-dr, -dc):
            return candidate
        if int(grid[r, c]) == int(wall):
            return None


def _translation_to_target(
    payload: Iterable[tuple[int, int, int]],
    target_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> tuple[int, int]:
    payload = list(payload)
    rows = [r for r, _, _ in payload]
    cols = [c for _, c, _ in payload]
    dr, dc = direction
    r0, c0, r1, c1 = target_bbox
    if (dr, dc) == (0, 1):
        return (0, int(c1 - 1 - max(cols)))
    if (dr, dc) == (0, -1):
        return (0, int(c0 + 1 - min(cols)))
    if (dr, dc) == (1, 0):
        return (int(r1 - 1 - max(rows)), 0)
    return (int(r0 + 1 - min(rows)), 0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push colored pixels out of gapped maroon rooms and let each rigid block slide into the opposite-facing room on the same channel.

    Input structure:
    - bg is the most common color.
    - Maroon rectangular frames form closed rooms whose 1-pixel border has exactly one bg gap on one side.
    - Some rooms contain colored pixels on bg; other aligned rooms are empty receivers.

    Rule:
    - For each room with colored pixels, use the gap side to determine the push direction.
    - Treat all colored pixels in that room as one rigid block and move it straight along that direction.
    - The block slides through the straight bg channel until it reaches the first room whose gap lies on the same line and faces back toward the source, then continues inside until its front edge hits the opposite wall.
    - Clear the source room's colored pixels to bg and keep the walls unchanged.

    Key insight:
    The missing border pixel is both the movement-direction cue and the alignment cue for pairing each source room with its opposite-facing receiver.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    wall = _wall_color(g, bg)
    rooms = _extract_rooms(g, bg, wall)
    if not rooms:
        return g.copy()

    gap_map = {room["gap"]: room for room in rooms}
    out = g.copy()
    clears: list[tuple[int, int]] = []
    placements: list[tuple[int, int, int]] = []

    for room in rooms:
        if not room["payload"]:
            continue
        target = _find_target(room, gap_map, g, wall)
        if target is None:
            continue
        dy, dx = _translation_to_target(room["payload"], target["bbox"], room["dir"])
        clears.extend((r, c) for r, c, _ in room["payload"])
        placements.extend((r + dy, c + dx, color) for r, c, color in room["payload"])

    for r, c in clears:
        out[r, c] = bg
    for r, c, color in placements:
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            out[r, c] = int(color)
    return out


def _reserve(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = mask.shape
    r0 = max(0, bbox[0] - pad)
    c0 = max(0, bbox[1] - pad)
    r1 = min(h - 1, bbox[2] + pad)
    c1 = min(w - 1, bbox[3] + pad)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _free(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    h, w = mask.shape
    r0 = max(0, bbox[0] - pad)
    c0 = max(0, bbox[1] - pad)
    r1 = min(h - 1, bbox[2] + pad)
    c1 = min(w - 1, bbox[3] + pad)
    return not bool(mask[r0 : r1 + 1, c0 : c1 + 1].any())


def _color_sequence(rng: random.Random, count: int, bg: int) -> list[int]:
    palette = [c for c in range(10) if c not in (int(bg), int(WALL))]
    distinct = rng.randint(2, min(4, count, len(palette)))
    chosen = rng.sample(palette, distinct)
    seq = [int(rng.choice(chosen)) for _ in range(count)]
    if len(set(seq)) == 1:
        seq[-1] = chosen[-1]
    return seq


def _payload_offsets(rng: random.Random, span: int) -> list[int]:
    count = rng.randint(2, min(span, 4))
    offsets = {0, span - 1}
    while len(offsets) < count:
        offsets.add(rng.randrange(span))
    return sorted(offsets)


def _draw_room(grid: np.ndarray, bbox: tuple[int, int, int, int], gap: tuple[int, int], wall: int, bg: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = wall
    grid[r1, c0 : c1 + 1] = wall
    grid[r0 : r1 + 1, c0] = wall
    grid[r0 : r1 + 1, c1] = wall
    grid[gap[0], gap[1]] = bg


def _pair_translation(pair: dict) -> tuple[int, int]:
    return _translation_to_target(pair["payload"], pair["target_bbox"], DIRS[pair["source_side"]])


def _expected_from_pairs(grid: np.ndarray, pairs: list[dict], bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for pair in pairs:
        for r, c, _ in pair["payload"]:
            out[r, c] = int(bg)
    for pair in pairs:
        dy, dx = _pair_translation(pair)
        for r, c, color in pair["payload"]:
            out[r + dy, c + dx] = int(color)
    return out


def _horizontal_pair(
    rng: random.Random,
    h: int,
    w: int,
    occupied: np.ndarray,
    used_rows: set[int],
    direction: str,
    bg: int,
) -> dict | None:
    left_w = rng.randint(5, 8)
    right_w = rng.randint(5, 8)
    room_h = rng.randint(4, 7)
    corridor = rng.randint(2, 6)
    total_w = left_w + corridor + right_w
    if room_h + 2 > h or total_w + 2 > w:
        return None

    for _ in range(80):
        r0 = rng.randint(1, h - room_h - 1)
        gap_off = rng.randint(1, room_h - 2)
        gap_row = r0 + gap_off
        if gap_row in used_rows:
            continue
        c0 = rng.randint(1, w - total_w - 1)
        union_bbox = (r0, c0, r0 + room_h - 1, c0 + total_w - 1)
        if not _free(occupied, union_bbox, pad=1):
            continue

        left_bbox = (r0, c0, r0 + room_h - 1, c0 + left_w - 1)
        right_bbox = (
            r0,
            c0 + left_w + corridor,
            r0 + room_h - 1,
            c0 + left_w + corridor + right_w - 1,
        )
        if direction == "right":
            source_bbox, target_bbox = left_bbox, right_bbox
            source_side, target_side = "right", "left"
        else:
            source_bbox, target_bbox = right_bbox, left_bbox
            source_side, target_side = "left", "right"

        max_span = min(source_bbox[3] - source_bbox[1] - 1, target_bbox[3] - target_bbox[1] - 1)
        if max_span < 2:
            continue
        span = rng.randint(2, min(5, max_span))
        offsets = _payload_offsets(rng, span)
        colors = _color_sequence(rng, len(offsets), bg)

        start_col = rng.randint(source_bbox[1] + 1, source_bbox[3] - span)
        payload = [(gap_row, start_col + off, color) for off, color in zip(offsets, colors)]

        used_rows.add(gap_row)
        _reserve(occupied, union_bbox, pad=1)
        return {
            "source_bbox": source_bbox,
            "target_bbox": target_bbox,
            "source_side": source_side,
            "target_side": target_side,
            "source_gap": _room_gap(source_bbox, source_side, gap_off),
            "target_gap": _room_gap(target_bbox, target_side, gap_off),
            "payload": [(int(r), int(c), int(color)) for r, c, color in payload],
        }
    return None


def _vertical_pair(
    rng: random.Random,
    h: int,
    w: int,
    occupied: np.ndarray,
    used_cols: set[int],
    direction: str,
    bg: int,
) -> dict | None:
    top_h = rng.randint(5, 8)
    bottom_h = rng.randint(5, 8)
    room_w = rng.randint(4, 7)
    corridor = rng.randint(2, 6)
    total_h = top_h + corridor + bottom_h
    if total_h + 2 > h or room_w + 2 > w:
        return None

    for _ in range(80):
        c0 = rng.randint(1, w - room_w - 1)
        gap_off = rng.randint(1, room_w - 2)
        gap_col = c0 + gap_off
        if gap_col in used_cols:
            continue
        r0 = rng.randint(1, h - total_h - 1)
        union_bbox = (r0, c0, r0 + total_h - 1, c0 + room_w - 1)
        if not _free(occupied, union_bbox, pad=1):
            continue

        top_bbox = (r0, c0, r0 + top_h - 1, c0 + room_w - 1)
        bottom_bbox = (
            r0 + top_h + corridor,
            c0,
            r0 + top_h + corridor + bottom_h - 1,
            c0 + room_w - 1,
        )
        if direction == "down":
            source_bbox, target_bbox = top_bbox, bottom_bbox
            source_side, target_side = "bottom", "top"
        else:
            source_bbox, target_bbox = bottom_bbox, top_bbox
            source_side, target_side = "top", "bottom"

        max_span = min(source_bbox[2] - source_bbox[0] - 1, target_bbox[2] - target_bbox[0] - 1)
        if max_span < 2:
            continue
        span = rng.randint(2, min(5, max_span))
        offsets = _payload_offsets(rng, span)
        colors = _color_sequence(rng, len(offsets), bg)

        start_row = rng.randint(source_bbox[0] + 1, source_bbox[2] - span)
        payload = [(start_row + off, gap_col, color) for off, color in zip(offsets, colors)]

        used_cols.add(gap_col)
        _reserve(occupied, union_bbox, pad=1)
        return {
            "source_bbox": source_bbox,
            "target_bbox": target_bbox,
            "source_side": source_side,
            "target_side": target_side,
            "source_gap": _room_gap(source_bbox, source_side, gap_off),
            "target_gap": _room_gap(target_bbox, target_side, gap_off),
            "payload": [(int(r), int(c), int(color)) for r, c, color in payload],
        }
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(20, 30)
    w = rng.randint(20, 30)
    bg = rng.choice([0, 1, 2, 5])
    grid = np.full((h, w), int(bg), dtype=int)

    occupied = np.zeros((h, w), dtype=bool)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    n_pairs = rng.randint(2, 3)
    directions = rng.sample(["left", "right", "up", "down"], n_pairs)

    pairs: list[dict] = []
    for direction in directions:
        if direction in ("left", "right"):
            pair = _horizontal_pair(rng, h, w, occupied, used_rows, direction, bg)
        else:
            pair = _vertical_pair(rng, h, w, occupied, used_cols, direction, bg)
        if pair is None:
            return None
        pairs.append(pair)

    for pair in pairs:
        _draw_room(grid, pair["source_bbox"], pair["source_gap"], WALL, bg)
        _draw_room(grid, pair["target_bbox"], pair["target_gap"], WALL, bg)
        for r, c, color in pair["payload"]:
            grid[r, c] = int(color)

    vals, counts = np.unique(grid, return_counts=True)
    if int(vals[counts.argmax()]) != int(bg):
        return None

    expected = _expected_from_pairs(grid, pairs, bg)
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    rooms = _extract_rooms(grid, bg, WALL)
    if len(rooms) != 2 * len(pairs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
