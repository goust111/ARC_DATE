from __future__ import annotations

import random
from collections import deque

import numpy as np


ORANGE = 7
GREY = 5
RED = 2
SPECIAL_COLORS = {ORANGE, GREY, RED}
STAMP_COLORS = (1, 3, 4, 6, 8, 9)

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Each marker is one protruding head cell plus a two-cell tail.
ARROW_PATTERNS = {
    "up": frozenset({(1, 0), (1, 1), (0, 1)}),
    "right": frozenset({(0, 0), (1, 0), (1, 1)}),
    "down": frozenset({(0, 0), (0, 1), (1, 0)}),
    "left": frozenset({(0, 0), (0, 1), (1, 1)}),
}
ARROW_SHAPE = (2, 2)

GEN_ATTEMPTS = 400
ROOM_COUNT_CHOICES = (1, 2, 3)
GRID_MIN = 22
GRID_MAX = 30
GAP_RANGE = (1, 4)
DISTRACTOR_COUNT_CHOICES = (0, 1, 2)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    if connectivity == 4:
        neighbors = DIRS4
    else:
        neighbors = DIRS4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
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
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return frozenset((r - r0, c - c0) for r, c in coords)


def _overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _extract_patch(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _frame_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _room_infos(grid: np.ndarray) -> list[dict]:
    rooms: list[dict] = []
    for comp in _components(grid == GREY, connectivity=4):
        r0, c0, r1, c1 = _bbox(comp)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        assert h >= 4 and w >= 4, "room must have an interior"
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        frame = _frame_mask((h, w))
        assert np.array_equal(patch == GREY, frame), "grey component must be a rectangular frame"

        ir0, ic0, ir1, ic1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
        interior = grid[ir0 : ir1 + 1, ic0 : ic1 + 1]
        assert np.all(np.isin(interior, [ORANGE, RED])), "room interior may only contain background and arrows"

        arrow_comps = _components(interior == RED, connectivity=4)
        assert 1 <= len(arrow_comps) <= 3, "each room must contain one to three arrows"
        directions = []
        for arrow in arrow_comps:
            rel = _normalize(arrow)
            matches = [name for name, pattern in ARROW_PATTERNS.items() if rel == pattern]
            assert len(matches) == 1, "unexpected arrow shape"
            directions.append(matches[0])
        assert len(set(directions)) == 1, "all arrows in one room must point the same way"

        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "interior_bbox": (ir0, ic0, ir1, ic1),
                "direction": directions[0],
                "count": len(arrow_comps),
            }
        )
    assert rooms, "expected at least one room"
    return rooms


def _stamp_infos(grid: np.ndarray) -> list[dict]:
    mask = ~np.isin(grid, list(SPECIAL_COLORS))
    infos: list[dict] = []
    for comp in _components(mask, connectivity=4):
        box = _bbox(comp)
        patch = _extract_patch(grid, box)
        infos.append({"bbox": box, "patch": patch})
    assert infos, "expected at least one stamp"
    return infos


def _nearest_stamp(room: dict, stamps: list[dict]) -> dict:
    r0, c0, r1, c1 = room["bbox"]
    direction = room["direction"]
    candidates: list[tuple[int, dict]] = []
    for stamp in stamps:
        sr0, sc0, sr1, sc1 = stamp["bbox"]
        if direction == "right":
            if sc0 > c1 and _overlap(r0, r1, sr0, sr1):
                candidates.append((sc0 - c1 - 1, stamp))
        elif direction == "left":
            if sc1 < c0 and _overlap(r0, r1, sr0, sr1):
                candidates.append((c0 - sc1 - 1, stamp))
        elif direction == "down":
            if sr0 > r1 and _overlap(c0, c1, sc0, sc1):
                candidates.append((sr0 - r1 - 1, stamp))
        else:
            if sr1 < r0 and _overlap(c0, c1, sc0, sc1):
                candidates.append((r0 - sr1 - 1, stamp))
    assert candidates, "room must have at least one stamp in its pointing direction"
    best_distance = min(distance for distance, _stamp in candidates)
    best = [stamp for distance, stamp in candidates if distance == best_distance]
    assert len(best) == 1, "nearest stamp must be unique"
    return best[0]


def _paste_patch(out: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    window = out[top : top + h, left : left + w]
    mask = patch != ORANGE
    window[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rooms = _room_infos(g)
    stamps = _stamp_infos(g)

    out = g.copy()
    for room in rooms:
        ir0, ic0, ir1, ic1 = room["interior_bbox"]
        interior = out[ir0 : ir1 + 1, ic0 : ic1 + 1]
        interior[interior == RED] = ORANGE

        stamp = _nearest_stamp(room, stamps)
        patch = stamp["patch"]
        sh, sw = patch.shape
        direction = room["direction"]
        count = room["count"]

        interior_h = ir1 - ir0 + 1
        interior_w = ic1 - ic0 + 1
        if direction in ("left", "right"):
            total_h = sh
            total_w = sw * count
            assert total_h <= interior_h and total_w <= interior_w
            start_r = ir0 + (interior_h - total_h) // 2
            start_c = ic0 if direction == "right" else ic1 - total_w + 1
            for i in range(count):
                _paste_patch(out, patch, start_r, start_c + i * sw)
        else:
            total_h = sh * count
            total_w = sw
            assert total_h <= interior_h and total_w <= interior_w
            start_r = ir0 if direction == "down" else ir1 - total_h + 1
            start_c = ic0 + (interior_w - total_w) // 2
            for i in range(count):
                _paste_patch(out, patch, start_r + i * sh, start_c)

    return out


def _block_bbox(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _bbox_free(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return not np.any(blocked[r0 : r1 + 1, c0 : c1 + 1])


def _choose_stamp_shape(direction: str) -> tuple[int, int]:
    if direction in ("left", "right"):
        return random.choice(((3, 2), (3, 3)))
    return random.choice(((2, 3), (3, 3)))


def _random_room_spec(h: int, w: int, color: int) -> dict:
    direction = random.choice(tuple(ARROW_PATTERNS))
    count = random.randint(1, 3)
    stamp_h, stamp_w = _choose_stamp_shape(direction)
    if direction in ("left", "right"):
        interior_h = random.randint(max(stamp_h + 1, 5), max(stamp_h + 1, 5) + 3)
        interior_w = random.randint(count * stamp_w + 1, count * stamp_w + 4)
    else:
        interior_h = random.randint(count * stamp_h + 1, count * stamp_h + 4)
        interior_w = random.randint(max(stamp_w + 1, 5), max(stamp_w + 1, 5) + 3)

    room_h = interior_h + 2
    room_w = interior_w + 2
    gap = random.randint(*GAP_RANGE)

    return {
        "direction": direction,
        "count": count,
        "stamp_shape": (stamp_h, stamp_w),
        "stamp_color": color,
        "room_shape": (room_h, room_w),
        "gap": gap,
    }


def _place_room(blocked: np.ndarray, spec: dict) -> dict | None:
    h, w = blocked.shape
    room_h, room_w = spec["room_shape"]
    stamp_h, stamp_w = spec["stamp_shape"]
    direction = spec["direction"]
    gap = spec["gap"]

    for _ in range(120):
        if direction == "right":
            max_r0 = h - room_h - 1
            max_c0 = w - room_w - gap - stamp_w - 1
            if max_r0 < 1 or max_c0 < 1:
                return None
            r0 = random.randint(1, max_r0)
            c0 = random.randint(1, max_c0)
            sr0_min = max(0, r0 - stamp_h + 1)
            sr0_max = min(h - stamp_h, r0 + room_h - 1)
            sr0 = random.randint(sr0_min, sr0_max)
            sc0 = c0 + room_w + gap
        elif direction == "left":
            max_r0 = h - room_h - 1
            min_c0 = stamp_w + gap + 1
            max_c0 = w - room_w - 1
            if max_r0 < 1 or min_c0 > max_c0:
                return None
            r0 = random.randint(1, max_r0)
            c0 = random.randint(min_c0, max_c0)
            sr0_min = max(0, r0 - stamp_h + 1)
            sr0_max = min(h - stamp_h, r0 + room_h - 1)
            sr0 = random.randint(sr0_min, sr0_max)
            sc0 = c0 - gap - stamp_w
        elif direction == "down":
            max_r0 = h - room_h - gap - stamp_h - 1
            max_c0 = w - room_w - 1
            if max_r0 < 1 or max_c0 < 1:
                return None
            r0 = random.randint(1, max_r0)
            c0 = random.randint(1, max_c0)
            sr0 = r0 + room_h + gap
            sc0_min = max(0, c0 - stamp_w + 1)
            sc0_max = min(w - stamp_w, c0 + room_w - 1)
            sc0 = random.randint(sc0_min, sc0_max)
        else:
            min_r0 = stamp_h + gap + 1
            max_r0 = h - room_h - 1
            max_c0 = w - room_w - 1
            if min_r0 > max_r0 or max_c0 < 1:
                return None
            r0 = random.randint(min_r0, max_r0)
            c0 = random.randint(1, max_c0)
            sr0 = r0 - gap - stamp_h
            sc0_min = max(0, c0 - stamp_w + 1)
            sc0_max = min(w - stamp_w, c0 + room_w - 1)
            sc0 = random.randint(sc0_min, sc0_max)

        room_bbox = (r0, c0, r0 + room_h - 1, c0 + room_w - 1)
        stamp_bbox = (sr0, sc0, sr0 + stamp_h - 1, sc0 + stamp_w - 1)
        if not _bbox_free(blocked, room_bbox):
            continue
        if not _bbox_free(blocked, stamp_bbox):
            continue
        return {
            **spec,
            "room_bbox": room_bbox,
            "stamp_bbox": stamp_bbox,
        }
    return None


def _arrow_cells(anchor_r: int, anchor_c: int, direction: str) -> set[tuple[int, int]]:
    return {(anchor_r + dr, anchor_c + dc) for dr, dc in ARROW_PATTERNS[direction]}


def _place_arrows(grid: np.ndarray, room: dict) -> bool:
    ir0, ic0, ir1, ic1 = room["interior_bbox"]
    direction = room["direction"]
    count = room["count"]
    pattern_h, pattern_w = ARROW_SHAPE

    anchors = [
        (r, c)
        for r in range(ir0, ir1 - pattern_h + 2)
        for c in range(ic0, ic1 - pattern_w + 2)
    ]
    random.shuffle(anchors)

    chosen: list[set[tuple[int, int]]] = []
    occupied: set[tuple[int, int]] = set()
    halo: set[tuple[int, int]] = set()
    for r, c in anchors:
        cells = _arrow_cells(r, c, direction)
        if cells & occupied:
            continue
        if cells & halo:
            continue
        chosen.append(cells)
        occupied |= cells
        for cr, cc in cells:
            for dr, dc in DIRS4:
                halo.add((cr + dr, cc + dc))
        if len(chosen) == count:
            break
    if len(chosen) != count:
        return False
    for cells in chosen:
        for r, c in cells:
            grid[r, c] = RED
    return True


def _draw_room(grid: np.ndarray, room: dict) -> None:
    r0, c0, r1, c1 = room["room_bbox"]
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY


def _draw_stamp(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _add_distractor_stamps(grid: np.ndarray, blocked: np.ndarray, colors: list[int]) -> None:
    count = random.choice(DISTRACTOR_COUNT_CHOICES)
    placed = 0
    random.shuffle(colors)
    for color in colors:
        if placed == count:
            break
        stamp_h, stamp_w = random.choice(((2, 2), (2, 3), (3, 2), (3, 3)))
        for _ in range(80):
            r0 = random.randint(0, grid.shape[0] - stamp_h)
            c0 = random.randint(0, grid.shape[1] - stamp_w)
            bbox = (r0, c0, r0 + stamp_h - 1, c0 + stamp_w - 1)
            if not _bbox_free(blocked, bbox):
                continue
            _draw_stamp(grid, bbox, color)
            _block_bbox(blocked, bbox, pad=1)
            placed += 1
            break


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), ORANGE, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    room_count = random.choice(ROOM_COUNT_CHOICES)
    colors = list(STAMP_COLORS)
    random.shuffle(colors)
    if room_count > len(colors):
        return None

    rooms_raw: list[dict] = []
    for idx in range(room_count):
        spec = _random_room_spec(h, w, colors[idx])
        placed = _place_room(blocked, spec)
        if placed is None:
            return None
        rooms_raw.append(placed)
        _block_bbox(blocked, placed["room_bbox"], pad=1)
        _block_bbox(blocked, placed["stamp_bbox"], pad=1)

    for room in rooms_raw:
        _draw_room(grid, room)
        _draw_stamp(grid, room["stamp_bbox"], room["stamp_color"])

    rooms: list[dict] = []
    for room in rooms_raw:
        r0, c0, r1, c1 = room["room_bbox"]
        room["interior_bbox"] = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
        if not _place_arrows(grid, room):
            return None
        rooms.append(room)

    _add_distractor_stamps(grid, blocked, colors[room_count:])

    try:
        parsed_rooms = _room_infos(grid)
        assert len(parsed_rooms) == room_count
        stamps = _stamp_infos(grid)
        for room in parsed_rooms:
            _nearest_stamp(room, stamps)
        out = main(grid)
        assert out.shape == grid.shape
        assert np.any(out != grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
