import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9

SIDE_ORDER = ("up", "right", "down", "left")
SIDE_TO_INDEX = {side: idx for idx, side in enumerate(SIDE_ORDER)}
MISSING_TO_TARGET = {
    (0, 1): "up",
    (0, 0): "left",
    (1, 0): "down",
    (1, 1): "right",
}
TARGET_TO_MISSING = {side: missing for missing, side in MISSING_TO_TARGET.items()}


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_value(grid: np.ndarray, value: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != value:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == value:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _is_rect_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    return set(coords) == expected


def _find_rooms(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rooms = []
    for comp in _components_of_value(grid, GREY):
        if _is_rect_frame(comp):
            rooms.append(_bbox(comp))
    rooms.sort()
    return rooms


def _find_arrow(grid: np.ndarray, room: tuple[int, int, int, int]) -> tuple[int, str]:
    r0, c0, r1, c1 = room
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (inner != BLACK) & (inner != GREY)
    coords = np.argwhere(mask)
    assert len(coords) == 3, "room interior must contain exactly one 3-cell arrow"
    colors = np.unique(inner[mask])
    assert len(colors) == 1, "arrow must be monochrome"
    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    assert max_r - min_r == 1 and max_c - min_c == 1, "arrow must occupy a 2x2 box"

    local = {(int(r - min_r), int(c - min_c)) for r, c in coords}
    missing = next(cell for cell in ((0, 0), (0, 1), (1, 0), (1, 1)) if cell not in local)
    return int(colors[0]), MISSING_TO_TARGET[missing]


def _side_strip_cells(
    grid: np.ndarray,
    room: tuple[int, int, int, int],
    side: str,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = room
    cells: list[tuple[int, int]] = []

    if side == "up":
        for r in range(max(0, r0 - 3), r0):
            for c in range(c0, c1 + 1):
                if int(grid[r, c]) == MAROON:
                    cells.append((r, c))
        if cells and not any(r == r0 - 1 for r, _ in cells):
            return []
        return cells

    if side == "down":
        for r in range(r1 + 1, min(h, r1 + 4)):
            for c in range(c0, c1 + 1):
                if int(grid[r, c]) == MAROON:
                    cells.append((r, c))
        if cells and not any(r == r1 + 1 for r, _ in cells):
            return []
        return cells

    if side == "left":
        for r in range(r0, r1 + 1):
            for c in range(max(0, c0 - 3), c0):
                if int(grid[r, c]) == MAROON:
                    cells.append((r, c))
        if cells and not any(c == c0 - 1 for _, c in cells):
            return []
        return cells

    if side == "right":
        for r in range(r0, r1 + 1):
            for c in range(c1 + 1, min(w, c1 + 4)):
                if int(grid[r, c]) == MAROON:
                    cells.append((r, c))
        if cells and not any(c == c1 + 1 for _, c in cells):
            return []
        return cells

    raise ValueError(side)


def _find_outer_pattern(
    grid: np.ndarray,
    room: tuple[int, int, int, int],
) -> tuple[str, tuple[int, int, int, int], np.ndarray]:
    found: list[tuple[str, list[tuple[int, int]]]] = []
    for side in SIDE_ORDER:
        cells = _side_strip_cells(grid, room, side)
        if cells:
            found.append((side, cells))
    assert len(found) == 1, "each room must have exactly one adjacent maroon pattern"
    side, cells = found[0]
    pr0, pc0, pr1, pc1 = _bbox(cells)
    mask = grid[pr0 : pr1 + 1, pc0 : pc1 + 1] == MAROON
    return side, (pr0, pc0, pr1, pc1), mask


def _rotation_steps(source_side: str, target_side: str) -> int:
    return (SIDE_TO_INDEX[target_side] - SIDE_TO_INDEX[source_side]) % 4


def _inner_stamp_top_left(
    inner_h: int,
    inner_w: int,
    side: str,
    stamp_h: int,
    stamp_w: int,
) -> tuple[int, int]:
    assert stamp_h <= inner_h and stamp_w <= inner_w
    if side == "up":
        return 0, (inner_w - stamp_w) // 2
    if side == "down":
        return inner_h - stamp_h, (inner_w - stamp_w) // 2
    if side == "left":
        return (inner_h - stamp_h) // 2, 0
    if side == "right":
        return (inner_h - stamp_h) // 2, inner_w - stamp_w
    raise ValueError(side)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each outside maroon stamp from its wall to the wall indicated by the room's L-arrow.
    Recolor the rotated stamp with the arrow color, erase the original outside stamp, and keep
    room frames plus arrows unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for room in _find_rooms(g):
        arrow_color, target_side = _find_arrow(g, room)
        source_side, stamp_bbox, stamp_mask = _find_outer_pattern(g, room)
        pr0, pc0, pr1, pc1 = stamp_bbox

        out[pr0 : pr1 + 1, pc0 : pc1 + 1][stamp_mask] = BLACK

        rotated = np.rot90(stamp_mask, -_rotation_steps(source_side, target_side))
        r0, c0, r1, c1 = room
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        off_r, off_c = _inner_stamp_top_left(inner_h, inner_w, target_side, *rotated.shape)
        dst_r0 = r0 + 1 + off_r
        dst_c0 = c0 + 1 + off_c
        target = out[dst_r0 : dst_r0 + rotated.shape[0], dst_c0 : dst_c0 + rotated.shape[1]]
        target[rotated] = arrow_color

    return out


# --------------- generator ---------------

GEN_RETRIES = 300
ROOM_TRIES = 80
ROOM_COUNT_CHOICES = (2, 3, 4)
SIZE_RANGE = (24, 36)
INNER_SIZE_RANGE = (6, 10)
ARROW_COLORS = [1, 2, 3, 4, 6, 7, 8]
ROOM_MARGIN = 4


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _touches_attachment_edge(mask: np.ndarray, side: str) -> bool:
    if side == "up":
        return bool(mask[-1, :].any())
    if side == "down":
        return bool(mask[0, :].any())
    if side == "left":
        return bool(mask[:, -1].any())
    if side == "right":
        return bool(mask[:, 0].any())
    raise ValueError(side)


def _random_pattern_mask(rng: random.Random, side: str) -> np.ndarray:
    for _ in range(60):
        h = rng.randint(1, 3)
        w = rng.randint(1, 3)
        total = h * w
        cell_count = 1 if total == 1 else rng.randint(2, min(5, total))

        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        while len(cells) < cell_count:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if _touches_attachment_edge(mask, side):
            return mask
    raise RuntimeError("failed to generate attached pattern")


def _arrow_cells_for_target(top: int, left: int, target_side: str) -> set[tuple[int, int]]:
    missing = TARGET_TO_MISSING[target_side]
    cells = set()
    for dr in range(2):
        for dc in range(2):
            if (dr, dc) != missing:
                cells.add((top + dr, left + dc))
    return cells


def _choose_arrow_position(
    rng: random.Random,
    inner_h: int,
    inner_w: int,
    target_side: str,
    stamp_mask: np.ndarray,
) -> tuple[int, int] | None:
    stamp_top, stamp_left = _inner_stamp_top_left(inner_h, inner_w, target_side, *stamp_mask.shape)
    stamp_cells = {
        (stamp_top + r, stamp_left + c)
        for r in range(stamp_mask.shape[0])
        for c in range(stamp_mask.shape[1])
        if stamp_mask[r, c]
    }
    candidates = []
    for r in range(inner_h - 1):
        for c in range(inner_w - 1):
            arrow_cells = _arrow_cells_for_target(r, c, target_side)
            if arrow_cells.isdisjoint(stamp_cells):
                candidates.append((r, c))
    if not candidates:
        return None
    return rng.choice(candidates)


def _outside_stamp_top_left(
    room: tuple[int, int, int, int],
    side: str,
    stamp_h: int,
    stamp_w: int,
) -> tuple[int, int]:
    r0, c0, r1, c1 = room
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    if side == "up":
        return r0 - stamp_h, c0 + 1 + (inner_w - stamp_w) // 2
    if side == "down":
        return r1 + 1, c0 + 1 + (inner_w - stamp_w) // 2
    if side == "left":
        return r0 + 1 + (inner_h - stamp_h) // 2, c0 - stamp_w
    if side == "right":
        return r0 + 1 + (inner_h - stamp_h) // 2, c1 + 1
    raise ValueError(side)


def _reserve_clear(used: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    return not used[r0 : r1 + 1, c0 : c1 + 1].any()


def _draw_frame(grid: np.ndarray, room: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = room
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _draw_arrow(
    grid: np.ndarray,
    room: tuple[int, int, int, int],
    target_side: str,
    arrow_top: int,
    arrow_left: int,
    color: int,
) -> None:
    r0, c0, _, _ = room
    for r, c in _arrow_cells_for_target(arrow_top, arrow_left, target_side):
        grid[r0 + 1 + r, c0 + 1 + c] = color


def _draw_pattern(
    grid: np.ndarray,
    top: int,
    left: int,
    mask: np.ndarray,
    color: int,
) -> None:
    sub = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    sub[mask] = color


def _gen_room_spec(
    rng: random.Random,
    grid_shape: tuple[int, int],
    used: np.ndarray,
    arrow_color: int,
) -> dict | None:
    h, w = grid_shape
    for _ in range(ROOM_TRIES):
        inner_h = rng.randint(*INNER_SIZE_RANGE)
        inner_w = rng.randint(*INNER_SIZE_RANGE)
        source_side = rng.choice(SIDE_ORDER)
        target_side = rng.choice(SIDE_ORDER)
        pattern_mask = _random_pattern_mask(rng, source_side)
        rotated_mask = np.rot90(pattern_mask, -_rotation_steps(source_side, target_side))
        if rotated_mask.shape[0] > inner_h or rotated_mask.shape[1] > inner_w:
            continue

        arrow_pos = _choose_arrow_position(rng, inner_h, inner_w, target_side, rotated_mask)
        if arrow_pos is None:
            continue

        outer_h = inner_h + 2
        outer_w = inner_w + 2
        min_r = ROOM_MARGIN
        max_r = h - ROOM_MARGIN - outer_h
        min_c = ROOM_MARGIN
        max_c = w - ROOM_MARGIN - outer_w
        if min_r > max_r or min_c > max_c:
            return None

        r0 = rng.randint(min_r, max_r)
        c0 = rng.randint(min_c, max_c)
        room = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        reserve_r0 = r0 - ROOM_MARGIN
        reserve_c0 = c0 - ROOM_MARGIN
        reserve_r1 = room[2] + ROOM_MARGIN
        reserve_c1 = room[3] + ROOM_MARGIN
        if not _reserve_clear(used, reserve_r0, reserve_c0, reserve_r1, reserve_c1):
            continue

        pattern_top, pattern_left = _outside_stamp_top_left(room, source_side, *pattern_mask.shape)
        if pattern_top < 0 or pattern_left < 0:
            continue
        if pattern_top + pattern_mask.shape[0] > h or pattern_left + pattern_mask.shape[1] > w:
            continue

        return {
            "room": room,
            "reserve": (reserve_r0, reserve_c0, reserve_r1, reserve_c1),
            "source_side": source_side,
            "target_side": target_side,
            "pattern_mask": pattern_mask,
            "pattern_top": pattern_top,
            "pattern_left": pattern_left,
            "arrow_top": arrow_pos[0],
            "arrow_left": arrow_pos[1],
            "arrow_color": arrow_color,
        }
    return None


def _draw_spec(grid: np.ndarray, spec: dict) -> None:
    room = spec["room"]
    _draw_frame(grid, room)
    _draw_arrow(
        grid,
        room,
        spec["target_side"],
        spec["arrow_top"],
        spec["arrow_left"],
        spec["arrow_color"],
    )
    _draw_pattern(
        grid,
        spec["pattern_top"],
        spec["pattern_left"],
        spec["pattern_mask"],
        MAROON,
    )


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*SIZE_RANGE)
    w = rng.randint(*SIZE_RANGE)
    n_rooms = rng.choice(ROOM_COUNT_CHOICES)
    if n_rooms > len(ARROW_COLORS):
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    used = np.zeros((h, w), dtype=bool)
    arrow_colors = rng.sample(ARROW_COLORS, n_rooms)
    specs = []
    for color in arrow_colors:
        spec = _gen_room_spec(rng, (h, w), used, color)
        if spec is None:
            return None
        rr0, rc0, rr1, rc1 = spec["reserve"]
        used[rr0 : rr1 + 1, rc0 : rc1 + 1] = True
        specs.append(spec)

    if all(spec["source_side"] == spec["target_side"] for spec in specs):
        return None
    if len({spec["target_side"] for spec in specs}) == 1:
        return None

    for spec in specs:
        _draw_spec(grid, spec)

    try:
        rooms = _find_rooms(grid)
        if len(rooms) != n_rooms:
            return None
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out != grid):
        return None
    if np.any(out == MAROON):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
