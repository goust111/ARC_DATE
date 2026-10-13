import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5

MIN_ROOM_SIDE = 2
ROOM_COUNT_RANGE = (3, 7)
INNER_SIDE_RANGE = (5, 10)
GRID_SIDE_MAX = 30
MARGIN_EXTRA_RANGE = (1, 3)
PARTITION_RETRIES = 120
GEN_RETRIES = 300


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


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
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    points = np.argwhere(grid != bg)
    if len(points) == 0:
        raise ValueError("grid has no foreground")
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _wall_color(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> int:
    r0, c0, r1, c1 = bbox
    border_values: list[int] = []
    border_values.extend(int(grid[r0, c]) for c in range(c0, c1 + 1) if int(grid[r0, c]) != bg)
    border_values.extend(int(grid[r1, c]) for c in range(c0, c1 + 1) if int(grid[r1, c]) != bg)
    border_values.extend(int(grid[r, c0]) for r in range(r0 + 1, r1) if int(grid[r, c0]) != bg)
    border_values.extend(int(grid[r, c1]) for r in range(r0 + 1, r1) if int(grid[r, c1]) != bg)
    if not border_values:
        raise ValueError("failed to infer wall color")
    return int(Counter(border_values).most_common(1)[0][0])


def _extract_rooms(
    grid: np.ndarray,
) -> tuple[int, int, tuple[int, int, int, int], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    frame = _frame_bbox(g, bg)
    wall = _wall_color(g, frame, bg)
    r0, c0, r1, c1 = frame

    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    comps = _components(inner != wall)
    rooms: list[dict[str, object]] = []

    for comp in comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        rr0, rr1 = min(rows), max(rows)
        cc0, cc1 = min(cols), max(cols)
        color = int(inner[comp[0][0], comp[0][1]])
        if color in (bg, wall):
            raise ValueError("room color collides with structural colors")
        if any(int(inner[r, c]) != color for r, c in comp):
            raise ValueError("room must be monochrome")
        if len(comp) != (rr1 - rr0 + 1) * (cc1 - cc0 + 1):
            raise ValueError("room must be rectangular")

        touches: list[str] = []
        exit_sides: list[str] = []
        if rr0 == 0:
            touches.append("top")
            segment = g[r0, c0 + 1 + cc0 : c0 + 2 + cc1]
            if int(np.count_nonzero(segment == bg)) == 1:
                exit_sides.append("top")
        if rr1 == inner.shape[0] - 1:
            touches.append("bottom")
            segment = g[r1, c0 + 1 + cc0 : c0 + 2 + cc1]
            if int(np.count_nonzero(segment == bg)) == 1:
                exit_sides.append("bottom")
        if cc0 == 0:
            touches.append("left")
            segment = g[r0 + 1 + rr0 : r0 + 2 + rr1, c0]
            if int(np.count_nonzero(segment == bg)) == 1:
                exit_sides.append("left")
        if cc1 == inner.shape[1] - 1:
            touches.append("right")
            segment = g[r0 + 1 + rr0 : r0 + 2 + rr1, c1]
            if int(np.count_nonzero(segment == bg)) == 1:
                exit_sides.append("right")

        if len(exit_sides) != 1:
            raise ValueError("each room must have exactly one exit side")

        rooms.append(
            {
                "bbox": (rr0, cc0, rr1, cc1),
                "color": color,
                "touches": tuple(touches),
                "exit": exit_sides[0],
            }
        )

    if not rooms:
        raise ValueError("no rooms found")
    return bg, wall, frame, rooms


def _paint_rect(
    grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int
) -> None:
    h, w = grid.shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(h - 1, int(r1))
    cc1 = min(w - 1, int(c1))
    if rr0 <= rr1 and cc0 <= cc1:
        grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition the grey frame into rooms, detect each room's border gap, and
    project that room outward as a rectangular strip.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _bg, _wall, frame, rooms = _extract_rooms(g)
    fr0, fc0, fr1, fc1 = frame

    for room in rooms:
        rr0, cc0, rr1, cc1 = room["bbox"]
        color = int(room["color"])
        side = str(room["exit"])
        room_h = int(rr1 - rr0 + 1)
        room_w = int(cc1 - cc0 + 1)
        abs_r0 = fr0 + 1 + rr0
        abs_r1 = fr0 + 1 + rr1
        abs_c0 = fc0 + 1 + cc0
        abs_c1 = fc0 + 1 + cc1

        if side == "top":
            _paint_rect(out, fr0 - room_h, abs_c0, fr0 - 1, abs_c1, color)
        elif side == "bottom":
            _paint_rect(out, fr1 + 1, abs_c0, fr1 + room_h, abs_c1, color)
        elif side == "left":
            _paint_rect(out, abs_r0, fc0 - room_w, abs_r1, fc0 - 1, color)
        elif side == "right":
            _paint_rect(out, abs_r0, fc1 + 1, abs_r1, fc1 + room_w, color)
        else:
            raise ValueError("unknown exit side")

    return out


def _choose_inner_size(rng: random.Random, target_rooms: int) -> tuple[int, int]:
    lo = INNER_SIDE_RANGE[0] if target_rooms <= 4 else INNER_SIDE_RANGE[0] + 1
    hi = INNER_SIDE_RANGE[1]
    if rng.random() < 0.33:
        return rng.randint(lo, hi - 1), rng.randint(lo + 1, hi)
    if rng.random() < 0.5:
        return rng.randint(lo + 1, hi), rng.randint(lo, hi - 1)
    return rng.randint(lo, hi), rng.randint(lo, hi)


def _copy_room(room: dict[str, object]) -> dict[str, object]:
    return {
        "r0": int(room["r0"]),
        "c0": int(room["c0"]),
        "r1": int(room["r1"]),
        "c1": int(room["c1"]),
        "touches": set(room["touches"]),
    }


def _partition_layout(
    rng: random.Random, target_rooms: int
) -> tuple[int, int, list[dict[str, object]], list[tuple[int, int, int, int]]] | None:
    for _ in range(PARTITION_RETRIES):
        inner_h, inner_w = _choose_inner_size(rng, target_rooms)
        rooms: list[dict[str, object]] = [
            {
                "r0": 0,
                "c0": 0,
                "r1": inner_h - 1,
                "c1": inner_w - 1,
                "touches": {"top", "bottom", "left", "right"},
            }
        ]
        walls: list[tuple[int, int, int, int]] = []

        while len(rooms) < target_rooms:
            moves: list[tuple[int, str]] = []
            for idx, room in enumerate(rooms):
                h = int(room["r1"]) - int(room["r0"]) + 1
                w = int(room["c1"]) - int(room["c0"]) + 1
                touches = set(room["touches"])
                if w >= 2 * MIN_ROOM_SIDE + 1 and ("top" in touches or "bottom" in touches):
                    moves.append((idx, "vertical"))
                if h >= 2 * MIN_ROOM_SIDE + 1 and ("left" in touches or "right" in touches):
                    moves.append((idx, "horizontal"))
            if not moves:
                break

            idx, orient = rng.choice(moves)
            room = rooms.pop(idx)
            r0 = int(room["r0"])
            c0 = int(room["c0"])
            r1 = int(room["r1"])
            c1 = int(room["c1"])
            touches = set(room["touches"])

            if orient == "vertical":
                cut = rng.randint(c0 + MIN_ROOM_SIDE, c1 - MIN_ROOM_SIDE)
                walls.append((r0, cut, r1, cut))
                common = touches & {"top", "bottom"}
                left_touches = set(common)
                right_touches = set(common)
                if "left" in touches:
                    left_touches.add("left")
                if "right" in touches:
                    right_touches.add("right")
                rooms.extend(
                    [
                        {"r0": r0, "c0": c0, "r1": r1, "c1": cut - 1, "touches": left_touches},
                        {"r0": r0, "c0": cut + 1, "r1": r1, "c1": c1, "touches": right_touches},
                    ]
                )
            else:
                cut = rng.randint(r0 + MIN_ROOM_SIDE, r1 - MIN_ROOM_SIDE)
                walls.append((cut, c0, cut, c1))
                common = touches & {"left", "right"}
                top_touches = set(common)
                bottom_touches = set(common)
                if "top" in touches:
                    top_touches.add("top")
                if "bottom" in touches:
                    bottom_touches.add("bottom")
                rooms.extend(
                    [
                        {"r0": r0, "c0": c0, "r1": cut - 1, "c1": c1, "touches": top_touches},
                        {"r0": cut + 1, "c0": c0, "r1": r1, "c1": c1, "touches": bottom_touches},
                    ]
                )

        if len(rooms) == target_rooms:
            return inner_h, inner_w, [_copy_room(room) for room in rooms], walls
    return None


def _match_sides(
    rng: random.Random, rooms: list[dict[str, object]]
) -> dict[str, int]:
    side_order = sorted(
        ("top", "bottom", "left", "right"),
        key=lambda side: sum(side in room["touches"] for room in rooms),
    )
    answer: dict[str, int] = {}
    used: set[int] = set()

    def dfs(pos: int) -> bool:
        if pos == len(side_order):
            return True
        side = side_order[pos]
        candidates = [i for i, room in enumerate(rooms) if i not in used and side in room["touches"]]
        rng.shuffle(candidates)
        for idx in candidates:
            used.add(idx)
            answer[side] = idx
            if dfs(pos + 1):
                return True
            used.remove(idx)
            del answer[side]
        return False

    dfs(0)
    return answer


def _assign_exits(
    rng: random.Random, rooms: list[dict[str, object]]
) -> list[str]:
    exits = ["" for _ in rooms]
    if len(rooms) >= 4:
        matched = _match_sides(rng, rooms)
        for side, idx in matched.items():
            exits[idx] = side
    for idx, room in enumerate(rooms):
        if not exits[idx]:
            exits[idx] = rng.choice(sorted(room["touches"]))
    return exits


def _needed_margins(
    rooms: list[dict[str, object]], exits: list[str]
) -> dict[str, int]:
    needs = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    for room, side in zip(rooms, exits):
        room_h = int(room["r1"]) - int(room["r0"]) + 1
        room_w = int(room["c1"]) - int(room["c0"]) + 1
        if side in ("top", "bottom"):
            needs[side] = max(needs[side], room_h)
        else:
            needs[side] = max(needs[side], room_w)
    return needs


def _render_input(
    rng: random.Random,
    inner_h: int,
    inner_w: int,
    rooms: list[dict[str, object]],
    walls: list[tuple[int, int, int, int]],
    exits: list[str],
) -> np.ndarray | None:
    needs = _needed_margins(rooms, exits)
    margins = {
        side: needs[side] + rng.randint(MARGIN_EXTRA_RANGE[0], MARGIN_EXTRA_RANGE[1])
        for side in ("top", "bottom", "left", "right")
    }
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    grid_h = margins["top"] + frame_h + margins["bottom"]
    grid_w = margins["left"] + frame_w + margins["right"]
    if grid_h > GRID_SIDE_MAX or grid_w > GRID_SIDE_MAX:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    fr0 = margins["top"]
    fc0 = margins["left"]
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1

    grid[fr0, fc0 : fc1 + 1] = GREY
    grid[fr1, fc0 : fc1 + 1] = GREY
    grid[fr0 : fr1 + 1, fc0] = GREY
    grid[fr0 : fr1 + 1, fc1] = GREY

    for wr0, wc0, wr1, wc1 in walls:
        grid[fr0 + 1 + wr0 : fr0 + 2 + wr1, fc0 + 1 + wc0 : fc0 + 2 + wc1] = GREY

    palette = [color for color in range(1, 10) if color != GREY]
    room_colors = rng.sample(palette, len(rooms))
    for room, color in zip(rooms, room_colors):
        grid[
            fr0 + 1 + int(room["r0"]) : fr0 + 2 + int(room["r1"]),
            fc0 + 1 + int(room["c0"]) : fc0 + 2 + int(room["c1"]),
        ] = int(color)
        room["color"] = int(color)

    for room, side in zip(rooms, exits):
        room["exit"] = side
        if side == "top":
            gap_c = rng.randint(int(room["c0"]), int(room["c1"]))
            grid[fr0, fc0 + 1 + gap_c] = BLACK
        elif side == "bottom":
            gap_c = rng.randint(int(room["c0"]), int(room["c1"]))
            grid[fr1, fc0 + 1 + gap_c] = BLACK
        elif side == "left":
            gap_r = rng.randint(int(room["r0"]), int(room["r1"]))
            grid[fr0 + 1 + gap_r, fc0] = BLACK
        else:
            gap_r = rng.randint(int(room["r0"]), int(room["r1"]))
            grid[fr0 + 1 + gap_r, fc1] = BLACK

    try:
        parsed_bg, parsed_wall, _frame, parsed_rooms = _extract_rooms(grid)
    except ValueError:
        return None
    if parsed_bg != BLACK or parsed_wall != GREY or len(parsed_rooms) != len(rooms):
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    target_rooms = rng.randint(ROOM_COUNT_RANGE[0], ROOM_COUNT_RANGE[1])
    layout = _partition_layout(rng, target_rooms)
    if layout is None:
        return None
    inner_h, inner_w, rooms, walls = layout
    exits = _assign_exits(rng, rooms)
    return _render_input(rng, inner_h, inner_w, rooms, walls, exits)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
