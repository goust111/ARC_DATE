from __future__ import annotations

from collections import deque

import numpy as np
import random


BG = 0
LAMP = 4
WALL = 7
WINDOW_PALETTE = (1, 2, 3, 5, 6, 8, 9)
DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_ORDER = ("top", "bottom", "left", "right")


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _flood_room_interior(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = grid.shape
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS.values():
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and (nr, nc) not in seen
                and int(grid[nr, nc]) in (BG, LAMP)
            ):
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _wall_cells(outer_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = outer_bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _window_candidates(
    window: tuple[int, int],
    outer_bbox: tuple[int, int, int, int],
) -> list[str]:
    r, c = window
    r0, c0, r1, c1 = outer_bbox
    candidates: list[str] = []
    if r == r0:
        candidates.append("top")
    if r == r1:
        candidates.append("bottom")
    if c == c0:
        candidates.append("left")
    if c == c1:
        candidates.append("right")
    return candidates


def _projection_score(window: tuple[int, int], lamp: tuple[int, int], direction: str) -> int:
    wr, wc = window
    lr, lc = lamp
    if direction == "top":
        return lr - wr
    if direction == "bottom":
        return wr - lr
    if direction == "left":
        return lc - wc
    return wc - lc


def _window_direction(
    window: tuple[int, int],
    lamp: tuple[int, int],
    outer_bbox: tuple[int, int, int, int],
) -> str:
    candidates = _window_candidates(window, outer_bbox)
    assert candidates, "window must lie on a room wall"
    return max(
        candidates,
        key=lambda name: (_projection_score(window, lamp, name), -DIR_ORDER.index(name)),
    )


def _collect_rooms(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    rooms = []
    seen_lamps: set[tuple[int, int]] = set()
    for lamp_rc in map(tuple, np.argwhere(g == LAMP)):
        lamp = (int(lamp_rc[0]), int(lamp_rc[1]))
        if lamp in seen_lamps:
            continue
        interior = _flood_room_interior(g, lamp)
        lamp_count = sum(1 for cell in interior if int(g[cell]) == LAMP)
        assert lamp_count == 1, "each room interior must contain exactly one lamp"
        seen_lamps |= {cell for cell in interior if int(g[cell]) == LAMP}

        ir0, ic0, ir1, ic1 = _bbox(interior)
        outer = (ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1)
        or0, oc0, or1, oc1 = outer
        assert 0 <= or0 <= or1 < g.shape[0]
        assert 0 <= oc0 <= oc1 < g.shape[1]

        windows: list[tuple[int, int]] = []
        window_colors: set[int] = set()
        for r, c in _wall_cells(outer):
            value = int(g[r, c])
            assert value != BG, "room boundary must be sealed"
            if value != WALL:
                assert value != LAMP, "lamp must stay inside the room"
                windows.append((r, c))
                window_colors.add(value)
        assert windows, "each room must have at least one window"
        assert len(window_colors) == 1, "all windows in a room must share one color"

        rooms.append(
            {
                "lamp": lamp,
                "outer_bbox": outer,
                "windows": windows,
                "window_color": int(next(iter(window_colors))),
            }
        )

    rooms.sort(key=lambda room: (room["outer_bbox"][0], room["outer_bbox"][1]))
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """Project colored rays out of each room's windows.

    Input structure
    - Background is black.
    - Orange rectangular room walls enclose black interiors.
    - Each room wall contains one or more same-colored windows.
    - Each room interior contains exactly one yellow lamp.

    Rule
    - Let W be the number of windows of a room.
    - Each window emits a ray of length W in the outward direction of its wall.
    - Corner windows choose the wall direction whose outward normal is closer to the lamp-to-window segment.
    - Rays paint only background cells and stop at obstacles or the grid boundary.
    - Rooms themselves remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for room in _collect_rooms(g):
        lamp = room["lamp"]
        outer_bbox = room["outer_bbox"]
        windows = room["windows"]
        ray_len = len(windows)
        color = room["window_color"]

        for window in windows:
            direction = _window_direction(window, lamp, outer_bbox)
            dr, dc = DIRS[direction]
            r, c = window
            for step in range(1, ray_len + 1):
                nr, nc = r + dr * step, c + dc * step
                if not _in_bounds(out.shape, nr, nc) or int(out[nr, nc]) != BG:
                    break
                out[nr, nc] = color

    return out


# --------------- generator ---------------

GEN_RETRIES = 400
GRID_MIN = 18
GRID_MAX = 30
ROOM_GAP = 1
PLACE_RETRIES = 250
SPEC_RETRIES = 120


def _local_border_cells(room_h: int, room_w: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    corners = [(0, 0), (0, room_w - 1), (room_h - 1, 0), (room_h - 1, room_w - 1)]
    edges = []
    for c in range(1, room_w - 1):
        edges.append((0, c))
        edges.append((room_h - 1, c))
    for r in range(1, room_h - 1):
        edges.append((r, 0))
        edges.append((r, room_w - 1))
    return corners, edges


def _is_corner(cell: tuple[int, int], room_h: int, room_w: int) -> bool:
    r, c = cell
    return r in (0, room_h - 1) and c in (0, room_w - 1)


def _make_room_spec(
    room_count: int,
    window_color: int,
    force_corner: bool,
    force_truncation: bool,
) -> dict | None:
    max_inner = 4 if room_count >= 3 else 5
    max_windows = 3 if room_count >= 3 else 4

    for _ in range(SPEC_RETRIES):
        inner_h = random.randint(2, max_inner)
        inner_w = random.randint(2, max_inner + 1)
        room_h = inner_h + 2
        room_w = inner_w + 2
        window_count = random.randint(1, max_windows)
        corners, edges = _local_border_cells(room_h, room_w)

        use_corner = force_corner or random.random() < 0.5
        windows: list[tuple[int, int]] = []
        if use_corner:
            windows.append(random.choice(corners))
        remaining = window_count - len(windows)
        if remaining > len(edges):
            continue
        windows.extend(random.sample(edges, remaining))
        random.shuffle(windows)

        lamps = [
            (r, c)
            for r in range(1, room_h - 1)
            for c in range(1, room_w - 1)
            if all(
                not (
                    _is_corner(window, room_h, room_w)
                    and abs(r - window[0]) == abs(c - window[1])
                )
                for window in windows
            )
        ]
        if not lamps:
            continue
        lamp = random.choice(lamps)
        outer_bbox = (0, 0, room_h - 1, room_w - 1)
        directions = [_window_direction(window, lamp, outer_bbox) for window in windows]
        used_dirs = sorted(set(directions), key=DIR_ORDER.index)
        truncate_dir = random.choice(used_dirs) if force_truncation else None

        return {
            "inner_h": inner_h,
            "inner_w": inner_w,
            "room_h": room_h,
            "room_w": room_w,
            "lamp": lamp,
            "windows": windows,
            "directions": directions,
            "window_count": window_count,
            "window_color": window_color,
            "used_dirs": used_dirs,
            "truncate_dir": truncate_dir,
            "has_corner_window": any(_is_corner(window, room_h, room_w) for window in windows),
        }
    return None


def _room_distances(
    shape: tuple[int, int],
    top: int,
    left: int,
    room_h: int,
    room_w: int,
) -> dict[str, int]:
    h, w = shape
    return {
        "top": top,
        "bottom": h - (top + room_h),
        "left": left,
        "right": w - (left + room_w),
    }


def _placement_satisfies(spec: dict, shape: tuple[int, int], top: int, left: int) -> bool:
    distances = _room_distances(shape, top, left, spec["room_h"], spec["room_w"])
    for direction in spec["used_dirs"]:
        dist = distances[direction]
        if direction == spec["truncate_dir"]:
            if not (1 <= dist < spec["window_count"]):
                return False
        elif dist < spec["window_count"]:
            return False
    return True


def _ray_cells_for_placement(
    spec: dict,
    shape: tuple[int, int],
    top: int,
    left: int,
    reserved_output: np.ndarray,
) -> list[tuple[int, int]] | None:
    cells: list[tuple[int, int]] = []
    seen = set()
    for (wr, wc), direction in zip(spec["windows"], spec["directions"]):
        dr, dc = DIRS[direction]
        ar, ac = top + wr, left + wc
        for step in range(1, spec["window_count"] + 1):
            nr, nc = ar + dr * step, ac + dc * step
            if not _in_bounds(shape, nr, nc):
                break
            if bool(reserved_output[nr, nc]) or (nr, nc) in seen:
                return None
            seen.add((nr, nc))
            cells.append((nr, nc))
    return cells


def _paint_room(grid: np.ndarray, top: int, left: int, spec: dict) -> None:
    room_h = spec["room_h"]
    room_w = spec["room_w"]
    grid[top : top + room_h, left : left + room_w] = WALL
    grid[top + 1 : top + room_h - 1, left + 1 : left + room_w - 1] = BG
    lamp_r, lamp_c = spec["lamp"]
    grid[top + lamp_r, left + lamp_c] = LAMP
    for wr, wc in spec["windows"]:
        grid[top + wr, left + wc] = spec["window_color"]


def _candidate_positions(spec: dict, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    room_h = spec["room_h"]
    room_w = spec["room_w"]
    positions = [
        (top, left)
        for top in range(h - room_h + 1)
        for left in range(w - room_w + 1)
        if _placement_satisfies(spec, shape, top, left)
    ]
    random.shuffle(positions)
    return positions


def _layout_rooms(shape: tuple[int, int], specs: list[dict]) -> np.ndarray | None:
    grid = np.full(shape, BG, dtype=int)
    blocked_rooms = np.zeros(shape, dtype=bool)
    reserved_output = np.zeros(shape, dtype=bool)

    ordered_specs = sorted(
        specs,
        key=lambda spec: (spec["truncate_dir"] is not None, -(spec["room_h"] * spec["room_w"])),
    )
    for spec in ordered_specs:
        placed = False
        attempts = 0
        for top, left in _candidate_positions(spec, shape):
            attempts += 1
            if attempts > PLACE_RETRIES:
                break

            room_h = spec["room_h"]
            room_w = spec["room_w"]
            pad_r0 = max(0, top - ROOM_GAP)
            pad_c0 = max(0, left - ROOM_GAP)
            pad_r1 = min(shape[0], top + room_h + ROOM_GAP)
            pad_c1 = min(shape[1], left + room_w + ROOM_GAP)
            if bool(np.any(blocked_rooms[pad_r0:pad_r1, pad_c0:pad_c1])):
                continue
            if bool(np.any(reserved_output[top : top + room_h, left : left + room_w])):
                continue

            ray_cells = _ray_cells_for_placement(spec, shape, top, left, reserved_output)
            if ray_cells is None:
                continue

            _paint_room(grid, top, left, spec)
            blocked_rooms[pad_r0:pad_r1, pad_c0:pad_c1] = True
            reserved_output[top : top + room_h, left : left + room_w] = True
            for rr, cc in ray_cells:
                reserved_output[rr, cc] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _room_count_for_shape(shape: tuple[int, int]) -> int:
    h, w = shape
    if min(h, w) >= 26:
        return random.randint(2, 4)
    if min(h, w) >= 22:
        return random.randint(1, 3)
    return random.randint(1, 2)


def _make_attempt() -> np.ndarray | None:
    shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
    room_count = _room_count_for_shape(shape)
    require_corner = random.random() < 0.5
    require_truncation = random.random() < 0.5

    colors = random.sample(WINDOW_PALETTE, room_count)
    force_corner_idx = random.randrange(room_count) if require_corner else None
    force_trunc_idx = random.randrange(room_count) if require_truncation else None

    specs: list[dict] = []
    for idx in range(room_count):
        spec = _make_room_spec(
            room_count=room_count,
            window_color=colors[idx],
            force_corner=idx == force_corner_idx,
            force_truncation=idx == force_trunc_idx,
        )
        if spec is None:
            return None
        specs.append(spec)

    grid = _layout_rooms(shape, specs)
    if grid is None:
        return None

    if require_corner and not any(spec["has_corner_window"] for spec in specs):
        return None
    if require_truncation and not any(spec["truncate_dir"] is not None for spec in specs):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
