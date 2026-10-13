import random
from collections import Counter, deque

import numpy as np


BG_COLOR = 0
WALL_COLOR = 5
WINDOW_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_MIN = 16
GRID_MAX = 30
MAX_GENERATION_ATTEMPTS = 400

MIN_INNER_H = 2
MAX_INNER_H = 6
MIN_INNER_W = 2
MAX_INNER_W = 7
MAX_ROOMS = 5
ROOM_PLACEMENT_ATTEMPTS = 120

WINDOW_CASES = ("none", "top", "bottom", "both")


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            comps.append(np.array(coords, dtype=int))

    return comps


def _find_rooms(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    rooms: list[dict] = []

    for coords in _components(grid == bg):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())

        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue

        inner_h = r1 - r0 + 1
        inner_w = c1 - c0 + 1
        if len(coords) != inner_h * inner_w:
            continue
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == bg):
            continue

        top = grid[r0 - 1, c0 : c1 + 1]
        bottom = grid[r1 + 1, c0 : c1 + 1]
        left = grid[r0 : r1 + 1, c0 - 1]
        right = grid[r0 : r1 + 1, c1 + 1]
        corners = np.array(
            [
                grid[r0 - 1, c0 - 1],
                grid[r0 - 1, c1 + 1],
                grid[r1 + 1, c0 - 1],
                grid[r1 + 1, c1 + 1],
            ],
            dtype=int,
        )

        ring = np.concatenate((top, bottom, left, right, corners))
        if np.any(ring == bg):
            continue

        wall_color, _ = Counter(map(int, ring)).most_common(1)[0]
        if not (np.all(left == wall_color) and np.all(right == wall_color) and np.all(corners == wall_color)):
            continue

        top_cols = [c0 + i for i, value in enumerate(top) if int(value) != wall_color]
        bottom_cols = [c0 + i for i, value in enumerate(bottom) if int(value) != wall_color]
        window_values = [int(top[col - c0]) for col in top_cols] + [int(bottom[col - c0]) for col in bottom_cols]
        if any(value == bg for value in window_values):
            continue

        window_colors = set(window_values)
        if len(window_colors) > 1:
            continue

        rooms.append(
            {
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "top_cols": top_cols,
                "bottom_cols": bottom_cols,
                "window_color": None if not window_values else window_values[0],
            }
        )

    rooms.sort(key=lambda room: (room["r0"], room["c0"]))
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow equal-height vertical pillars inward from each room's windows, with height set by the room's total window count.

    Input structure:
    - The grid has a black background plus one or more closed rectangular rooms outlined by a wall color.
    - Each room interior is a solid black rectangle enclosed by a one-cell-thick wall ring.
    - Some cells on a room's top wall, bottom wall, or both are replaced by same-colored window pixels; side walls stay unchanged.

    Rule:
    - Find every enclosed rectangular black room interior.
    - For each room, count all window pixels on its horizontal walls; call that total N.
    - From each top-wall window, fill N cells downward inside the room, and from each bottom-wall window, fill N cells upward, stopping before the opposite wall if needed.
    - Leave the walls and the original window pixels unchanged, and leave rooms with no windows unchanged.

    Key insight:
    - A room's window count controls pillar height as well as pillar count, so every window in the same room grows a same-height column.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for room in _find_rooms(g, bg):
        total_windows = len(room["top_cols"]) + len(room["bottom_cols"])
        if total_windows == 0:
            continue

        color = int(room["window_color"])
        max_len = room["r1"] - room["r0"] + 1
        length = min(total_windows, max_len)

        for col in room["top_cols"]:
            out[room["r0"] : room["r0"] + length, col] = color
        for col in room["bottom_cols"]:
            out[room["r1"] - length + 1 : room["r1"] + 1, col] = color

    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _sample_windows(case: str, inner_h: int, inner_w: int) -> tuple[list[int], list[int]]:
    if case == "none":
        return [], []

    if case in ("top", "bottom"):
        choices = list(range(1, inner_w + 1))
    else:
        choices = list(range(2, 2 * inner_w + 1))

    wants_truncation = random.random() < 0.5
    if wants_truncation:
        truncated = [n for n in choices if n > inner_h]
        if truncated:
            choices = truncated
    else:
        non_truncated = [n for n in choices if n <= inner_h]
        if non_truncated:
            choices = non_truncated

    total = random.choice(choices)
    if case == "top":
        n_top, n_bottom = total, 0
    elif case == "bottom":
        n_top, n_bottom = 0, total
    else:
        top_lo = max(1, total - inner_w)
        top_hi = min(inner_w, total - 1)
        n_top = random.randint(top_lo, top_hi)
        n_bottom = total - n_top

    top_cols = sorted(random.sample(range(inner_w), n_top))
    bottom_cols = sorted(random.sample(range(inner_w), n_bottom))
    return top_cols, bottom_cols


def _sample_room_layout(height: int, width: int) -> list[dict] | None:
    max_rooms = min(MAX_ROOMS, max(1, (height * width) // 110))
    target_rooms = random.randint(1, max_rooms)
    rooms: list[dict] = []
    blocked: list[tuple[int, int, int, int]] = []

    for _ in range(target_rooms):
        placed = False
        for _attempt in range(ROOM_PLACEMENT_ATTEMPTS):
            inner_h = random.randint(MIN_INNER_H, MAX_INNER_H)
            inner_w = random.randint(MIN_INNER_W, MAX_INNER_W)
            outer_h = inner_h + 2
            outer_w = inner_w + 2

            if outer_h + 2 > height or outer_w + 2 > width:
                continue

            r0 = random.randint(1, height - outer_h - 1)
            c0 = random.randint(1, width - outer_w - 1)
            box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            expanded = (box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1)
            if any(_boxes_overlap(expanded, other) for other in blocked):
                continue

            rooms.append(
                {
                    "box": box,
                    "inner_h": inner_h,
                    "inner_w": inner_w,
                }
            )
            blocked.append(expanded)
            placed = True
            break
        if not placed:
            return None

    cases = [random.choice(WINDOW_CASES) for _ in rooms]
    if all(case == "none" for case in cases):
        cases[random.randrange(len(cases))] = random.choice(("top", "bottom", "both"))
    if len(rooms) > 1 and all(case != "none" for case in cases) and random.random() < 0.5:
        cases[random.randrange(len(cases))] = "none"

    windowed = sum(case != "none" for case in cases)
    if windowed > len(WINDOW_COLORS):
        return None

    colors = random.sample(WINDOW_COLORS, windowed)
    color_index = 0
    for room, case in zip(rooms, cases):
        room["case"] = case
        top_cols, bottom_cols = _sample_windows(case, room["inner_h"], room["inner_w"])
        room["top_cols"] = top_cols
        room["bottom_cols"] = bottom_cols
        room["color"] = None
        if case != "none":
            room["color"] = int(colors[color_index])
            color_index += 1

    return rooms


def _render_pair(height: int, width: int, rooms: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), BG_COLOR, dtype=int)
    for room in rooms:
        r0, c0, r1, c1 = room["box"]
        inp[r0, c0 : c1 + 1] = WALL_COLOR
        inp[r1, c0 : c1 + 1] = WALL_COLOR
        inp[r0 : r1 + 1, c0] = WALL_COLOR
        inp[r0 : r1 + 1, c1] = WALL_COLOR

        if room["color"] is None:
            continue

        for inner_col in room["top_cols"]:
            inp[r0, c0 + 1 + inner_col] = room["color"]
        for inner_col in room["bottom_cols"]:
            inp[r1, c0 + 1 + inner_col] = room["color"]

    out = inp.copy()
    for room in rooms:
        total_windows = len(room["top_cols"]) + len(room["bottom_cols"])
        if total_windows == 0:
            continue

        r0, c0, r1, _c1 = room["box"]
        length = min(total_windows, room["inner_h"])
        color = int(room["color"])

        for inner_col in room["top_cols"]:
            col = c0 + 1 + inner_col
            out[r0 + 1 : r0 + 1 + length, col] = color
        for inner_col in room["bottom_cols"]:
            col = c0 + 1 + inner_col
            out[r1 - length : r1, col] = color

    return inp, out


def _valid_design(inp: np.ndarray, out: np.ndarray, rooms: list[dict]) -> bool:
    if not np.array_equal(main(inp), out):
        return False
    if len(_find_rooms(inp, BG_COLOR)) != len(rooms):
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    rooms = _sample_room_layout(height, width)
    if rooms is None:
        return None

    inp, out = _render_pair(height, width, rooms)
    if not _valid_design(inp, out, rooms):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-window pillar puzzle")
