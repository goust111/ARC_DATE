import random
from collections import Counter

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


BACKGROUND = Color.BLACK
CURSOR_COLOR = Color.GREY
OBSTACLE_COLOR = Color.TEAL
FRAME_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.MAROON]

DIR_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

MAX_GENERATE_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude=None) -> list[int]:
    colors = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _find_gap(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> tuple[tuple[int, int], str]:
    r0, c0, r1, c1 = bbox
    missing = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                if int(grid[r, c]) != int(color):
                    missing.append((r, c))

    assert len(missing) == 1
    gap = missing[0]
    assert gap not in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    gr, gc = gap
    if gr == r0:
        return gap, "up"
    if gr == r1:
        return gap, "down"
    if gc == c0:
        return gap, "left"
    return gap, "right"


def _measure_gap_distance(cursor: tuple[int, int] | None, bbox: tuple[int, int, int, int], direction: str) -> int:
    if cursor is None:
        return 0
    r0, c0, r1, c1 = bbox
    r, c = cursor
    if direction == "up":
        return int(r - r0)
    if direction == "down":
        return int(r1 - r)
    if direction == "left":
        return int(c - c0)
    return int(c1 - c)


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == BACKGROUND

    gray_all = {tuple(map(int, rc)) for rc in np.argwhere(g == CURSOR_COLOR)}
    assigned_gray = set()
    frames = []
    for color in colors_present(g, exclude={bg, CURSOR_COLOR, OBSTACLE_COLOR}):
        for coords in components_mask(g == color, connectivity=4):
            bbox = bbox_coords(coords)
            r0, c0, r1, c1 = bbox
            assert r1 - r0 >= 2 and c1 - c0 >= 2
            assert all(r in (r0, r1) or c in (c0, c1) for r, c in coords)

            gap, direction = _find_gap(g, color, bbox)
            inside_gray = [
                (r, c)
                for r, c in gray_all
                if r0 < r < r1 and c0 < c < c1
            ]
            assert len(inside_gray) <= 1
            cursor = inside_gray[0] if inside_gray else None
            if cursor is not None:
                assigned_gray.add(cursor)
            distance = _measure_gap_distance(cursor, bbox, direction)

            frames.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox,
                    "gap": gap,
                    "direction": direction,
                    "cursor": cursor,
                    "gap_distance": int(distance),
                }
            )

    assert frames
    assert assigned_gray == gray_all
    return frames


def _travel_until_contact(
    coords: np.ndarray,
    obstacle_mask: np.ndarray,
    direction: str,
    shape: tuple[int, int],
) -> int:
    dr, dc = DIR_TO_VEC[direction]
    h, w = shape
    best = None
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        steps = 0
        while True:
            nr = rr + dr
            nc = cc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if obstacle_mask[nr, nc]:
                break
            rr = nr
            cc = nc
            steps += 1
        best = steps if best is None else min(best, steps)
    assert best is not None
    return int(best)


def _shift_coords(coords: np.ndarray, direction: str, steps: int) -> np.ndarray:
    dr, dc = DIR_TO_VEC[direction]
    shift = np.array([dr * steps, dc * steps], dtype=int)
    return np.asarray(coords, dtype=int) + shift


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each gapped rectangular frame along its gap direction and stop with the cursor-implied gap.

    Input structure:
    - The background is black.
    - Several non-cyan colors form hollow rectangular frames; each frame has exactly one missing non-corner border cell.
    - Some frames contain one grey cursor pixel in their interior, and cyan solid bars act as fixed obstacles.

    Rule:
    - For each frame, use the side containing the missing border cell as the front side and translate the whole frame outward along that side's normal.
    - First compute how far the frame could move before any of its colored border cells would touch a cyan obstacle or the grid edge.
    - If the frame contains a grey cursor, measure its distance d to the front side along the motion axis and stop d cells before contact; frames without a cursor stop at contact.
    - Keep obstacle bars unchanged and preserve each frame's color, shape, gap placement, and cursor's position relative to the frame.

    Key insight:
    - The border gap decides the travel direction, while the grey cursor is an internal ruler whose front-side distance becomes the final external stopping gap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _extract_frames(g)
    obstacle_mask = (g == OBSTACLE_COLOR)

    out = g.copy()
    for frame in frames:
        coords = frame["coords"]
        out[coords[:, 0], coords[:, 1]] = BACKGROUND
        cursor = frame["cursor"]
        if cursor is not None:
            out[cursor] = BACKGROUND

    for frame in frames:
        contact_steps = _travel_until_contact(frame["coords"], obstacle_mask, frame["direction"], g.shape)
        gap_steps = int(frame["gap_distance"])
        assert contact_steps >= gap_steps
        travel = contact_steps - gap_steps

        moved = _shift_coords(frame["coords"], frame["direction"], travel)
        assert np.all(out[moved[:, 0], moved[:, 1]] == BACKGROUND)
        out[moved[:, 0], moved[:, 1]] = frame["color"]

        cursor = frame["cursor"]
        if cursor is not None:
            cr, cc = cursor
            dr, dc = DIR_TO_VEC[frame["direction"]]
            new_cursor = (cr + dr * travel, cc + dc * travel)
            assert out[new_cursor] == BACKGROUND
            out[new_cursor] = CURSOR_COLOR

    return out


def _covered_choices(values: list, n: int, extra_prefix: list | None = None) -> list:
    out = list(values)
    if extra_prefix is not None:
        out.extend(extra_prefix)
    while len(out) < n:
        out.append(random.choice(values))
    random.shuffle(out)
    return out[:n]


def _draw_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    direction: str,
    cursor: tuple[int, int] | None,
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    if direction == "up":
        gap = (r0, random.randint(c0 + 1, c1 - 1))
    elif direction == "down":
        gap = (r1, random.randint(c0 + 1, c1 - 1))
    elif direction == "left":
        gap = (random.randint(r0 + 1, r1 - 1), c0)
    else:
        gap = (random.randint(r0 + 1, r1 - 1), c1)

    grid[gap] = BACKGROUND
    if cursor is not None:
        grid[cursor] = CURSOR_COLOR
    return gap


def _gen_vertical() -> tuple[np.ndarray, list[dict]] | None:
    lane_count = random.randint(3, 4)
    lane_width = 6
    height = random.randint(19, 28)
    width = 2 + lane_count * lane_width + (lane_count - 1)
    grid = np.full((height, width), BACKGROUND, dtype=int)

    directions = _covered_choices(["up", "down"], lane_count)
    blockers = _covered_choices(["edge", "obstacle"], lane_count)
    cursors = _covered_choices([True, False], lane_count, extra_prefix=[True])
    colors = random.sample(FRAME_COLORS, lane_count)

    meta = []
    used_d = []
    for i in range(lane_count):
        lane_c0 = 1 + i * (lane_width + 1)
        lane_c1 = lane_c0 + lane_width - 1
        direction = directions[i]
        blocker = blockers[i]
        color = colors[i]

        frame_h = random.randint(3, 7)
        frame_w = random.randint(4, 5)
        c0 = lane_c0 + random.randint(0, lane_width - frame_w)
        c1 = c0 + frame_w - 1

        has_cursor = bool(cursors[i])
        if has_cursor:
            d = random.randint(1, frame_h - 2)
            used_d.append(d)
        else:
            d = 0
        min_contact = d + 1 if has_cursor else 1

        if direction == "up":
            start_min = min_contact + (2 if blocker == "obstacle" else 1)
            start_max = height - frame_h - 1
            if start_min > start_max:
                return None
            r0 = random.randint(start_min, start_max)
            r1 = r0 + frame_h - 1
            if blocker == "edge":
                blocker_coord = None
                contact = r0
            else:
                hi = r0 - min_contact - 1
                if 1 > hi:
                    return None
                blocker_coord = random.randint(1, hi)
                grid[blocker_coord, lane_c0 : lane_c1 + 1] = OBSTACLE_COLOR
                contact = r0 - blocker_coord - 1
            cursor = (r0 + d, random.randint(c0 + 1, c1 - 1)) if has_cursor else None
        else:
            start_min = 1
            start_max = height - frame_h - min_contact - (1 if blocker == "obstacle" else 0)
            if start_min > start_max:
                return None
            r0 = random.randint(start_min, start_max)
            r1 = r0 + frame_h - 1
            if blocker == "edge":
                blocker_coord = None
                contact = height - 1 - r1
            else:
                lo = r1 + min_contact + 1
                hi = height - 2
                if lo > hi:
                    return None
                blocker_coord = random.randint(lo, hi)
                grid[blocker_coord, lane_c0 : lane_c1 + 1] = OBSTACLE_COLOR
                contact = blocker_coord - r1 - 1
            cursor = (r1 - d, random.randint(c0 + 1, c1 - 1)) if has_cursor else None

        travel = contact - d
        if travel <= 0:
            return None

        bbox = (r0, c0, r1, c1)
        gap = _draw_frame(grid, bbox, color, direction, cursor)
        meta.append(
            {
                "color": color,
                "direction": direction,
                "cursor": cursor,
                "d": d,
                "gap": gap,
                "bbox": bbox,
                "blocker": blocker,
                "blocker_coord": blocker_coord,
                "contact": contact,
                "travel": travel,
            }
        )

    if len(used_d) >= 2 and len(set(used_d)) < 2:
        return None
    return grid, meta


def _gen_horizontal() -> tuple[np.ndarray, list[dict]] | None:
    lane_count = random.randint(3, 4)
    lane_height = 6
    height = 2 + lane_count * lane_height + (lane_count - 1)
    width = random.randint(19, 28)
    grid = np.full((height, width), BACKGROUND, dtype=int)

    directions = _covered_choices(["left", "right"], lane_count)
    blockers = _covered_choices(["edge", "obstacle"], lane_count)
    cursors = _covered_choices([True, False], lane_count, extra_prefix=[True])
    colors = random.sample(FRAME_COLORS, lane_count)

    meta = []
    used_d = []
    for i in range(lane_count):
        lane_r0 = 1 + i * (lane_height + 1)
        lane_r1 = lane_r0 + lane_height - 1
        direction = directions[i]
        blocker = blockers[i]
        color = colors[i]

        frame_h = random.randint(4, 5)
        frame_w = random.randint(3, 7)
        r0 = lane_r0 + random.randint(0, lane_height - frame_h)
        r1 = r0 + frame_h - 1

        has_cursor = bool(cursors[i])
        if has_cursor:
            d = random.randint(1, frame_w - 2)
            used_d.append(d)
        else:
            d = 0
        min_contact = d + 1 if has_cursor else 1

        if direction == "left":
            start_min = min_contact + (2 if blocker == "obstacle" else 1)
            start_max = width - frame_w - 1
            if start_min > start_max:
                return None
            c0 = random.randint(start_min, start_max)
            c1 = c0 + frame_w - 1
            if blocker == "edge":
                blocker_coord = None
                contact = c0
            else:
                hi = c0 - min_contact - 1
                if 1 > hi:
                    return None
                blocker_coord = random.randint(1, hi)
                grid[lane_r0 : lane_r1 + 1, blocker_coord] = OBSTACLE_COLOR
                contact = c0 - blocker_coord - 1
            cursor = (random.randint(r0 + 1, r1 - 1), c0 + d) if has_cursor else None
        else:
            start_min = 1
            start_max = width - frame_w - min_contact - (1 if blocker == "obstacle" else 0)
            if start_min > start_max:
                return None
            c0 = random.randint(start_min, start_max)
            c1 = c0 + frame_w - 1
            if blocker == "edge":
                blocker_coord = None
                contact = width - 1 - c1
            else:
                lo = c1 + min_contact + 1
                hi = width - 2
                if lo > hi:
                    return None
                blocker_coord = random.randint(lo, hi)
                grid[lane_r0 : lane_r1 + 1, blocker_coord] = OBSTACLE_COLOR
                contact = blocker_coord - c1 - 1
            cursor = (random.randint(r0 + 1, r1 - 1), c1 - d) if has_cursor else None

        travel = contact - d
        if travel <= 0:
            return None

        bbox = (r0, c0, r1, c1)
        gap = _draw_frame(grid, bbox, color, direction, cursor)
        meta.append(
            {
                "color": color,
                "direction": direction,
                "cursor": cursor,
                "d": d,
                "gap": gap,
                "bbox": bbox,
                "blocker": blocker,
                "blocker_coord": blocker_coord,
                "contact": contact,
                "travel": travel,
            }
        )

    if len(used_d) >= 2 and len(set(used_d)) < 2:
        return None
    return grid, meta


def _validate_instance(grid: np.ndarray, meta: list[dict]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    directions = {item["direction"] for item in meta}
    blockers = {item["blocker"] for item in meta}
    has_cursor = {item["cursor"] is not None for item in meta}
    if len(directions) < 2 or blockers != {"edge", "obstacle"} or has_cursor != {False, True}:
        return False

    cursor_ds = [item["d"] for item in meta if item["cursor"] is not None]
    if len(cursor_ds) >= 2 and len(set(cursor_ds)) < 2:
        return False

    frames_in = {frame["color"]: frame for frame in _extract_frames(grid)}
    if set(frames_in) != {item["color"] for item in meta}:
        return False

    obstacle_mask = (grid == OBSTACLE_COLOR)
    for item in meta:
        frame = frames_in[item["color"]]
        if frame["direction"] != item["direction"]:
            return False
        if frame["gap"] != item["gap"]:
            return False
        if frame["gap_distance"] != item["d"]:
            return False
        if frame["cursor"] != item["cursor"]:
            return False
        if _travel_until_contact(frame["coords"], obstacle_mask, frame["direction"], grid.shape) != item["contact"]:
            return False
        if item["travel"] <= 0:
            return False

    out = main(grid)
    if out.shape != grid.shape or np.any(out[grid == OBSTACLE_COLOR] != OBSTACLE_COLOR):
        return False

    frames_out = {frame["color"]: frame for frame in _extract_frames(out)}
    if set(frames_out) != set(frames_in):
        return False

    for item in meta:
        src = frames_in[item["color"]]
        dst = frames_out[item["color"]]
        moved = _shift_coords(src["coords"], item["direction"], item["travel"])
        if set(map(tuple, moved)) != set(map(tuple, dst["coords"])):
            return False
        if _travel_until_contact(dst["coords"], obstacle_mask, item["direction"], grid.shape) != item["d"]:
            return False
        if src["cursor"] is None:
            if dst["cursor"] is not None:
                return False
        else:
            dr, dc = DIR_TO_VEC[item["direction"]]
            expected_cursor = (src["cursor"][0] + dr * item["travel"], src["cursor"][1] + dc * item["travel"])
            if dst["cursor"] != expected_cursor:
                return False

    if np.array_equal(grid, out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        built = _gen_vertical() if random.random() < 0.5 else _gen_horizontal()
        if built is None:
            continue
        grid, meta = built
        if _validate_instance(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid gap-slider puzzle input")
