from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


GEN_ATTEMPTS = 600
PAD = 2
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
FILL_DIRS = ["up", "down", "left", "right"]


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(height: int, width: int, r: int, c: int) -> bool:
    return 0 <= r < height and 0 <= c < width


def _segment_from_neighbor(
    grid: np.ndarray, bg: int, r: int, c: int, dr: int, dc: int
) -> tuple[int, int | None]:
    height, width = grid.shape
    if not _in_bounds(height, width, r, c) or int(grid[r, c]) == bg:
        return 0, None
    color = int(grid[r, c])
    length = 0
    while _in_bounds(height, width, r, c) and int(grid[r, c]) == color:
        length += 1
        r += dr
        c += dc
    return length, color


def _blocked_bg_component_size(
    grid: np.ndarray, bg: int, start: tuple[int, int], blocked: tuple[int, int]
) -> int:
    height, width = grid.shape
    sr, sc = start
    if not _in_bounds(height, width, sr, sc) or int(grid[sr, sc]) != bg:
        return 0

    q = deque([start])
    seen = {start}
    size = 0
    while q:
        r, c = q.popleft()
        size += 1
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if (nr, nc) == blocked or not _in_bounds(height, width, nr, nc):
                continue
            if (nr, nc) in seen or int(grid[nr, nc]) != bg:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return size


def _fill_color(len_a: int, color_a: int, len_b: int, color_b: int) -> int:
    if len_a > len_b:
        return color_a
    if len_b > len_a:
        return color_b
    if color_a == color_b:
        return color_a
    return color_a


def _triangle_cells(r: int, c: int, direction: str, size: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for d in range(size):
        if direction == "up":
            rr = r - d
            for cc in range(c - d, c + d + 1):
                cells.append((rr, cc))
        elif direction == "down":
            rr = r + d
            for cc in range(c - d, c + d + 1):
                cells.append((rr, cc))
        elif direction == "left":
            cc = c - d
            for rr in range(r - d, r + d + 1):
                cells.append((rr, cc))
        else:
            cc = c + d
            for rr in range(r - d, r + d + 1):
                cells.append((rr, cc))
    return cells


def _find_doors(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    doors: list[dict] = []

    for r in range(height):
        for c in range(width):
            if int(g[r, c]) != bg:
                continue

            has_left = c > 0 and int(g[r, c - 1]) != bg
            has_right = c + 1 < width and int(g[r, c + 1]) != bg
            has_up = r > 0 and int(g[r - 1, c]) != bg
            has_down = r + 1 < height and int(g[r + 1, c]) != bg

            is_horizontal = has_left and has_right and not has_up and not has_down
            is_vertical = has_up and has_down and not has_left and not has_right
            if is_horizontal == is_vertical:
                continue

            if is_horizontal:
                left_len, left_color = _segment_from_neighbor(g, bg, r, c - 1, 0, -1)
                right_len, right_color = _segment_from_neighbor(g, bg, r, c + 1, 0, 1)
                if left_len == 0 or right_len == 0:
                    continue
                up_area = _blocked_bg_component_size(g, bg, (r - 1, c), (r, c))
                down_area = _blocked_bg_component_size(g, bg, (r + 1, c), (r, c))
                if up_area == down_area:
                    continue
                direction = "up" if up_area > down_area else "down"
                doors.append(
                    {
                        "r": r,
                        "c": c,
                        "direction": direction,
                        "size": min(left_len, right_len),
                        "color": _fill_color(left_len, int(left_color), right_len, int(right_color)),
                    }
                )
            else:
                up_len, up_color = _segment_from_neighbor(g, bg, r - 1, c, -1, 0)
                down_len, down_color = _segment_from_neighbor(g, bg, r + 1, c, 1, 0)
                if up_len == 0 or down_len == 0:
                    continue
                left_area = _blocked_bg_component_size(g, bg, (r, c - 1), (r, c))
                right_area = _blocked_bg_component_size(g, bg, (r, c + 1), (r, c))
                if left_area == right_area:
                    continue
                direction = "left" if left_area > right_area else "right"
                doors.append(
                    {
                        "r": r,
                        "c": c,
                        "direction": direction,
                        "size": min(up_len, down_len),
                        "color": _fill_color(up_len, int(up_color), down_len, int(down_color)),
                    }
                )

    doors.sort(key=lambda item: (item["r"], item["c"], item["direction"]))
    return doors


def _paint_triangle(out: np.ndarray, inp: np.ndarray, bg: int, door: dict) -> None:
    height, width = out.shape
    for r, c in _triangle_cells(door["r"], door["c"], door["direction"], door["size"]):
        if _in_bounds(height, width, r, c) and int(inp[r, c]) == bg:
            out[r, c] = int(door["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find single-cell wall gaps ("doors") and paint a right-isosceles triangular fan
    into the larger background side of each door.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()
    for door in _find_doors(g, bg):
        _paint_triangle(out, g, bg, door)
    return out


def _object_spec(colors: list[int], force_dual: bool = False) -> dict:
    direction = random.choice(FILL_DIRS)
    if len(colors) == 1:
        left_color = right_color = int(colors[0])
    else:
        if force_dual:
            left_color, right_color = map(int, colors)
        elif random.random() < 0.5:
            wall_color = int(random.choice(colors))
            left_color = right_color = wall_color
        else:
            left_color, right_color = map(int, colors)

    a = random.randint(2, 6)
    b = random.randint(2, 6)
    if left_color != right_color:
        while a == b:
            b = random.randint(2, 6)
    depth = random.randint(3, 6)
    size = min(a, b)

    if direction in ("up", "down"):
        height = depth + size
        width = a + b + 1
        if direction == "down":
            door_r = depth
            door_c = a
            back_r = 0
        else:
            door_r = size - 1
            door_c = a
            back_r = height - 1

        walls: dict[tuple[int, int], int] = {}
        for c in range(width):
            walls[(back_r, c)] = left_color if c <= door_c else right_color
        for c in range(width):
            if c == door_c:
                continue
            walls[(door_r, c)] = left_color if c < door_c else right_color
        r0 = min(back_r, door_r)
        r1 = max(back_r, door_r)
        for r in range(r0, r1 + 1):
            walls[(r, 0)] = left_color
            walls[(r, width - 1)] = right_color
    else:
        height = a + b + 1
        width = depth + size
        if direction == "right":
            door_r = a
            door_c = depth
            back_c = 0
        else:
            door_r = a
            door_c = size - 1
            back_c = width - 1

        walls = {}
        for r in range(height):
            walls[(r, back_c)] = left_color if r <= door_r else right_color
        for r in range(height):
            if r == door_r:
                continue
            walls[(r, door_c)] = left_color if r < door_r else right_color
        c0 = min(back_c, door_c)
        c1 = max(back_c, door_c)
        for c in range(c0, c1 + 1):
            walls[(0, c)] = left_color
            walls[(height - 1, c)] = right_color

    fill_color = _fill_color(a, left_color, b, right_color)
    triangle = set(_triangle_cells(door_r, door_c, direction, size))

    return {
        "direction": direction,
        "a": a,
        "b": b,
        "size": size,
        "height": height,
        "width": width,
        "door_local": (door_r, door_c),
        "walls": walls,
        "triangle": triangle,
        "fill_color": fill_color,
    }


def _try_place_object(
    grid: np.ndarray, occupied: np.ndarray, spec: dict
) -> tuple[int, int] | None:
    height, width = grid.shape
    obj_h = int(spec["height"])
    obj_w = int(spec["width"])

    positions = []
    for top in range(PAD, height - obj_h - PAD + 1):
        for left in range(PAD, width - obj_w - PAD + 1):
            r0 = top - PAD
            r1 = top + obj_h + PAD
            c0 = left - PAD
            c1 = left + obj_w + PAD
            if occupied[r0:r1, c0:c1].any():
                continue
            positions.append((top, left))
    if not positions:
        return None

    top, left = random.choice(positions)
    for (dr, dc), color in spec["walls"].items():
        grid[top + dr, left + dc] = int(color)

    r0 = top - PAD
    r1 = top + obj_h + PAD
    c0 = left - PAD
    c1 = left + obj_w + PAD
    occupied[r0:r1, c0:c1] = True
    return top, left


def _expected_output(inp: np.ndarray, doors: list[dict]) -> np.ndarray:
    bg = _bg_color(inp)
    out = inp.copy()
    for door in doors:
        _paint_triangle(out, inp, bg, door)
    return out


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(18, 28)
    width = random.randint(18, 28)
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)

    color_count = random.choice([1, 2])
    wall_colors = palette[:color_count]

    grid = np.full((height, width), bg, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    target_objects = random.randint(1, 4)
    placed_doors: list[dict] = []
    for idx in range(target_objects):
        spec = _object_spec(wall_colors, force_dual=(color_count == 2 and idx == 0))
        placed = _try_place_object(grid, occupied, spec)
        if placed is None:
            return None
        top, left = placed
        door_r, door_c = spec["door_local"]
        placed_doors.append(
            {
                "r": top + door_r,
                "c": left + door_c,
                "direction": spec["direction"],
                "size": spec["size"],
                "color": spec["fill_color"],
            }
        )

    vals, counts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    detected = _find_doors(grid, bg)
    expected_sorted = sorted(
        placed_doors, key=lambda item: (item["r"], item["c"], item["direction"])
    )
    if detected != expected_sorted:
        return None

    out = _expected_output(grid, placed_doors)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid door-and-fan input")
