import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r: int, c: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yield r + dr, c + dc


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(coords)
    return out


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _centered_interval(center: int, length: int):
    start = center - (length // 2)
    return start, start + length - 1


def _interval_conflict(interval, used, gap: int = 1):
    lo, hi = interval
    for ulo, uhi in used:
        if not (hi + gap < ulo or uhi + gap < lo):
            return True
    return False


def _valid_centers(lo: int, hi: int, length: int):
    vals = []
    for center in range(lo, hi + 1):
        start, end = _centered_interval(center, length)
        if lo <= start and end <= hi:
            vals.append(center)
    return vals


def _build_object_cells(r0: int, c0: int, size: int, direction: str, shaft_len: int, offset: int):
    coords = {(r, c) for r in range(r0, r0 + size) for c in range(c0, c0 + size)}
    if direction == "right":
        ray_r = r0 + offset
        for step in range(1, shaft_len + 1):
            coords.add((ray_r, c0 + size - 1 + step))
        base_c = c0 + size - 1 + shaft_len
        coords.update({(ray_r, base_c + 1), (ray_r - 1, base_c), (ray_r + 1, base_c)})
        tip = (ray_r, base_c + 1)
        ray_axis = ray_r
    elif direction == "left":
        ray_r = r0 + offset
        for step in range(1, shaft_len + 1):
            coords.add((ray_r, c0 - step))
        base_c = c0 - shaft_len
        coords.update({(ray_r, base_c - 1), (ray_r - 1, base_c), (ray_r + 1, base_c)})
        tip = (ray_r, base_c - 1)
        ray_axis = ray_r
    elif direction == "down":
        ray_c = c0 + offset
        for step in range(1, shaft_len + 1):
            coords.add((r0 + size - 1 + step, ray_c))
        base_r = r0 + size - 1 + shaft_len
        coords.update({(base_r + 1, ray_c), (base_r, ray_c - 1), (base_r, ray_c + 1)})
        tip = (base_r + 1, ray_c)
        ray_axis = ray_c
    else:
        ray_c = c0 + offset
        for step in range(1, shaft_len + 1):
            coords.add((r0 - step, ray_c))
        base_r = r0 - shaft_len
        coords.update({(base_r - 1, ray_c), (base_r, ray_c - 1), (base_r, ray_c + 1)})
        tip = (base_r - 1, ray_c)
        ray_axis = ray_c
    return coords, tip, ray_axis


def _detect_object(coords):
    pts = set(coords)
    rmin, cmin, rmax, cmax = _bbox(coords)
    candidates = []
    for size in (2, 3):
        for r0 in range(rmin, rmax - size + 2):
            for c0 in range(cmin, cmax - size + 2):
                square = {(r, c) for r in range(r0, r0 + size) for c in range(c0, c0 + size)}
                if not square.issubset(pts):
                    continue
                rem = pts - square
                shaft_len = len(rem) - 3
                if shaft_len < 1:
                    continue
                for direction in DIRS:
                    for offset in range(size):
                        expected, tip, ray_axis = _build_object_cells(
                            r0, c0, size, direction, shaft_len, offset
                        )
                        if expected == pts:
                            candidates.append(
                                {
                                    "size": size,
                                    "direction": direction,
                                    "shaft_len": shaft_len,
                                    "square": (r0, c0, r0 + size - 1, c0 + size - 1),
                                    "offset": offset,
                                    "tip": tip,
                                    "ray_axis": ray_axis,
                                }
                            )
    assert len(candidates) == 1, f"ambiguous arrow object: {candidates}"
    return candidates[0]


def _extract_walls(grid: np.ndarray):
    walls = []
    for coords in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        if r0 == r1:
            assert len(coords) == c1 - c0 + 1
            walls.append({"orientation": "horizontal", "row": r0, "c0": c0, "c1": c1})
        else:
            assert c0 == c1
            assert len(coords) == r1 - r0 + 1
            walls.append({"orientation": "vertical", "col": c0, "r0": r0, "r1": r1})
    assert walls
    return walls


def _find_wall(walls, obj):
    direction = obj["direction"]
    tip_r, tip_c = obj["tip"]
    if direction == "right":
        cands = [
            wall
            for wall in walls
            if wall["orientation"] == "vertical"
            and wall["col"] > tip_c
            and wall["r0"] <= obj["ray_axis"] <= wall["r1"]
        ]
        assert cands
        return min(cands, key=lambda wall: wall["col"])
    if direction == "left":
        cands = [
            wall
            for wall in walls
            if wall["orientation"] == "vertical"
            and wall["col"] < tip_c
            and wall["r0"] <= obj["ray_axis"] <= wall["r1"]
        ]
        assert cands
        return max(cands, key=lambda wall: wall["col"])
    if direction == "down":
        cands = [
            wall
            for wall in walls
            if wall["orientation"] == "horizontal"
            and wall["row"] > tip_r
            and wall["c0"] <= obj["ray_axis"] <= wall["c1"]
        ]
        assert cands
        return min(cands, key=lambda wall: wall["row"])
    cands = [
        wall
        for wall in walls
        if wall["orientation"] == "horizontal"
        and wall["row"] < tip_r
        and wall["c0"] <= obj["ray_axis"] <= wall["c1"]
    ]
    assert cands
    return max(cands, key=lambda wall: wall["row"])


def main(grid: np.ndarray) -> np.ndarray:
    """Project each colored square onto the nearest perpendicular grey wall using its arrow as direction-and-scale code.

    Input structure:
    - The background is black.
    - Grey pixels form straight horizontal or vertical wall segments near the grid edges.
    - Each non-grey foreground object is a solid 2x2 or 3x3 colored square with a same-colored arrow attached to one side.
    - The arrow consists of a one-pixel-wide shaft and a 3-pixel head pointing up, down, left, or right.

    Rule:
    - For each colored connected component, identify the embedded square, the arrow direction, and the shaft length.
    - Continue the arrow ray in that direction and find the nearest grey wall whose orientation is perpendicular to the arrow.
    - Draw a same-colored segment on that wall, centered on the ray/wall intersection, with length equal to square side length plus shaft length.
    - Keep the original squares, arrows, and all uncovered grey wall cells unchanged.

    Key insight:
    - The arrow encodes two instructions at once: its direction selects which wall receives the projection, and its shaft length increases the projection size beyond the source square.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    walls = _extract_walls(g)

    objects = []
    for color in sorted(set(map(int, np.unique(g))) - {BG, GREY}):
        for coords in _components(g == color):
            obj = _detect_object(coords)
            obj["color"] = color
            objects.append(obj)

    assert objects
    for obj in objects:
        wall = _find_wall(walls, obj)
        proj_len = obj["size"] + obj["shaft_len"]
        color = obj["color"]
        if wall["orientation"] == "vertical":
            start, end = _centered_interval(obj["ray_axis"], proj_len)
            assert wall["r0"] <= start and end <= wall["r1"]
            for r in range(start, end + 1):
                assert out[r, wall["col"]] in (GREY, color)
                out[r, wall["col"]] = color
        else:
            start, end = _centered_interval(obj["ray_axis"], proj_len)
            assert wall["c0"] <= start and end <= wall["c1"]
            for c in range(start, end + 1):
                assert out[wall["row"], c] in (GREY, color)
                out[wall["row"], c] = color
    return out


def _wall_specs(h: int, w: int):
    vert_max = min(15, h - 8)
    horiz_max = min(15, w - 8)
    assert vert_max >= 5 and horiz_max >= 5

    left_col = random.randint(0, 2)
    right_col = w - 1 - random.randint(0, 2)
    top_row = random.randint(0, 2)
    bottom_row = h - 1 - random.randint(0, 2)
    assert right_col - left_col >= 12
    assert bottom_row - top_row >= 12

    def vertical(col: int):
        length = random.randint(5, vert_max)
        r0 = random.randint(4, h - 4 - length)
        return {"orientation": "vertical", "col": col, "r0": r0, "r1": r0 + length - 1}

    def horizontal(row: int):
        length = random.randint(5, horiz_max)
        c0 = random.randint(4, w - 4 - length)
        return {"orientation": "horizontal", "row": row, "c0": c0, "c1": c0 + length - 1}

    return {
        "left": vertical(left_col),
        "right": vertical(right_col),
        "up": horizontal(top_row),
        "down": horizontal(bottom_row),
    }


def _cells_fit(coords, h: int, w: int):
    return all(0 <= r < h and 0 <= c < w for r, c in coords)


def _touches_occupied(occupied: np.ndarray, coords):
    h, w = occupied.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                    return True
    return False


def _mark_occupied(occupied: np.ndarray, coords):
    for r, c in coords:
        occupied[r, c] = True


def _candidate_positions(h: int, w: int, walls, direction: str, size: int, shaft_len: int, offset: int):
    if direction == "right":
        ray_r = None
        c_lo = walls["left"]["col"] + 4
        c_hi = walls["right"]["col"] - size - shaft_len - 2
        return [("h", c) for c in range(c_lo, c_hi + 1)]
    if direction == "left":
        c_lo = walls["left"]["col"] + shaft_len + 3
        c_hi = walls["right"]["col"] - size - 3
        return [("h", c) for c in range(c_lo, c_hi + 1)]
    if direction == "down":
        r_lo = walls["up"]["row"] + 4
        r_hi = walls["down"]["row"] - size - shaft_len - 2
        return [("v", r) for r in range(r_lo, r_hi + 1)]
    r_lo = walls["up"]["row"] + shaft_len + 3
    r_hi = walls["down"]["row"] - size - 3
    return [("v", r) for r in range(r_lo, r_hi + 1)]


def _place_objects(h: int, w: int, walls):
    n_objects = random.randint(2, 5)
    counts = Counter()
    directions = []
    for _ in range(n_objects):
        options = [d for d in DIRS if counts[d] < 2]
        direction = random.choice(options)
        counts[direction] += 1
        directions.append(direction)
    random.shuffle(directions)

    colors = random.sample(PALETTE, n_objects)
    occupied = np.zeros((h, w), dtype=bool)
    intervals = {key: [] for key in ("left", "right", "up", "down")}
    objects = []

    for color, direction in zip(colors, directions):
        size = random.choice((2, 3))
        shaft_len = random.randint(1, 4)
        proj_len = size + shaft_len
        wall = walls[direction]
        if wall["orientation"] == "vertical":
            centers = _valid_centers(wall["r0"], wall["r1"], proj_len)
        else:
            centers = _valid_centers(wall["c0"], wall["c1"], proj_len)
        random.shuffle(centers)
        placed = False

        for center in centers:
            interval = _centered_interval(center, proj_len)
            if _interval_conflict(interval, intervals[direction]):
                continue
            offsets = list(range(size))
            random.shuffle(offsets)
            for offset in offsets:
                if direction in ("left", "right"):
                    r0 = center - offset
                    if not (0 <= r0 and r0 + size <= h):
                        continue
                    positions = list(range(*{
                        "right": (walls["left"]["col"] + 4, walls["right"]["col"] - size - shaft_len - 1),
                        "left": (walls["left"]["col"] + shaft_len + 3, walls["right"]["col"] - size - 2),
                    }[direction]))
                    random.shuffle(positions)
                    for c0 in positions:
                        coords, tip, ray_axis = _build_object_cells(
                            r0, c0, size, direction, shaft_len, offset
                        )
                        if not _cells_fit(coords, h, w) or _touches_occupied(occupied, coords):
                            continue
                        if direction == "right" and tip[1] >= walls["right"]["col"] - 1:
                            continue
                        if direction == "left" and tip[1] <= walls["left"]["col"] + 1:
                            continue
                        _mark_occupied(occupied, coords)
                        intervals[direction].append(interval)
                        objects.append(
                            {
                                "color": color,
                                "size": size,
                                "shaft_len": shaft_len,
                                "direction": direction,
                                "offset": offset,
                                "square_r0": r0,
                                "square_c0": c0,
                                "coords": coords,
                                "tip": tip,
                                "ray_axis": ray_axis,
                            }
                        )
                        placed = True
                        break
                    if placed:
                        break
                else:
                    c0 = center - offset
                    if not (0 <= c0 and c0 + size <= w):
                        continue
                    positions = list(range(*{
                        "down": (walls["up"]["row"] + 4, walls["down"]["row"] - size - shaft_len - 1),
                        "up": (walls["up"]["row"] + shaft_len + 3, walls["down"]["row"] - size - 2),
                    }[direction]))
                    random.shuffle(positions)
                    for r0 in positions:
                        coords, tip, ray_axis = _build_object_cells(
                            r0, c0, size, direction, shaft_len, offset
                        )
                        if not _cells_fit(coords, h, w) or _touches_occupied(occupied, coords):
                            continue
                        if direction == "down" and tip[0] >= walls["down"]["row"] - 1:
                            continue
                        if direction == "up" and tip[0] <= walls["up"]["row"] + 1:
                            continue
                        _mark_occupied(occupied, coords)
                        intervals[direction].append(interval)
                        objects.append(
                            {
                                "color": color,
                                "size": size,
                                "shaft_len": shaft_len,
                                "direction": direction,
                                "offset": offset,
                                "square_r0": r0,
                                "square_c0": c0,
                                "coords": coords,
                                "tip": tip,
                                "ray_axis": ray_axis,
                            }
                        )
                        placed = True
                        break
                    if placed:
                        break
            if placed:
                break
        if not placed:
            return None
    return objects


def _render_input(h: int, w: int, walls, objects):
    grid = np.full((h, w), BG, dtype=int)
    for wall in walls.values():
        if wall["orientation"] == "vertical":
            grid[wall["r0"] : wall["r1"] + 1, wall["col"]] = GREY
        else:
            grid[wall["row"], wall["c0"] : wall["c1"] + 1] = GREY
    for obj in objects:
        for r, c in obj["coords"]:
            grid[r, c] = obj["color"]
    return grid


def _expected_output(inp: np.ndarray, walls, objects):
    out = inp.copy()
    for obj in objects:
        proj_len = obj["size"] + obj["shaft_len"]
        wall = walls[obj["direction"]]
        color = obj["color"]
        start, end = _centered_interval(obj["ray_axis"], proj_len)
        if wall["orientation"] == "vertical":
            out[start : end + 1, wall["col"]] = color
        else:
            out[wall["row"], start : end + 1] = color
    return out


def _generate_once():
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    if min(h, w) < 18:
        return None
    walls = _wall_specs(h, w)
    objects = _place_objects(h, w, walls)
    if objects is None:
        return None
    inp = _render_input(h, w, walls, objects)
    expected = _expected_output(inp, walls, objects)
    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid grid")
