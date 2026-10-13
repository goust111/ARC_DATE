import random
from collections import Counter

import numpy as np


BG = 0
MARKER = 7
DIRECTIONS = ("top", "bottom", "left", "right")
COLOR_POOL = (1, 2, 3, 4, 5, 6, 8, 9)
AREA_POOL = (4, 5, 6, 7, 8)
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(pts, dtype=int)
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return objects


def _best_gap_coords(grid: np.ndarray, bg: int, objects: list[dict]) -> np.ndarray:
    bg_coords = np.argwhere(grid == bg)
    assert bg_coords.size > 0
    obj_coords = [obj["coords"] for obj in objects]
    scores = []
    for r, c in bg_coords:
        distances = []
        for coords in obj_coords:
            d = np.abs(coords[:, 0] - r) + np.abs(coords[:, 1] - c)
            distances.append(int(d.min()))
        scores.append((sum(distances), max(distances)))
    best = min(scores)
    chosen = bg_coords[[score == best for score in scores]]
    assert chosen.size > 0
    return chosen


def _distance_to_gap(coords: np.ndarray, gap_coords: np.ndarray) -> np.ndarray:
    diffs = np.abs(coords[:, None, :] - gap_coords[None, :, :]).sum(axis=2)
    return diffs.min(axis=1)


def _tip_pixel(obj: dict, gap_coords: np.ndarray) -> tuple[int, int]:
    distances = _distance_to_gap(obj["coords"], gap_coords)
    best = int(distances.min())
    tips = obj["coords"][distances == best]
    assert len(tips) == 1
    return int(tips[0, 0]), int(tips[0, 1])


def _direction_of_object(obj: dict, gap_coords: np.ndarray) -> str:
    gap_center = gap_coords.mean(axis=0)
    obj_center = obj["coords"].mean(axis=0)
    dr = float(obj_center[0] - gap_center[0])
    dc = float(obj_center[1] - gap_center[1])
    if abs(dr) > abs(dc):
        return "top" if dr < 0 else "bottom"
    return "left" if dc < 0 else "right"


def _assign_directions(objects: list[dict]) -> dict[str, dict]:
    centers = [
        (
            float(obj["coords"][:, 0].mean()),
            float(obj["coords"][:, 1].mean()),
            obj,
        )
        for obj in objects
    ]
    top = min(centers, key=lambda item: item[0])[2]
    bottom = max(centers, key=lambda item: item[0])[2]
    left = min(centers, key=lambda item: item[1])[2]
    right = max(centers, key=lambda item: item[1])[2]
    assigned = {"top": top, "bottom": bottom, "left": left, "right": right}
    assert len({id(obj) for obj in assigned.values()}) == 4
    return assigned


def _refine_gap_coords(directions: dict[str, dict], grid: np.ndarray, bg: int) -> np.ndarray:
    top = directions["top"]["bbox"][2] + 1
    bottom = directions["bottom"]["bbox"][0] - 1
    left = directions["left"]["bbox"][3] + 1
    right = directions["right"]["bbox"][1] - 1
    assert top <= bottom and left <= right
    coords = []
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            assert int(grid[r, c]) == bg
            coords.append((r, c))
    return np.array(coords, dtype=int)


def _winner_cap(obj: dict, direction: str, tip: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = obj["bbox"]
    tip_r, tip_c = tip
    if direction == "top":
        row = r0 - 1
        return [(row, tip_c - 1), (row, tip_c), (row, tip_c + 1)]
    if direction == "bottom":
        row = r1 + 1
        return [(row, tip_c - 1), (row, tip_c), (row, tip_c + 1)]
    if direction == "left":
        col = c0 - 1
        return [(tip_r - 1, col), (tip_r, col), (tip_r + 1, col)]
    col = c1 + 1
    return [(tip_r - 1, col), (tip_r, col), (tip_r + 1, col)]


def _frame_cells(markers: list[tuple[int, int]]) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]]:
    rows = [r for r, _ in markers]
    cols = [c for _, c in markers]
    r0 = min(rows) + 1
    r1 = max(rows) - 1
    c0 = min(cols) + 1
    c1 = max(cols) - 1
    assert r0 < r1 and c0 < c1
    frame = []
    for c in range(c0, c1 + 1):
        frame.append((r0, c))
        frame.append((r1, c))
    for r in range(r0 + 1, r1):
        frame.append((r, c0))
        frame.append((r, c1))
    return (r0, c0, r1, c1), frame


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    objects = _find_components(g, bg)
    assert len(objects) == 4

    areas = [obj["area"] for obj in objects]
    largest = max(areas)
    assert areas.count(largest) == 1
    winner = objects[areas.index(largest)]

    directions = _assign_directions(objects)
    gap_coords = _refine_gap_coords(directions, g, bg)
    tips = {}
    for direction, obj in directions.items():
        tips[direction] = _tip_pixel(obj, gap_coords)

    winner_dir = next(direction for direction, obj in directions.items() if obj is winner)
    markers = [tips[direction] for direction in DIRECTIONS if direction != winner_dir]
    _, frame = _frame_cells(markers)
    cap = _winner_cap(winner, winner_dir, tips[winner_dir])

    out = g.copy()
    for direction in DIRECTIONS:
        if direction == winner_dir:
            continue
        r, c = tips[direction]
        out[r, c] = MARKER

    winner_color = int(winner["color"])
    h, w = g.shape
    for r, c in cap:
        assert 0 <= r < h and 0 <= c < w
        assert int(g[r, c]) == bg
        out[r, c] = winner_color
    for r, c in frame:
        assert 0 <= r < h and 0 <= c < w
        out[r, c] = winner_color

    return out, {
        "bg": bg,
        "objects": objects,
        "gap_coords": gap_coords,
        "winner_dir": winner_dir,
        "winner_color": winner_color,
        "tips": tips,
        "markers": markers,
        "frame": frame,
        "cap": cap,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let the largest center-facing object grow outward while the others donate tip markers that place a frame.

    Input structure:
    - The grid has a black background and exactly four 8-connected monochromatic foreground objects of distinct colors.
    - The four objects sit in the top, bottom, left, and right directions around a shared central background gap.
    - Each object has one unique inward tip pixel that is closest to the central gap, and exactly one object has the largest area.

    Rule:
    - Infer the central gap as the background region minimizing aggregate Manhattan distance to the four objects.
    - Keep the unique largest object as the winner, add a 3-pixel cap on its outer side, and keep its color for all added drawing.
    - Recolor the inward tip of each losing object to orange, then use the min/max rows and columns of those three orange markers to draw a rectangular frame one cell inside those extrema.

    Key insight:
    - The same largest-area decision controls three linked consequences at once: who grows, which three tips become markers, and which color paints the final frame.
    """
    return _solve_details(grid)[0]


def _transform_offsets(direction: str, offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    transformed = []
    for dr, dc in offsets:
        if direction == "top":
            rr, cc = dr, dc
        elif direction == "bottom":
            rr, cc = -dr, dc
        elif direction == "left":
            rr, cc = dc, dr
        else:
            rr, cc = dc, -dr
        transformed.append((rr, cc))
    return transformed


def _build_object(direction: str, tip: tuple[int, int], area: int, rng: random.Random) -> list[tuple[int, int]]:
    patterns = [
        [(0, 0), (-1, 0), (-2, 0), (-1, -1), (-1, 1), (-2, -1), (-2, 1), (-3, 0)],
        [(0, 0), (-1, 0), (-2, 0), (-1, 1), (-1, -1), (-2, 1), (-2, -1), (-3, 0)],
        [(0, 0), (-1, 0), (-2, 0), (-2, -1), (-1, -1), (-2, 1), (-1, 1), (-3, 0)],
    ]
    offsets = _transform_offsets(direction, rng.choice(patterns))[:area]
    tip_r, tip_c = tip
    return [(tip_r + dr, tip_c + dc) for dr, dc in offsets]


def _shape_is_irregular(coords: list[tuple[int, int]]) -> bool:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    h = max(rows) - min(rows) + 1
    w = max(cols) - min(cols) + 1
    return len(coords) < h * w


def _frame_expected_from_tips(winner_dir: str, tips: dict[str, tuple[int, int]]) -> list[tuple[int, int]]:
    markers = [tips[direction] for direction in DIRECTIONS if direction != winner_dir]
    return _frame_cells(markers)[1]


def _validate_instance(inp: np.ndarray, expected: dict) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if int(np.sum(inp != BG)) < 16:
        return False

    objects = _find_components(inp, BG)
    if len(objects) != 4:
        return False
    colors = [obj["color"] for obj in objects]
    if len(set(colors)) != 4 or MARKER in colors:
        return False
    areas = [obj["area"] for obj in objects]
    if len(set(areas)) != 4:
        return False

    derived_out, details = _solve_details(inp)
    if details["winner_dir"] != expected["winner_dir"]:
        return False
    if details["winner_color"] != expected["winner_color"]:
        return False
    if details["tips"] != expected["tips"]:
        return False
    if details["cap"] != expected["cap"]:
        return False
    if details["frame"] != expected["frame"]:
        return False

    gap_bbox = _bbox(expected["gap_coords"])
    if gap_bbox[2] - gap_bbox[0] + 1 != 3 or gap_bbox[3] - gap_bbox[1] + 1 != 3:
        return False

    for r, c in expected["cap"] + expected["frame"]:
        if int(inp[r, c]) != BG:
            return False

    expected_out = expected["output"]
    if not np.array_equal(derived_out, expected_out):
        return False

    marker_rows = [r for r, _ in expected["markers"]]
    marker_cols = [c for _, c in expected["markers"]]
    if max(marker_rows) - min(marker_rows) < 3 or max(marker_cols) - min(marker_cols) < 3:
        return False

    return True


def _generate_attempt(rng: random.Random) -> tuple[np.ndarray, dict] | None:
    h = rng.randint(14, 18)
    w = rng.randint(14, 18)
    gap_top = rng.randint(5, h - 9)
    gap_left = rng.randint(5, w - 9)
    gap_rows = [gap_top, gap_top + 1, gap_top + 2]
    gap_cols = [gap_left, gap_left + 1, gap_left + 2]
    gap_coords = np.array([(r, c) for r in gap_rows for c in gap_cols], dtype=int)

    winner_dir = rng.choice(DIRECTIONS)
    colors = rng.sample(COLOR_POOL, 4)
    colors_by_dir = dict(zip(rng.sample(list(DIRECTIONS), 4), colors))
    areas = rng.sample(AREA_POOL, 4)
    areas.sort(reverse=True)
    other_dirs = [direction for direction in DIRECTIONS if direction != winner_dir]
    rng.shuffle(other_dirs)
    areas_by_dir = {winner_dir: areas[0]}
    for direction, area in zip(other_dirs, areas[1:]):
        areas_by_dir[direction] = area

    tip_rows_lr = rng.sample([gap_top, gap_top + 2], 2)
    tip_cols_tb = rng.sample([gap_left, gap_left + 2], 2)
    tips = {
        "top": (gap_top - 1, gap_cols[1]),
        "bottom": (gap_top + 3, gap_cols[1]),
        "left": (gap_rows[1], gap_left - 1),
        "right": (gap_rows[1], gap_left + 3),
    }
    if winner_dir in ("top", "bottom"):
        tips["left"] = (tip_rows_lr[0], gap_left - 1)
        tips["right"] = (tip_rows_lr[1], gap_left + 3)
        tips["top"] = (gap_top - 1, rng.choice(gap_cols))
        tips["bottom"] = (gap_top + 3, rng.choice(gap_cols))
    else:
        tips["top"] = (gap_top - 1, tip_cols_tb[0])
        tips["bottom"] = (gap_top + 3, tip_cols_tb[1])
        tips["left"] = (rng.choice(gap_rows), gap_left - 1)
        tips["right"] = (rng.choice(gap_rows), gap_left + 3)

    objects = {}
    occupied = set()
    for direction in DIRECTIONS:
        coords = _build_object(direction, tips[direction], areas_by_dir[direction], rng)
        if not _shape_is_irregular(coords):
            return None
        if len(coords) != len(set(coords)):
            return None
        for r, c in coords:
            if not (0 <= r < h and 0 <= c < w):
                return None
            if (r, c) in occupied:
                return None
        occupied.update(coords)
        objects[direction] = coords

    for direction, coords in objects.items():
        if tips[direction] not in coords:
            return None
        non_tip = [pt for pt in coords if pt != tips[direction]]
        if any(abs(r - tips[direction][0]) + abs(c - tips[direction][1]) <= 1 for r, c in non_tip):
            pass

    winner_color = colors_by_dir[winner_dir]
    markers = [tips[direction] for direction in DIRECTIONS if direction != winner_dir]
    try:
        _, frame = _frame_cells(markers)
    except AssertionError:
        return None

    winner_obj = {
        "bbox": _bbox(np.array(objects[winner_dir], dtype=int)),
    }
    cap = _winner_cap(winner_obj, winner_dir, tips[winner_dir])

    for r, c in frame + cap:
        if not (0 <= r < h and 0 <= c < w):
            return None
        if (r, c) in occupied:
            return None

    inp = np.zeros((h, w), dtype=int)
    for direction in DIRECTIONS:
        color = colors_by_dir[direction]
        for r, c in objects[direction]:
            inp[r, c] = color

    out = inp.copy()
    for direction in DIRECTIONS:
        if direction == winner_dir:
            continue
        tr, tc = tips[direction]
        out[tr, tc] = MARKER
    for r, c in cap + frame:
        out[r, c] = winner_color

    expected = {
        "winner_dir": winner_dir,
        "winner_color": winner_color,
        "tips": tips,
        "markers": markers,
        "frame": frame,
        "cap": cap,
        "gap_coords": gap_coords,
        "output": out,
    }
    if not _validate_instance(inp, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        result = _generate_attempt(rng)
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid input")
