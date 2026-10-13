import random
from collections import Counter

import numpy as np


BG = 0
WALL = 5
OBJECT_COLORS = (1, 2, 3, 7)
MAX_ATTEMPTS = 400


SHAPES = [
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
]

SQUARE4 = ((0, 0), (0, 1), (1, 0), (1, 1))
LINE3V = ((0, 0), (1, 0), (2, 0))
L3 = ((0, 0), (1, 0), (1, 1))


def _shape_size(shape):
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_objects(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BG, WALL) or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "top": int(arr[:, 0].min()),
                    "bottom": int(arr[:, 0].max()),
                    "left": int(arr[:, 1].min()),
                }
            )
    return objects


def _wall_columns(grid):
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == WALL)]


def _chambers_from_walls(grid):
    walls = _wall_columns(grid)
    spans = []
    start = 0
    for wall in walls:
        spans.append((start, wall - 1))
        start = wall + 1
    spans.append((start, grid.shape[1] - 1))
    return spans


def _settle_objects(grid):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full_like(g, BG)
    out[:, _wall_columns(g)] = WALL
    occupied = out == WALL

    objects = _find_objects(g)
    objects.sort(key=lambda obj: (-obj["bottom"], -obj["top"], obj["left"]))

    for obj in objects:
        coords = obj["coords"]
        rows = coords[:, 0]
        cols = coords[:, 1]
        shift = 0
        while True:
            next_rows = rows + shift + 1
            if np.any(next_rows >= h):
                break
            if np.any(occupied[next_rows, cols]):
                break
            shift += 1
        dest_rows = rows + shift
        out[dest_rows, cols] = obj["color"]
        occupied[dest_rows, cols] = True

    return out


def _winner_for_chamber(chamber):
    pts = np.argwhere((chamber != BG) & (chamber != WALL))
    assert len(pts) > 0
    colors = sorted({int(chamber[r, c]) for r, c in pts})

    def score(color):
        locs = np.argwhere(chamber == color)
        count = int(locs.shape[0])
        bottom = int(locs[:, 0].max())
        left = int(locs[:, 1].min())
        top = int(locs[:, 0].min())
        return count, bottom, -left, -top

    return max(colors, key=score)


def _fill_chambers(settled):
    out = settled.copy()
    for c0, c1 in _chambers_from_walls(settled):
        chamber = settled[:, c0 : c1 + 1]
        winner = _winner_for_chamber(chamber)
        mask = chamber == BG
        out[:, c0 : c1 + 1][mask] = winner
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each floating object downward inside its vertical chamber, then fill each chamber by its settled majority color.

    Input structure:
    - The background is black, and 2-4 full-height grey columns act as vertical walls that split the grid into chambers.
    - Each non-wall foreground object is a monochromatic 4-connected component in blue, red, green, or orange.
    - Every object lies entirely inside one chamber and starts in the upper part of the grid without overlapping other objects.

    Rule:
    - Clear the colored objects, keep the grey walls, and process the objects in descending original bottom-edge order.
    - Each object only moves downward, stopping just before it would leave the grid or overlap a previously settled object.
    - After all objects settle, count the colored pixels inside each chamber and choose the chamber winner by highest frequency, then by lowest-reaching color, then by leftmost color.
    - Fill the remaining black cells of that chamber with the chosen winner, preserving the walls and the settled objects.

    Key insight:
    - The gravity order changes which objects become supports, so the color counts used for chamber filling must be computed only after simulating the settling step.
    """
    settled = _settle_objects(np.asarray(grid, dtype=int))
    return _fill_chambers(settled)


def _role_width(role):
    if role == "left_tie":
        return 6
    if role == "floor_majority":
        return 5
    return random.randint(4, 5)


def _single_blueprint(chamber_width, rng):
    choices = [shape for shape in SHAPES if _shape_size(shape)[1] <= chamber_width - 2]
    shape = rng.choice(choices)
    _, sw = _shape_size(shape)
    x = rng.randint(1, chamber_width - sw - 1)
    return [{"shape": shape, "x": x}]


def _pattern_blueprints(role, chamber_width, rng):
    if role == "bottom_tie":
        return [
            {"shape": SQUARE4, "x": 1, "tag": "stack_lower"},
            {"shape": SQUARE4, "x": 1, "tag": "stack_upper"},
        ]
    if role == "left_tie":
        return [
            {"shape": SQUARE4, "x": 1},
            {"shape": SQUARE4, "x": chamber_width - 3},
        ]
    if role == "stack_majority":
        return [
            {"shape": SQUARE4, "x": 1, "tag": "stack_lower"},
            {"shape": L3, "x": 1, "tag": "stack_upper"},
        ]
    if role == "floor_majority":
        return [
            {"shape": SQUARE4, "x": 1},
            {"shape": LINE3V, "x": chamber_width - 2},
        ]
    return _single_blueprint(chamber_width, rng)


def _choose_colors(count, rng):
    return rng.sample(OBJECT_COLORS, count)


def _assign_bottom_rows(objects, height, rng):
    max_bottom = min(height - 2, height // 2 + 2)
    available = list(range(2, max_bottom + 1))
    rng.shuffle(available)
    available.sort(reverse=True)

    stack_lower = [obj for obj in objects if obj.get("tag") == "stack_lower"]
    stack_upper = [obj for obj in objects if obj.get("tag") == "stack_upper"]
    assert len(stack_lower) <= 1 and len(stack_upper) <= 1

    if stack_lower:
        lower = stack_lower[0]
        height_lower, _ = _shape_size(lower["shape"])
        height_upper, _ = _shape_size(stack_upper[0]["shape"])
        candidates = []
        for lower_bottom in available:
            if lower_bottom < height_lower - 1:
                continue
            upper_choices = [b for b in available if b != lower_bottom and b >= height_upper - 1 and b <= lower_bottom - 3]
            if upper_choices:
                candidates.append((lower_bottom, upper_choices))
        assert candidates
        lower_bottom, upper_choices = rng.choice(candidates)
        upper_bottom = rng.choice(upper_choices)
        lower["bottom_row"] = lower_bottom
        stack_upper[0]["bottom_row"] = upper_bottom
        available.remove(lower_bottom)
        available.remove(upper_bottom)

    remaining = [obj for obj in objects if "bottom_row" not in obj]
    assert len(remaining) <= len(available)
    rng.shuffle(remaining)
    for obj, bottom in zip(remaining, available):
        shape_h, _ = _shape_size(obj["shape"])
        assert bottom >= shape_h - 1
        obj["bottom_row"] = bottom


def _cells_for_object(shape, top, left):
    return [(top + r, left + c) for r, c in shape]


def _can_place(grid, cells):
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in cell_set:
                    continue
                if int(grid[nr, nc]) not in (BG, WALL):
                    return False
    return True


def _place_objects(grid, spans, chamber_roles, rng):
    all_objects = []
    for idx, (c0, c1) in enumerate(spans):
        role = chamber_roles[idx]
        chamber_width = c1 - c0 + 1
        blueprints = _pattern_blueprints(role, chamber_width, rng)
        colors = _choose_colors(len(blueprints), rng)
        for blueprint, color in zip(blueprints, colors):
            obj = dict(blueprint)
            obj["color"] = color
            obj["chamber"] = idx
            obj["left_col"] = c0 + obj["x"]
            all_objects.append(obj)

    _assign_bottom_rows(all_objects, grid.shape[0], rng)

    for obj in sorted(all_objects, key=lambda item: item["bottom_row"], reverse=True):
        shape_h, _ = _shape_size(obj["shape"])
        top = obj["bottom_row"] - shape_h + 1
        cells = _cells_for_object(obj["shape"], top, obj["left_col"])
        if not _can_place(grid, cells):
            return None
        for r, c in cells:
            grid[r, c] = obj["color"]
        obj["top_row"] = top
    return all_objects


def _analyze_chambers(grid):
    info = []
    for c0, c1 in _chambers_from_walls(grid):
        chamber = grid[:, c0 : c1 + 1]
        colors = [int(v) for v in chamber.ravel() if int(v) not in (BG, WALL)]
        counts = Counter(colors)
        winner = _winner_for_chamber(chamber)
        ranking = []
        for color, count in counts.items():
            locs = np.argwhere(chamber == color)
            ranking.append(
                {
                    "color": color,
                    "count": count,
                    "bottom": int(locs[:, 0].max()),
                    "left": int(locs[:, 1].min()),
                }
            )
        ranking.sort(key=lambda item: (item["count"], item["bottom"], -item["left"]), reverse=True)
        info.append({"counts": counts, "winner": winner, "ranking": ranking})
    return info


def _valid_input(grid, roles):
    h, _ = grid.shape
    objects = _find_objects(grid)
    if not (5 <= len(objects) <= 8):
        return False
    if any(obj["bottom"] > min(h - 2, h // 2 + 2) for obj in objects):
        return False

    spans = _chambers_from_walls(grid)
    chamber_colors = []
    for idx, (c0, c1) in enumerate(spans):
        colors = set(int(v) for v in grid[:, c0 : c1 + 1].ravel() if int(v) not in (BG, WALL))
        if not colors:
            return False
        chamber_colors.append(colors)

    if not any(len(colors) >= 2 for colors in chamber_colors):
        return False

    settled = _settle_objects(grid)
    filled = _fill_chambers(settled)
    if np.array_equal(filled, grid):
        return False
    if np.sum(filled == BG) >= np.sum(settled == BG):
        return False

    chamber_info = _analyze_chambers(settled)
    if "bottom_tie" in roles:
        idx = roles.index("bottom_tie")
        ranking = chamber_info[idx]["ranking"]
        if len(ranking) < 2 or ranking[-1]["count"] == 0:
            return False
        if not (ranking[0]["count"] == ranking[1]["count"] and ranking[0]["bottom"] > ranking[1]["bottom"]):
            return False
    if "left_tie" in roles:
        idx = roles.index("left_tie")
        ranking = chamber_info[idx]["ranking"]
        if len(ranking) < 2:
            return False
        if not (ranking[0]["count"] == ranking[1]["count"] and ranking[0]["bottom"] == ranking[1]["bottom"] and ranking[0]["left"] < ranking[1]["left"]):
            return False
    return True


def _gen_attempt(rng):
    mode = rng.choices(["bottom_tie", "left_tie", "none"], weights=[4, 4, 2], k=1)[0]
    chamber_count = rng.choices([3, 4, 5], weights=[4, 4, 2], k=1)[0]
    height = rng.randint(14, 18)

    chamber_roles = ["single"] * chamber_count
    special = rng.sample(range(chamber_count), 2)
    if mode == "bottom_tie":
        chamber_roles[special[0]] = "bottom_tie"
        chamber_roles[special[1]] = "floor_majority"
    elif mode == "left_tie":
        chamber_roles[special[0]] = "left_tie"
        chamber_roles[special[1]] = "stack_majority"
    else:
        chamber_roles[special[0]] = "stack_majority"
        chamber_roles[special[1]] = "floor_majority"

    remaining = [idx for idx, role in enumerate(chamber_roles) if role == "single"]
    if remaining and rng.random() < 0.4:
        chamber_roles[rng.choice(remaining)] = "floor_majority"

    widths = [_role_width(role) for role in chamber_roles]
    width = sum(widths) + chamber_count - 1
    if width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    spans = []
    col = 0
    for idx, chamber_width in enumerate(widths):
        spans.append((col, col + chamber_width - 1))
        col += chamber_width
        if idx != chamber_count - 1:
            grid[:, col] = WALL
            col += 1

    placed = _place_objects(grid, spans, chamber_roles, rng)
    if placed is None:
        return None
    if not _valid_input(grid, chamber_roles):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
