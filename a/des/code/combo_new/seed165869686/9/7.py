import random
from collections import Counter

import numpy as np


FRAME_COLOR = 5
ARROW_COLOR = 8
DIRS = ("up", "down", "left", "right")
EDGE_COLOR_BY_DIR = {
    "up": "top",
    "down": "bottom",
    "left": "left",
    "right": "right",
}
ARROW_PATTERNS = {
    "up": frozenset({(0, 0), (0, 1), (1, 1)}),
    "right": frozenset({(0, 1), (1, 0), (1, 1)}),
    "down": frozenset({(0, 0), (1, 0), (1, 1)}),
    "left": frozenset({(0, 0), (0, 1), (1, 0)}),
}
DIR_BY_PATTERN = {pattern: direction for direction, pattern in ARROW_PATTERNS.items()}

GEN_ATTEMPTS = 500
FRAME_COUNT_RANGE = (4, 6)
GRID_SIDE_MAX = 30


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _mode_1d(values: np.ndarray) -> int:
    return int(Counter(map(int, values.tolist())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _normalize_points(points: np.ndarray) -> frozenset[tuple[int, int]]:
    r0 = int(points[:, 0].min())
    c0 = int(points[:, 1].min())
    return frozenset((int(r - r0), int(c - c0)) for r, c in points)


def _frame_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == FRAME_COLOR):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0 = min(rows)
        r1 = max(rows)
        c0 = min(cols)
        c1 = max(cols)
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        if side_h != side_w or side_h not in (5, 6):
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not bool((patch[0, :] == FRAME_COLOR).all()):
            continue
        if not bool((patch[-1, :] == FRAME_COLOR).all()):
            continue
        if not bool((patch[:, 0] == FRAME_COLOR).all()):
            continue
        if not bool((patch[:, -1] == FRAME_COLOR).all()):
            continue
        inner = patch[1:-1, 1:-1]
        if inner.shape[0] not in (3, 4) or inner.shape[0] != inner.shape[1]:
            continue
        if bool((inner == FRAME_COLOR).any()):
            continue
        if int((inner == ARROW_COLOR).sum()) != 3:
            continue
        boxes.append((r0, c0, r1, c1))
    boxes.sort()
    return boxes


def _arrow_direction(inner: np.ndarray) -> str:
    pts = np.argwhere(inner == ARROW_COLOR)
    assert pts.shape[0] == 3, "each frame must contain exactly three arrow pixels"
    norm = _normalize_points(pts)
    assert norm in DIR_BY_PATTERN, "unrecognized arrow pattern"
    return DIR_BY_PATTERN[norm]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed square with the border-strip color selected by its internal arrow.

    Input structure:
    - The grid has a single dominant background color.
    - The outermost top, bottom, left, and right edges each contain a 1-pixel color strip
      excluding the corner cells; the four strip colors are distinct.
    - The middle area contains several non-overlapping grey square frames with 1-pixel borders.
    - Each frame encloses a 3x3 or 4x4 interior containing one azure 3-pixel arrow marker.

    Rule:
    - Read the arrow direction inside each frame.
    - Map up/down/left/right to the corresponding top/bottom/left/right border strip.
    - Sample that strip's color.
    - Fill every background pixel inside the frame with the sampled color.
    - Keep the grey border and the azure arrow unchanged.

    Key insight:
    - The arrow does not move anything; it acts as an indirection marker that tells you which
      border strip provides the fill color for that frame.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _get_bg(g)

    edge_colors = {
        "up": _mode_1d(g[0, 1:-1]),
        "down": _mode_1d(g[-1, 1:-1]),
        "left": _mode_1d(g[1:-1, 0]),
        "right": _mode_1d(g[1:-1, -1]),
    }

    for r0, c0, r1, c1 in _frame_boxes(g):
        inner_in = g[r0 + 1 : r1, c0 + 1 : c1]
        inner_out = out[r0 + 1 : r1, c0 + 1 : c1]
        direction = _arrow_direction(inner_in)
        fill_color = edge_colors[direction]
        inner_out[inner_out == bg] = fill_color

    return out


def _boxes_conflict(
    box: tuple[int, int, int, int],
    placed: list[tuple[int, int, int, int]],
    gap: int = 1,
) -> bool:
    r0, c0, r1, c1 = box
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + gap < pr0 or pr1 + gap < r0 or c1 + gap < pc0 or pc1 + gap < c0):
            return True
    return False


def _choose_palette(rng: random.Random) -> tuple[int, dict[str, int]]:
    pool = [color for color in range(10) if color not in {FRAME_COLOR, ARROW_COLOR}]
    bg = rng.choice(pool)
    pool.remove(bg)
    top, bottom, left, right = rng.sample(pool, 4)
    return bg, {"up": top, "down": bottom, "left": left, "right": right}


def _frame_specs(rng: random.Random) -> list[tuple[int, str]]:
    count = rng.randint(*FRAME_COUNT_RANGE)
    directions = list(DIRS)
    while len(directions) < count:
        directions.append(rng.choice(DIRS))
    rng.shuffle(directions)

    sizes = [3, 4]
    while len(sizes) < count:
        sizes.append(rng.choice((3, 4)))
    rng.shuffle(sizes)
    return list(zip(sizes, directions))


def _grid_shape(rng: random.Random, frame_count: int) -> tuple[int, int]:
    min_side = 12 + 2 * frame_count
    h = rng.randint(min_side, GRID_SIDE_MAX)
    w = rng.randint(min_side, GRID_SIDE_MAX)
    return h, w


def _place_frames(
    rng: random.Random,
    h: int,
    w: int,
    specs: list[tuple[int, str]],
) -> list[tuple[int, int, int, str]]:
    placed_boxes: list[tuple[int, int, int, int]] = []
    placements: list[tuple[int, int, int, str]] = []
    order = sorted(specs, key=lambda item: item[0], reverse=True)

    for inner_size, direction in order:
        outer = inner_size + 2
        row_stop = h - outer - 1
        col_stop = w - outer - 1
        candidates = [(r, c) for r in range(2, row_stop) for c in range(2, col_stop)]
        rng.shuffle(candidates)
        chosen = None
        for r0, c0 in candidates:
            box = (r0, c0, r0 + outer - 1, c0 + outer - 1)
            if not _boxes_conflict(box, placed_boxes, gap=1):
                chosen = box
                break
        if chosen is None:
            return []
        placed_boxes.append(chosen)
        placements.append((chosen[0], chosen[1], inner_size, direction))

    rng.shuffle(placements)
    return placements


def _arrow_offset(rng: random.Random, inner_size: int, direction: str) -> tuple[int, int]:
    span = inner_size - 1
    if direction == "up":
        return 0, rng.randrange(span)
    if direction == "down":
        return inner_size - 2, rng.randrange(span)
    if direction == "left":
        return rng.randrange(span), 0
    return rng.randrange(span), inner_size - 2


def _draw_instance(
    h: int,
    w: int,
    bg: int,
    edge_colors: dict[str, int],
    placements: list[tuple[int, int, int, str]],
    rng: random.Random,
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)

    grid[0, 1:-1] = edge_colors["up"]
    grid[-1, 1:-1] = edge_colors["down"]
    grid[1:-1, 0] = edge_colors["left"]
    grid[1:-1, -1] = edge_colors["right"]

    for r0, c0, inner_size, direction in placements:
        outer = inner_size + 2
        r1 = r0 + outer - 1
        c1 = c0 + outer - 1
        grid[r0, c0 : c1 + 1] = FRAME_COLOR
        grid[r1, c0 : c1 + 1] = FRAME_COLOR
        grid[r0 : r1 + 1, c0] = FRAME_COLOR
        grid[r0 : r1 + 1, c1] = FRAME_COLOR

        ar0, ac0 = _arrow_offset(rng, inner_size, direction)
        for dr, dc in ARROW_PATTERNS[direction]:
            grid[r0 + 1 + ar0 + dr, c0 + 1 + ac0 + dc] = ARROW_COLOR

    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    h, w = grid.shape
    if h > 30 or w > 30 or h < 10 or w < 10:
        return False

    bg = _get_bg(grid)
    edge_lines = {
        "up": grid[0, 1:-1],
        "down": grid[-1, 1:-1],
        "left": grid[1:-1, 0],
        "right": grid[1:-1, -1],
    }
    edge_colors: dict[str, int] = {}
    for direction, line in edge_lines.items():
        values = set(map(int, np.unique(line)))
        if len(values) != 1:
            return False
        edge_colors[direction] = values.pop()
    if len(set(edge_colors.values())) != 4:
        return False
    if bg in edge_colors.values():
        return False
    if FRAME_COLOR in edge_colors.values() or ARROW_COLOR in edge_colors.values():
        return False
    if int(grid[0, 0]) != bg or int(grid[0, -1]) != bg or int(grid[-1, 0]) != bg or int(grid[-1, -1]) != bg:
        return False

    frames = _frame_boxes(grid)
    if len(frames) < 4:
        return False

    seen_dirs = set()
    seen_sizes = set()
    for i, box in enumerate(frames):
        r0, c0, r1, c1 = box
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        seen_sizes.add(int(inner.shape[0]))
        if not set(map(int, np.unique(inner))).issubset({bg, ARROW_COLOR}):
            return False
        try:
            seen_dirs.add(_arrow_direction(inner))
        except AssertionError:
            return False
        for other in frames[i + 1 :]:
            if _boxes_conflict(box, [other], gap=1):
                return False

    if seen_dirs != set(DIRS):
        return False
    if seen_sizes != {3, 4}:
        return False

    try:
        out = main(grid)
    except Exception:
        return False

    if np.array_equal(out, grid):
        return False
    if not bool((out == ARROW_COLOR).any()):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        specs = _frame_specs(rng)
        h, w = _grid_shape(rng, len(specs))
        placements = _place_frames(rng, h, w, specs)
        if not placements:
            continue
        bg, edge_colors = _choose_palette(rng)
        grid = _draw_instance(h, w, bg, edge_colors, placements, rng)
        if _validate_generated(grid):
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
