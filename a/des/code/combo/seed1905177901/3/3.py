import numpy as np


BG = 0
GREY = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
# Missing-corner code for the 2x2 L-shaped arrow.
MISSING_TO_DIR = {
    (0, 0): "U",
    (0, 1): "R",
    (1, 1): "D",
    (1, 0): "L",
}
DIR_TO_MISSING = {direction: corner for corner, direction in MISSING_TO_DIR.items()}

GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (14, 24)
OBJECT_COUNT_CHOICES = (1, 2, 3, 4)
FRAME_H_RANGE = (4, 8)
FRAME_W_RANGE = (4, 8)
MIN_TRAVEL = 2
GEN_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 250


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _centered_span(lo: int, hi: int, width: int) -> tuple[int, int]:
    full = hi - lo + 1
    width = max(1, min(width, full))
    slack = full - width
    start = lo + slack // 2
    return start, start + width - 1


def _travel_steps(box: tuple[int, int, int, int], direction: str, shape: tuple[int, int]) -> int:
    r0, c0, r1, c1 = box
    h, w = shape
    if direction == "U":
        return int(r0)
    if direction == "D":
        return int(h - 1 - r1)
    if direction == "L":
        return int(c0)
    return int(w - 1 - c1)


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _draw_moved_object(grid: np.ndarray, obj: dict) -> None:
    r0, c0, r1, c1 = obj["box"]
    color = int(obj["color"])
    direction = obj["direction"]
    steps = _travel_steps(obj["box"], direction, grid.shape)

    if direction in ("L", "R"):
        full_width = r1 - r0 + 1
        for t in range(steps):
            trail_width = max(1, full_width - t // 2)
            rr0, rr1 = _centered_span(r0, r1, trail_width)
            col = c0 + t if direction == "R" else c1 - t
            grid[rr0 : rr1 + 1, col] = color
    else:
        full_width = c1 - c0 + 1
        for t in range(steps):
            trail_width = max(1, full_width - t // 2)
            cc0, cc1 = _centered_span(c0, c1, trail_width)
            row = r1 - t if direction == "U" else r0 + t
            grid[row, cc0 : cc1 + 1] = color

    dr, dc = DIRS[direction]
    nr0 = r0 + dr * steps
    nc0 = c0 + dc * steps
    nr1 = r1 + dr * steps
    nc1 = c1 + dc * steps
    _draw_frame(grid, (nr0, nc0, nr1, nc1), color)
    for rr, cc in obj["arrow_rel"]:
        grid[nr0 + rr, nc0 + cc] = GREY


def _object_input_cells(obj: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = obj["box"]
    cells = set(_rect_border_cells(r0, c0, r1, c1))
    for rr, cc in obj["arrow_rel"]:
        cells.add((r0 + rr, c0 + cc))
    return cells


def _object_output_cells(obj: dict, shape: tuple[int, int]) -> set[tuple[int, int]]:
    canvas = np.zeros(shape, dtype=int)
    _draw_moved_object(canvas, obj)
    return {(int(r), int(c)) for r, c in np.argwhere(canvas != BG)}


def _detect_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    objects: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (BG, GREY)):
        for coords in _components(g == color):
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = _bbox(coords)
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue
            expected = set(_rect_border_cells(r0, c0, r1, c1))
            actual = {(int(r), int(c)) for r, c in coords}
            if actual != expected:
                continue
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            arrow = np.argwhere(inner == GREY)
            if arrow.shape[0] != 3:
                continue
            ar0, ac0 = arrow.min(axis=0)
            ar1, ac1 = arrow.max(axis=0)
            if int(ar1 - ar0) != 1 or int(ac1 - ac0) != 1:
                continue
            rel_arrow = {(int(r - ar0), int(c - ac0)) for r, c in arrow}
            missing = list({(0, 0), (0, 1), (1, 0), (1, 1)} - rel_arrow)
            if len(missing) != 1 or missing[0] not in MISSING_TO_DIR:
                continue
            arrow_rel = tuple(sorted((int(r) + 1, int(c) + 1) for r, c in arrow))
            objects.append(
                {
                    "color": int(color),
                    "box": (r0, c0, r1, c1),
                    "direction": MISSING_TO_DIR[missing[0]],
                    "arrow_rel": arrow_rel,
                }
            )
    objects.sort(key=lambda obj: (obj["box"][0], obj["box"][1], obj["color"]))
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored rectangular frame toward the edge indicated by its inner grey arrow.

    The grey arrow is a 3-pixel L in a 2x2 box; its missing corner determines the direction.
    While the frame translates, it leaves a same-colored tail whose transverse width shrinks by
    one cell every two pixels of travel, clamped at width one.
    """
    g = np.asarray(grid, dtype=int)
    objects = _detect_objects(g)
    if not objects:
        return g.copy()

    out = np.full_like(g, BG)
    for obj in objects:
        _draw_moved_object(out, obj)
    return out


def _boxes_disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _make_object(
    rng: np.random.Generator,
    shape: tuple[int, int],
    color: int,
    direction: str,
) -> dict | None:
    h, w = shape
    oh = int(rng.integers(FRAME_H_RANGE[0], min(FRAME_H_RANGE[1], h - 2) + 1))
    ow = int(rng.integers(FRAME_W_RANGE[0], min(FRAME_W_RANGE[1], w - 2) + 1))

    if oh < 4 or ow < 4:
        return None

    if direction == "U":
        if h - oh < MIN_TRAVEL:
            return None
        r0 = int(rng.integers(MIN_TRAVEL, h - oh + 1))
        c0 = int(rng.integers(0, w - ow + 1))
    elif direction == "D":
        if h - oh < MIN_TRAVEL:
            return None
        r0 = int(rng.integers(0, h - oh - MIN_TRAVEL + 1))
        c0 = int(rng.integers(0, w - ow + 1))
    elif direction == "L":
        if w - ow < MIN_TRAVEL:
            return None
        r0 = int(rng.integers(0, h - oh + 1))
        c0 = int(rng.integers(MIN_TRAVEL, w - ow + 1))
    else:
        if w - ow < MIN_TRAVEL:
            return None
        r0 = int(rng.integers(0, h - oh + 1))
        c0 = int(rng.integers(0, w - ow - MIN_TRAVEL + 1))

    r1 = r0 + oh - 1
    c1 = c0 + ow - 1

    arrow_r = int(rng.integers(1, oh - 2))
    arrow_c = int(rng.integers(1, ow - 2))
    missing = DIR_TO_MISSING[direction]
    arrow_rel = []
    for rr in (0, 1):
        for cc in (0, 1):
            if (rr, cc) != missing:
                arrow_rel.append((arrow_r + rr, arrow_c + cc))

    return {
        "color": int(color),
        "box": (r0, c0, r1, c1),
        "direction": direction,
        "arrow_rel": tuple(sorted(arrow_rel)),
    }


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    target_count = int(rng.choice(OBJECT_COUNT_CHOICES))
    directions = [str(rng.choice(np.array(["U", "D", "L", "R"], dtype=object))) for _ in range(target_count)]
    colors = rng.choice(np.array(FRAME_COLORS, dtype=int), size=target_count, replace=False)

    objects: list[dict] = []
    output_occ: set[tuple[int, int]] = set()
    input_boxes: list[tuple[int, int, int, int]] = []

    for color, direction in zip(colors.tolist(), directions, strict=True):
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            obj = _make_object(rng, (h, w), int(color), direction)
            if obj is None:
                continue

            r0, c0, r1, c1 = obj["box"]
            padded = (r0 - 1, c0 - 1, r1 + 1, c1 + 1)
            if any(not _boxes_disjoint(padded, other) for other in input_boxes):
                continue

            new_output = _object_output_cells(obj, (h, w))
            if output_occ & new_output:
                continue

            input_boxes.append(padded)
            output_occ |= new_output
            objects.append(obj)
            placed = True
            break
        if not placed:
            return None

    grid = np.full((h, w), BG, dtype=int)
    for obj in objects:
        _draw_frame(grid, obj["box"], int(obj["color"]))
        r0, c0, _, _ = obj["box"]
        for rr, cc in obj["arrow_rel"]:
            grid[r0 + rr, c0 + cc] = GREY

    if len(_detect_objects(grid)) != target_count:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if not np.all((out == BG) | (out == GREY) | np.isin(out, FRAME_COLORS)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
