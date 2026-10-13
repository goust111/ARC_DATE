import numpy as np
from collections import deque


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _cc4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _foreground_components(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for pts in _cc4(grid == color):
            comps.append((color, pts))
    comps.sort(key=lambda item: (item[0], _bbox(item[1])))
    return comps


def _nearest_edge(h: int, w: int, bbox_: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox_
    row2 = r0 + r1
    col2 = c0 + c1
    distances = {
        "top": row2,
        "bottom": 2 * (h - 1) - row2,
        "left": col2,
        "right": 2 * (w - 1) - col2,
    }
    return min(distances.items(), key=lambda item: item[1])[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flatten each interior foreground object onto the nearest grid edge.

    The nearest edge is decided by the object's bounding-box center. The output
    discards every original shape and paints a one-cell-thick line on that edge.
    The line starts at the center's projection to the edge and extends rightward
    (for top/bottom) or downward (for left/right).
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    out = np.full_like(g, bg)

    for color, pts in _foreground_components(g, bg):
        area = len(pts)
        r0, c0, r1, c1 = _bbox(pts)
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        edge = _nearest_edge(h, w, (r0, c0, r1, c1))

        if edge == "top":
            row = 0
            for offset in range(area):
                col = center_c + offset
                if 0 <= col < w:
                    out[row, col] = color
        elif edge == "bottom":
            row = h - 1
            for offset in range(area):
                col = center_c + offset
                if 0 <= col < w:
                    out[row, col] = color
        elif edge == "left":
            col = 0
            for offset in range(area):
                row = center_r + offset
                if 0 <= row < h:
                    out[row, col] = color
        else:
            col = w - 1
            for offset in range(area):
                row = center_r + offset
                if 0 <= row < h:
                    out[row, col] = color

    return out


EDGE_ORDER = ("top", "bottom", "left", "right")
GRID_HEIGHT_RANGE = (12, 20)
GRID_WIDTH_RANGE = (12, 20)
OBJECT_COUNT_RANGE = (2, 5)
ODD_SIZES = (3, 5, 7)
MAX_GENERATION_ATTEMPTS = 400
MAX_OBJECT_ATTEMPTS = 200


def _manhattan_path(a: tuple[int, int], b: tuple[int, int], horizontal_first: bool) -> list[tuple[int, int]]:
    r, c = a
    rr, cc = b
    path = [(r, c)]
    if horizontal_first:
        step = 1 if cc >= c else -1
        for nc in range(c + step, cc + step, step):
            path.append((r, nc))
        step = 1 if rr >= r else -1
        for nr in range(r + step, rr + step, step):
            path.append((nr, cc))
    else:
        step = 1 if rr >= r else -1
        for nr in range(r + step, rr + step, step):
            path.append((nr, c))
        step = 1 if cc >= c else -1
        for nc in range(c + step, cc + step, step):
            path.append((rr, nc))
    return path


def _random_spanning_seed(height: int, width: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    center = (height // 2, width // 2)
    top = (0, int(rng.integers(0, width)))
    bottom = (height - 1, int(rng.integers(0, width)))
    left = (int(rng.integers(0, height)), 0)
    right = (int(rng.integers(0, height)), width - 1)
    cells = {center}
    for endpoint in (top, bottom, left, right):
        cells.update(_manhattan_path(endpoint, center, bool(rng.integers(0, 2))))
    return cells


def _grow_connected(
    seed: set[tuple[int, int]],
    height: int,
    width: int,
    target_area: int,
    rng: np.random.Generator,
) -> set[tuple[int, int]]:
    cells = set(seed)
    while len(cells) < target_area:
        frontier = []
        for r in range(height):
            for c in range(width):
                if (r, c) in cells:
                    continue
                if any((nr, nc) in cells for nr, nc in _neighbors4(r, c)):
                    frontier.append((r, c))
        if not frontier:
            return set()
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    return cells


def _sample_center(edge: str, h: int, w: int, rng: np.random.Generator) -> tuple[int, int] | None:
    for _ in range(60):
        if edge == "top":
            cr = int(rng.integers(2, max(3, h // 3 + 1)))
            cc = int(rng.integers(2, w - 2))
        elif edge == "bottom":
            cr = int(rng.integers(max(2, (2 * h) // 3 - 1), h - 2))
            cc = int(rng.integers(2, w - 2))
        elif edge == "left":
            cr = int(rng.integers(2, h - 2))
            cc = int(rng.integers(2, max(3, w // 3 + 1)))
        else:
            cr = int(rng.integers(2, h - 2))
            cc = int(rng.integers(max(2, (2 * w) // 3 - 1), w - 2))

        dists = [cr, h - 1 - cr, cc, w - 1 - cc]
        if len(set(dists)) < 4:
            continue
        if EDGE_ORDER[int(np.argmin(dists))] == edge:
            return cr, cc
    return None


def _shape_for_object(
    edge: str,
    h: int,
    w: int,
    blocked: np.ndarray,
    rng: np.random.Generator,
) -> tuple[set[tuple[int, int]], dict[str, int]] | None:
    for _ in range(MAX_OBJECT_ATTEMPTS):
        center = _sample_center(edge, h, w, rng)
        if center is None:
            return None
        cr, cc = center

        row_radius_max = min(cr - 1, h - 2 - cr, 3)
        col_radius_max = min(cc - 1, w - 2 - cc, 3)
        height_choices = [size for size in ODD_SIZES if size // 2 <= row_radius_max]
        width_choices = [size for size in ODD_SIZES if size // 2 <= col_radius_max]
        if not height_choices or not width_choices:
            continue

        box_h = int(height_choices[int(rng.integers(0, len(height_choices)))])
        box_w = int(width_choices[int(rng.integers(0, len(width_choices)))])
        top = cr - box_h // 2
        left = cc - box_w // 2

        line_capacity = (w - cc) if edge in ("top", "bottom") else (h - cr)
        max_area = min(box_h * box_w, line_capacity)
        if max_area < 5:
            continue

        for _ in range(30):
            seed = _random_spanning_seed(box_h, box_w, rng)
            base_area = len(seed)
            if base_area > max_area:
                continue
            area = int(rng.integers(base_area, max_area + 1))
            local_cells = _grow_connected(seed, box_h, box_w, area, rng)
            if not local_cells:
                continue

            abs_cells = {(top + r, left + c) for r, c in local_cells}
            if any(blocked[r, c] for r, c in abs_cells):
                continue

            bbox_ = _bbox(list(abs_cells))
            if _nearest_edge(h, w, bbox_) != edge:
                continue
            if ((bbox_[0] + bbox_[2]) // 2, (bbox_[1] + bbox_[3]) // 2) != (cr, cc):
                continue

            return abs_cells, {
                "edge": edge,
                "center_r": cr,
                "center_c": cc,
                "area": area,
            }
    return None


def _mark_blocked(blocked: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True


def _expected_output(h: int, w: int, bg: int, specs: list[tuple[int, dict[str, int]]]) -> np.ndarray:
    out = np.full((h, w), bg, dtype=int)
    for color, spec in specs:
        if spec["edge"] == "top":
            out[0, spec["center_c"] : spec["center_c"] + spec["area"]] = color
        elif spec["edge"] == "bottom":
            out[h - 1, spec["center_c"] : spec["center_c"] + spec["area"]] = color
        elif spec["edge"] == "left":
            out[spec["center_r"] : spec["center_r"] + spec["area"], 0] = color
        else:
            out[spec["center_r"] : spec["center_r"] + spec["area"], w - 1] = color
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_HEIGHT_RANGE))
    w = int(rng.integers(*GRID_WIDTH_RANGE))
    obj_count = int(rng.integers(*OBJECT_COUNT_RANGE))
    bg = int(rng.integers(0, 10))
    colors = [c for c in range(10) if c != bg]
    rng.shuffle(colors)
    obj_colors = colors[:obj_count]
    edges = list(rng.permutation(np.array(EDGE_ORDER, dtype=object))[:obj_count])

    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    specs: list[tuple[int, dict[str, int]]] = []

    for color, edge in zip(obj_colors, edges):
        placed = _shape_for_object(str(edge), h, w, blocked, rng)
        if placed is None:
            return None
        cells, spec = placed
        for r, c in cells:
            grid[r, c] = color
        _mark_blocked(blocked, cells)
        specs.append((color, spec))

    if np.count_nonzero(grid == bg) <= max(spec["area"] for _, spec in specs):
        return None

    expected = _expected_output(h, w, bg, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
