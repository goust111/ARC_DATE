import numpy as np


BG = 0
COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 120
MIN_BASE_HEIGHT, MAX_BASE_HEIGHT = 10, 22
MIN_BASE_WIDTH, MAX_BASE_WIDTH = 18, 26
MIN_LINE_MARGIN = 3
MIN_LINE_SEPARATION = 6


WIDE_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2)),
)
TIE_SHAPES = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
)


def _normalize_shape(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    coords = np.array(shape, dtype=int)
    coords -= coords.min(axis=0)
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    return coords[order]


def _transpose_shape(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    coords = _normalize_shape(shape)
    coords = coords[:, [1, 0]]
    coords -= coords.min(axis=0)
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    return coords[order]


HORIZONTAL_SHAPES = tuple(_normalize_shape(shape) for shape in WIDE_SHAPES + TIE_SHAPES)
VERTICAL_SHAPES = tuple(_transpose_shape(shape) for shape in WIDE_SHAPES)


def _shape_dims(shape: np.ndarray) -> tuple[int, int]:
    return int(shape[:, 0].max()) + 1, int(shape[:, 1].max()) + 1


def _collect_full_lines(grid: np.ndarray) -> tuple[dict[int, list[int]], dict[int, list[int]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rows_by_color: dict[int, list[int]] = {}
    cols_by_color: dict[int, list[int]] = {}
    line_mask = np.zeros((h, w), dtype=bool)

    for r in range(h):
        color = int(g[r, 0])
        if color != BG and bool(np.all(g[r] == color)):
            rows_by_color.setdefault(color, []).append(r)
            line_mask[r, :] = True

    for c in range(w):
        color = int(g[0, c])
        if color != BG and bool(np.all(g[:, c] == color)):
            cols_by_color.setdefault(color, []).append(c)
            line_mask[:, c] = True

    return rows_by_color, cols_by_color, line_mask


def _find_components(grid: np.ndarray, line_mask: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or line_mask[r, c] or int(g[r, c]) == BG:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or line_mask[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    return comps


def _axis_from_bbox(comp: dict) -> str:
    r0, c0, r1, c1 = comp["bbox"]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    return "horizontal" if width >= height else "vertical"


def _scene_info(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    rows_by_color, cols_by_color, line_mask = _collect_full_lines(g)
    comps = _find_components(g, line_mask)
    return {
        "rows_by_color": rows_by_color,
        "cols_by_color": cols_by_color,
        "line_mask": line_mask,
        "components": comps,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each small colored object along its longest axis onto the matching full line, leaving only the painted path.

    Input structure:
    - The background is black.
    - Several non-black colors form full-width rows or full-height columns, one line per color.
    - Small 4-connected monochrome objects of area 2-5 are scattered away from the lines.
    - Each object's color matches exactly one full line, and its bounding-box longest axis
      determines whether it projects horizontally or vertically.

    Rule:
    - Detect every small object after excluding the full lines from the foreground.
    - If an object's bounding box is wider than tall, or tied, project it horizontally to the
      same-colored full column; if it is taller than wide, project it vertically to the
      same-colored full row.
    - For each object pixel, fill every black cell along that straight path until the matching
      line is reached.
    - Erase the original object pixels and keep the full lines unchanged.

    Key insight:
    - The destination is not chosen by nearest distance alone; the object's own shape selects the
      projection axis, which in turn identifies the orientation of the same-colored target line.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    info = _scene_info(g)
    rows_by_color = info["rows_by_color"]
    cols_by_color = info["cols_by_color"]
    comps = info["components"]
    line_mask = info["line_mask"]
    object_mask = (g != BG) & (~line_mask)

    for comp in comps:
        for r, c in comp["coords"]:
            out[int(r), int(c)] = BG

    for comp in comps:
        color = int(comp["color"])
        coords = comp["coords"]
        r0, c0, r1, c1 = comp["bbox"]
        axis = _axis_from_bbox(comp)

        if axis == "horizontal":
            assert len(cols_by_color.get(color, [])) == 1, "expected one matching full column"
            line_c = int(cols_by_color[color][0])
            assert line_c < c0 or line_c > c1, "matching column must lie outside the object bbox"
            step = -1 if line_c < c0 else 1
            for rr, cc in coords:
                r = int(rr)
                c = int(cc)
                for path_c in range(c + step, line_c, step):
                    if int(g[r, path_c]) == BG:
                        out[r, path_c] = color
                    else:
                        assert bool(object_mask[r, path_c]) and int(g[r, path_c]) == color, (
                            "non-black path cells must belong to the same source object"
                        )
                assert int(out[r, line_c]) == color, "target column color mismatch"
        else:
            assert len(rows_by_color.get(color, [])) == 1, "expected one matching full row"
            line_r = int(rows_by_color[color][0])
            assert line_r < r0 or line_r > r1, "matching row must lie outside the object bbox"
            step = -1 if line_r < r0 else 1
            for rr, cc in coords:
                r = int(rr)
                c = int(cc)
                for path_r in range(r + step, line_r, step):
                    if int(g[path_r, c]) == BG:
                        out[path_r, c] = color
                    else:
                        assert bool(object_mask[path_r, c]) and int(g[path_r, c]) == color, (
                            "non-black path cells must belong to the same source object"
                        )
                assert int(out[line_r, c]) == color, "target row color mismatch"

    return out


def _sample_line_positions(length: int, count: int, rng: np.random.Generator) -> list[int] | None:
    lo = MIN_LINE_MARGIN
    hi = length - MIN_LINE_MARGIN
    if hi - lo < count:
        return None

    for _ in range(PLACE_ATTEMPTS):
        positions = sorted(int(x) for x in rng.choice(np.arange(lo, hi), size=count, replace=False))
        if positions[0] < MIN_LINE_MARGIN or length - 1 - positions[-1] < MIN_LINE_MARGIN:
            continue
        if any(b - a < MIN_LINE_SEPARATION for a, b in zip(positions, positions[1:])):
            continue
        return positions
    return None


def _line_segments(width: int, positions: list[int], idx: int) -> list[tuple[str, int, int]]:
    line_c = positions[idx]
    prev_line = positions[idx - 1] if idx > 0 else None
    next_line = positions[idx + 1] if idx + 1 < len(positions) else None

    left_start = 1 if prev_line is None else prev_line + 2
    left_end = line_c - 2
    right_start = line_c + 2
    right_end = width - 2 if next_line is None else next_line - 2

    zones: list[tuple[str, int, int]] = []
    if left_start <= left_end:
        zones.append(("left", left_start, left_end))
    if right_start <= right_end:
        zones.append(("right", right_start, right_end))
    return zones


def _bbox_with_margin(coords: np.ndarray, height: int, width: int) -> tuple[slice, slice]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    rs = slice(max(0, int(r0) - 1), min(height, int(r1) + 2))
    cs = slice(max(0, int(c0) - 1), min(width, int(c1) + 2))
    return rs, cs


def _can_place_vertical_object(
    grid: np.ndarray,
    coords: np.ndarray,
    line_c: int,
    side: str,
) -> bool:
    h, w = grid.shape
    rs, cs = _bbox_with_margin(coords, h, w)
    if bool(np.any(grid[rs, cs] != BG)):
        return False

    for rr, cc in coords:
        r = int(rr)
        c = int(cc)
        if side == "left":
            if c >= line_c:
                return False
            path = grid[r, c + 1 : line_c]
        else:
            if c <= line_c:
                return False
            path = grid[r, line_c + 1 : c]
        if bool(np.any(path != BG)):
            return False
    return True


def _place_vertical_object(
    grid: np.ndarray,
    shape: np.ndarray,
    line_c: int,
    zone: tuple[str, int, int],
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    side, start_c, end_c = zone
    shape_h, shape_w = _shape_dims(shape)
    if end_c - start_c + 1 < shape_w or h - 2 < shape_h:
        return False

    row_choices = np.arange(1, h - shape_h)
    col_choices = np.arange(start_c, end_c - shape_w + 2)
    if row_choices.size == 0 or col_choices.size == 0:
        return False

    for _ in range(PLACE_ATTEMPTS):
        r0 = int(rng.choice(row_choices))
        c0 = int(rng.choice(col_choices))
        coords = shape + np.array([r0, c0], dtype=int)
        if not _can_place_vertical_object(grid, coords, line_c, side):
            continue
        color = int(grid[0, line_c])
        for rr, cc in coords:
            grid[int(rr), int(cc)] = color
        return True

    return False


def _gen_vertical_scene(height: int, width: int, allow_ties: bool, rng: np.random.Generator) -> np.ndarray | None:
    line_count_choices = [2, 3] if width >= 20 else [2]
    line_count = int(rng.choice(line_count_choices))
    positions = _sample_line_positions(width, line_count, rng)
    if positions is None:
        return None

    colors = [int(x) for x in rng.choice(COLORS, size=line_count, replace=False)]
    grid = np.zeros((height, width), dtype=int)
    for color, c in zip(colors, positions):
        grid[:, c] = color

    shapes = list(HORIZONTAL_SHAPES if allow_ties else tuple(_normalize_shape(shape) for shape in WIDE_SHAPES))
    object_plan = list(range(line_count))
    extra_count = int(rng.integers(0, line_count + 1))
    if extra_count:
        object_plan.extend(int(x) for x in rng.choice(np.arange(line_count), size=extra_count, replace=True))
    rng.shuffle(object_plan)

    for line_idx in range(line_count):
        if line_idx not in object_plan:
            object_plan.append(line_idx)

    for line_idx in object_plan:
        zones = _line_segments(width, positions, int(line_idx))
        if not zones:
            return None
        rng.shuffle(zones)

        placed = False
        shape_order = rng.permutation(len(shapes))
        for shape_idx in shape_order:
            shape = shapes[int(shape_idx)]
            shape_h, shape_w = _shape_dims(shape)
            zone_order = zones.copy()
            rng.shuffle(zone_order)
            for zone in zone_order:
                _side, start_c, end_c = zone
                if end_c - start_c + 1 < shape_w:
                    continue
                if height - 2 <= shape_h:
                    continue
                if _place_vertical_object(grid, shape, positions[int(line_idx)], zone, rng):
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None

    if not _is_valid_scene(grid):
        return None
    return grid


def _is_valid_scene(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    info = _scene_info(grid)
    rows_by_color = info["rows_by_color"]
    cols_by_color = info["cols_by_color"]
    comps = info["components"]
    if not comps:
        return False

    line_colors = set(rows_by_color) | set(cols_by_color)
    if len(line_colors) < 2:
        return False

    for color in line_colors:
        if color in rows_by_color and len(rows_by_color[color]) != 1:
            return False
        if color in cols_by_color and len(cols_by_color[color]) != 1:
            return False
        if color in rows_by_color and color in cols_by_color:
            return False

    horiz_count = 0
    vert_count = 0
    for comp in comps:
        color = int(comp["color"])
        area = int(len(comp["coords"]))
        if not (2 <= area <= 5):
            return False
        axis = _axis_from_bbox(comp)
        r0, c0, r1, c1 = comp["bbox"]

        if axis == "horizontal":
            horiz_count += 1
            cols = cols_by_color.get(color)
            if cols is None or len(cols) != 1:
                return False
            line_c = int(cols[0])
            if not (line_c < c0 or line_c > c1):
                return False
            step = -1 if line_c < c0 else 1
            for rr, cc in comp["coords"]:
                r = int(rr)
                c = int(cc)
                for path_c in range(c + step, line_c, step):
                    if int(grid[r, path_c]) not in (BG, color):
                        return False
        else:
            vert_count += 1
            rows = rows_by_color.get(color)
            if rows is None or len(rows) != 1:
                return False
            line_r = int(rows[0])
            if not (line_r < r0 or line_r > r1):
                return False
            step = -1 if line_r < r0 else 1
            for rr, cc in comp["coords"]:
                r = int(rr)
                c = int(cc)
                for path_r in range(r + step, line_r, step):
                    if int(grid[path_r, c]) not in (BG, color):
                        return False

    if horiz_count and vert_count:
        return False

    transformed = main(grid)
    if transformed.shape != grid.shape:
        return False
    if not bool(np.any(transformed != grid)):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        base_h = int(rng.integers(MIN_BASE_HEIGHT, MAX_BASE_HEIGHT + 1))
        base_w = int(rng.integers(MIN_BASE_WIDTH, MAX_BASE_WIDTH + 1))
        use_vertical_projection = bool(rng.integers(2))

        if use_vertical_projection:
            scene = _gen_vertical_scene(base_h, base_w, allow_ties=False, rng=rng)
            if scene is None:
                continue
            scene = scene.T
        else:
            scene = _gen_vertical_scene(base_h, base_w, allow_ties=True, rng=rng)
            if scene is None:
                continue

        if _is_valid_scene(scene):
            return scene

    raise RuntimeError("failed to generate a valid ARC projection scene")
