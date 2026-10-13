import numpy as np


BACKGROUND_COLOR = 8
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
RNG = np.random.default_rng()

# Generator controls.
GRID_HEIGHT_MIN = 18
GRID_HEIGHT_MAX = 30
GRID_WIDTH_MIN = 18
GRID_WIDTH_MAX = 30
OBJECT_COUNT_MIN = 3
OBJECT_COUNT_MAX = 6
SKELETON_LENGTH_MIN = 5
SKELETON_LENGTH_MAX = 10
TIE_LENGTHS = (5, 7, 9)
MAX_BRANCH_COUNT = 4
MAX_BRANCH_ARM = 3
MAX_BRANCH_TOTAL = 4
PLACEMENT_TRIES = 200
GENERATION_TRIES = 400
KIND_OPTIONS = ("H", "V", "T")
FOREGROUND_COLORS = tuple(color for color in range(10) if color != BACKGROUND_COLOR)


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _normalize_coords(coords) -> np.ndarray:
    arr = np.asarray(list(coords), dtype=int)
    arr = np.unique(arr, axis=0)
    arr = arr - arr.min(axis=0, keepdims=True)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _flip_coords(coords: np.ndarray, flip_rows: bool, flip_cols: bool) -> np.ndarray:
    arr = np.array(coords, dtype=int, copy=True)
    if flip_rows:
        arr[:, 0] = arr[:, 0].max() - arr[:, 0]
    if flip_cols:
        arr[:, 1] = arr[:, 1].max() - arr[:, 1]
    return _normalize_coords(map(tuple, arr))


def _transpose_coords(coords: np.ndarray) -> np.ndarray:
    arr = np.array(coords[:, [1, 0]], dtype=int, copy=True)
    return _normalize_coords(map(tuple, arr))


def _component_from_coords(coords: np.ndarray, color: int) -> dict:
    return {"color": int(color), "coords": np.asarray(coords, dtype=int)}


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    height, width = map(int, grid.shape)
    seen = np.zeros((height, width), dtype=bool)
    components = []

    for row in range(height):
        for col in range(width):
            if seen[row, col] or int(grid[row, col]) == bg:
                continue

            color = int(grid[row, col])
            stack = [(row, col)]
            seen[row, col] = True
            cells = []

            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.asarray(cells, dtype=int)
            components.append(_component_from_coords(coords, color))

    return components


def _component_mask(component: dict) -> tuple[np.ndarray, int, int]:
    coords = component["coords"]
    rows = coords[:, 0]
    cols = coords[:, 1]
    row0 = int(rows.min())
    col0 = int(cols.min())
    row1 = int(rows.max())
    col1 = int(cols.max())
    mask = np.zeros((row1 - row0 + 1, col1 - col0 + 1), dtype=bool)
    mask[rows - row0, cols - col0] = True
    return mask, row0, col0


def _best_horizontal_segment(component: dict) -> dict:
    mask, row0, col0 = _component_mask(component)
    color = int(component["color"])
    best = None

    for local_row in range(mask.shape[0]):
        local_col = 0
        while local_col < mask.shape[1]:
            if not mask[local_row, local_col]:
                local_col += 1
                continue

            start = local_col
            while local_col < mask.shape[1] and mask[local_row, local_col]:
                local_col += 1

            segment = {
                "axis": "H",
                "color": color,
                "length": int(local_col - start),
                "row": int(row0 + local_row),
                "col0": int(col0 + start),
                "col1": int(col0 + local_col - 1),
            }
            if best is None:
                best = segment
            elif segment["length"] > best["length"]:
                best = segment
            elif segment["length"] == best["length"] and (segment["row"], segment["col0"]) < (best["row"], best["col0"]):
                best = segment

    assert best is not None
    return best


def _best_vertical_segment(component: dict) -> dict:
    mask, row0, col0 = _component_mask(component)
    color = int(component["color"])
    best = None

    for local_col in range(mask.shape[1]):
        local_row = 0
        while local_row < mask.shape[0]:
            if not mask[local_row, local_col]:
                local_row += 1
                continue

            start = local_row
            while local_row < mask.shape[0] and mask[local_row, local_col]:
                local_row += 1

            segment = {
                "axis": "V",
                "color": color,
                "length": int(local_row - start),
                "col": int(col0 + local_col),
                "row0": int(row0 + start),
                "row1": int(row0 + local_row - 1),
            }
            if best is None:
                best = segment
            elif segment["length"] > best["length"]:
                best = segment
            elif segment["length"] == best["length"] and (segment["col"], segment["row0"]) < (best["col"], best["row0"]):
                best = segment

    assert best is not None
    return best


def _extract_skeleton(component: dict) -> tuple[dict, dict, dict]:
    horizontal = _best_horizontal_segment(component)
    vertical = _best_vertical_segment(component)
    chosen = horizontal if horizontal["length"] >= vertical["length"] else vertical
    return horizontal, vertical, chosen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each object's longest internal axis-aligned segment, group by orientation,
    and redraw only those skeletons.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    components = _find_components(g, bg)
    assert components, "expected at least one foreground object"

    horizontal_segments = []
    vertical_segments = []
    for component in components:
        _, _, skeleton = _extract_skeleton(component)
        if skeleton["axis"] == "H":
            horizontal_segments.append(skeleton)
        else:
            vertical_segments.append(skeleton)

    horizontal_segments.sort(key=lambda seg: (seg["row"], seg["col0"]))
    vertical_segments.sort(key=lambda seg: (seg["col"], seg["row0"]))

    h_height = 0 if not horizontal_segments else 2 * len(horizontal_segments) - 1
    h_width = max((seg["length"] for seg in horizontal_segments), default=0)
    v_height = max((seg["length"] for seg in vertical_segments), default=0)
    v_width = 0 if not vertical_segments else 2 * len(vertical_segments) - 1
    separator = 1 if horizontal_segments and vertical_segments else 0

    out_height = h_height + separator + v_height
    out_width = max(h_width, v_width)
    assert out_height > 0 and out_width > 0

    out = np.full((out_height, out_width), bg, dtype=int)

    for index, segment in enumerate(horizontal_segments):
        out[2 * index, : segment["length"]] = int(segment["color"])

    row_offset = h_height + separator
    for index, segment in enumerate(vertical_segments):
        out[row_offset : row_offset + segment["length"], 2 * index] = int(segment["color"])

    return out


def _shape_dims(coords: np.ndarray) -> tuple[int, int]:
    return int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1


def _shape_stats(coords: np.ndarray) -> tuple[int, int, str]:
    component = _component_from_coords(coords, color=1)
    horizontal, vertical, chosen = _extract_skeleton(component)
    return int(horizontal["length"]), int(vertical["length"]), str(chosen["axis"])


def _make_horizontal_shape(rng: np.random.Generator) -> np.ndarray:
    length = int(rng.integers(SKELETON_LENGTH_MIN, SKELETON_LENGTH_MAX + 1))
    coords = {(0, col) for col in range(length)}

    branch_limit = min(MAX_BRANCH_COUNT, length - 2)
    branch_count = int(rng.integers(1, branch_limit + 1))
    positions = [int(value) for value in rng.choice(np.arange(length), size=branch_count, replace=False)]

    for pos in positions:
        while True:
            up = int(rng.integers(0, MAX_BRANCH_ARM + 1))
            down = int(rng.integers(0, MAX_BRANCH_ARM + 1))
            if up + down == 0:
                continue
            if up + down + 1 <= min(MAX_BRANCH_TOTAL, length - 1):
                break
        for delta in range(1, up + 1):
            coords.add((-delta, pos))
        for delta in range(1, down + 1):
            coords.add((delta, pos))

    shape = _normalize_coords(coords)
    return _flip_coords(shape, flip_rows=bool(rng.integers(0, 2)), flip_cols=bool(rng.integers(0, 2)))


def _make_vertical_shape(rng: np.random.Generator) -> np.ndarray:
    return _transpose_coords(_make_horizontal_shape(rng))


def _make_tie_shape(rng: np.random.Generator) -> np.ndarray:
    length = int(rng.choice(TIE_LENGTHS))
    half = length // 2
    center = half
    coords = {(0, col) for col in range(length)}
    coords.update((row, center) for row in range(-half, half + 1))

    nub_candidates = []
    for col in range(length):
        if col == center:
            continue
        nub_candidates.append((-1, col))
        nub_candidates.append((1, col))
    for row in range(-half, half + 1):
        if row == 0:
            continue
        nub_candidates.append((row, center - 1))
        nub_candidates.append((row, center + 1))

    nub_count = int(rng.integers(1, min(4, len(nub_candidates)) + 1))
    for idx in rng.choice(np.arange(len(nub_candidates)), size=nub_count, replace=False):
        coords.add(nub_candidates[int(idx)])

    shape = _normalize_coords(coords)
    return _flip_coords(shape, flip_rows=bool(rng.integers(0, 2)), flip_cols=bool(rng.integers(0, 2)))


def _build_shape(kind: str, rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(60):
        if kind == "H":
            shape = _make_horizontal_shape(rng)
        elif kind == "V":
            shape = _make_vertical_shape(rng)
        else:
            assert kind == "T"
            shape = _make_tie_shape(rng)

        h_len, v_len, axis = _shape_stats(shape)
        if kind == "H" and axis == "H" and h_len > v_len:
            return shape
        if kind == "V" and axis == "V" and v_len > h_len:
            return shape
        if kind == "T" and axis == "H" and h_len == v_len:
            return shape
    return None


def _can_place(cells: np.ndarray, occupied: np.ndarray) -> bool:
    height, width = map(int, occupied.shape)
    for row, col in cells:
        r = int(row)
        c = int(col)
        if r <= 0 or c <= 0 or r >= height - 1 or c >= width - 1:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if occupied[r + dr, c + dc]:
                    return False
    return True


def _gen_attempt() -> np.ndarray | None:
    object_count = int(RNG.integers(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX + 1))

    for _ in range(40):
        kinds = RNG.choice(KIND_OPTIONS, size=object_count, replace=True).tolist()
        if any(kind == "V" for kind in kinds) and any(kind in ("H", "T") for kind in kinds):
            break
    else:
        return None

    colors = [int(value) for value in RNG.choice(FOREGROUND_COLORS, size=object_count, replace=False)]
    specs = []
    for kind, color in zip(kinds, colors):
        shape = _build_shape(kind, RNG)
        if shape is None:
            return None
        height, width = _shape_dims(shape)
        specs.append(
            {
                "kind": kind,
                "color": color,
                "coords": shape,
                "height": height,
                "width": width,
                "area": int(shape.shape[0]),
            }
        )

    max_shape_height = max(spec["height"] for spec in specs)
    max_shape_width = max(spec["width"] for spec in specs)
    grid_height_min = max(GRID_HEIGHT_MIN, max_shape_height + object_count + 4)
    grid_width_min = max(GRID_WIDTH_MIN, max_shape_width + object_count + 4)
    if grid_height_min > GRID_HEIGHT_MAX or grid_width_min > GRID_WIDTH_MAX:
        return None

    grid_height = int(RNG.integers(grid_height_min, GRID_HEIGHT_MAX + 1))
    grid_width = int(RNG.integers(grid_width_min, GRID_WIDTH_MAX + 1))
    grid = np.full((grid_height, grid_width), BACKGROUND_COLOR, dtype=int)
    occupied = np.zeros((grid_height, grid_width), dtype=bool)

    for spec in sorted(specs, key=lambda item: (-item["area"], -item["height"], -item["width"])):
        max_top_exclusive = grid_height - spec["height"]
        max_left_exclusive = grid_width - spec["width"]
        if max_top_exclusive <= 1 or max_left_exclusive <= 1:
            return None

        placed = False
        for _ in range(PLACEMENT_TRIES):
            top = int(RNG.integers(1, max_top_exclusive))
            left = int(RNG.integers(1, max_left_exclusive))
            cells = spec["coords"] + np.array([top, left], dtype=int)
            if not _can_place(cells, occupied):
                continue
            for row, col in cells:
                r = int(row)
                c = int(col)
                grid[r, c] = int(spec["color"])
                occupied[r, c] = True
            placed = True
            break

        if not placed:
            return None

    if get_bg(grid) != BACKGROUND_COLOR:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.size == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
