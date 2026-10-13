import random
from collections import Counter

import numpy as np


ColorId = int
Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


GRID_ROWS_RANGE = (12, 20)
GRID_COLS_RANGE = (12, 20)
HIDDEN_HEIGHT_RANGE = (3, 6)
HIDDEN_WIDTH_EXTRA = 4
HIDDEN_TOP_MARGIN = 2
MOVER_SOURCE_TOP = 2
OBSTACLE_HEIGHT_RANGE = (2, 4)
OBSTACLE_WIDTH_RANGE = (2, 5)
SIDE_OBSTACLE_RETRIES = 50
MAX_GENERATION_ATTEMPTS = 400
BRANCHES = ("hidden", "obstacle", "bottom")
_BRANCH_BAG: list[str] = []


def _get_bg(grid: np.ndarray) -> ColorId:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(set(map(int, grid.ravel())))
    if exclude is None:
        return colors
    return [color for color in colors if color not in exclude]


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    components: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _find_components(grid: np.ndarray, bg: int, exclude_colors: set[int]) -> list[dict]:
    components: list[dict] = []
    for color in _colors_present(grid, exclude={bg, *exclude_colors}):
        for coords in _find_color_components(grid, color):
            components.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return components


def _is_contiguous_run(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    return np.array_equal(values, np.arange(int(values[0]), int(values[-1]) + 1))


def _find_projection_rectangle(grid: np.ndarray, bg: int) -> tuple[int, BBox]:
    candidates: list[tuple[int, BBox]] = []
    for color in _colors_present(grid, exclude={bg}):
        mask = grid == int(color)
        row_cols = np.flatnonzero(mask[0])
        col_rows = np.flatnonzero(mask[:, 0])
        if row_cols.size == 0 or col_rows.size == 0:
            continue
        if not _is_contiguous_run(row_cols) or not _is_contiguous_run(col_rows):
            continue
        allowed = np.zeros_like(mask, dtype=bool)
        allowed[0, row_cols[0] : row_cols[-1] + 1] = True
        allowed[col_rows[0] : col_rows[-1] + 1, 0] = True
        if np.array_equal(mask, allowed):
            candidates.append(
                (
                    int(color),
                    (int(col_rows[0]), int(row_cols[0]), int(col_rows[-1]), int(row_cols[-1])),
                )
            )
    assert len(candidates) == 1, "expected exactly one projection-line color"
    return candidates[0]


def _is_solid_rectangle(component: dict, grid: np.ndarray) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if area != component["area"]:
        return False
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(component["color"])))


def _drop_distance(coords: np.ndarray, blocked: np.ndarray) -> int:
    height = blocked.shape[0]
    distance = 0
    while True:
        for r, c in coords:
            nr = int(r) + distance + 1
            nc = int(c)
            if nr >= height or blocked[nr, nc]:
                return distance
        distance += 1


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = patch != int(bg)
    window = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    window[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """Drop the movable object while treating the projected-but-undrawn rectangle as a solid blocker.

    Input structure
    - The dominant color is the background.
    - One color appears only as a horizontal segment on the first row and a vertical segment on the first column; together they encode an invisible rectangle's column span and row span.
    - The remaining foreground contains one movable 4-connected monochrome object and zero or more monochrome solid rectangular obstacles.

    Rule
    - Reconstruct the invisible rectangle from the edge segments, but keep it unpainted.
    - Let the movable object fall straight downward without changing shape or color.
    - The object stops at the last position before any next step would hit the grid bottom, a visible solid rectangle, or any cell of the invisible rectangle; all other painted cells stay unchanged.

    Key insight
    - The two edge segments are only projections, so the correct landing point is found by restoring their unseen rectangle as collision geometry rather than as visible output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    projection_color, hidden_bbox = _find_projection_rectangle(g, bg)
    components = _find_components(g, bg, exclude_colors={projection_color})
    assert components, "expected a movable object"

    if len(components) == 1:
        mover = components[0]
        obstacles: list[dict] = []
    else:
        mover_candidates = [component for component in components if not _is_solid_rectangle(component, g)]
        assert len(mover_candidates) == 1, "expected exactly one non-rectangle movable object"
        mover = mover_candidates[0]
        obstacles = [component for component in components if component is not mover]

    blocked = np.zeros_like(g, dtype=bool)
    hr0, hc0, hr1, hc1 = hidden_bbox
    blocked[hr0 : hr1 + 1, hc0 : hc1 + 1] = True
    for obstacle in obstacles:
        rows, cols = obstacle["coords"].T
        blocked[rows, cols] = True

    distance = _drop_distance(mover["coords"], blocked)
    out = g.copy()
    rows, cols = mover["coords"].T
    out[rows, cols] = bg
    out[rows + distance, cols] = int(mover["color"])
    return out


def _random_palette(bg: int, needed: int) -> list[int]:
    colors = [color for color in range(10) if color != int(bg)]
    random.shuffle(colors)
    return colors[:needed]


def _random_mover_patch(color: int, bg: int) -> np.ndarray:
    for _ in range(100):
        height = random.randint(2, 4)
        width = random.randint(2, 4)
        max_area = min(height * width - 1, 6)
        if max_area < 3:
            continue
        target_area = random.randint(3, max_area)
        cells = {(random.randrange(height), random.randrange(width))}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        r0, c0 = min(rows), min(cols)
        trimmed = {(r - r0, c - c0) for r, c in cells}
        trim_h = max(r for r, _c in trimmed) + 1
        trim_w = max(c for _r, c in trimmed) + 1
        if trim_h < 2 or trim_w < 2 or len(trimmed) == trim_h * trim_w:
            continue

        patch = np.full((trim_h, trim_w), int(bg), dtype=int)
        for r, c in trimmed:
            patch[r, c] = int(color)
        return patch
    raise RuntimeError("failed to generate a non-rectangular mover")


def _mask_bottom_row(patch: np.ndarray, bg: int) -> int:
    coords = np.argwhere(patch != int(bg))
    return int(coords[:, 0].max())


def _bbox_intersects(a: BBox, b: BBox, pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _render_grid(
    shape: tuple[int, int],
    bg: int,
    projection_color: int,
    hidden_bbox: BBox,
    mover_patch: np.ndarray,
    mover_top: int,
    mover_left: int,
    obstacles: list[tuple[int, int, int, int, int]],
) -> np.ndarray:
    height, width = shape
    grid = np.full((height, width), int(bg), dtype=int)
    hr0, hc0, hr1, hc1 = hidden_bbox
    grid[0, hc0 : hc1 + 1] = int(projection_color)
    grid[hr0 : hr1 + 1, 0] = int(projection_color)
    for top, left, obs_h, obs_w, color in obstacles:
        grid[top : top + obs_h, left : left + obs_w] = int(color)
    _paint_patch(grid, mover_patch, mover_top, mover_left, bg)
    return grid


def _expected_output(
    inp: np.ndarray,
    mover_patch: np.ndarray,
    mover_top: int,
    mover_left: int,
    dest_top: int,
    bg: int,
) -> np.ndarray:
    out = inp.copy()
    mask = mover_patch != int(bg)
    out[mover_top : mover_top + mover_patch.shape[0], mover_left : mover_left + mover_patch.shape[1]][mask] = int(bg)
    out[dest_top : dest_top + mover_patch.shape[0], mover_left : mover_left + mover_patch.shape[1]][mask] = int(
        mover_patch[mask][0]
    )
    return out


def _place_side_obstacles(
    shape: tuple[int, int],
    hidden_bbox: BBox,
    forbidden_boxes: list[BBox],
    protected_cols: set[int],
    colors: list[int],
    count_range: tuple[int, int],
) -> list[tuple[int, int, int, int, int]]:
    height, width = shape
    obstacles: list[tuple[int, int, int, int, int]] = []
    used_boxes = forbidden_boxes[:]
    max_count = random.randint(*count_range)
    color_index = 0

    for _ in range(max_count):
        placed = False
        for _retry in range(SIDE_OBSTACLE_RETRIES):
            obs_h = random.randint(*OBSTACLE_HEIGHT_RANGE)
            obs_w = random.randint(*OBSTACLE_WIDTH_RANGE)
            if height - obs_h - 1 <= 2 or width - obs_w - 1 <= 1:
                break
            top = random.randint(2, height - obs_h - 1)
            left = random.randint(1, width - obs_w - 1)
            box = (top, left, top + obs_h - 1, left + obs_w - 1)
            cols = set(range(left, left + obs_w))
            if cols & protected_cols:
                continue
            if _bbox_intersects(box, hidden_bbox, pad=0):
                continue
            if any(_bbox_intersects(box, used, pad=1) for used in used_boxes):
                continue
            color = int(colors[color_index % len(colors)])
            color_index += 1
            obstacles.append((top, left, obs_h, obs_w, color))
            used_boxes.append(box)
            placed = True
            break
        if not placed:
            break
    return obstacles


def _make_case(branch: str) -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(*GRID_ROWS_RANGE)
    width = random.randint(*GRID_COLS_RANGE)
    bg = random.randint(0, 9)
    palette = _random_palette(bg, 6)
    projection_color, mover_color = palette[0], palette[1]
    obstacle_colors = palette[2:]

    mover_patch = _random_mover_patch(mover_color, bg)
    patch_h, patch_w = mover_patch.shape
    mover_bottom = _mask_bottom_row(mover_patch, bg)

    hidden_h = random.randint(*HIDDEN_HEIGHT_RANGE)
    hidden_w_min = max(patch_w if branch != "bottom" else 3, 3)
    hidden_w_max = min(width - 3, patch_w + HIDDEN_WIDTH_EXTRA)
    if hidden_w_min > hidden_w_max:
        return None
    hidden_w = random.randint(hidden_w_min, hidden_w_max)

    hidden_row_min = mover_bottom + (7 if branch == "obstacle" else 4)
    hidden_row_max = height - hidden_h - 2
    if hidden_row_min > hidden_row_max:
        return None
    hidden_top = random.randint(hidden_row_min, hidden_row_max)
    hidden_left = random.randint(HIDDEN_TOP_MARGIN, width - hidden_w - 1)
    hidden_bbox = (hidden_top, hidden_left, hidden_top + hidden_h - 1, hidden_left + hidden_w - 1)

    mover_left_candidates: list[int] = []
    for left in range(1, width - patch_w):
        fully_inside_hidden = hidden_left <= left and left + patch_w - 1 <= hidden_left + hidden_w - 1
        fully_outside_hidden = left + patch_w - 1 < hidden_left or left > hidden_left + hidden_w - 1
        if branch == "bottom" and fully_outside_hidden:
            mover_left_candidates.append(left)
        if branch in {"hidden", "obstacle"} and fully_inside_hidden:
            mover_left_candidates.append(left)
    if not mover_left_candidates:
        return None
    mover_left = random.choice(mover_left_candidates)

    dest_top: int
    obstacles: list[tuple[int, int, int, int, int]] = []
    source_box: BBox
    forbidden_boxes: list[BBox]
    protected_cols = set(range(mover_left, mover_left + patch_w))

    if branch == "obstacle":
        obs_h = random.randint(2, 3)
        obstacle_top_min = mover_bottom + 4
        obstacle_top_max = hidden_top - obs_h - 2
        if obstacle_top_min > obstacle_top_max:
            return None
        obstacle_top = random.randint(obstacle_top_min, obstacle_top_max)
        dest_top = obstacle_top - 1 - mover_bottom
        if dest_top <= MOVER_SOURCE_TOP:
            return None
        mover_top = random.randint(MOVER_SOURCE_TOP, dest_top - 1)
        obstacles.append((obstacle_top, mover_left, obs_h, patch_w, int(obstacle_colors[0])))
        source_box = (mover_top, mover_left, mover_top + patch_h - 1, mover_left + patch_w - 1)
        forbidden_boxes = [source_box, (obstacle_top, mover_left, obstacle_top + obs_h - 1, mover_left + patch_w - 1)]
        obstacles.extend(
            _place_side_obstacles(
                (height, width),
                hidden_bbox,
                forbidden_boxes,
                protected_cols,
                obstacle_colors[1:] or obstacle_colors[:1],
                (0, 1),
            )
        )
    elif branch == "hidden":
        dest_top = hidden_top - 1 - mover_bottom
        if dest_top <= MOVER_SOURCE_TOP:
            return None
        mover_top = random.randint(MOVER_SOURCE_TOP, dest_top - 1)
        source_box = (mover_top, mover_left, mover_top + patch_h - 1, mover_left + patch_w - 1)
        forbidden_boxes = [source_box]
        obstacles.extend(
            _place_side_obstacles(
                (height, width),
                hidden_bbox,
                forbidden_boxes,
                protected_cols,
                obstacle_colors,
                (0, 2),
            )
        )
    else:
        dest_top = height - 1 - mover_bottom
        if dest_top <= MOVER_SOURCE_TOP:
            return None
        mover_top = random.randint(MOVER_SOURCE_TOP, dest_top - 1)
        source_box = (mover_top, mover_left, mover_top + patch_h - 1, mover_left + patch_w - 1)
        forbidden_boxes = [source_box]
        obstacles.extend(
            _place_side_obstacles(
                (height, width),
                hidden_bbox,
                forbidden_boxes,
                protected_cols,
                obstacle_colors,
                (0, 2),
            )
        )

    inp = _render_grid(
        (height, width),
        bg,
        projection_color,
        hidden_bbox,
        mover_patch,
        mover_top,
        mover_left,
        obstacles,
    )
    expected = _expected_output(inp, mover_patch, mover_top, mover_left, dest_top, bg)

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, actual):
        return None
    if not np.array_equal(actual, expected):
        return None
    if _get_bg(inp) != bg:
        return None
    return inp, expected


def _next_branch() -> str:
    global _BRANCH_BAG
    if not _BRANCH_BAG:
        _BRANCH_BAG = ["hidden", "bottom", "obstacle", "obstacle", "obstacle", "obstacle"]
        random.shuffle(_BRANCH_BAG)
    return _BRANCH_BAG.pop()


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        branch = _next_branch()
        generated = _make_case(branch)
        if generated is not None:
            return generated[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
