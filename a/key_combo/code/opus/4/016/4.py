import random

import numpy as np


BG = 0
GUIDE = 6
SQUARE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
BASE_SIZE = 9
GEN_ATTEMPTS = 500

CORNER_ANCHORS = {
    "tl": (1, 1),
    "tr": (1, 5),
    "bl": (5, 1),
    "br": (5, 5),
}

INWARD_SIDES = {
    "tl": ("right", "bottom"),
    "tr": ("left", "bottom"),
    "bl": ("right", "top"),
    "br": ("left", "top"),
}

SIDE_NEIGHBOR = {
    ("tl", "right"): "tr",
    ("tl", "bottom"): "bl",
    ("tr", "left"): "tl",
    ("tr", "bottom"): "br",
    ("bl", "right"): "br",
    ("bl", "top"): "tl",
    ("br", "left"): "bl",
    ("br", "top"): "tr",
}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _run_start_indices(matrix: np.ndarray) -> list[int]:
    starts = [0]
    for idx in range(1, matrix.shape[0]):
        if not np.array_equal(matrix[idx], matrix[idx - 1]):
            starts.append(idx)
    return starts


def _compress_grid(grid: np.ndarray) -> np.ndarray:
    row_starts = _run_start_indices(grid)
    col_starts = _run_start_indices(grid.T)
    return grid[np.ix_(row_starts, col_starts)].copy()


def _find_hollow_squares(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    out = []
    border_r = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    border_c = np.array([0, 1, 2, 0, 2, 0, 1, 2])
    for r in range(grid.shape[0] - 2):
        for c in range(grid.shape[1] - 2):
            patch = grid[r : r + 3, c : c + 3]
            color = int(patch[0, 0])
            if color in (BG, GUIDE):
                continue
            if int(patch[1, 1]) != BG:
                continue
            if np.all(patch[border_r, border_c] == color):
                out.append((r, c, r + 2, c + 2))
    return out


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Collapse duplicate row/column runs first, then crop the uniquely enclosed hollow square.

    Input structure:
    - The background is black.
    - Several 3x3 monochrome hollow squares are hidden inside a row/column-stretched grid.
    - Sparse magenta guide pixels sit only on background cells, and some of them disappear when duplicate rows or columns are removed.

    Rule:
    - Keep only the first row from every maximal run of identical adjacent rows, and only the first column from every maximal run of identical adjacent columns.
    - In the compressed grid, compute the bounding box of the remaining magenta guide pixels.
    - Find every 3x3 hollow square whose 8 border cells share one non-black, non-magenta color and whose center is black.
    - Select the unique hollow square whose 3x3 bbox is fully contained in the compressed guide bbox, crop it tightly, and replace any magenta pixel inside that crop with black.

    Key insight:
    - The guide mask only becomes meaningful after deduplication: before compression its bbox is stretched by disappearing guide pixels and can frame multiple candidates.
    """
    g = np.asarray(grid, dtype=int)
    compressed = _compress_grid(g)

    guide_mask = compressed == GUIDE
    guide_box = _bbox(guide_mask)
    assert guide_box is not None, "expected surviving guide pixels after compression"

    candidates = _find_hollow_squares(compressed)
    selected = [box for box in candidates if _contains(guide_box, box)]
    assert len(selected) == 1, "expected exactly one hollow square inside the compressed guide bbox"

    out = _crop_bbox(compressed, selected[0])
    out[out == GUIDE] = BG
    return out


def _paint_hollow_square(grid: np.ndarray, top: int, left: int, color: int) -> None:
    grid[top : top + 3, left : left + 3] = np.array(
        [[color, color, color], [color, BG, color], [color, color, color]],
        dtype=int,
    )


def _anchor_box(corner: str) -> tuple[int, int, int, int]:
    r, c = CORNER_ANCHORS[corner]
    return (r, c, r + 2, c + 2)


def _guide_box_for_target(corner: str) -> tuple[int, int, int, int]:
    r, c = CORNER_ANCHORS[corner]
    return (r - 1, c - 1, r + 3, c + 3)


def _expand_grid(grid: np.ndarray, row_runs: list[int], col_runs: list[int]) -> np.ndarray:
    return np.repeat(np.repeat(grid, row_runs, axis=0), col_runs, axis=1)


def _run_starts(runs: list[int]) -> list[int]:
    starts = []
    acc = 0
    for run in runs:
        starts.append(acc)
        acc += int(run)
    return starts


def _map_box_to_expanded(
    box: tuple[int, int, int, int],
    row_runs: list[int],
    col_runs: list[int],
) -> tuple[int, int, int, int]:
    row_starts = _run_starts(row_runs)
    col_starts = _run_starts(col_runs)
    r0, c0, r1, c1 = box
    return (
        row_starts[r0],
        col_starts[c0],
        row_starts[r1] + row_runs[r1] - 1,
        col_starts[c1] + col_runs[c1] - 1,
    )


def _pick_runs(length: int) -> list[int]:
    runs = [random.randint(1, 3) for _ in range(length)]
    for idx in (1, 7):
        runs[idx] = random.randint(2, 3)
    return runs


def _core_guide_cells(guide_box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = guide_box
    perimeter = [
        (r0, c0),
        (r0, c1),
        (r1, c0),
        (r1, c1),
        (r0, c0 + 2),
        (r1, c0 + 2),
        (r0 + 2, c0),
        (r0 + 2, c1),
    ]
    count = random.randint(4, len(perimeter))
    chosen = random.sample(perimeter, count)
    return chosen


def _decoy_base_cell(
    target_corner: str,
    side: str,
    guide_box: tuple[int, int, int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = guide_box
    neighbor = SIDE_NEIGHBOR[(target_corner, side)]
    nr, nc = CORNER_ANCHORS[neighbor]
    if side == "right":
        return random.choice([(r0, nc + 2), (r1, nc + 2)])
    if side == "left":
        return random.choice([(r0, nc), (r1, nc)])
    if side == "bottom":
        return random.choice([(nr + 2, c0), (nr + 2, c1)])
    return random.choice([(nr, c0), (nr, c1)])


def _decoy_points_expanded(
    guide_box: tuple[int, int, int, int],
    target_corner: str,
    chosen_sides: tuple[str, ...],
    row_runs: list[int],
    col_runs: list[int],
) -> list[tuple[int, int]]:
    row_starts = _run_starts(row_runs)
    col_starts = _run_starts(col_runs)

    decoy_points = []
    for side in chosen_sides:
        br, bc = _decoy_base_cell(target_corner, side, guide_box)
        rr = row_starts[br]
        cc = col_starts[bc]
        if side in ("left", "right"):
            cc += random.randint(1, col_runs[bc] - 1)
        else:
            rr += random.randint(1, row_runs[br] - 1)
        decoy_points.append((rr, cc))
    return decoy_points


def _adjacent_rows_cols_are_unique(grid: np.ndarray) -> bool:
    for idx in range(1, grid.shape[0]):
        if np.array_equal(grid[idx], grid[idx - 1]):
            return False
    for idx in range(1, grid.shape[1]):
        if np.array_equal(grid[:, idx], grid[:, idx - 1]):
            return False
    return True


def _build_base_scene(
    target_corner: str,
    chosen_sides: tuple[str, ...],
    present_corners: list[str],
) -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]:
    base = np.zeros((BASE_SIZE, BASE_SIZE), dtype=int)
    colors = random.sample(SQUARE_COLORS, len(present_corners))
    random.shuffle(colors)

    boxes = {}
    for corner, color in zip(present_corners, colors):
        r, c = CORNER_ANCHORS[corner]
        _paint_hollow_square(base, r, c, color)
        boxes[corner] = (r, c, r + 2, c + 2)

    assert target_corner in boxes
    return base, boxes


def _present_corners(target_corner: str, chosen_sides: tuple[str, ...]) -> list[str]:
    required = {target_corner}
    for side in chosen_sides:
        required.add(SIDE_NEIGHBOR[(target_corner, side)])

    all_corners = list(CORNER_ANCHORS)
    optional = [corner for corner in all_corners if corner not in required]
    target_count = random.choice([3, 4])
    if target_count < len(required):
        target_count = len(required)

    present = list(required)
    random.shuffle(optional)
    while len(present) < target_count:
        present.append(optional.pop())
    random.shuffle(present)
    return present


def _expected_output(base_grid: np.ndarray, target_corner: str) -> np.ndarray:
    out = _crop_bbox(base_grid, _anchor_box(target_corner))
    out[out == GUIDE] = BG
    return out


def _validate_instance(
    grid: np.ndarray,
    base_grid: np.ndarray,
    target_corner: str,
    boxes: dict[str, tuple[int, int, int, int]],
    row_runs: list[int],
    col_runs: list[int],
) -> bool:
    compressed = _compress_grid(grid)
    if compressed.shape != (BASE_SIZE, BASE_SIZE):
        return False
    if not _adjacent_rows_cols_are_unique(compressed):
        return False

    guide_box = _bbox(compressed == GUIDE)
    if guide_box != _guide_box_for_target(target_corner):
        return False

    candidates = _find_hollow_squares(compressed)
    if len(candidates) != len(boxes):
        return False

    selected = [box for box in candidates if _contains(guide_box, box)]
    if selected != [boxes[target_corner]]:
        return False

    if not np.array_equal(compressed, base_grid):
        return False

    input_guide_box = _bbox(grid == GUIDE)
    if input_guide_box is None:
        return False
    input_candidates = _find_hollow_squares(grid)
    raw_selected = [box for box in input_candidates if _contains(input_guide_box, box)]
    if len(raw_selected) == 1:
        return False

    if not np.array_equal(main(grid), _expected_output(base_grid, target_corner)):
        return False

    return True


def _generate_one() -> np.ndarray | None:
    target_corner = random.choice(list(CORNER_ANCHORS))
    side_count = random.choice([1, 2])
    chosen_sides = tuple(random.sample(INWARD_SIDES[target_corner], side_count))
    present = _present_corners(target_corner, chosen_sides)

    base_grid, boxes = _build_base_scene(target_corner, chosen_sides, present)
    guide_box = _guide_box_for_target(target_corner)

    core_cells = _core_guide_cells(guide_box)
    for r, c in core_cells:
        if base_grid[r, c] != BG:
            return None
        base_grid[r, c] = GUIDE

    if not _adjacent_rows_cols_are_unique(base_grid):
        return None

    row_runs = _pick_runs(BASE_SIZE)
    col_runs = _pick_runs(BASE_SIZE)

    target_r0, target_c0, target_r1, target_c1 = boxes[target_corner]
    if max(row_runs[target_r0 : target_r1 + 1]) == 1:
        row_runs[random.randint(target_r0, target_r1)] = random.randint(2, 3)
    if max(col_runs[target_c0 : target_c1 + 1]) == 1:
        col_runs[random.randint(target_c0, target_c1)] = random.randint(2, 3)

    expanded = _expand_grid(base_grid, row_runs, col_runs)

    if not _validate_instance(expanded, base_grid, target_corner, boxes, row_runs, col_runs):
        return None
    return expanded


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
