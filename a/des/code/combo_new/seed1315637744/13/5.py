import numpy as np


BACKGROUND = 0
GROUND_COLOR = 5
RECEIVER_COLOR = 4
OVERLAP_COLOR = 7
TOWER_COLORS = (1, 2, 3, 6, 8, 9)

MIN_HEIGHT = 1
MAX_HEIGHT = 5
MIN_GRID_HEIGHT = 12
MAX_GRID_HEIGHT = 22
MIN_GRID_WIDTH = 12
MAX_GRID_WIDTH = 28
MIN_TOWER_GAP = 2
GENERATION_RETRIES = 400

RECEIVER_NEIGHBORS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_ground_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    rows = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != bg and np.all(grid[r] == grid[r, 0])
    ]
    assert rows, "expected a full-width ground row"
    ground_row = max(rows)
    return ground_row, int(grid[ground_row, 0])


def _extract_towers(
    grid: np.ndarray, ground_row: int, ground_color: int, bg: int
) -> tuple[list[dict[str, int]], np.ndarray]:
    towers: list[dict[str, int]] = []
    tower_mask = np.zeros(grid.shape, dtype=bool)
    if ground_row == 0:
        return towers, tower_mask

    for c in range(grid.shape[1]):
        color = int(grid[ground_row - 1, c])
        if color == bg or color == ground_color:
            continue

        h = 0
        r = ground_row - 1
        while r >= 0 and int(grid[r, c]) == color:
            h += 1
            r -= 1

        top_r = ground_row - h
        tower_mask[top_r:ground_row, c] = True
        towers.append({"color": color, "col": c, "top_r": top_r, "height": h})

    assert towers, "expected at least one signal tower"
    assert len({tower["col"] for tower in towers}) == len(towers)
    assert len({tower["color"] for tower in towers}) == len(towers)
    return towers, tower_mask


def _receiver_positions(
    grid: np.ndarray, tower_mask: np.ndarray, ground_row: int, ground_color: int, bg: int
) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    receiver_colors = set()
    for r in range(ground_row):
        for c in range(grid.shape[1]):
            if tower_mask[r, c]:
                continue
            color = int(grid[r, c])
            if color == bg or color == ground_color:
                continue
            coords.append((r, c))
            receiver_colors.add(color)

    assert coords, "expected at least one receiver"
    assert len(receiver_colors) == 1, "receivers must share one color"
    return coords


def _covering_towers(r: int, c: int, towers: list[dict[str, int]]) -> list[dict[str, int]]:
    covering = []
    for tower in towers:
        reach = tower["height"]
        if abs(r - tower["top_r"]) <= reach and abs(c - tower["col"]) <= reach:
            covering.append(tower)
    return covering


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    ground_row, ground_color = _find_ground_row(g, bg)
    towers, tower_mask = _extract_towers(g, ground_row, ground_color, bg)
    receivers = _receiver_positions(g, tower_mask, ground_row, ground_color, bg)

    out = g.copy()
    for r, c in receivers:
        covering = _covering_towers(r, c, towers)
        if len(covering) == 1:
            out[r, c] = covering[0]["color"]
        elif len(covering) >= 2:
            out[r, c] = OVERLAP_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor receivers according to the square broadcast range of each vertical signal tower.

    Input structure:
    - The grid has a dominant background color and one full-width horizontal ground line.
    - Several one-cell-wide vertical towers of distinct colors rise upward from the row just above the ground line.
    - Towers can have different heights, and scattered single-pixel receivers share one common color in the upper area.

    Rule:
    - For each tower, measure its visible height and center a square broadcast region on the tower's top pixel.
    - The square side length is height * 2 + 1, so the tower reaches that many cells in every direction within the grid.
    - A receiver covered by exactly one tower changes to that tower's color; a receiver covered by two or more towers changes to orange; uncovered receivers stay unchanged.
    - The towers, ground line, and background remain fixed.

    Key insight:
    - The tower height is not decorative: it directly determines the broadcast square size, and receiver recoloring depends on whether zero, one, or multiple tower ranges include that receiver.
    """
    return _transform(grid)


def _sample_towers(
    rng: np.random.Generator, height: int, width: int, ground_row: int
) -> list[dict[str, int]] | None:
    max_towers = min(4, len(TOWER_COLORS), (width - 1) // 2)
    if max_towers < 2:
        return None

    tower_count = int(rng.integers(2, max_towers + 1))
    heights = rng.integers(MIN_HEIGHT, min(MAX_HEIGHT, ground_row) + 1, size=tower_count)
    colors = rng.choice(np.array(TOWER_COLORS), size=tower_count, replace=False)

    c1 = int(rng.integers(1, width - 1))
    overlap_candidates = [
        c
        for c in range(1, width - 1)
        if abs(c - c1) >= MIN_TOWER_GAP and abs(c - c1) <= int(heights[0] + heights[1])
    ]
    if not overlap_candidates:
        return None

    cols = [c1, int(rng.choice(overlap_candidates))]
    while len(cols) < tower_count:
        candidates = [
            c
            for c in range(1, width - 1)
            if all(abs(c - existing) >= MIN_TOWER_GAP for existing in cols)
        ]
        if not candidates:
            return None
        cols.append(int(rng.choice(candidates)))

    order = np.argsort(cols)
    towers = []
    for idx in order:
        h = int(heights[idx])
        col = int(cols[idx])
        towers.append(
            {
                "color": int(colors[idx]),
                "col": col,
                "top_r": int(ground_row - h),
                "height": h,
            }
        )
    return towers


def _coverage_classes(
    shape: tuple[int, int], ground_row: int, towers: list[dict[str, int]], tower_mask: np.ndarray
) -> dict[int, list[tuple[int, int]]]:
    height, width = shape
    classes = {0: [], 1: [], 2: []}
    for r in range(max(ground_row - 1, 0)):
        for c in range(width):
            if tower_mask[r, c]:
                continue
            covering = len(_covering_towers(r, c, towers))
            bucket = 2 if covering >= 2 else covering
            classes[bucket].append((r, c))
    return classes


def _reserve_neighbors(
    pos: tuple[int, int], blocked: set[tuple[int, int]], height: int, width: int
) -> None:
    r, c = pos
    for dr, dc in RECEIVER_NEIGHBORS:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height and 0 <= cc < width:
            blocked.add((rr, cc))


def _pick_spaced_position(
    rng: np.random.Generator,
    candidates: list[tuple[int, int]],
    blocked: set[tuple[int, int]],
    height: int,
    width: int,
) -> tuple[int, int] | None:
    if not candidates:
        return None
    order = np.array(candidates, dtype=int)
    rng.shuffle(order)
    for r, c in order:
        pos = (int(r), int(c))
        if pos in blocked:
            continue
        _reserve_neighbors(pos, blocked, height, width)
        return pos
    return None


def _place_receivers(
    rng: np.random.Generator, shape: tuple[int, int], classes: dict[int, list[tuple[int, int]]]
) -> list[tuple[int, int]] | None:
    height, width = shape
    blocked: set[tuple[int, int]] = set()
    chosen: list[tuple[int, int]] = []

    mandatory_order = sorted((0, 1, 2), key=lambda bucket: len(classes[bucket]))
    for bucket in mandatory_order:
        pos = _pick_spaced_position(rng, classes[bucket], blocked, height, width)
        if pos is None:
            return None
        chosen.append(pos)

    extra_targets = (
        [0] * int(rng.integers(0, 3))
        + [1] * int(rng.integers(1, 4))
        + [2] * int(rng.integers(0, 3))
    )
    rng.shuffle(extra_targets)
    for bucket in extra_targets:
        pos = _pick_spaced_position(rng, classes[bucket], blocked, height, width)
        if pos is not None:
            chosen.append(pos)

    return chosen


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(MIN_GRID_HEIGHT, MAX_GRID_HEIGHT + 1))
    width = int(rng.integers(MIN_GRID_WIDTH, MAX_GRID_WIDTH + 1))
    ground_row = int(rng.integers(MAX_HEIGHT + 2, height - 1))

    towers = _sample_towers(rng, height, width, ground_row)
    if towers is None:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[ground_row, :] = GROUND_COLOR

    tower_mask = np.zeros_like(grid, dtype=bool)
    for tower in towers:
        col = tower["col"]
        top_r = tower["top_r"]
        grid[top_r:ground_row, col] = tower["color"]
        tower_mask[top_r:ground_row, col] = True

    classes = _coverage_classes(grid.shape, ground_row, towers, tower_mask)
    if any(len(classes[bucket]) == 0 for bucket in (0, 1, 2)):
        return None

    receivers = _place_receivers(rng, grid.shape, classes)
    if receivers is None or len(receivers) < 4:
        return None

    for r, c in receivers:
        grid[r, c] = RECEIVER_COLOR

    detected = _receiver_positions(grid, tower_mask, ground_row, GROUND_COLOR, BACKGROUND)
    if len(detected) != len(receivers):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower puzzle")
