import numpy as np

from arc_utils import find_components, get_bg
import random


def _square_objects(g: np.ndarray, bg: int) -> list[dict]:
    objects = []
    for obj in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        r0, c0, r1, c1 = obj["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h == w and h >= 2 and int(obj["area"]) == h * w:
            objects.append({"side": int(h), "color": int(obj["color"]), "bbox": obj["bbox"]})
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """Render the filled-square object counts as compact nested stacks ordered by count rank.

    Input structure
    A large noisy grid has a dominant background, scattered non-square noise, and
    filled monochrome square objects. Each square side length has exactly one
    object color, and side lengths form consecutive sizes.

    Rule
    Ignore non-square components. If the lowest square in the scene is of the
    smallest side length, treat that one smallest square as a distractor and do
    not count it. For count rank k from left to right, create one block from all
    square sizes with at least k remaining objects; draw larger squares first and
    smaller squares over their upper-left corner, then append one background
    separator column.

    Key insight
    The output is a count histogram of square side lengths, but each histogram
    slot is rendered as nested upper-left square masks using the side colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    objects = _square_objects(g, bg)
    assert objects, "expected filled square objects"

    sides = sorted({obj["side"] for obj in objects})
    assert sides == list(range(sides[0], sides[-1] + 1)), "expected consecutive square side lengths"

    by_side = {side: [obj for obj in objects if obj["side"] == side] for side in sides}
    colors = {}
    for side, side_objects in by_side.items():
        side_colors = {obj["color"] for obj in side_objects}
        assert len(side_colors) == 1, "each side length should have one color"
        colors[side] = side_colors.pop()

    smallest = sides[0]
    lowest_row = max(obj["bbox"][2] for obj in objects)
    lowest_smallest = [obj for obj in by_side[smallest] if obj["bbox"][2] == lowest_row]
    if lowest_smallest:
        assert len(lowest_smallest) == 1, "expected at most one lowest smallest distractor"
        drop_bbox = lowest_smallest[0]["bbox"]
        by_side[smallest] = [obj for obj in by_side[smallest] if obj["bbox"] != drop_bbox]

    counts = {side: len(side_objects) for side, side_objects in by_side.items()}
    assert all(count > 0 for count in counts.values()), "each square size should remain represented"

    out_h = sides[-1] + 1
    groups = []
    for rank in range(max(counts.values())):
        present_sides = [side for side in sides if counts[side] > rank]
        assert present_sides, "rank should contain at least one square size"

        group_w = max(present_sides)
        group = np.full((out_h, group_w), bg, dtype=int)
        for side in sorted(present_sides, reverse=True):
            group[:side, :side] = colors[side]
        groups.append(group)

    out_w = sum(group.shape[1] + 1 for group in groups)
    out = np.full((out_h, out_w), bg, dtype=int)
    c = 0
    for group in groups:
        w = group.shape[1]
        out[:, c : c + w] = group
        c += w + 1

    return out

# --------------- generator (att1) ---------------

SIDES = [2, 3, 4]
ALL_COLORS = list(range(10))

# Canvas and object bounds.
GRID_SIZE_RANGE = (25, 28)
COUNT_RANGE = (2, 6)
SQUARE_GAP = 1

# Noise and distractor controls.
DISTRACTOR_PROB = 0.5
NOISE_COUNT_RANGE = (40, 70)

# Retry budget.
MAX_GENERATION_ATTEMPTS = 300


def _choose_palette() -> tuple[int, dict[int, int], int]:
    """Choose one background, three square colors, and one noise color."""
    bg = random.choice(ALL_COLORS)
    non_bg = [c for c in ALL_COLORS if c != bg]
    random.shuffle(non_bg)
    square_colors = {side: non_bg[i] for i, side in enumerate(SIDES)}
    return bg, square_colors, non_bg[len(SIDES)]


def _make_square_queue(
    counts: dict[int, int], square_colors: dict[int, int], has_distractor: bool
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    """Return regular squares plus the separately placed distractor candidate."""
    squares = [(side, square_colors[side]) for side in SIDES for _ in range(counts[side])]
    if has_distractor:
        squares.append((SIDES[0], square_colors[SIDES[0]]))

    random.shuffle(squares)
    if not has_distractor:
        return squares, None
    return squares[:-1], squares[-1]


def _free_positions(occupied: np.ndarray, side: int, min_r: int = 0) -> list[tuple[int, int]]:
    """List top-left positions whose padded square footprint is unoccupied."""
    H, W = occupied.shape
    valid = []
    for r in range(min_r, H - side + 1):
        for c in range(0, W - side + 1):
            r0 = max(0, r - SQUARE_GAP)
            r1 = min(H, r + side + SQUARE_GAP)
            c0 = max(0, c - SQUARE_GAP)
            c1 = min(W, c + side + SQUARE_GAP)
            if not occupied[r0:r1, c0:c1].any():
                valid.append((r, c))
    return valid


def _mark_occupied(occupied: np.ndarray, r: int, c: int, side: int) -> None:
    H, W = occupied.shape
    r0 = max(0, r - SQUARE_GAP)
    r1 = min(H, r + side + SQUARE_GAP)
    c0 = max(0, c - SQUARE_GAP)
    c1 = min(W, c + side + SQUARE_GAP)
    occupied[r0:r1, c0:c1] = True


def _place_square(
    grid: np.ndarray,
    occupied: np.ndarray,
    placed: list[tuple[int, int, int, int]],
    side: int,
    color: int,
    positions: list[tuple[int, int]] | None = None,
) -> bool:
    """Place one square at a random valid position and update grid state."""
    valid = _free_positions(occupied, side) if positions is None else positions
    if not valid:
        return False
    r, c = random.choice(valid)
    grid[r:r + side, c:c + side] = color
    _mark_occupied(occupied, r, c, side)
    placed.append((r, c, side, color))
    return True


def _place_squares(
    grid: np.ndarray,
    occupied: np.ndarray,
    regular_squares: list[tuple[int, int]],
    distractor_sq: tuple[int, int] | None,
) -> list[tuple[int, int, int, int]] | None:
    """Place all squares, requiring any distractor to sit below the others."""
    placed = []

    random.shuffle(regular_squares)
    for side, color in regular_squares:
        if not _place_square(grid, occupied, placed, side, color):
            return None

    if distractor_sq is None:
        return placed

    d_side, d_color = distractor_sq
    current_max_bottom = max(r + side - 1 for r, _, side, _ in placed) if placed else -1
    min_r = max(0, current_max_bottom - d_side + 2)
    valid = [
        (r, c)
        for r, c in _free_positions(occupied, d_side, min_r=min_r)
        if r + d_side - 1 > current_max_bottom
    ]
    if not _place_square(grid, occupied, placed, d_side, d_color, valid):
        return None
    return placed


def _add_noise(grid: np.ndarray, bg: int, noise_color: int) -> bool:
    """Scatter noise onto background cells only."""
    free_cells = list(zip(*np.where(grid == bg)))
    num_noise = random.randint(*NOISE_COUNT_RANGE)
    if len(free_cells) < num_noise:
        return False
    random.shuffle(free_cells)
    for r, c in free_cells[:num_noise]:
        grid[r, c] = noise_color
    return True


def _would_trigger_unintended_distractor(placed: list[tuple[int, int, int, int]]) -> bool:
    """Detect no-distractor samples where the solver would drop a small square."""
    lowest_row = max(r + side - 1 for r, _, side, _ in placed)
    return any(side == SIDES[0] and r + side - 1 == lowest_row for r, _, side, _ in placed)


def _render_expected_output(
    bg: int, counts: dict[int, int], square_colors: dict[int, int]
) -> np.ndarray:
    """Render the count-derived oracle output used to validate samples."""
    out_h = SIDES[-1] + 1
    groups = []
    for rank in range(max(counts.values())):
        present_sides = [side for side in SIDES if counts[side] > rank]
        group = np.full((out_h, max(present_sides)), bg, dtype=int)
        for side in sorted(present_sides, reverse=True):
            group[:side, :side] = square_colors[side]
        groups.append(group)

    out_w = sum(group.shape[1] + 1 for group in groups)
    out = np.full((out_h, out_w), bg, dtype=int)
    c = 0
    for group in groups:
        w = group.shape[1]
        out[:, c:c + w] = group
        c += w + 1
    return out


def _gen_attempt() -> np.ndarray | None:
    bg, square_colors, noise_color = _choose_palette()
    counts = {side: random.randint(*COUNT_RANGE) for side in SIDES}
    has_distractor = random.random() < DISTRACTOR_PROB
    regular_squares, distractor_sq = _make_square_queue(counts, square_colors, has_distractor)

    H = random.randint(*GRID_SIZE_RANGE)
    W = random.randint(*GRID_SIZE_RANGE)
    grid = np.full((H, W), bg, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    placed = _place_squares(grid, occupied, regular_squares, distractor_sq)
    if placed is None or not _add_noise(grid, bg, noise_color):
        return None

    if not has_distractor and _would_trigger_unintended_distractor(placed):
        return None

    expected_out = _render_expected_output(bg, counts, square_colors)

    try:
        actual_out = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual_out, expected_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {MAX_GENERATION_ATTEMPTS} attempts")
