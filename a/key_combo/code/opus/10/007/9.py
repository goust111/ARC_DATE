import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

WIDTH_CHOICES = (7, 9, 11, 13)
MIN_HEIGHT = 6
MAX_HEIGHT = 11
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    separator_cols = np.flatnonzero(np.all(g == RED, axis=0))
    assert len(separator_cols) == 1, "expected exactly one full-height red separator"
    sep = int(separator_cols[0])

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape and left.size > 0, "panels must have equal non-empty shapes"

    for panel in (left, right):
        h, w = panel.shape
        assert w >= 5, "panel must be wide enough to contain two bars and interior markers"
        assert np.all(panel[:, 0] == BLUE), "each panel must start with a blue edge bar"
        assert np.all(panel[:, w - 1] == GREEN), "each panel must end with a green edge bar"
        inner_colors = set(map(int, np.unique(panel[:, 1 : w - 1])))
        assert inner_colors <= {BLACK, ORANGE}, "panel interiors may only contain background or orange controls"

    return left, right


def _solve_from_panels(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    h, w = left.shape
    left_mask = left == ORANGE
    right_mask = right == ORANGE
    union_mask = left_mask | right_mask
    shared_mask = left_mask & right_mask

    merged = np.zeros((h, w), dtype=int)
    merged[:, 0] = BLUE
    merged[:, w - 1] = GREEN
    merged[union_mask] = ORANGE

    paints: dict[tuple[int, int], int] = {}
    stats = {
        "shared_controls": 0,
        "left_only_controls": 0,
        "right_only_controls": 0,
        "equidistant_controls": 0,
        "shared_success_controls": 0,
        "shared_non_equidistant_success_controls": 0,
        "unique_equidistant_success_controls": 0,
        "cancelled_extensions": 0,
        "visible_extension_colors": set(),
    }

    for r0, c0 in np.argwhere(union_mask):
        r = int(r0)
        c = int(c0)
        is_shared = bool(shared_mask[r, c])
        is_left_only = bool(left_mask[r, c] and not right_mask[r, c])
        is_right_only = bool(right_mask[r, c] and not left_mask[r, c])
        if is_shared:
            stats["shared_controls"] += 1
        elif is_left_only:
            stats["left_only_controls"] += 1
        else:
            stats["right_only_controls"] += 1

        left_dist = c
        right_dist = (w - 1) - c
        equidistant = left_dist == right_dist
        if equidistant:
            stats["equidistant_controls"] += 1

        targets: list[tuple[int, int, int]] = []
        if left_dist <= right_dist:
            targets.append((r, c - 1, YELLOW if is_shared else BLUE))
        if right_dist <= left_dist:
            targets.append((r, c + 1, YELLOW if is_shared else GREEN))

        success_count = 0
        for nr, nc, color in targets:
            assert _in_bounds(merged.shape, nr, nc), "extension must stay within the merged panel"
            if merged[nr, nc] != BLACK:
                stats["cancelled_extensions"] += 1
                continue
            key = (nr, nc)
            if key in paints:
                assert paints[key] == color, "conflicting extensions target the same cell"
            else:
                paints[key] = color
            success_count += 1
            stats["visible_extension_colors"].add(color)

        if is_shared and success_count > 0:
            stats["shared_success_controls"] += 1
            if not equidistant:
                stats["shared_non_equidistant_success_controls"] += 1
        if (not is_shared) and equidistant and success_count == 2:
            stats["unique_equidistant_success_controls"] += 1

    out = merged.copy()
    for (r, c), color in paints.items():
        out[r, c] = int(color)
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge the two orange control masks across the red divider and extend each merged control one step toward its nearer edge bar.

    Input structure:
    - A black grid is split into two equal panels by one full-height red separator column.
    - Inside each panel, the leftmost column is a full blue edge bar and the rightmost column is a full green edge bar.
    - Between the bars, each panel contains isolated orange control pixels; some positions appear in both panels and some in only one.

    Rule:
    - Remove the red separator and align the two panels cell by cell into one panel-sized canvas.
    - Keep the blue and green edge bars, and place an orange control pixel wherever either panel contains one at that position.
    - For each merged orange control pixel, compare its horizontal distance to the blue bar and the green bar: extend one cell toward the nearer bar, or one cell in both directions if the distances are equal.
    - If the control came from only one panel, each successful extension uses the color of the bar it moves toward; if it came from both panels, every successful extension uses yellow instead.
    - Any attempted extension that would land on an existing non-black cell is cancelled, and the original orange controls remain unchanged.

    Key insight:
    - The logical OR merge does more than combine masks: it creates a visible difference between shared controls and single-panel controls, because overlap changes only the extension color while the distance-to-bars rule still controls the direction.
    """
    left, right = _parse_input(grid)
    out, _stats = _solve_from_panels(left, right)
    return out


COUNT_TRIPLES = tuple(
    (shared, left_only, right_only)
    for shared in range(1, 4)
    for left_only in range(1, 4)
    for right_only in range(1, 4)
    if 3 <= shared + left_only <= 5
    if 3 <= shared + right_only <= 5
    if 4 <= shared + left_only + right_only <= 6
)


def _assemble_input(height: int, width: int, placements: list[tuple[int, int, str]]) -> np.ndarray:
    left = np.zeros((height, width), dtype=int)
    right = np.zeros((height, width), dtype=int)
    left[:, 0] = BLUE
    left[:, width - 1] = GREEN
    right[:, 0] = BLUE
    right[:, width - 1] = GREEN

    for r, c, label in placements:
        if label in {"L", "B"}:
            left[r, c] = ORANGE
        if label in {"R", "B"}:
            right[r, c] = ORANGE

    separator = np.full((height, 1), RED, dtype=int)
    return np.concatenate([left, separator, right], axis=1)


def _random_filler_column(rng: random.Random, width: int) -> int:
    mid = (width - 1) // 2
    weighted = [1, width - 2, mid] + list(range(2, width - 2))
    return rng.choice(weighted)


def _build_positions(
    rng: random.Random,
    height: int,
    width: int,
    counts: tuple[int, int, int],
) -> list[tuple[int, int, str]] | None:
    shared_count, left_only_count, right_only_count = counts
    total = shared_count + left_only_count + right_only_count
    if total > height:
        return None

    rows = list(range(height))
    rng.shuffle(rows)
    row_iter = iter(rows)
    mid = (width - 1) // 2
    compare_side = rng.choice(("left", "right"))
    compare_cols = list(range(2, mid)) if compare_side == "left" else list(range(mid + 1, width - 2))
    if not compare_cols:
        return None
    compare_col = rng.choice(compare_cols)

    remaining = {"B": shared_count, "L": left_only_count, "R": right_only_count}
    placements: list[tuple[int, int, str]] = []

    def take(row: int, col: int, label: str) -> None:
        placements.append((row, col, label))
        remaining[label] -= 1

    shared_row = next(row_iter)
    unique_same_row = next(row_iter)
    equidistant_row = next(row_iter)
    unique_same_label = "L" if compare_side == "left" else "R"
    unique_eq_label = "R" if compare_side == "left" else "L"

    take(shared_row, compare_col, "B")
    take(unique_same_row, compare_col, unique_same_label)
    take(equidistant_row, mid, unique_eq_label)

    available_cancel_labels = [label for label, count in remaining.items() if count > 0]
    if not available_cancel_labels:
        return None
    cancel_row = next(row_iter)
    cancel_label = rng.choice(available_cancel_labels)
    cancel_col = rng.choice((1, width - 2))
    take(cancel_row, cancel_col, cancel_label)

    leftover_labels = [
        label
        for label, count in remaining.items()
        for _ in range(count)
    ]
    rng.shuffle(leftover_labels)
    used_rows = {shared_row, unique_same_row, equidistant_row, cancel_row}
    for label in leftover_labels:
        row = next(row_iter)
        assert row not in used_rows
        col = _random_filler_column(rng, width)
        placements.append((row, col, label))
        used_rows.add(row)

    assert len(placements) == total
    return placements


def _validate_case(inp: np.ndarray, counts: tuple[int, int, int]) -> bool:
    try:
        left, right = _parse_input(inp)
        expected, stats = _solve_from_panels(left, right)
        observed = main(inp)
    except Exception:
        return False

    if not np.array_equal(observed, expected):
        return False

    shared_count, left_only_count, right_only_count = counts
    if int(np.count_nonzero(left == ORANGE)) != shared_count + left_only_count:
        return False
    if int(np.count_nonzero(right == ORANGE)) != shared_count + right_only_count:
        return False

    visible_colors = set(stats["visible_extension_colors"])
    if stats["shared_controls"] < 1:
        return False
    if stats["left_only_controls"] < 1 or stats["right_only_controls"] < 1:
        return False
    if stats["equidistant_controls"] < 1:
        return False
    if stats["shared_non_equidistant_success_controls"] < 1:
        return False
    if stats["unique_equidistant_success_controls"] < 1:
        return False
    if stats["cancelled_extensions"] < 1:
        return False
    if YELLOW not in visible_colors:
        return False
    if BLUE not in visible_colors or GREEN not in visible_colors:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    counts = rng.choice(COUNT_TRIPLES)
    total = sum(counts)
    width = rng.choice(WIDTH_CHOICES)
    height = rng.randint(max(MIN_HEIGHT, total), MAX_HEIGHT)
    placements = _build_positions(rng, height, width, counts)
    if placements is None:
        return None

    inp = _assemble_input(height, width, placements)
    if not _validate_case(inp, counts):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
