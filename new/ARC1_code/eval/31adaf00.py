import numpy as np

from arc_utils import Color
import random


Square = tuple[int, int, int]  # (r0, c0, side)


def _overlap(a: Square, b: Square) -> bool:
    ar, ac, aside = a
    br, bc, bside = b
    return ar < br + bside and br < ar + aside and ac < bc + bside and bc < ac + aside


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor every maximal non-overlapping black square pocket blue while leaving grey pixels unchanged.

    Input structure
    - The grid is a black background with grey obstacle pixels.
    - The relevant blank regions are axis-aligned all-black squares of side at least 2.
    - The output has the same shape as the input.

    Rule
    - Search for all all-black square candidate regions of side at least 2.
    - Process candidates from largest side length to smallest.
    - Paint a candidate blue if it does not overlap any larger square already selected.
    - Same-size candidates that survive larger-square blocking must be disjoint.

    Key insight
    - Long corridors and shifted protrusions are not selected; overlapping larger square rooms claim the cells first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert min(height, width) >= 2, "expected room for at least a 2x2 square"
    assert set(map(int, np.unique(g))) == {Color.BLACK, Color.GREY}, "expected black background and grey obstacles"

    selected: list[Square] = []
    for side in range(min(height, width), 1, -1):
        candidates: list[Square] = []
        for r in range(height - side + 1):
            for c in range(width - side + 1):
                square = (r, c, side)
                if any(_overlap(square, chosen) for chosen in selected):
                    continue
                if np.all(g[r : r + side, c : c + side] == Color.BLACK):
                    candidates.append(square)

        for i, square in enumerate(candidates):
            assert all(
                not _overlap(square, other) for other in candidates[i + 1 :]
            ), "expected same-size candidate squares to be disjoint"
        selected.extend(candidates)

    assert selected, "expected at least one black square pocket"

    out = g.copy()
    for r, c, side in selected:
        out[r : r + side, c : c + side] = Color.BLUE
    return out

# --------------- generator (o5x) ---------------

# Canvas.
_GEN_GRID_SIZE = 10
_GEN_SHAPE = (_GEN_GRID_SIZE, _GEN_GRID_SIZE)
# Pocket layout.
_GEN_POCKET_COUNT_RANGE = (3, 5)
_GEN_POCKET_SIDE_CHOICES = (2, 3)
_GEN_REQUIRED_SIDE = 2
_GEN_MIN_SELECTED_SIDE = 2
_GEN_NO_OWNER = -1
# Grey obstacle count.
_GEN_GREY_COUNT_RANGE = (35, 41)
_GEN_MAX_ATTEMPTS = 300


def _gen_pocket_mask(squares):
    mask = np.zeros(_GEN_SHAPE, dtype=bool)
    for r, c, side in squares:
        mask[r : r + side, c : c + side] = True
    return mask


def _gen_owner_grid(squares):
    owner = np.full(_GEN_SHAPE, _GEN_NO_OWNER, dtype=int)
    for idx, (r, c, side) in enumerate(squares):
        if np.any(owner[r : r + side, c : c + side] != _GEN_NO_OWNER):
            return None
        owner[r : r + side, c : c + side] = idx
    return owner


def _gen_min_square_cells(r, c):
    return (
        (r + dr, c + dc)
        for dc in range(_GEN_MIN_SELECTED_SIDE)
        for dr in range(_GEN_MIN_SELECTED_SIDE)
    )


def _gen_layout_ok(squares):
    owner = _gen_owner_grid(squares)
    if owner is None:
        return False

    limit = _GEN_GRID_SIZE - _GEN_MIN_SELECTED_SIDE + 1
    for r in range(limit):
        for c in range(limit):
            block = owner[r : r + _GEN_MIN_SELECTED_SIDE, c : c + _GEN_MIN_SELECTED_SIDE].ravel()
            if np.all(block != _GEN_NO_OWNER) and len(set(map(int, block))) > 1:
                return False
    return True


def _gen_choose_sides(count):
    sides = [random.choice(_GEN_POCKET_SIDE_CHOICES) for _ in range(count)]
    if _GEN_REQUIRED_SIDE not in sides:
        sides[random.randrange(count)] = _GEN_REQUIRED_SIDE
    random.shuffle(sides)
    return sides


def _gen_choose_pockets():
    count = random.randint(*_GEN_POCKET_COUNT_RANGE)
    squares = []
    for side in _gen_choose_sides(count):
        candidates = []
        for r in range(_GEN_GRID_SIZE - side + 1):
            for c in range(_GEN_GRID_SIZE - side + 1):
                trial = squares + [(r, c, side)]
                if _gen_layout_ok(trial):
                    candidates.append((r, c, side))
        if not candidates:
            return None
        squares.append(random.choice(candidates))
    return squares


def _gen_min_square_inside_pocket(r, c, squares):
    for sr, sc, side in squares:
        fits_rows = sr <= r and r + _GEN_MIN_SELECTED_SIDE <= sr + side
        fits_cols = sc <= c and c + _GEN_MIN_SELECTED_SIDE <= sc + side
        if fits_rows and fits_cols:
            return True
    return False


def _gen_required_windows(squares, pocket_mask):
    windows = []
    limit = _GEN_GRID_SIZE - _GEN_MIN_SELECTED_SIDE + 1
    for r in range(limit):
        for c in range(limit):
            if _gen_min_square_inside_pocket(r, c, squares):
                continue
            cells = tuple(_gen_min_square_cells(r, c))
            blockers = [(rr, cc) for rr, cc in cells if not pocket_mask[rr, cc]]
            if not blockers:
                return None
            windows.append(tuple(blockers))
    return windows


def _gen_add_missing_row_col_grey(grey, pocket_mask, target_grey):
    while True:
        missing_rows = {r for r in range(_GEN_GRID_SIZE) if all((r, c) not in grey for c in range(_GEN_GRID_SIZE))}
        missing_cols = {c for c in range(_GEN_GRID_SIZE) if all((r, c) not in grey for r in range(_GEN_GRID_SIZE))}
        if not missing_rows and not missing_cols:
            return True
        if len(grey) >= target_grey:
            return False

        best = []
        best_score = -1
        for r in range(_GEN_GRID_SIZE):
            for c in range(_GEN_GRID_SIZE):
                if pocket_mask[r, c] or (r, c) in grey:
                    continue
                score = int(r in missing_rows) + int(c in missing_cols)
                if score > best_score:
                    best = [(r, c)]
                    best_score = score
                elif score == best_score:
                    best.append((r, c))
        if best_score <= 0:
            return False
        grey.add(random.choice(best))


def _gen_cover_required_windows(windows, target_grey):
    cell_to_windows = {}
    for idx, blockers in enumerate(windows):
        for cell in blockers:
            cell_to_windows.setdefault(cell, []).append(idx)

    grey = set()
    uncovered = set(range(len(windows)))
    while uncovered:
        scored = []
        best_score = 0
        for cell, covered_windows in cell_to_windows.items():
            if cell in grey:
                continue
            score = sum(idx in uncovered for idx in covered_windows)
            if score > best_score:
                scored = [cell]
                best_score = score
            elif score == best_score and score > 0:
                scored.append(cell)
        if not scored:
            return None
        cell = random.choice(scored)
        grey.add(cell)
        uncovered.difference_update(cell_to_windows[cell])
        if len(grey) > target_grey:
            return None
    return grey


def _gen_fill_extra_grey(grey, pocket_mask, target_grey):
    extras = [
        (r, c)
        for r in range(_GEN_GRID_SIZE)
        for c in range(_GEN_GRID_SIZE)
        if not pocket_mask[r, c] and (r, c) not in grey
    ]
    random.shuffle(extras)
    while len(grey) < target_grey and extras:
        grey.add(extras.pop())
    return len(grey) == target_grey


def _gen_build_grey(squares, target_grey):
    pocket_mask = _gen_pocket_mask(squares)
    windows = _gen_required_windows(squares, pocket_mask)
    if windows is None:
        return None

    grey = _gen_cover_required_windows(windows, target_grey)
    if grey is None:
        return None
    if not _gen_add_missing_row_col_grey(grey, pocket_mask, target_grey):
        return None
    if not _gen_fill_extra_grey(grey, pocket_mask, target_grey):
        return None
    return grey


def _gen_render_pair(squares, grey):
    out = np.full(_GEN_SHAPE, Color.BLACK, dtype=int)
    for r, c in grey:
        out[r, c] = Color.GREY
    for r, c, side in squares:
        out[r : r + side, c : c + side] = Color.BLUE

    inp = out.copy()
    inp[inp == Color.BLUE] = Color.BLACK
    return inp, out


def _gen_attempt():
    squares = _gen_choose_pockets()
    if squares is None:
        return None

    target_grey = random.randint(*_GEN_GREY_COUNT_RANGE)
    grey = _gen_build_grey(squares, target_grey)
    if grey is None:
        return None

    inp, out = _gen_render_pair(squares, grey)
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input():
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 31adaf00 input")
