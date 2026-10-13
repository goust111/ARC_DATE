import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
SEP = 5
FG = 7

PANEL_HEIGHT = 12
PANEL_WIDTH = 5
MAX_GENERATION_ATTEMPTS = 500


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _separator_column(grid: np.ndarray, bg: int) -> int:
    h, w = map(int, grid.shape)
    assert w >= 3 and w % 2 == 1, "expected two equal panels separated by one column"

    candidates: list[int] = []
    for c in range(w):
        color = int(grid[0, c])
        if color == bg:
            continue
        if not np.all(grid[:, c] == color):
            continue
        if c == 0 or c == w - 1 or c != w - c - 1:
            continue
        without = np.concatenate((grid[:, :c], grid[:, c + 1 :]), axis=1)
        if np.any(without == color):
            continue
        candidates.append(c)

    assert len(candidates) == 1, "expected one unique solid separator column"
    return candidates[0]


def _comparison_panel(left: np.ndarray, right: np.ndarray, bg: int) -> np.ndarray:
    assert left.shape == right.shape, "panels must have the same shape"
    out = np.full(left.shape, bg, dtype=int)

    same = left == right
    out[same] = left[same]

    left_only = (left != bg) & (right == bg)
    right_only = (left == bg) & (right != bg)
    out[left_only] = RED
    out[right_only] = BLUE

    conflict = (left != bg) & (right != bg) & (left != right)
    assert not np.any(conflict), "aligned foreground pixels must match in color"
    return out


def _propagate(seed_panel: np.ndarray, bg: int) -> np.ndarray:
    out = seed_panel.copy()
    h, w = out.shape

    for c in range(w):
        red_rows = np.flatnonzero(seed_panel[:, c] == RED)
        blue_rows = np.flatnonzero(seed_panel[:, c] == BLUE)
        assert not (red_rows.size and blue_rows.size), "a column cannot contain both propagation directions"

        for r in red_rows:
            rr = int(r) + 1
            while rr < h and int(seed_panel[rr, c]) == bg:
                out[rr, c] = RED
                rr += 1

        for r in blue_rows:
            rr = int(r) - 1
            while rr >= 0 and int(seed_panel[rr, c]) == bg:
                out[rr, c] = BLUE
                rr -= 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare two separated panels and use the asymmetric differences as vertical propagation seeds.

    Input structure:
    - The grid contains two equal-width black-background panels separated by one solid gray divider column.
    - Each side panel uses orange foreground pixels, and most aligned cells match between the two panels.
    - A differing aligned cell has orange on exactly one side, which encodes both a seed location and its propagation direction.

    Rule:
    - Remove the divider and compare the left and right panels cell by cell.
    - Copy shared orange pixels, keep shared background black, mark left-only foreground as red seeds, and mark right-only foreground as blue seeds.
    - Then propagate every red seed downward through consecutive black cells in its own column, and propagate every blue seed upward through consecutive black cells in its own column.
    - Propagation stops before the first non-black cell in the pre-propagation comparison map or at the grid edge.

    Key insight:
    - The panel-difference map is only an intermediate representation: which side owns each mismatch determines whether that column should grow downward or upward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    sep_col = _separator_column(g, bg)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert left.shape == right.shape, "divider must split two equal panels"

    seed_panel = _comparison_panel(left, right, bg)
    return _propagate(seed_panel, bg)


def _sample_rows(rng: random.Random, candidates: Iterable[int], count_range: tuple[int, int]) -> list[int]:
    rows = list(candidates)
    if not rows:
        return []
    lo, hi = count_range
    hi = min(hi, len(rows))
    if hi < lo:
        return []
    count = rng.randint(lo, hi)
    return sorted(rng.sample(rows, count))


def _gen_seed_panel(rng: random.Random) -> np.ndarray | None:
    panel = np.full((PANEL_HEIGHT, PANEL_WIDTH), BG, dtype=int)

    diff_count = rng.randint(3, 5)
    diff_cols = sorted(rng.sample(range(PANEL_WIDTH), diff_count))
    red_count = rng.randint(1, diff_count - 1)
    red_cols = set(rng.sample(diff_cols, red_count))
    blue_cols = set(diff_cols) - red_cols

    edge_col = rng.choice(diff_cols)
    stopper_choices = [c for c in diff_cols if c != edge_col]
    stopper_col = rng.choice(stopper_choices)

    total_shared = 0
    edge_branch_seen = False
    stopper_branch_seen = False

    for c in range(PANEL_WIDTH):
        if c not in diff_cols:
            shared_rows = _sample_rows(rng, range(PANEL_HEIGHT), (1, 3))
            for r in shared_rows:
                panel[r, c] = FG
            total_shared += len(shared_rows)
            continue

        is_red = c in red_cols
        mode = "edge" if c == edge_col else "stopper" if c == stopper_col else rng.choice(("edge", "stopper"))

        if is_red:
            if mode == "edge":
                seed_row = rng.randint(1, PANEL_HEIGHT - 2)
                anchor_rows = _sample_rows(rng, range(0, seed_row), (0, 2))
                edge_branch_seen = True
            else:
                seed_row = rng.randint(1, PANEL_HEIGHT - 4)
                blocker_row = rng.randint(seed_row + 2, PANEL_HEIGHT - 2)
                panel[blocker_row, c] = FG
                anchor_rows = _sample_rows(rng, range(0, seed_row), (0, 2))
                stopper_branch_seen = True

            panel[seed_row, c] = RED
            for r in anchor_rows:
                panel[r, c] = FG
            total_shared += len(anchor_rows) + int(mode == "stopper")
        else:
            if mode == "edge":
                seed_row = rng.randint(1, PANEL_HEIGHT - 2)
                anchor_rows = _sample_rows(rng, range(seed_row + 1, PANEL_HEIGHT), (0, 2))
                edge_branch_seen = True
            else:
                seed_row = rng.randint(3, PANEL_HEIGHT - 2)
                blocker_row = rng.randint(1, seed_row - 2)
                panel[blocker_row, c] = FG
                anchor_rows = _sample_rows(rng, range(seed_row + 1, PANEL_HEIGHT), (0, 2))
                stopper_branch_seen = True

            panel[seed_row, c] = BLUE
            for r in anchor_rows:
                panel[r, c] = FG
            total_shared += len(anchor_rows) + int(mode == "stopper")

    if total_shared < 4 or not edge_branch_seen or not stopper_branch_seen:
        return None
    return panel


def _seed_panel_to_input(seed_panel: np.ndarray) -> np.ndarray:
    left = np.full(seed_panel.shape, BG, dtype=int)
    right = np.full(seed_panel.shape, BG, dtype=int)

    shared = seed_panel == FG
    left[shared] = FG
    right[shared] = FG

    left[seed_panel == RED] = FG
    right[seed_panel == BLUE] = FG

    separator = np.full((seed_panel.shape[0], 1), SEP, dtype=int)
    return np.concatenate((left, separator, right), axis=1)


def _validate_generated_case(inp: np.ndarray, seed_panel: np.ndarray) -> bool:
    if inp.shape != (PANEL_HEIGHT, PANEL_WIDTH * 2 + 1):
        return False

    bg = _dominant_color(inp)
    if bg != BG:
        return False

    sep_col = _separator_column(inp, bg)
    if sep_col != PANEL_WIDTH:
        return False

    left = inp[:, :sep_col]
    right = inp[:, sep_col + 1 :]
    if not set(np.unique(left)).issubset({BG, FG}):
        return False
    if not set(np.unique(right)).issubset({BG, FG}):
        return False

    rebuilt = _comparison_panel(left, right, bg)
    if not np.array_equal(rebuilt, seed_panel):
        return False

    red_rows, red_cols = np.where(seed_panel == RED)
    blue_rows, blue_cols = np.where(seed_panel == BLUE)
    diff_total = len(red_rows) + len(blue_rows)
    if not (3 <= diff_total <= 5):
        return False
    if not len(red_rows) or not len(blue_rows):
        return False
    if len(set(red_cols) | set(blue_cols)) != diff_total:
        return False

    edge_branch = False
    stopper_branch = False
    for c in range(PANEL_WIDTH):
        red_pos = np.flatnonzero(seed_panel[:, c] == RED)
        blue_pos = np.flatnonzero(seed_panel[:, c] == BLUE)
        if red_pos.size and blue_pos.size:
            return False
        if red_pos.size > 1 or blue_pos.size > 1:
            return False

        if red_pos.size == 1:
            r = int(red_pos[0])
            if r == PANEL_HEIGHT - 1 or int(seed_panel[r + 1, c]) != BG:
                return False
            stop_row = r + 1
            while stop_row < PANEL_HEIGHT and int(seed_panel[stop_row, c]) == BG:
                stop_row += 1
            if stop_row == PANEL_HEIGHT:
                edge_branch = True
            else:
                stopper_branch = True

        if blue_pos.size == 1:
            r = int(blue_pos[0])
            if r == 0 or int(seed_panel[r - 1, c]) != BG:
                return False
            stop_row = r - 1
            while stop_row >= 0 and int(seed_panel[stop_row, c]) == BG:
                stop_row -= 1
            if stop_row < 0:
                edge_branch = True
            else:
                stopper_branch = True

    if not edge_branch or not stopper_branch:
        return False

    expected = _propagate(seed_panel, BG)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    if int(np.count_nonzero(seed_panel == FG)) < 4:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        seed_panel = _gen_seed_panel(rng)
        if seed_panel is None:
            continue
        inp = _seed_panel_to_input(seed_panel)
        if _validate_generated_case(inp, seed_panel):
            return inp
    raise RuntimeError("failed to generate a valid panel-diff propagation input")
