import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

HEIGHT = 10
PANEL_WIDTH = 5
SEPARATOR_WIDTH = 1
FULL_WIDTH = PANEL_WIDTH * 2 + SEPARATOR_WIDTH
PERIOD_CHOICES = (2, 3)
SHIFT_CHOICES = (1, 2)
MAX_GENERATION_ATTEMPTS = 400


def _observed_height(panel: np.ndarray, fg: int) -> int:
    rows = np.where(np.any(panel == fg, axis=1))[0]
    if rows.size == 0:
        raise ValueError("panel must contain at least one foreground row")
    return int(rows[-1]) + 1


def _all_candidates(observed: np.ndarray) -> list[tuple[int, int]]:
    h, w = observed.shape
    candidates: list[tuple[int, int]] = []
    for period in range(1, min(h, 3) + 1):
        if h <= period:
            continue
        for shift in range(w):
            ok = True
            for r in range(h - period):
                if not np.array_equal(observed[r + period], np.roll(observed[r], shift)):
                    ok = False
                    break
            if ok:
                candidates.append((period, shift))
    return candidates


def _infer_period_and_shift(observed: np.ndarray) -> tuple[int, int]:
    candidates = _all_candidates(observed)
    if not candidates:
        raise ValueError("no valid period/shift pair found")
    candidates.sort()
    return candidates[0]


def _complete_panel(panel: np.ndarray, fg: int) -> np.ndarray:
    observed_h = _observed_height(panel, fg)
    observed = panel[:observed_h] == fg
    period, shift = _infer_period_and_shift(observed)

    full = np.zeros((panel.shape[0], panel.shape[1]), dtype=bool)
    full[:observed_h] = observed
    for r in range(observed_h, panel.shape[0]):
        full[r] = np.roll(full[r - period], shift)
    return full


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete two truncated periodic row patterns and output their pixelwise overlap.

    Input structure:
    - The grid contains a left blue panel and a right red panel separated by one full-height grey column.
    - Each side is the same width and uses black as background; only the top rows of each periodic pattern are shown.
    - Within each panel, rows repeat after a small vertical period, with the repeated row circularly shifted to the right.

    Rule:
    - Locate the grey separator column and split the grid into the two equal-width panels.
    - Independently infer the row period and horizontal circular shift for the visible blue rows on the left and red rows on the right.
    - Extend both panel patterns to the full grid height, then output a single panel marking green exactly where both completed panels contain foreground.

    Key insight:
    - The lower rows are missing information rather than true empty space, so each side must be reconstructed before taking the logical AND.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    if not sep_cols:
        raise ValueError("missing separator column")
    sep = sep_cols[0]
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    if left.shape != right.shape:
        raise ValueError("panels must have equal shape")

    left_full = _complete_panel(left, BLUE)
    right_full = _complete_panel(right, RED)

    out = np.zeros(left.shape, dtype=int)
    out[left_full & right_full] = GREEN
    return out


def _random_mask(width: int) -> np.ndarray:
    count = random.randint(1, 3)
    if random.random() < 0.55:
        run = min(count, width - 1)
        start = random.randint(0, width - run)
        cols = list(range(start, start + run))
    else:
        cols = random.sample(range(width), count)
    row = np.zeros(width, dtype=bool)
    row[cols] = True
    if row.all():
        row[random.randrange(width)] = False
    return row


def _build_full_panel(period: int, shift: int, width: int, height: int) -> np.ndarray:
    bases = [_random_mask(width) for _ in range(period)]
    full = np.zeros((height, width), dtype=bool)
    for r in range(height):
        base = bases[r % period]
        steps = r // period
        full[r] = np.roll(base, steps * shift)
    return full


def _make_input_from_full(full: np.ndarray, visible_h: int, fg: int) -> np.ndarray:
    panel = np.zeros(full.shape, dtype=int)
    visible = panel[:visible_h]
    visible[full[:visible_h]] = fg
    return panel


def _valid_observed_panel(panel: np.ndarray, fg: int, truth: tuple[int, int]) -> bool:
    observed_h = _observed_height(panel, fg)
    observed = panel[:observed_h] == fg
    candidates = _all_candidates(observed)
    return candidates == [truth]


def _nontrivial_overlap(
    left_full: np.ndarray,
    right_full: np.ndarray,
    left_visible_h: int,
    right_visible_h: int,
) -> bool:
    overlap = left_full & right_full
    if not overlap.any():
        return False

    overlap_count = int(overlap.sum())
    if overlap_count >= int(left_full.sum()) or overlap_count >= int(right_full.sum()):
        return False

    naive_left = np.zeros_like(left_full)
    naive_right = np.zeros_like(right_full)
    naive_left[:left_visible_h] = left_full[:left_visible_h]
    naive_right[:right_visible_h] = right_full[:right_visible_h]
    naive_overlap = naive_left & naive_right
    if np.array_equal(naive_overlap, overlap):
        return False

    hidden_rows = np.zeros(left_full.shape[0], dtype=bool)
    hidden_rows[left_visible_h:] = True
    hidden_rows[right_visible_h:] = True
    if not overlap[hidden_rows].any():
        return False

    if overlap_count > 12:
        return False
    return True


def _visible_height(period: int) -> int:
    lo = max(4, 2 * period)
    hi = 6
    return random.randint(lo, hi)


def _generate_attempt() -> np.ndarray | None:
    left_period = random.choice(PERIOD_CHOICES)
    right_period = random.choice(PERIOD_CHOICES)
    left_shift = random.choice(SHIFT_CHOICES)
    right_shift = random.choice(SHIFT_CHOICES)

    left_full = _build_full_panel(left_period, left_shift, PANEL_WIDTH, HEIGHT)
    right_full = _build_full_panel(right_period, right_shift, PANEL_WIDTH, HEIGHT)

    left_visible_h = _visible_height(left_period)
    right_visible_h = _visible_height(right_period)

    left_panel = _make_input_from_full(left_full, left_visible_h, BLUE)
    right_panel = _make_input_from_full(right_full, right_visible_h, RED)

    if not _valid_observed_panel(left_panel, BLUE, (left_period, left_shift)):
        return None
    if not _valid_observed_panel(right_panel, RED, (right_period, right_shift)):
        return None
    if not _nontrivial_overlap(left_full, right_full, left_visible_h, right_visible_h):
        return None

    grid = np.zeros((HEIGHT, FULL_WIDTH), dtype=int)
    grid[:, :PANEL_WIDTH] = left_panel
    grid[:, PANEL_WIDTH] = GREY
    grid[:, PANEL_WIDTH + 1 :] = right_panel

    expected = np.zeros((HEIGHT, PANEL_WIDTH), dtype=int)
    expected[left_full & right_full] = GREEN
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
