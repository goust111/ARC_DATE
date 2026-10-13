from __future__ import annotations

from collections import Counter
import random

import numpy as np


MARKER_COLOR = 4

# --------------- solver ---------------


def _mode_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    assert counts, "expected a non-empty grid"
    return int(counts.most_common(1)[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Activate rows and columns with odd foreground counts, then mark their empty intersections.

    Input structure
    - A black background contains scattered pixels of one foreground color.
    - The yellow marker color does not appear in the input.

    Rule
    - A row is activated when it contains an odd number of foreground pixels.
    - A column is activated when it contains an odd number of foreground pixels.
    - Paint yellow on every background cell at an activated-row / activated-column
      intersection.
    - Preserve all original foreground pixels.

    Key insight
    - The foreground pixels act as parity votes for their rows and columns; the
      output is the intersection grid induced by those odd votes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _mode_color(g)
    fg_mask = g != int(bg)

    active_rows = (np.sum(fg_mask, axis=1) % 2) == 1
    active_cols = (np.sum(fg_mask, axis=0) % 2) == 1

    out = g.copy()
    if bool(active_rows.any()) and bool(active_cols.any()):
        mark_mask = np.outer(active_rows, active_cols) & (g == int(bg))
        out[mark_mask] = int(MARKER_COLOR)
    return out


# --------------- generator ---------------

BACKGROUND_COLOR = 0
INPUT_COLOR_CHOICES = [1, 2, 3, 5, 6, 7, 8, 9]
GRID_HEIGHT_RANGE = (8, 16)
GRID_WIDTH_RANGE = (8, 16)
MAX_ACTIVE_PER_AXIS = 5
NOISE_RECT_RANGE = (1, 5)
GENERATION_ATTEMPTS = 400


def _toggle_rectangle(mask: np.ndarray, r1: int, r2: int, c1: int, c2: int) -> None:
    mask[r1, c1] = ~mask[r1, c1]
    mask[r1, c2] = ~mask[r1, c2]
    mask[r2, c1] = ~mask[r2, c1]
    mask[r2, c2] = ~mask[r2, c2]


def _candidate_pairs(H: int, W: int, scenario: str) -> list[tuple[int, int]]:
    row_limit = min(H - 1, MAX_ACTIVE_PER_AXIS)
    col_limit = min(W - 1, MAX_ACTIVE_PER_AXIS)
    pairs: list[tuple[int, int]] = []

    for parity in (0, 1):
        row_opts = [n for n in range(1, row_limit + 1) if n % 2 == parity]
        col_opts = [n for n in range(1, col_limit + 1) if n % 2 == parity]
        for row_count in row_opts:
            for col_count in col_opts:
                area = row_count * col_count
                if scenario == "all_new":
                    if 2 <= area <= 10:
                        pairs.append((row_count, col_count))
                else:
                    if row_count >= 2 and col_count >= 2 and 8 <= area <= 16:
                        pairs.append((row_count, col_count))
    return pairs


def _build_mask(H: int, W: int, active_rows: list[int], active_cols: list[int], scenario: str) -> np.ndarray | None:
    mask = np.zeros((H, W), dtype=bool)
    inactive_rows = [r for r in range(H) if r not in active_rows]
    inactive_cols = [c for c in range(W) if c not in active_cols]
    if not inactive_rows or not inactive_cols:
        return None

    support_row = random.choice(inactive_rows)
    support_col = random.choice(inactive_cols)

    for r in active_rows:
        mask[r, support_col] ^= True
    for c in active_cols:
        mask[support_row, c] ^= True
    if len(active_rows) % 2 == 1:
        mask[support_row, support_col] ^= True

    safe_modes = []
    if len(inactive_cols) >= 2:
        safe_modes.append("cols")
    if len(inactive_rows) >= 2:
        safe_modes.append("rows")
    if not safe_modes:
        return None

    for _ in range(random.randint(*NOISE_RECT_RANGE)):
        mode = random.choice(safe_modes)
        if mode == "cols":
            r1, r2 = sorted(random.sample(range(H), 2))
            c1, c2 = sorted(random.sample(inactive_cols, 2))
        else:
            r1, r2 = sorted(random.sample(inactive_rows, 2))
            c1, c2 = sorted(random.sample(range(W), 2))
        _toggle_rectangle(mask, r1, r2, c1, c2)

    if scenario == "mixed":
        r1, r2 = sorted(random.sample(active_rows, 2))
        c1, c2 = sorted(random.sample(active_cols, 2))
        _toggle_rectangle(mask, r1, r2, c1, c2)

    row_perm = np.random.permutation(H)
    col_perm = np.random.permutation(W)
    return mask[row_perm][:, col_perm]


def _validate_mask(mask: np.ndarray, scenario: str) -> bool:
    row_counts = np.sum(mask, axis=1)
    col_counts = np.sum(mask, axis=0)
    active_rows = row_counts % 2 == 1
    active_cols = col_counts % 2 == 1

    if not bool(active_rows.any()) or not bool(active_cols.any()):
        return False
    if bool(active_rows.all()) or bool(active_cols.all()):
        return False

    foreground_pixels = int(np.sum(mask))
    if foreground_pixels < 4:
        return False
    if foreground_pixels > (mask.shape[0] * mask.shape[1]) // 3:
        return False
    if int(np.count_nonzero(row_counts)) < 2 or int(np.count_nonzero(col_counts)) < 2:
        return False

    cross = mask[np.ix_(active_rows, active_cols)]
    occupied_cross = int(np.sum(cross))
    total_cross = int(cross.size)
    if total_cross == 0:
        return False
    if occupied_cross >= total_cross:
        return False
    if scenario == "all_new" and occupied_cross != 0:
        return False
    if scenario == "mixed" and occupied_cross == 0:
        return False
    return True


def _expected_output(inp: np.ndarray) -> np.ndarray:
    fg_mask = inp != BACKGROUND_COLOR
    active_rows = (np.sum(fg_mask, axis=1) % 2) == 1
    active_cols = (np.sum(fg_mask, axis=0) % 2) == 1
    out = inp.copy()
    out[np.outer(active_rows, active_cols) & (inp == BACKGROUND_COLOR)] = MARKER_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_HEIGHT_RANGE)
    W = random.randint(*GRID_WIDTH_RANGE)
    scenario = random.choice(["all_new", "mixed"])

    pairs = _candidate_pairs(H, W, scenario)
    if not pairs:
        return None
    row_count, col_count = random.choice(pairs)

    active_rows = sorted(random.sample(range(H), row_count))
    active_cols = sorted(random.sample(range(W), col_count))
    mask = _build_mask(H, W, active_rows, active_cols, scenario)
    if mask is None or not _validate_mask(mask, scenario):
        return None

    fg_color = random.choice(INPUT_COLOR_CHOICES)
    inp = np.full((H, W), BACKGROUND_COLOR, dtype=int)
    inp[mask] = int(fg_color)

    expected = _expected_output(inp)
    try:
        out = main(inp)
    except Exception:
        return None
    if not np.array_equal(out, expected):
        return None
    if np.any(inp == MARKER_COLOR):
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid parity-cross input")
