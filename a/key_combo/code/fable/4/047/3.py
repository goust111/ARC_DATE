from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(10))
FOREGROUND = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _mode_color(values: list[int]) -> int:
    counts = Counter(values)
    return counts.most_common(1)[0][0]


def _repair_dual_symmetry(rect: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = rect.shape
    out = rect.copy()
    changes = 0

    for r in range((h + 1) // 2):
        for c in range((w + 1) // 2):
            orbit = {
                (r, c),
                (r, w - 1 - c),
                (h - 1 - r, c),
                (h - 1 - r, w - 1 - c),
            }
            coords = sorted(orbit)
            colors = [int(rect[rr, cc]) for rr, cc in coords]
            color = _mode_color(colors)
            for rr, cc in coords:
                if int(out[rr, cc]) != color:
                    changes += 1
                    out[rr, cc] = color

    return out, changes


def _border_black_penalty(rect: np.ndarray) -> int:
    penalty = 0
    if not np.any(rect[0] != BACKGROUND):
        penalty += 1
    if not np.any(rect[-1] != BACKGROUND):
        penalty += 1
    if not np.any(rect[:, 0] != BACKGROUND):
        penalty += 1
    if not np.any(rect[:, -1] != BACKGROUND):
        penalty += 1
    return penalty


def _nonempty_rows(grid: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.any(grid != BACKGROUND, axis=1))


def _candidate_unshears(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    rows = _nonempty_rows(grid)
    assert rows.size > 0, "expected at least one non-background row"
    r0 = int(rows[0])
    active_rows = np.arange(r0, int(rows[-1]) + 1)
    assert np.array_equal(rows, active_rows), "object rows must be contiguous"

    h = int(active_rows.size)
    lefts = []
    rights = []
    row_cols = []
    for rr in active_rows:
        cols = np.flatnonzero(grid[rr] != BACKGROUND)
        assert cols.size > 0, "every active row must contain a visible pixel"
        row_cols.append(cols)
        lefts.append(int(cols.min()))
        rights.append(int(cols.max()))

    candidates: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for direction in (-1, 1):
        shifted_lefts = [lefts[i] - direction * i for i in range(h)]
        shifted_rights = [rights[i] - direction * i for i in range(h)]
        min_width = max(shifted_rights) - min(shifted_lefts) + 1
        max_width = min(int(grid.shape[1]), min_width + 4)

        for width in range(min_width, max_width + 1):
            base_lo = max(shifted_rights) - width + 1
            base_hi = min(shifted_lefts)
            for base in range(base_lo, base_hi + 1):
                starts = [base + direction * i for i in range(h)]
                if min(starts) < 0 or max(starts) + width > int(grid.shape[1]):
                    continue

                rect = np.full((h, width), BACKGROUND, dtype=int)
                valid = True
                for i, rr in enumerate(active_rows):
                    cols = row_cols[i]
                    shifted = cols - starts[i]
                    if int(shifted.min()) < 0 or int(shifted.max()) >= width:
                        valid = False
                        break
                    rect[i, shifted] = grid[rr, cols]
                if valid:
                    candidates.append(((direction, width, base, r0), rect))

    assert candidates, "failed to infer a valid hidden rectangle"
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Undo a row-by-row shear, then repair the hidden rectangle by two-axis symmetry voting.

    Input structure:
    - A black background contains one multicolor pattern occupying a sheared parallelogram.
    - Each successive row of the hidden rectangle was shifted by exactly one column left or right.
    - The hidden rectangle itself is not shown directly and may contain black pixels inside.

    Rule:
    - Infer the shear direction and row offsets that align the visible pixels back into an HxW rectangle.
    - In that recovered coordinate frame, group each cell with its horizontal, vertical, and central mirror partners.
    - Recolor any minority-error cells so every orbit agrees, then output the repaired rectangle.

    Key insight:
    - The only reliable symmetry lives in the un-sheared coordinate system, so solve by searching that latent frame and choosing the one that needs the fewest symmetry repairs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    best: tuple[tuple[int, int, int], np.ndarray] | None = None
    for _meta, rect in _candidate_unshears(g):
        repaired, changes = _repair_dual_symmetry(rect)
        score = (
            changes,
            _border_black_penalty(repaired),
            repaired.shape[0] * repaired.shape[1],
        )
        if best is None or score < best[0]:
            best = (score, repaired)

    assert best is not None
    return best[1]


def _orbit_representatives(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h // 2) for c in range(w // 2)]


def _make_symmetric_rect(h: int, w: int) -> np.ndarray:
    assert h % 2 == 0 and w % 2 == 0

    for _ in range(200):
        rect = np.full((h, w), BACKGROUND, dtype=int)
        reps = _orbit_representatives(h, w)
        n_colors = random.randint(3, 5)
        colors = [BACKGROUND] + random.sample(FOREGROUND, n_colors - 1)
        black_reps: set[tuple[int, int]] = set(random.sample(reps, random.randint(1, max(1, len(reps) // 4))))

        for r, c in reps:
            if (r, c) in black_reps:
                color = BACKGROUND
            else:
                palette = [col for col in colors if col != BACKGROUND] + [BACKGROUND]
                weights = [5] * (len(palette) - 1) + [2]
                color = random.choices(palette, weights=weights, k=1)[0]
            orbit = (
                (r, c),
                (r, w - 1 - c),
                (h - 1 - r, c),
                (h - 1 - r, w - 1 - c),
            )
            for rr, cc in orbit:
                rect[rr, cc] = color

        if len(set(map(int, np.unique(rect))) - {BACKGROUND}) < 2:
            continue
        if not np.any(rect == BACKGROUND):
            continue
        if np.any(np.sum(rect != BACKGROUND, axis=1) < 2):
            continue
        if np.any(np.sum(rect != BACKGROUND, axis=0) < 2):
            continue
        return rect

    raise RuntimeError("failed to build a non-degenerate symmetric rectangle")


def _pick_distinct_orbit_cells(h: int, w: int, count: int) -> list[tuple[int, int]]:
    reps = _orbit_representatives(h, w)
    chosen_reps = random.sample(reps, count)
    cells = []
    for r, c in chosen_reps:
        orbit = [
            (r, c),
            (r, w - 1 - c),
            (h - 1 - r, c),
            (h - 1 - r, w - 1 - c),
        ]
        cells.append(random.choice(orbit))
    return cells


def _corrupt_rect(rect: np.ndarray) -> np.ndarray:
    out = rect.copy()
    h, w = out.shape
    count = random.randint(2, min(4, len(_orbit_representatives(h, w))))
    for r, c in _pick_distinct_orbit_cells(h, w, count):
        current = int(out[r, c])
        choices = [color for color in PALETTE if color != current]
        out[r, c] = random.choice(choices)
    return out


def _shear(rect: np.ndarray, direction: int) -> np.ndarray:
    h, w = rect.shape
    out = np.zeros((h, w + h - 1), dtype=int)
    for r in range(h):
        start = r if direction == 1 else h - 1 - r
        out[r, start : start + w] = rect[r]
    return out


def _embed_with_margin(sheared: np.ndarray) -> np.ndarray:
    h, w = sheared.shape
    max_top = min(4, 30 - h)
    max_left = min(4, 30 - w)
    top = random.randint(0, max_top)
    bottom = random.randint(0, min(4, 30 - h - top))
    left = random.randint(0, max_left)
    right = random.randint(0, min(4, 30 - w - left))
    canvas = np.zeros((h + top + bottom, w + left + right), dtype=int)
    canvas[top : top + h, left : left + w] = sheared
    return canvas


def _sample_dimensions() -> tuple[int, int]:
    choices = []
    for h in range(4, 13, 2):
        for w in range(4, 13, 2):
            if w + h - 1 <= 24:
                choices.append((h, w))
    return random.choice(choices)


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h, w = _sample_dimensions()
    target = _make_symmetric_rect(h, w)
    corrupted = _corrupt_rect(target)
    direction = random.choice((-1, 1))
    sheared = _shear(corrupted, direction)
    grid = _embed_with_margin(sheared)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, target):
        return None
    return grid, target


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        attempt = _gen_attempt()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("failed to generate a valid input")
