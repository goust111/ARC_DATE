from __future__ import annotations

import random

import numpy as np


HOLE_SIZE = 3
SPAN_WIDTH = 9
PERIOD_CHOICES = (3, 4)
ACTIVE_COLORS = (1, 2)
MAX_GENERATION_ATTEMPTS = 400


def _span_end(grid: np.ndarray) -> int:
    cols = np.where(np.any(grid != 0, axis=0))[0]
    assert cols.size > 0, "expected a non-empty left motif"
    assert int(cols[0]) == 0, "expected the motif to start at the left edge"
    return int(cols[-1])


def _mirror_patch(prefix: np.ndarray, r0: int, c0: int) -> np.ndarray:
    width = prefix.shape[1]
    c1 = c0 + HOLE_SIZE - 1
    src_c0 = width - 1 - c1
    src_c1 = width - 1 - c0
    return prefix[r0 : r0 + HOLE_SIZE, src_c0 : src_c1 + 1][:, ::-1].copy()


def _hole_candidates(prefix: np.ndarray) -> list[tuple[int, int]]:
    h, w = prefix.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - HOLE_SIZE + 1):
        for c0 in range(w - HOLE_SIZE + 1):
            patch = prefix[r0 : r0 + HOLE_SIZE, c0 : c0 + HOLE_SIZE]
            if not np.all(patch == 0):
                continue
            source = _mirror_patch(prefix, r0, c0)
            if source.shape != (HOLE_SIZE, HOLE_SIZE):
                continue
            if np.all(source != 0):
                candidates.append((r0, c0))
    return candidates


def _repair_prefix(prefix: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    candidates = _hole_candidates(prefix)
    assert len(candidates) == 1, "expected exactly one mirror-repairable 3x3 hole"
    r0, c0 = candidates[0]
    repaired = prefix.copy()
    repaired[r0 : r0 + HOLE_SIZE, c0 : c0 + HOLE_SIZE] = _mirror_patch(prefix, r0, c0)
    return repaired, (r0, c0)


def _is_lr_symmetric(prefix: np.ndarray) -> bool:
    return bool(np.array_equal(prefix, prefix[:, ::-1]))


def _is_periodic(prefix: np.ndarray, period: int) -> bool:
    h, w = prefix.shape
    if period <= 0 or period > w:
        return False
    tile = prefix[:, :period]
    for c in range(w):
        if not np.array_equal(prefix[:, c], tile[:, c % period]):
            return False
    return True


def _minimal_period(prefix: np.ndarray) -> int:
    width = prefix.shape[1]
    for period in range(1, width + 1):
        if _is_periodic(prefix, period):
            return period
    raise AssertionError("expected a periodic repaired motif")


def _tile_full_width(prefix: np.ndarray, full_width: int, period: int) -> np.ndarray:
    tile = prefix[:, :period]
    out = np.empty((prefix.shape[0], full_width), dtype=int)
    for c in range(full_width):
        out[:, c] = tile[:, c % period]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the mirrored 3x3 hole so the hidden column period becomes visible, then tile that period across the full width.

    Input structure:
    - The background is black, and a left-aligned motif occupies the first several columns while the remaining right columns are all black.
    - Inside that left motif, exactly one compact 3x3 black hole lies in rows where the mirror-reflected 3x3 area is fully colored.
    - The complete left motif is horizontally mirror-symmetric and, once repaired, follows a smaller repeating column pattern.

    Rule:
    - Find the unique 3x3 all-black patch whose left-right mirror across the motif span is fully colored.
    - Copy colors from that mirrored 3x3 area to repair the hole.
    - On the repaired left motif, find the smallest column period that reproduces every visible column.
    - Repeat that period from column 0 across the entire grid width to form the output.

    Key insight:
    - The hole is placed so the true period is obscured until mirror symmetry restores the missing columns; only after repairing the symmetry can the minimal repeating unit be detected correctly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    end = _span_end(g)
    prefix = g[:, : end + 1]
    repaired, _hole = _repair_prefix(prefix)
    assert _is_lr_symmetric(repaired), "repaired motif must be left-right symmetric"

    period = _minimal_period(repaired)
    assert period in PERIOD_CHOICES, "expected the repaired motif to expose period 3 or 4"
    return _tile_full_width(repaired, g.shape[1], period)


def _row_word(period: int) -> list[int]:
    if period == 3:
        a = random.choice(ACTIVE_COLORS)
        b = ACTIVE_COLORS[0] if a == ACTIVE_COLORS[1] else ACTIVE_COLORS[1]
        return [a, b, a]

    a = random.choice(ACTIVE_COLORS)
    c = ACTIVE_COLORS[0] if a == ACTIVE_COLORS[1] else ACTIVE_COLORS[1]
    b = random.choice(ACTIVE_COLORS)
    return [a, b, c, b]


def _build_repaired_prefix(height: int, period: int) -> tuple[np.ndarray, int, int]:
    top_bg = random.randint(0, 1)
    bottom_bg = random.randint(0, 1)
    active_rows = height - top_bg - bottom_bg
    assert active_rows >= HOLE_SIZE

    prefix = np.zeros((height, SPAN_WIDTH), dtype=int)
    row_signatures: list[tuple[int, ...]] = []
    for r in range(top_bg, height - bottom_bg):
        word = _row_word(period)
        row_signatures.append(tuple(word))
        for c in range(SPAN_WIDTH):
            prefix[r, c] = word[c % period]

    # Keep the output varied and prevent all active rows from collapsing to one band.
    if len(set(row_signatures)) == 1 and active_rows >= 4:
        r = random.randint(top_bg, height - bottom_bg - 1)
        alt = _row_word(period)
        while tuple(alt) == row_signatures[0]:
            alt = _row_word(period)
        for c in range(SPAN_WIDTH):
            prefix[r, c] = alt[c % period]

    return prefix, top_bg, bottom_bg


def _hole_col_start(period: int) -> int:
    if period == 3:
        return random.choice([1, 5])
    return 6


def _build_input_from_repaired(repaired: np.ndarray, period: int, full_width: int) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((repaired.shape[0], full_width), dtype=int)
    inp[:, :SPAN_WIDTH] = repaired

    active = np.where(np.any(repaired != 0, axis=1))[0]
    assert active.size >= HOLE_SIZE
    r0 = random.randint(int(active[0]), int(active[-1]) - HOLE_SIZE + 1)
    c0 = _hole_col_start(period)
    inp[r0 : r0 + HOLE_SIZE, c0 : c0 + HOLE_SIZE] = 0
    return inp, np.array([r0, c0], dtype=int)


def _validate_semantics(inp: np.ndarray) -> bool:
    try:
        if inp.ndim != 2 or max(inp.shape) > 30:
            return False

        end = _span_end(inp)
        if end + 1 != SPAN_WIDTH:
            return False

        prefix = inp[:, :SPAN_WIDTH]
        holes = _hole_candidates(prefix)
        if len(holes) != 1:
            return False

        repaired, (r0, c0) = _repair_prefix(prefix)
        if not _is_lr_symmetric(repaired):
            return False

        period = _minimal_period(repaired)
        if period not in PERIOD_CHOICES:
            return False

        broken_period = _minimal_period(prefix)
        if broken_period == period:
            return False

        if period == 3 and c0 not in (1, 5):
            return False
        if period == 4 and c0 != 6:
            return False

        source = _mirror_patch(prefix, r0, c0)
        if np.any(source == 0):
            return False

        expected = _tile_full_width(repaired, inp.shape[1], period)
        actual = main(inp)
        if not np.array_equal(actual, expected):
            return False

        # The left motif must itself not already fill the whole grid.
        if inp.shape[1] <= SPAN_WIDTH:
            return False

        return True
    except AssertionError:
        return False


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(8, 10)
    full_width = random.randint(18, 26)
    period = random.choice(PERIOD_CHOICES)

    repaired, _top_bg, _bottom_bg = _build_repaired_prefix(height, period)
    if _minimal_period(repaired) != period:
        return None

    inp, _hole = _build_input_from_repaired(repaired, period, full_width)
    if not _validate_semantics(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
