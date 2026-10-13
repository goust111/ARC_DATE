"""ARC task 8: repair two aligned, differently colored panels across a divider."""

from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 300


def _split_panels(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Find the unique full grey row and return equally sized aligned panels."""
    full_grey_rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    assert len(full_grey_rows) == 1, "expected exactly one full grey divider row"
    divider = int(full_grey_rows[0])
    assert divider > 0 and divider + 1 < grid.shape[0], "divider must have panels on both sides"
    top = grid[:divider]
    bottom = grid[divider + 1 :]
    assert top.shape == bottom.shape, "the two aligned panels must have equal dimensions"
    return divider, top, bottom


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete both aligned panels by translating intact counterpart colors through their inferred palettes.

    Input structure:
    - One full grey divider row separates two same-sized, cell-aligned panels.
    - The panels depict the same nonperiodic structure with two one-to-one palettes.
    - Black cells are missing pixels; each panel has multiple holes, and aligned holes
      never occur on both sides at once.

    Rule:
    - At positions where both panels are intact, infer the bijection between top and
      bottom colors. Fill every top black cell from the aligned bottom color translated
      to the top palette, and fill bottom holes by the inverse translation. Preserve
      every intact pixel and the divider.

    Key insight:
    - One panel supplies the missing geometry while intact aligned pairs elsewhere
      supply the color dictionary needed to express that geometry in the local palette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30 in either dimension"
    divider, top, bottom = _split_panels(g)

    top_holes = top == BLACK
    bottom_holes = bottom == BLACK
    assert np.any(top_holes) and np.any(bottom_holes), "both panels must contain holes"
    assert not np.any(top_holes & bottom_holes), "aligned holes may not overlap"

    top_to_bottom: dict[int, int] = {}
    bottom_to_top: dict[int, int] = {}
    intact = ~top_holes & ~bottom_holes
    assert np.any(intact), "the panels need intact aligned evidence"
    for upper, lower in zip(top[intact].flat, bottom[intact].flat):
        upper, lower = int(upper), int(lower)
        assert upper != GREY and lower != GREY, "grey is reserved for the divider"
        if upper in top_to_bottom:
            assert top_to_bottom[upper] == lower, "top color has conflicting translations"
        if lower in bottom_to_top:
            assert bottom_to_top[lower] == upper, "bottom color has conflicting translations"
        top_to_bottom[upper] = lower
        bottom_to_top[lower] = upper

    # Every counterpart color needed by a hole must have an independently witnessed
    # translation. Thus no arbitrary fallback can affect the result.
    needed_bottom = set(map(int, bottom[top_holes]))
    needed_top = set(map(int, top[bottom_holes]))
    assert BLACK not in needed_bottom | needed_top
    assert needed_bottom <= bottom_to_top.keys(), "unseen bottom color needed for a top hole"
    assert needed_top <= top_to_bottom.keys(), "unseen top color needed for a bottom hole"

    out = g.copy()
    out_top = out[:divider]
    out_bottom = out[divider + 1 :]
    for color in needed_bottom:
        mask = top_holes & (bottom == color)
        out_top[mask] = bottom_to_top[color]
    for color in needed_top:
        mask = bottom_holes & (top == color)
        out_bottom[mask] = top_to_bottom[color]
    return out


def _has_simple_symmetry(pattern: np.ndarray) -> bool:
    """Reject visible global symmetries and short axial repetition."""
    variants = [np.flipud(pattern), np.fliplr(pattern), np.flip(pattern)]
    if pattern.shape[0] == pattern.shape[1]:
        variants.extend([pattern.T, np.rot90(pattern), np.rot90(pattern, 3)])
    if any(np.array_equal(pattern, candidate) for candidate in variants):
        return True
    h, w = pattern.shape
    for period in range(1, h // 2 + 1):
        if np.array_equal(pattern[period:], pattern[:-period]):
            return True
    for period in range(1, w // 2 + 1):
        if np.array_equal(pattern[:, period:], pattern[:, :-period]):
            return True
    return False


def _sample_holes(pattern: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Choose disjoint, dispersed holes that exercise several underlying colors."""
    h, w = pattern.shape
    positions = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(positions)

    def choose(forbidden: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
        candidates = [p for p in positions if p not in forbidden]
        for _ in range(80):
            chosen = set(random.sample(candidates, count))
            labels = {int(pattern[p]) for p in chosen}
            # Multiple translated colors and non-clumped damage rule out a single-color
            # fill and keep the repairs visually separable.
            if len(labels) < 2:
                continue
            if len({r for r, _ in chosen}) < 2 or len({c for _, c in chosen}) < 2:
                continue
            return chosen
        return None

    upper = choose(set())
    if upper is None:
        return None
    lower = choose(upper)
    if lower is None:
        return None
    upper_mask = np.zeros((h, w), dtype=bool)
    lower_mask = np.zeros((h, w), dtype=bool)
    for p in upper:
        upper_mask[p] = True
    for p in lower:
        lower_mask[p] = True
    return upper_mask, lower_mask


def _semantic_check(
    inp: np.ndarray,
    expected: np.ndarray,
    labels: np.ndarray,
    top_palette: tuple[int, ...],
    bottom_palette: tuple[int, ...],
    top_holes: np.ndarray,
    bottom_holes: np.ndarray,
) -> bool:
    """Check the described semantics against construction truth, not just closure."""
    h, w = labels.shape
    if inp.shape != (2 * h + 1, w) or max(inp.shape) > 30:
        return False
    full_rows = np.flatnonzero(np.all(inp == GREY, axis=1))
    if full_rows.tolist() != [h]:
        return False
    if int(top_holes.sum()) < 2 or int(bottom_holes.sum()) < 2:
        return False
    if np.any(top_holes & bottom_holes):
        return False

    perfect_top = np.take(np.asarray(top_palette), labels)
    perfect_bottom = np.take(np.asarray(bottom_palette), labels)
    constructed = np.vstack([perfect_top, np.full((1, w), GREY), perfect_bottom])
    if not np.array_equal(expected, constructed) or np.any(expected == BLACK):
        return False

    intact = ~(top_holes | bottom_holes)
    # Every dictionary entry is visibly witnessed at least twice after damage.
    if any(int(np.sum(intact & (labels == i))) < 2 for i in range(len(top_palette))):
        return False
    # Both cross-panel geometry and palette translation are indispensable: direct
    # copying the counterpart is wrong at every hole because palettes are disjoint.
    damaged = np.vstack([top_holes, np.zeros((1, w), bool), bottom_holes])
    if int(damaged.sum()) == 0 or np.array_equal(inp, expected):
        return False
    if any(color in bottom_palette for color in top_palette):
        return False
    try:
        actual = main(inp)
    except (AssertionError, ValueError, IndexError):
        return False
    if not np.array_equal(actual, expected):
        return False
    if not np.array_equal(actual[~damaged], inp[~damaged]):
        return False
    return bool(np.all(actual[damaged] != BLACK))


def _generation_attempt() -> np.ndarray | None:
    h = random.randint(5, 11)
    w = random.randint(6, 16)
    color_count = random.randint(2, 4)
    colors = random.sample(PALETTE, 2 * color_count)
    top_palette = tuple(colors[:color_count])
    bottom_palette = tuple(colors[color_count:])

    # Seed every structural label several times, then randomize the remaining cells.
    area = h * w
    flat = list(range(color_count)) * 5
    flat.extend(random.randrange(color_count) for _ in range(area - len(flat)))
    random.shuffle(flat)
    labels = np.asarray(flat, dtype=int).reshape(h, w)
    if _has_simple_symmetry(labels):
        return None

    hole_count = random.randint(2, min(8, max(2, area // 7)))
    holes = _sample_holes(labels, hole_count)
    if holes is None:
        return None
    top_holes, bottom_holes = holes

    perfect_top = np.take(np.asarray(top_palette), labels)
    perfect_bottom = np.take(np.asarray(bottom_palette), labels)
    top = perfect_top.copy()
    bottom = perfect_bottom.copy()
    top[top_holes] = BLACK
    bottom[bottom_holes] = BLACK
    inp = np.vstack([top, np.full((1, w), GREY), bottom]).astype(int)
    expected = np.vstack([perfect_top, np.full((1, w), GREY), perfect_bottom]).astype(int)

    if not _semantic_check(
        inp, expected, labels, top_palette, bottom_palette, top_holes, bottom_holes
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked instance of the dual-panel repair task."""
    for _ in range(GENERATION_ATTEMPTS):
        result = _generation_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid dual-panel color-translation input")
