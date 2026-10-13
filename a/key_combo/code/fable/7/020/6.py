from __future__ import annotations

import hashlib
import random

import numpy as np


BACKGROUND = 0
MASK_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Inclusive (r0, c0, r1, c1) bbox, following arc_utils conventions."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_4_connected(mask: np.ndarray) -> bool:
    """Return whether all true cells form one 4-connected component."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    height, width = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _wave_shift(distance_from_bottom: int) -> int:
    """Forward shift used by the bottom-anchored wave."""
    if distance_from_bottom % 2 == 0:
        return 0
    return -1 if (distance_from_bottom // 2) % 2 == 0 else 1


def _straighten(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Undo the wave while retaining gray pixels for the later completion step."""
    g = np.asarray(grid, dtype=int)
    box = _bbox(g != BACKGROUND)
    assert box is not None, "expected one nonempty foreground object"
    assert _is_4_connected(g != BACKGROUND), "expected one 4-connected foreground object"
    r0, c0, r1, c1 = box
    assert all(np.any(g[r] != BACKGROUND) for r in range(r0, r1 + 1)), (
        "every row in the object bbox must contain foreground"
    )

    out = np.zeros_like(g)
    for r in range(r0, r1 + 1):
        cols = np.flatnonzero(g[r] != BACKGROUND)
        shift = _wave_shift(r1 - r)
        restored_cols = cols - shift
        assert restored_cols.size > 0
        assert int(restored_cols.min()) >= 0 and int(restored_cols.max()) < g.shape[1], (
            "inverse row shift leaves the canvas"
        )
        out[r, restored_cols] = g[r, cols]

    straight_box = _bbox(out != BACKGROUND)
    assert straight_box is not None
    return out, straight_box


def main(grid: np.ndarray) -> np.ndarray:
    """Straighten the bottom-anchored row wave, then restore gray-masked cells by horizontal-axis symmetry.

    Input structure:
    - The background is black and one multicolor foreground object occupies every row of
      its bounding box.
    - The object's vertically mirror-symmetric original has been bent row by row into a
      one-cell horizontal wave whose phase is anchored at the bottom bounding-box row.
    - One or more nonoverlapping gray plus-shaped masks cover foreground pixels.

    Rule:
    - Starting at the bottom object row, leave alternate rows fixed and inversely shift
      the intervening rows right, left, right, left to undo the wave.
    - Only after straightening, replace every gray cell with the color at the vertically
      mirrored coordinate inside the straight object's own bounding box; keep all other
      cells black or unchanged.

    Key insight:
    - The visible symmetry is deliberately broken by the wave: its bottom-anchored phase
      must be inverted before the gray masks have valid mirror partners.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    assert np.any(g == MASK_COLOR), "expected at least one gray mask"

    straight, box = _straighten(g)
    r0, _c0, r1, _c1 = box
    gray_cells = np.argwhere(straight == MASK_COLOR)
    assert gray_cells.size > 0

    # Read every source from the still-masked straightened image, so restoration
    # cannot silently chain through another restored value.
    sources: list[tuple[int, int, int]] = []
    for r, c in gray_cells:
        mirror_r = r0 + r1 - int(r)
        source = int(straight[mirror_r, int(c)])
        assert source not in (BACKGROUND, MASK_COLOR), (
            "each gray cell needs one visible non-gray mirror source after straightening"
        )
        sources.append((int(r), int(c), source))

    out = straight.copy()
    for r, c, color in sources:
        out[r, c] = color
    assert not np.any(out == MASK_COLOR)
    return out


def _make_symmetric_object(height: int, width: int, colors: list[int]) -> np.ndarray:
    """Create a connected, nonrectangular, multicolor vertically symmetric sprite."""
    assert height % 2 == 0
    sprite = np.zeros((height, width), dtype=int)
    half = height // 2

    intervals: list[tuple[int, int]] = [(0, width - 1)]
    for _ in range(1, half):
        left = random.randint(0, 2)
        right = width - 1 - random.randint(0, 2)
        intervals.append((left, right))

    # Force at least one visible side indentation, while the top/bottom pair fixes
    # the true bbox and gives an unmistakable bottom anchor.
    if all(left == 0 and right == width - 1 for left, right in intervals[1:]):
        intervals[random.randrange(1, half)] = (1, width - 1)

    for r in range(half):
        left, right = intervals[r]
        band = r // 2
        for c in range(left, right + 1):
            color = colors[(c + c // 3 + band) % len(colors)]
            sprite[r, c] = color
            sprite[height - 1 - r, c] = color

    # Make every selected color structurally visible in a mirrored pair.
    for i, color in enumerate(colors):
        c = i % width
        sprite[0, c] = color
        sprite[-1, c] = color
    return sprite


def _paint_at_random_location(sprite: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = sprite.shape
    grid_h = random.randint(height + 4, min(30, height + 12))
    grid_w = random.randint(width + 6, min(30, width + 14))
    r0 = random.randint(2, grid_h - height - 2)
    c0 = random.randint(3, grid_w - width - 3)
    output = np.zeros((grid_h, grid_w), dtype=int)
    output[r0 : r0 + height, c0 : c0 + width] = sprite
    return output.copy(), output


def _bend(straight: np.ndarray) -> np.ndarray:
    box = _bbox(straight != BACKGROUND)
    assert box is not None
    r0, _c0, r1, _c1 = box
    bent = np.zeros_like(straight)
    for r in range(r0, r1 + 1):
        cols = np.flatnonzero(straight[r] != BACKGROUND)
        shifted = cols + _wave_shift(r1 - r)
        assert int(shifted.min()) >= 0 and int(shifted.max()) < straight.shape[1]
        bent[r, shifted] = straight[r, cols]
    return bent


def _plus_cells(r: int, c: int) -> frozenset[tuple[int, int]]:
    return frozenset(((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))


def _to_straight_coords(
    cells: set[tuple[int, int]] | frozenset[tuple[int, int]], bottom: int
) -> set[tuple[int, int]]:
    return {(r, c - _wave_shift(bottom - r)) for r, c in cells}


def _choose_cross_masks(bent: np.ndarray, count: int) -> set[tuple[int, int]] | None:
    box = _bbox(bent != BACKGROUND)
    assert box is not None
    r0, _c0, r1, _c1 = box
    height, width = bent.shape
    candidates: list[tuple[int, int, frozenset[tuple[int, int]]]] = []

    for r in range(max(1, r0), min(height - 1, r1 + 1)):
        for c in range(1, width - 1):
            cells = _plus_cells(r, c)
            if all(int(bent[rr, cc]) != BACKGROUND for rr, cc in cells):
                candidates.append((r, c, cells))
    random.shuffle(candidates)

    chosen_centers: list[tuple[int, int]] = []
    masked_bent: set[tuple[int, int]] = set()
    masked_straight: set[tuple[int, int]] = set()
    for center_r, center_c, cells in candidates:
        if cells & masked_bent:
            continue
        if any(abs(center_r - r) + abs(center_c - c) < 5 for r, c in chosen_centers):
            continue

        new_straight = _to_straight_coords(cells, r1)
        combined = masked_straight | new_straight
        mirrored = {(r0 + r1 - r, c) for r, c in combined}
        if combined & mirrored:
            continue

        chosen_centers.append((center_r, center_c))
        masked_bent.update(cells)
        masked_straight = combined
        if len(chosen_centers) == count:
            return masked_bent
    return None


def _wrong_top_anchored_result(grid: np.ndarray) -> np.ndarray | None:
    """A deliberate false shortcut used only by generator validation."""
    box = _bbox(grid != BACKGROUND)
    assert box is not None
    r0, _c0, r1, _c1 = box
    wrong = np.zeros_like(grid)
    for r in range(r0, r1 + 1):
        cols = np.flatnonzero(grid[r] != BACKGROUND)
        restored = cols - _wave_shift(r - r0)
        if restored.size == 0 or restored.min() < 0 or restored.max() >= grid.shape[1]:
            return None
        wrong[r, restored] = grid[r, cols]

    wrong_box = _bbox(wrong != BACKGROUND)
    assert wrong_box is not None
    wr0, _wc0, wr1, _wc1 = wrong_box
    original = wrong.copy()
    for r, c in np.argwhere(original == MASK_COLOR):
        source = int(original[wr0 + wr1 - int(r), int(c)])
        if source in (BACKGROUND, MASK_COLOR):
            return None
        wrong[int(r), int(c)] = source
    return wrong


def _direct_symmetry_result(grid: np.ndarray) -> np.ndarray:
    """Another deliberate false shortcut: mirror gray before undoing the wave."""
    box = _bbox(grid != BACKGROUND)
    assert box is not None
    r0, _c0, r1, _c1 = box
    direct = grid.copy()
    original = grid.copy()
    for r, c in np.argwhere(original == MASK_COLOR):
        source = int(original[r0 + r1 - int(r), int(c)])
        if source not in (BACKGROUND, MASK_COLOR):
            direct[int(r), int(c)] = source
    return direct


def _semantically_valid(
    inp: np.ndarray,
    expected: np.ndarray,
    bent_before_masks: np.ndarray,
    mask_cells: set[tuple[int, int]],
    cross_count: int,
) -> bool:
    """Check the stated rule, uniqueness conditions, and common wrong shortcuts."""
    if inp.ndim != 2 or max(inp.shape) > 30 or expected.shape != inp.shape:
        return False
    if len(mask_cells) != 5 * cross_count or cross_count < 2:
        return False
    if any(int(inp[r, c]) != MASK_COLOR for r, c in mask_cells):
        return False
    if not np.array_equal(inp != MASK_COLOR, bent_before_masks != MASK_COLOR):
        # Outside masks the bent object must be untouched.  At mask cells the
        # boolean comparison is intentionally allowed to differ.
        outside = np.ones(inp.shape, dtype=bool)
        rr, cc = zip(*mask_cells)
        outside[rr, cc] = False
        if not np.array_equal(inp[outside], bent_before_masks[outside]):
            return False

    try:
        actual = main(inp)
        straight_only, _ = _straighten(inp)
    except (AssertionError, ValueError, IndexError):
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(straight_only, expected) or not np.any(straight_only == MASK_COLOR):
        return False
    if np.array_equal(_direct_symmetry_result(inp), expected):
        return False
    wrong_top = _wrong_top_anchored_result(inp)
    if wrong_top is not None and np.array_equal(wrong_top, expected):
        return False
    if np.array_equal(bent_before_masks, expected):
        return False
    object_box = _bbox(expected != BACKGROUND)
    assert object_box is not None
    or0, _oc0, or1, _oc1 = object_box
    shifts = {_wave_shift(or1 - r) for r in range(or0, or1 + 1)}
    if shifts != {-1, 0, 1}:
        return False
    expected_box = _bbox(expected != BACKGROUND)
    if expected_box is None:
        return False
    er0, ec0, er1, ec1 = expected_box
    crop = expected[er0 : er1 + 1, ec0 : ec1 + 1]
    return bool(np.array_equal(crop, np.flipud(crop)))


_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.choice((8, 10, 12))
    width = random.randint(8, 13)
    color_count = random.randint(2, 4)
    colors = random.sample(PALETTE, color_count)
    sprite = _make_symmetric_object(height, width, colors)
    straight, expected = _paint_at_random_location(sprite)
    bent = _bend(straight)

    cross_count = random.choice((2, 2, 3))
    mask_cells = _choose_cross_masks(bent, cross_count)
    if mask_cells is None:
        return None
    inp = bent.copy()
    for r, c in mask_cells:
        inp[r, c] = MASK_COLOR

    if not _semantically_valid(inp, expected, bent, mask_cells, cross_count):
        return None
    signature = hashlib.md5(expected.tobytes() + str(expected.shape).encode()).hexdigest()
    if signature in _SEEN_OUTPUT_SIGNATURES:
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    """Generate a varied, validated example with two or three visible gray crosses."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_attempt()
        if case is None:
            continue
        inp, expected = case
        signature = hashlib.md5(expected.tobytes() + str(expected.shape).encode()).hexdigest()
        _SEEN_OUTPUT_SIGNATURES.add(signature)
        return inp
    raise RuntimeError("failed to generate a valid straighten-then-complete input")
