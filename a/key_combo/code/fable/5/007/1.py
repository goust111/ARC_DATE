from __future__ import annotations

import random

import numpy as np


# The solver is intentionally self-contained so SourceFileLoader can load this
# file without adding the shared solver directory to sys.path.  These helpers
# follow arc_utils.py's canonical (row, col) and inclusive-bbox conventions.


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _solid_rectangle_masks(grid: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Return colors whose complete occurrence set is one nontrivial rectangle."""
    g = np.asarray(grid, dtype=int)
    found: list[tuple[int, np.ndarray]] = []
    for color in np.unique(g):
        mask = g == int(color)
        box = _bbox(mask)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        height, width = r1 - r0 + 1, c1 - c0 + 1
        # Generated marker rectangles have both dimensions >= 2.  Requiring
        # this also prevents a lone pattern pixel or a line from posing as it.
        if height < 2 or width < 2:
            continue
        if int(mask.sum()) != height * width:
            continue
        if bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1])):
            found.append((int(color), mask))
    return found


def _phase_is_symmetric(
    grid: np.ndarray, marker_mask: np.ndarray, dr: int, dc: int
) -> bool:
    """Check both fixed-axis reflections after a toroidal (dr,dc) roll."""
    rolled = np.roll(grid, shift=(int(dr), int(dc)), axis=(0, 1))
    hidden = np.roll(marker_mask, shift=(int(dr), int(dc)), axis=(0, 1))

    reflected_h = np.flipud(rolled)
    hidden_h = np.flipud(hidden)
    jointly_visible = (~hidden) & (~hidden_h)
    if not bool(np.all(rolled[jointly_visible] == reflected_h[jointly_visible])):
        return False

    reflected_v = np.fliplr(rolled)
    hidden_v = np.fliplr(hidden)
    jointly_visible = (~hidden) & (~hidden_v)
    return bool(np.all(rolled[jointly_visible] == reflected_v[jointly_visible]))


def _valid_phases(grid: np.ndarray, marker_mask: np.ndarray) -> list[tuple[int, int]]:
    n = int(grid.shape[0])
    return [
        (dr, dc)
        for dr in range(n)
        for dc in range(n)
        if _phase_is_symmetric(grid, marker_mask, dr, dc)
    ]


def _restore_in_rolled_frame(
    rolled: np.ndarray, hidden: np.ndarray
) -> np.ndarray:
    """Fill hidden cells from their horizontal/vertical reflection orbits."""
    n = int(rolled.shape[0])
    restored = rolled.copy()
    for r, c in np.argwhere(hidden):
        r, c = int(r), int(c)
        orbit = {
            (r, c),
            (n - 1 - r, c),
            (r, n - 1 - c),
            (n - 1 - r, n - 1 - c),
        }
        visible_values = {
            int(rolled[rr, cc]) for rr, cc in orbit if not bool(hidden[rr, cc])
        }
        assert visible_values, "every hidden symmetry orbit needs a visible cell"
        assert len(visible_values) == 1, "visible orbit members must agree"
        restored[r, c] = visible_values.pop()
    return restored


def main(grid: np.ndarray) -> np.ndarray:
    """Infer a toroidal roll from biaxial symmetry, restore a covered rectangle, and crop out its patch.

    Input structure:
    - An odd-sized square grid contains a multicolor toroidal pattern and one
      absent-from-pattern color whose complete occurrence set is a solid rectangle.
    - After one unknown cyclic row/column roll, the unobscured pattern has both
      horizontal and vertical reflection symmetry about the rolled grid's axes.

    Rule:
    - Identify the unique nontrivial solid-color rectangle and ignore its cells
      while testing every cyclic roll.  Select the unique roll satisfying both
      reflections, restore each covered cell from a visible member of its
      four-cell reflection orbit, undo the roll, and output the rectangle crop.

    Key insight:
    - The reflection axes live in the unknown rolled coordinate frame, so their
      uniquely consistent phase must be inferred before symmetry can reveal the
      patch hidden by the marker rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square grid"
    n = int(g.shape[0])
    assert n % 2 == 1, "a unique toroidal reflection phase requires odd side length"

    markers = _solid_rectangle_masks(g)
    assert len(markers) == 1, "expected one uniquely identifiable solid rectangle"
    _marker_color, marker_mask = markers[0]
    box = _bbox(marker_mask)
    assert box is not None

    phases = _valid_phases(g, marker_mask)
    assert len(phases) == 1, "the visible pattern must determine one roll phase"
    dr, dc = phases[0]

    rolled = np.roll(g, shift=(dr, dc), axis=(0, 1))
    hidden = np.roll(marker_mask, shift=(dr, dc), axis=(0, 1))
    restored_rolled = _restore_in_rolled_frame(rolled, hidden)
    restored = np.roll(restored_rolled, shift=(-dr, -dc), axis=(0, 1))
    return _crop_bbox(restored, box)


# ---------------------------------------------------------------------------
# Generator: construct truth first, cover it, then validate every semantic clue.
# ---------------------------------------------------------------------------

_SIDE_LENGTHS = (9, 11, 13, 15, 17)
_RECT_SIDES = (2, 3, 4, 5)
_MAX_GENERATION_ATTEMPTS = 400


def _make_biaxial_pattern(n: int, colors: list[int], rng: np.random.Generator) -> np.ndarray:
    """Make a non-periodic-looking pattern by coloring reflection orbits."""
    center = n // 2
    bg_weight = float(rng.uniform(0.43, 0.57))
    ink_split = float(rng.uniform(0.38, 0.62))
    probs = np.array(
        [bg_weight, (1.0 - bg_weight) * ink_split, (1.0 - bg_weight) * (1.0 - ink_split)]
    )
    seed = rng.choice(np.asarray(colors, dtype=int), size=(center + 1, center + 1), p=probs)

    out = np.empty((n, n), dtype=int)
    for r in range(center + 1):
        for c in range(center + 1):
            value = int(seed[r, c])
            out[r, c] = value
            out[n - 1 - r, c] = value
            out[r, n - 1 - c] = value
            out[n - 1 - r, n - 1 - c] = value
    return out


def _orbit_visibility_requirements(hidden: np.ndarray) -> bool:
    """Require recoverability and rule out a fixed one-mate shortcut."""
    n = int(hidden.shape[0])
    has_visible_horizontal_mate = False
    has_visible_vertical_mate = False
    has_blocked_direct_mate = False
    for r, c in np.argwhere(hidden):
        r, c = int(r), int(c)
        h = (n - 1 - r, c)
        v = (r, n - 1 - c)
        hv = (n - 1 - r, n - 1 - c)
        if bool(hidden[h] and hidden[v] and hidden[hv]):
            return False
        has_visible_horizontal_mate |= not bool(hidden[h])
        has_visible_vertical_mate |= not bool(hidden[v])
        has_blocked_direct_mate |= bool(hidden[h] or hidden[v])
    return (
        has_visible_horizontal_mate
        and has_visible_vertical_mate
        and has_blocked_direct_mate
    )


def _choose_cover(
    picture: np.ndarray, dr: int, dc: int, rng: np.random.Generator
) -> np.ndarray | None:
    """Choose a rectangle crossing exactly one rolled symmetry axis."""
    n = int(picture.shape[0])
    center = n // 2
    candidates: list[np.ndarray] = []
    side_pairs = [(h, w) for h in _RECT_SIDES for w in _RECT_SIDES if h * w <= 20]
    rng.shuffle(side_pairs)
    for h, w in side_pairs:
        positions = [(r0, c0) for r0 in range(n - h + 1) for c0 in range(n - w + 1)]
        rng.shuffle(positions)
        for r0, c0 in positions:
            mask = np.zeros((n, n), dtype=bool)
            mask[r0 : r0 + h, c0 : c0 + w] = True
            rolled_mask = np.roll(mask, shift=(dr, dc), axis=(0, 1))
            crosses_horizontal = bool(rolled_mask[center, :].any())
            crosses_vertical = bool(rolled_mask[:, center].any())
            # Crossing both axes would cover the unrecoverable singleton at
            # their intersection; crossing neither makes every direct mate
            # visible and admits a simpler fixed-direction shortcut.
            if crosses_horizontal == crosses_vertical:
                continue
            if not _orbit_visibility_requirements(rolled_mask):
                continue
            patch = picture[r0 : r0 + h, c0 : c0 + w]
            if np.unique(patch).size < 2:
                continue
            candidates.append(mask)
        # Prefer the first randomly selected dimensions that have valid placements,
        # so rectangle shape/area remains broadly distributed.
        if candidates:
            return candidates[int(rng.integers(0, len(candidates)))]
    return None


def _single_axis_phase_count(
    grid: np.ndarray, marker_mask: np.ndarray, horizontal: bool
) -> int:
    """Count shifts satisfying only one axis; used to reject shortcut-friendly cases."""
    n = int(grid.shape[0])
    count = 0
    for dr in range(n):
        for dc in range(n):
            rolled = np.roll(grid, shift=(dr, dc), axis=(0, 1))
            hidden = np.roll(marker_mask, shift=(dr, dc), axis=(0, 1))
            reflected = np.flipud(rolled) if horizontal else np.fliplr(rolled)
            reflected_hidden = np.flipud(hidden) if horizontal else np.fliplr(hidden)
            known = (~hidden) & (~reflected_hidden)
            count += int(bool(np.all(rolled[known] == reflected[known])))
    return count


def _semantic_case_is_valid(
    inp: np.ndarray,
    expected: np.ndarray,
    marker_mask: np.ndarray,
    true_phase: tuple[int, int],
) -> bool:
    """Check ground-truth semantics, not merely main(generate_input()) closure."""
    n = int(inp.shape[0])
    dr, dc = true_phase
    if inp.shape != (n, n) or n > 30 or n % 2 != 1:
        return False
    if dr == 0 or dc == 0:
        return False

    markers = _solid_rectangle_masks(inp)
    if len(markers) != 1 or not np.array_equal(markers[0][1], marker_mask):
        return False

    phases = _valid_phases(inp, marker_mask)
    if phases != [(int(dr), int(dc))]:
        return False
    if _phase_is_symmetric(inp, marker_mask, 0, 0):
        return False

    rolled_mask = np.roll(marker_mask, shift=(dr, dc), axis=(0, 1))
    if not _orbit_visibility_requirements(rolled_mask):
        return False
    # Either axis alone must leave multiple phase choices; only their
    # conjunction is allowed to identify the full (dr,dc) phase.
    if _single_axis_phase_count(inp, marker_mask, horizontal=True) <= 1:
        return False
    if _single_axis_phase_count(inp, marker_mask, horizontal=False) <= 1:
        return False

    box = _bbox(marker_mask)
    if box is None or expected.shape == inp.shape or np.unique(expected).size < 2:
        return False
    try:
        actual = main(inp)
    except (AssertionError, ValueError, IndexError):
        return False
    return bool(np.array_equal(actual, expected))


def _generate_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.choice(_SIDE_LENGTHS))
    palette = [int(x) for x in rng.choice(np.arange(10), size=4, replace=False)]
    pattern_colors = palette[:3]
    marker_color = palette[3]

    canonical = _make_biaxial_pattern(n, pattern_colors, rng)
    # All pattern colors need repeated, spatially distributed evidence so none
    # can accidentally be mistaken for the rectangle marker.
    if any(int(np.count_nonzero(canonical == color)) < 8 for color in pattern_colors):
        return None

    dr = random.randint(1, n - 1)
    dc = random.randint(1, n - 1)
    picture = np.roll(canonical, shift=(-dr, -dc), axis=(0, 1))
    marker_mask = _choose_cover(picture, dr, dc, rng)
    if marker_mask is None:
        return None

    box = _bbox(marker_mask)
    assert box is not None
    expected = _crop_bbox(picture, box)
    inp = picture.copy()
    inp[marker_mask] = marker_color

    if not _semantic_case_is_valid(inp, expected, marker_mask, (dr, dc)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        result = _generate_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a unique, non-degenerate symmetry-completion case")
