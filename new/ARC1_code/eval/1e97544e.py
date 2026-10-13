from __future__ import annotations

import numpy as np

from arc_utils import colors_present, paste as _gen_paste, random_free_location as _gen_random_free_location
import random as _gen_random


def _phase_index(row: int, col: int, period: int) -> int:
    if row <= col:
        return col % period
    return (col + 1) % period


def _infer_blank_and_cycle(g: np.ndarray) -> tuple[int, tuple[int, ...]]:
    colors = colors_present(g)
    candidates: list[tuple[int, tuple[int, ...]]] = []

    for blank in colors:
        foreground = set(colors) - {int(blank)}
        period = len(foreground)
        if period < 2:
            continue

        cycle: list[int | None] = [None] * period
        ok = True
        for row in range(g.shape[0]):
            for col in range(g.shape[1]):
                value = int(g[row, col])
                if value == blank:
                    continue
                idx = _phase_index(row, col, period)
                if cycle[idx] is None:
                    cycle[idx] = value
                elif cycle[idx] != value:
                    ok = False
                    break
            if not ok:
                break

        if ok and all(value is not None for value in cycle):
            inferred = tuple(int(value) for value in cycle if value is not None)
            if len(set(inferred)) == period and set(inferred) == foreground:
                candidates.append((int(blank), inferred))

    assert len(candidates) == 1, "expected a unique blank color and cyclic pattern"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill blank gaps in a triangular cyclic color table.

    Input structure:
    - The grid is a square table with one blank color masking missing cells.
    - The visible non-blank pixels follow a repeated color cycle by column.
    - Cells on or above the main diagonal use the column's cycle phase; cells
      below the diagonal use the next cycle phase.

    Rule:
    - Infer the unique blank color and the ordered color cycle from all visible
      pixels.
    - Reconstruct the complete table with phase col mod n above/on the diagonal
      and phase col+1 mod n below it.
    - Replace only blank pixels with their reconstructed colors.

    Key insight:
    - The apparent holes are not independent objects; they are missing samples
      from one global periodic pattern with a diagonal phase shift.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == width, "expected a square cyclic table"

    blank, cycle = _infer_blank_and_cycle(g)
    assert np.any(g == blank), "expected at least one blank pixel to fill"
    period = len(cycle)
    cycle_array = np.asarray(cycle, dtype=int)

    rows, cols = np.indices(g.shape)
    phase = np.where(rows <= cols, cols % period, (cols + 1) % period)
    full_pattern = cycle_array[phase]
    assert np.array_equal(g[g != blank], full_pattern[g != blank]), "visible pixels must match the inferred table"

    out = g.copy()
    out[g == blank] = full_pattern[g == blank]
    return out

# --------------- generator (o5x) ---------------

# Grid and cyclic palette bounds.
GRID_SIZE, BLANK_COLOR = 23, 0
PERIOD_MIN, PERIOD_MAX = 6, 9

# Blank region counts and sizes.
REGION_COUNT_MIN, REGION_COUNT_MAX = 3, 5
TOTAL_BLANK_MIN, TOTAL_BLANK_MAX = 50, 73
PATCH_SIDE_MIN, PATCH_SIDE_MAX = 2, 7
L_PATCH_SIDE_MIN, L_PATCH_SIDE_MAX = 3, 7
PATCH_AREA_MIN, PATCH_AREA_MAX = 4, 43

# L-shape construction and sampling weights.
L_BAR_MIN, L_BAR_STOP = 2, 4
L_ROTATIONS = range(4)
RECT_PATCH_PROB = 0.78

# Placement and retry controls.
PLACEMENT_BORDER, PLACEMENT_PADDING, PLACEMENT_CONNECTIVITY = 0, 1, 4
MAX_GENERATION_ATTEMPTS = 300


def _gen_l_patch(height: int, width: int, bar_h: int, bar_w: int, rot: int) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    patch[:, :bar_w] = 1
    patch[-bar_h:, :] = 1
    return np.rot90(patch, int(rot))


def _gen_patch_library() -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]]]:
    rects: dict[int, list[np.ndarray]] = {}
    l_shapes: dict[int, list[np.ndarray]] = {}
    for height in range(PATCH_SIDE_MIN, PATCH_SIDE_MAX + 1):
        for width in range(PATCH_SIDE_MIN, PATCH_SIDE_MAX + 1):
            area = height * width
            if area <= PATCH_AREA_MAX:
                rects.setdefault(area, []).append(np.ones((height, width), dtype=int))

    for height in range(L_PATCH_SIDE_MIN, L_PATCH_SIDE_MAX + 1):
        for width in range(L_PATCH_SIDE_MIN, L_PATCH_SIDE_MAX + 1):
            for bar_h in range(L_BAR_MIN, min(L_BAR_STOP, height)):
                for bar_w in range(L_BAR_MIN, min(L_BAR_STOP, width)):
                    for rot in L_ROTATIONS:
                        patch = _gen_l_patch(height, width, bar_h, bar_w, rot)
                        area = int(np.sum(patch))
                        if PATCH_AREA_MIN <= area <= PATCH_AREA_MAX:
                            l_shapes.setdefault(area, []).append(patch)
    return rects, l_shapes


_RECT_PATCHES, _L_SHAPE_PATCHES = _gen_patch_library()
_PATCH_AREAS = sorted(set(_RECT_PATCHES) | set(_L_SHAPE_PATCHES))


def _gen_pick_region_patches(count: int, total: int) -> list[np.ndarray] | None:
    memo: dict[tuple[int, int], bool] = {}

    def feasible(slots: int, remaining: int) -> bool:
        key = (slots, remaining)
        if key in memo:
            return memo[key]
        if slots == 0:
            memo[key] = remaining == 0
            return memo[key]
        if remaining < PATCH_AREA_MIN * slots or remaining > PATCH_AREA_MAX * slots:
            memo[key] = False
            return False
        ok = any(feasible(slots - 1, remaining - area) for area in _PATCH_AREAS)
        memo[key] = ok
        return ok

    if not feasible(count, total):
        return None

    chosen: list[np.ndarray] = []
    remaining = int(total)
    for slots in range(count, 0, -1):
        possible = [area for area in _PATCH_AREAS if feasible(slots - 1, remaining - area)]
        if not possible:
            return None
        area = _gen_random.choice(possible)
        use_rect = area in _RECT_PATCHES and (
            area not in _L_SHAPE_PATCHES or _gen_random.random() < RECT_PATCH_PROB
        )
        choices = _RECT_PATCHES[area] if use_rect else _L_SHAPE_PATCHES[area]
        chosen.append(_gen_random.choice(choices).copy())
        remaining -= area

    _gen_random.shuffle(chosen)
    return chosen


def _gen_full_pattern(period: int) -> tuple[np.ndarray, np.ndarray]:
    offset = _gen_random.randrange(period)
    cycle = np.roll(np.arange(1, period + 1, dtype=int), offset)
    rows, cols = np.indices((GRID_SIZE, GRID_SIZE))
    phase = np.where(rows <= cols, cols % period, (cols + 1) % period)
    return cycle[phase], phase


def _gen_blank_mask(patches: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for patch in patches:
        try:
            row, col = _gen_random_free_location(
                canvas,
                patch,
                transparent=BLANK_COLOR,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None
        _gen_paste(canvas, patch, row, col, transparent=BLANK_COLOR)
    return canvas.astype(bool)


def _gen_attempt() -> np.ndarray | None:
    period = _gen_random.randint(PERIOD_MIN, PERIOD_MAX)
    output, phase = _gen_full_pattern(period)

    region_count = _gen_random.randint(REGION_COUNT_MIN, REGION_COUNT_MAX)
    total_blank = _gen_random.randint(TOTAL_BLANK_MIN, TOTAL_BLANK_MAX)
    patches = _gen_pick_region_patches(region_count, total_blank)
    if patches is None:
        return None

    mask = _gen_blank_mask(patches)
    if mask is None:
        return None
    if int(np.sum(mask)) != total_blank:
        return None
    for idx in range(period):
        if not np.any((phase == idx) & ~mask):
            return None

    inp = output.copy()
    inp[mask] = BLANK_COLOR
    try:
        if not np.array_equal(main(inp), output):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid 1e97544e input")
