from __future__ import annotations

import numpy as np

from arc_utils import Color, components_mask, is_contiguous, paste, random_free_location
import random


def _residue(r: int, c: int, period: int, mode: str) -> int:
    if mode == "down_right":
        return (int(r) + int(c)) % int(period)
    if mode == "down_left":
        return (int(c) - int(r)) % int(period)
    if mode == "up_right":
        return (int(r) - int(c)) % int(period)
    raise AssertionError(f"unknown diagonal mode: {mode}")


def _diagonal_cycle(g: np.ndarray) -> tuple[str, dict[int, int]]:
    colors = set(map(int, np.unique(g))) - {Color.BLACK}
    period = len(colors)
    assert period > 1, "expected a multi-color cycle"

    matches: list[tuple[str, dict[int, int]]] = []
    for mode in ("down_right", "down_left", "up_right"):
        residue_to_color: dict[int, int] = {}
        ok = True
        for r, c in np.argwhere(g != Color.BLACK):
            key = _residue(int(r), int(c), period, mode)
            color = int(g[r, c])
            if key in residue_to_color and residue_to_color[key] != color:
                ok = False
                break
            residue_to_color[key] = color
        if ok and len(residue_to_color) == period and set(residue_to_color.values()) == colors:
            matches.append((mode, residue_to_color))

    assert len(matches) == 1, "expected one uniquely implied diagonal color cycle"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill black holes by restoring the diagonal cyclic color pattern they interrupt.

    Input structure
    - The grid is a rectangular cyclic pattern of non-black colors with some
      black pixels removed.
    - All non-black colors in the cycle remain visible somewhere in the input.
    - Each diagonal residue modulo the number of non-black colors has one color.

    Rule
    - Infer the diagonal direction and residue-to-color cycle from the visible
      non-black pixels.
    - Replace every black pixel with the color assigned to its diagonal residue.
    - Preserve every non-black pixel.

    Key insight
    - The blanks are not objects; they are missing cells in a global repeating
      diagonal tile, so their colors are determined by row + column modulo the
      cycle length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mode, residue_to_color = _diagonal_cycle(g)
    period = len(residue_to_color)

    out = g.copy()
    for r, c in np.argwhere(out == Color.BLACK):
        key = _residue(int(r), int(c), period, mode)
        out[r, c] = residue_to_color[key]
    return out

# --------------- generator (o5x) ---------------

# Canvas and diagonal tile parameters.
CANVAS_SHAPE = (16, 16)
PERIOD_BOUNDS = (5, 8)
FIRST_TILE_COLOR = 1

# Hole geometry and black-cell budget.
HOLE_COUNT_BOUNDS = (3, 4)
TOTAL_BLACK_BOUNDS = (39, 43)
HOLE_SIDE_BOUNDS = (2, 7)
HOLE_AREA_BOUNDS = (4, 29)
REMOVED_CELL_DIVISOR = 5

# Placement separation, connectivity checks, and retry budget.
HOLE_PLACEMENT_PADDING = 1
HOLE_PADDING_CONNECTIVITY = 8
HOLE_CONNECTIVITY = 4
GENERATION_RETRIES = 300


def _gen_build_hole_choices_by_area() -> dict[int, list[tuple[int, int, int]]]:
    choices: dict[int, list[tuple[int, int, int]]] = {}
    min_side, max_side = HOLE_SIDE_BOUNDS
    min_area, max_area = HOLE_AREA_BOUNDS
    for h in range(min_side, max_side + 1):
        for w in range(min_side, max_side + 1):
            cells = h * w
            for removed in range(cells // REMOVED_CELL_DIVISOR + 1):
                area = cells - removed
                if min_area <= area <= max_area:
                    choices.setdefault(area, []).append((h, w, removed))
    return choices


HOLE_CHOICES_BY_AREA = _gen_build_hole_choices_by_area()
HOLE_AREA_VALUES = sorted(HOLE_CHOICES_BY_AREA)


def _gen_split_areas(total: int, count: int, values: list[int]) -> list[int] | None:
    memo: dict[tuple[int, int], bool] = {}

    def can_make(remaining: int, slots: int) -> bool:
        key = (remaining, slots)
        if key in memo:
            return memo[key]
        if slots == 0:
            ok = remaining == 0
        elif remaining < slots * values[0] or remaining > slots * values[-1]:
            ok = False
        else:
            ok = any(can_make(remaining - v, slots - 1) for v in values)
        memo[key] = ok
        return ok

    areas: list[int] = []
    remaining = int(total)
    for slots in range(int(count), 0, -1):
        if slots == 1:
            if remaining not in values:
                return None
            areas.append(remaining)
            break
        feasible = [v for v in values if can_make(remaining - v, slots - 1)]
        if not feasible:
            return None
        area = random.choice(feasible)
        areas.append(area)
        remaining -= area
    random.shuffle(areas)
    return areas


def _gen_make_hole(area: int) -> np.ndarray | None:
    h, w, removed = random.choice(HOLE_CHOICES_BY_AREA[int(area)])
    mask = np.ones((h, w), dtype=int)
    if removed == 0:
        return mask

    removable = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r in (0, h - 1) or c in (0, w - 1))
        and not (r in (0, h - 1) and c in (0, w - 1))
    ]
    random.shuffle(removable)

    n_removed = 0
    for r, c in removable:
        if n_removed == removed:
            break
        mask[r, c] = 0
        if is_contiguous(mask, background=0, connectivity=HOLE_CONNECTIVITY):
            n_removed += 1
        else:
            mask[r, c] = 1
    if n_removed != removed:
        return None
    return mask


def _gen_output_tile(period: int) -> np.ndarray:
    rr, cc = np.indices(CANVAS_SHAPE)
    return ((rr + cc) % int(period) + FIRST_TILE_COLOR).astype(int)


def _gen_hole_masks(total_black: int, hole_count: int) -> list[np.ndarray] | None:
    areas = _gen_split_areas(total_black, hole_count, HOLE_AREA_VALUES)
    if areas is None:
        return None

    masks: list[np.ndarray] = []
    for area in areas:
        mask = _gen_make_hole(area)
        if mask is None:
            return None
        masks.append(mask)
    return masks


def _gen_place_holes(masks: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.zeros(CANVAS_SHAPE, dtype=int)
    for patch in sorted(masks, key=lambda m: int(m.sum()), reverse=True):
        try:
            r0, c0 = random_free_location(
                canvas,
                patch,
                transparent=0,
                padding=HOLE_PLACEMENT_PADDING,
                padding_connectivity=HOLE_PADDING_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(canvas, patch, r0, c0, transparent=0)
    if len(components_mask(canvas != 0, connectivity=HOLE_CONNECTIVITY)) != len(masks):
        return None
    return canvas.astype(bool)


def _gen_keep_residues_visible(hole_mask: np.ndarray, period: int) -> np.ndarray | None:
    """Unblack cells only when a residue class would otherwise disappear."""
    rr, cc = np.indices(hole_mask.shape)
    residues = (rr + cc) % int(period)
    hole_mask = hole_mask.copy()
    for residue in range(int(period)):
        if np.any((residues == residue) & ~hole_mask):
            continue
        covered = np.argwhere((residues == residue) & hole_mask)
        if covered.size == 0:
            return None
        r, c = random.choice(covered.tolist())
        hole_mask[int(r), int(c)] = False
    return hole_mask


def _gen_attempt() -> np.ndarray | None:
    period = random.randint(*PERIOD_BOUNDS)
    out = _gen_output_tile(period)

    hole_count = random.randint(*HOLE_COUNT_BOUNDS)
    total_black = random.randint(*TOTAL_BLACK_BOUNDS)
    masks = _gen_hole_masks(total_black, hole_count)
    if masks is None:
        return None
    hole_mask = _gen_place_holes(masks)
    if hole_mask is None:
        return None
    hole_mask = _gen_keep_residues_visible(hole_mask, period)
    if hole_mask is None or not (TOTAL_BLACK_BOUNDS[0] <= int(hole_mask.sum()) <= TOTAL_BLACK_BOUNDS[1]):
        return None

    inp = out.copy()
    inp[hole_mask] = Color.BLACK
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise AssertionError("failed to generate a valid c3f564a4 input")
