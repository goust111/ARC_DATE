from __future__ import annotations

import numpy as np

from arc_utils import colors_present, find_components, get_bg, is_contiguous, paste
import itertools
import random
from functools import lru_cache


SMALL_MARKER_MAX_AREA = 3


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop every large foreground object straight down to the bottom while leaving small marker objects fixed.

    Input structure:
    - The grid has a dominant background color and exactly one foreground color.
    - The foreground is split into separated 4-connected objects.
    - Objects of area at most three are stationary markers; larger objects are moving sprites.

    Rule:
    - Clear each moving sprite from its original position.
    - Translate each moving sprite vertically downward until its bbox bottom reaches the grid bottom.
    - Repaint each translated sprite in its original color; small markers stay unchanged.

    Key insight:
    - Object area separates the active falling sprites from the small fixed markers, and all active sprites follow the same vertical path.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg = int(fg_colors[0])

    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[fg],
        return_format="dict",
    )
    assert components, "expected foreground objects"

    moving = [comp for comp in components if int(comp["area"]) > SMALL_MARKER_MAX_AREA]
    stationary = [comp for comp in components if int(comp["area"]) <= SMALL_MARKER_MAX_AREA]
    assert moving, "expected at least one moving sprite"
    assert stationary, "expected at least one fixed marker"

    out = g.copy()
    for comp in moving:
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    occupied = np.zeros((H, W), dtype=bool)
    for comp in moving:
        coords = comp["coords"]
        _r0, _c0, r1, _c1 = map(int, comp["bbox"])
        dr = H - 1 - r1
        assert dr >= 0, "moving sprite bbox must already be inside the grid"

        shifted = coords + np.array([dr, 0], dtype=int)
        assert bool(np.all((0 <= shifted[:, 0]) & (shifted[:, 0] < H)))
        assert bool(np.all((0 <= shifted[:, 1]) & (shifted[:, 1] < W)))
        assert not bool(occupied[shifted[:, 0], shifted[:, 1]].any()), "dropped sprites must not overlap"
        assert bool(np.all(out[shifted[:, 0], shifted[:, 1]] == bg)), "drop path must end on background"

        occupied[shifted[:, 0], shifted[:, 1]] = True
        out[shifted[:, 0], shifted[:, 1]] = int(comp["color"])

    return out

# --------------- generator (att2) ---------------

# Canvas and palette.
BG = 7
FG_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]
GRID_SIZE_RANGE = (7, 16)
MAX_FOREGROUND_LIMIT = (1, 2)  # reserved cells, divisor
# Mover geometry and count.
MOVER_COUNT_RANGE = (1, 3)
MOVER_WIDTH_OPTIONS = (2, 3, 4)
MOVER_HEIGHT_RANGE = (2, 5)
MOVER_AREA_RANGE = (4, 10)
MOVER_MIN_DENSITY = (3, 4)
MOVER_MIN_COLUMN_STRIDE = 3
# Marker geometry and placement.
MARKER_COUNT_RANGE = (1, 2)
MARKER_MAX_AREA = 3
MARKER_BOTTOM_FRACTION = (2, 3)
MARKER_SHAPE_TEMPLATES = (
    ((1,),), ((1, 1),), ((1,), (1,)), ((1, 1, 1),), ((1,), (1,), (1,)),
    ((1, 1), (1, 0)), ((1, 1), (0, 1)), ((1, 0), (1, 1)), ((0, 1), (1, 1)),
)
# Layout bounds.
EDGE_MARGIN = 1
MIN_COLUMN_GAP = 1
INPUT_TOP_FRACTION = (3, 4)
# Retry limits.
GEN_ATTEMPTS = 300


def _ceil_fraction(n: int, numerator: int, denominator: int) -> int:
    return (int(numerator) * int(n) + int(denominator) - 1) // int(denominator)


@lru_cache(None)
def _mask_options(h: int, w: int, area: int) -> tuple[tuple[int, ...], ...]:
    total = int(h) * int(w)
    holes = total - int(area)
    out: list[tuple[int, ...]] = []
    for removed in itertools.combinations(range(total), holes):
        mask = np.ones((h, w), dtype=bool)
        for idx in removed:
            mask[idx // w, idx % w] = False
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if is_contiguous(mask.astype(int), background=0, connectivity=4):
            out.append(tuple(np.flatnonzero(mask.ravel()).tolist()))
    return tuple(out)


def _sample_mover_mask(w: int, max_area: int) -> np.ndarray | None:
    choices: list[tuple[int, int]] = []
    for h in range(MOVER_HEIGHT_RANGE[0], MOVER_HEIGHT_RANGE[1] + 1):
        cells = h * int(w)
        lo = max(MOVER_AREA_RANGE[0], _ceil_fraction(cells, *MOVER_MIN_DENSITY))
        hi = min(MOVER_AREA_RANGE[1], cells, int(max_area))
        for area in range(lo, hi + 1):
            if _mask_options(h, int(w), area):
                choices.append((h, area))
    if not choices:
        return None

    h, area = random.choice(choices)
    flat = random.choice(_mask_options(h, int(w), area))
    mask = np.zeros(h * int(w), dtype=bool)
    mask[list(flat)] = True
    return mask.reshape(h, int(w))


def _patch(mask: np.ndarray, fg: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(fg)
    return patch


def _choose_widths(n: int, count: int) -> list[int]:
    widths = [
        list(ws)
        for ws in itertools.product(MOVER_WIDTH_OPTIONS, repeat=int(count))
        if sum(ws) + MIN_COLUMN_GAP * (int(count) - 1) <= int(n) - 2 * EDGE_MARGIN
    ]
    return random.choice(widths)


def _place_columns(n: int, widths: list[int]) -> list[int]:
    base = sum(widths) + MIN_COLUMN_GAP * (len(widths) - 1)
    extra = int(n) - 2 * EDGE_MARGIN - base
    spaces = [0] * (len(widths) + 1)
    for _ in range(extra):
        spaces[random.randrange(len(spaces))] += 1

    cols: list[int] = []
    c = EDGE_MARGIN + spaces[0]
    for i, w in enumerate(widths):
        cols.append(c)
        c += int(w)
        if i + 1 < len(widths):
            c += MIN_COLUMN_GAP + spaces[i + 1]
    return cols


def _input_top(n: int, h: int) -> int | None:
    frac_top = (INPUT_TOP_FRACTION[0] * int(n)) // INPUT_TOP_FRACTION[1] - int(h)
    max_top = min(int(n) - int(h) - EDGE_MARGIN, max(EDGE_MARGIN, frac_top))
    if max_top < EDGE_MARGIN:
        return None
    return random.randint(EDGE_MARGIN, max_top)


def _marker_shapes() -> list[np.ndarray]:
    return [np.array(x, dtype=bool) for x in MARKER_SHAPE_TEMPLATES]


def _touches_existing(coords: np.ndarray, occupied: np.ndarray) -> bool:
    n = occupied.shape[0]
    for r, c in coords:
        r = int(r)
        c = int(c)
        if occupied[r, c]:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < n and 0 <= cc < n and occupied[rr, cc]:
                return True
    return False


def _marker_candidates(n: int, occupied: np.ndarray, mover_cols: set[int]) -> dict[int, list[tuple[np.ndarray, int, int]]]:
    by_area: dict[int, list[tuple[np.ndarray, int, int]]] = {}
    max_bottom = min(int(n) - EDGE_MARGIN - 1, (MARKER_BOTTOM_FRACTION[0] * int(n)) // MARKER_BOTTOM_FRACTION[1])
    for mask in _marker_shapes():
        h, w = mask.shape
        max_r0 = min(int(n) - h - EDGE_MARGIN, max_bottom - h + 1)
        if max_r0 < EDGE_MARGIN:
            continue
        for r0 in range(EDGE_MARGIN, max_r0 + 1):
            for c0 in range(EDGE_MARGIN, int(n) - w):
                coords = np.argwhere(mask) + np.array([r0, c0], dtype=int)
                if any(int(c) in mover_cols for c in coords[:, 1]):
                    continue
                if _touches_existing(coords, occupied):
                    continue
                by_area.setdefault(int(mask.sum()), []).append((mask, r0, c0))
    return by_area


def _sample_mover_masks(n: int, widths: list[int], marker_count: int, max_fg: int) -> list[np.ndarray] | None:
    reserved_for_markers = MARKER_MAX_AREA * int(marker_count)
    mover_masks: list[np.ndarray] = []
    used_area = 0
    mover_count = len(widths)
    for i, w in enumerate(widths):
        remaining = mover_count - i - 1
        max_area = int(max_fg) - reserved_for_markers - used_area - MOVER_AREA_RANGE[0] * remaining
        mask = _sample_mover_mask(w, max_area)
        if mask is None:
            return None
        mover_masks.append(mask)
        used_area += int(mask.sum())
    return mover_masks


def _place_movers(inp: np.ndarray, out: np.ndarray, mover_masks: list[np.ndarray], cols: list[int], fg: int) -> tuple[np.ndarray, set[int]] | None:
    n = inp.shape[0]
    occupied = np.zeros((n, n), dtype=bool)
    mover_cols: set[int] = set()
    for mask, c0 in zip(mover_masks, cols):
        h, w = mask.shape
        final_r0 = n - h
        input_r0 = _input_top(n, h)
        if input_r0 is None:
            return None

        patch = _patch(mask, fg)
        paste(out, patch, final_r0, c0, transparent=BG)
        paste(inp, patch, input_r0, c0, transparent=BG)

        coords = np.argwhere(mask) + np.array([input_r0, c0], dtype=int)
        occupied[coords[:, 0], coords[:, 1]] = True
        mover_cols.update(range(c0, c0 + w))
    return occupied, mover_cols


def _place_markers(inp: np.ndarray, out: np.ndarray, fg: int, occupied: np.ndarray, mover_cols: set[int], marker_count: int) -> bool:
    n = inp.shape[0]
    for _ in range(marker_count):
        candidates = _marker_candidates(n, occupied, mover_cols)
        areas = [area for area, vals in candidates.items() if vals]
        if not areas:
            return False
        area = random.choice(areas)
        mask, r0, c0 = random.choice(candidates[area])
        patch = _patch(mask, fg)
        paste(inp, patch, r0, c0, transparent=BG)
        paste(out, patch, r0, c0, transparent=BG)

        coords = np.argwhere(mask) + np.array([r0, c0], dtype=int)
        occupied[coords[:, 0], coords[:, 1]] = True
    return True


def _valid_generated(inp: np.ndarray, out: np.ndarray, fg: int, mover_count: int, marker_count: int, max_fg: int) -> bool:
    if int(np.count_nonzero(inp == fg)) > int(max_fg):
        return False
    comps = find_components(inp, bg=BG, connectivity=4, monochromatic=True, colors=[fg], return_format="dict")
    if sum(int(c["area"]) > MARKER_MAX_AREA for c in comps) != int(mover_count):
        return False
    if sum(int(c["area"]) <= MARKER_MAX_AREA for c in comps) != int(marker_count):
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(actual, out))


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1])
    fg = random.choice(FG_COLORS)
    max_fg = (n * n - MAX_FOREGROUND_LIMIT[0]) // MAX_FOREGROUND_LIMIT[1]
    max_movers = min(MOVER_COUNT_RANGE[1], (n - EDGE_MARGIN) // MOVER_MIN_COLUMN_STRIDE)
    mover_count = random.randint(MOVER_COUNT_RANGE[0], max_movers)
    marker_count = random.randint(MARKER_COUNT_RANGE[0], MARKER_COUNT_RANGE[1])

    widths = _choose_widths(n, mover_count)
    cols = _place_columns(n, widths)
    mover_masks = _sample_mover_masks(n, widths, marker_count, max_fg)
    if mover_masks is None:
        return None

    out = np.full((n, n), BG, dtype=int)
    inp = np.full((n, n), BG, dtype=int)

    placement = _place_movers(inp, out, mover_masks, cols, fg)
    if placement is None:
        return None
    occupied, mover_cols = placement
    if not _place_markers(inp, out, fg, occupied, mover_cols, marker_count):
        return None

    if not _valid_generated(inp, out, fg, mover_count, marker_count, max_fg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid 6ca952ad input")
