from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _add_arc_utils_path() -> None:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if candidate.is_file():
            utils_dir = str(candidate.parent)
            if utils_dir not in sys.path:
                sys.path.append(utils_dir)
            return
    raise ImportError("could not locate new/solvers/arc_utils.py")


_add_arc_utils_path()

from arc_utils import Color, crop_bbox, find_components, paste  # noqa: E402


GRID_SIZE = 12
BG = int(Color.BLACK)
COUNTER_COLOR = int(Color.GREY)
CROSS_COLOR = int(Color.RED)
SPRITE_COLORS = [int(c) for c in (Color.BLUE, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)]
MAX_GEN_ATTEMPTS = 500

_GEN_RNG = np.random.default_rng()
_GEN_CALL_COUNT = 0


_BASE_MASKS = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1], [1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int).tolist())


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _all_mask_variants() -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for base in _BASE_MASKS:
        for k in range(4):
            rot = np.rot90(base, k)
            for candidate in (rot, np.fliplr(rot)):
                norm = _normalize_mask(candidate.astype(bool))
                key = _mask_key(norm)
                if key in seen:
                    continue
                seen.add(key)
                variants.append(norm)
    return variants


_SPRITE_MASKS = _all_mask_variants()


def _locate_cross(grid: np.ndarray) -> tuple[int, int, int]:
    candidates = []
    for color in np.unique(grid):
        col = int(color)
        if col in (BG, COUNTER_COLOR):
            continue
        rows = np.flatnonzero(np.all(grid == col, axis=1))
        cols = np.flatnonzero(np.all(grid == col, axis=0))
        if len(rows) == 1 and len(cols) == 1:
            candidates.append((col, int(rows[0]), int(cols[0])))
    assert len(candidates) == 1, "expected exactly one full-row/full-column cross"
    return candidates[0]


def _scene_without_guides(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    clean = g.copy()
    clean[clean[:, -1] == COUNTER_COLOR, -1] = BG
    cross = _locate_cross(clean)
    _, row, col = cross
    clean[row, :] = BG
    clean[:, col] = BG
    return clean, cross


def _selected_components(clean: np.ndarray, row_cut: int, col_cut: int) -> list[dict]:
    comps = find_components(
        clean,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    selected = []
    for comp in comps:
        coords = comp["coords"]
        if np.all(coords[:, 0] > int(row_cut)) and np.all(coords[:, 1] > int(col_cut)):
            selected.append(comp)
    selected.sort(key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))
    return selected


def _pack_selected(clean: np.ndarray, selected: list[dict]) -> np.ndarray:
    if not selected:
        return np.zeros((1, 1), dtype=int)

    crops = [crop_bbox(clean, tuple(map(int, comp["bbox"]))) for comp in selected]
    max_h = max(int(crop.shape[0]) for crop in crops)
    total_w = sum(int(crop.shape[1]) for crop in crops)
    out = np.full((max_h, total_w), BG, dtype=int)

    cursor = 0
    for crop in crops:
        paste(out, crop, 0, cursor, transparent=None)
        cursor += int(crop.shape[1])
    return out


def _solve_with_shift(grid: np.ndarray, *, row_shift: int, col_shift: int) -> tuple[np.ndarray, list[dict], np.ndarray]:
    clean, (_, row, col) = _scene_without_guides(grid)
    selected = _selected_components(clean, row + int(row_shift), col - int(col_shift))
    return clean, selected, _pack_selected(clean, selected)


def _selection_signature(grid: np.ndarray, *, row_shift: int, col_shift: int) -> tuple:
    clean, selected, _ = _solve_with_shift(grid, row_shift=row_shift, col_shift=col_shift)
    sig = []
    for comp in selected:
        bbox = tuple(map(int, comp["bbox"]))
        crop = crop_bbox(clean, bbox)
        sig.append((bbox[0], bbox[1], _mask_key(crop != BG), tuple(int(v) for v in crop[crop != BG])))
    return tuple(sig)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift the cross by the grey counter, then crop and concatenate the sprites fully inside the shifted lower-right quadrant.

    Input structure:
    - A 12x12 black grid contains one full horizontal row and one full vertical column of the same non-black, non-grey color, forming a cross.
    - The top of the rightmost column contains 1 to 3 grey counter pixels.
    - Several other small monochrome 4-connected sprites appear away from the cross.

    Rule:
    - Count the grey pixels in the last column as N, then erase those grey pixels.
    - Find the cross row R and cross column C, and use the shifted reference point (R + N, C - N).
    - Ignore the cross itself, keep only sprites whose every pixel satisfies r > R + N and c > C - N.
    - Crop each selected sprite to its bbox, sort them by original left-to-right position, and concatenate the crops horizontally with top alignment.

    Key insight:
    - The cross is only a movable partition tool: the grey counter changes which sprites fall into the lower-right quadrant, and only those cropped sprites appear in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    n = int(np.count_nonzero(g[:, -1] == COUNTER_COLOR))
    _clean, _selected, out = _solve_with_shift(g, row_shift=n, col_shift=n)
    return out


def _box_positions(box: tuple[int, int, int, int], mask: np.ndarray, *, top: int | None = None, left: int | None = None) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    h, w = map(int, mask.shape)
    if r1 - r0 + 1 < h or c1 - c0 + 1 < w:
        return []
    rows = [int(top)] if top is not None else list(range(r0, r1 - h + 2))
    cols = [int(left)] if left is not None else list(range(c0, c1 - w + 2))
    return [(r, c) for r in rows for c in cols if r0 <= r <= r1 - h + 1 and c0 <= c <= c1 - w + 1]


def _fits(grid: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = map(int, mask.shape)
    patch = grid[int(top) : int(top) + h, int(left) : int(left) + w]
    return np.all(patch[mask] == BG)


def _paint_sprite(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    paste(grid, patch, int(top), int(left), transparent=BG)


def _candidate_masks(max_h: int, max_w: int, *, style: str = "any") -> list[np.ndarray]:
    candidates = []
    for mask in _SPRITE_MASKS:
        h, w = map(int, mask.shape)
        if h > max_h or w > max_w:
            continue
        area = int(mask.sum())
        if not (2 <= area <= 5):
            continue
        if style == "tall" and not (w <= 2 and h >= 2 and h >= w):
            continue
        if style == "wide" and not (h <= 2 and w >= 2 and w >= h):
            continue
        candidates.append(mask)
    return candidates


def _place_random_mask(
    grid: np.ndarray,
    color: int,
    box: tuple[int, int, int, int],
    rng: np.random.Generator,
    *,
    style: str = "any",
    top: int | None = None,
    left: int | None = None,
) -> bool:
    r0, c0, r1, c1 = map(int, box)
    masks = _candidate_masks(r1 - r0 + 1, c1 - c0 + 1, style=style)
    if not masks:
        return False

    order = list(rng.permutation(len(masks)))
    for idx in order:
        mask = masks[int(idx)]
        positions = _box_positions(box, mask, top=top, left=left)
        if not positions:
            continue
        rng.shuffle(positions)
        for rr, cc in positions:
            if _fits(grid, mask, rr, cc):
                _paint_sprite(grid, mask, color, rr, cc)
                return True
    return False


def _place_bottom_row_bar(
    grid: np.ndarray,
    color: int,
    box: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> bool:
    r0, c0, r1, c1 = map(int, box)
    width = c1 - c0 + 1
    if width < 2:
        return False
    lengths = [L for L in (2, 3, 4) if L <= width]
    rng.shuffle(lengths)
    for length in lengths:
        starts = list(range(c0, c1 - length + 2))
        rng.shuffle(starts)
        for start in starts:
            mask = np.ones((1, length), dtype=bool)
            if _fits(grid, mask, r1, start):
                _paint_sprite(grid, mask, color, r1, start)
                return True
    return False


def _preferred_counter_value() -> int:
    global _GEN_CALL_COUNT
    seq = (2, 3, 1)
    value = seq[_GEN_CALL_COUNT % len(seq)]
    _GEN_CALL_COUNT += 1
    return int(value)


def _choose_scene_params(rng: np.random.Generator, n: int) -> tuple[int, int, int, int, int, int]:
    row_shifted_min = max(4, 2 * n)
    row_shifted = int(rng.integers(row_shifted_min, 8))
    col_shifted = int(rng.integers(1, 5))
    row = row_shifted - n
    col = col_shifted + n
    selected_count = int(rng.integers(2, 4))
    extra_outside = int(rng.integers(1, 3))
    return row, col, row_shifted, col_shifted, selected_count, extra_outside


def _build_candidate(n: int) -> np.ndarray | None:
    rng = _GEN_RNG
    row, col, row_shifted, col_shifted, selected_count, outside_extra = _choose_scene_params(rng, n)

    grid = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    grid[:n, -1] = COUNTER_COLOR
    grid[row, :] = CROSS_COLOR
    grid[:, col] = CROSS_COLOR

    total_needed = selected_count + 1 + outside_extra
    if total_needed > len(SPRITE_COLORS):
        return None
    colors = [int(c) for c in rng.choice(SPRITE_COLORS, size=total_needed, replace=False)]
    color_idx = 0

    if n >= 2:
        strip_box = (row_shifted + 1, col_shifted + 1, GRID_SIZE - 1, col - 1)
        ok = _place_random_mask(
            grid,
            colors[color_idx],
            strip_box,
            rng,
            style="tall",
            left=col_shifted + 1,
        )
        if not ok:
            return None
        color_idx += 1
        deep_needed = selected_count - 1
    else:
        deep_needed = selected_count

    deep_box = (row_shifted + 1, col + 1, GRID_SIZE - 1, GRID_SIZE - 1)
    if deep_needed <= 0:
        return None
    ok = _place_random_mask(
        grid,
        colors[color_idx],
        deep_box,
        rng,
        style="any",
        top=row_shifted + 1,
    )
    if not ok:
        return None
    color_idx += 1

    for _ in range(deep_needed - 1):
        ok = _place_random_mask(grid, colors[color_idx], deep_box, rng, style="any")
        if not ok:
            return None
        color_idx += 1

    row_strip_box = (row + 1, col + 1, row_shifted, GRID_SIZE - 1)
    if not _place_bottom_row_bar(grid, colors[color_idx], row_strip_box, rng):
        return None
    color_idx += 1

    zones = [
        (0, 0, row - 1, col - 1),
        (0, col + 1, row - 1, GRID_SIZE - 2),
        (row_shifted + 1, 0, GRID_SIZE - 1, col_shifted),
        (row + 1, 0, row_shifted, col_shifted),
    ]
    zone_order = list(rng.permutation(len(zones)))
    placed_extra = 0
    for zone_idx in zone_order:
        if placed_extra >= outside_extra:
            break
        zone = zones[int(zone_idx)]
        zr0, zc0, zr1, zc1 = zone
        if zr0 > zr1 or zc0 > zc1:
            continue
        ok = _place_random_mask(grid, colors[color_idx], zone, rng, style="any")
        if not ok:
            continue
        color_idx += 1
        placed_extra += 1

    if placed_extra != outside_extra:
        return None
    return grid


def _semantic_checks(grid: np.ndarray, n: int) -> bool:
    if grid is None or grid.shape != (GRID_SIZE, GRID_SIZE):
        return False
    if not np.all(grid[:n, -1] == COUNTER_COLOR):
        return False
    if np.any(grid[n:, -1] == COUNTER_COLOR):
        return False

    try:
        _, cross_row, cross_col = _locate_cross(grid)
    except AssertionError:
        return False

    if not (0 < cross_row < GRID_SIZE - 1 and 0 < cross_col < GRID_SIZE - 1):
        return False

    clean, selected, expected = _solve_with_shift(grid, row_shift=n, col_shift=n)
    comps = find_components(clean, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    if not (4 <= len(comps) <= 6):
        return False
    if len({int(comp["color"]) for comp in comps}) != len(comps):
        return False
    if len(selected) not in (2, 3):
        return False

    starts = [int(comp["bbox"][1]) for comp in selected]
    if len(set(starts)) != len(starts):
        return False

    if not np.array_equal(main(grid), expected):
        return False

    base_sig = _selection_signature(grid, row_shift=n, col_shift=n)
    if not base_sig:
        return False

    no_shift_sig = _selection_signature(grid, row_shift=0, col_shift=0)
    if no_shift_sig == base_sig:
        return False

    neighbor_counts = [k for k in {0, n - 1, n + 1} if 0 <= k <= 3 and k != n]
    if not neighbor_counts:
        return False
    for k in neighbor_counts:
        if _selection_signature(grid, row_shift=k, col_shift=k) == base_sig:
            return False

    if n >= 2:
        if _selection_signature(grid, row_shift=n, col_shift=0) == base_sig:
            return False
        if _selection_signature(grid, row_shift=0, col_shift=n) == base_sig:
            return False

    return True


def generate_input() -> np.ndarray:
    desired_n = _preferred_counter_value()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _build_candidate(desired_n)
        if grid is not None and _semantic_checks(grid, desired_n):
            return grid
    raise RuntimeError("failed to generate a valid counter-shifted quadrant-packing ARC input")
