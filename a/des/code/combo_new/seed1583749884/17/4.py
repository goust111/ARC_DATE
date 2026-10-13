from __future__ import annotations

import numpy as np
from numpy.random import default_rng


BLACK = 0
RED = 2
GREEN = 3
GRAY = 5

DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_MIN = 22
GRID_MAX = 30
PRIMITIVE_HEIGHTS = (2, 3, 4)
PRIMITIVE_WIDTHS = (2, 3, 4)
MAX_OBJECT_SIDE = 12
GEN_ATTEMPTS = 400
PRIMITIVE_ATTEMPTS = 120
INVALID_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 300


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components of a boolean mask."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_in_bbox(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    comps = _find_components(mask)
    return len(comps) == 1


def _is_tiled_copy(candidate_mask: np.ndarray, primitive_mask: np.ndarray) -> bool:
    ph, pw = primitive_mask.shape
    ch, cw = candidate_mask.shape
    if ch % ph != 0 or cw % pw != 0:
        return False
    tiled = np.tile(primitive_mask, (ch // ph, cw // pw))
    return np.array_equal(candidate_mask, tiled)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only grey objects whose shape is an exact gapless tiling of the red primitive.

    Input structure:
    - The background is black.
    - There is exactly one small red foreground object that serves as the primitive tile.
    - There are several separate grey foreground objects elsewhere on the grid.

    Rule:
    - Extract the red primitive's tight bounding-box mask.
    - For each grey object, extract its tight mask and test whether it is exactly the
      red mask repeated an integer number of times vertically and horizontally with no gaps.
    - If the grey object is such a tiling, recolor it green; otherwise erase it to black.
      Keep the red primitive unchanged.

    Key insight:
    - The red object defines a reusable shape unit; surviving grey objects are precisely
      the ones whose masks factor into a rectangular grid of identical primitive copies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    red_components = _find_components(g == RED)
    assert len(red_components) == 1, "expected exactly one red primitive"
    primitive_mask = _mask_in_bbox(red_components[0])

    out = g.copy()
    for component in _find_components(g == GRAY):
        component_mask = _mask_in_bbox(component)
        color = GREEN if _is_tiled_copy(component_mask, primitive_mask) else BLACK
        for r, c in component:
            out[r, c] = color
    return out


def _random_path(start: tuple[int, int], end: tuple[int, int], rng: np.random.Generator) -> list[tuple[int, int]]:
    path = [start]
    r, c = start
    tr, tc = end
    while (r, c) != (tr, tc):
        options: list[tuple[int, int]] = []
        if r < tr:
            options.append((r + 1, c))
        elif r > tr:
            options.append((r - 1, c))
        if c < tc:
            options.append((r, c + 1))
        elif c > tc:
            options.append((r, c - 1))
        r, c = options[int(rng.integers(len(options)))]
        path.append((r, c))
    return path


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]):
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
                    cells.append((r, c))
                    break
    return cells


def _generate_primitive_mask(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(PRIMITIVE_ATTEMPTS):
        h = int(rng.choice(PRIMITIVE_HEIGHTS))
        w = int(rng.choice(PRIMITIVE_WIDTHS))
        join = (int(rng.integers(h)), int(rng.integers(w)))
        seam_row = int(rng.integers(h))
        seam_col = int(rng.integers(w))

        cells = set(_random_path((seam_row, 0), join, rng))
        cells.update(_random_path((seam_row, w - 1), join, rng))
        cells.update(_random_path((0, seam_col), join, rng))
        cells.update(_random_path((h - 1, seam_col), join, rng))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        max_extra = min(h * w - int(mask.sum()), max(1, (h * w) // 2))
        extra = int(rng.integers(max_extra + 1))
        for _extra in range(extra):
            frontier = _frontier(mask)
            if not frontier:
                break
            r, c = frontier[int(rng.integers(len(frontier)))]
            mask[r, c] = True

        if not _is_connected(mask):
            continue
        if int(mask.sum()) < 3:
            continue
        if int(mask.sum()) == h * w and float(rng.random()) < 0.7:
            continue
        return mask
    return None


def _repeat_options(primitive_mask: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    ph, pw = primitive_mask.shape
    max_r = min(3, MAX_OBJECT_SIDE // ph)
    max_c = min(3, MAX_OBJECT_SIDE // pw)
    return {
        "horizontal": [(1, c) for c in range(2, max_c + 1)],
        "vertical": [(r, 1) for r in range(2, max_r + 1)],
        "both": [(r, c) for r in range(2, max_r + 1) for c in range(2, max_c + 1)],
    }


def _sample_valid_repeats(
    primitive_mask: np.ndarray, count: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    options = _repeat_options(primitive_mask)
    kinds = [kind for kind, values in options.items() if values]
    chosen_kinds: list[str] = []
    while len(chosen_kinds) < count:
        unused = [kind for kind in kinds if kind not in chosen_kinds]
        pool = unused if unused else kinds
        chosen_kinds.append(pool[int(rng.integers(len(pool)))])

    repeats: list[tuple[int, int]] = []
    for kind in chosen_kinds:
        pool = options[kind]
        repeats.append(pool[int(rng.integers(len(pool)))])
    rng.shuffle(repeats)
    return repeats


def _make_invalid_mask(primitive_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    all_options = []
    for values in _repeat_options(primitive_mask).values():
        all_options.extend(values)
    if not all_options:
        return None

    for _ in range(INVALID_ATTEMPTS):
        reps = all_options[int(rng.integers(len(all_options)))]
        mask = np.tile(primitive_mask, reps).copy()
        h, w = mask.shape

        addable = _frontier(mask)
        removable: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if not bool(mask[r, c]):
                    continue
                trial = mask.copy()
                trial[r, c] = False
                if trial.any() and _is_connected(trial):
                    removable.append((r, c))

        moves = []
        if addable:
            moves.append("add")
        if removable:
            moves.append("remove")
        if not moves:
            continue

        action = moves[int(rng.integers(len(moves)))]
        if action == "add":
            r, c = addable[int(rng.integers(len(addable)))]
            mask[r, c] = True
        else:
            r, c = removable[int(rng.integers(len(removable)))]
            mask[r, c] = False

        if _is_connected(mask) and not _is_tiled_copy(mask, primitive_mask):
            return mask
    return None


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    patch = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    patch[mask] = color


def _place_objects(
    objects: list[tuple[np.ndarray, int]], rng: np.random.Generator
) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    if any(mask.shape[0] + 2 > h or mask.shape[1] + 2 > w for mask, _ in objects):
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    order = sorted(
        range(len(objects)),
        key=lambda idx: objects[idx][0].shape[0] * objects[idx][0].shape[1],
        reverse=True,
    )

    for idx in order:
        mask, color = objects[idx]
        mh, mw = mask.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = int(rng.integers(1, h - mh))
            left = int(rng.integers(1, w - mw))
            r0 = top - 1
            c0 = left - 1
            r1 = top + mh + 1
            c1 = left + mw + 1
            if reserved[r0:r1, c0:c1].any():
                continue
            _paint_mask(grid, mask, top, left, color)
            reserved[r0:r1, c0:c1] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    primitive_mask = _generate_primitive_mask(rng)
    if primitive_mask is None:
        return None

    gray_count = int(rng.integers(3, 6))
    valid_count = int(rng.integers(1, gray_count))
    invalid_count = gray_count - valid_count

    valid_masks = [np.tile(primitive_mask, reps) for reps in _sample_valid_repeats(primitive_mask, valid_count, rng)]

    invalid_masks: list[np.ndarray] = []
    for _ in range(invalid_count):
        mask = _make_invalid_mask(primitive_mask, rng)
        if mask is None:
            return None
        invalid_masks.append(mask)

    objects: list[tuple[np.ndarray, int]] = [(primitive_mask, RED)]
    objects.extend((mask, GRAY) for mask in valid_masks)
    objects.extend((mask, GRAY) for mask in invalid_masks)
    rng.shuffle(objects)

    grid = _place_objects(objects, rng)
    if grid is None:
        return None
    if len(_find_components(grid == RED)) != 1:
        return None
    if len(_find_components(grid == GRAY)) != gray_count:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    green_components = _find_components(out == GREEN)
    if len(green_components) != valid_count:
        return None
    if np.count_nonzero(out == GRAY) != 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
