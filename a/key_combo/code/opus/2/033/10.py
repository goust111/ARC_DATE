from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREEN = 3
GRAY = 5
PINK = 6

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def bbox_w(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid)[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int,
    clip: bool = True,
) -> None:
    h, w = patch.shape
    hh, ww = dst.shape
    r0 = int(r0)
    c0 = int(c0)
    if not clip and (r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww):
        raise ValueError("patch out of bounds")
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(hh, r0 + h)
    cc1 = min(ww, c0 + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    pr0 = rr0 - r0
    pc0 = cc0 - c0
    view = patch[pr0 : pr0 + (rr1 - rr0), pc0 : pc0 + (cc1 - cc0)]
    mask = view != int(transparent)
    dst_view = dst[rr0:rr1, cc0:cc1]
    dst_view[mask] = view[mask]


def _find_components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    g = np.asarray(grid)
    seen = np.zeros(g.shape, dtype=bool)
    comps: list[dict] = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if seen[r, c] or int(g[r, c]) != int(color):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(g.shape, nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _component_touches_marker(comp: dict, marker: Coord) -> bool:
    mr, mc = marker
    cells = {tuple(map(int, rc)) for rc in comp["coords"]}
    return any((nr, nc) in cells for nr, nc in neighbors4(mr, mc))


def _seed_from_component(grid: np.ndarray, comp: dict, color: int, bg: int) -> np.ndarray:
    box = comp["bbox"]
    mask = np.zeros_like(grid, dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r), int(c)] = True
    local = crop_bbox(mask, box)
    return np.where(local, int(color), int(bg)).astype(int)


def _recolor_non_bg(patch: np.ndarray, bg: int, color: int) -> np.ndarray:
    out = np.full(patch.shape, int(bg), dtype=int)
    out[patch != int(bg)] = int(color)
    return out


def _iter_stamp_jobs(grid_shape: tuple[int, int], seed: np.ndarray, r0: int, c0: int, bg: int):
    h, w = seed.shape
    period = int(h + w)
    rotated = _recolor_non_bg(np.rot90(seed, k=3), bg, PINK)
    limit = int(grid_shape[0] + grid_shape[1] + period + 2)
    for diag in (1, -1):
        for d in range(-limit, limit + 1):
            patch = seed if d == 0 or abs(d) % 2 == 0 else rotated
            rr = int(r0 + d * period)
            cc = int(c0 + diag * d * period)
            yield diag, d, patch, rr, cc


def _intersects_canvas(grid_shape: tuple[int, int], patch_shape: tuple[int, int], r0: int, c0: int) -> bool:
    h, w = patch_shape
    return max(0, int(r0)) < min(int(grid_shape[0]), int(r0) + int(h)) and max(0, int(c0)) < min(
        int(grid_shape[1]), int(c0) + int(w)
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the grey-selected irregular green object along both diagonals with alternating rotated recolored copies.

    Input structure:
    - A black background contains 2-3 separated small green 4-connected objects.
    - Exactly one grey marker pixel is 4-adjacent to exactly one green object.
    - One pink single pixel appears elsewhere only as the alternate output color cue.

    Rule:
    - Find the green object touching the grey marker, crop that object to its bounding box, and record the box top-left as (r0, c0).
    - Erase the grey marker, the pink cue pixel, and all unselected green objects.
    - Stamp the cropped seed on both diagonals at top-lefts (r0 + d*(H+W), c0 + d*(H+W)) and (r0 + d*(H+W), c0 - d*(H+W)) for every integer d, clipping at the grid border.
    - Use the original green seed for d = 0 and other even |d|, and use the seed rotated 90 degrees clockwise and recolored pink for odd |d|.

    Key insight:
    - The grey marker chooses which irregular object becomes the seed, and that seed's bounding-box height plus width sets the diagonal zigzag period.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    gray_cells = np.argwhere(g == GRAY)
    assert gray_cells.shape[0] == 1, "expected one grey marker"
    marker = tuple(map(int, gray_cells[0]))

    green_components = _find_components_of_color(g, GREEN)
    assert 2 <= len(green_components) <= 3, "expected 2-3 green objects"

    selected = [comp for comp in green_components if _component_touches_marker(comp, marker)]
    assert len(selected) == 1, "grey marker must select exactly one green object"
    comp = selected[0]

    r0, c0, _r1, _c1 = comp["bbox"]
    seed = _seed_from_component(g, comp, GREEN, bg)
    out = np.full(g.shape, bg, dtype=int)

    for _diag, _d, patch, rr, cc in _iter_stamp_jobs(g.shape, seed, r0, c0, bg):
        paste(out, patch, rr, cc, transparent=bg, clip=True)
    return out


def _mask_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    box = bbox(mask)
    assert box is not None
    return crop_bbox(mask.astype(int), box).astype(bool)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    norm = _normalize_mask(mask)
    return norm.shape, norm.astype(np.uint8).tobytes()


def _unique_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for k in range(4):
        rotated = np.rot90(mask, k=k)
        for flipped in (rotated, np.fliplr(rotated)):
            norm = _normalize_mask(flipped)
            variants[_mask_key(norm)] = norm
    return list(variants.values())


_BASE_SHAPES = [
    _mask_from_rows(("10", "11")),
    _mask_from_rows(("010", "111")),
    _mask_from_rows(("10", "10", "11")),
    _mask_from_rows(("011", "110")),
    _mask_from_rows(("010", "110", "100")),
    _mask_from_rows(("100", "111", "001")),
    _mask_from_rows(("110", "111")),
    _mask_from_rows(("1000", "1111")),
    _mask_from_rows(("1110", "0111")),
    _mask_from_rows(("110", "111", "001")),
    _mask_from_rows(("100", "111", "100")),
]

_ALL_SHAPES: list[np.ndarray] = []
_SELECTABLE_SHAPES: list[np.ndarray] = []
for _base in _BASE_SHAPES:
    for _variant in _unique_variants(_base):
        if not any(np.array_equal(_variant, old) for old in _ALL_SHAPES):
            _ALL_SHAPES.append(_variant)
        if not np.array_equal(_variant, np.rot90(_variant, k=3)) and not any(
            np.array_equal(_variant, old) for old in _SELECTABLE_SHAPES
        ):
            _SELECTABLE_SHAPES.append(_variant)


GRID_MIN = 14
GRID_MAX = 18
GEN_RETRIES = 500
MAX_OTHER_PLACE_TRIES = 80
OBJECT_COUNTS = (2, 3)
MIN_VISIBLE_UNIQUE = 5
MIN_VISIBLE_PER_DIAG = 3
MAX_FILLED_RATIO = 0.35


def _placed_cells(mask: np.ndarray, r0: int, c0: int) -> set[Coord]:
    cells = set()
    for r, c in np.argwhere(mask):
        cells.add((int(r0) + int(r), int(c0) + int(c)))
    return cells


def _adjacent_background_cells(grid_shape: tuple[int, int], cells: set[Coord]) -> list[Coord]:
    out = set()
    for r, c in cells:
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid_shape, nr, nc) and (nr, nc) not in cells:
                out.add((int(nr), int(nc)))
    return list(out)


def _count_visible_stamps(grid_shape: tuple[int, int], seed: np.ndarray, r0: int, c0: int, bg: int) -> tuple[int, int, int]:
    unique = set()
    diag_counts = {1: 0, -1: 0}
    for diag, d, patch, rr, cc in _iter_stamp_jobs(grid_shape, seed, r0, c0, bg):
        if not _intersects_canvas(grid_shape, patch.shape, rr, cc):
            continue
        diag_counts[diag] += 1
        unique.add((int(rr), int(cc), patch.shape, 0 if d == 0 or abs(d) % 2 == 0 else 1))
    return len(unique), diag_counts[1], diag_counts[-1]


def _can_place_object(
    mask: np.ndarray,
    r0: int,
    c0: int,
    grid_shape: tuple[int, int],
    protected_cells: set[Coord],
) -> bool:
    cells = _placed_cells(mask, r0, c0)
    if not cells:
        return False
    for r, c in cells:
        if not in_bounds(grid_shape, r, c):
            return False
        if (r, c) in protected_cells:
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in protected_cells:
                    return False
    return True


def _paint_cells(grid: np.ndarray, cells: set[Coord], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _choose_selected_placement(grid_shape: tuple[int, int], seed_mask: np.ndarray) -> tuple[int, int, Coord] | None:
    h, w = seed_mask.shape
    candidates: list[tuple[int, int, Coord]] = []
    seed = np.where(seed_mask, GREEN, BLACK).astype(int)
    for r0 in range(grid_shape[0] - h + 1):
        for c0 in range(grid_shape[1] - w + 1):
            cells = _placed_cells(seed_mask, r0, c0)
            marker_options = _adjacent_background_cells(grid_shape, cells)
            if not marker_options:
                continue
            visible, diag_a, diag_b = _count_visible_stamps(grid_shape, seed, r0, c0, BLACK)
            if visible < MIN_VISIBLE_UNIQUE or diag_a < MIN_VISIBLE_PER_DIAG or diag_b < MIN_VISIBLE_PER_DIAG:
                continue
            random.shuffle(marker_options)
            candidates.append((int(r0), int(c0), marker_options[0]))
    if not candidates:
        return None
    return random.choice(candidates)


def _choose_pink_marker(grid_shape: tuple[int, int], forbidden_green: set[Coord], other_non_bg: set[Coord]) -> Coord | None:
    candidates = []
    for r in range(grid_shape[0]):
        for c in range(grid_shape[1]):
            cell = (int(r), int(c))
            if cell in other_non_bg:
                continue
            if any((nr, nc) in forbidden_green for nr, nc in neighbors4(r, c)):
                continue
            if any((rr, cc) in other_non_bg for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)):
                continue
            candidates.append(cell)
    if not candidates:
        return None
    return random.choice(candidates)


def _build_input_from_layout(
    grid_shape: tuple[int, int],
    objects: list[tuple[np.ndarray, int, int]],
    gray_cell: Coord,
    pink_cell: Coord,
) -> np.ndarray:
    grid = np.full(grid_shape, BLACK, dtype=int)
    for mask, r0, c0 in objects:
        _paint_cells(grid, _placed_cells(mask, r0, c0), GREEN)
    grid[int(gray_cell[0]), int(gray_cell[1])] = GRAY
    grid[int(pink_cell[0]), int(pink_cell[1])] = PINK
    return grid


def _gen_attempt() -> np.ndarray | None:
    grid_shape = (
        random.randint(GRID_MIN, GRID_MAX),
        random.randint(GRID_MIN, GRID_MAX),
    )
    object_count = random.choice(OBJECT_COUNTS)

    selected_mask = random.choice(_SELECTABLE_SHAPES)
    selected_layout = _choose_selected_placement(grid_shape, selected_mask)
    if selected_layout is None:
        return None
    sel_r0, sel_c0, gray_cell = selected_layout

    objects: list[tuple[np.ndarray, int, int]] = [(selected_mask, sel_r0, sel_c0)]
    selected_cells = _placed_cells(selected_mask, sel_r0, sel_c0)
    protected_cells = set(selected_cells)
    protected_cells.add(gray_cell)

    for _ in range(object_count - 1):
        placed = False
        for _try in range(MAX_OTHER_PLACE_TRIES):
            mask = random.choice(_ALL_SHAPES)
            h, w = mask.shape
            r0 = random.randint(0, grid_shape[0] - h)
            c0 = random.randint(0, grid_shape[1] - w)
            if not _can_place_object(mask, r0, c0, grid_shape, protected_cells):
                continue
            objects.append((mask, r0, c0))
            new_cells = _placed_cells(mask, r0, c0)
            protected_cells |= new_cells
            placed = True
            break
        if not placed:
            return None

    pink_cell = _choose_pink_marker(grid_shape, protected_cells - {gray_cell}, protected_cells)
    if pink_cell is None:
        return None

    inp = _build_input_from_layout(grid_shape, objects, gray_cell, pink_cell)
    try:
        out = main(inp)
    except AssertionError:
        return None

    if out.shape != inp.shape:
        return None
    colors = set(map(int, np.unique(out)))
    if GREEN not in colors or PINK not in colors:
        return None
    if np.array_equal(inp, out):
        return None
    filled_ratio = float(np.count_nonzero(out != BLACK)) / float(out.size)
    if filled_ratio > MAX_FILLED_RATIO:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 10")
