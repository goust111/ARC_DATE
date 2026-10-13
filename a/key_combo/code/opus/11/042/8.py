from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREY = 5


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _border_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }


def _inside_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _find_unique_marker(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == RED)
    assert coords.shape[0] == 1, "expected exactly one red marker pixel in the input"
    return int(coords[0, 0]), int(coords[0, 1])


def _valid_frame_component(coords: np.ndarray) -> bool:
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    if outer_h < 6 or outer_w < 6:
        return False
    if (outer_h - 2) * (outer_w - 2) < 16:
        return False
    return {tuple(map(int, rc)) for rc in coords.tolist()} == _border_cells(box)


def _find_unique_frame_box(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey_components = _components(grid == GREY)
    candidates = [coords for coords in grey_components if _valid_frame_component(coords)]
    assert len(candidates) == 1, "expected exactly one closed grey rectangular frame"
    return _bbox(candidates[0])


def _expected_output(grid: np.ndarray) -> np.ndarray:
    marker_r, marker_c = _find_unique_marker(grid)
    box = _find_unique_frame_box(grid)
    rr, cc = np.indices(grid.shape)
    inside = _inside_mask(grid.shape, box)
    same_parity = (rr % 2 == marker_r % 2) & (cc % 2 == marker_c % 2)
    out = np.array(grid, dtype=int, copy=True)
    out[(grid == BLACK) & inside & same_parity] = RED
    return out


def _frame_only_output(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    inside = _inside_mask(grid.shape, box)
    out[(grid == BLACK) & inside] = RED
    return out


def _parity_only_output(grid: np.ndarray, parity: tuple[int, int]) -> np.ndarray:
    rr, cc = np.indices(grid.shape)
    out = np.array(grid, dtype=int, copy=True)
    out[(grid == BLACK) & (rr % 2 == parity[0]) & (cc % 2 == parity[1])] = RED
    return out


def _box_parity_output(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    parity: tuple[int, int],
) -> np.ndarray:
    rr, cc = np.indices(grid.shape)
    inside = _inside_mask(grid.shape, box)
    out = np.array(grid, dtype=int, copy=True)
    out[(grid == BLACK) & inside & (rr % 2 == parity[0]) & (cc % 2 == parity[1])] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the intersection of the frame interior and the marker-defined parity lattice with red.

    Input structure:
    - A black background grid contains exactly one one-pixel-thick grey rectangular frame with a black interior.
    - Several extra grey pixels may appear outside the frame as isolated distractors and are not part of the rectangle.
    - Exactly one red marker pixel lies outside the frame and defines a row-parity and column-parity pair.

    Rule:
    - Find the unique closed grey rectangle and take only its strict interior, excluding the grey border.
    - Read the red marker's row and column parity, then select all cells whose row parity and column parity both match it.
    - Recolor a cell red iff it is currently black and satisfies both constraints at once: it is strictly inside the frame and on the marker's parity lattice.

    Key insight:
    - The output is the overlap of two independent selectors: a geometric selector from the rectangle interior and a coordinate-parity selector from the lone red marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert set(map(int, np.unique(g))) <= {BLACK, RED, GREY}, "unexpected colors in input"

    marker_r, marker_c = _find_unique_marker(g)
    box = _find_unique_frame_box(g)
    r0, c0, r1, c1 = box
    assert not (r0 <= marker_r <= r1 and c0 <= marker_c <= c1), "marker must be outside the frame"

    out = np.array(g, dtype=int, copy=True)
    for r in range(r0 + 1, r1):
        if r % 2 != marker_r % 2:
            continue
        for c in range(c0 + 1, c1):
            if c % 2 == marker_c % 2 and g[r, c] == BLACK:
                out[r, c] = RED
    return out


# --------------- generator ---------------

_GEN_BAGS: dict[str, list[object]] = {}
_GEN_RETRY_LIMIT = 400

_GRID_HEIGHT_CHOICES = (11, 12, 13, 13, 14, 15, 16, 17)
_GRID_WIDTH_CHOICES = (11, 12, 13, 13, 14, 15, 16, 17)
_ORIENTATION_OPTIONS = ("wide", "tall", "square")
_PARITY_OPTIONS = ((0, 0), (0, 1), (1, 0), (1, 1))
_MARKER_MODE_OPTIONS = ("near", "far", "edge")
_DISTRACTOR_COUNT_OPTIONS = (3, 4, 5)
_DISTRACTOR_MODE_OPTIONS = ("corner", "row", "col", "ring", "far")


def _bag_pick(name: str, options: tuple[object, ...] | list[object]) -> object:
    bag = _GEN_BAGS.get(name)
    if not bag:
        bag = list(options)
        random.shuffle(bag)
        _GEN_BAGS[name] = bag
    return _GEN_BAGS[name].pop()


def _choose_frame_dims(mode: str) -> tuple[int, int]:
    if mode == "wide":
        inner_h = random.randint(4, 6)
        inner_w = random.randint(max(6, inner_h + 1), 8)
    elif mode == "tall":
        inner_w = random.randint(4, 6)
        inner_h = random.randint(max(6, inner_w + 1), 8)
    else:
        inner_h = random.randint(4, 7)
        inner_w = min(8, max(4, inner_h + random.choice((-1, 0, 1))))
        if inner_w < 4:
            inner_w = 4
    return inner_h, inner_w


def _distance_to_box(r: int, c: int, box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    row_gap = 0 if r0 <= r <= r1 else (r0 - r if r < r0 else r - r1)
    col_gap = 0 if c0 <= c <= c1 else (c0 - c if c < c0 else c - c1)
    return row_gap, col_gap


def _chebyshev_to_box(r: int, c: int, box: tuple[int, int, int, int]) -> int:
    row_gap, col_gap = _distance_to_box(r, c, box)
    return max(row_gap, col_gap)


def _outside_box(r: int, c: int, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return not (r0 <= r <= r1 and c0 <= c <= c1)


def _safe_from_points(cell: tuple[int, int], points: list[tuple[int, int]], pad: int = 1) -> bool:
    r, c = cell
    return all(max(abs(r - pr), abs(c - pc)) > pad for pr, pc in points)


def _distractor_pools(shape: tuple[int, int], box: tuple[int, int, int, int]) -> dict[str, list[tuple[int, int]]]:
    h, w = shape
    r0, c0, r1, c1 = box
    pools = {name: [] for name in _DISTRACTOR_MODE_OPTIONS}
    for r in range(h):
        for c in range(w):
            if not _outside_box(r, c, box):
                continue
            row_gap, col_gap = _distance_to_box(r, c, box)
            cheb = max(row_gap, col_gap)
            if cheb < 2:
                continue
            if row_gap > 0 and col_gap > 0 and cheb <= 3:
                pools["corner"].append((r, c))
            if r in (r0, r1) and (c < c0 - 1 or c > c1 + 1):
                pools["row"].append((r, c))
            if c in (c0, c1) and (r < r0 - 1 or r > r1 + 1):
                pools["col"].append((r, c))
            if cheb == 2:
                pools["ring"].append((r, c))
            pools["far"].append((r, c))
    return pools


def _pick_from_pool(
    candidates: list[tuple[int, int]],
    used: list[tuple[int, int]],
) -> tuple[int, int] | None:
    random.shuffle(candidates)
    for cell in candidates:
        if cell in used:
            continue
        if _safe_from_points(cell, used, pad=1):
            return cell
    return None


def _place_distractors(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    count: int,
) -> list[tuple[int, int]] | None:
    pools = _distractor_pools(grid.shape, box)
    selected: list[tuple[int, int]] = []
    mode_cycle = list(_DISTRACTOR_MODE_OPTIONS)
    random.shuffle(mode_cycle)

    while len(selected) < count:
        made_progress = False
        for mode in mode_cycle:
            cell = _pick_from_pool(pools[mode][:], selected)
            if cell is None:
                continue
            selected.append(cell)
            made_progress = True
            if len(selected) == count:
                break
        if len(selected) == count:
            break
        if not made_progress:
            return None
        mode_cycle = ["far"]

    for r, c in selected:
        grid[r, c] = GREY
    return selected


def _marker_candidates(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    distractors: list[tuple[int, int]],
    parity: tuple[int, int],
) -> dict[str, list[tuple[int, int]]]:
    h, w = grid.shape
    candidates = {name: [] for name in _MARKER_MODE_OPTIONS}
    grey_points = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)]
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLACK or not _outside_box(r, c, box):
                continue
            if (r % 2, c % 2) != parity:
                continue
            if not _safe_from_points((r, c), grey_points, pad=1):
                continue
            cheb = _chebyshev_to_box(r, c, box)
            if cheb == 2:
                candidates["near"].append((r, c))
            if r in (0, h - 1) or c in (0, w - 1):
                candidates["edge"].append((r, c))
            if cheb >= 4:
                candidates["far"].append((r, c))
    return candidates


def _place_marker(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    distractors: list[tuple[int, int]],
    parity: tuple[int, int],
) -> tuple[int, int] | None:
    candidates = _marker_candidates(grid, box, distractors, parity)
    preferred_mode = _bag_pick("marker_mode", list(_MARKER_MODE_OPTIONS))
    for mode in (preferred_mode, "near", "far", "edge"):
        pool = candidates.get(mode, [])
        if pool:
            cell = random.choice(pool)
            grid[cell] = RED
            return cell
    return None


def _random_grid_shape(outer_h: int, outer_w: int) -> tuple[int, int]:
    min_h = max(outer_h + 3, 11)
    min_w = max(outer_w + 3, 11)
    height_choices = [h for h in _GRID_HEIGHT_CHOICES if h >= min_h]
    width_choices = [w for w in _GRID_WIDTH_CHOICES if w >= min_w]
    if not height_choices or not width_choices:
        raise RuntimeError("frame does not fit generator bounds")
    return random.choice(height_choices), random.choice(width_choices)


def _place_frame(grid: np.ndarray, inner_h: int, inner_w: int) -> tuple[int, int, int, int]:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    h, w = grid.shape
    r0 = random.randint(1, h - outer_h - 1)
    c0 = random.randint(1, w - outer_w - 1)
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    return r0, c0, r1, c1


def _validate_input(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert set(map(int, np.unique(grid))) <= {BLACK, RED, GREY}

    marker_r, marker_c = _find_unique_marker(grid)
    box = _find_unique_frame_box(grid)
    r0, c0, r1, c1 = box
    assert not (r0 <= marker_r <= r1 and c0 <= marker_c <= c1)

    grey_components = _components(grid == GREY)
    frame_sizes = [len(comp) for comp in grey_components if _valid_frame_component(comp)]
    distractor_sizes = [len(comp) for comp in grey_components if not _valid_frame_component(comp)]
    assert frame_sizes == [2 * (r1 - r0 + 1) + 2 * (c1 - c0 + 1) - 4]
    assert 3 <= len(distractor_sizes) <= 5
    assert all(size == 1 for size in distractor_sizes)

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert interior.size >= 16
    assert np.all(interior == BLACK)

    rr, cc = np.indices(grid.shape)
    inside = _inside_mask(grid.shape, box)
    parity = (marker_r % 2, marker_c % 2)
    same_parity = (rr % 2 == parity[0]) & (cc % 2 == parity[1])

    assert np.count_nonzero((grid == BLACK) & inside & same_parity) >= 4
    assert np.any((grid == BLACK) & inside & ~same_parity)
    assert np.any((grid == BLACK) & ~inside & same_parity)

    expected = _expected_output(grid)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert np.count_nonzero(actual != grid) >= 4

    frame_only = _frame_only_output(grid, box)
    parity_only = _parity_only_output(grid, parity)
    assert not np.array_equal(actual, frame_only)
    assert not np.array_equal(actual, parity_only)
    for alt_parity in _PARITY_OPTIONS:
        alt = _box_parity_output(grid, box, alt_parity)
        if alt_parity == parity:
            assert np.array_equal(alt, actual)
        else:
            assert not np.array_equal(alt, actual)


def _attempt_generate() -> np.ndarray | None:
    orientation = _bag_pick("orientation", list(_ORIENTATION_OPTIONS))
    inner_h, inner_w = _choose_frame_dims(str(orientation))
    height, width = _random_grid_shape(inner_h + 2, inner_w + 2)
    grid = np.full((height, width), BLACK, dtype=int)
    box = _place_frame(grid, inner_h, inner_w)

    distractor_count = int(_bag_pick("distractor_count", list(_DISTRACTOR_COUNT_OPTIONS)))
    distractors = _place_distractors(grid, box, distractor_count)
    if distractors is None:
        return None

    parity = _bag_pick("parity", list(_PARITY_OPTIONS))
    marker = _place_marker(grid, box, distractors, parity)  # type: ignore[arg-type]
    if marker is None:
        return None

    try:
        _validate_input(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
