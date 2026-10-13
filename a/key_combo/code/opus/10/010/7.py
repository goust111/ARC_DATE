import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

GEN_RETRY_LIMIT = 800


TEMPLATE_LIBRARY = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[0, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=int),
    np.array([[0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 0, 1], [0, 0, 0, 1]], dtype=int),
    np.array([[0, 1, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0]], dtype=int),
    np.array([[0, 1, 0, 0], [1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 1, 0]], dtype=int),
    np.array([[0, 0, 1, 0], [1, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0]], dtype=int),
]


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_template(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    comps = _component_coords(grid == BLUE)
    assert len(comps) == 1, "expected exactly one blue template component"
    coords = comps[0]
    r0, c0, r1, c1 = _bbox(coords)
    patch = (grid[r0 : r1 + 1, c0 : c1 + 1] == BLUE).astype(int)
    assert patch.shape[0] in (3, 4) and patch.shape[1] in (3, 4)
    assert 4 <= int(patch.sum()) < patch.size
    assert patch[0, 0] == 0
    assert patch[1, 1] == 1
    assert np.all(patch.sum(axis=0) > 0) and np.all(patch.sum(axis=1) > 0)
    return patch, (r0, c0)


def _extract_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    comps = _component_coords(grid == RED)
    markers: list[tuple[int, int, int]] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == w, "red marker must be square"
        k = h
        assert k in (2, 3), "red marker size must be 2 or 3"
        assert coords.shape[0] == k * k, "red marker must be solid"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == RED), "red marker square must be filled"
        markers.append((r0, c0, k))
    markers.sort()
    assert 3 <= len(markers) <= 5
    return markers


def _scaled_blue_cells(template: np.ndarray, k: int) -> list[tuple[int, int]]:
    h, w = template.shape
    cells: list[tuple[int, int]] = []
    for tr in range(h):
        for tc in range(w):
            if template[tr, tc] != 1:
                continue
            for dr in range(k):
                for dc in range(k):
                    cells.append((tr * k + dr, tc * k + dc))
    return cells


def _composite_after_stamps(
    grid: np.ndarray,
    *,
    erase_all_first: bool,
    full_bbox: bool = False,
) -> np.ndarray:
    template, _ = _extract_template(grid)
    markers = _extract_markers(grid)

    out = np.asarray(grid, dtype=int).copy()
    if erase_all_first:
        out[out == RED] = BLACK

    h, w = template.shape
    blue_cells = None if full_bbox else _scaled_blue_cells
    for r0, c0, k in markers:
        if not erase_all_first:
            out[r0 : r0 + k, c0 : c0 + k] = BLACK

        if full_bbox:
            for rr in range(r0, r0 + h * k):
                for cc in range(c0, c0 + w * k):
                    if _in_bounds(out.shape, rr, cc) and out[rr, cc] == BLACK:
                        out[rr, cc] = BLUE
        else:
            for dr, dc in blue_cells(template, k):
                rr = r0 + dr
                cc = c0 + dc
                if _in_bounds(out.shape, rr, cc) and out[rr, cc] == BLACK:
                    out[rr, cc] = BLUE
    return out


def _mark_blank_lines(composite: np.ndarray) -> np.ndarray:
    row_mask = np.all(composite == BLACK, axis=1)
    col_mask = np.all(composite == BLACK, axis=0)
    out = composite.copy()
    out[row_mask[:, None] | col_mask[None, :]] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp scaled copies of the blue template at every red square and color the composite's blank rows and columns green.

    Input structure:
    - The grid has black background, one irregular blue template near the top-left, and 3-5 detached red solid squares elsewhere.
    - Each red square is a 2x2 or 3x3 marker whose side length is also its stamp scale.
    - Green is absent from the input.

    Rule:
    - Take the blue template's minimal bounding box as the source pattern.
    - Process red squares in top-left to bottom-right order; before each stamp, erase only that square to black.
    - Scale every blue template cell into a k x k blue block, align the scaled pattern's top-left corner to the marker's top-left corner, and paint only cells that are currently black.
    - After all stamps, recolor every all-black row and every all-black column to green, leaving the rest of the composite unchanged.

    Key insight:
    - The final green guide lines depend on the negative space left after all scale-dependent, obstacle-aware stamps have been applied, so they cannot be inferred from the input alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert not np.any(g == GREEN), "green must be absent from the input"

    composite = _composite_after_stamps(g, erase_all_first=False)
    return _mark_blank_lines(composite)


def _can_place_square(occupied: np.ndarray, r0: int, c0: int, k: int, *, padding: int = 1) -> bool:
    h, w = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + k > h or c0 + k > w:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(h, r0 + k + padding)
    cc1 = min(w, c0 + k + padding)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _paint_square(grid: np.ndarray, r0: int, c0: int, k: int, color: int) -> None:
    grid[r0 : r0 + k, c0 : c0 + k] = color


def _place_random_marker(
    occupied: np.ndarray,
    stamp_shape: tuple[int, int],
    h: int,
    w: int,
    k: int,
    row_floor: int,
    col_floor: int,
) -> tuple[int, int] | None:
    stamp_h = stamp_shape[0] * k
    stamp_w = stamp_shape[1] * k
    candidates: list[tuple[int, int]] = []
    for r0 in range(row_floor, h - stamp_h + 1):
        for c0 in range(col_floor, w - stamp_w + 1):
            if _can_place_square(occupied, r0, c0, k):
                candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _validate_generated(grid: np.ndarray, forced_late_marker: tuple[int, int, int]) -> bool:
    try:
        template, _ = _extract_template(grid)
        markers = _extract_markers(grid)
    except AssertionError:
        return False

    if {k for _, _, k in markers} != {2, 3}:
        return False
    if template[0, 0] != 0 or template[1, 1] != 1:
        return False

    true_comp = _composite_after_stamps(grid, erase_all_first=False)
    erase_all_comp = _composite_after_stamps(grid, erase_all_first=True)
    bbox_comp = _composite_after_stamps(grid, erase_all_first=True, full_bbox=True)
    final_out = _mark_blank_lines(true_comp)

    if np.array_equal(true_comp, erase_all_comp):
        return False
    if np.array_equal(true_comp, bbox_comp):
        return False
    if np.array_equal(final_out, _mark_blank_lines(grid)):
        return False

    br, bc, bk = forced_late_marker
    block_true = true_comp[br : br + bk, bc : bc + bk]
    block_naive = erase_all_comp[br : br + bk, bc : bc + bk]
    if not np.any((block_naive == BLUE) & (block_true == BLACK)):
        return False

    blank_rows_in = np.all(grid == BLACK, axis=1)
    blank_cols_in = np.all(grid == BLACK, axis=0)
    blank_rows_out = np.all(true_comp == BLACK, axis=1)
    blank_cols_out = np.all(true_comp == BLACK, axis=0)
    if not blank_rows_out.any() or not blank_cols_out.any():
        return False
    if blank_rows_out.all() or blank_cols_out.all():
        return False
    if np.array_equal(blank_rows_in, blank_rows_out) and np.array_equal(blank_cols_in, blank_cols_out):
        return False

    if not np.array_equal(main(grid), final_out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    template = random.choice(TEMPLATE_LIBRARY).copy()
    th, tw = template.shape

    grid = np.zeros((h, w), dtype=int)
    tr = random.randint(0, 1)
    tc = random.randint(0, 1)
    grid[tr : tr + th, tc : tc + tw][template == 1] = BLUE
    occupied = grid != BLACK

    # Force one order-sensitive interaction: a 3x3 marker stamps over a later 2x2 marker.
    max_a_r = h - max(th * 3, 4 + th * 2)
    max_a_c = w - max(tw * 3, 4 + tw * 2)
    min_a_r = tr + th + 2
    min_a_c = tc + tw + 2
    a_candidates: list[tuple[int, int]] = []
    for ar in range(min_a_r, max_a_r + 1):
        for ac in range(min_a_c, max_a_c + 1):
            br = ar + 4
            bc = ac + 4
            if not _can_place_square(occupied, ar, ac, 3):
                continue
            probe = occupied.copy()
            _paint_square(probe, ar, ac, 3, True)
            if not _can_place_square(probe, br, bc, 2):
                continue
            a_candidates.append((ar, ac))
    if not a_candidates:
        return None

    ar, ac = random.choice(a_candidates)
    br, bc = ar + 4, ac + 4
    _paint_square(grid, ar, ac, 3, RED)
    _paint_square(grid, br, bc, 2, RED)
    occupied = grid != BLACK
    markers = [(ar, ac, 3), (br, bc, 2)]

    total_markers = random.randint(3, 5)
    while len(markers) < total_markers:
        k = random.choice([2, 3])
        pos = _place_random_marker(occupied, template.shape, h, w, k, tr + th + 1, tc + tw + 1)
        if pos is None:
            return None
        mr, mc = pos
        _paint_square(grid, mr, mc, k, RED)
        occupied = grid != BLACK
        markers.append((mr, mc, k))

    if not _validate_generated(grid, (br, bc, 2)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
