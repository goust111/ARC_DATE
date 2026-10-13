from __future__ import annotations

import random

import numpy as np


BLACK = 0
OUTLINE = 5
MARKER_COLORS = (1, 2, 3, 4)

GRID_H_RANGE = (13, 20)
GRID_W_RANGE = (13, 20)
SHAPE_COUNT_CHOICES = (1, 2, 2)
PEAK_RANGE = (2, 4)
PLATEAU_RANGE = (0, 2)
MIN_SHAPE_GAP = 2
MIN_FILL_CELLS = 12
MIN_MARKER_GAP = 4
GENERATION_ATTEMPTS = 800


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _scanline_pairs(outline_mask: np.ndarray) -> list[tuple[int, int, int]]:
    pairs: list[tuple[int, int, int]] = []
    for r in range(outline_mask.shape[0]):
        cols = np.flatnonzero(outline_mask[r])
        if cols.size % 2 != 0:
            raise AssertionError("each row must have an even number of outline pixels")
        for left, right in zip(cols[0::2], cols[1::2]):
            pairs.append((r, int(left), int(right)))
    return pairs


def _span_mask(outline_mask: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(outline_mask, dtype=bool)
    for r, left, right in _scanline_pairs(outline_mask):
        if right - left > 1:
            mask[r, left + 1 : right] = True
    return mask


def _nearest_marker_color(r: int, c: int, markers: list[tuple[int, int, int]]) -> int:
    best = min(markers, key=lambda item: (abs(item[0] - r) + abs(item[1] - c), item[0], item[1]))
    return int(best[2])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each scanline-bounded contour span using the color of the nearest marker pixel.

    Input structure:
    - The grid uses black background cells, grey diagonal outline pixels, and 2-4 colored marker pixels.
    - Grey pixels appear in rows with an even count so they can be paired left-to-right into scanline endpoints.
    - Marker pixels may lie inside or outside the outlined spans, but never on top of grey outline pixels.

    Rule:
    - Preserve every grey outline pixel and every marker pixel.
    - In each row, pair grey pixels as (1st,2nd), (3rd,4th), and so on, then consider the cells strictly between each pair.
    - Recolor only the black cells inside those spans.
    - The new color of each filled cell is the color of the marker with minimum Manhattan distance; ties break by smaller marker row, then smaller marker column.

    Key insight:
    - The outline tells you where filling is allowed, while the markers independently tell you which color each allowed pixel receives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    outline_mask = g == OUTLINE
    markers = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere((g != BLACK) & (g != OUTLINE))]
    assert len(markers) >= 2, "expected at least two marker pixels"

    out = g.copy()
    filled_any = False
    for r, left, right in _scanline_pairs(outline_mask):
        for c in range(left + 1, right):
            if int(g[r, c]) != BLACK:
                continue
            out[r, c] = _nearest_marker_color(r, c, markers)
            filled_any = True

    assert filled_any, "expected at least one fillable black pixel"
    return out


def _make_shape_patch() -> np.ndarray:
    peak = random.randint(*PEAK_RANGE)
    plateau = random.randint(*PLATEAU_RANGE)
    radii = list(range(peak + 1)) + [peak] * plateau + list(range(peak - 1, -1, -1))

    drift = [0]
    for i in range(1, len(radii)):
        if radii[i] == radii[i - 1]:
            drift.append(drift[-1] + random.choice((-1, 0, 1)))
        else:
            drift.append(drift[-1])

    left_cols = [drift[i] - radii[i] for i in range(len(radii))]
    right_cols = [drift[i] + radii[i] + 1 for i in range(len(radii))]
    min_col = min(left_cols)
    max_col = max(right_cols)

    patch = np.zeros((len(radii), max_col - min_col + 1), dtype=bool)
    for r, (left, right) in enumerate(zip(left_cols, right_cols)):
        patch[r, left - min_col] = True
        patch[r, right - min_col] = True
    return patch


def _boxes_far_apart(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_shapes(h: int, w: int) -> np.ndarray | None:
    grid = np.full((h, w), BLACK, dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []

    for _ in range(random.choice(SHAPE_COUNT_CHOICES)):
        patch = _make_shape_patch()
        ph, pw = patch.shape
        if ph + 2 > h or pw + 2 > w:
            return None

        locations = [(r0, c0) for r0 in range(1, h - ph) for c0 in range(1, w - pw)]
        random.shuffle(locations)

        placed = False
        for r0, c0 in locations:
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(not _boxes_far_apart(box, other, MIN_SHAPE_GAP) for other in placed_boxes):
                continue
            for rr, cc in np.argwhere(patch):
                grid[r0 + int(rr), c0 + int(cc)] = OUTLINE
            placed_boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    return grid


def _pick_cell(
    candidates: list[tuple[int, int]],
    chosen: list[tuple[int, int]],
    min_gap: int,
    *,
    forbidden: set[tuple[int, int]] | None = None,
) -> tuple[int, int] | None:
    forbidden = forbidden or set()
    random.shuffle(candidates)
    for cell in candidates:
        if cell in forbidden:
            continue
        if all(abs(cell[0] - rr) + abs(cell[1] - cc) >= min_gap for rr, cc in chosen):
            return cell
    return None


def _find_tie_setup(
    grid: np.ndarray,
    span_black_cells: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], str] | None:
    h, w = grid.shape
    goals = span_black_cells[:]
    random.shuffle(goals)
    for pr, pc in goals:
        modes = ["vertical", "horizontal"]
        random.shuffle(modes)
        for mode in modes:
            for dist in (2, 3, 4):
                if mode == "vertical":
                    a = (pr - dist, pc)
                    b = (pr + dist, pc)
                else:
                    a = (pr, pc - dist)
                    b = (pr, pc + dist)
                if not (_in_bounds((h, w), *a) and _in_bounds((h, w), *b)):
                    continue
                if int(grid[a]) != BLACK or int(grid[b]) != BLACK:
                    continue
                return (pr, pc), a, b, mode
    return None


def _tie_holds(
    point: tuple[int, int],
    markers: list[tuple[int, int, int]],
) -> tuple[bool, int | None]:
    pr, pc = point
    dists = sorted((abs(r - pr) + abs(c - pc), r, c, color) for r, c, color in markers)
    if len(dists) < 2 or dists[0][0] != dists[1][0]:
        return False, None
    return True, int(dists[0][3])


def _semantic_ok(
    grid: np.ndarray,
    *,
    require_tie: bool,
    tie_point: tuple[int, int] | None,
) -> bool:
    outline_mask = grid == OUTLINE
    try:
        spans = _span_mask(outline_mask)
        out = main(grid)
    except Exception:
        return False

    markers = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere((grid != BLACK) & (grid != OUTLINE))]
    filled_mask = (grid == BLACK) & (out != BLACK)
    if int(filled_mask.sum()) < MIN_FILL_CELLS:
        return False
    if len({int(out[r, c]) for r, c in np.argwhere(filled_mask)}) < 2:
        return False
    if not np.any((grid == BLACK) & (~spans)):
        return False
    if not any(spans[r, c] for r, c, _ in markers):
        return False
    if not any(not spans[r, c] for r, c, _ in markers):
        return False

    for r, c, color in markers:
        if int(out[r, c]) != color:
            return False

    if require_tie:
        if tie_point is None:
            return False
        holds, expected = _tie_holds(tie_point, markers)
        if not holds or expected is None or int(out[tie_point]) != expected:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = _place_shapes(h, w)
    if grid is None:
        return None

    outline_mask = grid == OUTLINE
    spans = _span_mask(outline_mask)
    span_black_cells = [tuple(map(int, cell)) for cell in np.argwhere(spans & (grid == BLACK))]
    outside_black_cells = [tuple(map(int, cell)) for cell in np.argwhere((~spans) & (grid == BLACK))]
    if len(span_black_cells) < MIN_FILL_CELLS or not outside_black_cells:
        return None

    require_tie = random.random() < 0.45
    colors = list(MARKER_COLORS)
    random.shuffle(colors)

    chosen_cells: list[tuple[int, int]] = []
    marker_specs: list[tuple[int, int, int]] = []
    tie_point: tuple[int, int] | None = None

    if require_tie:
        tie_setup = _find_tie_setup(grid, span_black_cells)
        if tie_setup is None:
            return None
        tie_point, a, b, _mode = tie_setup
        chosen_cells.extend([a, b])
        marker_specs.append((a[0], a[1], colors.pop()))
        marker_specs.append((b[0], b[1], colors.pop()))

    inside_need = not any(spans[r, c] for r, c in chosen_cells)
    outside_need = not any(not spans[r, c] for r, c in chosen_cells)

    if inside_need:
        cell = _pick_cell(span_black_cells[:], chosen_cells, MIN_MARKER_GAP)
        if cell is None:
            return None
        chosen_cells.append(cell)
        marker_specs.append((cell[0], cell[1], colors.pop()))

    if outside_need:
        cell = _pick_cell(outside_black_cells[:], chosen_cells, MIN_MARKER_GAP)
        if cell is None:
            return None
        chosen_cells.append(cell)
        marker_specs.append((cell[0], cell[1], colors.pop()))

    target_count = random.randint(max(2, len(marker_specs)), 4)
    candidate_pool = span_black_cells + outside_black_cells
    while len(marker_specs) < target_count and colors:
        cell = _pick_cell(candidate_pool[:], chosen_cells, MIN_MARKER_GAP, forbidden=set(chosen_cells))
        if cell is None:
            break
        chosen_cells.append(cell)
        marker_specs.append((cell[0], cell[1], colors.pop()))

    if len(marker_specs) < 2:
        return None

    for r, c, color in marker_specs:
        grid[r, c] = color

    if not _semantic_ok(grid, require_tie=require_tie, tie_point=tie_point):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
