from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
SQUARE_FILL = 8
BAR_FILL = 2
OUTLINE_COLORS = (1, 3, 4, 6, 7, 9)

GRID_ROWS_RANGE = (17, 20)
GRID_COLS_RANGE = (17, 20)
OUTLINE_COUNT_OPTIONS = (2, 3)
VALID_COUNT_PAIRS = (
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (1, 4),
    (4, 1),
    (2, 3),
    (3, 2),
)
MAX_GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    if isinstance(coords, np.ndarray):
        rows = coords[:, 0]
        cols = coords[:, 1]
        return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        coords = []
        while q:
            rr, cc = q.popleft()
            coords.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _row_span_interior_cells(grid: np.ndarray, coords: np.ndarray) -> list[tuple[int, int]]:
    occupied = {tuple(map(int, rc)) for rc in coords}
    rows = sorted({int(r) for r, _c in occupied})
    cells: list[tuple[int, int]] = []
    for r in rows:
        cols = sorted(c for rr, c in occupied if rr == r)
        if len(cols) < 2:
            continue
        left, right = cols[0], cols[-1]
        for c in range(left + 1, right):
            if (r, c) not in occupied and int(grid[r, c]) == BG:
                cells.append((r, c))
    return cells


def _primitive_candidates(mask: np.ndarray) -> tuple[list[tuple[int, str]], int]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    index = {coord: i for i, coord in enumerate(coords)}
    occ = set(coords)
    candidates: list[tuple[int, str]] = []
    seen_masks: set[tuple[int, str]] = set()
    for r, c in coords:
        square = ((r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1))
        if all(cell in occ for cell in square):
            bits = sum(1 << index[cell] for cell in square)
            key = (bits, "square")
            if key not in seen_masks:
                seen_masks.add(key)
                candidates.append(key)
        hbar = ((r, c), (r, c + 1), (r, c + 2))
        if all(cell in occ for cell in hbar):
            bits = sum(1 << index[cell] for cell in hbar)
            key = (bits, "bar")
            if key not in seen_masks:
                seen_masks.add(key)
                candidates.append(key)
        vbar = ((r, c), (r + 1, c), (r + 2, c))
        if all(cell in occ for cell in vbar):
            bits = sum(1 << index[cell] for cell in vbar)
            key = (bits, "bar")
            if key not in seen_masks:
                seen_masks.add(key)
                candidates.append(key)
    return candidates, len(coords)


def _decomposition_summary(mask: np.ndarray) -> tuple[int, tuple[int, int] | None]:
    candidates, n_cells = _primitive_candidates(mask)
    if n_cells == 0:
        return 0, None
    if not candidates:
        return 0, None

    cell_to_candidates = [[] for _ in range(n_cells)]
    for idx, (bits, _kind) in enumerate(candidates):
        for cell_idx in range(n_cells):
            if bits & (1 << cell_idx):
                cell_to_candidates[cell_idx].append(idx)

    full_mask = (1 << n_cells) - 1
    solutions: list[tuple[int, int]] = []

    def dfs(remaining: int, squares: int, bars: int) -> None:
        if len(solutions) >= 2:
            return
        if remaining == 0:
            solutions.append((squares, bars))
            return

        best_cell = None
        best_options = None
        for cell_idx in range(n_cells):
            if not (remaining & (1 << cell_idx)):
                continue
            options = [
                cand_idx
                for cand_idx in cell_to_candidates[cell_idx]
                if candidates[cand_idx][0] & remaining == candidates[cand_idx][0]
            ]
            if not options:
                return
            if best_options is None or len(options) < len(best_options):
                best_cell = cell_idx
                best_options = options
                if len(best_options) == 1:
                    break

        assert best_cell is not None and best_options is not None
        for cand_idx in best_options:
            bits, kind = candidates[cand_idx]
            dfs(
                remaining ^ bits,
                squares + (1 if kind == "square" else 0),
                bars + (1 if kind == "bar" else 0),
            )

    dfs(full_mask, 0, 0)
    if len(solutions) != 1:
        return len(solutions), None
    return 1, solutions[0]


def _extract_grey_mask(grid: np.ndarray) -> np.ndarray:
    comps = _components_of_color(grid, GREY)
    assert len(comps) == 1, "expected exactly one grey composite"
    coords = comps[0]
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local = coords - np.array([r0, c0], dtype=int)
    patch[local[:, 0], local[:, 1]] = True
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose the grey composite into squares and bars, then fill the outline cavities whose row-span areas match those counts.

    Input structure:
    - The background is black, plus one grey 4-connected composite made only from 2x2 squares and 1x3 or 3x1 bars.
    - Elsewhere, 2 to 3 monochrome one-pixel-thick outline objects of distinct colors enclose black row-span cavities.
    - Exactly one outline cavity has area equal to the number of square primitives, and exactly one has area equal to the number of bar primitives.

    Rule:
    - Uniquely decompose the grey composite so every grey cell belongs to exactly one allowed primitive.
    - Count squares as S and bars as R.
    - For each outline object, compute its cavity as the black cells strictly between the leftmost and rightmost outline pixels on each occupied row.
    - Fill the cavity whose area is S with teal and the cavity whose area is R with red; leave other cavities black and keep the grey composite unchanged.

    Key insight:
    - The grey object is an encoder: its primitive decomposition produces two counts, and those counts select which distant outline interiors get filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    grey_mask = _extract_grey_mask(g)
    solution_count, counts = _decomposition_summary(grey_mask)
    assert solution_count == 1 and counts is not None, "grey composite must have a unique valid decomposition"
    square_count, bar_count = counts
    assert square_count > 0 and bar_count > 0 and square_count != bar_count

    outline_colors = sorted(int(c) for c in np.unique(g) if int(c) not in {BG, GREY})
    assert 2 <= len(outline_colors) <= 3

    out = g.copy()
    area_hits = {square_count: 0, bar_count: 0}
    for color in outline_colors:
        comps = _components_of_color(g, color)
        assert len(comps) == 1, "each outline color should form one object"
        interior = _row_span_interior_cells(g, comps[0])
        area = len(interior)
        if area == square_count:
            area_hits[square_count] += 1
            for r, c in interior:
                out[r, c] = SQUARE_FILL
        elif area == bar_count:
            area_hits[bar_count] += 1
            for r, c in interior:
                out[r, c] = BAR_FILL

    assert area_hits[square_count] == 1 and area_hits[bar_count] == 1
    return out


def _shape_cells(kind: str, anchor: tuple[int, int], vertical: bool = False) -> set[tuple[int, int]]:
    r, c = anchor
    if kind == "square":
        return {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
    if vertical:
        return {(r, c), (r + 1, c), (r + 2, c)}
    return {(r, c), (r, c + 1), (r, c + 2)}


def _generate_grey_patch(target_counts: tuple[int, int]) -> np.ndarray | None:
    want_squares, want_bars = target_counts
    primitives = ["square"] * want_squares + ["bar"] * want_bars
    random.shuffle(primitives)

    occupied: set[tuple[int, int]] = set()
    for idx, kind in enumerate(primitives):
        placed = False
        for _ in range(180):
            vertical = kind == "bar" and random.random() < 0.5
            if idx == 0:
                anchor = (0, 0)
            else:
                r0, c0, r1, c1 = _bbox(occupied)
                anchor = (
                    random.randint(r0 - 3, r1 + 1),
                    random.randint(c0 - 3, c1 + 1),
                )
            cells = _shape_cells(kind, anchor, vertical)
            if occupied & cells:
                continue
            if occupied:
                adjacent = any(
                    (nr, nc) in occupied
                    for r, c in cells
                    for nr, nc in _neighbors4(r, c)
                )
                if not adjacent:
                    continue
            occupied |= cells
            placed = True
            break
        if not placed:
            return None

    r0, c0, r1, c1 = _bbox(occupied)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h > 7 or w > 7:
        return None
    patch = np.zeros((h, w), dtype=int)
    for r, c in occupied:
        patch[r - r0, c - c0] = GREY

    solution_count, counts = _decomposition_summary(patch == GREY)
    if solution_count != 1 or counts != target_counts:
        return None
    return patch


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs = []
    for h in range(1, area + 1):
        if area % h == 0:
            pairs.append((h, area // h))
    return sorted(pairs, key=lambda hw: (max(hw), abs(hw[0] - hw[1])))


def _make_outline_patch(area: int, color: int) -> np.ndarray:
    interior_h, interior_w = random.choice(_factor_pairs(area)[: min(3, len(_factor_pairs(area)))])
    ring = np.zeros((interior_h + 2, interior_w + 2), dtype=int)
    ring[0, :] = color
    ring[-1, :] = color
    ring[:, 0] = color
    ring[:, -1] = color

    patch = ring
    if random.random() < 0.7:
        side = random.choice(("top", "bottom", "left", "right"))
        seg_len = random.choice((1, 2))
        if side in {"top", "bottom"} and patch.shape[1] >= 3:
            seg_len = min(seg_len, patch.shape[1] - 2)
            start = random.randint(1, patch.shape[1] - seg_len - 1)
            grown = np.zeros((patch.shape[0] + 1, patch.shape[1]), dtype=int)
            if side == "top":
                grown[1:, :] = patch
                grown[0, start : start + seg_len] = color
            else:
                grown[:-1, :] = patch
                grown[-1, start : start + seg_len] = color
            patch = grown
        elif side in {"left", "right"} and patch.shape[0] >= 3:
            seg_len = min(seg_len, patch.shape[0] - 2)
            start = random.randint(1, patch.shape[0] - seg_len - 1)
            grown = np.zeros((patch.shape[0], patch.shape[1] + 1), dtype=int)
            if side == "left":
                grown[:, 1:] = patch
                grown[start : start + seg_len, 0] = color
            else:
                grown[:, :-1] = patch
                grown[start : start + seg_len, -1] = color
            patch = grown

    coords = np.argwhere(patch == color)
    assert len(coords) > 0
    assert len(_components_of_color(patch, color)) == 1
    assert len(_row_span_interior_cells(patch, coords)) == area
    return patch


def _quadrant_regions(h: int, w: int) -> list[tuple[int, int, int, int]]:
    mid_r = h // 2
    mid_c = w // 2
    return [
        (1, 1, mid_r - 1, mid_c - 1),
        (1, mid_c, mid_r - 1, w - 2),
        (mid_r, 1, h - 2, mid_c - 1),
        (mid_r, mid_c, h - 2, w - 2),
    ]


def _place_patch(grid: np.ndarray, patch: np.ndarray, region: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = region
    max_r = r1 - patch.shape[0] + 1
    max_c = c1 - patch.shape[1] + 1
    if max_r < r0 or max_c < c0:
        return False
    top = random.randint(r0, max_r)
    left = random.randint(c0, max_c)
    sub = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    if np.any((patch != BG) & (sub != BG)):
        return False
    sub[patch != BG] = patch[patch != BG]
    return True


def _validate_instance(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30

    grey_mask = _extract_grey_mask(grid)
    solution_count, counts = _decomposition_summary(grey_mask)
    assert solution_count == 1 and counts is not None
    square_count, bar_count = counts
    assert square_count > 0 and bar_count > 0 and square_count != bar_count

    outline_colors = sorted(int(c) for c in np.unique(grid) if int(c) not in {BG, GREY})
    assert 2 <= len(outline_colors) <= 3

    areas = []
    hits = {square_count: 0, bar_count: 0}
    for color in outline_colors:
        comps = _components_of_color(grid, color)
        assert len(comps) == 1
        interior = _row_span_interior_cells(grid, comps[0])
        area = len(interior)
        assert area > 0
        areas.append(area)
        if area == square_count:
            hits[square_count] += 1
        if area == bar_count:
            hits[bar_count] += 1
    assert len(set(areas)) == len(areas)
    assert hits[square_count] == 1
    assert hits[bar_count] == 1

    out = main(grid)
    for color in outline_colors:
        coords = _components_of_color(grid, color)[0]
        interior = _row_span_interior_cells(grid, coords)
        area = len(interior)
        expected = BG
        if area == square_count:
            expected = SQUARE_FILL
        elif area == bar_count:
            expected = BAR_FILL
        for r, c in interior:
            assert int(out[r, c]) == expected
    assert np.array_equal(out[grid == GREY], grid[grid == GREY])


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        h = random.randint(*GRID_ROWS_RANGE)
        w = random.randint(*GRID_COLS_RANGE)
        grid = np.zeros((h, w), dtype=int)

        count_pair = random.choice(VALID_COUNT_PAIRS)
        grey_patch = _generate_grey_patch(count_pair)
        if grey_patch is None:
            continue

        square_count, bar_count = count_pair
        outline_count = random.choice(OUTLINE_COUNT_OPTIONS)
        decoy_pool = [a for a in range(1, 9) if a not in {square_count, bar_count}]
        areas = [square_count, bar_count]
        if outline_count == 3:
            areas.append(random.choice(decoy_pool))
        random.shuffle(areas)

        colors = random.sample(OUTLINE_COLORS, outline_count)
        patches = [_make_outline_patch(area, color) for area, color in zip(areas, colors)]

        quadrants = _quadrant_regions(h, w)
        grey_quadrant = random.randrange(4)
        if not _place_patch(grid, grey_patch, quadrants[grey_quadrant]):
            continue

        remaining_quadrants = [i for i in range(4) if i != grey_quadrant]
        random.shuffle(remaining_quadrants)
        ok = True
        for patch, quad_idx in zip(patches, remaining_quadrants):
            if not _place_patch(grid, patch, quadrants[quad_idx]):
                ok = False
                break
        if not ok:
            continue

        try:
            _validate_instance(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
