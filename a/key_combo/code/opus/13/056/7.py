from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7

GAP_TO_FILL = {
    0: YELLOW,
    1: GREEN,
    2: PINK,
    3: ORANGE,
}

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _flood_exterior_black(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    exterior = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def add(r: int, c: int) -> None:
        if grid[r, c] == BLACK and not exterior[r, c]:
            exterior[r, c] = True
            queue.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if _in_bounds(grid.shape, rr, cc):
                add(rr, cc)

    return exterior


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))

    comps.sort(key=lambda pts: (int(pts[:, 0].min()), int(pts[:, 1].min())))
    return comps


def _straight_run_lengths(cells: set[tuple[int, int]]) -> list[int]:
    remaining = set(cells)
    lengths: list[int] = []
    while remaining:
        start = remaining.pop()
        queue = deque([start])
        run = [start]
        while queue:
            r, c = queue.popleft()
            for dr, dc in DIRS4:
                nxt = (r + dr, c + dc)
                if nxt in remaining:
                    remaining.remove(nxt)
                    queue.append(nxt)
                    run.append(nxt)
        rows = {r for r, _ in run}
        cols = {c for _, c in run}
        assert len(rows) == 1 or len(cols) == 1, "gap run must be straight"
        if len(rows) == 1:
            ordered = sorted(c for _, c in run)
        else:
            ordered = sorted(r for r, _ in run)
        assert ordered == list(range(ordered[0], ordered[-1] + 1)), "gap run must be contiguous"
        lengths.append(len(run))
    lengths.sort()
    return lengths


def _gap_total_for_region(grid: np.ndarray, coords: np.ndarray) -> int:
    offset_cells: set[tuple[int, int]] = set()
    blue_contacts = 0

    for r, c in coords:
        r = int(r)
        c = int(c)
        for dr, dc in DIRS4:
            br, bc = r + dr, c + dc
            if not _in_bounds(grid.shape, br, bc) or int(grid[br, bc]) != BLUE:
                continue
            blue_contacts += 1
            orr, occ = br + dr, bc + dc
            assert _in_bounds(grid.shape, orr, occ), "outer wall offset must stay in-bounds"
            offset_cells.add((orr, occ))

    assert blue_contacts > 0, "each enclosed region must touch an inner blue wall"
    assert offset_cells, "expected measurable outer wall positions"
    assert all(int(grid[r, c]) in (BLACK, RED) for r, c in offset_cells), (
        "outer wall positions must be red or black gaps"
    )

    gap_cells = {(r, c) for r, c in offset_cells if int(grid[r, c]) == BLACK}
    run_lengths = _straight_run_lengths(gap_cells)
    assert len(run_lengths) <= 2, "each frame may have at most two gap runs"
    assert all(1 <= length <= 3 for length in run_lengths), "gap run length must be 1-3"
    return len(gap_cells)


def _analyze_input(grid: np.ndarray) -> list[tuple[np.ndarray, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    exterior = _flood_exterior_black(g)
    enclosed_mask = (g == BLACK) & ~exterior
    regions = _mask_components(enclosed_mask)
    assert 2 <= len(regions) <= 4, "expected 2-4 enclosed regions"

    analyzed: list[tuple[np.ndarray, int]] = []
    totals: list[int] = []
    for coords in regions:
        total = _gap_total_for_region(g, coords)
        assert total in GAP_TO_FILL, "gap total must be encodable"
        analyzed.append((coords, total))
        totals.append(total)

    assert len(set(totals)) == len(totals), "each frame must have a distinct total gap length"
    return analyzed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed chamber with the color encoded by the total length of the outer red-wall gaps.

    Input structure:
    - The grid has a black background plus separated double-wall frames made from red outer walls and blue inner reinforcement walls.
    - Each frame contains one enclosed black chamber; the red outer wall may have 0-2 black gap runs, while the inner blue wall still keeps the chamber closed to 4-neighbor escape.
    - Different frames have different total red-gap lengths among 0, 1, 2, and 3.

    Rule:
    - Treat red and blue cells as walls, then flood-fill black cells reachable from the grid border by 4-neighbor moves.
    - Every remaining black component is an enclosed chamber.
    - For each chamber, look at the blue wall cells that touch it, step one more cell outward in the same direction, and count how many of those outer-wall positions are black gaps in the red wall.
    - Fill the chamber by the fixed code G=0->yellow(4), G=1->green(3), G=2->pink(6), G=3->orange(7), leaving all walls unchanged.

    Key insight:
    - Chamber detection depends on combined red-blue wall topology, but the fill color depends only on the measured total length of the missing red outer-wall segments.
    """
    g = np.asarray(grid, dtype=int)
    regions = _analyze_input(g)

    out = g.copy()
    for coords, total in regions:
        out[coords[:, 0], coords[:, 1]] = GAP_TO_FILL[total]
    return out


GRID_H_RANGE = (16, 20)
GRID_W_RANGE = (16, 20)
FRAME_COUNT_OPTIONS = (2, 3, 4)
FRAME_COUNT_WEIGHTS = (1, 2, 5)
FRAME_SIZE_MIN = 6
FRAME_SIZE_MAX = 8
MAX_ATTEMPTS = 400

GAP_PATTERNS = {
    0: [()],
    1: [(1,)],
    2: [(2,), (1, 1)],
    3: [(3,), (1, 2), (2, 1)],
}
SIDES = ("top", "bottom", "left", "right")


def _frame_cells(h: int, w: int) -> list[tuple[int, int, int, int]]:
    mid_r = h // 2
    mid_c = w // 2
    return [
        (1, mid_r - 2, 1, mid_c - 2),
        (1, mid_r - 2, mid_c + 1, w - 2),
        (mid_r + 1, h - 2, 1, mid_c - 2),
        (mid_r + 1, h - 2, mid_c + 1, w - 2),
    ]


def _draw_rect_ring(grid: np.ndarray, r0: int, c0: int, height: int, width: int, color: int) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _side_coords(r0: int, c0: int, height: int, width: int, side: str) -> list[tuple[int, int]]:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _carve_gaps(grid: np.ndarray, r0: int, c0: int, height: int, width: int, total_gap: int) -> None:
    pattern = random.choice(GAP_PATTERNS[total_gap])
    if not pattern:
        return
    sides = random.sample(SIDES, k=len(pattern))
    for side, length in zip(sides, pattern):
        coords = _side_coords(r0, c0, height, width, side)
        start = random.randint(0, len(coords) - length)
        for r, c in coords[start : start + length]:
            grid[r, c] = BLACK


def _build_candidate() -> tuple[np.ndarray, list[int]] | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)

    frame_count = random.choices(FRAME_COUNT_OPTIONS, weights=FRAME_COUNT_WEIGHTS, k=1)[0]
    chosen_totals = random.sample(list(GAP_TO_FILL), k=frame_count)
    random.shuffle(chosen_totals)

    cells = _frame_cells(h, w)
    chosen_cells = random.sample(cells, k=frame_count)

    for (rmin, rmax, cmin, cmax), total_gap in zip(chosen_cells, chosen_totals):
        cell_h = rmax - rmin + 1
        cell_w = cmax - cmin + 1
        if cell_h < FRAME_SIZE_MIN or cell_w < FRAME_SIZE_MIN:
            return None

        height = random.randint(FRAME_SIZE_MIN, min(FRAME_SIZE_MAX, cell_h))
        width = random.randint(FRAME_SIZE_MIN, min(FRAME_SIZE_MAX, cell_w))
        r0 = random.randint(rmin, rmax - height + 1)
        c0 = random.randint(cmin, cmax - width + 1)

        _draw_rect_ring(grid, r0, c0, height, width, RED)
        _draw_rect_ring(grid, r0 + 1, c0 + 1, height - 2, width - 2, BLUE)
        _carve_gaps(grid, r0, c0, height, width, total_gap)

    return grid, sorted(chosen_totals)


def _validate_generated(grid: np.ndarray, intended_totals: list[int]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not set(map(int, np.unique(grid))).issubset({BLACK, BLUE, RED}):
        return False

    try:
        analyzed = _analyze_input(grid)
        if sorted(total for _, total in analyzed) != intended_totals:
            return False
        out = main(grid)
    except AssertionError:
        return False

    for coords, total in analyzed:
        expected = GAP_TO_FILL[total]
        if not np.all(out[coords[:, 0], coords[:, 1]] == expected):
            return False

    wall_mask = grid != BLACK
    if not np.array_equal(out[wall_mask], grid[wall_mask]):
        return False
    if len(set(int(v) for v in out.flatten()) & set(GAP_TO_FILL.values())) < 2:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_candidate()
        if built is None:
            continue
        grid, intended_totals = built
        if _validate_generated(grid, intended_totals):
            return grid
    raise RuntimeError("failed to generate a valid wall-gap encoding input")
