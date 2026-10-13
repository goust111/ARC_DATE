from __future__ import annotations

import random
from collections import deque

import numpy as np


AZURE = 8
GREEN = 3
LEGEND_COUNT = 4
PALETTE = [0, 1, 2, 4, 5, 6, 7, 9]

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _legend_mapping(g: np.ndarray) -> dict[int, int]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    swatches: list[tuple[int, int, int, int]] = []

    for sr in range(h):
        for sc in range(w):
            color = int(g[sr, sc])
            if seen[sr, sc] or color in (AZURE, GREEN):
                continue

            q: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            cells: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in DIR4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(g[rr, cc]) == color:
                        seen[rr, cc] = True
                        q.append((rr, cc))

            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            swatches.append((min(rows), min(cols), max(rows), color))

    assert len(swatches) >= LEGEND_COUNT, "expected a vertical color legend"
    swatches.sort(key=lambda x: (x[0], x[1], x[2]))
    colors = [color for _, _, _, color in swatches]
    assert len(set(colors)) == len(colors), "legend swatch colors must be distinct"
    return {i + 1: int(color) for i, color in enumerate(colors)}


def _azure_regions(g: np.ndarray):
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(g[sr, sc]) != AZURE:
                continue

            q: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            cells: list[tuple[int, int]] = []
            touches_edge = False
            boundary_colors: set[int] = set()
            perimeter = 0

            while q:
                r, c = q.popleft()
                cells.append((r, c))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_edge = True

                for dr, dc in DIR4:
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < h and 0 <= cc < w):
                        touches_edge = True
                        continue
                    v = int(g[rr, cc])
                    if v == AZURE:
                        if not seen[rr, cc]:
                            seen[rr, cc] = True
                            q.append((rr, cc))
                    else:
                        boundary_colors.add(v)
                        if v == GREEN:
                            perimeter += 1

            yield cells, touches_edge, boundary_colors, perimeter


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed Azure room by the legend color for its floor(perimeter / area) ratio.

    Input structure:
    - Azure is the background and Green pixels are one-cell-thick walls.
    - The Green walls enclose several Azure regions that do not touch the grid edge.
    - One corner contains a vertical stack of colored legend swatches, ordered top-to-bottom for ratios 1, 2, 3, 4, ...

    Rule:
    - Find each 4-connected Azure region that does not touch the grid edge and whose boundary is Green wall.
    - Let A be its number of Azure cells and P be the number of 4-neighbor sides where those cells touch Green wall.
    - Compute R = floor(P / A), then fill the whole region with the R-th color from the corner legend.
    - Preserve all Green walls and all legend swatches.

    Key insight: The wall-contact perimeter divided by area measures compactness, so compact rooms and thin corridors select different legend entries.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    legend = _legend_mapping(g)

    for cells, touches_edge, boundary_colors, perimeter in _azure_regions(g):
        if touches_edge or boundary_colors != {GREEN}:
            continue
        ratio = int(perimeter // len(cells))
        assert ratio in legend, "closed region ratio must appear in the legend"
        fill = legend[ratio]
        for r, c in cells:
            out[r, c] = fill

    return out


def _mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    shifted = {(r - min_r, c - min_c) for r, c in cells}
    h = max(r for r, _ in shifted) + 1
    w = max(c for _, c in shifted) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shifted:
        mask[r, c] = True
    return mask


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return _trim(out)


def _perimeter(mask: np.ndarray) -> int:
    cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    total = 0
    for r, c in cells:
        for dr, dc in DIR4:
            if (r + dr, c + dc) not in cells:
                total += 1
    return total


def _ratio(mask: np.ndarray) -> int:
    return int(_perimeter(mask) // int(mask.sum()))


def _random_polyomino(area: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIR4:
                p = (r + dr, c + dc)
                if p not in cells:
                    frontier.add(p)
        cells.add(random.choice(list(frontier)))
    return _mask_from_cells(cells)


def _mask_for_ratio(target: int) -> np.ndarray:
    if target == 4:
        return np.ones((1, 1), dtype=bool)
    if target == 3:
        return _random_transform(np.ones((1, 2), dtype=bool))

    if target == 2 and random.random() < 0.6:
        length = random.randint(3, 7)
        return _random_transform(np.ones((1, length), dtype=bool))

    area_range = (6, 12) if target == 1 else (3, 7)
    for _ in range(100):
        mask = _random_polyomino(random.randint(*area_range))
        if _ratio(mask) == target:
            return _random_transform(mask)

    if target == 1:
        h, w = random.choice([(2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (4, 3)])
        return _random_transform(np.ones((h, w), dtype=bool))

    fallback = random.choice(
        [
            np.array([[1, 1], [1, 0]], dtype=bool),
            np.ones((2, 2), dtype=bool),
            np.ones((1, 3), dtype=bool),
        ]
    )
    return _random_transform(fallback)


def _room_geometry(mask: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]:
    interior = {(int(r) + 1, int(c) + 1) for r, c in np.argwhere(mask)}
    walls: set[tuple[int, int]] = set()
    for r, c in interior:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                p = (r + dr, c + dc)
                if p not in interior:
                    walls.add(p)
    return interior, walls, (mask.shape[0] + 2, mask.shape[1] + 2)


def _blocked_neighborhood(blocked: np.ndarray, cells: set[tuple[int, int]], r0: int, c0: int) -> None:
    h, w = blocked.shape
    for r, c in cells:
        ar, ac = r0 + r, c0 + c
        for rr in range(max(0, ar - 1), min(h, ar + 2)):
            for cc in range(max(0, ac - 1), min(w, ac + 2)):
                blocked[rr, cc] = True


def _can_place(blocked: np.ndarray, cells: set[tuple[int, int]], r0: int, c0: int) -> bool:
    return all(not blocked[r0 + r, c0 + c] for r, c in cells)


def _place_legend(grid: np.ndarray, blocked: np.ndarray, colors: list[int]) -> None:
    h, w = grid.shape
    block = 2
    gap = random.choice([0, 1])
    total_h = LEGEND_COUNT * block + (LEGEND_COUNT - 1) * gap
    corner = random.choice(["TL", "TR", "BL", "BR"])

    r0 = 0 if corner[0] == "T" else h - total_h
    c0 = 0 if corner[1] == "L" else w - block
    swatch_cells: set[tuple[int, int]] = set()

    for i, color in enumerate(colors):
        rr = r0 + i * (block + gap)
        grid[rr : rr + block, c0 : c0 + block] = int(color)
        for r in range(rr, rr + block):
            for c in range(c0, c0 + block):
                swatch_cells.add((r, c))

    _blocked_neighborhood(blocked, swatch_cells, 0, 0)


def _place_room(
    grid: np.ndarray,
    expected: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    fill: int,
) -> bool:
    interior, walls, (ph, pw) = _room_geometry(mask)
    all_cells = interior | walls
    h, w = grid.shape

    positions = [(r, c) for r in range(1, h - ph) for c in range(1, w - pw)]
    random.shuffle(positions)
    for r0, c0 in positions:
        if not _can_place(blocked, all_cells, r0, c0):
            continue

        for r, c in walls:
            grid[r0 + r, c0 + c] = GREEN
            expected[r0 + r, c0 + c] = GREEN
        for r, c in interior:
            expected[r0 + r, c0 + c] = int(fill)

        _blocked_neighborhood(blocked, all_cells, r0, c0)
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    grid = np.full((h, w), AZURE, dtype=int)
    expected = grid.copy()
    blocked = np.zeros((h, w), dtype=bool)

    legend_colors = random.sample(PALETTE, LEGEND_COUNT)
    _place_legend(grid, blocked, legend_colors)
    expected[:, :] = grid

    ratios = [1, 2, 3, 4]
    ratios.extend(random.choices([1, 2, 3, 4], k=random.randint(0, 4)))
    random.shuffle(ratios)

    for ratio in ratios:
        mask = _mask_for_ratio(ratio)
        if _ratio(mask) != ratio:
            return None
        if not _place_room(grid, expected, blocked, mask, legend_colors[ratio - 1]):
            return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if int(np.sum(grid == AZURE)) <= int(np.sum(grid != AZURE)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 500 attempts")
