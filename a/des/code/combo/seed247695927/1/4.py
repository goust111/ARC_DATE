import random
from collections import Counter, deque

import numpy as np


AZURE = 8
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 9)
HOLE_FACTORS = {
    1: ((1, 1),),
    2: ((2, 1), (1, 2)),
    3: ((3, 1), (1, 3)),
    4: ((2, 2),),
    6: ((3, 2), (2, 3)),
}
LEGEND_HOLE_CHOICES = (1, 2, 3, 4, 6)
GENERATION_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _hole_count_from_component(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        patch[r - r0, c - c0] = True

    h, w = patch.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if patch[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = r == 0 or r == h - 1 or c == 0 or c == w - 1
            while q:
                cr, cc = q.popleft()
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if patch[nr, nc] or seen[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    if nr == 0 or nr == h - 1 or nc == 0 or nc == w - 1:
                        touches_border = True
                    q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the grid by the full horizontal grey divider.
    The lower panel is a hole-count legend: each colored hollow frame maps its
    number of enclosed background regions to its color.
    Recolor each grey frame in the upper panel to the color whose lower-panel
    legend object has the same hole count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    divider_rows = [r for r in range(g.shape[0]) if np.all(g[r] == GREY)]
    assert len(divider_rows) == 1, "expected exactly one full grey divider row"
    split = divider_rows[0]

    top = g[:split]
    bottom = g[split + 1 :]
    assert top.size > 0 and bottom.size > 0

    legend: dict[int, int] = {}
    lower_colors = [int(c) for c in np.unique(bottom) if int(c) not in (AZURE, GREY)]
    for color in lower_colors:
        comps = _components(bottom == color)
        assert len(comps) == 1, "each legend color should form one object"
        hole_count = _hole_count_from_component(bottom, comps[0], color)
        assert hole_count not in legend, "legend hole counts must be unique"
        legend[hole_count] = color
    assert legend, "legend must be non-empty"

    out = g.copy()
    for comp in _components(top == GREY):
        hole_count = _hole_count_from_component(top, comp, GREY)
        assert hole_count in legend, "upper-panel object hole count missing from legend"
        new_color = legend[hole_count]
        for r, c in comp:
            out[r, c] = new_color
    return out


def _make_partitioned_frame(color: int, holes: int, chamber_span_range: tuple[int, int]) -> np.ndarray:
    rows, cols = random.choice(HOLE_FACTORS[holes])
    chamber_h = random.randint(*chamber_span_range)
    chamber_w = random.randint(*chamber_span_range)
    height = 2 + rows * chamber_h + (rows - 1)
    width = 2 + cols * chamber_w + (cols - 1)

    patch = np.full((height, width), AZURE, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color

    cursor = 1 + chamber_w
    for _ in range(cols - 1):
        patch[:, cursor] = color
        cursor += 1 + chamber_w

    cursor = 1 + chamber_h
    for _ in range(rows - 1):
        patch[cursor, :] = color
        cursor += 1 + chamber_h

    if random.random() < 0.5:
        patch = patch.T
    return patch


def _can_place(occupied: np.ndarray, top: int, left: int, patch: np.ndarray) -> bool:
    h, w = patch.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(occupied.shape[0], top + h + 1)
    c1 = min(occupied.shape[1], left + w + 1)
    return not occupied[r0:r1, c0:c1].any()


def _place_patch(grid: np.ndarray, occupied: np.ndarray, row0: int, panel_top: int, patch: np.ndarray) -> bool:
    panel_h, panel_w = occupied.shape
    h, w = patch.shape
    if h > panel_h or w > panel_w:
        return False

    positions = [
        (r, c)
        for r in range(panel_h - h + 1)
        for c in range(panel_w - w + 1)
        if _can_place(occupied, r, c, patch)
    ]
    if not positions:
        return False

    top, left = random.choice(positions)
    mask = patch != AZURE
    target = grid[row0 + top : row0 + top + h, left : left + w]
    target[mask] = patch[mask]
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(panel_h, top + h + 1)
    c1 = min(panel_w, left + w + 1)
    occupied[r0:r1, c0:c1] = True
    return True


def _sample_top_holes(legend_holes: list[int], object_count: int) -> list[int]:
    holes = legend_holes[:]
    while len(holes) < object_count:
        holes.append(random.choice(legend_holes))
    random.shuffle(holes)
    return holes


def _gen_attempt() -> np.ndarray | None:
    legend_count = random.randint(2, 4)
    legend_holes = random.sample(LEGEND_HOLE_CHOICES, legend_count)
    top_count = random.randint(legend_count, legend_count + 2)
    top_holes = _sample_top_holes(legend_holes, top_count)

    legend_colors = random.sample(PALETTE, legend_count)
    hole_to_color = dict(zip(legend_holes, legend_colors))

    top_patches = [
        _make_partitioned_frame(GREY, holes, chamber_span_range=(1, 3))
        for holes in top_holes
    ]
    bottom_patches = [
        _make_partitioned_frame(hole_to_color[holes], holes, chamber_span_range=(1, 2))
        for holes in legend_holes
    ]

    max_top_h = max(p.shape[0] for p in top_patches)
    max_bottom_h = max(p.shape[0] for p in bottom_patches)
    max_w = max(max(p.shape[1] for p in top_patches), max(p.shape[1] for p in bottom_patches))

    grid_h = random.randint(max_top_h + max_bottom_h + 9, max_top_h + max_bottom_h + 18)
    grid_w = random.randint(max_w + 10, max_w + 24)
    divider = random.randint(max_top_h + 2, grid_h - max_bottom_h - 3)

    grid = np.full((grid_h, grid_w), AZURE, dtype=int)
    grid[divider, :] = GREY

    top_occ = np.zeros((divider, grid_w), dtype=bool)
    bottom_occ = np.zeros((grid_h - divider - 1, grid_w), dtype=bool)

    for patch in sorted(top_patches, key=lambda x: x.shape[0] * x.shape[1], reverse=True):
        if not _place_patch(grid, top_occ, 0, 0, patch):
            return None
    for patch in sorted(bottom_patches, key=lambda x: x.shape[0] * x.shape[1], reverse=True):
        if not _place_patch(grid, bottom_occ, divider + 1, divider + 1, patch):
            return None

    if Counter(grid.ravel()).most_common(1)[0][0] != AZURE:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
