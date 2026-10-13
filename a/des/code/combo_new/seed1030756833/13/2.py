import random
from collections import deque

import numpy as np


AZURE = 8
BLACK = 0
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(sr, sc)])
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    comps.sort(key=lambda pts: (min(r for r, _ in pts), min(c for _, c in pts), len(pts)))
    return [np.array(comp, dtype=int) for comp in comps]


def _bbox(coords: np.ndarray | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    if isinstance(coords, set):
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        return min(rows), min(cols), max(rows), max(cols)
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _parse_legend(row: np.ndarray) -> dict[int, int]:
    legend: dict[int, int] = {}
    c = 0
    w = len(row)
    while c < w:
        value = int(row[c])
        if value in (AZURE, BLACK):
            c += 1
            continue
        if c + 1 >= w:
            break
        target = int(row[c + 1])
        if target in (AZURE, BLACK):
            c += 1
            continue
        legend[value] = target
        c += 2
        if c < w and int(row[c]) == BLACK:
            c += 1
    return legend


def _is_solid_rectangle(cells: set[tuple[int, int]]) -> tuple[bool, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(cells)
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1), (r0, c0, r1, c1)


def _merge_candidate(
    moved: np.ndarray,
    fixed: np.ndarray,
    union_bbox: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], set[tuple[int, int]]] | None:
    r0, c0, r1, c1 = union_bbox
    fixed_cells = {(int(r), int(c)) for r, c in fixed}
    rotated = {(r0 + r1 - int(r), c0 + c1 - int(c)) for r, c in moved}
    if rotated & fixed_cells:
        return None
    merged = rotated | fixed_cells
    ok, rect = _is_solid_rectangle(merged)
    if not ok:
        return None
    return rect, merged


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored scattered fragments, rotate one fragment by 180 degrees, and recolor the completed rectangles from the bottom legend.

    Input structure:
    - The background is azure.
    - The bottom row is a legend made of adjacent two-cell pairs: fragment color then target color, with black separators between pairs.
    - Above the legend, each legend fragment color appears as exactly two monochrome 4-connected fragments.
    - For each same-colored pair, rotating one fragment around the geometric center of the pair's joint bounding box lets it join the other fragment as a solid rectangle.

    Rule:
    - Read the bottom legend left to right to map each fragment color to its target recolor.
    - For each fragment color, find its two 4-connected components above the legend row and compute their joint bounding box.
    - Try the two possible 180-degree rotations around that joint box center, using the first top-left component when both are valid.
    - Clear the original fragments above the legend, then draw the completed solid rectangle in the target color.
    - Preserve the legend row unchanged.

    Key insight:
    - The bottom row supplies a color mapping, while the two separated same-colored fragments supply a spatial rotation-and-merge relation; both clues are needed to know the output color and location.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, _w = g.shape
    assert h >= 2, "grid must contain a work area and a legend row"

    legend = _parse_legend(g[-1])
    assert legend, "expected at least one legend pair"

    plans: list[tuple[tuple[int, int, int, int], int]] = []
    cells_to_clear: list[np.ndarray] = []
    work = g[:-1]

    for color, target in legend.items():
        comps = _components(work == color)
        assert len(comps) == 2, "each legend fragment color must have exactly two fragments"
        abs_comps = [comp.copy() for comp in comps]
        for comp in abs_comps:
            cells_to_clear.append(comp)

        all_coords = np.vstack(abs_comps)
        union_bbox = _bbox(all_coords)
        candidates = []
        for idx, moved in enumerate(abs_comps):
            fixed = abs_comps[1 - idx]
            cand = _merge_candidate(moved, fixed, union_bbox)
            if cand is not None:
                rect, merged = cand
                candidates.append((idx, rect, merged))
        assert candidates, "a fragment pair must admit a 180-degree rectangular merge"

        # Components are already ordered by their visible top-left position.
        _idx, rect, _merged = min(candidates, key=lambda item: (item[0], item[1]))
        plans.append((rect, int(target)))

    for comp in cells_to_clear:
        out[comp[:, 0], comp[:, 1]] = AZURE

    occupied = np.zeros(g.shape, dtype=bool)
    for r0, c0, r1, c1, target in [(a, b, c, d, e) for (a, b, c, d), e in plans]:
        assert r1 < h - 1, "merged rectangles must stay above the legend row"
        region = occupied[r0 : r1 + 1, c0 : c1 + 1]
        assert not region.any(), "completed rectangles must not overlap"
        out[r0 : r1 + 1, c0 : c1 + 1] = target
        region[:, :] = True

    out[-1] = g[-1]
    return out


# ---------------- generator ----------------

GRID_ROWS = (16, 30)
GRID_COLS = (16, 30)
PAIR_COUNT_RANGE = (2, 4)
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 9]
PLACEMENT_RETRIES = 300
GENERATION_RETRIES = 400


BASE_A = {
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 2),
    (3, 2),
    (4, 2),
}
BASE_B = {
    (0, 4),
    (0, 5),
    (1, 4),
    (1, 5),
    (2, 4),
    (2, 5),
}
BASE_SHAPE = (5, 6)


def _transform_cells(
    cells: set[tuple[int, int]],
    shape: tuple[int, int],
    rot: int,
    flip: bool,
) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    h, w = shape
    pts = set(cells)
    for _ in range(rot % 4):
        pts = {(c, h - 1 - r) for r, c in pts}
        h, w = w, h
    if flip:
        pts = {(r, w - 1 - c) for r, c in pts}
    return pts, (h, w)


def _scale_cells(cells: set[tuple[int, int]], scale: int) -> set[tuple[int, int]]:
    if scale == 1:
        return set(cells)
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(scale):
            for dc in range(scale):
                out.add((r * scale + dr, c * scale + dc))
    return out


def _normalize_pair(
    a_cells: set[tuple[int, int]],
    b_cells: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]:
    all_cells = a_cells | b_cells
    r0, c0, r1, c1 = _bbox(all_cells)
    a = {(r - r0, c - c0) for r, c in a_cells}
    b = {(r - r0, c - c0) for r, c in b_cells}
    return a, b, (r1 - r0 + 1, c1 - c0 + 1)


def _sample_pair_template() -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]:
    rot = random.randrange(4)
    flip = random.random() < 0.5
    scale = 2 if random.random() < 0.25 else 1
    a, shape = _transform_cells(BASE_A, BASE_SHAPE, rot, flip)
    b, _shape = _transform_cells(BASE_B, BASE_SHAPE, rot, flip)
    a = _scale_cells(a, scale)
    b = _scale_cells(b, scale)
    return _normalize_pair(a, b)


def _padded_box(box: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _stamp_cells(grid: np.ndarray, cells: set[tuple[int, int]], r0: int, c0: int, color: int) -> None:
    for r, c in cells:
        grid[r0 + r, c0 + c] = color


def _draw_legend(grid: np.ndarray, fragment_colors: list[int], target_colors: list[int]) -> None:
    legend_len = 3 * len(fragment_colors) - 1
    start = random.randint(0, grid.shape[1] - legend_len)
    row = grid.shape[0] - 1
    c = start
    for i, (frag, target) in enumerate(zip(fragment_colors, target_colors)):
        grid[row, c] = frag
        grid[row, c + 1] = target
        c += 2
        if i + 1 < len(fragment_colors):
            grid[row, c] = BLACK
            c += 1


def _valid_component_count(grid: np.ndarray, colors: list[int]) -> bool:
    work = grid[:-1]
    for color in colors:
        if len(_components(work == color)) != 2:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    needed = 2 * n_pairs
    colors = random.sample(COLOR_POOL, needed)
    fragment_colors = colors[:n_pairs]
    target_colors = colors[n_pairs:]

    rows = random.randint(*GRID_ROWS)
    cols = random.randint(max(GRID_COLS[0], 3 * n_pairs - 1), GRID_COLS[1])
    grid = np.full((rows, cols), AZURE, dtype=int)
    _draw_legend(grid, fragment_colors, target_colors)

    occupied_boxes: list[tuple[int, int, int, int]] = []
    for color in fragment_colors:
        a_cells, b_cells, (ph, pw) = _sample_pair_template()
        if ph > rows - 2 or pw > cols - 2:
            return None

        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(0, rows - 2 - ph)
            c0 = random.randint(0, cols - pw)
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            padded = _padded_box(box, 1, rows - 2, cols - 1)
            if any(_boxes_overlap(padded, old) for old in occupied_boxes):
                continue
            _stamp_cells(grid, a_cells, r0, c0, color)
            _stamp_cells(grid, b_cells, r0, c0, color)
            occupied_boxes.append(padded)
            break
        else:
            return None

    if not _valid_component_count(grid, fragment_colors):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotation-merge legend input")
