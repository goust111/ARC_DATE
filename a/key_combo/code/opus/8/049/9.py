from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
MARKER_COLORS = (1, 2, 3)
DIVIDER = 5
SHAPE_COLORS = (4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500


def _find_divider_rows(grid: np.ndarray) -> list[int]:
    width = grid.shape[1]
    allowed = set(MARKER_COLORS + (DIVIDER,))
    rows = [
        r
        for r in range(grid.shape[0])
        if int(np.count_nonzero(grid[r] == DIVIDER)) > width // 2
        and all(int(v) in allowed for v in grid[r])
    ]
    assert len(rows) >= 2, "expected at least two divider rows"
    return rows


def _find_divider_cols(grid: np.ndarray) -> list[int]:
    height = grid.shape[0]
    allowed = set(MARKER_COLORS + (DIVIDER,))
    cols = [
        c
        for c in range(grid.shape[1])
        if int(np.count_nonzero(grid[:, c] == DIVIDER)) > height // 2
        and all(int(v) in allowed for v in grid[:, c])
    ]
    assert len(cols) >= 2, "expected at least two divider cols"
    return cols


def _layout(grid: np.ndarray) -> tuple[list[int], list[int]]:
    div_rows = _find_divider_rows(grid)
    div_cols = _find_divider_cols(grid)
    assert div_rows[0] == 0 and div_rows[-1] == grid.shape[0] - 1
    assert div_cols[0] == 0 and div_cols[-1] == grid.shape[1] - 1
    assert all(b - a >= 2 for a, b in zip(div_rows, div_rows[1:])), "regions must be non-empty"
    assert all(b - a >= 2 for a, b in zip(div_cols, div_cols[1:])), "regions must be non-empty"
    assert (len(div_cols) - 1) % 2 == 0, "region column count must be even"
    return div_rows, div_cols


def _region_bounds(div_rows: list[int], div_cols: list[int], rr: int, cc: int) -> tuple[int, int, int, int]:
    return div_rows[rr] + 1, div_rows[rr + 1] - 1, div_cols[cc] + 1, div_cols[cc + 1] - 1


def _region_patch(grid: np.ndarray, div_rows: list[int], div_cols: list[int], rr: int, cc: int) -> np.ndarray:
    r0, r1, c0, c1 = _region_bounds(div_rows, div_cols, rr, cc)
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _corner_colors(grid: np.ndarray, div_rows: list[int], div_cols: list[int], rr: int, cc: int) -> tuple[int, int, int, int]:
    return (
        int(grid[div_rows[rr], div_cols[cc]]),
        int(grid[div_rows[rr], div_cols[cc + 1]]),
        int(grid[div_rows[rr + 1], div_cols[cc]]),
        int(grid[div_rows[rr + 1], div_cols[cc + 1]]),
    )


def _active_mask(grid: np.ndarray, div_rows: list[int], div_cols: list[int]) -> np.ndarray:
    rows = len(div_rows) - 1
    cols = len(div_cols) - 1
    active = np.zeros((rows, cols), dtype=bool)
    for rr in range(rows):
        for cc in range(cols):
            corners = _corner_colors(grid, div_rows, div_cols, rr, cc)
            active[rr, cc] = len(set(corners)) == 1 and corners[0] in MARKER_COLORS
    return active


def _active_side(active: np.ndarray) -> str:
    coords = np.argwhere(active)
    assert len(coords) > 0, "expected at least one active region"
    half = active.shape[1] // 2
    cols = coords[:, 1]
    left = bool(np.all(cols < half))
    right = bool(np.all(cols >= half))
    assert left ^ right, "active regions must lie on one side of the vertical region-axis"
    return "left" if left else "right"


def _source_cols(side: str, col_count: int) -> range:
    half = col_count // 2
    return range(0, half) if side == "left" else range(half, col_count)


def _target_col(col_count: int, cc: int) -> int:
    return col_count - 1 - cc


def _apply_rule(grid: np.ndarray, *, mirror_all_source: bool, flip: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    div_rows, div_cols = _layout(g)
    active = _active_mask(g, div_rows, div_cols)
    side = _active_side(active)
    rows, cols = active.shape

    out = g.copy()
    for rr in range(rows):
        for cc in range(cols):
            if not active[rr, cc]:
                r0, r1, c0, c1 = _region_bounds(div_rows, div_cols, rr, cc)
                out[r0 : r1 + 1, c0 : c1 + 1] = BACKGROUND

    for rr in range(rows):
        for cc in _source_cols(side, cols):
            if not mirror_all_source and not active[rr, cc]:
                continue
            if mirror_all_source or active[rr, cc]:
                tgt = _target_col(cols, cc)
                if not mirror_all_source:
                    assert not active[rr, tgt], "mirror target cannot also be active"
                src_patch = _region_patch(g, div_rows, div_cols, rr, cc)
                patch = np.fliplr(src_patch) if flip else src_patch.copy()
                r0, r1, c0, c1 = _region_bounds(div_rows, div_cols, rr, tgt)
                out[r0 : r1 + 1, c0 : c1 + 1] = patch
    return out


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    height, width = shape
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(int(x) for x in coords[0])
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c, mask.shape):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(coords)


def _random_connected_mask(height: int, width: int, size: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    seed = (random.randrange(height), random.randrange(width))
    selected = {seed}
    frontier = set(_neighbors4(seed[0], seed[1], (height, width)))
    while len(selected) < size:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in selected:
            continue
        selected.add(cell)
        for nb in _neighbors4(cell[0], cell[1], (height, width)):
            if nb not in selected:
                frontier.add(nb)
    for r, c in selected:
        mask[r, c] = True
    return mask


def _make_patch(height: int, width: int, *, require_lr_asym: bool, forbidden: np.ndarray | None = None) -> np.ndarray:
    max_area = min(4, height * width)
    for _ in range(200):
        size = random.randint(1, max_area)
        mask = _random_connected_mask(height, width, size)
        if require_lr_asym and np.array_equal(mask, np.fliplr(mask)):
            continue
        color = random.choice(SHAPE_COLORS)
        patch = np.zeros((height, width), dtype=int)
        patch[mask] = color
        if forbidden is not None and np.array_equal(patch, forbidden):
            continue
        return patch
    raise RuntimeError("failed to build a suitable region patch")


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _choose_vertex_colors(active: np.ndarray) -> np.ndarray | None:
    rows, cols = active.shape
    dsu = _DSU((rows + 1) * (cols + 1))

    def vid(r: int, c: int) -> int:
        return r * (cols + 1) + c

    for rr in range(rows):
        for cc in range(cols):
            if active[rr, cc]:
                corners = [vid(rr, cc), vid(rr, cc + 1), vid(rr + 1, cc), vid(rr + 1, cc + 1)]
                anchor = corners[0]
                for other in corners[1:]:
                    dsu.union(anchor, other)

    for rr in range(rows):
        for cc in range(cols):
            roots = {dsu.find(vid(rr, cc)), dsu.find(vid(rr, cc + 1)), dsu.find(vid(rr + 1, cc)), dsu.find(vid(rr + 1, cc + 1))}
            if not active[rr, cc] and len(roots) == 1:
                return None

    roots = sorted({dsu.find(i) for i in range((rows + 1) * (cols + 1))})
    for _ in range(80):
        root_colors = {root: random.choice(MARKER_COLORS) for root in roots}
        if len(set(root_colors.values())) < 2:
            continue
        ok = True
        for rr in range(rows):
            for cc in range(cols):
                corners = (
                    root_colors[dsu.find(vid(rr, cc))],
                    root_colors[dsu.find(vid(rr, cc + 1))],
                    root_colors[dsu.find(vid(rr + 1, cc))],
                    root_colors[dsu.find(vid(rr + 1, cc + 1))],
                )
                is_active = len(set(corners)) == 1
                if is_active != bool(active[rr, cc]):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        vertices = np.zeros((rows + 1, cols + 1), dtype=int)
        for rr in range(rows + 1):
            for cc in range(cols + 1):
                vertices[rr, cc] = root_colors[dsu.find(vid(rr, cc))]
        return vertices
    return None


def _layout_choices() -> list[tuple[int, int, int, int]]:
    choices: list[tuple[int, int, int, int]] = []
    for rows in (2, 3):
        for cols in (2, 4):
            for region_h in range(2, 7):
                height = rows * (region_h + 1) + 1
                if not (11 <= height <= 15):
                    continue
                for region_w in range(2, 7):
                    width = cols * (region_w + 1) + 1
                    if 11 <= width <= 15:
                        choices.append((rows, cols, region_h, region_w))
    return choices


LAYOUT_CHOICES = _layout_choices()


def _build_grid(rows: int, cols: int, region_h: int, region_w: int, vertices: np.ndarray, patches: list[list[np.ndarray]]) -> np.ndarray:
    height = rows * (region_h + 1) + 1
    width = cols * (region_w + 1) + 1
    grid = np.full((height, width), BACKGROUND, dtype=int)
    div_rows = [i * (region_h + 1) for i in range(rows + 1)]
    div_cols = [i * (region_w + 1) for i in range(cols + 1)]
    grid[div_rows, :] = DIVIDER
    grid[:, div_cols] = DIVIDER

    for vr in range(rows + 1):
        for vc in range(cols + 1):
            grid[div_rows[vr], div_cols[vc]] = int(vertices[vr, vc])

    for rr in range(rows):
        for cc in range(cols):
            r0, r1, c0, c1 = _region_bounds(div_rows, div_cols, rr, cc)
            grid[r0 : r1 + 1, c0 : c1 + 1] = patches[rr][cc]

    return grid


def _choose_active_regions(rows: int, cols: int) -> tuple[np.ndarray, str]:
    side = random.choice(("left", "right"))
    half = cols // 2
    source_cells = [
        (rr, cc)
        for rr in range(rows)
        for cc in (range(0, half) if side == "left" else range(half, cols))
    ]
    active_count = random.randint(1, len(source_cells) - 1)
    chosen = set(random.sample(source_cells, active_count))
    active = np.zeros((rows, cols), dtype=bool)
    for rr, cc in chosen:
        active[rr, cc] = True
    return active, side


def _validate_structure(grid: np.ndarray) -> bool:
    div_rows, div_cols = _layout(grid)
    active = _active_mask(grid, div_rows, div_cols)
    _active_side(active)

    for rr in range(len(div_rows) - 1):
        for cc in range(len(div_cols) - 1):
            patch = _region_patch(grid, div_rows, div_cols, rr, cc)
            if np.any(patch == DIVIDER) or np.any(np.isin(patch, MARKER_COLORS)):
                return False
            mask = patch != BACKGROUND
            if not (1 <= int(mask.sum()) <= 4):
                return False
            if not _is_connected(mask):
                return False
    return True


def _validate_semantics(grid: np.ndarray) -> bool:
    div_rows, div_cols = _layout(grid)
    active = _active_mask(grid, div_rows, div_cols)
    side = _active_side(active)
    rows, cols = active.shape
    half_cols = list(_source_cols(side, cols))

    if not np.any(~active[:, half_cols]):
        return False

    out = main(grid)
    erase_only = _apply_rule(grid, mirror_all_source=False, flip=True)
    for rr in range(rows):
        for cc in range(cols):
            if active[rr, cc]:
                tgt = _target_col(cols, cc)
                src = _region_patch(grid, div_rows, div_cols, rr, cc)
                mirrored = np.fliplr(src)
                tgt_in = _region_patch(grid, div_rows, div_cols, rr, tgt)
                tgt_out = _region_patch(out, div_rows, div_cols, rr, tgt)
                if np.array_equal(src, mirrored):
                    return False
                if np.array_equal(tgt_in, mirrored):
                    return False
                if not np.array_equal(tgt_out, mirrored):
                    return False

    erase_only = erase_only.copy()
    for rr in range(rows):
        for cc in range(cols):
            if active[rr, cc]:
                tgt = _target_col(cols, cc)
                r0, r1, c0, c1 = _region_bounds(div_rows, div_cols, rr, tgt)
                erase_only[r0 : r1 + 1, c0 : c1 + 1] = BACKGROUND

    all_source = _apply_rule(grid, mirror_all_source=True, flip=True)
    no_flip = _apply_rule(grid, mirror_all_source=False, flip=False)
    if np.array_equal(out, erase_only):
        return False
    if np.array_equal(out, all_source):
        return False
    if np.array_equal(out, no_flip):
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _generate_candidate() -> np.ndarray | None:
    rows, cols, region_h, region_w = random.choice(LAYOUT_CHOICES)
    active, side = _choose_active_regions(rows, cols)
    if _active_side(active) != side:
        return None

    vertices = _choose_vertex_colors(active)
    if vertices is None:
        return None

    patches: list[list[np.ndarray | None]] = [[None for _ in range(cols)] for _ in range(rows)]

    for rr in range(rows):
        for cc in _source_cols(side, cols):
            if active[rr, cc]:
                src = _make_patch(region_h, region_w, require_lr_asym=True)
                patches[rr][cc] = src
                tgt = _target_col(cols, cc)
                patches[rr][tgt] = _make_patch(region_h, region_w, require_lr_asym=False, forbidden=np.fliplr(src))

    for rr in range(rows):
        for cc in range(cols):
            if patches[rr][cc] is None:
                patches[rr][cc] = _make_patch(region_h, region_w, require_lr_asym=False)

    final_patches = [[patches[rr][cc] for cc in range(cols)] for rr in range(rows)]
    grid = _build_grid(rows, cols, region_h, region_w, vertices, final_patches)

    if not _validate_structure(grid):
        return None
    if not _validate_semantics(grid):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Activate unanimous-corner regions, mirror their contents across the region-grid midline, and erase the rest.

    Input structure:
    - Grey divider bars partition the canvas into a rectangular grid of regions with an even number of region columns.
    - Each region is surrounded by four single-pixel corner markers placed on the divider intersections, colored blue, red, or green.
    - Every region interior contains a small 4-connected shape made of 1 to 4 non-marker pixels.

    Rule:
    - Find the grey divider rows and columns to recover the region layout.
    - A region is active exactly when its four surrounding corner markers all have the same color.
    - Erase every inactive region interior to black, keep every active region interior unchanged, and preserve all divider and marker pixels.
    - For each active region, copy its interior into the horizontally symmetric region on the opposite side of the region-grid vertical midline, using a left-right flip inside the target region.

    Key insight:
    - The unanimous corner markers are a vote that selects which side supplies content, and the even region grid makes the mirror target of each active region unambiguous.
    """
    return _apply_rule(grid, mirror_all_source=False, flip=True)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
