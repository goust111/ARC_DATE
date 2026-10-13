from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GRAY = 5
SEED_COLORS = (1, 2, 4, 7)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))

GEN_RETRIES = 400
GRID_H_RANGE = (14, 19)
GRID_W_RANGE = (14, 19)
EXTRA_BLOCKER_PROB = 0.55


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIRS4 if int(connectivity) == 4 else DIRS8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _normalize_shape(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    coords = coords - coords.min(axis=0)
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    out = np.zeros((h, w), dtype=bool)
    out[coords[:, 0], coords[:, 1]] = True
    return out


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            transformed = _normalize_shape(np.rot90(base, k))
            key = (tuple(map(int, transformed.shape)), transformed.tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append(transformed)
    return out


SEED_BASE_SHAPES = [
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
]
SEED_SHAPES = [variant for base in SEED_BASE_SHAPES for variant in _unique_transforms(base)]

BLOCKER_BASE_SHAPES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
]
BLOCKER_SHAPES = [variant for base in BLOCKER_BASE_SHAPES for variant in _unique_transforms(base)]


def _component_dicts(grid: np.ndarray, color: int) -> list[dict]:
    comps = []
    for coords in components_mask(grid == int(color), connectivity=4):
        comps.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return comps


def _ordered_seed_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for color in SEED_COLORS:
        comps.extend(_component_dicts(grid, color))
    assert len(comps) == 2, "expected exactly two seed regions"
    colors = {int(comp["color"]) for comp in comps}
    assert len(colors) == 2, "seed colors must be distinct"
    for comp in comps:
        assert 3 <= int(comp["area"]) <= 5, "seed size must be 3..5"
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    return comps


def _segment_labels(blocked: np.ndarray, axis: int) -> np.ndarray:
    h, w = map(int, blocked.shape)
    labels = np.full((h, w), -1, dtype=int)
    sid = 0
    if axis == 1:
        for r in range(h):
            c = 0
            while c < w:
                if bool(blocked[r, c]):
                    c += 1
                    continue
                while c < w and not bool(blocked[r, c]):
                    labels[r, c] = sid
                    c += 1
                sid += 1
    else:
        for c in range(w):
            r = 0
            while r < h:
                if bool(blocked[r, c]):
                    r += 1
                    continue
                while r < h and not bool(blocked[r, c]):
                    labels[r, c] = sid
                    r += 1
                sid += 1
    return labels


def _visible_mask(grid: np.ndarray, seed: dict, *, use_blockers: bool = True) -> np.ndarray:
    blocked = grid == GRAY if use_blockers else np.zeros_like(grid, dtype=bool)
    row_labels = _segment_labels(blocked, axis=1)
    col_labels = _segment_labels(blocked, axis=0)
    row_ids = {int(row_labels[r, c]) for r, c in seed["coords"]}
    col_ids = {int(col_labels[r, c]) for r, c in seed["coords"]}
    row_seen = np.isin(row_labels, list(row_ids))
    col_seen = np.isin(col_labels, list(col_ids))
    return (row_seen | col_seen) & (grid == BG)


def _solve_from_analysis(grid: np.ndarray, seeds: list[dict]) -> tuple[np.ndarray, dict]:
    vis_a = _visible_mask(grid, seeds[0], use_blockers=True)
    vis_b = _visible_mask(grid, seeds[1], use_blockers=True)

    only_a = vis_a & ~vis_b
    only_b = vis_b & ~vis_a
    both = vis_a & vis_b

    out = np.array(grid, dtype=int, copy=True)
    color_a = int(seeds[0]["color"])
    color_b = int(seeds[1]["color"])
    out[only_a] = color_a
    out[only_b] = color_b

    parity = (np.indices(grid.shape).sum(axis=0) % 2) == 0
    out[both & parity] = color_a
    out[both & ~parity] = color_b

    return out, {"vis_a": vis_a, "vis_b": vis_b, "only_a": only_a, "only_b": only_b, "both": both}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition background cells by the four-direction line of sight of two seed regions, then use chessboard parity to arbitrate contested cells.

    Input structure:
    - The grid has black background, exactly two small 4-connected seed regions in distinct non-gray colors, and several gray 4-connected blocker components.
    - Each seed has 3..5 cells and blockers are the only cells that stop line-of-sight rays; seed and blocker cells remain fixed.
    - Coordinates use the ARC `(row, col)` convention, and seed A is the one whose bbox top-left comes earlier in row-major order.

    Rule:
    - For every black background cell, test visibility to each seed independently: along the same row or column, the cell sees that seed iff a ray reaches any seed cell before hitting a gray blocker or the grid edge.
    - If only seed A is visible, recolor the cell to A's color; if only seed B is visible, recolor it to B's color.
    - If both seeds are visible, recolor by checkerboard parity: even `(row + col)` goes to A, odd goes to B; cells visible to neither seed stay black.
    - Seed cells and gray blocker cells are copied through unchanged.

    Key insight:
    - The result is not a simple nearest-seed fill: blockers first carve two orthogonal visibility territories, and parity matters only on the nonempty overlap of those territories.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(get_bg(g)) == BG, "background must be black"
    seeds = _ordered_seed_components(g)
    out, _ = _solve_from_analysis(g, seeds)
    return out


def _shape_cells(mask: np.ndarray, origin: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, origin)
    return [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)]


def _can_place(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = map(int, grid.shape)
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != BG:
            return False
    for r, c in cells:
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BG and (nr, nc) not in cell_set:
                return False
    return True


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _seed_bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.array(cells, dtype=int)
    return bbox_coords(arr)


def _pick_seed_shapes(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    return rng.choice(SEED_SHAPES), rng.choice(SEED_SHAPES)


def _pick_layout(
    h: int, w: int, shape_a: np.ndarray, shape_b: np.ndarray, rng: random.Random
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    ha, wa = map(int, shape_a.shape)
    hb, wb = map(int, shape_b.shape)
    mode = rng.choices(["diagonal", "same_row", "same_col"], weights=[4, 3, 3])[0]

    if mode == "same_row":
        row_hi = h - max(ha, hb) - 4
        if row_hi < 2 or 2 > w // 2 - wa - 2 or w // 2 + 1 > w - wb - 3:
            return None
        row = rng.randint(2, row_hi)
        col_a = rng.randint(2, max(2, w // 2 - wa - 2))
        col_b = rng.randint(max(col_a + wa + 3, w // 2 + 1), w - wb - 3)
        return (row, col_a), (row, col_b)

    if mode == "same_col":
        col_hi = w - max(wa, wb) - 4
        if col_hi < 2 or 2 > h // 2 - ha - 2 or h // 2 + 1 > h - hb - 3:
            return None
        col = rng.randint(2, col_hi)
        row_a = rng.randint(2, max(2, h // 2 - ha - 2))
        row_b = rng.randint(max(row_a + ha + 3, h // 2 + 1), h - hb - 3)
        return (row_a, col), (row_b, col)

    row_a_hi = h // 2 - ha - 2
    col_a_hi = w // 2 - wa - 2
    row_b_lo = h // 2 + 1
    col_b_lo = w // 2 + 1
    if row_a_hi < 2 or col_a_hi < 2 or row_b_lo > h - hb - 3 or col_b_lo > w - wb - 3:
        return None
    row_a = rng.randint(2, row_a_hi)
    col_a = rng.randint(2, col_a_hi)
    row_b = rng.randint(row_b_lo, h - hb - 3)
    col_b = rng.randint(col_b_lo, w - wb - 3)
    return (row_a, col_a), (row_b, col_b)


def _functional_blockers(
    grid: np.ndarray,
    seed_a: list[tuple[int, int]],
    seed_b: list[tuple[int, int]],
    rng: random.Random,
) -> bool:
    h, w = map(int, grid.shape)
    r0a, c0a, r1a, c1a = _seed_bbox(seed_a)
    r0b, c0b, r1b, c1b = _seed_bbox(seed_b)

    if c0a < 2 or r0a < 2 or c1b > w - 3 or r1b > h - 3:
        return False

    west_row = rng.choice(seed_a)[0]
    north_col = rng.choice(seed_a)[1]
    east_row = rng.choice(seed_b)[0]
    south_col = rng.choice(seed_b)[1]

    blocker_cells = [
        (west_row, rng.randint(0, c0a - 2)),
        (rng.randint(0, r0a - 2), north_col),
        (east_row, rng.randint(c1b + 2, w - 1)),
        (rng.randint(r1b + 2, h - 1), south_col),
    ]
    if len(set(blocker_cells)) != 4:
        return False
    for r, c in blocker_cells:
        if not _can_place(grid, [(r, c)]):
            return False
        grid[r, c] = GRAY
    return True


def _place_extra_blockers(grid: np.ndarray, rng: random.Random) -> bool:
    target_total = 4 + int(rng.random() < EXTRA_BLOCKER_PROB) + int(rng.random() < EXTRA_BLOCKER_PROB / 2)
    current = len(components_mask(grid == GRAY, connectivity=4))
    tries = 0

    while current < target_total and tries < 120:
        tries += 1
        shape = rng.choice(BLOCKER_SHAPES[1:])
        bh, bw = map(int, shape.shape)
        r0 = rng.randint(0, grid.shape[0] - bh)
        c0 = rng.randint(0, grid.shape[1] - bw)
        cells = _shape_cells(shape, (r0, c0))
        if not _can_place(grid, cells):
            continue
        _paint_cells(grid, cells, GRAY)
        current += 1
    return 3 <= current <= 6


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        seeds = _ordered_seed_components(grid)
    except AssertionError:
        return False

    blocker_comps = components_mask(grid == GRAY, connectivity=4)
    if not (3 <= len(blocker_comps) <= 6):
        return False
    if not all(1 <= int(comp.shape[0]) <= 3 for comp in blocker_comps):
        return False

    try:
        out, info = _solve_from_analysis(grid, seeds)
    except AssertionError:
        return False

    only_a = info["only_a"]
    only_b = info["only_b"]
    both = info["both"]
    hidden = (grid == BG) & ~info["vis_a"] & ~info["vis_b"]

    if int(np.count_nonzero(only_a)) < 2 or int(np.count_nonzero(only_b)) < 2:
        return False
    if int(np.count_nonzero(hidden)) < 2:
        return False
    if int(np.count_nonzero(both)) < 2:
        return False

    contested_parities = (np.indices(grid.shape).sum(axis=0) % 2)[both]
    if set(map(int, contested_parities.tolist())) != {0, 1}:
        return False

    no_block_grid = np.where(grid == GRAY, BG, grid)
    no_block_out = main(no_block_grid)
    if np.array_equal(no_block_out, out):
        return False

    color_a = int(seeds[0]["color"])
    color_b = int(seeds[1]["color"])
    shortcut_a = out.copy()
    shortcut_b = out.copy()
    shortcut_a[both] = color_a
    shortcut_b[both] = color_b
    if np.array_equal(shortcut_a, out) or np.array_equal(shortcut_b, out):
        return False

    if np.array_equal(out, grid):
        return False
    if not np.array_equal(main(grid), out):
        return False
    return True


def _attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)

    shape_a, shape_b = _pick_seed_shapes(rng)
    layout = _pick_layout(h, w, shape_a, shape_b, rng)
    if layout is None:
        return None
    origin_a, origin_b = layout

    cells_a = _shape_cells(shape_a, origin_a)
    cells_b = _shape_cells(shape_b, origin_b)
    if not _can_place(grid, cells_a):
        return None
    _paint_cells(grid, cells_a, -1)
    if not _can_place(grid, cells_b):
        return None
    grid[:] = np.where(grid == -1, BG, grid)

    color_a, color_b = rng.sample(list(SEED_COLORS), 2)
    _paint_cells(grid, cells_a, color_a)
    _paint_cells(grid, cells_b, color_b)

    if not _functional_blockers(grid, cells_a, cells_b, rng):
        return None
    if not _place_extra_blockers(grid, rng):
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid visibility-parity ARC input")
