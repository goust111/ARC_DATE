from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREEN = 3
WALL = 5
ORANGE = 7

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLORED_LOOP_SIZES = ((3, 4), (4, 3), (4, 4))
EMPTY_LOOP_SIZES = ((3, 3), (3, 4), (4, 3))
MAX_GENERATION_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            queue = deque([(sr, sc)])
            seen[sr, sc] = True
            cells: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _ring_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = set()
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    return cells


def _interior_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _expanded_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 - pad
        or br1 + pad < ar0 - pad
        or ac1 + pad < bc0 - pad
        or bc1 + pad < ac0 - pad
    )


def _adjacent_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        out[rr, cc] = True
        for dr, dc in NEIGHBORS4:
            nr, nc = rr + dr, cc + dc
            if _in_bounds(mask.shape, nr, nc):
                out[nr, nc] = True
    return out


def _polyomino(allowed: np.ndarray, size: int) -> set[tuple[int, int]] | None:
    candidates = [tuple(map(int, rc)) for rc in np.argwhere(allowed)]
    random.shuffle(candidates)
    for seed in candidates[:80]:
        cells = {seed}
        while len(cells) < size:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if _in_bounds(allowed.shape, nr, nc) and allowed[nr, nc] and (nr, nc) not in cells:
                        frontier.extend([(nr, nc)] * (1 + sum((nr + er, nc + ec) in cells for er, ec in NEIGHBORS4)))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) == size:
            return cells
    return None


def _multisource_path(
    shape: tuple[int, int],
    starts: set[tuple[int, int]],
    targets: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    queue = deque()
    prev: dict[tuple[int, int], tuple[int, int] | None] = {}
    for cell in starts:
        queue.append(cell)
        prev[cell] = None
    while queue:
        cell = queue.popleft()
        if cell in targets:
            path = [cell]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            path.reverse()
            return path
        r, c = cell
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if nxt in prev or nxt in blocked or not _in_bounds(shape, nr, nc):
                continue
            prev[nxt] = cell
            queue.append(nxt)
    return None


def _compare_halves(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    out = np.zeros(top.shape, dtype=int)
    top_fg = top != BG
    bottom_fg = bottom != BG
    out[top_fg & bottom_fg] = WALL
    out[top_fg & ~bottom_fg] = GREEN
    out[~top_fg & bottom_fg] = ORANGE
    return out


def _classify_regions(compare_grid: np.ndarray) -> list[tuple[np.ndarray, set[int], bool]]:
    info: list[tuple[np.ndarray, set[int], bool]] = []
    for coords in _components(compare_grid != WALL):
        vals = {int(compare_grid[r, c]) for r, c in coords}
        marker_colors = vals - {BG}
        has_black = BG in vals
        info.append((coords, marker_colors, has_black))
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """Compare two stacked masks to create walls and markers, then flood-fill chambers with a single marker color.

    Input structure:
    - The grid is split into two equally sized square halves stacked vertically on a black background.
    - The top half uses green foreground pixels, the bottom half uses orange foreground pixels.
    - Overlapping foreground positions are not pre-labeled as walls; the wall structure must be inferred by comparing aligned cells in the two halves.

    Rule:
    - Compare aligned cells from the two halves to build an intermediate square grid: overlap becomes gray wall, top-only becomes green marker, bottom-only becomes orange marker, and double background stays black.
    - Treat gray as a barrier and find every 4-connected non-gray region.
    - If a region contains only green markers or only orange markers, flood-fill the whole region with that one marker color; regions with no markers or mixed marker colors stay unchanged.
    - Keep all gray wall pixels unchanged.

    Key insight:
    - One cell-by-cell comparison simultaneously creates structure, signals, and empty space: the overlap defines walls, one-sided pixels define the chamber's color cue, and the remaining black cells are what later get filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h % 2 == 0, "expected two stacked halves"
    n = h // 2
    assert n == w, "expected two stacked square halves"

    top = g[:n, :]
    bottom = g[n:, :]
    compare_grid = _compare_halves(top, bottom)
    out = compare_grid.copy()

    for coords in _components(compare_grid != WALL):
        colors = {int(compare_grid[r, c]) for r, c in coords} - {BG}
        if colors == {GREEN}:
            out[coords[:, 0], coords[:, 1]] = GREEN
        elif colors == {ORANGE}:
            out[coords[:, 0], coords[:, 1]] = ORANGE

    return out


def _random_box(n: int, size_choices: tuple[tuple[int, int], ...], existing: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    choices = list(size_choices)
    random.shuffle(choices)
    for h, w in choices:
        cells = [(r, c) for r in range(1, n - h) for c in range(1, n - w)]
        random.shuffle(cells)
        for r0, c0 in cells[:120]:
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if all(not _expanded_overlap(box, old, pad=1) for old in existing):
                return box
    return None


def _place_loops(n: int) -> list[dict] | None:
    loop_count = 3 if random.random() < 0.55 else 2
    roles = ["green", "orange"]
    if loop_count == 3:
        roles.append(random.choices(["empty", "green", "orange"], weights=[6, 2, 2], k=1)[0])

    loops: list[dict] = []
    boxes: list[tuple[int, int, int, int]] = []
    for role in roles:
        size_choices = COLORED_LOOP_SIZES if role != "empty" else EMPTY_LOOP_SIZES
        box = _random_box(n, size_choices, boxes)
        if box is None:
            return None
        boxes.append(box)
        loops.append({"role": role, "box": box, "ring": _ring_cells(box), "interior": _interior_cells(box)})

    random.shuffle(loops)
    return loops


def _build_compare_grid(n: int) -> np.ndarray | None:
    loops = _place_loops(n)
    if loops is None:
        return None

    interior_cells = {cell for loop in loops for cell in loop["interior"]}
    wall_cells: set[tuple[int, int]] = set()
    connected: set[tuple[int, int]] = set()

    for idx, loop in enumerate(loops):
        wall_cells |= loop["ring"]
        if idx == 0:
            connected |= loop["ring"]
            continue
        path = _multisource_path((n, n), connected, set(loop["ring"]), interior_cells)
        if path is None:
            return None
        wall_cells |= set(path)
        connected = wall_cells.copy()

    compare_grid = np.zeros((n, n), dtype=int)
    for r, c in wall_cells:
        compare_grid[r, c] = WALL

    for loop in loops:
        role = loop["role"]
        interior = list(loop["interior"])
        if role == "empty":
            continue
        if len(interior) < 2:
            return None
        marker = random.choice(interior)
        compare_grid[marker] = GREEN if role == "green" else ORANGE

    wall_mask = compare_grid == WALL
    blocked_inside = np.zeros((n, n), dtype=bool)
    for r, c in interior_cells:
        blocked_inside[r, c] = True
    eligible = (compare_grid == BG) & ~blocked_inside & ~_adjacent_mask(wall_mask)

    cluster_roles = ["green", "orange"]
    if random.random() < 0.35:
        cluster_roles.append("green")
    if random.random() < 0.35:
        cluster_roles.append("orange")

    painted = np.zeros((n, n), dtype=bool)
    for idx, role in enumerate(cluster_roles):
        size = random.randint(2, 4) if idx < 2 else random.randint(1, 3)
        allowed = eligible & ~_adjacent_mask(painted)
        cluster = _polyomino(allowed, size)
        if cluster is None:
            return None
        color = GREEN if role == "green" else ORANGE
        for r, c in cluster:
            compare_grid[r, c] = color
            painted[r, c] = True

    if len(_components(compare_grid == WALL)) != 1:
        return None

    transforms = [
        lambda x: x,
        lambda x: np.rot90(x, 1),
        lambda x: np.rot90(x, 2),
        lambda x: np.rot90(x, 3),
        lambda x: np.fliplr(x),
        lambda x: np.flipud(x),
        lambda x: np.transpose(x),
        lambda x: np.fliplr(np.transpose(x)),
    ]
    return np.array(random.choice(transforms)(compare_grid), dtype=int)


def _validate_semantics(inp: np.ndarray) -> None:
    n = inp.shape[1]
    top = inp[:n, :]
    bottom = inp[n:, :]

    assert set(map(int, np.unique(top))) <= {BG, GREEN}
    assert set(map(int, np.unique(bottom))) <= {BG, ORANGE}

    top_count = len(_components(top == GREEN))
    bottom_count = len(_components(bottom == ORANGE))
    assert 2 <= top_count <= 3
    assert 2 <= bottom_count <= 3

    compare_grid = _compare_halves(top, bottom)
    out = main(inp)
    regions = _classify_regions(compare_grid)
    assert len(_components(compare_grid == WALL)) == 1

    has_green_only = False
    has_orange_only = False
    has_mixed = False
    has_empty = False

    for coords, colors, has_black in regions:
        after = {int(out[r, c]) for r, c in coords}
        if colors == {GREEN}:
            has_green_only = True
            assert has_black, "single-green region must expose visible fill"
            assert after == {GREEN}
        elif colors == {ORANGE}:
            has_orange_only = True
            assert has_black, "single-orange region must expose visible fill"
            assert after == {ORANGE}
        elif colors == {GREEN, ORANGE}:
            has_mixed = True
            before = compare_grid[coords[:, 0], coords[:, 1]]
            assert np.array_equal(before, out[coords[:, 0], coords[:, 1]])
        else:
            assert colors == set()
            has_empty = True
            before = compare_grid[coords[:, 0], coords[:, 1]]
            assert np.array_equal(before, out[coords[:, 0], coords[:, 1]])

    assert has_green_only
    assert has_orange_only
    assert has_mixed
    assert has_empty or len(regions) >= 3
    assert np.any((compare_grid == BG) & (out == GREEN))
    assert np.any((compare_grid == BG) & (out == ORANGE))


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        n = random.randint(11, 15)
        compare_grid = _build_compare_grid(n)
        if compare_grid is None:
            continue

        top = np.where((compare_grid == WALL) | (compare_grid == GREEN), GREEN, BG)
        bottom = np.where((compare_grid == WALL) | (compare_grid == ORANGE), ORANGE, BG)
        inp = np.vstack([top, bottom]).astype(int)

        try:
            _validate_semantics(inp)
        except AssertionError:
            continue
        return inp

    raise RuntimeError("failed to generate a valid input")
