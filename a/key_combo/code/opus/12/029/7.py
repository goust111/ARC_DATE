from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

BLOCK_COLORS = (BLUE, RED, GREEN, YELLOW)
CORNERS = ((0, 0), (0, -1), (-1, 0), (-1, -1))
INF = 10**9
GEN_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _corner_coords(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    return [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]


def _has_grey_ring(grid: np.ndarray, r0: int, c0: int) -> bool:
    if r0 < 1 or c0 < 1 or r0 + 2 >= grid.shape[0] or c0 + 2 >= grid.shape[1]:
        return False
    for r in range(r0 - 1, r0 + 3):
        for c in range(c0 - 1, c0 + 3):
            inside_block = r0 <= r <= r0 + 1 and c0 <= c <= c0 + 1
            if inside_block:
                continue
            if int(grid[r, c]) != GREY:
                return False
    return True


def _find_block(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    candidates = []
    for r0 in range(g.shape[0] - 1):
        for c0 in range(g.shape[1] - 1):
            patch = g[r0 : r0 + 2, c0 : c0 + 2]
            values = [int(v) for v in patch.ravel()]
            if set(values) != set(BLOCK_COLORS):
                continue
            if len(set(values)) != 4:
                continue
            if _has_grey_ring(g, r0, c0):
                candidates.append((r0, c0))
    assert len(candidates) == 1, "expected exactly one ring-enclosed 2x2 color block"
    return candidates[0]


def _prepare_seeded_grid(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    r0, c0 = _find_block(g)
    block = g[r0 : r0 + 2, c0 : c0 + 2].copy()

    seeded = g.copy()
    seeded[r0 : r0 + 2, c0 : c0 + 2] = BLACK

    corner_colors = [
        int(block[0, 0]),
        int(block[0, 1]),
        int(block[1, 0]),
        int(block[1, 1]),
    ]
    for (r, c), color in zip(_corner_coords(g.shape), corner_colors):
        seeded[r, c] = color
    return seeded, corner_colors


def _bfs_dist(traversable: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    h, w = traversable.shape
    dist = np.full((h, w), INF, dtype=int)
    sr, sc = seed
    if not traversable[sr, sc]:
        return dist

    q = deque([seed])
    dist[sr, sc] = 0
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and traversable[rr, cc] and dist[rr, cc] == INF:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def _component_size(traversable: np.ndarray, seed: tuple[int, int]) -> int:
    h, w = traversable.shape
    sr, sc = seed
    if not traversable[sr, sc]:
        return 0

    seen = {seed}
    q = deque([seed])
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and traversable[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen)


def _analyze_flood(seeded: np.ndarray, corner_colors: list[int]) -> dict:
    traversable = seeded != GREY
    corners = _corner_coords(seeded.shape)
    dists = np.stack([_bfs_dist(traversable, corner) for corner in corners], axis=0)
    finite = dists < INF
    min_dist = dists.min(axis=0)
    candidates = finite & (dists == min_dist[None, :, :])
    candidate_counts = candidates.sum(axis=0)

    unique_areas = [
        int(np.count_nonzero(candidates[idx] & (candidate_counts == 1)))
        for idx in range(4)
    ]

    owner = np.full(seeded.shape, -1, dtype=int)
    tie_records = []
    for r, c in np.argwhere(candidate_counts > 0):
        seed_ids = [int(idx) for idx in np.flatnonzero(candidates[:, r, c])]
        if len(seed_ids) == 1:
            owner[r, c] = seed_ids[0]
            continue

        best_area = max(unique_areas[idx] for idx in seed_ids)
        leaders = [idx for idx in seed_ids if unique_areas[idx] == best_area]
        winner = min(leaders, key=lambda idx: corner_colors[idx])
        owner[r, c] = winner
        tie_records.append(
            {
                "pos": (int(r), int(c)),
                "candidates": tuple(seed_ids),
                "leaders": tuple(leaders),
                "winner": int(winner),
            }
        )

    final_areas = [int(np.count_nonzero(owner == idx)) for idx in range(4)]
    return {
        "owner": owner,
        "tie_mask": candidate_counts > 1,
        "unique_areas": unique_areas,
        "final_areas": final_areas,
        "tie_records": tie_records,
    }


def _render_output(seeded: np.ndarray, corner_colors: list[int], owner: np.ndarray) -> np.ndarray:
    out = seeded.copy()
    for idx, color in enumerate(corner_colors):
        out[owner == idx] = int(color)
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    seeded, corner_colors = _prepare_seeded_grid(grid)
    analysis = _analyze_flood(seeded, corner_colors)
    return _render_output(seeded, corner_colors, analysis["owner"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the 2x2 color block to the four corners, then let those corners simultaneously claim the black maze by flood-fill.

    Input structure:
    - A black grid contains exactly one 2x2 block using the four colors blue, red, green, and yellow.
    - That 2x2 block is fully enclosed by a one-cell-thick grey wall ring, so it serves only as a color key.
    - Additional grey wall pixels form obstacle corridors and chambers while the four outer corners remain black.

    Rule:
    - Read the 2x2 block by relative position: its top-left, top-right, bottom-left, and bottom-right colors move to the matching grid corners.
    - Clear the original 2x2 block to black, keep every grey wall unchanged, and start a simultaneous 4-neighbor flood from the four colored corners through non-grey cells.
    - A cell claimed first by one corner takes that corner's color.
    - If multiple corner wavefronts reach a cell in the same minimum number of steps, assign that tied cell to the seed with the larger strictly-first-claimed area; if those areas are equal, prefer the smaller seed color id.

    Key insight:
    - The corner colors are both the moved payload and the flood seeds, and conflict cells can only be resolved by combining shortest-path reachability with a global comparison of the seeds' territory sizes.
    """
    return _solve(np.asarray(grid, dtype=int))


def _place_block_and_ring(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    r0 = random.randint(max(2, h // 2 - 2), min(h - 4, h // 2))
    c0 = random.randint(max(2, w // 2 - 2), min(w - 4, w // 2))

    colors = list(BLOCK_COLORS)
    random.shuffle(colors)
    grid[r0 : r0 + 2, c0 : c0 + 2] = np.array(colors, dtype=int).reshape(2, 2)

    grid[r0 - 1 : r0 + 3, c0 - 1] = GREY
    grid[r0 - 1 : r0 + 3, c0 + 2] = GREY
    grid[r0 - 1, c0 - 1 : c0 + 3] = GREY
    grid[r0 + 2, c0 - 1 : c0 + 3] = GREY
    return r0, c0


def _protected_mask(shape: tuple[int, int], block_rc: tuple[int, int]) -> np.ndarray:
    h, w = shape
    r0, c0 = block_rc
    protected = np.zeros((h, w), dtype=bool)
    protected[r0 - 1 : r0 + 3, c0 - 1 : c0 + 3] = True
    for r, c in _corner_coords(shape):
        protected[r, c] = True
    return protected


def _paint_wall_cells(grid: np.ndarray, protected: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = grid.shape
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w and not protected[r, c]:
            grid[r, c] = GREY


def _draw_vertical_wall(
    grid: np.ndarray,
    protected: np.ndarray,
    col: int,
    gap_rows: list[int],
) -> None:
    cells = [(r, col) for r in range(1, grid.shape[0] - 1) if r not in gap_rows]
    _paint_wall_cells(grid, protected, cells)


def _draw_horizontal_wall(
    grid: np.ndarray,
    protected: np.ndarray,
    row: int,
    gap_cols: list[int],
) -> None:
    cells = [(row, c) for c in range(1, grid.shape[1] - 1) if c not in gap_cols]
    _paint_wall_cells(grid, protected, cells)


def _random_segment(
    shape: tuple[int, int],
    protected: np.ndarray,
    mirror: str | None,
) -> list[tuple[int, int]] | None:
    h, w = shape
    vertical = random.random() < 0.5
    length = random.randint(2, 4)

    if mirror == "lr":
        if vertical:
            r0 = random.randint(1, h - length - 1)
            c0 = random.randint(1, max(1, w // 2 - 3))
            left = [(r0 + k, c0) for k in range(length)]
            right = [(r, w - 1 - c) for r, c in left]
        else:
            r0 = random.randint(1, h - 2)
            c0 = random.randint(1, max(1, w // 2 - length - 1))
            left = [(r0, c0 + k) for k in range(length)]
            right = [(r, w - 1 - c) for r, c in left]
        cells = left + right
    elif mirror == "ud":
        if vertical:
            r0 = random.randint(1, max(1, h // 2 - length - 1))
            c0 = random.randint(1, w - 2)
            top = [(r0 + k, c0) for k in range(length)]
            bottom = [(h - 1 - r, c) for r, c in top]
        else:
            r0 = random.randint(1, max(1, h // 2 - 2))
            c0 = random.randint(1, w - length - 1)
            top = [(r0, c0 + k) for k in range(length)]
            bottom = [(h - 1 - r, c) for r, c in top]
        cells = top + bottom
    else:
        if vertical:
            r0 = random.randint(1, h - length - 1)
            c0 = random.randint(1, w - 2)
            cells = [(r0 + k, c0) for k in range(length)]
        else:
            r0 = random.randint(1, h - 2)
            c0 = random.randint(1, w - length - 1)
            cells = [(r0, c0 + k) for k in range(length)]

    if any(protected[r, c] for r, c in cells):
        return None
    return cells


def _build_asymmetric_walls(grid: np.ndarray, protected: np.ndarray) -> None:
    h, w = grid.shape
    valid_cols = [c for c in range(2, w - 2) if not protected[:, c].any()]
    valid_rows = [r for r in range(2, h - 2) if not protected[r, :].any()]
    col = random.choice(valid_cols)
    row = random.choice(valid_rows)

    gap_rows = random.sample(range(2, h - 2), 2)
    gap_cols = random.sample(range(2, w - 2), 2)
    _draw_vertical_wall(grid, protected, col, gap_rows)
    _draw_horizontal_wall(grid, protected, row, gap_cols)

    for _ in range(random.randint(1, 3)):
        cells = _random_segment(grid.shape, protected, mirror=None)
        if cells is not None:
            _paint_wall_cells(grid, protected, cells)


def _build_symmetric_walls(grid: np.ndarray, protected: np.ndarray) -> None:
    h, w = grid.shape
    if random.random() < 0.5:
        mirror = "lr"
        valid_rows = [r for r in range(2, h - 2) if not protected[r, :].any()]
        row = random.choice(valid_rows)
        gap = random.randint(2, max(2, w // 2 - 2))
        gap_cols = sorted({gap, w - 1 - gap})
        _draw_horizontal_wall(grid, protected, row, gap_cols)
    else:
        mirror = "ud"
        valid_cols = [c for c in range(2, w - 2) if not protected[:, c].any()]
        col = random.choice(valid_cols)
        gap = random.randint(2, max(2, h // 2 - 2))
        gap_rows = sorted({gap, h - 1 - gap})
        _draw_vertical_wall(grid, protected, col, gap_rows)

    for _ in range(random.randint(1, 2)):
        cells = _random_segment(grid.shape, protected, mirror=mirror)
        if cells is not None:
            _paint_wall_cells(grid, protected, cells)


def _branch_satisfied(analysis: dict, corner_colors: list[int], branch: str) -> bool:
    if branch == "asym":
        return any(len(record["leaders"]) == 1 for record in analysis["tie_records"])
    return any(
        len(record["leaders"]) >= 2
        and record["winner"] == min(record["leaders"], key=lambda idx: corner_colors[idx])
        for record in analysis["tie_records"]
    )


def _validate_generated_input(grid: np.ndarray, branch: str) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if any(int(g[r, c]) != BLACK for r, c in _corner_coords(g.shape)):
        return False
    if not set(map(int, np.unique(g))).issubset({BLACK, BLUE, RED, GREEN, YELLOW, GREY}):
        return False

    try:
        seeded, corner_colors = _prepare_seeded_grid(g)
        analysis = _analyze_flood(seeded, corner_colors)
    except AssertionError:
        return False

    traversable = seeded != GREY
    corner_sizes = [_component_size(traversable, corner) for corner in _corner_coords(g.shape)]
    if min(corner_sizes) < 3:
        return False
    if np.count_nonzero(analysis["tie_mask"]) < 3:
        return False
    if min(analysis["final_areas"]) < 3:
        return False
    if not _branch_satisfied(analysis, corner_colors, branch):
        return False

    out = _render_output(seeded, corner_colors, analysis["owner"])
    if np.array_equal(out, g):
        return False
    if set(map(int, np.unique(out))) & set(BLOCK_COLORS) != set(BLOCK_COLORS):
        return False
    return True


def _generate_attempt(branch: str) -> np.ndarray | None:
    h = random.randint(13, 17)
    w = random.randint(13, 17)
    grid = np.zeros((h, w), dtype=int)

    block_rc = _place_block_and_ring(grid)
    protected = _protected_mask(grid.shape, block_rc)

    if branch == "asym":
        _build_asymmetric_walls(grid, protected)
    else:
        _build_symmetric_walls(grid, protected)

    return grid if _validate_generated_input(grid, branch) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        branch = "sym" if random.random() < 0.75 else "asym"
        grid = _generate_attempt(branch)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-7 input")
