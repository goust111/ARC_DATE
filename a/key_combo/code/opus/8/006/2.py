from __future__ import annotations

import random
from collections import deque

import numpy as np


CHECKER_PATTERN = np.array(
    [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ],
    dtype=int,
)
RING_COLORS = (2, 3, 7, 9)
OUTPUT_COLORS = (1, 2, 3, 4)


def _adjacent_sites(top: int, left: int) -> tuple[tuple[int, int], ...]:
    return ((top - 1, left + 1), (top + 3, left + 1), (top + 1, left - 1), (top + 1, left + 3))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _neighbors2(node: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = node
    return ((r - 2, c), (r + 2, c), (r, c - 2), (r, c + 2))


def _find_checker(grid: np.ndarray) -> tuple[int, int]:
    hits: list[tuple[int, int]] = []
    h, w = grid.shape
    for r in range(h - 2):
        for c in range(w - 2):
            if np.array_equal(grid[r : r + 3, c : c + 3], CHECKER_PATTERN):
                hits.append((r, c))
    assert len(hits) == 1, "expected exactly one checkerboard"
    return hits[0]


def _find_rings(grid: np.ndarray) -> list[tuple[int, int]]:
    rings: list[tuple[int, int]] = []
    h, w = grid.shape
    for r in range(h - 2):
        for c in range(w - 2):
            window = grid[r : r + 3, c : c + 3]
            border = np.array(
                [
                    window[0, 0],
                    window[0, 1],
                    window[0, 2],
                    window[1, 0],
                    window[1, 2],
                    window[2, 0],
                    window[2, 1],
                    window[2, 2],
                ]
            )
            if int(window[1, 1]) == 0 and int(border[0]) != 0 and np.all(border == border[0]):
                rings.append((r, c))
    return rings


def _follow_path(grid: np.ndarray, start: tuple[int, int]) -> tuple[int, int]:
    guide = {tuple(x) for x in np.argwhere(grid == 5)}
    assert start in guide, "path start must be a grey guide pixel"
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxt = [node for node in _neighbors2(cur) if node in guide and node != prev]
        if not nxt:
            return cur
        assert len(nxt) == 1, "guide path must be unbranched"
        prev, cur = cur, nxt[0]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_sprite(grid: np.ndarray) -> np.ndarray:
    comps = _components(grid == 8)
    assert len(comps) == 1, "expected one teal sprite"
    r0, c0, r1, c1 = _bbox(comps[0])
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    checker_top, checker_left = _find_checker(g)

    starts = [
        site
        for site in _adjacent_sites(checker_top, checker_left)
        if 0 <= site[0] < g.shape[0] and 0 <= site[1] < g.shape[1] and int(g[site]) == 5
    ]
    assert len(starts) == 1, "checkerboard must touch one guide pixel"
    terminal = _follow_path(g, starts[0])

    rings = _find_rings(g)
    targets = [(r, c) for r, c in rings if terminal in _adjacent_sites(r, c)]
    assert len(targets) == 1, "terminal guide pixel must point to one ring"
    ring_r, ring_c = targets[0]

    rr0 = max(0, ring_r - 2)
    cc0 = max(0, ring_c - 2)
    rr1 = min(g.shape[0], ring_r + 5)
    cc1 = min(g.shape[1], ring_c + 5)
    repeat_count = int(np.sum(g[rr0:rr1, cc0:cc1] == 6))
    assert 1 <= repeat_count <= 4, "marker count must be between 1 and 4"

    sprite = _extract_sprite(g)
    mask = sprite != 0
    h, w = sprite.shape
    out = np.zeros((h, w * repeat_count), dtype=int)
    for i in range(repeat_count):
        out[:, i * w : (i + 1) * w][mask] = OUTPUT_COLORS[i]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the guide path to a target ring, count nearby markers, then repeat and recolor the teal sprite.

    Input structure:
    - The grid has a black background, one 3x3 blue-black checkerboard, and 2 or 3 distinct 3x3 ring objects.
    - Grey guide pixels lie on a two-cell lattice and form one unbranched path starting next to the checkerboard.
    - A small teal 4-connected sprite sits elsewhere, and pink singleton markers appear only near the path-selected ring.

    Rule:
    - Find the unique grey guide pixel adjacent to the checkerboard and follow grey pixels in two-cell orthogonal steps to the path endpoint.
    - Identify the unique ring whose side-adjacent guide site matches that terminal guide pixel.
    - Count pink markers inside the target ring's bounding box expanded by 2 cells on every side; call that count N.
    - Crop the teal sprite to its bounding box, repeat that cropped tile horizontally N times, and recolor copy 1/2/3/4 to blue/red/green/yellow while leaving background black.

    Key insight:
    - The path chooses which ring matters, and only markers near that chosen ring determine how many sprite copies appear and therefore which copy-index color each one receives.
    """
    return _solve(grid)


def _to_actual(node: tuple[int, int]) -> tuple[int, int]:
    return 1 + 2 * node[0], 1 + 2 * node[1]


def _in_lattice(node: tuple[int, int], lattice_h: int, lattice_w: int) -> bool:
    return 0 <= node[0] < lattice_h and 0 <= node[1] < lattice_w


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _object_rect(center: tuple[int, int]) -> tuple[int, int, int, int]:
    rr, cc = _to_actual(center)
    return rr - 1, cc - 1, rr + 1, cc + 1


def _reserve_rect(reserved: np.ndarray, rect: tuple[int, int, int, int], pad: int = 0) -> None:
    h, w = reserved.shape
    r0, c0, r1, c1 = rect
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _paint_checker(grid: np.ndarray, center: tuple[int, int]) -> None:
    rr, cc = _to_actual(center)
    grid[rr - 1 : rr + 2, cc - 1 : cc + 2] = CHECKER_PATTERN


def _paint_ring(grid: np.ndarray, center: tuple[int, int], color: int) -> None:
    rr, cc = _to_actual(center)
    grid[rr - 1 : rr + 2, cc - 1 : cc + 2] = int(color)
    grid[rr, cc] = 0


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]]) -> None:
    for node in path:
        rr, cc = _to_actual(node)
        grid[rr, cc] = 5


def _target_center_candidates(
    end_node: tuple[int, int],
    path_nodes: set[tuple[int, int]],
    checker_center: tuple[int, int],
    lattice_h: int,
    lattice_w: int,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for cand in ((end_node[0] - 1, end_node[1]), (end_node[0] + 1, end_node[1]), (end_node[0], end_node[1] - 1), (end_node[0], end_node[1] + 1)):
        if not _in_lattice(cand, lattice_h, lattice_w):
            continue
        if cand == checker_center or cand in path_nodes:
            continue
        rr, cc = _to_actual(cand)
        if rr < 3 or cc < 3 or rr > 2 * lattice_h - 2 or cc > 2 * lattice_w - 2:
            continue
        if sum(_manhattan(cand, node) == 1 for node in path_nodes) != 1:
            continue
        if _manhattan(cand, checker_center) < 4:
            continue
        out.append(cand)
    random.shuffle(out)
    return out


def _grow_path(lattice_h: int, lattice_w: int) -> tuple[tuple[int, int], list[tuple[int, int]], tuple[int, int]] | None:
    left_limit = max(1, min(3, lattice_w // 3 + 1))
    row_choices = [r for r in range(1, max(2, lattice_h - 1))]
    if not row_choices:
        row_choices = list(range(lattice_h))
    checker_options = [(r, c) for r in row_choices for c in range(left_limit)]
    random.shuffle(checker_options)

    for checker_center in checker_options:
        dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        random.shuffle(dirs)
        for dr, dc in dirs:
            start = (checker_center[0] + dr, checker_center[1] + dc)
            if not _in_lattice(start, lattice_h, lattice_w):
                continue
            if _manhattan(start, checker_center) != 1:
                continue

            target_len = random.randint(4, 8)
            path = [start]
            visited = {start}

            def rec(cur: tuple[int, int]) -> bool:
                if len(path) == target_len:
                    return bool(_target_center_candidates(cur, visited, checker_center, lattice_h, lattice_w))
                candidates = []
                for nxt in _neighbors4(*cur):
                    if not _in_lattice(nxt, lattice_h, lattice_w):
                        continue
                    if nxt == checker_center or nxt in visited:
                        continue
                    if _manhattan(nxt, checker_center) == 1:
                        continue
                    if sum(_manhattan(nxt, old) == 1 for old in visited) != 1:
                        continue
                    candidates.append(nxt)
                random.shuffle(candidates)
                for nxt in candidates:
                    visited.add(nxt)
                    path.append(nxt)
                    if rec(nxt):
                        return True
                    path.pop()
                    visited.remove(nxt)
                return False

            if rec(start):
                target_choices = _target_center_candidates(path[-1], visited, checker_center, lattice_h, lattice_w)
                if target_choices:
                    return checker_center, path[:], target_choices[0]
    return None


def _pick_extra_rings(
    checker_center: tuple[int, int],
    target_center: tuple[int, int],
    path_nodes: set[tuple[int, int]],
    lattice_h: int,
    lattice_w: int,
    total_rings: int,
) -> list[tuple[int, int]] | None:
    others: list[tuple[int, int]] = []
    candidates = [(r, c) for r in range(lattice_h) for c in range(lattice_w)]
    random.shuffle(candidates)
    for cand in candidates:
        if cand in path_nodes or cand in {checker_center, target_center}:
            continue
        if sum(_manhattan(cand, node) == 1 for node in path_nodes) != 0:
            continue
        if _manhattan(cand, checker_center) < 4 or _manhattan(cand, target_center) < 4:
            continue
        if any(_manhattan(cand, old) < 4 for old in others):
            continue
        others.append(cand)
        if len(others) == total_rings - 1:
            return others
    return None


def _random_sprite_cells() -> list[tuple[int, int]]:
    size = random.randint(3, 6)
    cells = {(random.randint(0, 2), random.randint(0, 2))}
    while len(cells) < size:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < 3 and 0 <= nc < 3 and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(tuple(frontier)))
    return list(cells)


def _place_sprite(grid: np.ndarray, reserved: np.ndarray) -> bool:
    h, w = grid.shape
    for _ in range(80):
        cells = _random_sprite_cells()
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        rmin, cmin = min(rows), min(cols)
        norm = [(r - rmin, c - cmin) for r, c in cells]
        sh = max(r for r, _ in norm) + 1
        sw = max(c for _, c in norm) + 1
        row_lo = max(0, h - 6)
        row_hi = h - sh
        if row_lo > row_hi:
            row_lo = 0
        positions = [(r0, c0) for r0 in range(row_lo, row_hi + 1) for c0 in range(0, w - sw + 1)]
        random.shuffle(positions)
        for r0, c0 in positions:
            ok = True
            for rr in range(max(0, r0 - 1), min(h, r0 + sh + 1)):
                for cc in range(max(0, c0 - 1), min(w, c0 + sw + 1)):
                    if reserved[rr, cc]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            for dr, dc in norm:
                grid[r0 + dr, c0 + dc] = 8
            reserved[r0 : r0 + sh, c0 : c0 + sw] = True
            return True
    return False


def _place_markers(
    grid: np.ndarray,
    reserved: np.ndarray,
    target_center: tuple[int, int],
    other_rings: list[tuple[int, int]],
    count: int,
) -> bool:
    h, w = grid.shape
    tr, tc = _to_actual(target_center)
    candidates: list[tuple[int, int]] = []
    banned = np.zeros_like(reserved, dtype=bool)
    for center in other_rings:
        r0, c0, r1, c1 = _object_rect(center)
        _reserve_rect(banned, (r0, c0, r1, c1), pad=2)
    for rr in range(max(0, tr - 3), min(h, tr + 4)):
        for cc in range(max(0, tc - 3), min(w, tc + 4)):
            if tr - 1 <= rr <= tr + 1 and tc - 1 <= cc <= tc + 1:
                continue
            if reserved[rr, cc] or banned[rr, cc]:
                continue
            candidates.append((rr, cc))
    if len(candidates) < count:
        return False
    for rr, cc in random.sample(candidates, count):
        grid[rr, cc] = 6
        reserved[rr, cc] = True
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    lattice_h = len(range(1, h - 1, 2))
    lattice_w = len(range(1, w - 1, 2))
    if lattice_h < 7 or lattice_w < 7:
        return None

    path_spec = _grow_path(lattice_h, lattice_w)
    if path_spec is None:
        return None
    checker_center, path, target_center = path_spec
    total_rings = random.randint(2, 3)
    extra_rings = _pick_extra_rings(checker_center, target_center, set(path), lattice_h, lattice_w, total_rings)
    if extra_rings is None:
        return None

    _paint_checker(grid, checker_center)
    _reserve_rect(reserved, _object_rect(checker_center), pad=1)
    ring_colors = random.sample(RING_COLORS, total_rings)
    _paint_ring(grid, target_center, ring_colors[0])
    _reserve_rect(reserved, _object_rect(target_center), pad=1)
    for center, color in zip(extra_rings, ring_colors[1:]):
        _paint_ring(grid, center, color)
        _reserve_rect(reserved, _object_rect(center), pad=1)

    _paint_path(grid, path)
    for node in path:
        rr, cc = _to_actual(node)
        reserved[rr, cc] = True

    marker_count = random.randint(1, 4)
    if not _place_markers(grid, reserved, target_center, extra_rings, marker_count):
        return None
    if not _place_sprite(grid, reserved):
        return None

    out = _solve(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
