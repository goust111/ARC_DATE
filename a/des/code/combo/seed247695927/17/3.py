from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
STEP4 = [(-2, 0), (2, 0), (0, -2), (0, 2)]
SIDES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 400
GRID_H_RANGE = (15, 23)
GRID_W_RANGE = (15, 23)
RING_COUNT_RANGE = (2, 5)
DISTRACTOR_PATH_PROB = 0.6


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(max(counts, key=counts.get))


def _checker_patch(a: int, b: int) -> np.ndarray:
    patch = np.full((3, 3), a, dtype=int)
    patch[0, 1] = b
    patch[1, 0] = b
    patch[1, 2] = b
    patch[2, 1] = b
    return patch


def _ring_patch(color: int, bg: int) -> np.ndarray:
    patch = np.full((3, 3), color, dtype=int)
    patch[1, 1] = bg
    return patch


def _is_checker_patch(patch: np.ndarray, bg: int) -> bool:
    if patch.shape != (3, 3) or np.any(patch == bg):
        return False
    a = int(patch[1, 1])
    b = int(patch[0, 1])
    if a == b:
        return False
    return (
        int(patch[0, 0]) == a
        and int(patch[0, 2]) == a
        and int(patch[2, 0]) == a
        and int(patch[2, 2]) == a
        and int(patch[1, 1]) == a
        and int(patch[1, 0]) == b
        and int(patch[1, 2]) == b
        and int(patch[2, 1]) == b
    )


def _is_ring_patch(patch: np.ndarray, bg: int) -> bool:
    if patch.shape != (3, 3) or int(patch[1, 1]) != bg:
        return False
    ring_vals = [int(patch[r, c]) for r in range(3) for c in range(3) if (r, c) != (1, 1)]
    return len(set(ring_vals)) == 1 and ring_vals[0] != bg


def _detect_objects(grid: np.ndarray) -> tuple[tuple[int, int], list[tuple[int, int]], set[tuple[int, int]]]:
    bg = _get_bg(grid)
    h, w = grid.shape
    checker = None
    rings: list[tuple[int, int]] = []
    occupied: set[tuple[int, int]] = set()
    for r in range(h - 2):
        for c in range(w - 2):
            patch = grid[r : r + 3, c : c + 3]
            if _is_checker_patch(patch, bg):
                assert checker is None, "expected exactly one checker object"
                checker = (r, c)
                occupied.update((r + rr, c + cc) for rr in range(3) for cc in range(3))
            elif _is_ring_patch(patch, bg):
                rings.append((r, c))
                occupied.update((r + rr, c + cc) for rr in range(3) for cc in range(3) if (rr, cc) != (1, 1))
    assert checker is not None, "missing checker object"
    assert rings, "missing ring objects"
    return checker, rings, occupied


def _adjacent_to_box(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    return {(r0 - 1, c0 + 1), (r0 + 3, c0 + 1), (r0 + 1, c0 - 1), (r0 + 1, c0 + 3)}


def _neighbors2(node: tuple[int, int], nodes: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = node
    out = []
    for dr, dc in STEP4:
        nxt = (r + dr, c + dc)
        if nxt in nodes:
            out.append(nxt)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Return the 3x3 ring reached by following the guide-pixel color attached to the checker.

    Input structure:
    - A dominant background color.
    - One 3x3 checker object.
    - Several 3x3 ring objects.
    - Sparse guide pixels that connect by horizontal/vertical steps of length two.

    Rule:
    - Find the unique guide pixel adjacent to the checker.
    - Follow the connected guide path in that pixel's color using step size two.
    - Output the unique ring object adjacent to the far end of that path.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    checker, rings, object_cells = _detect_objects(g)

    guide_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(g != bg)
        if (int(r), int(c)) not in object_cells
    }
    assert guide_cells, "expected guide pixels"

    checker_adj = _adjacent_to_box(checker)
    starts = [pos for pos in guide_cells if pos in checker_adj]
    assert len(starts) == 1, "checker must have a unique adjacent guide pixel"
    start = starts[0]
    path_color = int(g[start])

    same_color_nodes = {pos for pos in guide_cells if int(g[pos]) == path_color}
    assert start in same_color_nodes

    q = deque([start])
    seen = {start}
    while q:
        node = q.popleft()
        for nxt in _neighbors2(node, same_color_nodes):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)

    hit_rings = []
    for ring in rings:
        adj = _adjacent_to_box(ring)
        if any(node in adj for node in seen):
            hit_rings.append(ring)
    assert len(hit_rings) == 1, "path must indicate a unique ring"

    r0, c0 = hit_rings[0]
    return g[r0 : r0 + 3, c0 : c0 + 3].copy()


def _bbox_clear(a: tuple[int, int], b: tuple[int, int], gap: int = 2) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    return abs(ar0 - br0) >= 3 + gap or abs(ac0 - bc0) >= 3 + gap


def _place_objects(h: int, w: int, ring_count: int) -> list[tuple[int, int]] | None:
    slots = [(r, c) for r in range(2, h - 4, 2) for c in range(2, w - 4, 2)]
    random.shuffle(slots)
    picked: list[tuple[int, int]] = []
    for pos in slots:
        if all(_bbox_clear(pos, old) for old in picked):
            picked.append(pos)
            if len(picked) == ring_count + 1:
                return picked
    return None


def _object_adjacent_nodes(objects: list[tuple[int, int]]) -> set[tuple[int, int]]:
    blocked = set()
    for box in objects:
        blocked |= _adjacent_to_box(box)
    return blocked


def _bfs_path(
    h: int,
    w: int,
    start: tuple[int, int],
    end: tuple[int, int],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    q = deque([start])
    prev = {start: None}
    dirs = STEP4[:]
    while q:
        cur = q.popleft()
        if cur == end:
            break
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = cur[0] + dr, cur[1] + dc
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nr % 2 == 0 or nc % 2 == 0:
                continue
            if nxt in forbidden or nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)
    if end not in prev:
        return None

    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _random_side_node(box: tuple[int, int]) -> tuple[int, int]:
    r0, c0 = box
    return random.choice([(r0 - 1, c0 + 1), (r0 + 3, c0 + 1), (r0 + 1, c0 - 1), (r0 + 1, c0 + 3)])


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], color: int) -> None:
    for r, c in path:
        grid[r, c] = color


def _add_distractor_path(
    grid: np.ndarray,
    bg: int,
    box: tuple[int, int],
    color: int,
    occupied_guides: set[tuple[int, int]],
    blocked_nodes: set[tuple[int, int]],
) -> bool:
    start = _random_side_node(box)
    if start in occupied_guides:
        return False
    candidates = []
    for dr, dc in STEP4:
        p1 = (start[0] + dr, start[1] + dc)
        p2 = (start[0] + 2 * dr, start[1] + 2 * dc)
        path = [start]
        ok = True
        for node in (p1, p2):
            r, c = node
            if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
                ok = False
                break
            if node in blocked_nodes or node in occupied_guides or r % 2 == 0 or c % 2 == 0:
                ok = False
                break
            path.append(node)
        if ok:
            candidates.append(path)
    if not candidates:
        return False
    path = random.choice(candidates)
    _paint_path(grid, path, color)
    occupied_guides.update(path)
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        bg = random.randrange(10)
        ring_count = random.randint(*RING_COUNT_RANGE)

        object_positions = _place_objects(h, w, ring_count)
        if object_positions is None:
            continue
        checker = object_positions[0]
        rings = object_positions[1:]
        target = random.choice(rings)

        colors = [c for c in range(10) if c != bg]
        random.shuffle(colors)
        if len(colors) < ring_count + 3:
            continue
        checker_a, checker_b = colors[0], colors[1]
        path_color = colors[2]
        ring_colors = colors[3 : 3 + ring_count]

        grid = np.full((h, w), bg, dtype=int)
        cr, cc = checker
        grid[cr : cr + 3, cc : cc + 3] = _checker_patch(checker_a, checker_b)
        for (r0, c0), color in zip(rings, ring_colors):
            grid[r0 : r0 + 3, c0 : c0 + 3] = _ring_patch(color, bg)

        start = _random_side_node(checker)
        end = _random_side_node(target)
        blocked = _object_adjacent_nodes([checker] + rings) - {start, end}
        path = _bfs_path(h, w, start, end, blocked)
        if path is None or len(path) < 2:
            continue
        _paint_path(grid, path, path_color)
        occupied_guides = set(path)

        extra_colors = [c for c in colors[3 + ring_count :] if c not in {checker_a, checker_b, path_color}]
        random.shuffle(extra_colors)
        for ring in rings:
            if ring == target or not extra_colors or random.random() >= DISTRACTOR_PATH_PROB:
                continue
            color = extra_colors.pop()
            _add_distractor_path(grid, bg, ring, color, occupied_guides, blocked | occupied_guides)

        out = main(grid)
        tr, tc = target
        expected = grid[tr : tr + 3, tc : tc + 3]
        if not np.array_equal(out, expected):
            continue
        return grid

    raise RuntimeError("failed to generate valid input")
