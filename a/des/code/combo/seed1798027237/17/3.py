import random
from collections import Counter, defaultdict, deque

import numpy as np


MARKER_COLOR = 4  # yellow
SEPARATOR_COLOR = 5  # grey
PATH_COLOR = 8  # azure

SLOT_SIZE = 6
GEN_ATTEMPTS = 300
RNG = random.Random()


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(int(v) for v in grid.ravel()).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), min(cs), max(rs), max(cs)


def _find_separator_col(grid: np.ndarray, bg: int) -> int:
    candidates: list[int] = []
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if int(col[0]) != bg and bool(np.all(col == col[0])):
            candidates.append(c)
    exact = [c for c in candidates if int(grid[0, c]) == SEPARATOR_COLOR]
    if exact:
        assert len(exact) == 1
        return int(exact[0])
    assert len(candidates) == 1
    return int(candidates[0])


def _find_marker(grid: np.ndarray, sep_col: int) -> tuple[int, int]:
    pts = np.argwhere(grid[:, :sep_col] == MARKER_COLOR)
    if pts.shape[0] == 1:
        return int(pts[0, 0]), int(pts[0, 1])

    # Structural fallback: a singleton non-bg pixel directly above a colored node.
    h, _w = grid.shape
    for r in range(h - 1):
        for c in range(sep_col):
            if int(grid[r, c]) == _bg_color(grid):
                continue
            if int(grid[r + 1, c]) in (PATH_COLOR, SEPARATOR_COLOR):
                continue
            neighbors = [
                (r - 1, c),
                (r + 1, c),
                (r, c - 1),
                (r, c + 1),
            ]
            if sum(
                1
                for nr, nc in neighbors
                if 0 <= nr < h and 0 <= nc < sep_col and int(grid[nr, nc]) != _bg_color(grid)
            ) == 1:
                return r, c
    raise AssertionError("no valid start marker found")


def _extract_scene(
    grid: np.ndarray,
) -> tuple[int, int, list[list[tuple[int, int]]], list[int], list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    sep_col = _find_separator_col(g, bg)
    marker_r, marker_c = _find_marker(g, sep_col)
    assert 0 <= marker_r < h and 0 <= marker_c < sep_col

    left = g[:, :sep_col]
    pattern_mask = (
        (left != bg)
        & (left != PATH_COLOR)
        & (left != MARKER_COLOR)
        & (left != SEPARATOR_COLOR)
    )
    patterns = _components(pattern_mask)
    assert patterns, "no patterns found"

    pattern_id = -np.ones(left.shape, dtype=int)
    bboxes: list[tuple[int, int, int, int]] = []
    for idx, comp in enumerate(patterns):
        for r, c in comp:
            pattern_id[r, c] = idx
        bboxes.append(_bbox(comp))

    assert marker_r + 1 < h, "marker must sit above the start pattern"
    start_id = int(pattern_id[marker_r + 1, marker_c])
    assert start_id >= 0, "marker must point to the start pattern"

    path_comps = _components(left == PATH_COLOR)
    graph: dict[int, set[int]] = defaultdict(set)
    for pid in range(len(patterns)):
        graph[pid]

    for comp in path_comps:
        touched: set[int] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < sep_col and int(pattern_id[nr, nc]) >= 0:
                    touched.add(int(pattern_id[nr, nc]))
        assert len(touched) == 2, "every path segment must connect exactly two patterns"
        a, b = sorted(touched)
        graph[a].add(b)
        graph[b].add(a)

    if len(patterns) == 1:
        order = [start_id]
    else:
        degrees = [len(graph[i]) for i in range(len(patterns))]
        assert degrees.count(1) == 2, "pattern graph must be a chain"
        assert all(d in (1, 2) for d in degrees), "pattern graph must be unbranched"

        order = [start_id]
        prev = -1
        cur = start_id
        while True:
            nxts = [n for n in graph[cur] if n != prev]
            if not nxts:
                break
            assert len(nxts) == 1, "chain traversal must be unambiguous"
            prev, cur = cur, int(nxts[0])
            order.append(cur)

    assert len(order) == len(patterns), "all patterns must be reachable from the marker"

    return bg, sep_col, patterns, order, bboxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Traverse the path-linked node chain from the yellow start marker and stamp the
    visited node patterns onto the blank canvas to the right of the grey divider.
    """
    g = np.asarray(grid, dtype=int)
    bg, sep_col, _patterns, order, bboxes = _extract_scene(g)

    out = g.copy()
    out[:, sep_col + 1 :] = bg
    cursor = 0
    canvas_w = g.shape[1] - sep_col - 1

    for pid in order:
        r0, c0, r1, c1 = bboxes[pid]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        ph, pw = patch.shape
        assert cursor + pw <= canvas_w, "canvas is too narrow for the stamped sequence"
        out[:ph, sep_col + 1 + cursor : sep_col + 1 + cursor + pw] = patch
        cursor += pw + 1

    return out


def _direction(a: tuple[int, int], b: tuple[int, int]) -> str:
    ar, ac = a
    br, bc = b
    if br == ar - 1 and bc == ac:
        return "U"
    if br == ar + 1 and bc == ac:
        return "D"
    if br == ar and bc == ac - 1:
        return "L"
    if br == ar and bc == ac + 1:
        return "R"
    raise AssertionError("slots must be 4-neighbors")


def _opposite(direction: str) -> str:
    return {"U": "D", "D": "U", "L": "R", "R": "L"}[direction]


def _walk_has_turn(path: list[tuple[int, int]]) -> bool:
    if len(path) < 3:
        return False
    dirs = [_direction(path[i], path[i + 1]) for i in range(len(path) - 1)]
    return any(dirs[i] != dirs[i - 1] for i in range(1, len(dirs)))


def _walk_spans_both_axes(path: list[tuple[int, int]]) -> bool:
    rows = {r for r, _ in path}
    cols = {c for _, c in path}
    return len(rows) > 1 and len(cols) > 1


def _random_walk(
    rows: int,
    cols: int,
    length: int,
    want_turn: bool,
    want_biaxial: bool,
) -> list[tuple[int, int]] | None:
    starts = [(r, c) for r in range(rows) for c in range(cols)]
    RNG.shuffle(starts)

    def dfs(path: list[tuple[int, int]], used: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(path) == length:
            if want_turn and not _walk_has_turn(path):
                return None
            if want_biaxial and not _walk_spans_both_axes(path):
                return None
            return path[:]

        r, c = path[-1]
        nbrs = []
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in used:
                nbrs.append((nr, nc))
        RNG.shuffle(nbrs)

        # Prefer neighbors with fewer onward choices so the search avoids easy dead ends.
        nbrs.sort(
            key=lambda cell: sum(
                1
                for ar, ac in (
                    (cell[0] - 1, cell[1]),
                    (cell[0] + 1, cell[1]),
                    (cell[0], cell[1] - 1),
                    (cell[0], cell[1] + 1),
                )
                if 0 <= ar < rows and 0 <= ac < cols and (ar, ac) not in used
            )
        )

        for cell in nbrs:
            path.append(cell)
            used.add(cell)
            res = dfs(path, used)
            if res is not None:
                return res
            used.remove(cell)
            path.pop()
        return None

    for start in starts:
        res = dfs([start], {start})
        if res is not None:
            return res
    return None


def _pattern_geometry(slot_r: int, slot_c: int, size: int) -> tuple[int, int, int, int]:
    top = slot_r * SLOT_SIZE + (4 - size)
    left = slot_c * SLOT_SIZE + (4 - size)
    anchor_r = slot_r * SLOT_SIZE + 2
    anchor_c = slot_c * SLOT_SIZE + 2
    return top, left, anchor_r, anchor_c


def _port(top: int, left: int, size: int, anchor_r: int, anchor_c: int, direction: str) -> tuple[int, int]:
    if direction == "U":
        return top - 1, anchor_c
    if direction == "D":
        return top + size, anchor_c
    if direction == "L":
        return anchor_r, left - 1
    if direction == "R":
        return anchor_r, left + size
    raise AssertionError("bad direction")


def _random_patch(size: int, bg: int) -> np.ndarray:
    pool = [c for c in range(10) if c not in {bg, MARKER_COLOR, SEPARATOR_COLOR, PATH_COLOR}]
    chosen = RNG.sample(pool, RNG.randint(2, min(4, len(pool))))
    flat = [RNG.choice(chosen) for _ in range(size * size)]
    if len(set(flat)) == 1:
        flat[-1] = chosen[1]
    return np.array(flat, dtype=int).reshape(size, size)


def _build_attempt() -> np.ndarray | None:
    pattern_count = RNG.randint(2, 5)

    for _ in range(40):
        slot_rows = RNG.randint(2, 4)
        slot_cols = RNG.randint(2, 5)
        if slot_rows * slot_cols >= pattern_count:
            break
    else:
        return None

    want_turn = pattern_count >= 3 and slot_rows > 1 and slot_cols > 1 and RNG.random() < 0.7
    want_biaxial = pattern_count >= 4 and slot_rows > 1 and slot_cols > 1 and RNG.random() < 0.6
    want_nonspatial = pattern_count >= 3 and RNG.random() < 0.6
    path_slots = _random_walk(slot_rows, slot_cols, pattern_count, want_turn, want_biaxial)
    if path_slots is None:
        return None

    sizes = [RNG.choice([2, 3]) for _ in range(pattern_count)]
    bg_candidates = [c for c in range(10) if c not in {MARKER_COLOR, SEPARATOR_COLOR, PATH_COLOR}]
    bg = RNG.choice(bg_candidates)

    left_w = slot_cols * SLOT_SIZE
    height = slot_rows * SLOT_SIZE
    canvas_w = sum(sizes) + (pattern_count - 1) + RNG.randint(0, 3)
    width = left_w + 1 + canvas_w

    grid = np.full((height, width), bg, dtype=int)
    grid[:, left_w] = SEPARATOR_COLOR

    top_lefts: list[tuple[int, int]] = []
    anchors: list[tuple[int, int]] = []
    node_dirs: list[set[str]] = [set() for _ in range(pattern_count)]

    for i in range(pattern_count - 1):
        d = _direction(path_slots[i], path_slots[i + 1])
        node_dirs[i].add(d)
        node_dirs[i + 1].add(_opposite(d))

    for i, (slot_r, slot_c) in enumerate(path_slots):
        top, left, anchor_r, anchor_c = _pattern_geometry(slot_r, slot_c, sizes[i])
        top_lefts.append((top, left))
        anchors.append((anchor_r, anchor_c))
        patch = _random_patch(sizes[i], bg)
        grid[top : top + sizes[i], left : left + sizes[i]] = patch

    for i in range(pattern_count - 1):
        top_a, left_a = top_lefts[i]
        top_b, left_b = top_lefts[i + 1]
        anchor_ar, anchor_ac = anchors[i]
        anchor_br, anchor_bc = anchors[i + 1]
        d = _direction(path_slots[i], path_slots[i + 1])
        pa = _port(top_a, left_a, sizes[i], anchor_ar, anchor_ac, d)
        pb = _port(top_b, left_b, sizes[i + 1], anchor_br, anchor_bc, _opposite(d))
        if pa[0] == pb[0]:
            r = pa[0]
            c0, c1 = sorted((pa[1], pb[1]))
            grid[r, c0 : c1 + 1] = PATH_COLOR
        else:
            c = pa[1]
            r0, r1 = sorted((pa[0], pb[0]))
            grid[r0 : r1 + 1, c] = PATH_COLOR

    start_top, start_left = top_lefts[0]
    start_size = sizes[0]
    _anchor_r, start_anchor_c = anchors[0]
    top_cols = list(range(start_left, start_left + start_size))
    blocked_cols = {start_anchor_c} if "U" in node_dirs[0] else set()
    marker_cols = [c for c in top_cols if c not in blocked_cols]
    if not marker_cols:
        marker_cols = top_cols
    marker_c = RNG.choice(marker_cols)
    marker_r = start_top - 1
    grid[marker_r, marker_c] = MARKER_COLOR

    try:
        bg2, sep_col, patterns, order, bboxes = _extract_scene(grid)
    except AssertionError:
        return None

    if bg2 != bg or sep_col != left_w:
        return None
    if len(patterns) != pattern_count or len(order) != pattern_count:
        return None
    spatial_order = sorted(range(len(bboxes)), key=lambda pid: (bboxes[pid][1], bboxes[pid][0]))
    if want_nonspatial and order == spatial_order:
        return None
    stamp_w = sum((b[3] - b[1] + 1) for b in bboxes) + (len(bboxes) - 1)
    if stamp_w > canvas_w:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
