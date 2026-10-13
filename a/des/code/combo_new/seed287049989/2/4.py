from __future__ import annotations

from collections import Counter, deque, defaultdict
import random

import numpy as np


MAROON = 9
MAX_GENERATION_ATTEMPTS = 400
PAIR_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)


def _neighbors4(r: int, c: int, h: int, w: int) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return tuple(out)


def _degree_grid(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    deg = np.zeros((h, w), dtype=int)
    coords = np.argwhere(mask)
    for r, c in coords:
        total = 0
        for rr, cc in _neighbors4(int(r), int(c), h, w):
            total += int(mask[rr, cc])
        deg[int(r), int(c)] = total
    return deg


def _connected_tree(mask: np.ndarray, deg: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = (int(coords[0, 0]), int(coords[0, 1]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c, h, w):
            if mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    edges = int(deg[mask].sum() // 2)
    return len(seen) == int(mask.sum()) and edges == int(mask.sum()) - 1


def _path_between(mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    q = deque([start])
    while q and goal not in parent:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c, h, w):
            if mask[rr, cc] and (rr, cc) not in parent:
                parent[(rr, cc)] = (r, c)
                q.append((rr, cc))
    assert goal in parent, "expected a connected path between same-color endpoints"

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _analyze_paths(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, dict[int, tuple[tuple[int, int], tuple[int, int]]], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    active = g != bg
    deg = _degree_grid(active)
    assert _connected_tree(active, deg), "non-background pixels must form a single tree"

    counts = Counter(map(int, g.ravel()))
    non_bg_colors = [int(color) for color in counts if int(color) != bg]
    candidates: list[int] = []
    for color in non_bg_colors:
        ok = True
        for other in non_bg_colors:
            if other == color:
                continue
            pts = np.argwhere(g == int(other))
            if len(pts) != 2:
                ok = False
                break
            if any(int(deg[int(r), int(c)]) != 1 for r, c in pts):
                ok = False
                break
        if ok:
            candidates.append(int(color))
    assert candidates, "failed to infer the grey network color"
    network_color = max(candidates, key=lambda color: counts[int(color)])

    endpoints: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    parity_counts: dict[int, int] = {}
    for color in non_bg_colors:
        if color == network_color:
            continue
        pts = np.argwhere(g == int(color))
        assert len(pts) == 2, "each endpoint color must appear exactly twice"
        pair = ((int(pts[0, 0]), int(pts[0, 1])), (int(pts[1, 0]), int(pts[1, 1])))
        assert int(deg[pair[0]]) == 1 and int(deg[pair[1]]) == 1, "colored endpoints must be leaves"
        path = _path_between(active, pair[0], pair[1])
        branch_count = sum(1 for r, c in path if int(g[r, c]) == network_color and int(deg[r, c]) >= 3)
        endpoints[int(color)] = pair
        parity_counts[int(color)] = int(branch_count)

    return network_color, active, deg, endpoints, parity_counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each same-color endpoint path across a grey tree network, then switch odd-branch paths to maroon.

    Input structure:
    - The background is black, and the non-background pixels form one 4-connected tree.
    - Most non-background pixels are grey orthogonal line segments, with some leaf pixels replaced by colored endpoints.
    - Each endpoint color appears exactly twice, and grey junction pixels can have degree at least 3.

    Rule:
    - For each pair of same-color endpoints, follow their unique simple path through the tree.
    - Recolor every grey pixel on that path to the endpoint color.
    - Count how many grey junction pixels of degree at least 3 lie on the path; if that count is odd, replace the path color with maroon instead.
    - Keep the colored endpoints themselves unchanged, and leave grey pixels that belong to no colored path as grey.

    Key insight:
    - The non-background structure is a tree, so pairing equal-colored leaves reduces the task to unique path extraction plus a parity test on branch nodes visited by each path.
    """
    g = np.asarray(grid, dtype=int)
    network_color, active, deg, endpoints, _parity_counts = _analyze_paths(g)

    out = g.copy()
    for color in sorted(endpoints):
        start, goal = endpoints[int(color)]
        path = _path_between(active, start, goal)
        branch_count = sum(1 for r, c in path if int(g[r, c]) == network_color and int(deg[r, c]) >= 3)
        fill_color = MAROON if branch_count % 2 == 1 else int(color)
        for r, c in path:
            if int(g[r, c]) == network_color:
                out[r, c] = fill_color
    return out


def _composition(total: int, parts: int) -> list[int]:
    shares = [0] * parts
    for _ in range(total):
        shares[random.randrange(parts)] += 1
    return shares


def _interval_starts(width: int, lengths: list[int], gap: int = 1) -> list[int]:
    min_width = 2 + sum(lengths) + gap * max(0, len(lengths) - 1)
    assert width >= min_width
    extra = width - min_width
    gaps = _composition(extra, len(lengths) + 1)
    starts: list[int] = []
    c = 1 + gaps[0]
    for i, length in enumerate(lengths):
        starts.append(c)
        c += int(length)
        if i + 1 < len(lengths):
            c += int(gap) + gaps[i + 1]
    return starts


def _parity_plan(num_pairs: int) -> list[int]:
    odd_pool = (1, 3)
    even_pool = (2, 4)
    odd_count = random.randint(1, num_pairs - 1)
    even_count = num_pairs - odd_count
    counts = [random.choice(odd_pool) for _ in range(odd_count)]
    counts.extend(random.choice(even_pool) for _ in range(even_count))
    random.shuffle(counts)
    return [int(v) for v in counts]


def _pair_lengths(branch_counts: list[int], num_pairs: int) -> list[int]:
    cap = 6 if num_pairs >= 4 else 7
    lengths: list[int] = []
    for branches in branch_counts:
        lo = int(branches) + 2
        hi = min(cap, lo + 1)
        lengths.append(random.randint(lo, hi))
    return lengths


def _side_assignment(num_pairs: int) -> list[int]:
    n_above = random.randint(max(0, num_pairs - 2), min(2, num_pairs))
    n_below = num_pairs - n_above
    sides = [-1] * n_above + [1] * n_below
    random.shuffle(sides)
    return sides


def _endpoint_rows(num_pairs: int) -> tuple[int, list[int]]:
    sides = _side_assignment(num_pairs)
    n_above = sides.count(-1)
    n_below = sides.count(1)
    top_margin = random.randint(2, 4)
    bottom_margin = random.randint(2, 4)
    backbone_row = top_margin + 4 * n_above
    height = backbone_row + 4 * n_below + bottom_margin + 1
    assert height <= 30

    above_rows = [backbone_row - 4 * i for i in range(1, n_above + 1)]
    below_rows = [backbone_row + 4 * i for i in range(1, n_below + 1)]
    random.shuffle(above_rows)
    random.shuffle(below_rows)

    rows: list[int] = []
    for side in sides:
        if side < 0:
            rows.append(int(above_rows.pop()))
        else:
            rows.append(int(below_rows.pop()))
    return int(backbone_row), rows


def _paint_line(grid: np.ndarray, points: list[tuple[int, int]], color: int) -> None:
    for r, c in points:
        grid[int(r), int(c)] = int(color)


def _build_canonical_input() -> np.ndarray | None:
    num_pairs = random.randint(2, 4)
    branch_counts = _parity_plan(num_pairs)
    lengths = _pair_lengths(branch_counts, num_pairs)
    min_width = 2 + sum(lengths) + (num_pairs - 1)
    if min_width > 30:
        return None
    width = random.randint(min_width, 30)
    starts = _interval_starts(width, lengths, gap=1)
    backbone_row, rows = _endpoint_rows(num_pairs)
    height = backbone_row + 1
    for row in rows:
        height = max(height, int(row) + 3)
    if height > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    colors = random.sample(PAIR_COLOR_POOL, num_pairs)
    attachment_cols: list[int] = []

    for color, row, start, length, branches in zip(colors, rows, starts, lengths, branch_counts):
        end = start + length - 1
        internal_cols = list(range(start + 1, end))
        if len(internal_cols) < branches:
            return None
        branch_cols = sorted(random.sample(internal_cols, branches))
        attach_col = random.choice(branch_cols)
        attachment_cols.append(int(attach_col))

        _paint_line(grid, [(row, c) for c in range(start, end + 1)], 5)

        if row < backbone_row:
            stem = [(rr, attach_col) for rr in range(row + 1, backbone_row + 1)]
            branch_dir = -1
            max_stub = min(2, row)
        else:
            stem = [(rr, attach_col) for rr in range(backbone_row, row)]
            branch_dir = 1
            max_stub = min(2, height - 1 - row)
        _paint_line(grid, stem, 5)
        if max_stub <= 0:
            return None

        for branch_col in branch_cols:
            if branch_col == attach_col:
                continue
            stub_len = random.randint(1, max_stub)
            stub = [(row + branch_dir * delta, branch_col) for delta in range(1, stub_len + 1)]
            _paint_line(grid, stub, 5)

        grid[row, start] = int(color)
        grid[row, end] = int(color)

    back_start = max(0, min(attachment_cols) - random.randint(0, 1))
    back_end = min(width - 1, max(attachment_cols) + random.randint(0, 1))
    _paint_line(grid, [(backbone_row, c) for c in range(back_start, back_end + 1)], 5)

    free_backbone = [c for c in range(back_start + 1, back_end) if c not in set(attachment_cols)]
    random.shuffle(free_backbone)
    extra_branches = min(len(free_backbone), random.randint(0, 2))
    for c in free_backbone[:extra_branches]:
        if backbone_row > height - 1 - backbone_row:
            direction = -1
            max_len = min(2, backbone_row)
        else:
            direction = 1
            max_len = min(2, height - 1 - backbone_row)
        if max_len <= 0:
            continue
        stub_len = random.randint(1, max_len)
        stub = [(backbone_row + direction * delta, c) for delta in range(1, stub_len + 1)]
        if any(grid[r, cc] != 0 for r, cc in stub):
            continue
        _paint_line(grid, stub, 5)

    if random.random() < 0.5:
        grid = grid.T
    if random.random() < 0.5:
        grid = np.flipud(grid)
    if random.random() < 0.5:
        grid = np.fliplr(grid)
    return np.asarray(grid, dtype=int)


def _disjoint_colored_paths(grid: np.ndarray) -> bool:
    network_color, active, _deg, endpoints, _parity_counts = _analyze_paths(grid)
    used: dict[tuple[int, int], int] = {}
    for color, (start, goal) in endpoints.items():
        for cell in _path_between(active, start, goal):
            if int(grid[cell]) != network_color:
                continue
            prev = used.get(cell)
            if prev is not None and prev != int(color):
                return False
            used[cell] = int(color)
    return True


def _gen_attempt() -> np.ndarray | None:
    grid = _build_canonical_input()
    if grid is None:
        return None

    try:
        network_color, _active, _deg, _endpoints, parity_counts = _analyze_paths(grid)
    except AssertionError:
        return None

    if network_color != 5:
        return None
    parities = {count % 2 for count in parity_counts.values()}
    if parities != {0, 1}:
        return None
    if not _disjoint_colored_paths(grid):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input for puzzle 4")
