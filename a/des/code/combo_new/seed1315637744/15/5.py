from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


STAMP_MASKS = [
    np.array([[0, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[0, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[0, 1, 0], [0, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=int),
]
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
CARDINAL_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_CCW_FROM_BOTTOM = {
    (-1, 0): 2,  # up
    (0, 1): 3,   # right -> clockwise 90
    (1, 0): 0,   # down
    (0, -1): 1,  # left -> counterclockwise 90
}
MAX_GENERATION_ATTEMPTS = 400
HEADER_HEIGHT = 5


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_templates(grid: np.ndarray) -> tuple[dict[int, np.ndarray], int]:
    templates: dict[int, np.ndarray] = {}
    last_row = -1
    for comp in _components_mask(grid != 0):
        if len(comp) <= 1:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        assert (r1 - r0, c1 - c0) == (2, 2), "stamp templates must occupy 3x3 boxes"
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "each stamp template must be monochrome"
        color = colors.pop()
        stamp = np.zeros((3, 3), dtype=int)
        for r, c in comp:
            stamp[r - r0, c - c0] = color
        templates[color] = stamp
        last_row = max(last_row, r1)
    assert templates, "at least one stamp template is required"
    return templates, last_row


def _find_work_start(grid: np.ndarray, last_template_row: int) -> int:
    for r in range(last_template_row + 1, grid.shape[0]):
        if np.all(grid[r] == 0):
            return r + 1
        if np.any(grid[r] != 0):
            return r
    return last_template_row + 1


def _build_visibility_graph(points: list[tuple[int, int]]) -> dict[tuple[int, int], set[tuple[int, int]]]:
    rows: dict[int, list[int]] = defaultdict(list)
    cols: dict[int, list[int]] = defaultdict(list)
    adj: dict[tuple[int, int], set[tuple[int, int]]] = {point: set() for point in points}
    for r, c in points:
        rows[r].append(c)
        cols[c].append(r)
    for r, cs in rows.items():
        cs = sorted(cs)
        for c0, c1 in zip(cs, cs[1:]):
            p0, p1 = (r, c0), (r, c1)
            adj[p0].add(p1)
            adj[p1].add(p0)
    for c, rs in cols.items():
        rs = sorted(rs)
        for r0, r1 in zip(rs, rs[1:]):
            p0, p1 = (r0, c), (r1, c)
            adj[p0].add(p1)
            adj[p1].add(p0)
    return adj


def _graph_components(
    points: list[tuple[int, int]], adj: dict[tuple[int, int], set[tuple[int, int]]]
) -> list[list[tuple[int, int]]]:
    comps: list[list[tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for start in sorted(points):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[tuple[int, int]] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(sorted(comp))
    return comps


def _ordered_chain(
    comp: list[tuple[int, int]], adj: dict[tuple[int, int], set[tuple[int, int]]]
) -> list[tuple[int, int]]:
    if len(comp) == 1:
        return comp
    endpoints = sorted(point for point in comp if len(adj[point]) == 1)
    assert len(endpoints) == 2, "marker components must be simple chains"
    order = [endpoints[0]]
    prev = None
    cur = endpoints[0]
    while True:
        nxts = [point for point in adj[cur] if point != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "marker chains may not branch"
        nxt = nxts[0]
        order.append(nxt)
        prev, cur = cur, nxt
    assert len(order) == len(comp), "chain traversal must visit every marker exactly once"
    return order


def _normalize_dir(dr: int, dc: int) -> tuple[int, int]:
    if dr != 0:
        dr = 1 if dr > 0 else -1
    if dc != 0:
        dc = 1 if dc > 0 else -1
    return dr, dc


def _marker_directions(points: list[tuple[int, int]]) -> dict[tuple[int, int], tuple[int, int] | None]:
    adj = _build_visibility_graph(points)
    assert all(len(neigh) <= 2 for neigh in adj.values()), "marker graph degree must stay <= 2"
    directions: dict[tuple[int, int], tuple[int, int] | None] = {}
    for comp in _graph_components(points, adj):
        edge_count = sum(len(adj[point]) for point in comp) // 2
        if len(comp) == 1:
            directions[comp[0]] = None
            continue
        assert edge_count == len(comp) - 1, "marker components must be paths, not cycles"
        order = _ordered_chain(comp, adj)
        for idx, point in enumerate(order):
            if idx < len(order) - 1:
                other = order[idx + 1]
            else:
                other = order[idx - 1]
            directions[point] = _normalize_dir(other[0] - point[0], other[1] - point[1])
    return directions


def _dominant_side(stamp: np.ndarray) -> int:
    mask = stamp != 0
    counts = [
        int(mask[0, :].sum()),
        int(mask[:, 2].sum()),
        int(mask[2, :].sum()),
        int(mask[:, 0].sum()),
    ]
    best = max(counts)
    assert counts.count(best) == 1, "stamp bottom edge must be unique"
    return counts.index(best)


def _rotate_stamp(stamp: np.ndarray, direction: tuple[int, int] | None) -> np.ndarray:
    if direction is None:
        return stamp
    dominant = _dominant_side(stamp)
    normalize_to_bottom = (dominant - 2) % 4
    k = (normalize_to_bottom + DIR_TO_CCW_FROM_BOTTOM[direction]) % 4
    return np.rot90(stamp, k)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp template sprites onto the lower work area, rotating each one by its marker-chain direction.

    Input structure:
    - The background is black.
    - The upper part contains several 3x3 monochrome stamp templates, separated from one another by background columns.
    - Each template is asymmetric and has a unique dominant edge: among its top, right, bottom, and left borders, one border contains the most foreground pixels.
    - The lower part is a work area containing only single-pixel colored markers; markers that share a row or column with no marker between them form simple visibility chains.

    Rule:
    - Match every lower marker to the upper stamp template of the same color.
    - For each marker chain, traverse the chain from one endpoint to the other. A non-terminal marker uses the direction toward the next marker in that chain; the final endpoint uses the direction back toward the previous marker. Isolated markers keep the original template orientation.
    - Rotate the chosen 3x3 stamp so that its dominant edge follows the direction rule (down = original orientation, right = clockwise quarter-turn, left = counterclockwise quarter-turn, up = half-turn).
    - Output only the lower work area, stamping each rotated template with the marker as the center cell of the 3x3 pattern. When stamps overlap, later markers in top-to-bottom, left-to-right order overwrite earlier ones.

    Key insight:
    - The marker color selects which stamp to use, but the stamp orientation is controlled by the direction of the marker chain rather than by the marker's absolute position.
    """
    g = np.asarray(grid, dtype=int)
    templates, last_template_row = _extract_templates(g)
    work_start = _find_work_start(g, last_template_row)
    work = g[work_start:, :]
    assert work.size > 0, "work area must be present below the templates"

    markers: list[tuple[int, int, int]] = []
    for r, c in zip(*np.where(work != 0), strict=False):
        markers.append((int(r), int(c), int(work[r, c])))
    assert markers, "work area must contain at least one marker"

    directions = _marker_directions([(r, c) for r, c, _ in markers])
    out = np.zeros_like(work)
    for r, c, color in sorted(markers):
        assert color in templates, "every marker color must have a matching stamp template"
        stamp = _rotate_stamp(templates[color], directions[(r, c)])
        for dr in range(3):
            for dc in range(3):
                value = int(stamp[dr, dc])
                if value == 0:
                    continue
                rr, cc = r + dr - 1, c + dc - 1
                assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "stamps must stay inside the work area"
                out[rr, cc] = value
    return out


def _point_ok(point: tuple[int, int], points: list[tuple[int, int]], h: int, w: int) -> bool:
    r, c = point
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if point in points:
        return False
    for pr, pc in points:
        if abs(pr - r) + abs(pc - c) == 1:
            return False
    return True


def _segment_clear(
    start: tuple[int, int], end: tuple[int, int], points: list[tuple[int, int]]
) -> bool:
    (r0, c0), (r1, c1) = start, end
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return all(not (r == r0 and lo < c < hi) for r, c in points)
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        return all(not (c == c0 and lo < r < hi) for r, c in points)
    return False


def _is_path_layout(points: list[tuple[int, int]]) -> bool:
    adj = _build_visibility_graph(points)
    if any(len(neigh) > 2 for neigh in adj.values()):
        return False
    for comp in _graph_components(points, adj):
        edge_count = sum(len(adj[point]) for point in comp) // 2
        if len(comp) == 1:
            continue
        if edge_count != len(comp) - 1:
            return False
    return True


def _place_single(
    points: list[tuple[int, int]], h: int, w: int, rng: random.Random
) -> tuple[int, int] | None:
    for _ in range(120):
        point = (rng.randint(1, h - 2), rng.randint(1, w - 2))
        if not _point_ok(point, points, h, w):
            continue
        if _is_path_layout(points + [point]):
            return point
    return None


def _place_chain(
    points: list[tuple[int, int]], size: int, h: int, w: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    distances = [2, 3, 4]
    for _ in range(160):
        start = _place_single(points, h, w, rng)
        if start is None:
            return None
        chain = [start]
        prev_dir: tuple[int, int] | None = None
        ok = True
        while len(chain) < size:
            cur = chain[-1]
            candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
            dirs = CARDINAL_DIRS[:]
            rng.shuffle(dirs)
            for dr, dc in dirs:
                if prev_dir is not None and (dr, dc) == (-prev_dir[0], -prev_dir[1]):
                    continue
                dists = distances[:]
                rng.shuffle(dists)
                for dist in dists:
                    nxt = (cur[0] + dr * dist, cur[1] + dc * dist)
                    all_points = points + chain
                    if not _point_ok(nxt, all_points, h, w):
                        continue
                    if not _segment_clear(cur, nxt, all_points):
                        continue
                    if _is_path_layout(all_points + [nxt]):
                        candidates.append((nxt, (dr, dc)))
            if not candidates:
                ok = False
                break
            nxt, prev_dir = rng.choice(candidates)
            chain.append(nxt)
        if ok and len(chain) == size and _is_path_layout(points + chain):
            return chain
    return None


def _count_overlaps(points: list[tuple[int, int]]) -> int:
    count = 0
    for i, (r0, c0) in enumerate(points):
        for r1, c1 in points[i + 1 :]:
            if abs(r0 - r1) <= 2 and abs(c0 - c1) <= 2:
                count += 1
    return count


def _assign_colors(
    points: list[tuple[int, int]], colors: list[int], rng: random.Random
) -> list[tuple[int, int, int]]:
    assignments = colors[:]
    while len(assignments) < len(points):
        assignments.append(rng.choice(colors))
    rng.shuffle(assignments)
    return [(r, c, assignments[idx]) for idx, (r, c) in enumerate(points)]


def _gen_marker_layout(
    h: int, w: int, colors: list[int], rng: random.Random
) -> list[tuple[int, int, int]] | None:
    want_isolated = rng.random() < 0.5
    want_overlap = rng.random() < 0.45
    want_large_chain = rng.random() < 0.6
    want_multi_dir = rng.random() < 0.7

    n_chains = rng.randint(1, 3)
    n_isolated = rng.randint(1, 2) if want_isolated else 0
    sizes: list[int] = []
    for idx in range(n_chains):
        if idx == 0 and want_large_chain:
            sizes.append(rng.randint(3, 4))
        else:
            sizes.append(rng.randint(2, 4))
    if sum(sizes) + n_isolated < max(4, len(colors)):
        sizes[0] = max(sizes[0], max(4, len(colors)) - (sum(sizes) + n_isolated - sizes[0]))
        sizes[0] = min(sizes[0], 4)
    if sum(sizes) + n_isolated < max(4, len(colors)):
        return None

    component_sizes = sizes + [1] * n_isolated
    rng.shuffle(component_sizes)
    points: list[tuple[int, int]] = []
    for size in component_sizes:
        if size == 1:
            point = _place_single(points, h, w, rng)
            if point is None:
                return None
            points.append(point)
        else:
            chain = _place_chain(points, size, h, w, rng)
            if chain is None:
                return None
            points.extend(chain)

    if not _is_path_layout(points):
        return None
    directions = _marker_directions(points)
    dirset = {direction for direction in directions.values() if direction is not None}
    overlap_count = _count_overlaps(points)
    isolated_count = sum(direction is None for direction in directions.values())

    if want_isolated and isolated_count == 0:
        return None
    if not want_isolated and isolated_count != 0:
        return None
    if want_multi_dir and len(dirset) < 2:
        return None
    if want_overlap and overlap_count == 0:
        return None
    if not want_overlap and overlap_count != 0:
        return None
    if overlap_count > 2:
        return None

    return _assign_colors(sorted(points), colors, rng)


def _place_templates(
    grid: np.ndarray, colors: list[int], masks: list[np.ndarray], rng: random.Random
) -> bool:
    gap_count = len(colors) - 1
    min_total = 3 * len(colors) + gap_count
    if min_total > grid.shape[1] - 2:
        return False
    for _ in range(60):
        gaps = [rng.randint(1, 3) for _ in range(gap_count)]
        total_width = 3 * len(colors) + sum(gaps)
        if total_width <= grid.shape[1] - 2:
            break
    else:
        return False
    left = rng.randint(1, grid.shape[1] - total_width - 1)
    col = left
    for idx, color in enumerate(colors):
        mask = masks[idx]
        patch = np.where(mask, color, 0)
        grid[1:4, col : col + 3] = patch
        col += 3
        if idx < len(gaps):
            col += gaps[idx]
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_templates = rng.choice([2, 3, 4])
    min_width = max(14, 3 * n_templates + (n_templates - 1) + 2)
    work_w = rng.randint(min_width, 30)
    work_h = rng.randint(10, 25)
    if HEADER_HEIGHT + work_h > 30:
        return None

    colors = rng.sample(PALETTE, n_templates)
    mask_indices = rng.sample(range(len(STAMP_MASKS)), n_templates)
    masks = [STAMP_MASKS[idx] for idx in mask_indices]

    base = np.zeros((HEADER_HEIGHT + work_h, work_w), dtype=int)
    if not _place_templates(base, colors, masks, rng):
        return None

    for _ in range(180):
        markers = _gen_marker_layout(work_h, work_w, colors, rng)
        if markers is None:
            continue
        grid = base.copy()
        for r, c, color in markers:
            grid[HEADER_HEIGHT + r, c] = color
        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.shape != (work_h, work_w):
            continue
        if max(out.shape) > 30:
            continue
        if np.count_nonzero(out) == 0:
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-chain ARC input")
