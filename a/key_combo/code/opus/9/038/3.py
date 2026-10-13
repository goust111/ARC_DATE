import random
from collections import defaultdict

import numpy as np


BLACK = 0
ORANGE = 7
TEAL = 8
MAROON = 9

MAX_GEN_ATTEMPTS = 400

BASE_PATHS = {
    2: [
        [(0, 0), (0, 1), (1, 1)],
    ],
    3: [
        [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
        [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)],
    ],
}


def _in_bounds(shape, r, c):
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8_including_self(r, c):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            yield r + dr, c + dc


def _chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _bbox(points):
    rows = [p[0] for p in points]
    cols = [p[1] for p in points]
    return min(rows), min(cols), max(rows), max(cols)


def _build_graph(cells):
    cells = set(cells)
    graph = defaultdict(list)
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells:
                graph[(r, c)].append((nr, nc))
    return graph


def _count_simple_paths(cells, start, end, limit=2):
    graph = _build_graph(cells)
    if start not in graph and start != end:
        return 0
    count = 0
    seen = {start}

    def dfs(node):
        nonlocal count
        if count >= limit:
            return
        if node == end:
            count += 1
            return
        for nxt in graph[node]:
            if nxt in seen:
                continue
            seen.add(nxt)
            dfs(nxt)
            seen.remove(nxt)
            if count >= limit:
                return

    dfs(start)
    return count


def _unique_simple_path(cells, start, end):
    cells = set(cells)
    graph = _build_graph(cells)
    if start not in cells or end not in cells:
        return None
    found = []
    path = [start]
    seen = {start}

    def dfs(node):
        if len(found) > 1:
            return
        if node == end:
            found.append(path.copy())
            return
        for nxt in graph[node]:
            if nxt in seen:
                continue
            seen.add(nxt)
            path.append(nxt)
            dfs(nxt)
            path.pop()
            seen.remove(nxt)
            if len(found) > 1:
                return

    dfs(start)
    if len(found) != 1:
        return None
    return found[0]


def _forbidden_mask(grid):
    forbidden = np.zeros(grid.shape, dtype=bool)
    for r, c in np.argwhere(grid == ORANGE):
        for nr, nc in _neighbors8_including_self(int(r), int(c)):
            if _in_bounds(grid.shape, nr, nc):
                forbidden[nr, nc] = True
    return forbidden


def _template_from_path(path):
    r0, c0, r1, c1 = _bbox(path)
    n_rows = r1 - r0 + 1
    n_cols = c1 - c0 + 1
    if n_rows != n_cols:
        raise ValueError("path bbox must be square")
    template = np.zeros((n_rows, n_cols), dtype=int)
    for r, c in path:
        template[r - r0, c - c0] = MAROON
    return template


def _expand_template(template):
    n = template.shape[0]
    out = np.zeros((n * n, n * n), dtype=int)
    mask = template == MAROON
    for br, bc in np.argwhere(mask):
        rs = int(br) * n
        cs = int(bc) * n
        out[rs : rs + n, cs : cs + n][mask] = MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Find the unique blocker-safe endpoint path, crop it to a square template, and self-stamp that template into a larger square.

    Input structure:
    - A black background contains a compact teal track network, exactly two maroon endpoint pixels on the track, and one or two orange blocker pixels near the track.
    - Each blocker defines a forbidden 8-neighborhood, but the endpoints and the intended safe route remain outside those forbidden cells.

    Rule:
    - Mark every orange pixel and its 8-neighborhood as forbidden.
    - In the remaining passable teal/maroon cells, find the unique 4-connected simple path between the two maroon endpoints.
    - Crop that path to its bounding box, recolor all path cells maroon in the cropped template, then create an N^2 by N^2 canvas where N is the template size.
    - For every maroon template cell at local position (r, c), stamp the whole template into output block (r, c); leave all other blocks black.

    Key insight:
    - The discovered safe path is only an intermediate object: its cropped shape simultaneously decides where copies go and what each copy contains.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    endpoints = [tuple(map(int, p)) for p in np.argwhere(g == MAROON)]
    if len(endpoints) != 2:
        raise ValueError("expected exactly two maroon endpoints")

    forbidden = _forbidden_mask(g)
    for endpoint in endpoints:
        if forbidden[endpoint]:
            raise ValueError("endpoint cannot be forbidden")

    passable_mask = ((g == TEAL) | (g == MAROON)) & ~forbidden
    passable_cells = {tuple(map(int, p)) for p in np.argwhere(passable_mask)}
    path = _unique_simple_path(passable_cells, endpoints[0], endpoints[1])
    if path is None:
        raise ValueError("expected a unique simple path")

    template = _template_from_path(path)
    return _expand_template(template)


def _transform_coord(coord, n, rot, flip):
    r, c = coord
    if flip:
        c = n - 1 - c
    for _ in range(rot % 4):
        r, c = c, n - 1 - r
    return r, c


def _transform_path(seq, n):
    rot = random.randrange(4)
    flip = bool(random.randrange(2))
    return [_transform_coord(coord, n, rot, flip) for coord in seq]


def _shift_path(seq, top, left):
    return [(r + top, c + left) for r, c in seq]


def _corner_detours(seq):
    detours = []
    occupied = set(seq)
    for idx in range(1, len(seq) - 1):
        prev_cell = seq[idx - 1]
        cur_cell = seq[idx]
        next_cell = seq[idx + 1]
        dr1 = cur_cell[0] - prev_cell[0]
        dc1 = cur_cell[1] - prev_cell[1]
        dr2 = next_cell[0] - cur_cell[0]
        dc2 = next_cell[1] - cur_cell[1]
        if dr1 == dr2 or dc1 == dc2:
            continue
        detour = (prev_cell[0] + next_cell[0] - cur_cell[0], prev_cell[1] + next_cell[1] - cur_cell[1])
        if detour in occupied:
            continue
        detours.append((idx, detour))
    return detours


def _candidate_blockers(shape, detour_cell, safe_cells):
    candidates = []
    for br, bc in _neighbors8_including_self(detour_cell[0], detour_cell[1]):
        blocker = (br, bc)
        if blocker == detour_cell or not _in_bounds(shape, br, bc):
            continue
        if blocker in safe_cells:
            continue
        if any(_chebyshev(blocker, cell) <= 1 for cell in safe_cells):
            continue
        candidates.append(blocker)
    random.shuffle(candidates)
    return candidates


def _path_set_from_ordered(seq):
    return set(seq), seq[0], seq[-1]


def _find_branch(shape, path_seq, forbidden, blocked_cells):
    path_cells, start, end = _path_set_from_ordered(path_seq)
    bbox = _bbox(path_cells)
    occupied = set(path_cells) | set(blocked_cells)
    candidates = []
    for r, c in path_cells:
        for nr, nc in _neighbors4(r, c):
            first = (nr, nc)
            if not _in_bounds(shape, nr, nc):
                continue
            if first in occupied or first in forbidden:
                continue
            if sum((adj in path_cells) for adj in _neighbors4(nr, nc)) != 1:
                continue

            outside = not (bbox[0] <= nr <= bbox[2] and bbox[1] <= nc <= bbox[3])
            second = (nr + (nr - r), nc + (nc - c))
            if _in_bounds(shape, second[0], second[1]):
                if second not in occupied and second not in forbidden:
                    if sum((adj in path_cells) for adj in _neighbors4(second[0], second[1])) == 0:
                        if sum((adj == first) for adj in _neighbors4(second[0], second[1])) == 1:
                            candidates.append(([first, second], outside, 2))
            candidates.append(([first], outside, 1))

    random.shuffle(candidates)
    candidates.sort(key=lambda item: (not item[1], random.random(), -item[2]))
    for branch, outside, _length in candidates:
        branch_cells = set(branch)
        if len(branch_cells) != len(branch):
            continue
        if any(cell in forbidden for cell in branch_cells):
            continue
        cells = set(path_cells) | branch_cells
        unique = _unique_simple_path(cells, start, end)
        if unique is None:
            continue
        if set(unique) != set(path_cells):
            continue
        if branch_cells - set(path_cells):
            return branch
    return None


def _validate_semantics(path_seq, branch_cells, detour_pairs, blockers, shape):
    path_cells, start, end = _path_set_from_ordered(path_seq)
    safe_cells = path_cells | set(branch_cells)
    blocked_cells = {detour for _idx, detour in detour_pairs}

    forbidden = set()
    for blocker in blockers:
        for cell in _neighbors8_including_self(blocker[0], blocker[1]):
            if _in_bounds(shape, cell[0], cell[1]):
                forbidden.add(cell)

    if forbidden & safe_cells:
        return False
    if not blocked_cells <= forbidden:
        return False

    after_cells = safe_cells | (blocked_cells - forbidden)
    unique_after = _unique_simple_path(after_cells, start, end)
    if unique_after is None or set(unique_after) != path_cells:
        return False
    if after_cells == path_cells:
        return False
    if _bbox(after_cells) == _bbox(path_cells):
        return False

    full_cells = safe_cells | blocked_cells
    if _count_simple_paths(full_cells, start, end, limit=2) < 2:
        return False

    for i in range(len(blockers)):
        partial_forbidden = set()
        for j, blocker in enumerate(blockers):
            if i == j:
                continue
            for cell in _neighbors8_including_self(blocker[0], blocker[1]):
                if _in_bounds(shape, cell[0], cell[1]):
                    partial_forbidden.add(cell)
        partial_cells = full_cells - partial_forbidden
        partial_unique = _unique_simple_path(partial_cells, start, end)
        if partial_unique is not None and set(partial_unique) == path_cells:
            return False

    template = _template_from_path(path_seq)
    if template.shape[0] not in (2, 3):
        return False
    if max(_expand_template(template).shape) > 30:
        return False
    return True


def _render_grid(shape, path_seq, branch_cells, detour_pairs, blockers):
    grid = np.zeros(shape, dtype=int)
    path_cells, start, end = _path_set_from_ordered(path_seq)
    full_track = path_cells | set(branch_cells) | {detour for _idx, detour in detour_pairs}

    for r, c in full_track:
        grid[r, c] = TEAL
    grid[start] = MAROON
    grid[end] = MAROON
    for blocker in blockers:
        grid[blocker] = ORANGE
    return grid


def _gen_attempt():
    n = random.choice([2, 3, 3])
    base_seq = random.choice(BASE_PATHS[n])
    seq = _transform_path(base_seq, n)

    h = random.randint(max(7, n + 4), 9)
    w = random.randint(max(7, n + 4), 9)

    top = random.randint(1, h - n - 1)
    left = random.randint(1, w - n - 1)
    path_seq = _shift_path(seq, top, left)
    path_cells, start, end = _path_set_from_ordered(path_seq)

    detours = _corner_detours(path_seq)
    if not detours:
        return None

    desired_blockers = 1
    if len(detours) >= 2 and n == 3 and random.random() < 0.55:
        desired_blockers = 2

    random.shuffle(detours)
    chosen_detours = []
    blockers = []
    for detour_info in detours:
        _idx, detour_cell = detour_info
        safe_cells = path_cells
        candidates = _candidate_blockers((h, w), detour_cell, safe_cells)
        for blocker in candidates:
            if any(_chebyshev(blocker, other) <= 1 for other in blockers):
                continue
            if any(_chebyshev(blocker, other_detour) <= 1 for _k, other_detour in chosen_detours):
                continue
            if any(_chebyshev(other_blocker, detour_cell) <= 1 for other_blocker in blockers):
                continue
            chosen_detours.append(detour_info)
            blockers.append(blocker)
            break
        if len(blockers) == desired_blockers:
            break
    if len(blockers) != desired_blockers:
        return None

    forbidden = set()
    for blocker in blockers:
        for cell in _neighbors8_including_self(blocker[0], blocker[1]):
            if _in_bounds((h, w), cell[0], cell[1]):
                forbidden.add(cell)

    branch = _find_branch((h, w), path_seq, forbidden, {detour for _idx, detour in chosen_detours})
    if branch is None:
        return None

    if not _validate_semantics(path_seq, branch, chosen_detours, blockers, (h, w)):
        return None

    grid = _render_grid((h, w), path_seq, branch, chosen_detours, blockers)
    forbidden_mask = _forbidden_mask(grid)
    if forbidden_mask[start] or forbidden_mask[end]:
        return None
    path = _unique_simple_path(
        {tuple(map(int, p)) for p in np.argwhere(((grid == TEAL) | (grid == MAROON)) & ~forbidden_mask)},
        start,
        end,
    )
    if path is None or set(path) != path_cells:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
