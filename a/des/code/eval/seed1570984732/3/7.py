import random
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
ORANGE = 7
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SOURCE_COLORS = [1, 2, 3, 6, 8, 9]


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _find_containers(grid: np.ndarray):
    h, w = grid.shape
    border_mask = np.zeros_like(grid, dtype=bool)
    containers = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != BLACK:
                continue
            frame = []
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr = r + dr
                    cc = c + dc
                    if int(grid[rr, cc]) != GREY:
                        ok = False
                        break
                    frame.append((rr, cc))
                if not ok:
                    break
            if not ok:
                continue

            attach_candidates = []
            for dr, dc in DIR4:
                rr = r + 2 * dr
                cc = c + 2 * dc
                if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == GREY:
                    attach_candidates.append((rr, cc))
            containers.append(
                {
                    "center": (r, c),
                    "frame": tuple(frame),
                    "attach_candidates": tuple(attach_candidates),
                }
            )
            for rr, cc in frame:
                border_mask[rr, cc] = True
    assert containers, "expected at least one 3x3 grey container"
    return containers, border_mask


def _source_block_and_color(grid: np.ndarray):
    colors = [int(c) for c in np.unique(grid) if int(c) not in (BLACK, GREY)]
    assert len(colors) == 1, "expected exactly one non-black non-grey source color"
    source_color = int(colors[0])
    pts = [tuple(map(int, p)) for p in np.argwhere(grid == source_color)]
    assert len(pts) == 4, "source must be a 2x2 solid block"
    rs = sorted(r for r, _ in pts)
    cs = sorted(c for _, c in pts)
    r0, r1 = rs[0], rs[-1]
    c0, c1 = cs[0], cs[-1]
    assert r1 == r0 + 1 and c1 == c0 + 1, "source must be 2x2"
    expected = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    assert set(pts) == expected, "source must be a full 2x2 block"
    return source_color, expected


def _analyze(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    containers, container_mask = _find_containers(g)
    pipe_mask = (g == GREY) & (~container_mask)
    assert bool(pipe_mask.any()), "grey pipe network is empty"

    source_color, source_block = _source_block_and_color(g)

    degrees = {}
    for r, c in map(tuple, np.argwhere(pipe_mask)):
        degrees[(int(r), int(c))] = sum(
            1 for rr, cc in _neighbors4(int(r), int(c), h, w) if bool(pipe_mask[rr, cc])
        )

    root_candidates = set()
    for r, c in source_block:
        for rr, cc in _neighbors4(r, c, h, w):
            if bool(pipe_mask[rr, cc]):
                root_candidates.add((rr, cc))
    assert root_candidates, "source block must touch the pipe network"
    root_endpoints = [p for p in root_candidates if degrees.get(p, 0) == 1]
    assert len(root_endpoints) == 1, "root endpoint must be unique"
    root = root_endpoints[0]

    parsed_containers = []
    leaf_attachments = set()
    for item in containers:
        attachments = [p for p in item["attach_candidates"] if bool(pipe_mask[p])]
        assert len(attachments) == 1, "each container must touch exactly one pipe leaf"
        attach = attachments[0]
        leaf_attachments.add(attach)
        parsed_containers.append({"center": item["center"], "attach": attach})

    pipe_leaves = {p for p, deg in degrees.items() if deg == 1 and p != root}
    assert pipe_leaves == leaf_attachments, "all non-root pipe leaves must terminate in containers"

    visited = {root}
    branch_counts = {root: 0}
    parent = {root: None}
    q = deque([root])
    while q:
        cur = q.popleft()
        cur_count = branch_counts[cur]
        cur_degree = degrees[cur]
        for nxt in _neighbors4(cur[0], cur[1], h, w):
            if not bool(pipe_mask[nxt]):
                continue
            if nxt == parent[cur]:
                continue
            if nxt in visited:
                continue
            visited.add(nxt)
            parent[nxt] = cur
            branch_counts[nxt] = cur_count + (1 if cur_degree >= 3 else 0)
            q.append(nxt)

    assert visited == set(degrees), "pipe network must be connected"
    return {
        "source_color": source_color,
        "containers": parsed_containers,
        "branch_counts": branch_counts,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the source color through the grey pipe tree and fill each terminal
    container according to how many branch points lie on its root-to-leaf path.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)
    out = g.copy()

    for container in info["containers"]:
        count = int(info["branch_counts"][container["attach"]])
        r, c = container["center"]
        if count == 0:
            out[r, c] = int(info["source_color"])
        elif count == 1:
            out[r, c] = ORANGE
        elif count == 2:
            out[r, c] = YELLOW
        else:
            out[r, c] = BLACK
    return out


def _source_cells(root, direction, offset):
    dr, dc = direction
    pr, pc = -dc, dr
    base = (root[0] + dr + offset * pr, root[1] + dc + offset * pc)
    cells = set()
    for a in (0, 1):
        for b in (0, 1):
            cells.add((base[0] + a * dr + b * pr, base[1] + a * dc + b * pc))
    return cells


def _frame_cells(endpoint, direction):
    center = (endpoint[0] + 2 * direction[0], endpoint[1] + 2 * direction[1])
    cells = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            cells.add((center[0] + dr, center[1] + dc))
    return cells


def _transform_point(pt, rot):
    r, c = pt
    if rot == 0:
        return r, c
    if rot == 1:
        return c, -r
    if rot == 2:
        return -r, -c
    return -c, r


def _place_feature_set(points, rot, shift_r, shift_c):
    return {
        (_transform_point(pt, rot)[0] + shift_r, _transform_point(pt, rot)[1] + shift_c)
        for pt in points
    }


def _build_attempt(rng: random.Random):
    family = rng.choice([0, 1, 2, 3])
    branch_levels = {0: 0, 1: 1, 2: 2, 3: rng.choice([3, 4])}[family]

    trunk_steps = [rng.randint(4, 7) for _ in range(branch_levels + 1)]
    branch_positions = []
    acc = 0
    for step in trunk_steps[:-1]:
        acc += step
        branch_positions.append(acc)
    final_leaf_x = sum(trunk_steps)

    pipe = {(0, x) for x in range(final_leaf_x + 1)}
    frames = [_frame_cells((0, final_leaf_x), (0, 1))]

    branch_signs = []
    for i in range(branch_levels):
        if i == 0 or rng.random() < 0.5:
            sign = rng.choice([-1, 1])
        else:
            sign = -branch_signs[-1]
        branch_signs.append(sign)

    for x, sign in zip(branch_positions, branch_signs):
        length = rng.randint(2, 5)
        for step in range(1, length + 1):
            pipe.add((sign * step, x))
        frames.append(_frame_cells((sign * length, x), (sign, 0)))

        if rng.random() < 0.35:
            extra_len = rng.randint(2, 4)
            opp = -sign
            for step in range(1, extra_len + 1):
                pipe.add((opp * step, x))
            frames.append(_frame_cells((opp * extra_len, x), (opp, 0)))

    source = _source_cells((0, 0), (0, -1), rng.choice([0, -1]))

    all_points = set(pipe) | source
    for frame in frames:
        all_points |= frame

    rot = rng.randrange(4)
    rotated = [_transform_point(pt, rot) for pt in all_points]
    rows = [r for r, _ in rotated]
    cols = [c for _, c in rotated]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    top = rng.randint(1, 5)
    bottom = rng.randint(1, 6)
    left = rng.randint(1, 5)
    right = rng.randint(1, 6)
    shift_r = top - r0
    shift_c = left - c0
    h = (r1 - r0 + 1) + top + bottom
    w = (c1 - c0 + 1) + left + right

    grid = np.zeros((h, w), dtype=int)
    source_color = rng.choice(SOURCE_COLORS)

    placed_pipe = _place_feature_set(pipe, rot, shift_r, shift_c)
    placed_source = _place_feature_set(source, rot, shift_r, shift_c)
    placed_frames = [_place_feature_set(frame, rot, shift_r, shift_c) for frame in frames]

    occupied = set()
    for frame in placed_frames:
        if occupied & frame:
            return None
        occupied |= frame
    if occupied & placed_pipe:
        return None
    if occupied & placed_source:
        return None
    if placed_pipe & placed_source:
        return None

    for r, c in placed_pipe:
        grid[r, c] = GREY
    for frame in placed_frames:
        for r, c in frame:
            grid[r, c] = GREY
    for r, c in placed_source:
        grid[r, c] = source_color

    info = _analyze(grid)
    counts = sorted(int(info["branch_counts"][item["attach"]]) for item in info["containers"])
    if family == 0 and counts != [0]:
        return None
    if family == 1 and (not counts or max(counts) != 1):
        return None
    if family == 2 and (1 not in counts or 2 not in counts or max(counts) != 2):
        return None
    if family == 3 and max(counts) < 3:
        return None

    out = main(grid)
    expected_centers = {tuple(item["center"]): int(info["branch_counts"][item["attach"]]) for item in info["containers"]}
    for (r, c), count in expected_centers.items():
        val = int(out[r, c])
        if count == 0 and val != source_color:
            return None
        if count == 1 and val != ORANGE:
            return None
        if count == 2 and val != YELLOW:
            return None
        if count >= 3 and val != BLACK:
            return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
