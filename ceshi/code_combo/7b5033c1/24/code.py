# Combo variant of 7b5033c1 (index 24)
# combo_spec: D2/2+D6/0+D7/0
# Name: 八连通+噪声+标记起点

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "arc2_solver_select"))
    from arc_utils import get_bg


# Design analysis for combo D2/2 + D6/0 + D7/0
# a) Concrete behavior and trace:
#    - The true object is a multicolor simple 8-connected snake. A single red marker
#      pixel (MARK=2) is placed diagonally next to exactly one snake endpoint, and
#      several noise pixels are added off the path. Some noise cells are also only
#      diagonally adjacent to the snake, so under 8-connectivity the full foreground
#      is no longer "one clean chain"; it is a tree made of the main chain plus leaf
#      singletons.
#    - Example trace on a tiny sketch:
#        path = [(2,2),(3,3),(4,4),(5,5)]
#        marker = (1,1)  -> 8-neighbor only to (2,2)
#        noise  = (4,2)  -> diagonal leaf off (3,3)
#      Original solver: collect all non-bg cells, count only 4-neighbors, and start
#      at min(endpoint). That breaks twice: diagonal path steps disconnect the snake,
#      and the marker/noise are misinterpreted as path cells.
#      Fused solver: identify the red marker first, use its unique 8-neighbor as the
#      start endpoint, keep only that 8-connected foreground tree, then traverse the
#      main chain by always moving into the candidate with the largest remaining
#      reachable subtree. The leaf singleton noise is therefore ignored while the
#      diagonal snake is preserved.
#    - The key difference is structural: the intended object is no longer "all
#      foreground", but "the unique long branch attached to the marker-indicated
#      endpoint inside an 8-connected foreground tree".
# b) Complexity calibration:
#    - Diagonal path steps are mandatory and numerous, so the D6 change is visually
#      obvious.
#    - At least two noise pixels are diagonal leaves attached to interior snake cells,
#      so a naive 8-connected trace would branch incorrectly.
#    - Additional far isolated noise keeps D7 visible even if a solver only inspects
#      the marker region.
#    - The freedom is controlled: one fixed marker color, one snake, 2-4 attached
#      leaves, and 1-2 far leaves. This keeps the rule inferable from a few samples.
# c) Hidden assumptions broken from the original main():
#    - `mask = g != bg` no longer means "these are exactly the path cells".
#    - 4-neighbor degree counting no longer finds the path because valid steps may be
#      diagonal.
#    - `start = min(ends)` is invalid because the marker now decides direction.
#    - "there is only one relevant foreground component" is also false because far
#      noise can form separate components.
#    - Even after switching to 8-neighbors, "the foreground component is a chain"
#      fails because attached diagonal noise creates degree-3 branch nodes. The new
#      solver therefore traces the main branch from the marked endpoint by subtree
#      size, rather than trying to consume the whole component.
# d) Placement legality:
#    - The snake is generated as a self-avoiding 8-connected chain where each new cell
#      may 8-touch only its predecessor. This prevents accidental cycles/shortcuts.
#    - The marker is placed on a diagonal neighbor of the chosen endpoint and is
#      required to 8-touch exactly that endpoint and no other snake cell.
#    - Attached noise cells are diagonal-only leaves: they are not on the snake, not
#      4-neighbor to it, 8-neighbor to exactly one interior snake cell, and 8-isolated
#      from other extras. Far noise is 8-isolated from both snake and extras.
# e) Visual observability:
#    - A human can see the red marker beside one end, and that end's color appears
#      first in the output strip.
#    - The snake visibly contains diagonal segments, so 4-connectivity cannot explain
#      the examples.
#    - Some extra colored pixels visibly sit off the main zig-zag and do not appear in
#      the output, signalling that not every non-background cell should be read.
# f) Visual cleanliness:
#    - Only one special marker color is reserved.
#    - Noise cells are singletons and are kept separated from each other, avoiding
#      dense clutter blobs.
#    - Noise is banned from both endpoints and their immediate path neighbors so the
#      main chain still reads cleanly and the end behavior is unambiguous.
# g) Color-change inferability:
#    - There is no hidden color remapping. Path colors are copied directly into the
#      output order. Noise reuses path colors on purpose, so color alone cannot solve
#      the task; the structure does.
# 9) New solver assumptions and generator guarantees:
#    - There is exactly one red marker pixel and its unique non-marker 8-neighbor is
#      the intended start endpoint.
#    - The marker-connected non-marker foreground is a tree whose backbone is the true
#      snake and whose off-backbone nodes are singleton leaves.
#    - No noise leaf attaches to either endpoint or its immediate successor, so the
#      greedy "largest remaining subtree" traversal is unambiguous all the way to the
#      far endpoint.
#    - `generate_input()` enforces these conditions before accepting a sample.


MARK = 2
DIRS8: list[tuple[int, int]] = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]
DIAG_DIRS: list[tuple[int, int]] = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
DIR_TO_IDX = {d: i for i, d in enumerate(DIRS8)}
TURN_ORDER = [0, 1, 7, 2, 6, 3, 5, 4]
INITIAL_DIR_ORDER = [2, 3, 4, 5, 6, 7, 0, 1]


def _neighbors8(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(int(r + dr), int(c + dc)) for dr, dc in DIRS8]


def _components8(points: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining = set(points)
    comps: list[list[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = [start]
        while stack:
            cur = stack.pop()
            for nxt in _neighbors8(cur):
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
                    comp.append(nxt)
        comps.append(comp)
    return comps


def _component_from(start: tuple[int, int], points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    seen = {start}
    stack = [start]
    comp = [start]
    while stack:
        cur = stack.pop()
        for nxt in _neighbors8(cur):
            if nxt in points and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
                comp.append(nxt)
    return comp


def _degree_map(points: list[tuple[int, int]] | set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    pset = set(points)
    return {p: sum(1 for q in _neighbors8(p) if q in pset) for p in pset}


def _dist_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = abs(int(a[0]) - int(b[0]))
    dc = abs(int(a[1]) - int(b[1]))
    return max(dr, dc), dr + dc


def _dir_idx(a: tuple[int, int], b: tuple[int, int]) -> int:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    dr = 0 if dr == 0 else dr // abs(dr)
    dc = 0 if dc == 0 else dc // abs(dc)
    return DIR_TO_IDX[(dr, dc)]


def _turn_rank(prev_idx: int | None, next_idx: int) -> int:
    if prev_idx is None:
        return INITIAL_DIR_ORDER.index(next_idx)
    delta = (next_idx - prev_idx) % 8
    return TURN_ORDER.index(delta)


def _reachable_size(
    seed: tuple[int, int],
    blocked: set[tuple[int, int]],
    points: set[tuple[int, int]],
) -> int:
    if seed in blocked or seed not in points:
        return 0
    seen = {seed}
    stack = [seed]
    total = 0
    while stack:
        cur = stack.pop()
        total += 1
        for nxt in _neighbors8(cur):
            if nxt in points and nxt not in blocked and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return total


def _component_score(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    deg = _degree_map(comp)
    end_count = sum(d == 1 for d in deg.values())
    chain_like = int(end_count == 2 and all(d in (1, 2) for d in deg.values()))
    r0, c0 = min(comp)
    return chain_like, len(comp), -int(r0), -int(c0)


def _trace_backbone(points: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    pset = set(points)
    order = [start]
    visited = {start}
    prev: tuple[int, int] | None = None
    cur = start
    prev_dir: int | None = None

    while True:
        candidates = [q for q in _neighbors8(cur) if q in pset and q != prev and q not in visited]
        if not candidates:
            break

        blocked = set(visited)
        blocked.add(cur)
        scored: list[tuple[int, int, tuple[int, int]]] = []
        for q in candidates:
            reach = _reachable_size(q, blocked, pset)
            turn = _turn_rank(prev_dir, _dir_idx(cur, q))
            scored.append((-int(reach), int(turn), q))

        _, _, nxt = min(scored)
        order.append(nxt)
        visited.add(nxt)
        prev_dir = _dir_idx(cur, nxt)
        prev, cur = cur, nxt

    return order


def _extract_structure(
    grid: np.ndarray,
) -> tuple[tuple[int, int] | None, tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    marker_pts = [(int(r), int(c)) for r, c in np.argwhere(g == MARK)]
    marker = marker_pts[0] if len(marker_pts) == 1 else None

    fg = {(int(r), int(c)) for r, c in np.argwhere((g != bg) & (g != MARK))}
    if not fg:
        return marker, (0, 0), [], []

    start: tuple[int, int] | None = None
    component: list[tuple[int, int]] | None = None
    if marker is not None:
        marker_nbrs = [q for q in _neighbors8(marker) if q in fg]
        if len(marker_nbrs) == 1:
            start = marker_nbrs[0]
            component = _component_from(start, fg)

    if component is None:
        comps = _components8(fg)
        component = max(comps, key=_component_score)
        deg = _degree_map(component)
        ends = [p for p, d in deg.items() if d == 1]
        if marker is not None and ends:
            start = min(ends, key=lambda p: (_dist_key(marker, p), p))
        elif ends:
            start = min(ends)
        else:
            start = min(component)

    order = _trace_backbone(component, start)
    return marker, start, component, order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten the marked 8-connected snake into a single-column strip.

    Rules in the combo variant:
    - The red marker indicates which endpoint is the start.
    - The snake itself is 8-connected and may include diagonal steps.
    - Extra non-background singleton pixels are noise and must not appear in output.
    """
    g = np.array(grid, dtype=int)
    _marker, _start, _component, order = _extract_structure(g)
    if not order:
        return np.zeros((0, 1), dtype=int)
    cols = [int(g[r, c]) for r, c in order]
    return np.array(cols, dtype=int).reshape((-1, 1))


def _legal_next(
    path_set: set[tuple[int, int]],
    cur: tuple[int, int],
    nxt: tuple[int, int],
    H: int,
    W: int,
) -> bool:
    r, c = nxt
    if r < 1 or r >= H - 1 or c < 1 or c >= W - 1:
        return False
    if nxt in path_set:
        return False
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if (rr, cc) in (cur, nxt):
                continue
            if (rr, cc) in path_set:
                return False
    return True


def _build_path(target_len: int, H: int, W: int) -> list[tuple[int, int]] | None:
    start = (random.randint(2, H - 3), random.randint(2, W - 3))
    path = [start]
    path_set = {start}
    turns = 0
    prev_dir: tuple[int, int] | None = None

    while len(path) < target_len:
        dirs = list(DIAG_DIRS)
        random.shuffle(dirs)
        chosen: tuple[int, int] | None = None
        for dr, dc in dirs:
            nxt = (int(path[-1][0] + dr), int(path[-1][1] + dc))
            if _legal_next(path_set, path[-1], nxt, H, W):
                chosen = (dr, dc)
                break

        if chosen is None:
            return None

        if prev_dir is not None and chosen != prev_dir:
            turns += 1
        prev_dir = chosen
        nxt = (int(path[-1][0] + chosen[0]), int(path[-1][1] + chosen[1]))
        path.append(nxt)
        path_set.add(nxt)

    if turns < 6:
        return None

    deg = _degree_map(path)
    if sum(1 for d in deg.values() if d == 1) != 2:
        return None
    if any(d > 2 for d in deg.values()):
        return None
    return path


def _random_partition(total: int, num_parts: int, min_len: int, max_len: int) -> list[int] | None:
    if num_parts * min_len > total or num_parts * max_len < total:
        return None
    parts = [min_len] * num_parts
    remaining = total - sum(parts)
    for _ in range(remaining):
        eligible = [i for i in range(num_parts) if parts[i] < max_len]
        if not eligible:
            return None
        parts[random.choice(eligible)] += 1
    random.shuffle(parts)
    return parts


def _choose_marker_cell(
    path_set: set[tuple[int, int]],
    endpoint: tuple[int, int],
    H: int,
    W: int,
) -> tuple[int, int] | None:
    offsets = list(DIAG_DIRS)
    random.shuffle(offsets)
    for dr, dc in offsets:
        cell = (int(endpoint[0] + dr), int(endpoint[1] + dc))
        r, c = cell
        if not (0 <= r < H and 0 <= c < W):
            continue
        if cell in path_set:
            continue
        path_nbrs = [p for p in _neighbors8(cell) if p in path_set]
        if path_nbrs != [endpoint]:
            continue
        return cell
    return None


def _attached_noise_candidates(
    path: list[tuple[int, int]],
    path_set: set[tuple[int, int]],
    extras: set[tuple[int, int]],
    H: int,
    W: int,
) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for idx in range(2, len(path) - 2):
        anchor = path[idx]
        for dr, dc in DIAG_DIRS:
            cell = (int(anchor[0] + dr), int(anchor[1] + dc))
            r, c = cell
            if not (0 <= r < H and 0 <= c < W):
                continue
            if cell in path_set or cell in extras:
                continue
            if any(nb in extras for nb in _neighbors8(cell)):
                continue
            path_nbrs = [p for p in _neighbors8(cell) if p in path_set]
            if path_nbrs != [anchor]:
                continue
            out.append((int(cell[0]), int(cell[1]), int(idx)))
    random.shuffle(out)
    return out


def _far_noise_candidates(
    path_set: set[tuple[int, int]],
    extras: set[tuple[int, int]],
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            cell = (int(r), int(c))
            if cell in path_set or cell in extras:
                continue
            if any(nb in path_set for nb in _neighbors8(cell)):
                continue
            if any(nb in extras for nb in _neighbors8(cell)):
                continue
            out.append(cell)
    random.shuffle(out)
    return out


def _place_noise(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    marker: tuple[int, int],
    palette: list[int],
) -> tuple[int, int] | None:
    H, W = map(int, grid.shape)
    path_set = set(path)
    extras = {marker}

    attached_candidates = _attached_noise_candidates(path, path_set, extras, H, W)
    if len(attached_candidates) < 2:
        return None

    attached_count_target = min(len(attached_candidates), random.randint(2, 4))
    attached = 0
    used_indices: set[int] = set()
    for r, c, idx in attached_candidates:
        cell = (r, c)
        if idx in used_indices:
            continue
        if any(nb in extras for nb in _neighbors8(cell)):
            continue
        grid[cell] = int(random.choice(palette))
        extras.add(cell)
        used_indices.add(idx)
        attached += 1
        if attached >= attached_count_target:
            break

    if attached < 2:
        return None

    far_candidates = _far_noise_candidates(path_set, extras, H, W)
    far_target = random.randint(1, 2)
    if len(far_candidates) < far_target:
        return None

    far = 0
    for cell in far_candidates:
        if any(nb in extras for nb in _neighbors8(cell)):
            continue
        grid[cell] = int(random.choice(palette))
        extras.add(cell)
        far += 1
        if far >= far_target:
            break

    if far < 1:
        return None

    return attached, far


def _path_colors(total: int, palette: list[int]) -> list[int] | None:
    num_segments = random.randint(max(4, len(palette)), 7)
    seg_lengths = _random_partition(total, num_segments, min_len=2, max_len=8)
    if seg_lengths is None:
        return None

    shuffled_palette = list(palette)
    random.shuffle(shuffled_palette)
    seg_colors = list(shuffled_palette[: min(num_segments, len(shuffled_palette))])
    while len(seg_colors) < num_segments:
        choices = [c for c in palette if c != seg_colors[-1]]
        seg_colors.append(random.choice(choices))
    for i in range(1, len(seg_colors)):
        if seg_colors[i] == seg_colors[i - 1]:
            choices = [c for c in palette if c != seg_colors[i - 1]]
            seg_colors[i] = random.choice(choices)

    colors: list[int] = []
    for seg_len, col in zip(seg_lengths, seg_colors):
        colors.extend([int(col)] * int(seg_len))
    return colors


def _gen_attempt() -> np.ndarray | None:
    H, W = 16, 16
    bg = random.choice([c for c in range(10) if c != MARK])
    palette = random.sample([c for c in range(10) if c not in (bg, MARK)], random.randint(3, 5))
    target_len = random.randint(22, 28)

    path = _build_path(target_len, H, W)
    if path is None:
        return None

    colors = _path_colors(len(path), palette)
    if colors is None:
        return None

    ends = [path[0], path[-1]]
    marked_start = random.choice(ends)
    marker = _choose_marker_cell(set(path), marked_start, H, W)
    if marker is None:
        return None

    grid = np.full((H, W), bg, dtype=int)
    for (r, c), col in zip(path, colors):
        grid[r, c] = int(col)
    grid[marker] = int(MARK)

    noise_stats = _place_noise(grid, path, marker, palette)
    if noise_stats is None:
        return None
    attached_noise, far_noise = noise_stats

    expected_colors = colors if marked_start == path[0] else list(reversed(colors))
    expected_out = np.array(expected_colors, dtype=int).reshape((-1, 1))
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    marker_nbrs = [
        q
        for q in _neighbors8(marker)
        if 0 <= q[0] < H and 0 <= q[1] < W and grid[q] != bg and grid[q] != MARK
    ]
    if marker_nbrs != [marked_start]:
        return None

    if attached_noise < 2 or far_noise < 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate combo input after 800 attempts")
