# Combo variant of 7b5033c1 (index 8)
# combo_spec: D2/2+D6/0
# Name: 标记起点+八连通遍历

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


# Design analysis for the fused variant
# 1) Concrete behavior and trace:
#    - The grid contains one long multicolor snake plus one isolated marker pixel.
#      The snake is a simple 8-connected chain: consecutive cells may meet
#      orthogonally or diagonally, but no non-consecutive pair 8-touches.
#    - Example:
#        snake = (3,3)->(4,4)->(4,5)->(5,6)->(6,6)
#        ends  = (3,3), (6,6)
#        marker at (6,8)
#      Original task would fail twice:
#        a) it assumes every foreground cell belongs to the path, so the marker
#           pollutes endpoint detection;
#        b) it only follows 4-neighbors, so the diagonal links are invisible.
#      Fused logic instead:
#        a) split foreground by 8-connectivity and keep the largest component as
#           the snake;
#        b) find 8-degree-1 endpoints inside that component;
#        c) choose the endpoint nearest to the isolated marker;
#        d) traverse through 8-neighbors and output colors as an N x 1 column.
#      If the marker points at (6,6), the output order is the reverse of what
#      "top-left endpoint first" would produce.
# 2) Complexity calibration:
#    - The marker is always a single isolated red pixel and the path is always the
#      only large foreground component, so object roles stay visually readable.
#    - The generator forces several diagonal edges and several turns, making the
#      D6 change visible instead of leaving mostly-4-connected degenerate samples.
#    - The marker sometimes points to the top-left endpoint and sometimes to the
#      other endpoint, so the D2 change is not reducible to a fixed reversal bias.
# 3) Old hidden assumptions that break:
#    - `mask = g != bg` no longer means "this is the whole snake"; one foreground
#      cell is just a symbol, so solver must separate components first.
#    - 4-neighbor degree counting and traversal are invalid because diagonal steps
#      are legal snake edges now.
#    - `start = min(ends)` is invalid because marker geometry chooses the start.
#    - Picking the next step by bare lexicographic order is unsafe once 8-way
#      motion is allowed, so traversal keeps a direction-priority tie-breaker.
# 4) Placement legality:
#    - Every new snake cell must be in bounds, unused, and may 8-touch only its
#      immediate predecessor. This preserves a single simple chain with two ends.
#    - The marker must be in bounds, off the snake, and not 8-neighbor to any
#      snake cell; otherwise 8-connected component splitting would merge it into
#      the snake and destroy the symbol semantics.
#    - The marker must be strictly closer to the chosen endpoint than to the
#      other endpoint under the solver's (Chebyshev, Manhattan) distance rule.
# 5) Visual observability:
#    - Across samples, humans can see one isolated red cue near one endpoint and
#      can also see several diagonal snake segments. The intended rule is thus:
#      "read from the marked end, following diagonal-capable snake connectivity."
# 6) Visual cleanliness:
#    - The path lives in one connected shape and the marker is the only detached
#      foreground cell. There are no stacked symbols, overlapping objects, or
#      marker-path tangencies that would make the scene look noisy.
# 7) Color inference:
#    - Colors are preserved rather than remapped. The only color semantics added
#      is that red (2) marks the start cue and never appears on the snake itself.
# 8) Quantitative anti-degeneracy target:
#    - The generator rejects paths with too few diagonal edges. Post-generation
#      validation can therefore count diagonal transitions and marked-end reversals
#      to confirm both fused dimensions are actually expressed in the samples.
# 9) New solver assumptions and matching generator guarantees:
#    - Solver assumes the largest 8-connected foreground component is the snake.
#      Generator guarantees one path of length about 20-28 and one 1-pixel marker.
#    - Solver assumes the snake has exactly two 8-degree-1 endpoints and all other
#      path cells have degree <= 2. Generator enforces that when constructing the
#      walk and rechecks it before accepting a grid.
#    - Solver assumes the marker-selected endpoint is unique. Generator only keeps
#      marker placements that are strictly nearer to one endpoint than the other.


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
DIR_TO_IDX = {d: i for i, d in enumerate(DIRS8)}
TURN_ORDER = [0, 1, 7, 2, 6, 3, 5, 4]
INITIAL_DIR_ORDER = [2, 3, 4, 5, 6, 7, 0, 1]
MARK_OFFSETS: list[tuple[int, int]] = [
    (-2, 0),
    (0, 2),
    (2, 0),
    (0, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -2),
    (-2, 2),
    (2, 2),
    (2, -2),
]
DIAG_DIRS = {(-1, -1), (-1, 1), (1, 1), (1, -1)}


def _neighbors8(rc: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = rc
    return [(int(r + dr), int(c + dc)) for dr, dc in DIRS8]


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((int(rr), int(cc)))
            for nr, nc in _neighbors8((rr, cc)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
        comps.append(comp)
    return comps


def _endpoint_degrees(path_pts: list[tuple[int, int]]) -> tuple[dict[tuple[int, int], int], list[tuple[int, int]]]:
    pset = set(path_pts)
    deg: dict[tuple[int, int], int] = {}
    ends: list[tuple[int, int]] = []
    for p in path_pts:
        d = sum(1 for q in _neighbors8(p) if q in pset)
        deg[p] = int(d)
        if d == 1:
            ends.append(p)
    return deg, ends


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


def _trace_path(path_pts: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    pset = set(path_pts)
    order = [start]
    visited = {start}
    prev: tuple[int, int] | None = None
    cur = start
    prev_dir: int | None = None

    for _ in range(len(path_pts) - 1):
        candidates: list[tuple[int, tuple[int, int]]] = []
        for q in _neighbors8(cur):
            if q not in pset or q == prev or q in visited:
                continue
            candidates.append((_turn_rank(prev_dir, _dir_idx(cur, q)), q))
        if not candidates:
            break
        _, nxt = min(candidates, key=lambda item: (item[0], item[1]))
        order.append(nxt)
        visited.add(nxt)
        prev_dir = _dir_idx(cur, nxt)
        prev, cur = cur, nxt
    return order


def _choose_start(ends: list[tuple[int, int]], extra_pts: list[tuple[int, int]]) -> tuple[int, int]:
    if not extra_pts:
        return min(ends)
    marker = min(extra_pts, key=lambda pt: min(_dist_key(pt, end) for end in ends))
    scored = [(_dist_key(marker, end), end) for end in ends]
    scored.sort()
    return scored[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten a multicolor 8-connected snake into a single-column strip.

    The isolated red marker selects which endpoint to start from.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    mask = g != bg
    if not np.any(mask):
        return np.zeros((0, 1), dtype=int)

    comps = _components8(mask)
    path_pts = max(comps, key=len)
    extra_pts = [pt for comp in comps if comp is not path_pts for pt in comp]

    deg, ends = _endpoint_degrees(path_pts)
    assert len(ends) == 2, "expected one simple 8-connected snake component"
    assert all(d <= 2 for d in deg.values()), "snake unexpectedly branches under 8-connectivity"

    start = _choose_start(ends, extra_pts)
    order = _trace_path(path_pts, start)
    assert len(order) == len(path_pts), "8-connected trace did not consume the whole snake"

    cols = [int(g[r, c]) for r, c in order]
    return np.array(cols, dtype=int).reshape((-1, 1))


def _legal_next(
    path_set: set[tuple[int, int]],
    cur: tuple[int, int],
    nxt: tuple[int, int],
    h: int,
    w: int,
) -> bool:
    r, c = nxt
    if r < 2 or r >= h - 2 or c < 2 or c >= w - 2:
        return False
    if nxt in path_set:
        return False
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if (rr, cc) == cur or (rr, cc) == nxt:
                continue
            if (rr, cc) in path_set:
                return False
    return True


def _future_options(
    path_set: set[tuple[int, int]],
    cur: tuple[int, int],
    nxt: tuple[int, int],
    h: int,
    w: int,
) -> int:
    tmp = set(path_set)
    tmp.add(nxt)
    count = 0
    for dr, dc in DIRS8:
        cand = (int(nxt[0] + dr), int(nxt[1] + dc))
        if cand == cur:
            continue
        if _legal_next(tmp, nxt, cand, h, w):
            count += 1
    return count


def _build_path(target_len: int, h: int, w: int) -> list[tuple[int, int]] | None:
    start = (random.randint(3, h - 4), random.randint(3, w - 4))
    path = [start]
    path_set = {start}
    prev_dir: tuple[int, int] | None = None
    turns = 0
    diag_steps = 0

    min_diagonal_steps = 6
    min_turns = 5

    while len(path) < target_len:
        remaining = target_len - len(path)
        candidates: list[tuple[tuple[int, int], int, int, int, float]] = []
        dirs = list(DIRS8)
        random.shuffle(dirs)
        for dr, dc in dirs:
            nxt = (int(path[-1][0] + dr), int(path[-1][1] + dc))
            if not _legal_next(path_set, path[-1], nxt, h, w):
                continue
            if remaining > 1 and _future_options(path_set, path[-1], nxt, h, w) == 0:
                continue
            is_diag = 1 if (dr, dc) in DIAG_DIRS else 0
            turn = 1 if prev_dir is not None and (dr, dc) != prev_dir else 0
            need_diag = max(0, min_diagonal_steps - diag_steps)
            need_turn = max(0, min_turns - turns)
            # Favor diagonal steps and turns early enough to satisfy visibility
            # constraints without making the walk deterministic.
            score = 0.0
            if is_diag:
                score += 2.0 if need_diag >= remaining else 1.0
            if turn:
                score += 1.5 if need_turn >= remaining else 0.6
            score += random.random() * 0.2
            openness = _future_options(path_set, path[-1], nxt, h, w)
            candidates.append(((dr, dc), is_diag, turn, openness, score))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (-item[4], -item[3], item[0]))
        chosen_dir, is_diag, turn, _open, _score = candidates[0]
        nxt = (int(path[-1][0] + chosen_dir[0]), int(path[-1][1] + chosen_dir[1]))
        path.append(nxt)
        path_set.add(nxt)
        diag_steps += int(is_diag)
        turns += int(turn)
        prev_dir = chosen_dir

    if diag_steps < min_diagonal_steps or turns < min_turns:
        return None

    deg, ends = _endpoint_degrees(path)
    if len(ends) != 2:
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
    chosen_end: tuple[int, int],
    other_end: tuple[int, int],
    h: int,
    w: int,
) -> tuple[int, int] | None:
    offsets = list(MARK_OFFSETS)
    random.shuffle(offsets)
    offsets.sort(key=lambda d: (max(abs(d[0]), abs(d[1])), abs(d[0]) + abs(d[1])))
    for dr, dc in offsets:
        nr, nc = (int(chosen_end[0] + dr), int(chosen_end[1] + dc))
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if (nr, nc) in path_set:
            continue
        if any((rr, cc) in path_set for rr, cc in _neighbors8((nr, nc))):
            continue
        if _dist_key((nr, nc), chosen_end) >= _dist_key((nr, nc), other_end):
            continue
        return nr, nc
    return None


def _diag_edge_count(path: list[tuple[int, int]]) -> int:
    count = 0
    for a, b in zip(path, path[1:]):
        if abs(int(a[0]) - int(b[0])) == 1 and abs(int(a[1]) - int(b[1])) == 1:
            count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    h, w = 16, 16
    bg = random.choice([c for c in range(10) if c != MARK])
    non_bg_count = random.randint(3, 5)
    palette = random.sample([c for c in range(10) if c not in (bg, MARK)], non_bg_count)
    target_len = random.randint(22, 28)

    path = _build_path(target_len, h, w)
    if path is None:
        return None

    total = len(path)
    num_segments = random.randint(max(4, non_bg_count), 7)
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

    _deg, ends = _endpoint_degrees(path)
    if len(ends) != 2:
        return None

    lex_start = min(ends)
    # Bias slightly toward the non-lexicographic end so the marker rule is often
    # behaviorally visible, but still allow both directions across samples.
    if random.random() < 0.6:
        marked_start = ends[0] if ends[0] != lex_start else ends[1]
    else:
        marked_start = lex_start
    other_end = ends[0] if marked_start == ends[1] else ends[1]

    path_set = set(path)
    marker = _choose_marker_cell(path_set, marked_start, other_end, h, w)
    if marker is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    for (r, c), col in zip(path, colors):
        grid[r, c] = int(col)
    grid[marker] = int(MARK)

    # Anti-degeneracy checks: diagonal connectivity must matter, and the marker
    # must be an actually isolated symbol under 8-connectivity.
    if _diag_edge_count(path) < 6:
        return None
    if any((rr, cc) in path_set for rr, cc in _neighbors8(marker)):
        return None
    if _dist_key(marker, marked_start) >= _dist_key(marker, other_end):
        return None

    expected_order = _trace_path(path, marked_start)
    expected_out = np.array([int(grid[r, c]) for r, c in expected_order], dtype=int).reshape((-1, 1))
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate fused D2/2+D6/0 input after 800 attempts")
