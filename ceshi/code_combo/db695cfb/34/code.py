# Combo variant of db695cfb (index 34)
# combo_spec: D3/2+D9/0+D14/2
# Name: 迭代传播+遇线停止+交叉触发新延伸

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg


_BACKSLASH = 0
_SLASH = 1
_DIRS = {
    _BACKSLASH: ((1, 1), (-1, -1)),
    _SLASH: ((1, -1), (-1, 1)),
}
_PERP = {_BACKSLASH: _SLASH, _SLASH: _BACKSLASH}

_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_H_RANGE = (10, 22)
_W_RANGE = (10, 22)
_GEN_RETRIES = 400
_PLACEMENT_RETRIES = 120


# Design analysis for the fused rule:
# - Concrete trace: a blue "\" segment contains a magenta seed T1. T1 emits "/" rays.
#   One ray passes through another magenta pixel T2, so T2 emits "\" rays in the next wave.
#   That new "\" branch hits the endpoint of a blue "/" segment, stops there, and the endpoint
#   immediately re-emits along "/" so the line grows past the endpoint on its free side.
# - Complexity calibration: every generated puzzle includes all three mechanisms exactly in that
#   sparse chain (seed -> relay -> collision -> re-extension), which makes the fusion visible
#   without flooding the board with ambiguous overlaps.
# - Broken original assumptions: rays no longer always reach the border, trigger handling is no
#   longer one-shot, and simple overwrite order is no longer enough. The solver must therefore
#   track line occupancy per diagonal orientation and remember which trigger-orientation pairs
#   were already processed.
# - Placement legality: the generator constructs coordinates algebraically, rejects overlaps,
#   rejects accidental extra blue diagonal pairings, and requires the intended free-side cell
#   beyond the collision endpoint to stay in bounds.
# - Visual observability and cleanliness: only the original two functional colors are used
#   repeatedly (blue anchors, magenta propagation). Samples read as a clear zig-zag relay
#   rather than a dense tangle, so a human can spot the iterative and collision-driven growth.
# - New solver assumptions audited against generation: every collision-triggered re-extension
#   relies on striking an endpoint of the opposite blue segment, so the emitted direction has at
#   least one open side; the generator enforces that geometry directly.


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _in_bounds(h: int, w: int, rc: tuple[int, int]) -> bool:
    r, c = rc
    return 0 <= r < h and 0 <= c < w


def _step(rc: tuple[int, int], d: tuple[int, int], n: int = 1) -> tuple[int, int]:
    return rc[0] + d[0] * n, rc[1] + d[1] * n


def _find_line_color(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    best = None  # (dist, color)
    for col, pts in by_color.items():
        if len(pts) < 2:
            continue
        for i in range(len(pts)):
            r0, c0 = pts[i]
            for j in range(i + 1, len(pts)):
                r1, c1 = pts[j]
                dr = r1 - r0
                dc = c1 - c0
                if dr == 0 or abs(dr) != abs(dc):
                    continue
                cand = (abs(dr), col)
                if best is None or cand > best:
                    best = cand

    assert best is not None, "at least one diagonal-aligned same-color pair must exist"
    return int(best[1])


def _base_segments(line_points: list[tuple[int, int]]) -> list[tuple[int, list[tuple[int, int]]]]:
    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in line_points:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    segments: list[tuple[int, list[tuple[int, int]]]] = []
    for pts in groups_a.values():
        if len(pts) >= 2:
            pts = sorted(pts)
            segments.append((_BACKSLASH, _diag_path(pts[0], pts[-1])))
    for pts in groups_b.values():
        if len(pts) >= 2:
            pts = sorted(pts)
            segments.append((_SLASH, _diag_path(pts[0], pts[-1])))
    return segments


def _simulate(grid: np.ndarray, want_debug: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    line_color = _find_line_color(g, bg)

    non_line_colors = [int(v) for v in g.flatten() if int(v) not in (bg, line_color)]
    if non_line_colors:
        ray_color = int(Counter(non_line_colors).most_common(1)[0][0])
    else:
        ray_color = next(c for c in range(10) if c not in (bg, line_color))

    out = g.copy()
    line_points = [(int(r), int(c)) for r, c in np.argwhere(g == line_color)]
    seed_cells = {(int(r), int(c)): int(g[r, c]) for r, c in np.argwhere((g != bg) & (g != line_color))}

    cell_orients: dict[tuple[int, int], set[int]] = defaultdict(set)
    queued: set[tuple[int, int, int]] = set()
    processed: set[tuple[int, int, int]] = set()
    queue: list[tuple[int, int, int, int]] = []
    debug = {
        "relay_hits": 0,
        "collision_triggers": 0,
        "stopped_on_line": 0,
        "processed_triggers": 0,
        "ray_color": ray_color,
    }

    def enqueue_trigger(r: int, c: int, emit_orient: int, color: int) -> bool:
        key = (int(r), int(c), int(emit_orient))
        if key in queued or key in processed:
            return False
        queue.append((int(r), int(c), int(emit_orient), int(color)))
        queued.add(key)
        return True

    for orient, path in _base_segments(line_points):
        for r, c in path:
            cell_orients[(r, c)].add(orient)
            if int(out[r, c]) == bg:
                out[r, c] = line_color
            if (r, c) in seed_cells:
                out[r, c] = seed_cells[(r, c)]
                enqueue_trigger(r, c, _PERP[orient], seed_cells[(r, c)])

    def extend_branch(r0: int, c0: int, d: tuple[int, int], orient: int, color: int) -> None:
        r, c = r0 + d[0], c0 + d[1]
        while 0 <= r < h and 0 <= c < w:
            pos = (r, c)
            existing = cell_orients.get(pos, set())

            if orient in existing:
                debug["stopped_on_line"] += 1
                return

            if existing:
                prev_orients = set(existing)
                cell_orients[pos].add(orient)
                out[r, c] = color
                debug["stopped_on_line"] += 1
                for prev in prev_orients:
                    if prev != orient and enqueue_trigger(r, c, prev, color):
                        debug["collision_triggers"] += 1
                return

            if int(out[r, c]) == line_color and pos not in seed_cells:
                return

            cell_orients[pos].add(orient)
            if pos in seed_cells:
                out[r, c] = seed_cells[pos]
                if enqueue_trigger(r, c, _PERP[orient], seed_cells[pos]):
                    debug["relay_hits"] += 1
            else:
                out[r, c] = color

            r += d[0]
            c += d[1]

    while queue:
        r, c, emit_orient, color = queue.pop(0)
        key = (r, c, emit_orient)
        queued.discard(key)
        if key in processed:
            continue
        processed.add(key)
        debug["processed_triggers"] += 1
        for d in _DIRS[emit_orient]:
            extend_branch(r, c, d, emit_orient, color)

    return (out, debug) if want_debug else out


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)
    line_color = _find_line_color(g, bg)

    non_bg = [(r, c, int(g[r, c])) for r in range(h) for c in range(w) if int(g[r, c]) != bg]
    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    def draw_segment(path: list[tuple[int, int]], orient: int) -> None:
        perp_dirs = _DIRS[_PERP[orient]]
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                triggers.setdefault((r, c, v), set()).update(perp_dirs)

    for orient, path in _base_segments(by_color.get(line_color, [])):
        draw_segment(path, orient)

    for (r0, c0, tcol), dirs in triggers.items():
        for dr, dc in dirs:
            r, c = r0 + dr, c0 + dc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += dr
                c += dc

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect blue diagonal endpoint pairs, then let magenta activations build an alternating diagonal network.

    Input structure:
    - Background is the dominant color. One color supplies diagonal endpoint pairs that define the initial blue segments.
    - Other non-background pixels are magenta-like trigger seeds: some lie on blue diagonals, others may sit off those diagonals waiting to be hit by later rays.

    Rule:
    - First fill each endpoint pair into a complete blue diagonal segment.
    - Any trigger touched by a segment or by a later ray emits a new ray pair on the perpendicular diagonal orientation.
    - Rays stop as soon as they meet an already drawn line, but that collision cell becomes a fresh trigger that re-emits along the encountered line's orientation.
    - Repeat until no new trigger-orientation pair remains.

    Key insight:
    - The output is not a single two-step construction anymore: magenta pixels relay activation across alternating diagonal directions, and line-line collisions both cap growth and seed the next wave.
    """

    return _simulate(grid, want_debug=False)


def _pairwise_accidental_alignment(points: list[tuple[int, int]], allowed: set[frozenset[tuple[int, int]]]) -> bool:
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p = points[i]
            q = points[j]
            if abs(p[0] - q[0]) != abs(p[1] - q[1]):
                continue
            if frozenset((p, q)) not in allowed:
                return True
    return False


def _try_place_distractors(inp: np.ndarray, occupied: set[tuple[int, int]], forbidden: set[tuple[int, int]]) -> None:
    h, w = inp.shape
    blue_points = [(int(r), int(c)) for r, c in np.argwhere(inp == _LINE_COLOR)]

    if random.random() < 0.45:
        for _ in range(40):
            rc = (random.randrange(h), random.randrange(w))
            if rc in occupied or rc in forbidden:
                continue
            if any(abs(rc[0] - r) == abs(rc[1] - c) for r, c in blue_points):
                continue
            inp[rc] = _LINE_COLOR
            occupied.add(rc)
            blue_points.append(rc)
            break

    if random.random() < 0.55:
        for _ in range(40):
            rc = (random.randrange(h), random.randrange(w))
            if rc in occupied or rc in forbidden:
                continue
            inp[rc] = _TRIGGER_COLOR
            occupied.add(rc)
            break


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_H_RANGE)
    w = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    for _ in range(_PLACEMENT_RETRIES):
        base_orient = random.choice((_BACKSLASH, _SLASH))
        perp_orient = _PERP[base_orient]
        d_base = random.choice(_DIRS[base_orient])
        d_perp = random.choice(_DIRS[perp_orient])

        a_left = random.randint(2, 5)
        a_right = random.randint(2, 5)
        hop_1 = random.randint(2, 4)
        hop_2 = random.randint(2, 4)
        b_len = random.randint(3, 6)

        t1 = (random.randrange(h), random.randrange(w))
        a0 = _step(t1, (-d_base[0], -d_base[1]), a_left)
        a1 = _step(t1, d_base, a_right)
        t2 = _step(t1, d_perp, hop_1)
        endpoint = _step(t2, d_base, hop_2)
        b0 = _step(endpoint, (-d_perp[0], -d_perp[1]), b_len)
        free_side = _step(endpoint, d_perp, 1)

        pts = [t1, a0, a1, t2, endpoint, b0, free_side]
        if not all(_in_bounds(h, w, p) for p in pts):
            continue

        path_a = _diag_path(a0, a1)
        path_b = _diag_path(b0, endpoint)
        if len(set(path_a) & set(path_b)) != 0:
            continue
        if t2 in path_a or t2 in path_b or t1 in path_b:
            continue

        input_blue = [a0, a1, b0, endpoint]
        allowed_pairs = {frozenset((a0, a1)), frozenset((b0, endpoint))}
        if _pairwise_accidental_alignment(input_blue, allowed_pairs):
            continue

        inp = np.full((h, w), bg, dtype=int)
        inp[a0] = _LINE_COLOR
        inp[a1] = _LINE_COLOR
        inp[b0] = _LINE_COLOR
        inp[endpoint] = _LINE_COLOR
        inp[t1] = _TRIGGER_COLOR
        inp[t2] = _TRIGGER_COLOR

        occupied = {a0, a1, b0, endpoint, t1, t2}
        out, debug = _simulate(inp, want_debug=True)
        legacy = _legacy_main(inp)

        if debug["relay_hits"] < 1:
            continue
        if debug["collision_triggers"] < 1:
            continue
        if debug["stopped_on_line"] < 1:
            continue
        if np.array_equal(out, legacy):
            continue
        if int(out[free_side]) != _TRIGGER_COLOR:
            continue
        if int(out[endpoint]) != _TRIGGER_COLOR:
            continue

        forbidden = {(int(r), int(c)) for r, c in np.argwhere(out != bg)}
        _try_place_distractors(inp, occupied, forbidden)
        return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
