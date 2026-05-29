# Combo variant of 332f06d7 (index 37)
# combo_spec: D1/1+D16/3
# Name: 对角线反射+弹跳平台

from __future__ import annotations

from collections import Counter
import random

import numpy as np


"""
Design analysis

a) Concrete behavior and trace
   - The original task painted a blue winding corridor and solved motion by BFS over all feasible n×n placements.
     A black square could translate freely in 4-neighbor steps until it reached the red target or the farthest
     chokepoint-adjacent placement.
   - In this combo, the traversable blue region is the union of diagonal n×n placements only. The mover no longer
     performs undirected search; it has a direction state `(dr, dc)` with `dr, dc in {-1, 1}` and advances exactly
     one diagonal placement per step.
   - Two visible reflector types coexist:
     1. Yellow vertical walls sit immediately left or right of a bounce state. They flip only the horizontal
        component `dc`, so `(1, 1) -> (1, -1)` or `(-1, -1) -> (-1, 1)`.
     2. Red horizontal platforms sit immediately above or below a bounce state. They flip only the vertical
        component `dr`, so `(1, 1) -> (-1, 1)` or `(-1, -1) -> (1, -1)`.
   - Small trace:
     Suppose the mover starts at the left edge with unique diagonal neighbor `(1, 1)`. It advances down-right for
     three placements, lands on a state whose support row is a red bar, so its next outgoing direction becomes
     `(-1, 1)`. Two steps later it touches a yellow wall on its right edge, so the next outgoing direction becomes
     `(-1, -1)`. The original BFS solver would have treated both red/yellow pixels as non-traversable noise and
     would never recover this directional rebound sequence.

b) Complexity calibration
   - Every generated sample contains both bounce types, so the fusion is not visually reducible to the diagonal-only
     reference or the platform-only reference.
   - Bounce choices are kept to two stable categories with fixed orientations: red horizontal means vertical bounce,
     yellow vertical means horizontal reflection. This is enough structure for humans to infer the rule from 2-4
     examples without introducing many color- or orientation-dependent regimes.
   - The generator still supports reachable and miss cases. The miss case places the cyan destination on a decoy
     diagonal spur that is visibly blue but not on the actual reflected route.

c) Implicit assumption audit against the original solver
   - Original `cand.sort(key=lambda s: int(s["col"]))` assumed the two important squares could be chosen by numeric
     color order. That breaks here because red is now a platform and yellow is a wall. The new solver reads the
     mover and destination by fixed visible colors instead.
   - Original `trav = river | mover | dest` followed by 4-neighbor BFS assumed motion is undirected connectivity.
     That fails under diagonal stateful motion, so the new solver computes the same feasible n×n placements but then
     runs a deterministic simulator with direction state plus bounce objects.
   - Original "pick target or farthest reachable placement" assumed a scalar BFS distance over placements. That also
     fails here: the relevant endpoint is the last state on the simulated reflected trajectory, not the BFS frontier.
   - New assumptions introduced:
     1. There is exactly one black solid square and one cyan solid square of equal size.
     2. Yellow reflector components are thin vertical bars adjacent to bounce states.
     3. Red reflector components are thin horizontal bars adjacent to bounce states.
     4. The start has exactly one feasible diagonal successor.
     The generator enforces all four.

d) Placement legality
   - The generator plans top-left placement states first, then paints each state as an n×n blue block.
   - Yellow/red reflectors are placed only on background cells just outside those blocks, never overlapping the mover,
     destination, or blue track.
   - A strict feasible-state-set check rejects any attempt whose painted blue pixels accidentally create extra n×n
     placements beyond the intended main route and optional decoy spur.

e) Visual observability
   - The only square objects are the black mover and cyan destination; red objects are always 1-pixel-high bars and
     yellow objects are always 1-pixel-wide bars. This makes role separation human-visible rather than numeric.
   - Across samples, the mover repeatedly turns upward at red bars and left/right at yellow bars, creating a repeated
     cross-example cue that the two obstacle families implement different reflection axes.

f) Visual cleanliness
   - The blue route is a sparse diagonal ribbon. Reflectors are outside the ribbon, so no cell carries multiple
     semantics.
   - Decoy spurs are kept several cells away from the main route to avoid accidental touch/merge and to keep the
     destination visibly distinct from the true path in miss cases.

g) Change inferability
   - No hidden color ranking or random tie-breaking is used by the solver.
   - Color usage stays limited to six stable roles: black mover, blue track, red platform, green background, yellow
     wall, cyan destination. The geometric form of each colored object reinforces its role, so the color changes are
     inferable from structure rather than memorized ids.
"""


_BG = 3
_RIVER = 1
_PLATFORM = 2
_WALL = 4
_DEST = 8
_MOVER = 0


def _corner_majority_bg(g: np.ndarray) -> int:
    corners = [int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]
    return int(Counter(corners).most_common(1)[0][0])


def _find_solid_square(grid: np.ndarray, color: int) -> tuple[int, int, int]:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    n = int(r1 - r0 + 1)
    assert n == int(c1 - c0 + 1)
    patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    assert bool((patch == int(color)).all())
    assert int(patch.size) == int(coords.shape[0])
    return int(r0), int(c0), int(n)


def _feasible_mask(grid: np.ndarray, n: int) -> np.ndarray:
    h2 = int(grid.shape[0] - n + 1)
    w2 = int(grid.shape[1] - n + 1)
    assert h2 > 0 and w2 > 0
    trav = (grid == int(_RIVER)) | (grid == int(_MOVER)) | (grid == int(_DEST))
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[int(r) : int(r + n), int(c) : int(c + n)].all()):
                feasible[int(r), int(c)] = True
    return feasible


def _diag_dirs() -> list[tuple[int, int]]:
    return [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _unique_start_dir(feasible: np.ndarray, sr: int, sc: int) -> tuple[int, int]:
    h2, w2 = feasible.shape
    dirs: list[tuple[int, int]] = []
    for dr, dc in _diag_dirs():
        rr, cc = int(sr + dr), int(sc + dc)
        if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[int(rr), int(cc)]):
            dirs.append((int(dr), int(dc)))
    assert len(dirs) == 1
    return dirs[0]


def _touch_platform(grid: np.ndarray, r: int, c: int, n: int, dr: int) -> bool:
    rr = int(r + n) if int(dr) == 1 else int(r - 1)
    if not (0 <= rr < grid.shape[0]):
        return False
    return bool((grid[int(rr), int(c) : int(c + n)] == int(_PLATFORM)).all())


def _touch_wall(grid: np.ndarray, r: int, c: int, n: int, dc: int) -> bool:
    cc = int(c + n) if int(dc) == 1 else int(c - 1)
    if not (0 <= cc < grid.shape[1]):
        return False
    return bool((grid[int(r) : int(r + n), int(cc)] == int(_WALL)).any())


def _trace_motion(
    grid: np.ndarray,
    n: int,
    sr: int,
    sc: int,
    dr: int,
    dc: int,
    dest: tuple[int, int],
) -> dict:
    feasible = _feasible_mask(grid, n)
    h2, w2 = feasible.shape
    assert bool(feasible[int(sr), int(sc)])

    cur_r, cur_c = int(sr), int(sc)
    path = [(int(cur_r), int(cur_c))]
    platform_bounces = 0
    wall_bounces = 0
    seen: set[tuple[int, int, int, int]] = set()

    while True:
        state = (int(cur_r), int(cur_c), int(dr), int(dc))
        if state in seen:
            return {
                "final": (int(cur_r), int(cur_c)),
                "path": path,
                "platform_bounces": int(platform_bounces),
                "wall_bounces": int(wall_bounces),
            }
        seen.add(state)

        if _touch_platform(grid, cur_r, cur_c, n, dr):
            dr = -int(dr)
            platform_bounces += 1
        if _touch_wall(grid, cur_r, cur_c, n, dc):
            dc = -int(dc)
            wall_bounces += 1

        nr, nc = int(cur_r + dr), int(cur_c + dc)
        if not (0 <= nr < h2 and 0 <= nc < w2 and bool(feasible[int(nr), int(nc)])):
            return {
                "final": (int(cur_r), int(cur_c)),
                "path": path,
                "platform_bounces": int(platform_bounces),
                "wall_bounces": int(wall_bounces),
            }

        cur_r, cur_c = int(nr), int(nc)
        path.append((int(cur_r), int(cur_c)))
        if (int(cur_r), int(cur_c)) == (int(dest[0]), int(dest[1])):
            return {
                "final": (int(cur_r), int(cur_c)),
                "path": path,
                "platform_bounces": int(platform_bounces),
                "wall_bounces": int(wall_bounces),
            }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    assert H == W
    assert int(_corner_majority_bg(g)) == int(_BG)

    sr, sc, n = _find_solid_square(g, _MOVER)
    dr0, dc0, n2 = _find_solid_square(g, _DEST)
    assert int(n) == int(n2)

    feasible = _feasible_mask(g, n)
    assert bool(feasible[int(sr), int(sc)])
    init_dr, init_dc = _unique_start_dir(feasible, sr, sc)
    traced = _trace_motion(g, n, sr, sc, init_dr, init_dc, (dr0, dc0))
    tr, tc = traced["final"]

    out = g.copy()
    out[int(sr) : int(sr + n), int(sc) : int(sc + n)] = int(_RIVER)
    out[int(tr) : int(tr + n), int(tc) : int(tc + n)] = int(_MOVER)
    return out


def _candidate_starts(h2: int, w2: int) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for c in range(1, w2 - 1):
        out.append((0, c, 1, random.choice([-1, 1])))
        out.append((h2 - 1, c, -1, random.choice([-1, 1])))
    for r in range(1, h2 - 1):
        out.append((r, 0, random.choice([-1, 1]), 1))
        out.append((r, w2 - 1, random.choice([-1, 1]), -1))
    random.shuffle(out)
    return out


def _too_close(p: tuple[int, int], states: list[tuple[int, int]]) -> bool:
    if not states:
        return False
    prev = states[-1]
    for q in states[:-1]:
        if max(abs(int(p[0]) - int(q[0])), abs(int(p[1]) - int(q[1]))) <= 1:
            return True
    return p == prev


def _walk_segment(
    states: list[tuple[int, int]],
    used: set[tuple[int, int]],
    cur: tuple[int, int],
    direction: tuple[int, int],
    seg_len: int,
    h2: int,
    w2: int,
) -> list[tuple[int, int]] | None:
    dr, dc = direction
    added: list[tuple[int, int]] = []
    rr, cc = cur
    for _ in range(seg_len):
        rr += int(dr)
        cc += int(dc)
        p = (int(rr), int(cc))
        if not (0 <= p[0] < h2 and 0 <= p[1] < w2):
            return None
        if p in used or _too_close(p, states + added):
            return None
        added.append(p)
    return added


def _bounce_dir(direction: tuple[int, int], kind: str) -> tuple[int, int]:
    dr, dc = direction
    if kind == "platform":
        return -int(dr), int(dc)
    if kind == "wall":
        return int(dr), -int(dc)
    raise ValueError(kind)


def _bounce_sequence() -> list[str]:
    n_bounces = random.choice([2, 3])
    seq = ["platform", "wall"]
    if n_bounces == 3:
        seq.append(random.choice(["platform", "wall"]))
    if n_bounces == 2:
        random.shuffle(seq)
    else:
        if random.random() < 0.5:
            seq = [seq[1], seq[0], seq[2]]
    return seq


def _plan_main_states(h2: int, w2: int) -> tuple[list[tuple[int, int]], list[tuple[int, str]], tuple[int, int]] | None:
    min_seg = 2
    max_seg = 5
    for sr, sc, dr0, dc0 in _candidate_starts(h2, w2)[:50]:
        bounce_kinds = _bounce_sequence()
        states = [(int(sr), int(sc))]
        used = {(int(sr), int(sc))}
        cur = (int(sr), int(sc))
        cur_dir = (int(dr0), int(dc0))
        bounces: list[tuple[int, str]] = []
        ok = True

        for seg_idx in range(len(bounce_kinds) + 1):
            remaining = len(bounce_kinds) - seg_idx
            kind = bounce_kinds[seg_idx] if seg_idx < len(bounce_kinds) else None
            options: list[tuple[list[tuple[int, int]], tuple[int, int] | None]] = []
            for seg_len in range(max_seg, min_seg - 1, -1):
                added = _walk_segment(states, used, cur, cur_dir, seg_len, h2, w2)
                if added is None:
                    continue
                endpoint = added[-1]
                if kind is None:
                    options.append((added, None))
                    continue
                next_dir = _bounce_dir(cur_dir, kind)
                nxt = (int(endpoint[0] + next_dir[0]), int(endpoint[1] + next_dir[1]))
                if not (0 <= nxt[0] < h2 and 0 <= nxt[1] < w2):
                    continue
                future_states = states + added
                if nxt in used or nxt in set(added) or _too_close(nxt, future_states):
                    continue
                # Leave enough room for the rest of the path after the forced bounce.
                if remaining > 1:
                    probe = _walk_segment(future_states, used | set(added), endpoint, next_dir, 1, h2, w2)
                    if probe is None:
                        continue
                options.append((added, next_dir))
            if not options:
                ok = False
                break

            added, next_dir = random.choice(options)
            for p in added:
                states.append(p)
                used.add(p)
            cur = states[-1]
            if kind is not None:
                bounces.append((len(states) - 1, kind))
                assert next_dir is not None
                cur_dir = (int(next_dir[0]), int(next_dir[1]))

        if ok and len(states) >= 8:
            return states, bounces, (int(dr0), int(dc0))
    return None


def _paint_states(grid: np.ndarray, states: list[tuple[int, int]], n: int, color: int) -> None:
    for r, c in states:
        grid[int(r) : int(r + n), int(c) : int(c + n)] = int(color)


def _platform_cells_at(
    state: tuple[int, int],
    n: int,
    incoming_dir: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    H, W = shape
    r, c = state
    dr, _dc = incoming_dir
    rr = int(r + n) if int(dr) == 1 else int(r - 1)
    if not (0 <= rr < H):
        return []
    c0 = max(0, int(c - 1))
    c1 = min(W - 1, int(c + n))
    return [(int(rr), int(cc)) for cc in range(int(c0), int(c1 + 1))]


def _wall_cells_at(
    state: tuple[int, int],
    n: int,
    incoming_dir: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    H, W = shape
    r, c = state
    _dr, dc = incoming_dir
    cc = int(c + n) if int(dc) == 1 else int(c - 1)
    if not (0 <= cc < W):
        return []
    r0 = max(0, int(r - 1))
    r1 = min(H - 1, int(r + n))
    return [(int(rr), int(cc)) for rr in range(int(r0), int(r1 + 1))]


def _incoming_dirs(states: list[tuple[int, int]]) -> list[tuple[int, int] | None]:
    inc: list[tuple[int, int] | None] = [None]
    for i in range(1, len(states)):
        pr, pc = states[i - 1]
        r, c = states[i]
        inc.append((int(r - pr), int(c - pc)))
    return inc


def _feasible_state_set(grid: np.ndarray, n: int) -> set[tuple[int, int]]:
    feasible = _feasible_mask(grid, n)
    return {(int(r), int(c)) for r, c in np.argwhere(feasible)}


def _make_spur(main_states: list[tuple[int, int]], h2: int, w2: int) -> list[tuple[int, int]] | None:
    used = set(main_states)
    for _ in range(120):
        length = random.randint(1, 3)
        direction = random.choice(_diag_dirs())
        start = (random.randint(0, h2 - 1), random.randint(0, w2 - 1))
        if any(max(abs(int(start[0]) - int(q[0])), abs(int(start[1]) - int(q[1]))) <= 2 for q in used):
            continue
        spur = [start]
        rr, cc = start
        ok = True
        for _ in range(length):
            rr += int(direction[0])
            cc += int(direction[1])
            p = (int(rr), int(cc))
            if not (0 <= p[0] < h2 and 0 <= p[1] < w2):
                ok = False
                break
            if p in used or p in spur:
                ok = False
                break
            if any(max(abs(int(p[0]) - int(q[0])), abs(int(p[1]) - int(q[1]))) <= 1 for q in main_states):
                ok = False
                break
            spur.append(p)
        if ok:
            return spur
    return None


def _expected_output(grid: np.ndarray, final_pos: tuple[int, int], start: tuple[int, int], n: int) -> np.ndarray:
    out = np.array(grid, copy=True)
    sr, sc = start
    tr, tc = final_pos
    out[int(sr) : int(sr + n), int(sc) : int(sc + n)] = int(_RIVER)
    out[int(tr) : int(tr + n), int(tc) : int(tc + n)] = int(_MOVER)
    return out


def _size_and_grid() -> tuple[int, int]:
    n = random.choice([1, 2, 3, 4])
    if n == 1:
        G = random.randint(10, 16)
    elif n == 2:
        G = random.randint(12, 18)
    elif n == 3:
        G = random.randint(14, 20)
    else:
        G = random.randint(16, 20)
    return n, G


def _gen_attempt() -> np.ndarray | None:
    reachable = random.random() < 0.55
    n, G = _size_and_grid()
    H = W = G
    h2 = H - n + 1
    w2 = W - n + 1
    if min(h2, w2) < 7:
        return None

    planned = _plan_main_states(h2, w2)
    if planned is None:
        return None
    main_states, bounces, init_dir = planned

    spur_states: list[tuple[int, int]] = []
    if not reachable:
        spur_states = _make_spur(main_states, h2, w2)
        if spur_states is None:
            return None

    dest_state = main_states[-1] if reachable else spur_states[-1]
    intended_states = set(main_states) | set(spur_states)

    grid = np.full((H, W), int(_BG), dtype=int)
    _paint_states(grid, main_states, n, _RIVER)
    if spur_states:
        _paint_states(grid, spur_states, n, _RIVER)

    inc = _incoming_dirs(main_states)
    obstacle_cells: dict[tuple[int, int], int] = {}
    for idx, kind in bounces:
        incoming = inc[idx]
        assert incoming is not None
        if kind == "platform":
            cells = _platform_cells_at(main_states[idx], n, incoming, grid.shape)
            color = int(_PLATFORM)
        else:
            cells = _wall_cells_at(main_states[idx], n, incoming, grid.shape)
            color = int(_WALL)
        if not cells:
            return None
        for cell in cells:
            r, c = cell
            if grid[int(r), int(c)] != int(_BG):
                return None
            prev = obstacle_cells.get((int(r), int(c)))
            if prev is not None and int(prev) != int(color):
                return None
            obstacle_cells[(int(r), int(c))] = int(color)

    for (r, c), color in obstacle_cells.items():
        grid[int(r), int(c)] = int(color)

    sr, sc = main_states[0]
    dr0, dc0 = dest_state
    grid[int(sr) : int(sr + n), int(sc) : int(sc + n)] = int(_MOVER)
    grid[int(dr0) : int(dr0 + n), int(dc0) : int(dc0 + n)] = int(_DEST)

    if int(_corner_majority_bg(grid)) != int(_BG):
        return None

    feasible_states = _feasible_state_set(grid, n)
    if feasible_states != intended_states:
        return None

    try:
        got_dir = _unique_start_dir(_feasible_mask(grid, n), sr, sc)
    except AssertionError:
        return None
    if got_dir != init_dir:
        return None

    for idx, kind in bounces:
        incoming = inc[idx]
        assert incoming is not None
        r, c = main_states[idx]
        has_platform = _touch_platform(grid, r, c, n, incoming[0])
        has_wall = _touch_wall(grid, r, c, n, incoming[1])
        if kind == "platform":
            if not has_platform or has_wall:
                return None
        else:
            if not has_wall or has_platform:
                return None

    traced = _trace_motion(grid, n, sr, sc, got_dir[0], got_dir[1], dest_state)
    final_pos = traced["final"]
    expected = _expected_output(grid, final_pos, (sr, sc), n)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    planned_platforms = sum(1 for _idx, kind in bounces if kind == "platform")
    planned_walls = sum(1 for _idx, kind in bounces if kind == "wall")
    if traced["platform_bounces"] != planned_platforms:
        return None
    if traced["wall_bounces"] != planned_walls:
        return None
    if reachable and final_pos != dest_state:
        return None
    if not reachable and final_pos != main_states[-1]:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 600 attempts")
