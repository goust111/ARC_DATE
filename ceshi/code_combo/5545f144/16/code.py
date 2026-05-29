# Combo variant of 5545f144 (index 16)
# combo_spec: D5/0+D12/0+D22/2
# Name: 标记转化+静态障碍+到达终点标记停止

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

# ---------------------------------------------------------------------------
# Design analysis for the fused combo
# ---------------------------------------------------------------------------
# a) Concrete behavior and trace
#    The base task has one actor color shared by the triangular sprite and the
#    ordinary single-pixel markers. This combo adds three coupled changes:
#    - ordinary markers become persistent trail pixels after being visited,
#    - one future marker has a dedicated terminal color and ends the simulation,
#    - a static 2x2 obstacle block blocks line-of-sight to a decoy marker.
#    Small trace:
#    - Last input frame: head=(6,6), the real next stop is the terminal at
#      (6,10), older visited stops are already recolored trail dots, and a decoy
#      actor-colored dot sits at (2,6) with one obstacle cell between it and the
#      head. Without the combo changes, the ordinary solver would either remove
#      visited dots, keep moving past the terminal, or treat the aligned decoy as
#      another candidate. In the fused rule, the decoy is ignored because the
#      obstacle blocks that ray, the head moves onto the terminal, and the output
#      keeps the recolored trail plus any leftover ordinary dots.
#
# b) Complexity calibration
#    To make all three changes visible, every generated case uses at least two
#    frames, at least one visible trail dot in the rightmost frame, exactly one
#    visible terminal-colored dot, and exactly one visible 2x2 obstacle block.
#    The terminal is always the immediate next real step from the last frame, so
#    the output still leaves ordinary markers behind and the stop-early effect is
#    obvious. One blocked decoy is enough to expose the obstacle rule without
#    turning the scene into a maze.
#
# c) Implicit assumption audit
#    The original main() relied on three assumptions that now break:
#    - one non-background color was enough to identify the actor; now actor,
#      trail, terminal, obstacle, and divider are distinct colors.
#    - every aligned in-bounds marker was reachable; now obstacle cells can hide
#      an aligned decoy, so line-of-sight must be checked before choosing.
#    - the process ended only when every marker disappeared; now ordinary markers
#      can remain because reaching the terminal stops the simulation early, and
#      visited ordinary stops must persist as trail pixels.
#    The fused solver therefore detects the actor by template matching, infers
#    obstacle/trail/terminal roles from framewise color behavior, and simulates a
#    unified state with `regular markers + terminal markers + visited trail`.
#
# d) Placement legality
#    The generator builds an ordinary valid path first, then recolors the next
#    real stop after the last input frame into the terminal, and finally inserts
#    one blocked decoy plus one 2x2 obstacle block. It rejects cases where the
#    obstacle or decoy would overlap any sprite pose, marker, trail stop, or the
#    terminal. It also preserves the D5-style trail clarity constraint so later
#    poses do not stamp over earlier visible trail dots.
#
# e) Visual observability
#    A human can infer the combo from repeated visual cues: trail-colored dots
#    accumulate across snapshots, one differently colored future dot is special
#    because the output sprite ends exactly there, and at least one ordinary dot
#    remains stranded behind a repeated 2x2 obstacle block. No hidden ordering or
#    numeric color semantics are used.
#
# f) Visual cleanliness
#    Object vocabularies stay simple: the sprite keeps the original templates,
#    ordinary and trail markers stay isolated single pixels, the terminal is one
#    single pixel, and the obstacle is one 2x2 block. The generator keeps at
#    least one-cell separation between the blocked decoy and all sprite/trail
#    cells, so the scene remains legible.
#
# g) Change inferability
#    The color-role mapping is structurally signaled rather than arbitrary:
#    trail is absent in frame 0 but appears later, terminal is a singleton future
#    dot visible from frame 0 onward, and obstacle is the only persistent 2x2
#    block color. That makes the color changes recoverable from a few examples.

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left


def _split_frames(g: np.ndarray) -> tuple[list[np.ndarray], int]:
    """
    Input is multiple frames concatenated horizontally, separated by full-height
    divider columns of a non-background color. Return list of frames and bg.
    """
    h, w = g.shape
    bg = _bg_color(g)

    solid_cols = [c for c in range(w) if len(set(map(int, g[:, c]))) == 1]
    div_cols = [c for c in solid_cols if int(g[0, c]) != bg]

    cuts = [-1] + div_cols + [w]
    frames: list[np.ndarray] = []
    for i in range(len(cuts) - 1):
        a = cuts[i] + 1
        b = cuts[i + 1]
        frames.append(g[:, a:b])
    return frames, bg


TEMPLATES: dict[str, dict[int, list[tuple[int, int]]]] = {
    "hollow3": {
        0: [(0, 0), (1, -1), (1, 1)],
        1: [(0, 0), (-1, -1), (1, -1)],
        2: [(0, 0), (-1, -1), (-1, 1)],
        3: [(0, 0), (-1, 1), (1, 1)],
    },
    "filled4": {
        0: [(0, 0), (1, -1), (1, 0), (1, 1)],
        1: [(0, 0), (-1, -1), (0, -1), (1, -1)],
        2: [(0, 0), (-1, -1), (-1, 0), (-1, 1)],
        3: [(0, 0), (-1, 1), (0, 1), (1, 1)],
    },
    "stem4": {
        0: [(0, 0), (1, 0), (2, -1), (2, 1)],
        1: [(0, 0), (0, -1), (-1, -2), (1, -2)],
        2: [(0, 0), (-1, 0), (-2, -1), (-2, 1)],
        3: [(0, 0), (0, 1), (-1, 2), (1, 2)],
    },
    "midfilled6": {
        0: [(0, 0), (1, -1), (1, 0), (1, 1), (2, -1), (2, 1)],
        1: [(0, 0), (-1, -1), (0, -1), (1, -1), (-1, -2), (1, -2)],
        2: [(0, 0), (-1, -1), (-1, 0), (-1, 1), (-2, -1), (-2, 1)],
        3: [(0, 0), (-1, 1), (0, 1), (1, 1), (-1, 2), (1, 2)],
    },
}


def _find_agent(
    frame: np.ndarray, actor: int
) -> tuple[str, int, tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]] | None:
    pts = set(map(tuple, np.argwhere(frame == actor)))
    if not pts:
        return None

    best = None  # (score, template_name, dir, head, placed)
    for head in pts:
        hy, hx = int(head[0]), int(head[1])
        for tname, dirs in TEMPLATES.items():
            for d, offs in dirs.items():
                placed = {(hy + dy, hx + dx) for dy, dx in offs}
                if placed.issubset(pts):
                    leftover = len(pts - placed)
                    score = (len(placed), -leftover)
                    if best is None or score > best[0]:
                        best = (score, tname, d, (hy, hx), placed)

    if best is None:
        return None
    _, tname, d, head, placed = best
    markers = pts - placed
    return tname, d, head, set(map(tuple, placed)), set(map(tuple, markers))


def _template_bounds(tname: str, d: int) -> tuple[int, int, int, int]:
    offs = TEMPLATES[tname][d]
    rs = [dr for dr, _ in offs]
    cs = [dc for _, dc in offs]
    return min(rs), max(rs), min(cs), max(cs)


def _template_patch(tname: str, d: int, actor: int, bg: int) -> tuple[np.ndarray, int, int]:
    r0, r1, c0, c1 = _template_bounds(tname, d)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for dr, dc in TEMPLATES[tname][d]:
        patch[dr - r0, dc - c0] = actor
    return patch, -r0, -c0


def _sprite_cells(head: tuple[int, int], tname: str, d: int) -> set[tuple[int, int]]:
    hr, hc = head
    return {(hr + dr, hc + dc) for dr, dc in TEMPLATES[tname][d]}


def _direction(a: tuple[int, int], b: tuple[int, int]) -> int | None:
    ar, ac = a
    br, bc = b
    if ac == bc and br < ar:
        return 0
    if ar == br and bc > ac:
        return 1
    if ac == bc and br > ar:
        return 2
    if ar == br and bc < ac:
        return 3
    return None


def _dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _touching(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _cells_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    out: list[tuple[int, int]] = []
    if ar == br:
        step = 1 if bc > ac else -1
        for c in range(ac + step, bc, step):
            out.append((ar, c))
    elif ac == bc:
        step = 1 if br > ar else -1
        for r in range(ar + step, br, step):
            out.append((r, ac))
    return out


def _placement_valid(
    head: tuple[int, int],
    tname: str,
    d: int,
    h: int,
    w: int,
    obstacles: set[tuple[int, int]],
) -> bool:
    for y, x in _sprite_cells(head, tname, d):
        if not (0 <= y < h and 0 <= x < w):
            return False
        if (y, x) in obstacles:
            return False
    return True


def _aligned_candidates(
    head: tuple[int, int], dir_idx: int, markers: set[tuple[int, int]]
) -> list[tuple[int, tuple[int, int]]]:
    hy, hx = head
    out: list[tuple[int, tuple[int, int]]] = []
    for y, x in markers:
        if dir_idx == 0 and x == hx and y < hy:
            out.append((hy - y, (y, x)))
        elif dir_idx == 1 and y == hy and x > hx:
            out.append((x - hx, (y, x)))
        elif dir_idx == 2 and x == hx and y > hy:
            out.append((y - hy, (y, x)))
        elif dir_idx == 3 and y == hy and x < hx:
            out.append((hx - x, (y, x)))
    out.sort()
    return out


def _first_reachable_on_dir(
    head: tuple[int, int],
    dir_idx: int,
    markers: set[tuple[int, int]],
    tname: str,
    h: int,
    w: int,
    obstacles: set[tuple[int, int]],
) -> tuple[int, tuple[int, int]] | None:
    for dist, pt in _aligned_candidates(head, dir_idx, markers):
        if any(cell in obstacles for cell in _cells_between(head, pt)):
            continue
        if not _placement_valid(pt, tname, dir_idx, h, w, obstacles):
            continue
        return dist, pt
    return None


def _reachable_choices(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    tname: str,
    h: int,
    w: int,
    obstacles: set[tuple[int, int]],
) -> list[tuple[int, int, tuple[int, int]]]:
    reachable: list[tuple[int, int, tuple[int, int]]] = []
    for dd in range(4):
        found = _first_reachable_on_dir(head, dd, markers, tname, h, w, obstacles)
        if found is not None:
            dist, pt = found
            reachable.append((dist, dd, pt))
    reachable.sort()
    return reachable


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    tname: str,
    h: int,
    w: int,
    obstacles: set[tuple[int, int]],
) -> tuple[int, tuple[int, int]] | None:
    reachable = _reachable_choices(head, d, markers, tname, h, w, obstacles)
    if not reachable:
        return None

    dirs_with_candidates = {dd for _, dd, _ in reachable}
    assert len(dirs_with_candidates) == 1, (
        f"head={head} dir={d}: reachable markers in multiple directions "
        f"{dirs_with_candidates}, expected exactly one after obstacle filtering"
    )

    _, dd, pt = reachable[0]
    assert dd != (d + 2) % 4, (
        f"head={head} dir={d}: would reverse to dir={dd}, not allowed"
    )
    return dd, pt


def _detect_actor_color(frame: np.ndarray, bg: int) -> int:
    best = None  # (score, color)
    for color in sorted(set(map(int, frame.ravel())) - {bg}):
        found = _find_agent(frame, color)
        if found is None:
            continue
        _, _, _, placed, markers = found
        score = (len(placed), -len(markers))
        if best is None or score > best[0]:
            best = (score, int(color))
    assert best is not None, "could not identify actor color"
    return best[1]


def _is_single_2x2_block(frame: np.ndarray, color: int) -> bool:
    pts = [tuple(map(int, pt)) for pt in np.argwhere(frame == color)]
    if len(pts) != 4:
        return False
    rows = sorted({r for r, _ in pts})
    cols = sorted({c for _, c in pts})
    if len(rows) != 2 or len(cols) != 2:
        return False
    return set(pts) == {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}


def _detect_special_colors(
    frames: list[np.ndarray], bg: int, actor: int, div_colors: set[int]
) -> tuple[int, int, int]:
    colors = set(map(int, np.concatenate([frame.ravel() for frame in frames])))
    candidates = sorted(colors - {bg, actor} - div_colors)

    obstacle_candidates = [
        color for color in candidates if all(_is_single_2x2_block(frame, color) for frame in frames)
    ]
    assert len(obstacle_candidates) == 1, f"expected one obstacle color, got {obstacle_candidates}"
    obstacle = obstacle_candidates[0]

    trail_candidates = [
        color
        for color in candidates
        if color != obstacle and np.count_nonzero(frames[0] == color) == 0 and np.count_nonzero(frames[-1] == color) > 0
    ]
    assert len(trail_candidates) == 1, f"expected one trail color, got {trail_candidates}"
    trail = trail_candidates[0]

    terminal_candidates = [color for color in candidates if color not in {obstacle, trail}]
    assert len(terminal_candidates) == 1, f"expected one terminal color, got {terminal_candidates}"
    terminal = terminal_candidates[0]
    return trail, terminal, obstacle


def _render_state(
    h: int,
    w: int,
    bg: int,
    actor: int,
    trail: int,
    terminal: int,
    obstacle: int,
    head: tuple[int, int],
    d: int,
    tname: str,
    future_markers: list[tuple[int, int]],
    visited_markers: list[tuple[int, int]],
    terminal_marker: tuple[int, int] | None,
    obstacles: set[tuple[int, int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    for y, x in obstacles:
        frame[y, x] = obstacle
    for y, x in visited_markers:
        frame[y, x] = trail
    for y, x in future_markers:
        frame[y, x] = actor
    if terminal_marker is not None:
        frame[terminal_marker] = terminal
    patch, ar, ac = _template_patch(tname, d, actor, bg)
    paste(frame, patch, head[0] - ar, head[1] - ac, transparent=bg)
    return frame


def _simulate_to_terminal(
    frame: np.ndarray,
    actor: int,
    trail: int,
    terminal: int,
    obstacle: int,
) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, _agent, regular_markers = found
    regular_markers = {tuple(map(int, pt)) for pt in regular_markers}
    terminal_markers = {tuple(map(int, pt)) for pt in np.argwhere(frame == terminal)}
    obstacles = {tuple(map(int, pt)) for pt in np.argwhere(frame == obstacle)}
    visited = {tuple(map(int, pt)) for pt in np.argwhere(frame == trail)}
    visited.add((int(head[0]), int(head[1])))

    remaining = set(regular_markers) | set(terminal_markers)
    max_steps = h * w * 4
    for _ in range(max_steps):
        choice = _choose_target(head, d, remaining, tname, h, w, obstacles)
        if choice is None:
            break

        nd, (ty, tx) = choice
        head = (ty, tx)
        d = nd
        if head in terminal_markers:
            break
        regular_markers.discard(head)
        remaining.discard(head)
        visited.add(head)

    leftover_terminal = sorted(pt for pt in terminal_markers if pt != head)
    return _render_state(
        h=h,
        w=w,
        bg=bg,
        actor=actor,
        trail=trail,
        terminal=terminal,
        obstacle=obstacle,
        head=head,
        d=d,
        tname=tname,
        future_markers=sorted(regular_markers),
        visited_markers=sorted(visited),
        terminal_marker=leftover_terminal[0] if leftover_terminal else None,
        obstacles=obstacles,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a triangular sprite through the latest snapshot, leaving a trail, respecting obstacles, and stopping on a special endpoint.

    Input structure:
    - Frames are concatenated left-to-right with full-height divider columns; the most common color is the background.
    - Each frame contains one actor color shared by the sprite and the ordinary future markers.
    - A second content color appears only after frame 0 and marks already-visited stops, one persistent 2x2 block color marks static obstacles, and one singleton future marker color marks the terminal endpoint.

    Rule:
    - Read the rightmost frame as the current state.
    - From the head, only same-row or same-column markers with clear line-of-sight past the obstacle block are reachable; after obstacle filtering, exactly one direction remains valid and the sprite moves to the nearest marker on that half-line.
    - When the sprite lands on an ordinary marker, that stop becomes part of the trail; when it lands on the special terminal marker, the simulation stops immediately even if other ordinary markers remain.
    - Output the final single frame with the sprite at the terminal, all visited ordinary stops recolored as trail, every leftover ordinary marker still visible, and the obstacle block unchanged.

    Key insight: Three visual signals must be fused at once: the triangle gives direction, the obstacle removes fake aligned candidates, and the distinct endpoint color overrides the old “consume everything” stopping rule while the trail preserves the visited history.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    start = frames[-1]
    actor = _detect_actor_color(start, bg)
    trail, terminal, obstacle = _detect_special_colors(frames, bg, actor, div_colors)
    return _simulate_to_terminal(start, actor, trail, terminal, obstacle)


# --------------- generator (trail + endpoint stop + blocked decoy) ---------------

N_FRAMES_RANGE = (2, 5)
FRAME_MAX = 16
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 8, 15

STEPS_RANGE = (4, 13)
STEP_DIST_RANGE = (2, 8)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

MAX_ATTEMPTS = 2000
SEARCH_MARKER_LIMIT = 32
SEARCH_BLOCK_LIMIT = 32

_STEP_AVAILABILITY = {
    n: sum((n % f == 0) and (n // f >= 2) for f in range(2, 5))
    for n in range(*STEPS_RANGE)
}


def _shift_interval(
    placements: list[tuple[tuple[int, int], int]], h: int, w: int, tname: str
) -> tuple[int, int, int, int] | None:
    tr_lo, tr_hi = -10**9, 10**9
    tc_lo, tc_hi = -10**9, 10**9
    for (hr, hc), d in placements:
        r0, r1, c0, c1 = _template_bounds(tname, d)
        tr_lo = max(tr_lo, -hr - r0)
        tr_hi = min(tr_hi, h - 1 - hr - r1)
        tc_lo = max(tc_lo, -hc - c0)
        tc_hi = min(tc_hi, w - 1 - hc - c1)
        if tr_lo > tr_hi or tc_lo > tc_hi:
            return None
    return tr_lo, tr_hi, tc_lo, tc_hi


def _trail_clear_of_pose(
    visited_heads: list[tuple[int, int]],
    head: tuple[int, int],
    d: int,
    tname: str,
) -> bool:
    if not visited_heads:
        return True
    return _sprite_cells(head, tname, d).isdisjoint(set(visited_heads))


def _marker_compatible_with_existing(
    nxt: tuple[int, int],
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    step_dirs: list[int],
    tname: str,
) -> bool:
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, state_dirs[j])
        if any(_touching(nxt, cell) for cell in sprite):
            return False
        rel_dir = _direction(old_head, nxt)
        if rel_dir is None:
            continue
        if rel_dir != step_dirs[j]:
            return False
        if j < len(heads) - 1 and _dist(old_head, nxt) <= _dist(old_head, heads[j + 1]):
            return False
    return True


def _validate_path(step_dirs: list[int], heads: list[tuple[int, int]]) -> tuple[bool, int, list[int]]:
    n_steps = len(step_dirs)
    dir_set = set(step_dirs)
    if len(dir_set) < MIN_DIR_DIVERSITY:
        return False, 0, []
    if not (dir_set & {0, 2} and dir_set & {1, 3}):
        return False, 0, []
    turns = sum(step_dirs[i] != step_dirs[i - 1] for i in range(1, n_steps))
    if turns < MIN_TURNS_ACCEPT or not (TURN_RATIO_LO <= turns / n_steps <= TURN_RATIO_HI):
        return False, turns, []
    dists = [_dist(heads[i], heads[i + 1]) for i in range(n_steps)]
    if len(set(dists)) < MIN_DIST_VARIETY:
        return False, turns, dists
    return True, turns, dists


def _make_step_dirs(rng: np.random.Generator, n_steps: int) -> list[int]:
    turn_min = max(TURN_MIN_ABS, int(np.ceil(TURN_RATIO_LO * n_steps)))
    turn_max = min(n_steps - 1, int(np.floor(TURN_RATIO_HI * n_steps)))
    n_turns = int(rng.integers(turn_min, turn_max + 1))
    n_segments = n_turns + 1

    segments = [int(rng.integers(0, 4))]
    for _ in range(n_segments - 1):
        segments.append(int((segments[-1] + rng.choice([1, -1])) % 4))

    lengths = np.ones(n_segments, dtype=int)
    for _ in range(n_steps - n_segments):
        lengths[int(rng.integers(0, n_segments))] += 1

    out: list[int] = []
    for d, ln in zip(segments, lengths):
        out.extend([d] * int(ln))
    return out


def _rollout_path(
    rng: np.random.Generator, tname: str, step_dirs: list[int], h: int, w: int
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]] | None:
    n_steps = len(step_dirs)
    d0 = step_dirs[0]
    heads: list[tuple[int, int]] = [(0, 0)]
    state_dirs: list[int] = [d0]
    fit_placements: list[tuple[tuple[int, int], int]] = [((0, 0), d0)]

    for i in range(n_steps):
        cur = heads[-1]
        cur_dir = state_dirs[-1]
        next_dir = step_dirs[i]
        if next_dir == (cur_dir + 2) % 4:
            return None

        dr, dc = DIRS4[next_dir]
        dist_order = list(rng.permutation(np.arange(*STEP_DIST_RANGE)))
        chosen = None
        for dist in dist_order:
            nxt = (cur[0] + int(dist) * dr, cur[1] + int(dist) * dc)
            if nxt in heads:
                continue
            if any(_touching(nxt, m) for m in heads[1:]):
                continue
            if not _marker_compatible_with_existing(nxt, heads, state_dirs, step_dirs, tname):
                continue
            if not _trail_clear_of_pose(heads[1:], nxt, next_dir, tname):
                continue
            trial_fit = fit_placements + [(nxt, cur_dir), (nxt, next_dir)]
            if _shift_interval(trial_fit, h, w, tname) is None:
                continue
            chosen = nxt
            break

        if chosen is None:
            return None

        heads.append(chosen)
        state_dirs.append(next_dir)
        fit_placements.extend([(chosen, cur_dir), (chosen, next_dir)])

    return heads, state_dirs, fit_placements


def _sample_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int, int, int, int, int, str]:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0 and n // n_frames >= 2]
    if not multiples:
        raise AssertionError(f"no valid step counts for n_frames={n_frames}")
    weights = np.array([(n ** 3) / max(_STEP_AVAILABILITY[n], 1) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 6 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = list(map(int, rng.permutation(10)))
    bg, actor, trail, obstacle, terminal, divider = palette[:6]

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames, n_steps, gap, h, w, bg, actor, trail, obstacle, terminal, divider, tname


def _all_sprite_cells(
    heads: list[tuple[int, int]], state_dirs: list[int], tname: str
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for head, d in zip(heads, state_dirs):
        out |= _sprite_cells(head, tname, d)
    return out


def _block_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = top_left
    return {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}


def _candidate_block_toplefts(
    head: tuple[int, int], marker: tuple[int, int], h: int, w: int
) -> list[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in _cells_between(head, marker):
        for tr in (r - 1, r):
            for tc in (c - 1, c):
                if 0 <= tr < h - 1 and 0 <= tc < w - 1:
                    out.add((tr, tc))
    return list(out)


def _terminal_pose_is_clean(
    terminal_idx: int,
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
) -> bool:
    return _trail_clear_of_pose(heads[1:terminal_idx], heads[terminal_idx], state_dirs[terminal_idx], tname)


def _find_obstacle_setup(
    rng: np.random.Generator,
    start_idx: int,
    terminal_idx: int,
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
    h: int,
    w: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], dict] | None:
    actor_cells = _all_sprite_cells(heads, state_dirs, tname) | set(heads[1:])
    start_head = heads[start_idx]
    actual_dir = state_dirs[start_idx + 1]
    terminal_marker = heads[terminal_idx]
    visible_stops = set(heads[1:terminal_idx + 1])
    remaining = set(heads[start_idx + 1 :])

    candidate_dirs = [dd for dd in range(4) if dd != actual_dir]
    rng.shuffle(candidate_dirs)
    for dd in candidate_dirs:
        dr, dc = DIRS4[dd]
        candidate_markers: list[tuple[int, int]] = []
        for dist in range(3, max(h, w) + 1):
            m = (start_head[0] + dist * dr, start_head[1] + dist * dc)
            if not (0 <= m[0] < h and 0 <= m[1] < w):
                break
            if m in actor_cells or m == terminal_marker:
                continue
            if any(_touching(m, cell) for cell in actor_cells):
                continue
            if not _placement_valid(m, tname, dd, h, w, set()):
                continue
            candidate_markers.append(m)

        rng.shuffle(candidate_markers)
        for marker in candidate_markers[:SEARCH_MARKER_LIMIT]:
            block_tops = _candidate_block_toplefts(start_head, marker, h, w)
            rng.shuffle(block_tops)
            for top_left in block_tops[:SEARCH_BLOCK_LIMIT]:
                obstacles = _block_cells(top_left)
                if obstacles & actor_cells:
                    continue
                if any(_touching(cell, seen) for cell in obstacles for seen in visible_stops):
                    continue
                if marker in obstacles or terminal_marker in obstacles:
                    continue
                trial_remaining = set(remaining) | {marker}
                free_choices = _reachable_choices(start_head, state_dirs[start_idx], trial_remaining, tname, h, w, set())
                obs_choices = _reachable_choices(start_head, state_dirs[start_idx], trial_remaining, tname, h, w, obstacles)
                if not free_choices or not obs_choices:
                    continue
                if len({dd0 for _, dd0, _ in free_choices}) <= 1:
                    continue
                if obs_choices[0][1:] != (state_dirs[start_idx + 1], heads[start_idx + 1]):
                    continue
                if len({dd0 for _, dd0, _ in obs_choices}) != 1:
                    continue
                return obstacles, {marker}, {
                    "blocked_marker": marker,
                    "blocked_dir": dd,
                    "free_dir_count": len({dd0 for _, dd0, _ in free_choices}),
                }
    return None


def _assemble_input(frames: list[np.ndarray], bg: int, divider: int) -> np.ndarray:
    n_frames = len(frames)
    h, w = frames[0].shape
    total_w = n_frames * w + (n_frames - 1)
    inp = np.full((h, total_w), bg, dtype=int)
    for i, fr in enumerate(frames):
        c0 = i * (w + 1)
        paste(inp, fr, 0, c0)
        if i < n_frames - 1:
            inp[:, c0 + w] = divider
    return inp


def _build_case(rng: np.random.Generator) -> dict | None:
    sampled = _sample_params(rng)
    n_frames, n_steps, gap, h, w, bg, actor, trail, obstacle, terminal, divider, tname = sampled
    if gap < 2:
        return None

    step_dirs = _make_step_dirs(rng, n_steps)
    rollout = _rollout_path(rng, tname, step_dirs, h, w)
    if rollout is None:
        return None
    heads, state_dirs, fit_placements = rollout

    ok, turns, dists = _validate_path(step_dirs, heads)
    if not ok:
        return None

    shift = _shift_interval(fit_placements, h, w, tname)
    if shift is None:
        return None
    tr_lo, tr_hi, tc_lo, tc_hi = shift
    tr = int(rng.integers(tr_lo, tr_hi + 1))
    tc = int(rng.integers(tc_lo, tc_hi + 1))
    heads = [(r + tr, c + tc) for r, c in heads]
    placements = [(heads[i], state_dirs[i]) for i in range(n_steps + 1)]

    last_idx = (n_frames - 1) * gap
    terminal_idx = last_idx + 1
    if terminal_idx >= n_steps:
        return None
    terminal_marker = heads[terminal_idx]
    if not _terminal_pose_is_clean(terminal_idx, heads, state_dirs, tname):
        return None

    obstacle_setup = _find_obstacle_setup(rng, last_idx, terminal_idx, heads, state_dirs, tname, h, w)
    if obstacle_setup is None:
        return None
    obstacles, blocked_markers, obstacle_meta = obstacle_setup

    blocked_list = sorted(blocked_markers)
    frames: list[np.ndarray] = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        visited = heads[1 : si + 1]
        future_regular = [pt for pt in heads[si + 1 :] if pt != terminal_marker] + blocked_list
        visible_terminal = terminal_marker if si < terminal_idx else None
        frames.append(
            _render_state(
                h=h,
                w=w,
                bg=bg,
                actor=actor,
                trail=trail,
                terminal=terminal,
                obstacle=obstacle,
                head=head,
                d=d,
                tname=tname,
                future_markers=future_regular,
                visited_markers=visited,
                terminal_marker=visible_terminal,
                obstacles=obstacles,
            )
        )

    out_frame = _render_state(
        h=h,
        w=w,
        bg=bg,
        actor=actor,
        trail=trail,
        terminal=terminal,
        obstacle=obstacle,
        head=placements[terminal_idx][0],
        d=placements[terminal_idx][1],
        tname=tname,
        future_markers=[pt for pt in heads[terminal_idx + 1 :] if pt != terminal_marker] + blocked_list,
        visited_markers=heads[1:terminal_idx],
        terminal_marker=None,
        obstacles=obstacles,
    )
    inp = _assemble_input(frames, bg, divider)

    found = _find_agent(frames[-1], actor)
    if found is None:
        return None
    f_tname, f_dir, f_head, _f_agent, f_markers = found
    expected_last_markers = set(heads[last_idx + 1 :]) - {terminal_marker} | set(blocked_markers)
    if (f_tname, f_dir, f_head) != (tname, placements[last_idx][1], placements[last_idx][0]):
        return None
    if f_markers != expected_last_markers:
        return None

    if np.count_nonzero(frames[0] == trail) != 0:
        return None
    if np.count_nonzero(frames[-1] == trail) <= 0:
        return None
    if np.count_nonzero(frames[-1] == terminal) != 1:
        return None
    if not _is_single_2x2_block(frames[-1], obstacle):
        return None
    if np.count_nonzero(out_frame == trail) <= 0:
        return None
    if np.count_nonzero(out_frame == actor) <= len(TEMPLATES[tname][placements[terminal_idx][1]]):
        return None
    if not np.array_equal(main(inp), out_frame):
        return None

    return {
        "input": inp,
        "output": out_frame,
        "meta": {
            "frame_height": h,
            "frame_width": w,
            "n_frames": n_frames,
            "template": tname,
            "bg": bg,
            "actor": actor,
            "trail": trail,
            "obstacle": obstacle,
            "terminal": terminal,
            "divider": divider,
            "total_steps": n_steps,
            "gap": gap,
            "turns": turns,
            "turn_ratio": turns / n_steps,
            "direction_diversity": len(set(step_dirs)),
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "terminal_idx": terminal_idx,
            "terminal_marker": terminal_marker,
            "step_distances": dists,
            "blocked_markers": blocked_list,
            "obstacle_cells": sorted(obstacles),
            "visible_trail_in_last_frame": int(np.count_nonzero(frames[-1] == trail)),
            "leftover_actor_markers_in_output": int(np.count_nonzero(out_frame == actor) - len(TEMPLATES[tname][placements[terminal_idx][1]])),
            "obstacle_free_dir_count_from_last_frame": obstacle_meta["free_dir_count"],
            "blocked_dir_from_last_frame": obstacle_meta["blocked_dir"],
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
