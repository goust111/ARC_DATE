# Combo variant of 5545f144 (index 8)
# combo_spec: D12/0+D22/0
# Name: 静态障碍+部分消耗即停止

"""
Design analysis for combo D12/0 + D22/0.

a) Concrete behavior and trace
- The scene still contains horizontally concatenated snapshots of one triangular
  sprite plus actor-colored single-pixel markers. The fusion adds one obstacle
  color used for static 2x2 blocks that remain present in every frame and in
  the output.
- Original task trace: if the head is at H and actor-colored dots are aligned
  on exactly one visible half-line, the sprite jumps to the nearest dot there,
  re-faces that direction, and repeats until no dot remains anywhere.
- Combo trace: aligned dots are only candidates if no obstacle cell lies
  strictly between the head and that dot on the same row or column, and the
  landing sprite would not overlap an obstacle. Example:
  head=(6,4), true next marker=(2,4), decoy=(6,9), obstacle cells include
  (6,6),(6,7). The original solver would see both Up and Right and fail the
  single-direction invariant. The combo first discards the right decoy because
  the obstacle blocks line of sight, so the sprite moves upward. After all real
  path markers are consumed, the blocked dot can remain forever; the run stops
  at "no reachable marker remains" rather than "all markers were consumed".

b) Complexity calibration
- The new vocabulary stays small: one obstacle color and 2x2 obstacle blocks
  only. This is enough to make occlusion visually obvious without turning the
  task into a general maze.
- Every generated case includes at least one marker that is visible from the
  latest frame on a cardinal ray but is blocked by an obstacle, so the fusion
  effect is present in the sample itself instead of only in hidden replay.
- The consumable route keeps the original path complexity ranges, so the task is
  still about following multiple turns rather than reading a trivial one-step
  obstacle puzzle.

c) Implicit-assumption audit of the original main()
- Original actor-color inference assumed exactly one non-bg, non-divider color.
  That breaks once obstacles introduce another content color, so the combo must
  detect the actor by matching sprite templates on the latest frame.
- Original reachability assumed "aligned and in-bounds" was sufficient. That is
  false here: obstacle cells between head and marker invalidate the candidate,
  and obstacle overlap at the landed placement is also illegal.
- Original simulation assumed leftover markers implied a next move existed and
  asserted on being stuck. The combo replaces that with a normal stop when no
  non-reverse, obstacle-free target remains.
- Original output erased all non-sprite content. The combo must repaint static
  obstacles and any stranded blocked markers in the final frame.

d) Placement legality
- The generator first builds a clean original-style path, then inserts a 2x2
  obstacle block plus a blocked decoy marker only if oracle replay confirms the
  intended consumable suffix is unchanged.
- Obstacles never overlap any sprite cell on any visited state, and blocked
  markers never overlap or touch any sprite placement or path marker.
- The global translation step still guarantees every visible sprite placement
  fits inside the frame; obstacle insertion is rejected if it violates that.

e) Visual observability
- A human can see the new rule because the obstacle color repeats as a compact
  block and at least one actor-colored dot survives behind it in the output.
- The difference from the original is inferable from examples: some aligned dots
  are not eaten, and the shared cause is the visible obstacle block on the ray.
- No hidden dependence on numeric color ids or coordinate ordering is used; only
  visible alignment, distance, and occlusion matter.

f) Visual cleanliness
- Obstacles are limited to 2x2 blocks, markers remain single pixels, and the
  original triangle templates are unchanged, so object categories stay distinct.
- The generator enforces spacing so blocked decoys do not visually merge with
  the sprite body or with real path markers.
- The output therefore remains readable: one sprite, one or more obstacle cells,
  and a small number of leftover dots.

g) Change inferability
- The only added semantic color is "obstacle", and it repeats as a consistent
  rigid shape. The actor color continues to mean both sprite and markers.
- The fused behavior is structurally supported by repetition: obstacle blocks are
  static across frames, blocked dots survive to the answer, and oracle checks
  ensure this survival is caused by obstacle-filtered reachability rather than
  by accidental generator noise.
"""

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

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
        elif dir_idx == 2 and x == hx and y > hy:
            out.append((y - hy, (y, x)))
        elif dir_idx == 3 and y == hy and x < hx:
            out.append((hx - x, (y, x)))
        elif dir_idx == 1 and y == hy and x > hx:
            out.append((x - hx, (y, x)))
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
        if dd == (d + 2) % 4:
            continue
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
        f"{dirs_with_candidates}, expected exactly one after filtering"
    )

    _, dd, pt = reachable[0]
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


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    static_cells = {
        (int(y), int(x)): int(frame[y, x])
        for y, x in np.argwhere((frame != bg) & (frame != actor))
    }
    obstacles = set(static_cells)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, agent, markers = found
    agent = set((int(y), int(x)) for y, x in agent)
    markers = set((int(y), int(x)) for y, x in markers)

    max_steps = h * w * 4
    for _ in range(max_steps):
        choice = _choose_target(head, d, markers, tname, h, w, obstacles)
        if choice is None:
            break

        nd, (ty, tx) = choice
        head = (ty, tx)
        d = nd
        agent = _sprite_cells(head, tname, d)
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for (y, x), color in static_cells.items():
        out[y, x] = color
    for y, x in markers:
        out[y, x] = actor
    for y, x in agent:
        out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue the triangular sprite through the latest snapshot, but obstacle blocks can strand actor-colored dots.

    Input structure:
    - The input is 1-4 frames concatenated horizontally, optionally separated by full-height solid divider columns; the most common color is background.
    - Each frame contains one actor color shared by the triangle sprite and the single-pixel markers.
    - A separate static obstacle color appears as fixed blocks that persist across all snapshots.

    Rule:
    - Read the rightmost frame as the current state.
    - From the sprite head, a marker is legal only if it lies on the same row or column, is not in the exact reverse direction, has no obstacle cell strictly between the head and marker, and the landed sprite would not overlap any obstacle.
    - After filtering, at most one cardinal direction can contain legal markers; move to the nearest one on that half-line and rotate to that travel direction.
    - Stop when no legal marker remains, even if some blocked markers are still visible, then output the final frame with the sprite, the static obstacles, and any leftover actor-colored dots.

    Key insight: The actor still follows nearest aligned dots, but now the decisive step is obstacle-filtered line of sight; dots hidden behind a wall stop mattering, and once every remaining dot is blocked or illegal the simulation simply ends.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)
    start = frames[-1]
    actor = _detect_actor_color(start, bg)
    return _simulate_to_end(start, actor)


N_FRAMES_RANGE = (1, 5)
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

OBSTACLE_BLOCK_SIZE = 2
MAX_ATTEMPTS = 400
SEARCH_MARKER_LIMIT = 32
SEARCH_BLOCK_LIMIT = 32

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(1, 5)) for n in range(*STEPS_RANGE)}


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


def _sample_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int, int, int, int | None, str]:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0]
    weights = np.array([(n**3) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 6 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = rng.permutation(10)
    bg = int(palette[0])
    actor = int(palette[1])
    obstacle = int(palette[2])
    divider = int(palette[3]) if n_frames > 1 else None

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames, n_steps, gap, h, w, bg, actor, obstacle, divider, tname


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


def _block_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = top_left
    return {
        (r, c),
        (r + 1, c),
        (r, c + 1),
        (r + 1, c + 1),
    }


def _all_sprite_cells(
    heads: list[tuple[int, int]], state_dirs: list[int], tname: str
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for head, d in zip(heads, state_dirs):
        out |= _sprite_cells(head, tname, d)
    return out


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


def _planned_suffix_matches(
    start_idx: int,
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
    h: int,
    w: int,
    obstacles: set[tuple[int, int]],
    blocked_markers: set[tuple[int, int]],
) -> bool:
    markers = set(heads[start_idx + 1 :]) | set(blocked_markers)
    head = heads[start_idx]
    d = state_dirs[start_idx]
    try:
        for idx in range(start_idx, len(heads) - 1):
            choice = _choose_target(head, d, markers, tname, h, w, obstacles)
            if choice != (state_dirs[idx + 1], heads[idx + 1]):
                return False
            head = heads[idx + 1]
            d = state_dirs[idx + 1]
            markers.discard(head)
        return _choose_target(head, d, markers, tname, h, w, obstacles) is None and markers == set(blocked_markers)
    except AssertionError:
        return False


def _obstacle_witness_count(
    start_idx: int,
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
    h: int,
    w: int,
    obstacles: set[tuple[int, int]],
    blocked_markers: set[tuple[int, int]],
) -> tuple[int, bool]:
    markers = set(heads[start_idx + 1 :]) | set(blocked_markers)
    count = 0
    start_diff = False
    for idx in range(start_idx, len(heads) - 1):
        head = heads[idx]
        d = state_dirs[idx]
        obs_sig = tuple((dd, pt) for _, dd, pt in _reachable_choices(head, d, markers, tname, h, w, obstacles))
        free_sig = tuple((dd, pt) for _, dd, pt in _reachable_choices(head, d, markers, tname, h, w, set()))
        if obs_sig != free_sig:
            count += 1
            if idx == start_idx:
                start_diff = True
        markers.discard(heads[idx + 1])
    return count, start_diff


def _find_obstacle_setup(
    rng: np.random.Generator,
    start_idx: int,
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
    h: int,
    w: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], dict] | None:
    actor_cells = _all_sprite_cells(heads, state_dirs, tname) | set(heads[1:])
    start_head = heads[start_idx]
    actual_dir = state_dirs[start_idx + 1]

    candidate_dirs = [dd for dd in range(4) if dd != actual_dir]
    rng.shuffle(candidate_dirs)

    for dd in candidate_dirs:
        dr, dc = DIRS4[dd]
        candidate_markers: list[tuple[int, int]] = []
        for dist in range(3, max(h, w) + 1):
            m = (start_head[0] + dist * dr, start_head[1] + dist * dc)
            if not (0 <= m[0] < h and 0 <= m[1] < w):
                break
            if m in actor_cells:
                continue
            if any(_touching(m, cell) for cell in actor_cells):
                continue
            if not _placement_valid(m, tname, dd, h, w, set()):
                continue
            candidate_markers.append(m)

        if not candidate_markers:
            continue
        rng.shuffle(candidate_markers)

        for marker in candidate_markers[:SEARCH_MARKER_LIMIT]:
            block_tops = _candidate_block_toplefts(start_head, marker, h, w)
            if not block_tops:
                continue
            rng.shuffle(block_tops)

            for top_left in block_tops[:SEARCH_BLOCK_LIMIT]:
                block = _block_cells(top_left)
                if block & actor_cells:
                    continue
                if marker in block:
                    continue
                obstacles = set(block)
                blocked_markers = {marker}
                if not _planned_suffix_matches(
                    start_idx, heads, state_dirs, tname, h, w, obstacles, blocked_markers
                ):
                    continue
                witness_count, start_diff = _obstacle_witness_count(
                    start_idx, heads, state_dirs, tname, h, w, obstacles, blocked_markers
                )
                if witness_count <= 0 or not start_diff:
                    continue
                return obstacles, blocked_markers, {
                    "blocked_marker": marker,
                    "blocked_dir": dd,
                    "witness_count": witness_count,
                }

    return None


def _render_frame(
    h: int,
    w: int,
    bg: int,
    actor: int,
    obstacle: int,
    head: tuple[int, int],
    d: int,
    tname: str,
    markers: list[tuple[int, int]],
    obstacles: set[tuple[int, int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    for y, x in obstacles:
        frame[y, x] = obstacle
    patch, ar, ac = _template_patch(tname, d, actor, bg)
    paste(frame, patch, head[0] - ar, head[1] - ac, transparent=bg)
    for m in markers:
        frame[m] = actor
    return frame


def _assemble_input(frames: list[np.ndarray], bg: int, divider: int | None) -> np.ndarray:
    n_frames = len(frames)
    h, w = frames[0].shape
    total_w = n_frames * w + (n_frames - 1)
    inp = np.full((h, total_w), bg, dtype=int)
    for i, fr in enumerate(frames):
        c0 = i * (w + 1)
        paste(inp, fr, 0, c0)
        if divider is not None and i < n_frames - 1:
            inp[:, c0 + w] = divider
    return inp


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, gap, h, w, bg, actor, obstacle, divider, tname = _sample_params(rng)
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
    obstacle_setup = _find_obstacle_setup(rng, last_idx, heads, state_dirs, tname, h, w)
    if obstacle_setup is None:
        return None
    obstacles, blocked_markers, obstacle_meta = obstacle_setup

    frames = []
    blocked_list = sorted(blocked_markers)
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        visible_markers = list(heads[si + 1 :]) + blocked_list
        frames.append(
            _render_frame(h, w, bg, actor, obstacle, head, d, tname, visible_markers, obstacles)
        )

    fhead, fdir = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, obstacle, fhead, fdir, tname, blocked_list, obstacles)
    inp = _assemble_input(frames, bg, divider)

    found = _find_agent(frames[-1], actor)
    if found is None:
        return None
    f_tname, f_dir, f_head, _f_agent, f_markers = found
    expected_last_markers = set(heads[last_idx + 1 :]) | set(blocked_markers)
    if (f_tname, f_dir, f_head) != (tname, placements[last_idx][1], placements[last_idx][0]):
        return None
    if f_markers != expected_last_markers:
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
            "obstacle": obstacle,
            "divider": divider,
            "total_steps": n_steps,
            "gap": gap,
            "turns": turns,
            "turn_ratio": turns / n_steps,
            "direction_diversity": len(set(step_dirs)),
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "step_distances": dists,
            "reachable_markers_in_first_frame": n_steps,
            "blocked_markers": sorted(blocked_markers),
            "visible_markers_in_last_input_frame": gap + len(blocked_markers),
            "obstacle_cells": sorted(obstacles),
            "obstacle_witness_count": obstacle_meta["witness_count"],
            "blocked_dir_from_last_frame": obstacle_meta["blocked_dir"],
            "stop_leaves_marker_count": len(blocked_markers),
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
