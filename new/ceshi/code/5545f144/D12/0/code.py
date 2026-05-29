# Variant of 5545f144
# Dimension: D12 — 障碍物 (meta)
# Change: 网格中无障碍物——除精灵和标记外全为背景色，精灵可自由移动到任何背景格。... → 静态障碍色块

"""
Design analysis for the D12 obstacle variant.

a) Concrete behavior and trace
- The actor is still a single-color triangular sprite, and ordinary markers are
  still single pixels of that same color. The new object is a static 2x2 block
  obstacle in one extra color. Obstacles are present in every frame and remain
  in the output.
- The last input frame is still the simulation start state for the solver. The
  sprite only reacts to markers that lie on one of the four cardinal half-lines
  from its head. The D12 change is that an aligned marker is ignored when an
  obstacle cell lies strictly between the head and that marker on the same ray.
- Small trace:
  head=(5,5), next real marker=(1,5), blocked decoy=(5,10), obstacle block
  contains (5,7) and (5,8). In the original task, the solver sees markers both
  up and right, which is ambiguous. In the variant, the right marker is filtered
  out because the obstacle occupies the open segment between head and marker, so
  only the upward marker remains reachable and the path continues unchanged.

b) Complexity calibration
- The obstacle mechanism is deliberately binary and local, matching the source
  idea from collision_stop: obstacles are hard blockers, not a general maze.
- To keep the change visible, every generated case contains at least one marker
  that is aligned from the last input frame but blocked by an obstacle. The
  output retains that leftover blocked marker, so the obstacle effect is visible
  without reading color ids or hidden metadata.
- The obstacle vocabulary is intentionally small: one obstacle color and 2x2
  obstacle blocks only. This keeps the scene readable and avoids accidental
  shape confusion with the actor templates.

c) Implicit assumptions in the original main() that break here
- Actor-color inference assumed exactly one non-bg/non-divider color. That is no
  longer true once obstacles add a second content color, so the new solver must
  identify the actor by template matching on the last frame.
- Reachability previously meant “aligned and still inside the grid”. That relied
  on the old world containing only background outside the actor/markers. In the
  variant, alignment alone is insufficient; obstacle-blocked rays must be
  rejected before choosing a direction.
- The old loop asserted that markers remaining implies some legal move exists.
  That fails once blocked decoys are allowed, so the new stop condition becomes
  “no reachable marker remains”, leaving blocked markers untouched.
- Rendering the old output on clean background was safe because every marker was
  guaranteed to disappear. Here the final frame must preserve static obstacles
  and any stranded blocked markers.

d) Placement legality guaranteed by the generator
- Obstacles never overlap any sprite placement on the full planned path.
- Blocked decoys never overlap or touch any sprite placement or any real path
  marker, so actor detection stays stable and the scenes remain legible.
- The actual consumable path is generated first using the original constraints,
  then obstacles/blocked decoys are added only if an oracle replay proves that
  the reachable path is unchanged and the decoy survives to the end.

e) Visual observability
- The new rule is inferable from 2-4 examples because obstacle blocks use a
  separate repeated color, blocked markers remain in the output, and at least
  one blocked marker is already aligned in the rightmost frame. This creates a
  visible “dot behind a wall does not get eaten” pattern.

f) Visual cleanliness
- Obstacles are only 2x2 blocks, markers remain single pixels, and the actor
  keeps its original templates. The generator preserves at least one-cell actor
  isolation around blocked decoys, so the sprite, ordinary markers, blocked
  markers, and obstacle blocks stay visually separable.

g) Color inference
- No color remapping is introduced. Colors only encode object categories:
  background, actor+markers, divider, and obstacle. The task logic depends on
  geometry and occlusion, not on numeric color ordering.
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


# Templates are defined as offsets relative to the head pixel at (0, 0).
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
    Move a triangular sprite through the latest snapshot, but obstacle blocks can leave dots stranded.

    Input structure:
    - Frames are concatenated horizontally and split by full-height divider columns; the background is the most common color.
    - Each frame contains one actor color shared by the triangle sprite and the single-pixel markers.
    - A separate static obstacle color forms 2x2 blocks that stay fixed across the snapshots.

    Rule:
    - Start from the rightmost frame.
    - A marker only counts if it is on the same row or column as the head, there is no obstacle cell strictly between the head and that marker, and the landed sprite would not cover an obstacle.
    - After obstacle filtering, exactly one cardinal direction can contain reachable markers; move to the nearest one on that half-line and rotate to face that direction.
    - Stop when no reachable marker remains, then output the final single frame with the sprite, the static obstacles, and any blocked leftover markers.

    Key insight: Aligned dots are no longer enough; the decisive step is to discard every line-of-sight candidate hidden behind an obstacle block before choosing the next move.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)
    start = frames[-1]
    actor = _detect_actor_color(start, bg)
    return _simulate_to_end(start, actor)


# --------------- generator (path-first + obstacle-decoy oracle) ---------------

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
    weights = np.array([(n ** 3) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
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
    out_markers = blocked_list
    out_frame = _render_frame(h, w, bg, actor, obstacle, fhead, fdir, tname, out_markers, obstacles)
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
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
