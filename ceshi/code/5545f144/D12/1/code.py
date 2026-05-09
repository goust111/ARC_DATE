# Variant of 5545f144
# Dimension: D12 — 障碍物 (meta)
# Change: 网格中无障碍物——除精灵和标记外全为背景色，精灵可自由移动到任何背景格。... → 墙壁边界

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left

# ---------------------------------------------------------------------------
# Design analysis for the D12 wall variant
#
# spec:
#   preserved_core:
#     - Input is still a horizontal strip of time snapshots separated by divider columns.
#     - The sprite is still a single-color triangular arrow whose tip encodes facing.
#     - Markers are still single pixels of the same color as the sprite, and the output
#       is still the final single-frame state reached by consuming all remaining markers
#       starting from the rightmost snapshot.
#   D12 change:
#     - Every frame now also contains static wall segments in one extra color.
#     - Walls are straight horizontal/vertical segments and stay fixed across frames.
#     - A marker is only visible if it lies on a cardinal ray from the head before the
#       first wall cell on that ray; markers hidden behind a wall do not count.
#     - On landing, the re-stamped triangle must fit in-bounds and avoid wall cells.
#     - Output keeps the walls, because the final state is a frame of the same maze
#       with markers removed.
#
# concrete_trace:
#   - Suppose the head is at (5,5), facing right.
#   - There is a future marker at (5,9) and another at (2,5).
#   - In the obstacle-free original task, both rays would be considered, so the state
#     would be ambiguous and violate the "exactly one direction" rule.
#   - In this variant, a horizontal wall crossing column 5 at row 4 blocks the upward
#     ray, while the rightward ray stays open; only (5,9) is visible, so the move is
#     well-defined.
#   - This is the exact place where D12 changes the pipeline: frame rendering now needs
#     wall placement, `_choose_target()` must stop ray scans at walls, and generator
#     must construct wall segments that erase otherwise-ambiguous rays.
#
# complexity_calibration:
#   - Only one extra semantic object type is introduced: static walls in a single color.
#   - The movement rule remains discrete and local: same sprite templates, same markers,
#     same cardinal motion, same no-180° constraint.
#   - To make the variant visibly different from the original, the generator requires at
#     least one post-last-frame state where ignoring walls would expose markers on more
#     than one ray and therefore change or invalidate the solve.
#
# implicit-assumption_audit:
#   - Original `_split_frames()` treated every full-height solid non-bg column as a
#     divider. That is unsafe once walls exist, so divider detection is tightened to a
#     color-level rule: divider colors are used only by full-height solid columns.
#   - Original `main()` assumed the unique non-bg/non-divider color must be the actor.
#     That breaks with wall colors present, so actor detection now checks each candidate
#     color for a triangle + isolated-marker decomposition.
#   - Original `_choose_target()` only checked open-grid bounds and used the current
#     orientation's footprint. With walls, the landing pose itself must avoid walls, so
#     fit checks use the destination orientation.
#   - Generator must enforce that walls never overlap any marker or any sprite pose that
#     appears during the simulated path; otherwise solver/generator agreement would be
#     accidental.
#
# placement_legality_and_cleanliness:
#   - Wall segments are placed only on cells that are background for every head position
#     and every sprite pose in the rollout.
#   - Walls are short segments (length 2-4), so they read as clear barriers without
#     degenerating into frame dividers or filling the canvas with clutter.
#   - Marker placement still avoids 8-neighbor touching with previous markers/sprite
#     cells, so the actor color can be decomposed reliably by the solver.
# ---------------------------------------------------------------------------


def _find_divider_colors(g: np.ndarray, bg: int) -> set[int]:
    """Divider colors appear only as full-height solid columns."""
    h, w = g.shape
    divider_colors: set[int] = set()
    for color in set(map(int, g.ravel())) - {bg}:
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if cols and int(np.count_nonzero(g == color)) == len(cols) * h:
            divider_colors.add(color)
    return divider_colors


def _split_frames(g: np.ndarray) -> tuple[list[np.ndarray], int]:
    """
    Input is multiple frames concatenated horizontally, separated by full-height
    divider columns whose color is reserved for dividers only.
    """
    h, w = g.shape
    bg = _bg_color(g)
    divider_colors = _find_divider_colors(g, bg)
    div_cols = [
        c
        for c in range(w)
        if int(g[0, c]) in divider_colors and np.all(g[:, c] == g[0, c])
    ]

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


def _markers_are_isolated(markers: set[tuple[int, int]]) -> bool:
    for y, x in markers:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                if (y + dy, x + dx) in markers:
                    return False
    return True


def _find_actor_color(frame: np.ndarray, candidate_colors: set[int]) -> int:
    best = None  # (score, color)
    for color in candidate_colors:
        found = _find_agent(frame, color)
        if found is None:
            continue
        _tname, _d, _head, placed, markers = found
        if not _markers_are_isolated(markers):
            continue
        score = (len(placed), -len(markers))
        if best is None or score > best[0]:
            best = (score, int(color))
    assert best is not None, "Could not identify actor color in the latest frame"
    return best[1]


def _sprite_cells(head: tuple[int, int], tname: str, d: int) -> set[tuple[int, int]]:
    hr, hc = head
    return {(hr + dr, hc + dc) for dr, dc in TEMPLATES[tname][d]}


def _wall_mask(frame: np.ndarray, wall_colors: set[int]) -> np.ndarray:
    if not wall_colors:
        return np.zeros_like(frame, dtype=bool)
    return np.isin(frame, list(wall_colors))


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


def _can_place_agent(
    head: tuple[int, int],
    offs: list[tuple[int, int]],
    h: int,
    w: int,
    wall_mask: np.ndarray,
    respect_walls: bool,
) -> bool:
    hy, hx = head
    for oy, ox in offs:
        ny = hy + oy
        nx = hx + ox
        if not (0 <= ny < h and 0 <= nx < w):
            return False
        if respect_walls and wall_mask[ny, nx]:
            return False
    return True


def _ray_candidates(
    head: tuple[int, int],
    dir_idx: int,
    markers: set[tuple[int, int]],
    wall_mask: np.ndarray,
    respect_walls: bool,
) -> list[tuple[int, tuple[int, int]]]:
    h, w = wall_mask.shape
    hy, hx = head
    dy, dx = DIRS4[dir_idx]
    y, x = hy + dy, hx + dx
    dist = 1
    out: list[tuple[int, tuple[int, int]]] = []
    while 0 <= y < h and 0 <= x < w:
        if respect_walls and wall_mask[y, x]:
            break
        if (y, x) in markers:
            out.append((dist, (y, x)))
        y += dy
        x += dx
        dist += 1
    return out


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    tname: str,
    h: int,
    w: int,
    wall_mask: np.ndarray,
    respect_walls: bool,
) -> tuple[int, tuple[int, int]] | None:
    reachable: list[tuple[int, int, tuple[int, int]]] = []  # (dist, dir, target)
    for dd in range(4):
        offs = TEMPLATES[tname][dd]
        for dist, pt in _ray_candidates(head, dd, markers, wall_mask, respect_walls):
            if _can_place_agent(pt, offs, h, w, wall_mask, respect_walls):
                reachable.append((dist, dd, pt))

    if not reachable:
        return None

    dirs_with_candidates = {dd for _, dd, _ in reachable}
    assert len(dirs_with_candidates) == 1, (
        f"head={head} dir={d}: visible markers exist on multiple directions "
        f"{dirs_with_candidates}, expected exactly one after wall occlusion"
    )

    reachable.sort()
    _, dd, pt = reachable[0]
    assert dd != (d + 2) % 4, (
        f"head={head} dir={d}: would reverse to dir={dd}, not allowed"
    )
    return dd, pt


def _simulate_final_pose(
    frame: np.ndarray,
    actor: int,
    wall_colors: set[int],
    *,
    respect_walls: bool,
) -> tuple[str, int, tuple[int, int]]:
    h, w = frame.shape
    wall_mask = _wall_mask(frame, wall_colors)

    found = _find_agent(frame, actor)
    assert found is not None, "Could not detect the sprite in the latest frame"
    tname, d, head, agent, markers = found
    markers = set((int(y), int(x)) for y, x in markers)

    if respect_walls:
        for y, x in agent:
            assert not wall_mask[y, x], "Initial sprite overlaps a wall"

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        choice = _choose_target(head, d, markers, tname, h, w, wall_mask, respect_walls)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        nd, head = choice
        d = nd
        agent = _sprite_cells(head, tname, d)
        if respect_walls:
            assert all(not wall_mask[y, x] for y, x in agent), (
                f"Landing pose at head={head} overlaps a wall"
            )
        markers.discard(head)
    else:
        raise AssertionError("Simulation exceeded step budget")

    assert not markers, "Simulation ended before consuming every marker"
    return tname, d, head


def _simulate_to_end(
    frame: np.ndarray,
    actor: int,
    wall_colors: set[int],
    *,
    respect_walls: bool,
) -> np.ndarray:
    bg = _bg_color(frame)
    wall_mask = _wall_mask(frame, wall_colors)
    tname, d, head = _simulate_final_pose(
        frame, actor, wall_colors, respect_walls=respect_walls
    )
    out = np.where(wall_mask, frame, bg).astype(int)
    for y, x in _sprite_cells(head, tname, d):
        out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance a triangular sprite through a walled frame, consuming visible markers.

    Input structure:
    - Frames are concatenated left-to-right and may be separated by solid divider columns.
    - Each frame contains one arrow-shaped sprite, same-color single-pixel markers, and
      static straight wall segments in a different color.

    Rule:
    - Start from the rightmost frame.
    - From the sprite head, scan each cardinal ray until the first wall or the frame edge.
    - Only markers seen before that stop are candidates; exactly one direction has any.
    - Move the sprite to the nearest visible marker on that ray, re-face it to that
      travel direction, and repeat until no markers remain. Walls persist in the output.

    Key insight: The last snapshot still determines the whole suffix of the simulation,
    but "aligned" now means "aligned and not hidden behind a wall segment."
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    divider_colors = _find_divider_colors(g, bg)
    colors = set(map(int, g.ravel())) - {bg} - divider_colors
    start = frames[-1]
    actor = _find_actor_color(start, colors)
    wall_colors = colors - {actor}

    return _simulate_to_end(start, actor, wall_colors, respect_walls=True)


# --------------- generator (wall-aware variant) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (1, 5)          # 1..4 frames
FRAME_MAX = 16                   # sampled H/W upper bound is 15
H_FLOOR, H_CAP = 9, 15
W_FLOOR, W_CAP = 8, 15

# --- CFG: path complexity ---
STEPS_RANGE = (4, 13)            # 4..12 inclusive
STEP_DIST_RANGE = (2, 8)         # 2..7 inclusive
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: wall complexity ---
WALL_SEG_LEN_MAX = 4
WALL_SEG_TARGET_RANGE = (1, 4)   # 1..3 extra effective wall segments
MAX_WALL_ATTEMPTS = 60
MAX_REQUIREMENTS = 6

# --- CFG: retry budget ---
MAX_ATTEMPTS = 1200

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(1, 5)) for n in range(*STEPS_RANGE)}


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
) -> tuple[int, int, int, int, int, int, int, int | None, int, str]:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0]
    weights = np.array([(n ** 3) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    h_lo = min(H_CAP, max(H_FLOOR, 7 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 6 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = rng.permutation(10)
    bg = int(palette[0])
    actor = int(palette[1])
    divider = int(palette[2]) if n_frames > 1 else None
    wall = int(palette[3] if n_frames > 1 else palette[2])

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames, n_steps, gap, h, w, bg, actor, divider, wall, tname


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


def _marker_compatible_with_existing(
    nxt: tuple[int, int],
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    step_dirs: list[int],
    tname: str,
) -> bool:
    """
    Allow future markers on different rays: those are the ones walls will hide later.
    Still forbid anything that would pre-empt an already-planned next marker on the
    intended ray, and keep markers visually isolated from sprite cells.
    """
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, state_dirs[j])
        if any(_touching(nxt, cell) for cell in sprite):
            return False
        rel_dir = _direction(old_head, nxt)
        if rel_dir is None:
            continue
        if rel_dir == step_dirs[j] and j < len(heads) - 1:
            if _dist(old_head, nxt) <= _dist(old_head, heads[j + 1]):
                return False
    return True


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


def _visibility_requirements(
    heads: list[tuple[int, int]], step_dirs: list[int]
) -> list[tuple[int, int, int]]:
    """
    For every state i, gather each off-path ray that contains any future marker.
    Requirement tuple = (state_index, blocked_direction, nearest_future_index_on_that_ray).
    """
    reqs: list[tuple[int, int, int]] = []
    for i, need_dir in enumerate(step_dirs):
        nearest_by_dir: dict[int, tuple[int, int]] = {}
        for j in range(i + 2, len(heads)):
            dd = _direction(heads[i], heads[j])
            if dd is None or dd == need_dir:
                continue
            dist = _dist(heads[i], heads[j])
            prev = nearest_by_dir.get(dd)
            if prev is None or dist < prev[0]:
                nearest_by_dir[dd] = (dist, j)
        for dd, (_dist_to_marker, j) in sorted(nearest_by_dir.items()):
            reqs.append((i, dd, j))
    return reqs


def _requirements_after(last_idx: int, reqs: list[tuple[int, int, int]]) -> int:
    return sum(i >= last_idx for i, _dd, _j in reqs)


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


def _occupied_cells(
    heads: list[tuple[int, int]],
    placements: list[tuple[tuple[int, int], int]],
    tname: str,
) -> set[tuple[int, int]]:
    occ = set(heads)
    for head, d in placements:
        occ |= _sprite_cells(head, tname, d)
    return occ


def _candidate_block_positions(
    src: tuple[int, int], dst: tuple[int, int]
) -> list[tuple[int, int]]:
    dd = _direction(src, dst)
    assert dd is not None
    dy, dx = DIRS4[dd]
    y, x = src[0] + dy, src[1] + dx
    out: list[tuple[int, int]] = []
    while (y, x) != dst:
        out.append((y, x))
        y += dy
        x += dx
    return out


def _propose_segment(
    rng: np.random.Generator,
    cross: tuple[int, int],
    ray_dir: int,
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    y, x = cross
    if cross in blocked:
        return None

    if ray_dir in (0, 2):
        axis = [(0, -1), (0, 1)]  # horizontal wall blocks vertical ray
    else:
        axis = [(-1, 0), (1, 0)]  # vertical wall blocks horizontal ray

    free_neg = 0
    ny, nx = y + axis[0][0], x + axis[0][1]
    while 0 <= ny < h and 0 <= nx < w and (ny, nx) not in blocked and free_neg < WALL_SEG_LEN_MAX - 1:
        free_neg += 1
        ny += axis[0][0]
        nx += axis[0][1]

    free_pos = 0
    ny, nx = y + axis[1][0], x + axis[1][1]
    while 0 <= ny < h and 0 <= nx < w and (ny, nx) not in blocked and free_pos < WALL_SEG_LEN_MAX - 1:
        free_pos += 1
        ny += axis[1][0]
        nx += axis[1][1]

    options: list[tuple[int, int, int]] = []
    max_len = min(WALL_SEG_LEN_MAX, 1 + free_neg + free_pos)
    for length in range(max_len, 1, -1):
        need = length - 1
        lo = max(0, need - free_pos)
        hi = min(free_neg, need)
        for neg_used in range(lo, hi + 1):
            pos_used = need - neg_used
            options.append((length, neg_used, pos_used))

    if not options:
        return None

    options.sort(reverse=True)
    top_len = options[0][0]
    top = [opt for opt in options if opt[0] == top_len]
    length, neg_used, pos_used = top[int(rng.integers(0, len(top)))]

    cells: list[tuple[int, int]] = []
    for k in range(neg_used, 0, -1):
        cells.append((y + axis[0][0] * k, x + axis[0][1] * k))
    cells.append((y, x))
    for k in range(1, pos_used + 1):
        cells.append((y + axis[1][0] * k, x + axis[1][1] * k))
    return cells


def _place_walls(
    rng: np.random.Generator,
    h: int,
    w: int,
    heads: list[tuple[int, int]],
    placements: list[tuple[tuple[int, int], int]],
    tname: str,
    reqs: list[tuple[int, int, int]],
    last_idx: int,
) -> tuple[set[tuple[int, int]], list[list[tuple[int, int]]]] | None:
    occupied = _occupied_cells(heads, placements, tname)
    need_suffix = {(i, dd, j) for i, dd, j in reqs if i >= last_idx}
    order_base = [req for req in reqs if req in need_suffix] + [req for req in reqs if req not in need_suffix]
    target_extra = int(rng.integers(*WALL_SEG_TARGET_RANGE))

    for _ in range(MAX_WALL_ATTEMPTS):
        walls: set[tuple[int, int]] = set()
        segments: list[list[tuple[int, int]]] = []
        ok = True
        order = list(order_base)
        tail = order[len(need_suffix):]
        rng.shuffle(tail)
        order = order[: len(need_suffix)] + tail

        for i, dd, j in order:
            cross_positions = _candidate_block_positions(heads[i], heads[j])
            if not cross_positions:
                ok = False
                break

            scored: list[tuple[int, tuple[int, int]]] = []
            for pos in cross_positions:
                score = _dist(pos, heads[j])  # prefer farther-from-source blockers
                scored.append((score, pos))
            scored.sort(reverse=True)
            picks = [pos for _score, pos in scored]
            if len(picks) > 6:
                picks = picks[:6]
            rng.shuffle(picks)

            segment = None
            blocked = occupied | walls
            for cross in picks:
                trial = _propose_segment(rng, cross, dd, h, w, blocked)
                if trial is not None:
                    segment = trial
                    break
            if segment is None:
                ok = False
                break
            walls.update(segment)
            segments.append(segment)

        if not ok:
            continue

        # Optional extra walls on already-blocked rays, purely for stronger visual cues.
        extras_added = 0
        extra_candidates = list(reqs)
        rng.shuffle(extra_candidates)
        for i, dd, j in extra_candidates:
            if extras_added >= target_extra:
                break
            blocked = occupied | walls
            cross_positions = _candidate_block_positions(heads[i], heads[j])
            rng.shuffle(cross_positions)
            for cross in cross_positions[:4]:
                seg = _propose_segment(rng, cross, dd, h, w, blocked)
                if seg is None:
                    continue
                if any(cell in walls for cell in seg):
                    continue
                walls.update(seg)
                segments.append(seg)
                extras_added += 1
                break

        return walls, segments
    return None


def _render_frame(
    h: int,
    w: int,
    bg: int,
    actor: int,
    wall: int,
    head: tuple[int, int],
    d: int,
    tname: str,
    markers: list[tuple[int, int]],
    walls: set[tuple[int, int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    for wy, wx in walls:
        frame[wy, wx] = wall
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


def _ray_blocked(src: tuple[int, int], dst: tuple[int, int], walls: set[tuple[int, int]]) -> bool:
    dd = _direction(src, dst)
    if dd is None:
        return False
    dy, dx = DIRS4[dd]
    y, x = src[0] + dy, src[1] + dx
    while (y, x) != dst:
        if (y, x) in walls:
            return True
        y += dy
        x += dx
    return False


def _count_blocked_requirements(
    heads: list[tuple[int, int]],
    reqs: list[tuple[int, int, int]],
    walls: set[tuple[int, int]],
    *,
    start_idx: int,
) -> int:
    total = 0
    for i, _dd, j in reqs:
        if i < start_idx:
            continue
        if _ray_blocked(heads[i], heads[j], walls):
            total += 1
    return total


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, gap, h, w, bg, actor, divider, wall, tname = _sample_params(rng)
    step_dirs = _make_step_dirs(rng, n_steps)

    rollout = _rollout_path(rng, tname, step_dirs, h, w)
    if rollout is None:
        return None
    heads, state_dirs, fit_placements = rollout

    ok, turns, dists = _validate_path(step_dirs, heads)
    if not ok:
        return None

    reqs = _visibility_requirements(heads, step_dirs)
    if not reqs or len(reqs) > MAX_REQUIREMENTS:
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
    reqs = _visibility_requirements(heads, step_dirs)
    if _requirements_after(last_idx, reqs) == 0:
        return None

    wall_data = _place_walls(rng, h, w, heads, placements, tname, reqs, last_idx)
    if wall_data is None:
        return None
    walls, wall_segments = wall_data

    blocked_after_last = _count_blocked_requirements(
        heads, reqs, walls, start_idx=last_idx
    )
    if blocked_after_last == 0:
        return None

    frames = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        frames.append(
            _render_frame(h, w, bg, actor, wall, head, d, tname, heads[si + 1 :], walls)
        )
    fhead, fdir = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, wall, fhead, fdir, tname, [], walls)
    inp = _assemble_input(frames, bg, divider)

    # Oracle: actor detection, marker recovery, wall-aware solve, and no-wall divergence.
    start = frames[-1]
    actor_guess = _find_actor_color(start, set(map(int, start.ravel())) - {bg})
    if actor_guess != actor:
        return None

    found = _find_agent(start, actor)
    if found is None:
        return None
    f_tname, f_dir, f_head, _f_agent, f_markers = found
    if (f_tname, f_dir, f_head) != (tname, placements[last_idx][1], placements[last_idx][0]):
        return None
    if f_markers != set(heads[last_idx + 1 :]):
        return None

    solved = main(inp)
    if not np.array_equal(solved, out_frame):
        return None

    try:
        no_wall_out = _simulate_to_end(start, actor, {wall}, respect_walls=False)
    except Exception:
        no_wall_out = None
    if no_wall_out is not None and np.array_equal(no_wall_out, out_frame):
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
            "wall": wall,
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
            "count_in_first_frame": n_steps,
            "count_in_last_input_frame": gap,
            "visibility_requirements": list(reqs),
            "blocked_requirements_after_last_frame": blocked_after_last,
            "n_wall_segments": len(wall_segments),
            "wall_segments": [list(seg) for seg in wall_segments],
            "no_wall_solver_matches": bool(
                no_wall_out is not None and np.array_equal(no_wall_out, out_frame)
            ),
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        try:
            case = _build_case(rng)
        except Exception:
            case = None
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
