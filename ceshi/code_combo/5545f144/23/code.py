# Combo variant of 5545f144 (index 23)
# combo_spec: D2/1+D12/1
# Name: L形精灵+墙壁边界

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left
#
# ------------------------------ Design analysis ------------------------------
# a) Concrete fusion + behavior trace
#    The moving actor is no longer any triangle family: it is a single 5-pixel
#    L pentomino whose missing corner indicates facing. We use one fixed family:
#      Up:    #.
#             ##
#             ##
#      Right: ###
#             ##.
#      Down:  ##
#             ##
#             .#
#      Left:  .##
#             ###
#    Walls are straight horizontal/vertical segments in a separate color and stay
#    fixed across all frames. Example: if the latest frame shows a Right-facing L
#    with head at (5, 5), a marker at (5, 9), and another marker at (2, 5), then
#    the obstacle-free logic would see both right and up rays and become invalid.
#    In the fused task, a horizontal wall crossing row 4 blocks the upward ray, so
#    only the rightward marker is visible; the head lands on (5, 9), the L stays
#    Right, and later turns are again determined by whichever unblocked ray is the
#    only visible one. The fusion therefore changes both object recognition (find
#    an L, not a triangle) and motion semantics (alignment now means line-of-sight
#    before the first wall, and the arrival L pose must not overlap walls).
#
# b) Complexity calibration
#    The variant keeps just one L template family and one wall color. That makes
#    the D2 change visually obvious while avoiding a combinatorial explosion of
#    sprite subclasses or wall types. Walls are short 2-4 pixel segments: long
#    enough to read as barriers, short enough not to impersonate frame dividers or
#    fill the grid with clutter. The generator also forces at least one suffix-step
#    requirement where walls hide an otherwise-visible off-path marker, so the D12
#    behavior is not decorative.
#
# c) Hidden-assumption review against the original solver
#    - Original `_find_agent()` searched for any triangle subset inside all actor
#      pixels. That is unsafe here because the L shape is larger and actor-colored
#      dots share the same color; the fused solver first splits the actor color into
#      4-connected components and demands exactly one 5-cell component matching an
#      L rotation, with all other actor components being singleton markers.
#    - Original `_split_frames()` treated every solid non-bg column as a divider.
#      With walls present, divider detection must be color-level: divider colors may
#      only appear as full-height solid columns.
#    - Original `main()` assumed a unique non-bg/non-divider color was the actor.
#      Walls break that assumption, so actor detection now tries each candidate
#      color and keeps the one that yields an L sprite plus isolated singleton
#      markers in the latest frame.
#    - Original fit logic only cared about in-bounds placement. In the fused task,
#      the destination L pose must fit in-bounds and avoid wall cells.
#    - Original output was a clean background. Here the final frame keeps the static
#      walls, because they are part of the scene state rather than transient marks.
#
# d) Placement legality
#    Generation remains path-first. Future marker cells are future head positions.
#    We reject candidate heads that touch any earlier marker or any earlier sprite
#    cell, and we maintain a global translation interval so every displayed L pose
#    fits inside the frame. Walls are placed only on cells that are background for
#    every marker and every displayed sprite pose, so no overlap is possible in any
#    snapshot or in the final output.
#
# e) Visual observability
#    A human can see both changes directly: the actor is a chunky asymmetric L, and
#    the walls are visibly blocking some row/column alignments that would otherwise
#    be ambiguous. Because the missing corner rotates with motion, 90-degree turns
#    remain legible across examples.
#
# f) Visual tidiness
#    The actor remains a single compact connected block; markers remain isolated
#    single pixels; wall segments are short and sparse. This keeps the actor color
#    decomposable and prevents the maze from turning into an unreadable thicket.
#
# g) Change inferability
#    There is no color remapping. Colors carry only stable roles: background,
#    actor+markers, optional divider, and walls. The fused rule is inferable from
#    geometry alone rather than from hidden color-id ordering.
#
# 9) New main() assumptions and how the generator guarantees them
#    - In the rightmost frame, exactly one actor-colored 4-connected component has
#      size 5 and matches an L rotation; every other actor component is a singleton
#      marker. Marker-isolation checks during rollout guarantee this.
#    - The latest frame always has at least one remaining marker (`gap >= 1`), so
#      actor-color identification can require a sprite-plus-markers decomposition and
#      avoid mistaking a wall component for the actor.
#    - Divider colors appear only as full-height solid columns; walls never do.
#    - During simulation, among all line-of-sight markers whose arrival L pose is
#      legal, exactly one direction is visible at each step; wall placement is built
#      from explicit visibility requirements and verified by the solver oracle.
#    - At least one post-last-frame off-path ray is hidden by a wall, and ignoring
#      wall occlusion would change or invalidate the solve. The generator measures
#      this and rejects degenerate cases.


TEMPLATES: dict[str, dict[int, list[tuple[int, int]]]] = {
    "ell5": {
        0: [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)],  # Up
        1: [(0, 0), (0, -1), (1, -1), (0, -2), (1, -2)],  # Right
        2: [(0, 0), (-1, 0), (-1, -1), (-2, 0), (-2, -1)],  # Down
        3: [(0, 0), (0, 1), (-1, 1), (0, 2), (-1, 2)],  # Left
    },
}
TEMPLATE_NAME = "ell5"
SPRITE_SIZE = 5


def _find_divider_colors(g: np.ndarray, bg: int) -> set[int]:
    """Divider colors are reserved for full-height solid columns only."""
    h, w = g.shape
    divider_colors: set[int] = set()
    for color in set(map(int, g.ravel())) - {bg}:
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if cols and int(np.count_nonzero(g == color)) == len(cols) * h:
            divider_colors.add(int(color))
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


def _components4(pts: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(pts)
    comps: list[set[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        comp = {seed}
        stack = [seed]
        while stack:
            y, x = stack.pop()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (ny, nx) in remaining:
                    remaining.remove((ny, nx))
                    comp.add((ny, nx))
                    stack.append((ny, nx))
        comps.append(comp)
    return comps


def _sprite_cells(head: tuple[int, int], d: int) -> set[tuple[int, int]]:
    hy, hx = head
    return {(hy + dy, hx + dx) for dy, dx in TEMPLATES[TEMPLATE_NAME][d]}


def _match_component(
    comp: set[tuple[int, int]],
) -> tuple[str, int, tuple[int, int], set[tuple[int, int]]] | None:
    if len(comp) != SPRITE_SIZE:
        return None

    matches: list[tuple[str, int, tuple[int, int], set[tuple[int, int]]]] = []
    for head in comp:
        hy, hx = int(head[0]), int(head[1])
        for d, offs in TEMPLATES[TEMPLATE_NAME].items():
            placed = {(hy + dy, hx + dx) for dy, dx in offs}
            if placed == comp:
                matches.append((TEMPLATE_NAME, d, (hy, hx), placed))

    if len(matches) != 1:
        return None
    return matches[0]


def _find_agent(
    frame: np.ndarray, actor: int
) -> tuple[str, int, tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]] | None:
    pts = set(map(tuple, np.argwhere(frame == actor)))
    if not pts:
        return None

    sprite_match = None
    sprite_comp = None
    for comp in _components4(pts):
        matched = _match_component(comp)
        if matched is None:
            if len(comp) != 1:
                return None
            continue
        if sprite_match is not None:
            return None
        sprite_match = matched
        sprite_comp = comp

    if sprite_match is None or sprite_comp is None:
        return None

    tname, d, head, placed = sprite_match
    markers = pts - sprite_comp
    return tname, d, head, placed, markers


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
        found = _find_agent(frame, int(color))
        if found is None:
            continue
        _tname, _d, _head, placed, markers = found
        if not markers or not _markers_are_isolated(markers):
            continue
        score = (len(placed), len(markers))
        if best is None or score > best[0]:
            best = (score, int(color))
    assert best is not None, "Could not identify actor color in the latest frame"
    return best[1]


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


def _can_place_sprite(
    head: tuple[int, int],
    d: int,
    h: int,
    w: int,
    wall_mask: np.ndarray,
    respect_walls: bool,
) -> bool:
    hy, hx = head
    for oy, ox in TEMPLATES[TEMPLATE_NAME][d]:
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
    h: int,
    w: int,
    wall_mask: np.ndarray,
    respect_walls: bool,
) -> tuple[int, tuple[int, int]] | None:
    reachable: list[tuple[int, int, tuple[int, int]]] = []
    for dd in range(4):
        for dist, pt in _ray_candidates(head, dd, markers, wall_mask, respect_walls):
            if _can_place_sprite(pt, dd, h, w, wall_mask, respect_walls):
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

        choice = _choose_target(head, d, markers, h, w, wall_mask, respect_walls)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        nd, head = choice
        d = nd
        agent = _sprite_cells(head, d)
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
    _tname, d, head = _simulate_final_pose(
        frame, actor, wall_colors, respect_walls=respect_walls
    )
    out = np.where(wall_mask, frame, bg).astype(int)
    for y, x in _sprite_cells(head, d):
        out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance an L-shaped sprite through a walled frame and return the final pose.

    Input structure:
    - The input is one or more same-height frames concatenated left-to-right, with
      optional divider columns whose color is reserved for full-height separators.
    - Inside each frame, one color is shared by a connected 5-pixel L sprite and by
      isolated 1-pixel markers, while a different color draws static straight walls.

    Rule:
    - Read only the rightmost frame as the current state.
    - From the sprite head, scan each cardinal ray until the first wall or the frame
      edge; only markers seen before that stop are candidates.
    - Exactly one direction has any legal visible target, and the nearest such marker
      becomes the next head position.
    - After each landing, consume that marker, rotate the L so its opening matches the
      travel direction, and continue until no markers remain. The output keeps walls.

    Key insight: the actor is the unique 5-cell L component, and wall occlusion turns
    raw row/column alignment into a line-of-sight constraint that disambiguates turns.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    divider_colors = _find_divider_colors(g, bg)
    colors = set(map(int, g.ravel())) - {bg} - divider_colors
    start = frames[-1]
    actor = _find_actor_color(start, colors)
    wall_colors = colors - {actor}

    return _simulate_to_end(start, actor, wall_colors, respect_walls=True)


# --------------- generator (L-sprite + wall-aware variant) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (1, 5)  # 1..4 frames
FRAME_MAX = 16
H_FLOOR, H_CAP = 9, 15
W_FLOOR, W_CAP = 8, 15

# --- CFG: path complexity ---
STEPS_RANGE = (4, 13)  # 4..12 inclusive
STEP_DIST_RANGE = (2, 8)  # 2..7 inclusive
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

# --- CFG: wall complexity ---
WALL_SEG_LEN_MAX = 4
WALL_SEG_TARGET_RANGE = (1, 4)  # 1..3 extra effective wall segments
MAX_WALL_ATTEMPTS = 80
MAX_REQUIREMENTS = 7

# --- CFG: retry budget ---
MAX_ATTEMPTS = 1500
_TARGETED_FRAME_ATTEMPTS = {4: 420, 3: 260, 2: 180, 1: 120}

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(1, 5)) for n in range(*STEPS_RANGE)}
_PREFERRED_STEPS_BY_FRAMES = {
    1: [11, 10, 9, 8, 7, 6, 5, 4, 12],
    2: [10, 8, 6, 12, 4],
    3: [9, 12, 6],
    4: [12],
}
_FRAME_TARGET_CYCLE = [4, 3, 2, 1]
_generate_call_count = 0


def _template_bounds(d: int) -> tuple[int, int, int, int]:
    offs = TEMPLATES[TEMPLATE_NAME][d]
    rs = [dr for dr, _ in offs]
    cs = [dc for _, dc in offs]
    return min(rs), max(rs), min(cs), max(cs)


def _template_patch(d: int, actor: int, bg: int) -> tuple[np.ndarray, int, int]:
    r0, r1, c0, c1 = _template_bounds(d)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for dr, dc in TEMPLATES[TEMPLATE_NAME][d]:
        patch[dr - r0, dc - c0] = actor
    return patch, -r0, -c0


def _shift_interval(
    placements: list[tuple[tuple[int, int], int]], h: int, w: int
) -> tuple[int, int, int, int] | None:
    tr_lo, tr_hi = -10**9, 10**9
    tc_lo, tc_hi = -10**9, 10**9
    for (hr, hc), d in placements:
        r0, r1, c0, c1 = _template_bounds(d)
        tr_lo = max(tr_lo, -hr - r0)
        tr_hi = min(tr_hi, h - 1 - hr - r1)
        tc_lo = max(tc_lo, -hc - c0)
        tc_hi = min(tc_hi, w - 1 - hc - c1)
        if tr_lo > tr_hi or tc_lo > tc_hi:
            return None
    return tr_lo, tr_hi, tc_lo, tc_hi


def _sample_params(
    rng: np.random.Generator,
    forced_n_frames: int | None = None,
) -> tuple[int, int, int, int, int, int, int, int | None, int]:
    n_frames = (
        int(forced_n_frames)
        if forced_n_frames is not None
        else int(rng.integers(*N_FRAMES_RANGE))
    )
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0]
    preferred = [
        n for n in _PREFERRED_STEPS_BY_FRAMES.get(n_frames, []) if n in multiples
    ]
    if preferred:
        multiples = preferred
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
    return n_frames, n_steps, gap, h, w, bg, actor, divider, wall


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
) -> bool:
    """
    Allow off-path rays so walls have something to hide later.
    Still forbid anything that would pre-empt the intended next marker, and keep
    markers visually isolated from every displayed L pose.
    """
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, state_dirs[j])
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
    rng: np.random.Generator, step_dirs: list[int], h: int, w: int
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
            if not _marker_compatible_with_existing(nxt, heads, state_dirs, step_dirs):
                continue
            trial_fit = fit_placements + [(nxt, next_dir)]
            if _shift_interval(trial_fit, h, w) is None:
                continue
            chosen = nxt
            break

        if chosen is None:
            return None

        heads.append(chosen)
        state_dirs.append(next_dir)
        fit_placements.append((chosen, next_dir))

    return heads, state_dirs, fit_placements


def _visibility_requirements(
    heads: list[tuple[int, int]], step_dirs: list[int]
) -> list[tuple[int, int, int]]:
    """
    For every state i, gather each off-path ray containing any future marker.
    Requirement tuple = (state_index, blocked_direction, nearest_future_index_on_ray).
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
) -> set[tuple[int, int]]:
    occ = set(heads)
    for head, d in placements:
        occ |= _sprite_cells(head, d)
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
    _length, neg_used, pos_used = top[int(rng.integers(0, len(top)))]

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
    reqs: list[tuple[int, int, int]],
    last_idx: int,
) -> tuple[set[tuple[int, int]], list[list[tuple[int, int]]]] | None:
    occupied = _occupied_cells(heads, placements)
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

            scored = sorted(
                ((_dist(pos, heads[j]), pos) for pos in cross_positions),
                reverse=True,
            )
            picks = [pos for _score, pos in scored[:6]]
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
    markers: list[tuple[int, int]],
    walls: set[tuple[int, int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    for wy, wx in walls:
        frame[wy, wx] = wall
    patch, ar, ac = _template_patch(d, actor, bg)
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


def _build_case(
    rng: np.random.Generator,
    forced_n_frames: int | None = None,
) -> dict | None:
    n_frames, n_steps, gap, h, w, bg, actor, divider, wall = _sample_params(
        rng, forced_n_frames=forced_n_frames
    )
    step_dirs = _make_step_dirs(rng, n_steps)

    rollout = _rollout_path(rng, step_dirs, h, w)
    if rollout is None:
        return None
    heads, state_dirs, fit_placements = rollout

    ok, turns, dists = _validate_path(step_dirs, heads)
    if not ok:
        return None

    reqs = _visibility_requirements(heads, step_dirs)
    if not reqs or len(reqs) > MAX_REQUIREMENTS:
        return None

    shift = _shift_interval(fit_placements, h, w)
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

    wall_data = _place_walls(rng, h, w, heads, placements, reqs, last_idx)
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
        frames.append(_render_frame(h, w, bg, actor, wall, head, d, heads[si + 1 :], walls))

    fhead, fdir = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, wall, fhead, fdir, [], walls)
    inp = _assemble_input(frames, bg, divider)

    start = frames[-1]
    actor_guess = _find_actor_color(start, set(map(int, start.ravel())) - {bg})
    if actor_guess != actor:
        return None

    found = _find_agent(start, actor)
    if found is None:
        return None
    f_tname, f_dir, f_head, f_agent, f_markers = found
    expected_agent = _sprite_cells(placements[last_idx][0], placements[last_idx][1])
    if (f_tname, f_dir, f_head) != (TEMPLATE_NAME, placements[last_idx][1], placements[last_idx][0]):
        return None
    if f_agent != expected_agent:
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
            "template": TEMPLATE_NAME,
            "sprite_pixels": SPRITE_SIZE,
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
    global _generate_call_count
    rng = np.random.default_rng()
    cycle_idx = _generate_call_count % len(_FRAME_TARGET_CYCLE)
    _generate_call_count += 1
    target_order = _FRAME_TARGET_CYCLE[cycle_idx:] + _FRAME_TARGET_CYCLE[:cycle_idx]

    for target in target_order:
        for _ in range(_TARGETED_FRAME_ATTEMPTS[target]):
            try:
                case = _build_case(rng, forced_n_frames=target)
            except Exception:
                case = None
            if case is not None:
                return case["input"]

    remaining_attempts = MAX_ATTEMPTS - sum(
        _TARGETED_FRAME_ATTEMPTS[target] for target in target_order
    )
    for _ in range(max(0, remaining_attempts)):
        try:
            case = _build_case(rng)
        except Exception:
            case = None
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
