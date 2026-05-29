# Combo variant of 5545f144 (index 1)
# combo_spec: D4/1+D14/0
# Name: 优先级方向+多方向最近

# Design analysis for the combo variant
# ====================================
# This file starts from the original 5545f144 task: a monochrome triangular
# sprite moves from marker to marker across the rightmost snapshot and the
# output is the final sprite pose. The combo keeps the visual vocabulary and
# frame cadence, but fuses two changes:
# - D14/0 relaxes the old "exactly one occupied half-line" constraint and
#   chooses the globally nearest reachable marker across the legal directions.
# - D4/1 supplies the facing-relative tiebreak among equally near legal moves:
#   forward, then right, then left.
#
# a) Concrete behavior and trace
# - Original task:
#   head=(5,5), facing Up, markers at (2,5) and (5,7).
#   This state was illegal because both Up and Right were populated.
# - D14-only task:
#   the same state becomes legal and the sprite would choose the globally
#   nearest aligned marker.
# - Combo task implemented here:
#   the state is legal and all forward/right/left candidates are compared by
#   distance first. If one side is strictly closer, it wins even against a
#   higher-priority direction. If two or three legal directions tie at the same
#   minimal distance, the triangle's facing breaks the tie by forward > right >
#   left.
# - Difference:
#   the original uniqueness constraint disappears; D14 now controls the primary
#   choice order, while D4 still matters on equal-distance ambiguities.
#
# b) Complexity calibration
# - If generated cases only ever expose one occupied direction, the combo would
#   collapse back to the original task.
# - If ambiguous states never contain equal-distance ties, the combo would
#   visually collapse to pure D14.
# - If ambiguous states always resolve by priority despite unequal distances,
#   the combo would collapse to pure D4.
# - To prevent both degeneracies, accepted cases must show both branches:
#   a visible multi-direction state where a closer lower-priority marker wins,
#   and another visible state where equal nearest distances are broken by the
#   facing priority chain.
#
# c) Hidden assumption audit
# - Original `_choose_target()` assumed all reachable markers sat on exactly one
#   cardinal half-line. That assumption is invalid and is replaced by a
#   reachable-candidate collection plus lexicographic choice by
#   (distance, priority rank).
# - Original generator logic rejected any future marker aligned to a different
#   direction from an old head. That encoded uniqueness and must be relaxed.
#   The new legality test only forbids future markers that would beat the
#   intended next move under the fused comparator (shorter distance, or equal
#   distance with better priority rank).
# - The original no-180 rule still matters. Reverse-aligned markers may exist,
#   but the combo solver never chooses them; the generator must guarantee that a
#   non-reverse priority direction is always available until all markers vanish.
#
# d) Placement legality
# - Every rendered sprite pose must fit the frame.
# - Every candidate head used by the solver must also fit both as a translated
#   copy of the current orientation and as the landing orientation; this keeps
#   solver and generator reachability identical.
# - Markers remain isolated single pixels and are kept away from stamped sprite
#   cells so `_find_agent()` still reliably detects the actor in the last frame.
#
# e) Visual observability
# - The crucial observable cue is geometric, not numeric: several aligned
#   options are visible from the head at once, and different samples show both
#   "closer side beats farther straight" and "equal-distance tie follows the
#   facing priority chain."
# - The generator prefers frame selections where those two branches are visible
#   directly in the shown snapshots, and requires the rightmost suffix to
#   contain at least one genuine multi-direction combo decision.
#
# f) Visual cleanliness
# - The actor and markers still share one color, so spacing remains important.
#   Candidate markers are rejected if they touch any earlier sprite body, which
#   keeps the triangle silhouette legible and prevents accidental merged shapes.
#
# g) Inference burden
# - No hidden use of color ids or arbitrary ordering is introduced.
# - The only new latent variables are a nearest-first distance comparison plus a
#   small three-way priority tiebreak tied directly to the triangle tip. Both
#   are structural and can be inferred from repeated ambiguous states rather
#   than from arbitrary symbols.
#
# New main()-level assumptions and generator guarantees
# -----------------------------------------------------
# - `main()` assumes the head orientation is recoverable from the last frame and
#   that every unfinished state has at least one reachable marker on forward,
#   right, or left.
# - The generator enforces those assumptions by validating the translated full
#   path with the same nearest-then-priority chooser used in the solver and by
#   rejecting any case whose visible snapshots or rightmost-frame suffix
#   collapse back to the original task or to either single variant.

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


def _priority_dirs(d: int) -> list[int]:
    return [d, (d + 1) % 4, (d + 3) % 4]


def _priority_rank(facing: int, move_dir: int) -> int | None:
    order = _priority_dirs(facing)
    return order.index(move_dir) if move_dir in order else None


def _choice_key(facing: int, move_dir: int, dist: int) -> tuple[int, int] | None:
    rank = _priority_rank(facing, move_dir)
    if rank is None:
        return None
    return dist, rank


# Templates are defined as offsets relative to the head pixel at (0, 0).
TEMPLATES: dict[str, dict[int, list[tuple[int, int]]]] = {
    # Hollow triangle (3 pixels)
    "hollow3": {
        0: [(0, 0), (1, -1), (1, 1)],  # Up
        1: [(0, 0), (-1, -1), (1, -1)],  # Right
        2: [(0, 0), (-1, -1), (-1, 1)],  # Down
        3: [(0, 0), (-1, 1), (1, 1)],  # Left
    },
    # Filled triangle (4 pixels: tip + base of 3)
    "filled4": {
        0: [(0, 0), (1, -1), (1, 0), (1, 1)],
        1: [(0, 0), (-1, -1), (0, -1), (1, -1)],
        2: [(0, 0), (-1, -1), (-1, 0), (-1, 1)],
        3: [(0, 0), (-1, 1), (0, 1), (1, 1)],
    },
    # Arrow-like triangle (4 pixels: tip + stem + two base corners)
    "stem4": {
        0: [(0, 0), (1, 0), (2, -1), (2, 1)],
        1: [(0, 0), (0, -1), (-1, -2), (1, -2)],
        2: [(0, 0), (-1, 0), (-2, -1), (-2, 1)],
        3: [(0, 0), (0, 1), (-1, 2), (1, 2)],
    },
    # 3-row triangle (6 pixels): tip + filled middle row + base corners
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


def _direction(a: tuple[int, int], b: tuple[int, int]) -> int | None:
    """Cardinal direction from a to b, or None if not axis-aligned."""
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


def _fits_head(head: tuple[int, int], tname: str, d: int, h: int, w: int) -> bool:
    hy, hx = head
    for oy, ox in TEMPLATES[tname][d]:
        ny = hy + oy
        nx = hx + ox
        if not (0 <= ny < h and 0 <= nx < w):
            return False
    return True


def _reachable_by_dir(
    head: tuple[int, int],
    cur_d: int,
    markers: set[tuple[int, int]],
    tname: str,
    h: int,
    w: int,
) -> dict[int, list[tuple[int, tuple[int, int]]]]:
    """
    Group aligned reachable markers by direction.

    A target is reachable only if the current orientation translated to that head
    and the landing orientation both fit inside the grid.
    """
    out: dict[int, list[tuple[int, tuple[int, int]]]] = {0: [], 1: [], 2: [], 3: []}
    for pt in markers:
        pt = (int(pt[0]), int(pt[1]))
        dd = _direction(head, pt)
        if dd is None:
            continue
        if not _fits_head(pt, tname, cur_d, h, w):
            continue
        if not _fits_head(pt, tname, dd, h, w):
            continue
        out[dd].append((_dist(head, pt), pt))
    for dd in out:
        out[dd].sort()
    return out


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    tname: str,
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    reachable = _reachable_by_dir(head, d, markers, tname, h, w)
    candidates: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
    for dd in _priority_dirs(d):
        if not reachable[dd]:
            continue
        dist, pt = reachable[dd][0]
        key = _choice_key(d, dd, dist)
        if key is None:
            continue
        candidates.append((key, dd, pt))
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1], item[2][0], item[2][1]))
        _, dd, pt = candidates[0]
        return dd, pt
    return None


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, agent, markers = found
    markers = set((int(y), int(x)) for y, x in markers)
    agent = set((int(y), int(x)) for y, x in agent)

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        choice = _choose_target(head, d, markers, tname, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        nd, (ty, tx) = choice

        head = (ty, tx)
        d = nd
        agent = {(ty + oy, tx + ox) for oy, ox in TEMPLATES[tname][d]}
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue the triangle sprite from the latest snapshot until every marker is eaten.

    Input structure:
    - The input is one to four equal-sized frames concatenated left-to-right,
      optionally separated by solid divider columns in a non-background color.
    - Inside each frame, one non-background actor color is shared by both the
      triangle sprite and all remaining one-pixel markers.

    Rule:
    - Read only the rightmost frame as the current state.
    - From the sprite head, inspect aligned markers on the forward, right, and
      left half-lines that keep the sprite valid in-bounds.
    - Choose the globally nearest reachable marker among those legal
      directions. If several legal markers tie at the same minimum distance,
      break the tie by fixed facing-relative priority:
      straight first, then right, then left.
    - Move to that marker, consume it, rotate to the travel direction, and
      repeat until no markers remain. Return a clean frame containing only the
      final sprite.

    Key insight: the triangle tip is not just a locator for the head; it defines
    a three-way direction ranking that only resolves equal-distance ties; a
    strictly closer side marker still wins over a farther straight option.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    # Actor color is the non-bg, non-divider color.
    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    # Use the latest frame (rightmost) as the starting state.
    start = frames[-1]
    return _simulate_to_end(start, actor)


# --------------- generator (combo: nearest across directions, priority on ties) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_CHOICES = (1, 2, 3, 4)
FRAME_MAX = 16
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 6, 15

# --- CFG: path complexity ---
STEPS_RANGE = (4, 13)
STEP_DIST_RANGE = (2, 8)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

# --- CFG: variant observability ---
MIN_GAP_FOR_VARIANT = 2
MIN_SUFFIX_MULTI_PRIORITY_STEPS = 1
MIN_SUFFIX_COMBO_STEPS = 1
MIN_VISIBLE_NEAREST_OVER_PRIORITY = 1
MIN_VISIBLE_PRIORITY_TIEBREAK = 1
MIN_VISIBLE_MULTI_SNAPSHOTS = 1

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 5000

_FRAME_AVAILABILITY = {
    n: sum(n % f == 0 and n // f >= MIN_GAP_FOR_VARIANT for f in N_FRAMES_CHOICES)
    for n in range(*STEPS_RANGE)
}


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


def _touching(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _shift_interval(
    placements: list[tuple[tuple[int, int], int]], h: int, w: int, tname: str
) -> tuple[int, int, int, int] | None:
    """Interval of translations that keep every recorded placement in-grid."""
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
) -> tuple[int, int, int, int, str]:
    """Sample (n_steps, h, w, bg, actor, tname)."""
    multiples = [n for n in range(*STEPS_RANGE) if _FRAME_AVAILABILITY[n] > 0]
    weights = np.array(
        [(n**3) / (_FRAME_AVAILABILITY[n] ** 2) for n in multiples],
        dtype=float,
    )
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = rng.permutation(10)
    bg, actor = int(palette[0]), int(palette[1])

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_steps, h, w, bg, actor, tname


def _make_step_dirs(rng: np.random.Generator, n_steps: int) -> list[int]:
    """Produce a length-n step-direction sequence with many turns but no reversals."""
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
    Validate a candidate marker against all previously-placed heads.

    Lower-priority aligned markers are allowed, but only if they lose under the
    fused comparator (shorter distance first, then facing-relative priority).
    """
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, state_dirs[j])
        if any(_touching(nxt, cell) for cell in sprite):
            return False

        rel_dir = _direction(old_head, nxt)
        if rel_dir is None:
            continue

        chosen_dir = step_dirs[j]
        chosen_key = _choice_key(
            state_dirs[j],
            chosen_dir,
            _dist(old_head, heads[j + 1]) if j < len(heads) - 1 else _dist(old_head, nxt),
        )
        rel_key = _choice_key(state_dirs[j], rel_dir, _dist(old_head, nxt))
        if chosen_key is None:
            return False
        if rel_key is None:
            continue
        if rel_key < chosen_key:
            return False
        if j < len(heads) - 1 and rel_key == chosen_key:
            return False
    return True


def _rollout_path(
    rng: np.random.Generator, tname: str, step_dirs: list[int], h: int, w: int
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]] | None:
    """
    Greedy path rollout.

    Returns (heads, state_dirs, fit_placements), where fit_placements includes
    both actual states and translated-current-orientation checks used by the
    solver's reachability test.
    """
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


def _validate_path(step_dirs: list[int], heads: list[tuple[int, int]]) -> tuple[bool, int, list[int]]:
    """Check post-hoc path-quality constraints. Returns (ok, turns, step_distances)."""
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


def _trace_priority_profile(
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
    h: int,
    w: int,
) -> list[dict] | None:
    """
    Validate the translated path against the combo rule and collect observability stats.
    """
    n_steps = len(heads) - 1
    profile: list[dict] = []
    for j in range(n_steps):
        head = heads[j]
        facing = state_dirs[j]
        actual_target = heads[j + 1]
        actual_dir = _direction(head, actual_target)
        if actual_dir is None:
            return None

        reachable = _reachable_by_dir(head, facing, set(heads[j + 1 :]), tname, h, w)
        priority_dirs = _priority_dirs(facing)
        visible_priority = [dd for dd in priority_dirs if reachable[dd]]
        if not visible_priority:
            return None

        candidates: list[tuple[tuple[int, int], int, int, tuple[int, int]]] = []
        for dd in visible_priority:
            dist, pt = reachable[dd][0]
            key = _choice_key(facing, dd, dist)
            if key is None:
                continue
            candidates.append((key, dd, dist, pt))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[3][0], item[3][1]))
        chosen_key, chosen_dir, chosen_dist, chosen_target = candidates[0]
        if chosen_target != actual_target:
            return None
        if chosen_dir != actual_dir or chosen_dir != state_dirs[j + 1]:
            return None

        multi_priority = len(visible_priority) >= 2
        first_priority_dir = visible_priority[0]
        first_priority_dist = reachable[first_priority_dir][0][0]
        nearest_dist = min(item[2] for item in candidates)
        nearest_dirs = [item[1] for item in candidates if item[2] == nearest_dist]
        nearest_targets = [item[3] for item in candidates if item[2] == nearest_dist]
        nearest_over_priority = multi_priority and chosen_dir != first_priority_dir
        priority_tiebreak = multi_priority and len(nearest_dirs) >= 2
        strict_priority_tiebreak = priority_tiebreak and chosen_dir == first_priority_dir
        combo_active = nearest_over_priority or strict_priority_tiebreak

        profile.append(
            {
                "step": j,
                "chosen_dir": chosen_dir,
                "chosen_target": chosen_target,
                "chosen_dist": chosen_dist,
                "chosen_key": chosen_key,
                "visible_priority_dirs": list(visible_priority),
                "all_visible_dirs": [dd for dd in range(4) if reachable[dd]],
                "multi_priority": multi_priority,
                "combo_active": combo_active,
                "nearest_over_priority": nearest_over_priority,
                "priority_tiebreak": priority_tiebreak,
                "strict_priority_tiebreak": strict_priority_tiebreak,
                "first_priority_dir": first_priority_dir,
                "first_priority_dist": first_priority_dist,
                "nearest_dirs": list(nearest_dirs),
                "nearest_targets": list(nearest_targets),
                "nearest_dist": nearest_dist,
            }
        )
    return profile


def _frame_options_from_profile(
    n_steps: int, profile: list[dict]
) -> list[tuple[int, int, int, int, int, int, int, int]]:
    """Return valid frame choices with visibility metadata for the fused rule."""
    out: list[tuple[int, int, int, int, int, int, int, int]] = []
    for n_frames in N_FRAMES_CHOICES:
        if n_steps % n_frames != 0:
            continue
        gap = n_steps // n_frames
        if gap < MIN_GAP_FOR_VARIANT:
            continue
        last_idx = (n_frames - 1) * gap
        suffix_multi = sum(int(item["multi_priority"]) for item in profile[last_idx:])
        suffix_combo = sum(int(item["combo_active"]) for item in profile[last_idx:])
        visible_multi = sum(int(profile[fi * gap]["multi_priority"]) for fi in range(n_frames))
        visible_nearest_over = sum(
            int(profile[fi * gap]["nearest_over_priority"]) for fi in range(n_frames)
        )
        visible_priority_tiebreak = sum(
            int(profile[fi * gap]["strict_priority_tiebreak"]) for fi in range(n_frames)
        )
        last_frame_multi = int(profile[last_idx]["multi_priority"])
        last_frame_combo = int(profile[last_idx]["combo_active"])
        if suffix_multi < MIN_SUFFIX_MULTI_PRIORITY_STEPS:
            continue
        if suffix_combo < MIN_SUFFIX_COMBO_STEPS:
            continue
        if visible_multi < MIN_VISIBLE_MULTI_SNAPSHOTS:
            continue
        if visible_nearest_over < MIN_VISIBLE_NEAREST_OVER_PRIORITY:
            continue
        if visible_priority_tiebreak < MIN_VISIBLE_PRIORITY_TIEBREAK:
            continue
        if not last_frame_multi or not last_frame_combo:
            continue
        out.append(
            (
                n_frames,
                gap,
                last_idx,
                suffix_multi,
                suffix_combo,
                visible_multi,
                visible_nearest_over,
                visible_priority_tiebreak,
            )
        )
    return out


def _render_frame(
    h: int,
    w: int,
    bg: int,
    actor: int,
    head: tuple[int, int],
    d: int,
    tname: str,
    markers: list[tuple[int, int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
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


def _rotate_head(head: tuple[int, int], turns: int) -> tuple[int, int]:
    r, c = head
    turns %= 4
    for _ in range(turns):
        r, c = c, -r
    return r, c


def _rotate_dir(d: int, turns: int) -> int:
    return (d + turns) % 4


def _combo_path_family(
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]]:
    """
    Build a path family that always exposes both combo branches.

    Visible step 2: equal-distance tie between forward and right; forward wins.
    Visible step 4: a closer lower-priority side marker beats a farther forward marker.
    """
    up1 = int(rng.integers(2, 4))
    up2 = int(rng.integers(2, 4))
    tie_dist = int(rng.integers(3, 5))
    far_right = int(rng.integers(tie_dist + 1, tie_dist + 3))

    heads = [
        (0, 0),
        (-up1, 0),
        (-up1 - up2, 0),
        (-up1 - up2 - tie_dist, 0),
        (-up1 - up2 - tie_dist, tie_dist),
        (-up1 - up2, tie_dist),
        (-up1 - up2, tie_dist + far_right),
        (-up1 - up2 - tie_dist, tie_dist + far_right),
    ]
    state_dirs = [0, 0, 0, 0, 1, 2, 1, 0]

    turns = int(rng.integers(0, 4))
    if turns:
        heads = [_rotate_head(head, turns) for head in heads]
        state_dirs = [_rotate_dir(d, turns) for d in state_dirs]

    fit_placements: list[tuple[tuple[int, int], int]] = [(heads[0], state_dirs[0])]
    for i in range(len(heads) - 1):
        fit_placements.append((heads[i + 1], state_dirs[i]))
        fit_placements.append((heads[i + 1], state_dirs[i + 1]))
    return heads, state_dirs, fit_placements


def _choose_translation_and_frames(
    rng: np.random.Generator,
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    fit_placements: list[tuple[tuple[int, int], int]],
    tname: str,
    h: int,
    w: int,
) -> tuple[list[tuple[int, int]], list[dict], int, int, int] | None:
    shift = _shift_interval(fit_placements, h, w, tname)
    if shift is None:
        return None
    tr_lo, tr_hi, tc_lo, tc_hi = shift

    translations = [
        (int(tr), int(tc))
        for tr in range(tr_lo, tr_hi + 1)
        for tc in range(tc_lo, tc_hi + 1)
    ]
    rng.shuffle(translations)

    for tr, tc in translations:
        shifted_heads = [(r + tr, c + tc) for r, c in heads]
        profile = _trace_priority_profile(shifted_heads, state_dirs, tname, h, w)
        if profile is None:
            continue
        frame_options = _frame_options_from_profile(len(profile), profile)
        if not frame_options:
            continue
        multi_frame_options = [item for item in frame_options if item[0] > 1]
        candidates = multi_frame_options if multi_frame_options else frame_options
        weights = np.array(
            [1.0 + item[3] + 2.0 * item[4] + 0.5 * item[5] + item[6] + item[7] for item in candidates],
            dtype=float,
        )
        weights /= weights.sum()
        pick = int(rng.choice(len(candidates), p=weights))
        n_frames, gap, last_idx, _suffix_multi, _suffix_combo, _visible_multi, _visible_nearest, _visible_tie = candidates[pick]
        return shifted_heads, profile, n_frames, gap, last_idx
    return None


def _build_case(rng: np.random.Generator) -> dict | None:
    _n_steps, _sample_h, _sample_w, bg, actor, tname = _sample_params(rng)
    heads, state_dirs, fit_placements = _combo_path_family(rng)
    n_steps = len(heads) - 1
    step_dirs = [state_dirs[i + 1] for i in range(n_steps)]

    path_h = max(r for r, _ in heads) - min(r for r, _ in heads) + 1
    path_w = max(c for _, c in heads) - min(c for _, c in heads) + 1
    h = int(rng.integers(max(H_FLOOR, path_h + 4), FRAME_MAX + 1))
    w = int(rng.integers(max(W_FLOOR, path_w + 4), FRAME_MAX + 1))

    shift = _shift_interval(fit_placements, h, w, tname)
    if shift is None:
        return None
    tr_lo, tr_hi, tc_lo, tc_hi = shift
    tr = int(rng.integers(tr_lo, tr_hi + 1))
    tc = int(rng.integers(tc_lo, tc_hi + 1))
    heads = [(r + tr, c + tc) for r, c in heads]

    profile = _trace_priority_profile(heads, state_dirs, tname, h, w)
    if profile is None or len(profile) != n_steps:
        return None
    if not profile[2]["strict_priority_tiebreak"]:
        return None
    if not profile[4]["nearest_over_priority"]:
        return None

    n_frames = 3
    frame_indices = [0, 2, 4]
    gap = 2
    last_idx = frame_indices[-1]
    placements = [(heads[i], state_dirs[i]) for i in range(n_steps + 1)]

    palette = [c for c in range(10) if c not in {bg, actor}]
    divider = int(rng.choice(palette))
    frames = [
        _render_frame(h, w, bg, actor, placements[si][0], placements[si][1], tname, heads[si + 1 :])
        for si in frame_indices
    ]

    fhead, fdir = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, fhead, fdir, tname, [])
    inp = _assemble_input(frames, bg, divider)

    found = _find_agent(frames[-1], actor)
    if found is None:
        return None
    f_tname, f_dir, f_head, _f_agent, f_markers = found
    if (f_tname, f_dir, f_head) != (tname, placements[last_idx][1], placements[last_idx][0]):
        return None
    if f_markers != set(heads[last_idx + 1 :]):
        return None
    if not np.array_equal(main(inp), out_frame):
        return None

    suffix_multi_steps = sum(int(item["multi_priority"]) for item in profile[last_idx:])
    suffix_combo_steps = sum(int(item["combo_active"]) for item in profile[last_idx:])
    visible_multi_snapshots = sum(int(profile[si]["multi_priority"]) for si in frame_indices)
    visible_combo_snapshots = sum(int(profile[si]["combo_active"]) for si in frame_indices)
    visible_nearest_over_snapshots = sum(
        int(profile[si]["nearest_over_priority"]) for si in frame_indices
    )
    visible_priority_tiebreak_snapshots = sum(
        int(profile[si]["strict_priority_tiebreak"]) for si in frame_indices
    )

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
            "divider": divider,
            "total_steps": n_steps,
            "gap": gap,
            "turns": sum(step_dirs[i] != step_dirs[i - 1] for i in range(1, n_steps)),
            "turn_ratio": sum(step_dirs[i] != step_dirs[i - 1] for i in range(1, n_steps)) / n_steps,
            "direction_diversity": len(set(step_dirs)),
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "step_distances": [_dist(heads[i], heads[i + 1]) for i in range(n_steps)],
            "count_in_first_frame": n_steps,
            "count_in_last_input_frame": n_steps - last_idx,
            "multi_priority_steps_total": sum(int(item["multi_priority"]) for item in profile),
            "multi_priority_steps_suffix": suffix_multi_steps,
            "combo_steps_total": sum(int(item["combo_active"]) for item in profile),
            "combo_steps_suffix": suffix_combo_steps,
            "nearest_over_priority_total": sum(int(item["nearest_over_priority"]) for item in profile),
            "strict_priority_tiebreak_total": sum(
                int(item["strict_priority_tiebreak"]) for item in profile
            ),
            "visible_multi_snapshots": visible_multi_snapshots,
            "visible_combo_snapshots": visible_combo_snapshots,
            "visible_nearest_over_priority_snapshots": visible_nearest_over_snapshots,
            "visible_priority_tiebreak_snapshots": visible_priority_tiebreak_snapshots,
            "last_frame_multi_priority": int(profile[last_idx]["multi_priority"]),
            "last_frame_combo_active": int(profile[last_idx]["combo_active"]),
            "last_frame_visible_priority_dirs": list(profile[last_idx]["visible_priority_dirs"]),
            "last_frame_nearest_over_priority": int(profile[last_idx]["nearest_over_priority"]),
            "last_frame_priority_tiebreak": int(profile[last_idx]["strict_priority_tiebreak"]),
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
