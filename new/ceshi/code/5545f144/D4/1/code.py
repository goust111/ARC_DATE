# Variant of 5545f144
# Dimension: D4 — 路径规则：移动方向选择 (path)
# Change: 每步恰好一个基数方向的半线上有可达标记，精灵移向该半线上最近的标记；不允许180度反转，只允许前进或90度转弯。... → 优先直行、其次右转

# Design analysis for the D4 variant
# ==================================
# Reference-task note:
# - The prompt cites `e87109e9` as the inspiration for turning rules.
# - Within the explicitly permitted resources for this task, its JSON examples are
#   present but no solver `.py` was available to read, so the implementation below
#   uses the stated "turning_rules" idea as the baseline and keeps the rest of
#   `5545f144` unchanged.
#
# a) Concrete behavior and trace
# - The visual vocabulary stays the same: a single-color triangular sprite plus
#   same-color single-pixel markers inside each frame, with optional divider
#   columns between horizontally concatenated frames.
# - What changes is the local movement decision. In the original task, from the
#   current head there had to be markers on exactly one visible half-line
#   (same row/column, one side). In this variant, markers may appear on multiple
#   visible half-lines at once, but the facing direction orders them:
#   forward > right > left; reverse is still disallowed.
# - Example:
#   current head=(5,5), facing Up.
#   markers at (2,5) [forward], (5,9) [right], (5,2) [left].
#   Old rule: invalid/ambiguous because multiple directions are occupied.
#   New rule: move to (2,5), because forward outranks right and left.
#   The unchosen markers stay on the board and may be consumed later after the
#   sprite reaches a position from which they become the highest-priority option.
#
# b) Complexity calibration
# - If the generator kept producing unique-direction states, the variant would
#   collapse back into the original behavior. To prevent that, accepted samples
#   must contain at least one decision in the suffix represented by the last
#   input frame where >=2 visible directions are occupied and the chosen step is
#   determined by the priority rule.
# - The sprite templates remain varied and asymmetric enough to keep direction
#   recognition easy. No extra symbols or colors are introduced.
#
# c) Hidden assumptions in the original solver
# - `_choose_target()` assumed that all reachable markers belong to exactly one
#   direction and asserted that uniqueness. This no longer holds; it must instead
#   group reachable markers by relative direction and scan them in priority order.
# - `_marker_compatible_with_existing()` in the generator previously rejected any
#   future marker aligned with a past head in a different direction. That was safe
#   only because uniqueness was mandatory. In the variant, a lower-priority future
#   marker may coexist with the chosen higher-priority one, so compatibility must
#   compare relative priority rather than demand uniqueness.
# - The no-reverse assumption remains valid and is preserved in both solver and
#   generator.
#
# d) Placement legality
# - The generator still builds a full path first, then renders snapshots from it.
# - Every actual landing position must fit the sprite before and after the turn,
#   markers remain isolated from all stamped sprite cells of earlier states, and
#   the detector `_find_agent()` is oracle-checked on the last input frame.
# - A final end-to-end oracle check `main(input) == output` remains mandatory.
#
# e) Visual observability
# - The new rule is human-visible because some last-frame states now show markers
#   simultaneously in front of the triangle and on one side. The output proves the
#   sprite first continues/turns according to the facing-based priority chain
#   rather than any "only one legal direction exists" assumption.
#
# f) Visual cleanliness
# - The original spacing discipline is retained: markers are kept away from sprite
#   bodies so the actor detector is not confused and the frame stays legible.
#
# g) Color inference
# - No new color logic is introduced. Background, actor, and divider colors keep
#   their original roles, so the variant remains inferable from geometry alone.
#
# New main()-level assumptions and how the generator guarantees them
# ---------------------------------------------------------------
# - `main()` now assumes that if visible candidates exist, at least one of the
#   priority directions (forward/right/left) contains a reachable marker and that
#   following the priority chain eventually consumes all markers.
# - The generator guarantees this by constructing a path whose step sequence never
#   reverses, by validating the translated path against the same priority chooser
#   used by the solver, and by rejecting any case where the last-frame suffix does
#   not contain a genuine multi-direction priority decision.

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


def _reachable_by_dir(
    head: tuple[int, int],
    markers: set[tuple[int, int]],
    offs: list[tuple[int, int]],
    h: int,
    w: int,
) -> dict[int, list[tuple[int, tuple[int, int]]]]:
    hy, hx = head

    def fits(target: tuple[int, int]) -> bool:
        ty, tx = target
        dy, dx = ty - hy, tx - hx
        for oy, ox in offs:
            ny = hy + oy + dy
            nx = hx + ox + dx
            if not (0 <= ny < h and 0 <= nx < w):
                return False
        return True

    out: dict[int, list[tuple[int, tuple[int, int]]]] = {0: [], 1: [], 2: [], 3: []}
    for y, x in markers:
        if x == hx and y < hy:
            dd = 0
            dist = hy - y
        elif y == hy and x > hx:
            dd = 1
            dist = x - hx
        elif x == hx and y > hy:
            dd = 2
            dist = y - hy
        elif y == hy and x < hx:
            dd = 3
            dist = hx - x
        else:
            continue
        if fits((y, x)):
            out[dd].append((dist, (y, x)))

    for dd in out:
        out[dd].sort()
    return out


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    offs: list[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    reachable = _reachable_by_dir(head, markers, offs, h, w)
    for dd in _priority_dirs(d):
        if reachable[dd]:
            _, pt = reachable[dd][0]
            return dd, pt
    return None


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, _agent, markers = found
    markers = set((int(y), int(x)) for y, x in markers)
    agent = set(_agent)

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        offs = TEMPLATES[tname][d]
        choice = _choose_target(head, d, markers, offs, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        nd, (ty, tx) = choice

        head = (ty, tx)
        d = nd
        offs = TEMPLATES[tname][d]
        agent = {(ty + oy, tx + ox) for oy, ox in offs}

        # Markers are only consumed when the head lands on them.
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance a triangular sprite through the final snapshot, consuming aligned markers.

    Input structure:
    - Several frames may be concatenated left-to-right with solid divider columns;
      the most common color is the background.
    - Each frame contains one triangular arrow sprite plus same-color single-pixel
      markers that remain to be consumed.

    Rule:
    - Read the rightmost frame as the current state.
    - From the sprite head, inspect aligned markers on the forward, right, and
      left half-lines that keep the current sprite footprint inside the grid.
      If multiple directions are occupied, break ties by priority:
      straight first, then right, then left. Within the chosen direction, move
      to the nearest marker.
    - Land on that marker, consume it, rotate the sprite to the travel direction,
      and repeat until no markers remain. Output only the final sprite body.

    Key insight: the arrow tip does more than show where the next unique marker is;
    it defines the tie-breaking order among several simultaneously visible directions.
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


# --------------- generator ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (1, 5)          # rng.integers low/high+1 -> 1..4 frames
FRAME_MAX = 16                   # rng.integers high for H, W -> 8..15 / 6..15
H_FLOOR, H_CAP = 8, 15           # clamps for computed H lower bound
W_FLOOR, W_CAP = 6, 15           # clamps for computed W lower bound

# --- CFG: path complexity ---
STEPS_RANGE = (4, 13)            # candidate total_steps (4..12 inclusive)
STEP_DIST_RANGE = (2, 8)         # per-step manhattan distance choices (2..7 inclusive)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3                 # absolute floor on #turns used during segment sampling
MIN_TURNS_ACCEPT = 2             # post-hoc minimum for accepted paths
MIN_DIR_DIVERSITY = 3            # >= 3 of {U,R,D,L} must appear
MIN_DIST_VARIETY = 2             # >= 2 distinct step distances

# --- CFG: variant observability ---
MIN_AMBIGUOUS_TAIL_STEPS = 1

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 1500

# Precomputed: how many divisors in 1..4 each step count has (used to reweight n_steps
# so that step counts compatible with fewer frame counts aren't over-represented).
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


def _sprite_cells(head: tuple[int, int], tname: str, d: int) -> set[tuple[int, int]]:
    hr, hc = head
    return {(hr + dr, hc + dc) for dr, dc in TEMPLATES[tname][d]}


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


def _touching(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _shift_interval(
    placements: list[tuple[tuple[int, int], int]], h: int, w: int, tname: str
) -> tuple[int, int, int, int] | None:
    """Interval of (tr, tc) translations that keep every placement's template in-grid."""
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
) -> tuple[int, int, int, int, int, int, int, int | None, str]:
    """Sample (n_frames, n_steps, gap, h, w, bg, actor, divider, tname)."""
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0]
    weights = np.array([(n ** 3) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = rng.permutation(10)
    bg, actor = int(palette[0]), int(palette[1])
    divider = int(palette[2]) if n_frames > 1 else None

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames, n_steps, gap, h, w, bg, actor, divider, tname


def _make_step_dirs(rng: np.random.Generator, n_steps: int) -> list[int]:
    """Produce a length-n sequence of step directions with turn ratio in [LO, HI]."""
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
    """Validate candidate marker against all previously-placed heads.

    Preserves sprite/marker spacing and enforces that any aligned future marker is
    either lower priority than the chosen next step from that past head, or on the
    same direction but farther away than the actual next marker.
    """
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, state_dirs[j])
        if any(_touching(nxt, cell) for cell in sprite):
            return False

        rel_dir = _direction(old_head, nxt)
        if rel_dir is None:
            continue

        chosen_dir = step_dirs[j]
        chosen_rank = _priority_rank(state_dirs[j], chosen_dir)
        rel_rank = _priority_rank(state_dirs[j], rel_dir)
        if rel_rank is None:
            continue
        if chosen_rank is None:
            return False
        if rel_rank < chosen_rank:
            return False
        if j < len(heads) - 1 and rel_rank == chosen_rank:
            if _dist(old_head, nxt) <= _dist(old_head, heads[j + 1]):
                return False
    return True


def _rollout_path(
    rng: np.random.Generator, tname: str, step_dirs: list[int], h: int, w: int
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]] | None:
    """Greedy path rollout. Returns (heads, state_dirs, fit_placements) or None on stuck step."""
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


def _priority_stats(
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
    h: int,
    w: int,
    start_idx: int = 0,
) -> dict[str, int]:
    stats = {
        "ambiguous_steps": 0,
        "straight_priority_wins": 0,
        "right_over_left_wins": 0,
    }
    n_steps = len(heads) - 1
    for j in range(start_idx, n_steps):
        head = heads[j]
        d = state_dirs[j]
        actual_dir = _direction(heads[j], heads[j + 1])
        if actual_dir is None:
            continue
        reachable = _reachable_by_dir(head, set(heads[j + 1 :]), TEMPLATES[tname][d], h, w)
        visible = [dd for dd in _priority_dirs(d) if reachable[dd]]
        if len(visible) >= 2:
            stats["ambiguous_steps"] += 1
            if actual_dir == d:
                stats["straight_priority_wins"] += 1
            elif actual_dir == (d + 1) % 4 and (d + 3) % 4 in visible:
                stats["right_over_left_wins"] += 1
    return stats


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, gap, h, w, bg, actor, divider, tname = _sample_params(rng)
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
    total_stats = _priority_stats(heads, state_dirs, tname, h, w, 0)
    tail_stats = _priority_stats(heads, state_dirs, tname, h, w, last_idx)
    if tail_stats["ambiguous_steps"] < MIN_AMBIGUOUS_TAIL_STEPS:
        return None
    if tail_stats["straight_priority_wins"] + tail_stats["right_over_left_wins"] == 0:
        return None

    frames = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, heads[si + 1 :]))
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
            "turns": turns,
            "turn_ratio": turns / n_steps,
            "direction_diversity": len(set(step_dirs)),
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "step_distances": dists,
            "count_in_first_frame": n_steps,
            "count_in_last_input_frame": gap,
            "ambiguous_steps_total": total_stats["ambiguous_steps"],
            "ambiguous_steps_tail": tail_stats["ambiguous_steps"],
            "straight_priority_wins_tail": tail_stats["straight_priority_wins"],
            "right_over_left_wins_tail": tail_stats["right_over_left_wins"],
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
