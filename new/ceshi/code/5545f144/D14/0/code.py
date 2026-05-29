# Variant of 5545f144
# Dimension: D14 — 唯一方向约束 (match_cond)
# Change: 每步恰好一个基数方向半线上有可达标记（CONSTRAINT 1），确保移动方向无歧义。... → 允许多方向有标记但取最近

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left

# ---------------------------------------------------------------------------
# Design analysis for the D14 variant
#
# variant_spec:
#   changed_dimension:
#     original: from each head position, reachable markers could appear on only
#       one cardinal half-line; that unique direction removed all ambiguity.
#     variant: markers may appear on multiple half-lines at once. The chosen
#       move is now the globally nearest reachable marker among all aligned
#       markers. To keep the rule visible and deterministic, the generator
#       enforces a unique minimum distance at every step.
#   concrete_grid_behavior:
#     sprite: unchanged monochrome triangle template whose tip shows current
#       facing. Markers remain isolated single pixels in the same color.
#     frames: unchanged horizontal snapshot strip with optional divider columns.
#     visual cue: at least one visible snapshot contains a head aligned with
#       markers in multiple directions. Humans can compare distances from the
#       tip and see that the shorter one should be consumed first.
#   traced_example:
#     original_logic:
#       Suppose the head is at (5,5), one marker is at (5,8), another at
#       (2,5). Under the original task this state was illegal because both
#       right and up were populated.
#     variant_logic:
#       The same state is now legal if, say, the up marker is distance 3 and
#       the right marker is distance 4. The sprite moves up first, consumes the
#       closer marker, then later may come back to consume the farther one.
#     difference:
#       The old `_choose_target` asserted that only one direction existed.
#       The new solver sorts all aligned reachable markers by distance and uses
#       the unique closest one.
#   complexity_calibration:
#     preserved:
#       - same four triangle templates
#       - same frame cadence / divider structure
#       - same no-actual-180-degree-reversal rule
#       - same path complexity floors on turns, direction diversity, and varied
#         distances
#     relaxed_only_where_needed:
#       The generator no longer forbids future markers from lying on other
#       half-lines of an earlier head. It only forbids them from being as near
#       as or nearer than that step's true next marker, because that would
#       change the nearest-choice result.
#   hidden_assumption_audit:
#     `_choose_target` previously depended on `len(dirs_with_candidates) == 1`.
#       This is invalid now and must be replaced by a unique-nearest check.
#     `_marker_compatible_with_existing` previously rejected any candidate whose
#       direction from an older head differed from the older step direction.
#       That rejection encoded the old uniqueness rule and must be relaxed.
#     frame selection matters:
#       because `main()` solves from the rightmost frame only, the generator
#       must ensure that the suffix starting at the last input frame genuinely
#       uses the new rule; otherwise the sample could silently degenerate back
#       to original behavior. Frame count is therefore selected only after the
#       full path is known. Accepted cases must contain at least one visible
#       snapshot with a multi-direction choice, and the suffix from the rightmost
#       frame to the end must also contain at least one such step.
#   placement_legality:
#     markers stay isolated from prior markers and from every stamped sprite
#     body, and all actual sprite states must fit in-grid. Candidate "shortcut"
#     markers considered by the solver are also filtered by in-bounds checks for
#     the landing orientation so the solver and generator agree on reachability.
#   visual_observability:
#     No hidden color-id ordering is used. The distinguishing feature is purely
#     geometric: multiple aligned options are visible, and shorter Manhattan
#     distance along a row/column wins. The generator requires at least one
#     input snapshot to expose such a multi-direction state, and it also
#     guarantees that the rightmost-frame suffix cannot be solved by the old
#     unique-direction rule.
#   neatness:
#     markers remain single pixels with non-touching spacing, so the sprite is
#     still easy to detect even though markers share its color.
# ---------------------------------------------------------------------------


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


def _reachable_targets(
    head: tuple[int, int],
    cur_d: int,
    markers: set[tuple[int, int]],
    tname: str,
    h: int,
    w: int,
) -> list[tuple[int, int, tuple[int, int]]]:
    """All aligned markers whose translated current stamp and landing stamp fit."""
    out: list[tuple[int, int, tuple[int, int]]] = []
    for pt in markers:
        pt = (int(pt[0]), int(pt[1]))
        dd = _direction(head, pt)
        if dd is None:
            continue
        if not _fits_head(pt, tname, cur_d, h, w):
            continue
        if not _fits_head(pt, tname, dd, h, w):
            continue
        out.append((_dist(head, pt), dd, pt))
    out.sort(key=lambda item: (item[0], item[1], item[2][0], item[2][1]))
    return out


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    tname: str,
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    reachable = _reachable_targets(head, d, markers, tname, h, w)
    if not reachable:
        return None

    # The variant allows multiple directions, but the closest marker must be
    # unique so distance alone resolves the next move.
    if len(reachable) >= 2:
        assert reachable[0][0] < reachable[1][0], (
            f"head={head} dir={d}: closest markers tie at distance "
            f"{reachable[0][0]}, expected a unique nearest marker"
        )

    _, dd, pt = reachable[0]

    # Actual 180° reversals are still disallowed.
    assert dd != (d + 2) % 4, (
        f"head={head} dir={d}: would reverse to dir={dd}, not allowed"
    )

    return dd, pt


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, _agent, markers = found
    markers = set((int(y), int(x)) for y, x in markers)
    agent = {(head[0] + oy, head[1] + ox) for oy, ox in TEMPLATES[tname][d]}

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

        # Markers are only consumed when the head lands on them.
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue the triangle sprite's chase from the latest snapshot until every marker is consumed.

    Input structure:
    - Several equal-sized frames are concatenated left-to-right, with optional
      full-height divider columns in a non-background color.
    - Inside each frame, one actor color is shared by both the triangle sprite
      and all remaining single-pixel markers.

    Rule:
    - Read only the rightmost frame as the current state.
    - From the sprite head, inspect every aligned marker on the four cardinal
      half-lines; among all reachable markers, move to the unique nearest one,
      even if multiple directions are populated.
    - Landing on that marker consumes it and the triangle reorients to the move
      direction; actual 180° reversals still do not occur.
    - Repeat until no markers remain, then return a clean frame containing only
      the final sprite body.

    Key insight: the triangle tip gives the current head and facing, but the
    move choice is no longer "the only non-empty direction" — it is the
    shortest visible aligned distance across all directions.
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


# --------------- generator (D14 variant: nearest across multiple directions) ---------------

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

# --- CFG: variant visibility ---
MIN_GAP_FOR_VARIANT = 2
MIN_SUFFIX_MULTI_DIR_STEPS = 1
MIN_VISIBLE_MULTI_SNAPSHOTS = 1

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 3000

# Reweight step counts so counts compatible with fewer visible frame cadences
# are not over-represented.
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
) -> tuple[int, int, int, int, str]:
    """Sample (n_steps, h, w, bg, actor, tname)."""
    multiples = [n for n in range(*STEPS_RANGE) if _FRAME_AVAILABILITY[n] > 0]
    weights = np.array(
        [(n ** 3) / (_FRAME_AVAILABILITY[n] ** 2) for n in multiples],
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


def _rollout_path(
    rng: np.random.Generator, tname: str, step_dirs: list[int], h: int, w: int
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]] | None:
    """Greedy path rollout. Returns (heads, state_dirs, placements) or None on stuck step."""
    n_steps = len(step_dirs)
    d0 = step_dirs[0]
    heads: list[tuple[int, int]] = [(0, 0)]
    state_dirs: list[int] = [d0]
    placements: list[tuple[tuple[int, int], int]] = [((0, 0), d0)]

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
            if not _marker_compatible_with_existing(nxt, heads, state_dirs, tname):
                continue
            trial_placements = placements + [(nxt, next_dir)]
            if _shift_interval(trial_placements, h, w, tname) is None:
                continue
            chosen = nxt
            break

        if chosen is None:
            return None

        heads.append(chosen)
        state_dirs.append(next_dir)
        placements.append((chosen, next_dir))

    return heads, state_dirs, placements


def _marker_compatible_with_existing(
    nxt: tuple[int, int],
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
) -> bool:
    """Validate candidate marker against all previously-placed heads.

    The old task forbade aligned markers on other directions entirely.
    This variant allows them, but any extra aligned marker seen from an older
    head must be strictly farther than that step's true next marker.
    """
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, state_dirs[j])
        if any(_touching(nxt, cell) for cell in sprite):
            return False
        rel_dir = _direction(old_head, nxt)
        if rel_dir is None:
            continue
        if j < len(heads) - 1 and _dist(old_head, nxt) <= _dist(old_head, heads[j + 1]):
            return False
    return True


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


def _trace_rule_profile(
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
    h: int,
    w: int,
) -> list[dict] | None:
    """Check the full path against the new nearest-across-directions rule."""
    n_steps = len(heads) - 1
    profile: list[dict] = []
    for j in range(n_steps):
        markers = set(heads[j + 1 :])
        reachable = _reachable_targets(heads[j], state_dirs[j], markers, tname, h, w)
        if not reachable:
            return None
        if len(reachable) >= 2 and reachable[0][0] >= reachable[1][0]:
            return None
        if reachable[0][1] == (state_dirs[j] + 2) % 4:
            return None
        if reachable[0][2] != heads[j + 1]:
            return None
        if reachable[0][1] != state_dirs[j + 1]:
            return None
        dirs = sorted({dd for _, dd, _ in reachable})
        profile.append(
            {
                "step": j,
                "dir_count": len(dirs),
                "dirs": dirs,
                "chosen_dir": reachable[0][1],
                "chosen_target": reachable[0][2],
                "nearest_dist": reachable[0][0],
                "multi_dir": len(dirs) >= 2,
            }
        )
    return profile


def _frame_options_from_profile(
    n_steps: int, profile: list[dict]
) -> list[tuple[int, int, int, int, int, int]]:
    """Return valid frame choices with visibility metadata."""
    out: list[tuple[int, int, int, int, int, int]] = []
    for n_frames in N_FRAMES_CHOICES:
        if n_steps % n_frames != 0:
            continue
        gap = n_steps // n_frames
        if gap < MIN_GAP_FOR_VARIANT:
            continue
        last_idx = (n_frames - 1) * gap
        suffix_multi = sum(int(item["multi_dir"]) for item in profile[last_idx:])
        visible_multi = sum(int(profile[fi * gap]["multi_dir"]) for fi in range(n_frames))
        if suffix_multi < MIN_SUFFIX_MULTI_DIR_STEPS:
            continue
        if visible_multi < MIN_VISIBLE_MULTI_SNAPSHOTS:
            continue
        last_frame_multi = int(profile[last_idx]["multi_dir"])
        out.append((n_frames, gap, last_idx, suffix_multi, visible_multi, last_frame_multi))
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


def _choose_translation_and_frames(
    rng: np.random.Generator,
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    placements: list[tuple[tuple[int, int], int]],
    tname: str,
    h: int,
    w: int,
) -> tuple[list[tuple[int, int]], list[dict], int, int, int] | None:
    shift = _shift_interval(placements, h, w, tname)
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
        profile = _trace_rule_profile(shifted_heads, state_dirs, tname, h, w)
        if profile is None:
            continue
        frame_options = _frame_options_from_profile(len(profile), profile)
        if not frame_options:
            continue
        multi_frame_options = [item for item in frame_options if item[0] > 1]
        candidates = multi_frame_options if multi_frame_options else frame_options
        weights = np.array([1.0 + item[5] for item in candidates], dtype=float)
        weights /= weights.sum()
        pick = int(rng.choice(len(candidates), p=weights))
        n_frames, gap, last_idx, _suffix_multi, _visible_multi, _last_frame_multi = candidates[pick]
        return shifted_heads, profile, n_frames, gap, last_idx
    return None


def _build_case(rng: np.random.Generator) -> dict | None:
    n_steps, h, w, bg, actor, tname = _sample_params(rng)
    step_dirs = _make_step_dirs(rng, n_steps)

    rollout = _rollout_path(rng, tname, step_dirs, h, w)
    if rollout is None:
        return None
    heads, state_dirs, placements = rollout

    ok, turns, dists = _validate_path(step_dirs, heads)
    if not ok:
        return None

    chosen = _choose_translation_and_frames(rng, heads, state_dirs, placements, tname, h, w)
    if chosen is None:
        return None
    heads, profile, n_frames, gap, last_idx = chosen

    palette = [c for c in range(10) if c not in {bg, actor}]
    divider = int(rng.choice(palette)) if n_frames > 1 else None
    placements = [(heads[i], state_dirs[i]) for i in range(n_steps + 1)]

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

    suffix_multi_steps = sum(int(item["multi_dir"]) for item in profile[last_idx:])
    last_frame_dir_count = int(profile[last_idx]["dir_count"])
    visible_multi_snapshots = sum(int(profile[fi * gap]["multi_dir"]) for fi in range(n_frames))

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
            "multi_dir_steps_total": sum(int(item["multi_dir"]) for item in profile),
            "multi_dir_steps_suffix": suffix_multi_steps,
            "visible_multi_snapshots": visible_multi_snapshots,
            "last_frame_multi": int(profile[last_idx]["multi_dir"]),
            "last_frame_dir_count": last_frame_dir_count,
            "last_frame_dirs": list(profile[last_idx]["dirs"]),
            "profile_dir_counts": [int(item["dir_count"]) for item in profile],
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
