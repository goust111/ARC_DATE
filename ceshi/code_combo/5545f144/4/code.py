# Combo variant of 5545f144 (index 4)
# combo_spec: D3/2+D7/1
# Name: 方向指示标记+前瞻朝向

import numpy as np

from arc_utils import components_mask
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

# Marker templates are centered on the landing cell. The arrow direction becomes
# the post-consumption facing, so each marker simultaneously stores the turn
# instruction and the lookahead cue for the next leg.
MARKER_TEMPLATES: dict[int, list[tuple[int, int]]] = {
    0: [(-1, 0), (0, 0), (1, -1), (1, 0), (1, 1)],
    1: [(-1, -1), (0, -1), (1, -1), (0, 0), (0, 1)],
    2: [(-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)],
    3: [(-1, 1), (0, -1), (0, 0), (0, 1), (1, 1)],
}


# ---------------------------------------------------------------------------
# Design analysis for combo D3/2 + D7/1
#
# a) Concreteization and behavior trace
# - Original state = one triangle sprite plus same-color single-pixel markers.
#   From the current head, geometry alone determined a unique half-line with
#   reachable markers; landing re-faced the sprite toward the move direction.
# - Combo state = one actor-colored triangle plus several marker-colored 5-pixel
#   arrow glyphs. The center of each glyph is still the landing cell, but the
#   glyph orientation tells the sprite how to face after that marker is eaten.
# - Small trace with heads H0 -> H1 -> H2:
#   1. At H0 the sprite already faces the half-line toward H1, so it moves to
#      the nearest marker center on that facing line.
#   2. The consumed marker at H1 carries an arrow, say "Down".
#   3. After landing, the sprite must face Down; the generator also guarantees
#      that the next marker H2 really lies on that downward half-line.
#   4. Therefore the same visible arrow on the marker is both a turn command
#      (D3) and a one-step preview of the next leg (D7).
#   The main difference from the original is that post-landing orientation is no
#   longer derived from the just-finished move. It is stored explicitly on the
#   consumed marker and must agree with the future path.
#
# b) Complexity calibration
# - Only one extra object family is introduced: a compact 5-cell arrow glyph in
#   one dedicated color. Four orientations are enough to encode direction while
#   keeping recognition stable.
# - The sprite templates, snapshot cadence, and path-length ranges stay close to
#   the source task, so the family resemblance remains visible.
# - To make the D7 part observable, generation uses 2..4 frames and forces every
#   snapshot boundary after the first to occur right after a turn. To make the D3
#   part observable, generation also creates at least one state where future
#   markers occupy multiple directions around the head, so geometry alone is not
#   reduced to the original "single allowed half-line" condition.
#
# c) Implicit-assumption audit for the old main()
# - Old color parsing assumed one non-bg non-divider color. That fails because the
#   combo adds a marker color, so actor and marker parsing must be separated.
# - Old _find_agent() tolerated leftover same-color marker pixels around the
#   sprite. That no longer works because markers are multi-pixel glyphs in a
#   different color; the actor should be detected exactly.
# - Old target selection scanned all four directions and relied on uniqueness.
#   The combo intentionally drops that assumption: the current sprite facing alone
#   chooses the active half-line, and other future markers may exist elsewhere.
# - Old state update set d = move_direction after landing. That is invalid now:
#   d must come from the consumed marker, and when markers remain that direction
#   must actually lead to the next consumed marker.
#
# d) Placement legality
# - Every visible sprite stamp, every translated pre-turn sprite stamp at landing,
#   and every marker glyph must fit fully in-grid after one global translation.
# - Marker glyphs must stay disjoint from all earlier visible sprite placements
#   and from one another, including 8-neighborhood touching, so components remain
#   easy to segment.
# - For each past state, any later marker that lies on the already-committed
#   facing half-line must be strictly farther than the intended next marker;
#   otherwise the nearest-target rule would consume the wrong one first.
#
# e) Visual observability
# - Humans can see both control signals: the triangle tip shows "where to go now",
#   while the consumed marker's own arrow tells "how to face next".
# - The frames visibly differ from the original because later snapshots show a
#   sprite orientation that previews the next leg rather than repeating the
#   previous leg. The markers are also visibly richer than plain dots.
# - No hidden ordering by color id or coordinate sort is used; only visible shape,
#   alignment, and proximity matter.
#
# f) Visual cleanliness
# - The actor and markers use different colors, so the triangle never visually
#   merges with the arrow glyphs.
# - Marker glyphs are kept non-touching and away from concurrently visible sprite
#   states, avoiding cluttered overlaps.
# - Because each frame contains only one sprite and a moderate number of compact
#   arrows, the last snapshot remains readable even with multi-step lookahead.
#
# g) Change inferability
# - There is only one added semantic color: "directional marker". Every marker in
#   that color follows the same four-shape orientation rule, repeated within a
#   case, so the mapping is learnable from a few examples.
# - The crucial combined behavior is structurally supported by repeated events:
#   some markers request a turn, some visible snapshots preview the next segment,
#   and some states contain off-axis future markers, making the fusion rule
#   inferable rather than a one-off coincidence.
# ---------------------------------------------------------------------------


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


def _template_bounds(offs: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [dr for dr, _ in offs]
    cs = [dc for _, dc in offs]
    return min(rs), max(rs), min(cs), max(cs)


def _sprite_bounds(tname: str, d: int) -> tuple[int, int, int, int]:
    return _template_bounds(TEMPLATES[tname][d])


def _marker_bounds(d: int) -> tuple[int, int, int, int]:
    return _template_bounds(MARKER_TEMPLATES[d])


def _sprite_cells(head: tuple[int, int], tname: str, d: int) -> set[tuple[int, int]]:
    hr, hc = head
    return {(hr + dr, hc + dc) for dr, dc in TEMPLATES[tname][d]}


def _marker_cells(anchor: tuple[int, int], d: int) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(ar + dr, ac + dc) for dr, dc in MARKER_TEMPLATES[d]}


def _fits_cells(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _template_patch(tname: str, d: int, color: int, bg: int) -> tuple[np.ndarray, int, int]:
    r0, r1, c0, c1 = _sprite_bounds(tname, d)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for dr, dc in TEMPLATES[tname][d]:
        patch[dr - r0, dc - c0] = color
    return patch, -r0, -c0


def _marker_patch(d: int, color: int, bg: int) -> tuple[np.ndarray, int, int]:
    r0, r1, c0, c1 = _marker_bounds(d)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for dr, dc in MARKER_TEMPLATES[d]:
        patch[dr - r0, dc - c0] = color
    return patch, -r0, -c0


def _find_agent_exact(
    frame: np.ndarray, actor: int
) -> tuple[str, int, tuple[int, int], set[tuple[int, int]]] | None:
    pts = set(map(tuple, np.argwhere(frame == actor)))
    if not pts:
        return None

    best = None
    for head in pts:
        hy, hx = int(head[0]), int(head[1])
        for tname, dirs in TEMPLATES.items():
            for d, offs in dirs.items():
                placed = {(hy + dy, hx + dx) for dy, dx in offs}
                if placed == pts:
                    score = len(placed)
                    if best is None or score > best[0]:
                        best = (score, tname, d, (hy, hx), placed)

    if best is None:
        return None
    _, tname, d, head, placed = best
    return tname, d, head, set(map(tuple, placed))


def _match_marker_component(coords: np.ndarray) -> tuple[int, tuple[int, int]] | None:
    pts = {(int(r), int(c)) for r, c in coords}
    if len(pts) != 5:
        return None
    for anchor in pts:
        ay, ax = anchor
        for d, offs in MARKER_TEMPLATES.items():
            placed = {(ay + dy, ax + dx) for dy, dx in offs}
            if placed == pts:
                return d, (ay, ax)
    return None


def _find_markers(frame: np.ndarray, marker_color: int) -> dict[tuple[int, int], int]:
    mask = frame == marker_color
    comps = components_mask(mask, connectivity=4)
    markers: dict[tuple[int, int], int] = {}
    for coords in comps:
        found = _match_marker_component(coords)
        assert found is not None, "marker component did not match any directional glyph"
        d, anchor = found
        assert anchor not in markers, "duplicate marker anchor detected"
        markers[anchor] = d
    return markers


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: dict[tuple[int, int], int],
) -> tuple[tuple[int, int], int] | None:
    hy, hx = head
    candidates: list[tuple[int, tuple[int, int], int]] = []
    for (y, x), nd in markers.items():
        if d == 0 and x == hx and y < hy:
            candidates.append((hy - y, (y, x), nd))
        elif d == 1 and y == hy and x > hx:
            candidates.append((x - hx, (y, x), nd))
        elif d == 2 and x == hx and y > hy:
            candidates.append((y - hy, (y, x), nd))
        elif d == 3 and y == hy and x < hx:
            candidates.append((hx - x, (y, x), nd))
    if not candidates:
        return None
    candidates.sort()
    _, anchor, nd = candidates[0]
    return anchor, nd


def _simulate_to_end(frame: np.ndarray, actor: int, marker_color: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent_exact(frame, actor)
    assert found is not None, "failed to detect actor sprite"
    tname, d, head, agent = found
    markers = _find_markers(frame, marker_color)

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        choice = _choose_target(head, d, markers)
        assert choice is not None, f"stuck with {len(markers)} markers at head={head} dir={d}"
        head, next_dir = choice
        del markers[head]

        if markers:
            assert _choose_target(head, next_dir, markers) is not None, (
                f"marker at {head} points to dir={next_dir}, but no next marker is reachable there"
            )
        d = next_dir
        agent = _sprite_cells(head, tname, d)
        assert _fits_cells(agent, h, w), f"sprite would leave grid at head={head} dir={d}"

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue a triangular sprite simulation whose markers also encode future facing.

    Input structure:
    - Frames are concatenated left-to-right with optional solid divider columns; the background is the most frequent color.
    - Each frame contains one actor-colored triangle plus marker-colored mini-arrow glyphs whose center cells are future landing points.

    Rule:
    - Read only the rightmost frame as the current state.
    - The sprite moves from its head to the nearest marker center on the half-line indicated by its current facing.
    - When a marker is consumed, the marker's own arrow sets the new facing; if markers remain, that arrow direction must also be the direction of the next marker that will be consumed from the new head.
    - Repeat until every marker glyph is removed; output only the final sprite on a clean background.

    Key insight: The triangle tip tells you which marker is eaten now, while the eaten marker's arrow tells you how the sprite must face for the next leg, so every step couples local consumption with one-step lookahead.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)
    start = frames[-1]

    colors = sorted(set(map(int, start.ravel())) - {bg})
    actor = None
    for color in colors:
        if _find_agent_exact(start, color) is not None:
            actor = int(color)
            break
    assert actor is not None, "failed to identify actor color"

    marker_colors = [int(c) for c in colors if int(c) != actor]
    assert len(marker_colors) == 1, "expected exactly one marker color"
    marker_color = marker_colors[0]
    return _simulate_to_end(start, actor, marker_color)


# --------------- generator (combo D3/2 + D7/1) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (2, 5)          # 2..4 frames so lookahead is visible
FRAME_MAX = 16
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 7, 15           # wider floor to accommodate 5-cell marker glyphs

# --- CFG: path complexity ---
STEPS_RANGE = (4, 11)            # 4..10 inclusive
STEP_DIST_RANGE = (2, 8)         # 2..7 inclusive
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 2
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 8
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 3000

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(2, 5)) for n in range(*STEPS_RANGE)}


def _shift_interval(
    sprite_fit_placements: list[tuple[tuple[int, int], int]],
    marker_placements: list[tuple[tuple[int, int], int]],
    h: int,
    w: int,
    tname: str,
) -> tuple[int, int, int, int] | None:
    tr_lo, tr_hi = -10**9, 10**9
    tc_lo, tc_hi = -10**9, 10**9

    for (hr, hc), d in sprite_fit_placements:
        r0, r1, c0, c1 = _sprite_bounds(tname, d)
        tr_lo = max(tr_lo, -hr - r0)
        tr_hi = min(tr_hi, h - 1 - hr - r1)
        tc_lo = max(tc_lo, -hc - c0)
        tc_hi = min(tc_hi, w - 1 - hc - c1)
        if tr_lo > tr_hi or tc_lo > tc_hi:
            return None

    for (mr, mc), d in marker_placements:
        r0, r1, c0, c1 = _marker_bounds(d)
        tr_lo = max(tr_lo, -mr - r0)
        tr_hi = min(tr_hi, h - 1 - mr - r1)
        tc_lo = max(tc_lo, -mc - c0)
        tc_hi = min(tc_hi, w - 1 - mc - c1)
        if tr_lo > tr_hi or tc_lo > tc_hi:
            return None

    return tr_lo, tr_hi, tc_lo, tc_hi


def _sample_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int, int, int, int, str]:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0]
    weights = np.array([(n ** 3) / max(1, _STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    h_lo = min(H_CAP, max(H_FLOOR, 7 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 5 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = rng.permutation(10)
    bg, actor, marker_color, divider = [int(x) for x in palette[:4]]

    weights_t = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    weights_t /= weights_t.sum()
    tname = str(rng.choice(list(TEMPLATES), p=weights_t))
    return n_frames, n_steps, gap, h, w, bg, actor, marker_color, divider, tname


def _make_move_dirs(rng: np.random.Generator, n_steps: int) -> list[int]:
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


def _sample_final_dir(rng: np.random.Generator, last_move_dir: int) -> int:
    opts = np.array([last_move_dir, (last_move_dir + 1) % 4, (last_move_dir - 1) % 4], dtype=int)
    probs = np.array([0.25, 0.375, 0.375], dtype=float)
    return int(rng.choice(opts, p=probs))


def _marker_compatible_with_existing(
    nxt: tuple[int, int],
    next_dir: int,
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    sprite_placements: list[tuple[tuple[int, int], int]],
    marker_placements: list[tuple[tuple[int, int], int]],
    tname: str,
) -> bool:
    nxt_cells = _marker_cells(nxt, next_dir)

    for head, d in sprite_placements:
        sprite = _sprite_cells(head, tname, d)
        if any(_touching(a, b) for a in nxt_cells for b in sprite):
            return False

    for anchor, d in marker_placements:
        other = _marker_cells(anchor, d)
        if any(_touching(a, b) for a in nxt_cells for b in other):
            return False

    for j, old_head in enumerate(heads[:-1]):
        rel_dir = _direction(old_head, nxt)
        if rel_dir != state_dirs[j]:
            continue
        if _dist(old_head, nxt) <= _dist(old_head, heads[j + 1]):
            return False

    return True


def _rollout_path(
    rng: np.random.Generator,
    tname: str,
    move_dirs: list[int],
    final_dir: int,
    h: int,
    w: int,
) -> tuple[
    list[tuple[int, int]],
    list[int],
    list[tuple[tuple[int, int], int]],
    list[tuple[tuple[int, int], int]],
    list[tuple[tuple[int, int], int]],
] | None:
    state_dirs = list(move_dirs) + [final_dir]
    n_steps = len(move_dirs)

    heads: list[tuple[int, int]] = [(0, 0)]
    sprite_placements: list[tuple[tuple[int, int], int]] = [((0, 0), state_dirs[0])]
    fit_placements: list[tuple[tuple[int, int], int]] = [((0, 0), state_dirs[0])]
    marker_placements: list[tuple[tuple[int, int], int]] = []

    for i in range(n_steps):
        cur = heads[-1]
        move_dir = state_dirs[i]
        next_dir = state_dirs[i + 1]
        dr, dc = DIRS4[move_dir]
        dist_order = list(rng.permutation(np.arange(*STEP_DIST_RANGE)))
        chosen = None

        for dist in dist_order:
            nxt = (cur[0] + int(dist) * dr, cur[1] + int(dist) * dc)
            if nxt in heads:
                continue
            if any(_touching(nxt, m) for m in heads[1:]):
                continue
            if not _marker_compatible_with_existing(
                nxt,
                next_dir,
                heads,
                state_dirs,
                sprite_placements,
                marker_placements,
                tname,
            ):
                continue

            trial_fit = fit_placements + [(nxt, move_dir), (nxt, next_dir)]
            trial_markers = marker_placements + [(nxt, next_dir)]
            if _shift_interval(trial_fit, trial_markers, h, w, tname) is None:
                continue

            chosen = nxt
            break

        if chosen is None:
            return None

        heads.append(chosen)
        sprite_placements.append((chosen, next_dir))
        fit_placements.extend([(chosen, move_dir), (chosen, next_dir)])
        marker_placements.append((chosen, next_dir))

    return heads, state_dirs, sprite_placements, fit_placements, marker_placements


def _visible_lookahead_frames(move_dirs: list[int], state_dirs: list[int], gap: int, n_frames: int) -> int:
    visible = 0
    for fi in range(1, n_frames):
        si = fi * gap
        if state_dirs[si] != move_dirs[si - 1]:
            visible += 1
    return visible


def _instruction_turns(move_dirs: list[int], marker_placements: list[tuple[tuple[int, int], int]]) -> int:
    return sum(int(d != move_dirs[i]) for i, (_head, d) in enumerate(marker_placements))


def _branch_stats(
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    markers: list[tuple[tuple[int, int], int]],
) -> int:
    branching = 0
    for i, head in enumerate(heads[:-1]):
        facing = state_dirs[i]
        hy, hx = head
        dirs = set()
        for (y, x), _ in markers[i:]:
            rel = _direction((hy, hx), (y, x))
            if rel is not None:
                dirs.add(rel)
        if len(dirs) >= 2 and any(
            (facing == 0 and x == hx and y < hy)
            or (facing == 1 and y == hy and x > hx)
            or (facing == 2 and x == hx and y > hy)
            or (facing == 3 and y == hy and x < hx)
            for (y, x), _ in markers[i:]
        ):
            branching += 1
    return branching


def _validate_path(
    move_dirs: list[int],
    state_dirs: list[int],
    heads: list[tuple[int, int]],
    gap: int,
    n_frames: int,
    marker_placements: list[tuple[tuple[int, int], int]],
) -> tuple[bool, int, list[int], int, int, int]:
    n_steps = len(move_dirs)
    dir_set = set(move_dirs)
    if len(dir_set) < MIN_DIR_DIVERSITY:
        return False, 0, [], 0, 0, 0
    if not (dir_set & {0, 2} and dir_set & {1, 3}):
        return False, 0, [], 0, 0, 0

    turns = sum(move_dirs[i] != move_dirs[i - 1] for i in range(1, n_steps))
    if turns < MIN_TURNS_ACCEPT or not (TURN_RATIO_LO <= turns / n_steps <= TURN_RATIO_HI):
        return False, turns, [], 0, 0, 0

    dists = [_dist(heads[i], heads[i + 1]) for i in range(n_steps)]
    if len(set(dists)) < MIN_DIST_VARIETY:
        return False, turns, dists, 0, 0, 0

    boundary_turns = 0
    for fi in range(1, n_frames):
        si = fi * gap
        if move_dirs[si] == move_dirs[si - 1]:
            return False, turns, dists, boundary_turns, 0, 0
        boundary_turns += 1

    visible_diff_frames = _visible_lookahead_frames(move_dirs, state_dirs, gap, n_frames)
    instruction_turns = _instruction_turns(move_dirs, marker_placements)
    branching_heads = _branch_stats(heads, state_dirs, marker_placements)
    return True, turns, dists, boundary_turns, visible_diff_frames, branching_heads + instruction_turns


def _render_frame(
    h: int,
    w: int,
    bg: int,
    actor: int,
    marker_color: int,
    head: tuple[int, int],
    d: int,
    tname: str,
    markers: list[tuple[tuple[int, int], int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    patch, ar, ac = _template_patch(tname, d, actor, bg)
    paste(frame, patch, head[0] - ar, head[1] - ac, transparent=bg)
    for anchor, md in markers:
        mp, mr, mc = _marker_patch(md, marker_color, bg)
        paste(frame, mp, anchor[0] - mr, anchor[1] - mc, transparent=bg)
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
    n_frames, n_steps, gap, h, w, bg, actor, marker_color, divider, tname = _sample_params(rng)
    move_dirs = _make_move_dirs(rng, n_steps)
    final_dir = _sample_final_dir(rng, move_dirs[-1])

    rollout = _rollout_path(rng, tname, move_dirs, final_dir, h, w)
    if rollout is None:
        return None
    heads, state_dirs, sprite_placements, fit_placements, marker_placements = rollout

    ok, turns, dists, boundary_turns, visible_diff_frames, combo_strength = _validate_path(
        move_dirs, state_dirs, heads, gap, n_frames, marker_placements
    )
    if not ok:
        return None

    shift = _shift_interval(fit_placements, marker_placements, h, w, tname)
    if shift is None:
        return None
    tr_lo, tr_hi, tc_lo, tc_hi = shift
    tr = int(rng.integers(tr_lo, tr_hi + 1))
    tc = int(rng.integers(tc_lo, tc_hi + 1))

    heads = [(r + tr, c + tc) for r, c in heads]
    sprite_placements = [((r + tr, c + tc), d) for (r, c), d in sprite_placements]
    fit_placements = [((r + tr, c + tc), d) for (r, c), d in fit_placements]
    marker_placements = [((r + tr, c + tc), d) for (r, c), d in marker_placements]

    frames = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = sprite_placements[si]
        frames.append(
            _render_frame(h, w, bg, actor, marker_color, head, d, tname, marker_placements[si:])
        )

    fhead, fdir = sprite_placements[-1]
    out_frame = _render_frame(h, w, bg, actor, marker_color, fhead, fdir, tname, [])
    inp = _assemble_input(frames, bg, divider)

    found_actor = _find_agent_exact(frames[-1], actor)
    if found_actor is None:
        return None
    f_tname, f_seen_dir, f_seen_head, _f_agent = found_actor
    last_idx = (n_frames - 1) * gap
    if (f_tname, f_seen_dir, f_seen_head) != (
        tname,
        sprite_placements[last_idx][1],
        sprite_placements[last_idx][0],
    ):
        return None

    found_markers = _find_markers(frames[-1], marker_color)
    expect_markers = {anchor: d for anchor, d in marker_placements[last_idx:]}
    if found_markers != expect_markers:
        return None

    if not np.array_equal(main(inp), out_frame):
        return None

    branching_heads = _branch_stats(heads, state_dirs, marker_placements)
    instruction_turns = _instruction_turns(move_dirs, marker_placements)
    visible_diff_frames = _visible_lookahead_frames(move_dirs, state_dirs, gap, n_frames)
    validated_lookahead_steps = 0
    for i in range(n_steps - 1):
        nxt = _choose_target(heads[i + 1], marker_placements[i][1], dict(marker_placements[i + 1 :]))
        if nxt is not None:
            validated_lookahead_steps += 1

    if branching_heads == 0:
        return None
    if visible_diff_frames == 0:
        return None
    if instruction_turns == 0:
        return None
    if validated_lookahead_steps != n_steps - 1:
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
            "marker_color": marker_color,
            "divider": divider,
            "total_steps": n_steps,
            "gap": gap,
            "turns": turns,
            "turn_ratio": turns / n_steps,
            "direction_diversity": len(set(move_dirs)),
            "move_dirs": list(move_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "marker_dirs": [d for _anchor, d in marker_placements],
            "step_distances": dists,
            "count_in_first_frame": n_steps,
            "count_in_last_input_frame": gap,
            "boundary_turns": boundary_turns,
            "visible_lookahead_frames": visible_diff_frames,
            "branching_heads": branching_heads,
            "instruction_turns": instruction_turns,
            "validated_lookahead_steps": validated_lookahead_steps,
            "final_dir_differs_from_last_move": state_dirs[-1] != move_dirs[-1],
            "combo_strength": combo_strength,
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
