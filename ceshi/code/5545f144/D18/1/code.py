# Variant of 5545f144
# Dimension: D18 — 步距约束 (meta)
# Change: 每步移动的曼哈顿距离为2-7格，且整条路径至少有2种不同步距。... → 递增步距

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left

# Design analysis for the D18 variant
# -----------------------------------
# a) Concrete behavior:
#    The triangle / divider / color scheme stays the same. The visible change is
#    in how the remaining dots are interpreted from the rightmost input frame.
#    If there are F frames and the rightmost frame still shows g dots, then that
#    frame is the state after (F-1)*g moves, so the remaining jump lengths are
#    forced to be (F-1)*g+2, (F-1)*g+3, ..., F*g+1. Example: F=2 and g=3 means
#    the sprite has already taken jumps 2,3,4; the remaining jumps must be 5,6,7.
#    Under the original rule a nearer aligned dot would be consumed first; under
#    the variant the sprite must ignore that nearer decoy unless it lies exactly
#    at the next required distance. That is the key local divergence.
# b) Complexity calibration:
#    If the generator kept the original "nearest dot wins" geometry, then the new
#    step-length rule would be visually present in metadata only and would not
#    affect the output on the same input. To make D18 operational, the generator
#    now permits extra aligned dots at wrong distances, while preserving a unique
#    dot at the required distance for every step. This creates samples where the
#    old nearest-step logic is wrong or ambiguous, so the distance schedule is
#    genuinely needed.
# c) Hidden assumptions audit:
#    - Original _choose_target assumed that at most one cardinal half-line from
#      the current head contained any reachable dots, and then chose the nearest.
#      That premise is intentionally broken here; solver-side target selection is
#      replaced with "pick the unique dot at the required distance".
#    - Original generator rejected any future marker that was closer than the
#      immediate next marker from an earlier head. That would suppress the new
#      behavior, so compatibility checking is relaxed: only collisions at the
#      exact required distance remain forbidden.
#    - _find_agent still assumes one actor color shared by sprite and markers.
#      The generator preserves that invariant and validates that detection on the
#      last input frame recovers the intended template, head, facing, and markers.
# d) Placement legality:
#    Future marker cells are still the future head positions. The generator keeps
#    every stamped sprite orientation inside the frame, avoids marker/sprite body
#    overlap or touching against all earlier states, prevents duplicated heads,
#    and ensures that for every earlier head there is exactly one future marker
#    at that step's required distance.
# e) Visual observability:
#    The stride rule is now inferable from structure only: count frames, count the
#    dots left in the last frame, infer which jump length comes next, then follow
#    the only dot at that distance. No color-id ordering or hidden numeric code is
#    used. In practice the generator fixes to the single-frame / six-step regime,
#    because shorter visible suffixes never leave enough room for a closer decoy
#    marker to coexist with the required one. That keeps D18 active instead of
#    collapsing back to the original nearest-dot behavior.
# f) Visual cleanliness:
#    Marker cells are still single pixels. To make the new distance rule actually
#    matter, some future markers are allowed to sit flush against an earlier sprite
#    body, but never on top of it; the generator then validates that _find_agent
#    still recovers the intended triangle exactly. Marker heads still avoid each
#    other so the dot field does not dissolve into an unstructured mass.
# g) Color inference:
#    Colors are unchanged from the original task: one background color, one actor
#    color shared by sprite and markers, and an optional divider color. The D18
#    change is entirely geometric, so no extra color mapping burden is added.


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


def _fits_template(head: tuple[int, int], tname: str, d: int, h: int, w: int) -> bool:
    hy, hx = head
    for oy, ox in TEMPLATES[tname][d]:
        ny = hy + oy
        nx = hx + ox
        if not (0 <= ny < h and 0 <= nx < w):
            return False
    return True


def _first_remaining_distance(n_frames: int, remaining_steps: int) -> int:
    return (n_frames - 1) * remaining_steps + 2


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    tname: str,
    h: int,
    w: int,
    expected_dist: int,
) -> tuple[int, tuple[int, int]] | None:
    hy, hx = head
    reachable: list[tuple[int, tuple[int, int]]] = []
    for y, x in markers:
        if x == hx and y < hy:
            dd, dist = 0, hy - y
        elif y == hy and x > hx:
            dd, dist = 1, x - hx
        elif x == hx and y > hy:
            dd, dist = 2, y - hy
        elif y == hy and x < hx:
            dd, dist = 3, hx - x
        else:
            continue
        if dist != expected_dist:
            continue
        if not _fits_template((y, x), tname, dd, h, w):
            continue
        reachable.append((dd, (y, x)))

    if not reachable:
        return None

    dirs_with_candidates = {dd for dd, _ in reachable}
    assert len(dirs_with_candidates) == 1, (
        f"head={head} dir={d}: markers at distance {expected_dist} appear in multiple "
        f"directions {dirs_with_candidates}, expected exactly one"
    )

    dd, pt = reachable[0]
    assert dd != (d + 2) % 4, (
        f"head={head} dir={d}: expected-distance target would reverse to dir={dd}"
    )
    return dd, pt


def _choose_target_nearest(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    offs: list[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    """Original nearest-marker rule, kept only for generator-side degeneration checks."""
    hy, hx = head

    def candidates(dir_idx: int) -> list[tuple[int, tuple[int, int]]]:
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

    def fits(target: tuple[int, int]) -> bool:
        ty, tx = target
        dy, dx = ty - hy, tx - hx
        for oy, ox in offs:
            ny = hy + oy + dy
            nx = hx + ox + dx
            if not (0 <= ny < h and 0 <= nx < w):
                return False
        return True

    reachable: list[tuple[int, int, tuple[int, int]]] = []
    for dd in range(4):
        for dist, pt in candidates(dd):
            if fits(pt):
                reachable.append((dist, dd, pt))

    if not reachable:
        return None

    dirs_with_candidates = {dd for _, dd, _ in reachable}
    assert len(dirs_with_candidates) == 1, (
        f"head={head} dir={d}: nearest-rule ambiguity in directions {dirs_with_candidates}"
    )

    reachable.sort()
    _, dd, pt = reachable[0]
    assert dd != (d + 2) % 4, (
        f"head={head} dir={d}: nearest rule would reverse to dir={dd}"
    )
    return dd, pt


def _simulate_to_end(frame: np.ndarray, actor: int, n_frames: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, agent, markers = found
    markers = set((int(y), int(x)) for y, x in markers)
    remaining_steps = len(markers)
    expected_dist = _first_remaining_distance(n_frames, remaining_steps)

    for _ in range(remaining_steps):
        choice = _choose_target(head, d, markers, tname, h, w, expected_dist)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}; "
            f"expected a marker exactly {expected_dist} cells away"
        )
        nd, (ty, tx) = choice
        head = (ty, tx)
        d = nd
        agent = {(ty + oy, tx + ox) for oy, ox in TEMPLATES[tname][d]}
        markers.discard(head)
        expected_dist += 1

    assert not markers, f"{len(markers)} markers remained after the scheduled jumps"

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def _simulate_to_end_nearest(frame: np.ndarray, actor: int) -> np.ndarray:
    """Old solver behavior, used only to ensure generated cases are non-degenerate."""
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, agent, markers = found
    markers = set((int(y), int(x)) for y, x in markers)

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break
        choice = _choose_target_nearest(head, d, markers, TEMPLATES[tname][d], h, w)
        assert choice is not None, (
            f"Old nearest rule got stuck with {len(markers)} markers at head={head} dir={d}"
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
    Advance a triangular sprite through the remaining markers using scheduled jump lengths.

    Input structure:
    - The input is a horizontal strip of snapshots, optionally separated by solid divider columns; this variant's generator
      keeps a single starting snapshot so the whole increasing 2,3,4,... schedule remains visible.
    - That snapshot contains one triangular sprite plus the still-unconsumed single-pixel markers in the same actor color.

    Rule:
    - Read the rightmost snapshot as the current state.
    - Let F be the number of snapshots and g be the number of markers left in that rightmost snapshot.
    - Because the snapshots are evenly spaced and the first jump is 2 cells, the remaining head-to-marker distances must be
      (F-1)*g+2, (F-1)*g+3, ..., F*g+1.
    - For each required distance, exactly one cardinal half-line contains the marker at that distance; move the head there,
      consume that marker, and re-face the sprite in that direction.
    - After all remaining markers are consumed, output only the final sprite on a clean background.

    Key insight: the frame cadence tells you which jump number the last input frame represents, so the correct target is not
    always the nearest aligned dot; it is the aligned dot at the next required distance in the 2,3,4,... schedule.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    # The latest frame contains only bg + actor, so detect the moving color there.
    actor = next(iter(set(map(int, frames[-1].ravel())) - {bg}))

    # Use the latest frame (rightmost) as the starting state.
    start = frames[-1]
    return _simulate_to_end(start, actor, len(frames))


# --------------- generator (D18 increasing-step variant) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (1, 2)          # rng.integers low/high+1 -> always 1 frame
FRAME_MAX = 16                   # rng.integers high for H, W -> 8..15 / 6..15
H_FLOOR, H_CAP = 13, 15          # larger boards help place 2..7 jumps plus close decoys
W_FLOOR, W_CAP = 13, 15

# --- CFG: path complexity ---
STEPS_RANGE = (6, 7)             # always 6 steps, so visible distances are exactly 2..7
STEP_DIST_START = 2
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_LAST_FRAME_MARKERS = 6       # with one frame, all six scheduled jumps remain visible
MIN_VISIBLE_CONFLICTS = 1        # at least one later marker is closer than the required next jump
TEMPLATE_OPTIONS = ("hollow3", "filled4")

# --- CFG: retry budget ---
MAX_ATTEMPTS = 500

_DIR_SEQUENCE_CACHE: dict[int, list[tuple[int, ...]]] = {}


def _sample_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int, int, int | None, str]:
    """Sample (n_frames, n_steps, gap, h, w, bg, actor, divider, tname)."""
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [
        n
        for n in range(*STEPS_RANGE)
        if n % n_frames == 0 and n // n_frames >= MIN_LAST_FRAME_MARKERS
    ]
    n_steps = int(rng.choice(multiples))
    gap = n_steps // n_frames

    h_lo = min(H_CAP, max(H_FLOOR, 7 + n_steps))
    w_lo = min(W_CAP, max(W_FLOOR, 7 + n_steps))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = rng.permutation(10)
    bg, actor = int(palette[0]), int(palette[1])
    divider = int(palette[2]) if n_frames > 1 else None
    tname = str(rng.choice(TEMPLATE_OPTIONS))
    return n_frames, n_steps, gap, h, w, bg, actor, divider, tname


def _step_distances(n_steps: int) -> list[int]:
    return [STEP_DIST_START + i for i in range(n_steps)]


def _direction_constraints(step_dirs: list[int]) -> tuple[bool, int]:
    n_steps = len(step_dirs)
    dir_set = set(step_dirs)
    if len(dir_set) < MIN_DIR_DIVERSITY:
        return False, 0
    if not (dir_set & {0, 2} and dir_set & {1, 3}):
        return False, 0
    turns = sum(step_dirs[i] != step_dirs[i - 1] for i in range(1, n_steps))
    if turns < MIN_TURNS_ACCEPT or not (TURN_RATIO_LO <= turns / n_steps <= TURN_RATIO_HI):
        return False, turns
    return True, turns


def _dir_sequences(n_steps: int) -> list[tuple[int, ...]]:
    if n_steps in _DIR_SEQUENCE_CACHE:
        return _DIR_SEQUENCE_CACHE[n_steps]

    out: list[tuple[int, ...]] = []

    def rec(seq: tuple[int, ...]) -> None:
        if len(seq) == n_steps:
            ok, _ = _direction_constraints(list(seq))
            if ok:
                out.append(seq)
            return
        if not seq:
            choices = range(4)
        else:
            prev = seq[-1]
            choices = (prev, (prev + 1) % 4, (prev - 1) % 4)
        for d in choices:
            rec(seq + (int(d),))

    rec(tuple())
    _DIR_SEQUENCE_CACHE[n_steps] = out
    return out


def _marker_compatible_with_existing(
    nxt: tuple[int, int],
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    step_dirs: list[int],
    step_dists: list[int],
    tname: str,
) -> bool:
    """Validate candidate marker against all previously-placed heads.

    Compared with the original task, wrong-distance aligned markers are allowed;
    only exact-distance conflicts are forbidden because the solver now keys off
    the jump-length schedule instead of "nearest wins". Markers may sit flush
    against an earlier sprite body, but they may not overlap any already-stamped
    sprite pixel.
    """
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, state_dirs[j])
        if nxt in sprite:
            return False
        rel_dir = _direction(old_head, nxt)
        if rel_dir is None:
            continue
        dist = _dist(old_head, nxt)
        if dist != step_dists[j]:
            continue
        if j == len(heads) - 1:
            if rel_dir != step_dirs[j]:
                return False
            if rel_dir == (state_dirs[j] + 2) % 4:
                return False
        else:
            return False
    return True


def _rollout_path(
    tname: str,
    step_dirs: list[int],
    step_dists: list[int],
    h: int,
    w: int,
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]] | None:
    """Roll out a fixed direction / distance schedule from the origin."""
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
        dist = step_dists[i]
        nxt = (cur[0] + dist * dr, cur[1] + dist * dc)
        if nxt in heads:
            return None
        if any(_touching(nxt, m) for m in heads[1:]):
            return None
        if not _marker_compatible_with_existing(nxt, heads, state_dirs, step_dirs, step_dists, tname):
            return None
        trial_placements = placements + [(nxt, next_dir)]
        if _shift_interval(trial_placements, h, w, tname) is None:
            return None

        heads.append(nxt)
        state_dirs.append(next_dir)
        placements = trial_placements

    return heads, state_dirs, placements


def _validate_path(
    step_dirs: list[int], step_dists: list[int], heads: list[tuple[int, int]]
) -> tuple[bool, int, list[int]]:
    """Check post-hoc path-quality constraints. Returns (ok, turns, step_distances)."""
    ok, turns = _direction_constraints(step_dirs)
    if not ok:
        return False, turns, []

    dists = [_dist(heads[i], heads[i + 1]) for i in range(len(step_dirs))]
    if dists != step_dists:
        return False, turns, dists
    if dists[0] != STEP_DIST_START:
        return False, turns, dists
    if any(dists[i] != dists[i - 1] + 1 for i in range(1, len(dists))):
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


def _count_visible_conflicts(
    heads: list[tuple[int, int]], step_dists: list[int], start_idx: int
) -> int:
    """Count states from the last input frame onward that contain a closer aligned decoy."""
    conflicts = 0
    for j in range(start_idx, len(heads) - 1):
        old_head = heads[j]
        expected = step_dists[j]
        has_closer = any(
            _direction(old_head, future) is not None and _dist(old_head, future) < expected
            for future in heads[j + 1 :]
        )
        conflicts += int(has_closer)
    return conflicts


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, gap, h, w, bg, actor, divider, tname = _sample_params(rng)
    step_dists = _step_distances(n_steps)
    last_idx = (n_frames - 1) * gap

    candidates = _dir_sequences(n_steps)
    order = list(rng.permutation(len(candidates)))
    for idx in order:
        step_dirs = list(candidates[idx])
        rollout = _rollout_path(tname, step_dirs, step_dists, h, w)
        if rollout is None:
            continue
        heads, state_dirs, placements = rollout

        ok, turns, dists = _validate_path(step_dirs, step_dists, heads)
        if not ok:
            continue

        visible_conflicts = _count_visible_conflicts(heads, step_dists, last_idx)
        if visible_conflicts < MIN_VISIBLE_CONFLICTS:
            continue

        shift = _shift_interval(placements, h, w, tname)
        if shift is None:
            continue
        tr_lo, tr_hi, tc_lo, tc_hi = shift
        tr = int(rng.integers(tr_lo, tr_hi + 1))
        tc = int(rng.integers(tc_lo, tc_hi + 1))
        heads_shifted = [(r + tr, c + tc) for r, c in heads]
        placements_shifted = [(heads_shifted[i], state_dirs[i]) for i in range(n_steps + 1)]

        frames = []
        for fi in range(n_frames):
            si = fi * gap
            head, d = placements_shifted[si]
            frames.append(_render_frame(h, w, bg, actor, head, d, tname, heads_shifted[si + 1 :]))

        fhead, fdir = placements_shifted[-1]
        out_frame = _render_frame(h, w, bg, actor, fhead, fdir, tname, [])
        inp = _assemble_input(frames, bg, divider)

        found = _find_agent(frames[-1], actor)
        if found is None:
            continue
        f_tname, f_dir, f_head, _f_agent, f_markers = found
        if (f_tname, f_dir, f_head) != (tname, placements_shifted[last_idx][1], placements_shifted[last_idx][0]):
            continue
        if f_markers != set(heads_shifted[last_idx + 1 :]):
            continue
        if not np.array_equal(main(inp), out_frame):
            continue

        old_status = "matched"
        try:
            old_out = _simulate_to_end_nearest(frames[-1], actor)
            if np.array_equal(old_out, out_frame):
                continue
            old_status = "different"
        except Exception as exc:  # old rule may now be ambiguous on purpose
            old_status = type(exc).__name__

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
                "heads": list(heads_shifted),
                "step_distances": dists,
                "remaining_step_distances_last_frame": dists[last_idx:],
                "count_in_first_frame": n_steps,
                "count_in_last_input_frame": gap,
                "visible_conflicts_last_frame": visible_conflicts,
                "old_nearest_status": old_status,
            },
        }
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
