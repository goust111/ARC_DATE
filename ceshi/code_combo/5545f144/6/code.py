# Combo variant of 5545f144 (index 6)
# combo_spec: D9/2+D19/0
# Name: 倒序帧+从最左帧开始模拟

import importlib.util

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
# - The visible vocabulary stays close to the base task: one triangular actor,
#   same-color single-pixel markers, and optional full-height divider columns
#   between equal-sized horizontal frames.
# - The fused rule uses the same simulation dynamics but changes how the strip is
#   interpreted:
#     * D9/2 reverses the movie strip, so the visible order is
#       [step (F-1)g] ... [step g] [step 0].
#     * D19/0 changes the designated simulation panel to the left edge of that
#       strip rather than the right edge.
# - To keep the fusion operative instead of cosmetic, every panel gets one extra
#   reachable branch marker. The leftmost panel is therefore locally ambiguous:
#   the true suffix markers are exactly the ones that persist across all reverse-
#   time snapshots, while each injected branch appears in only one panel.
# - Small trace with F=3 and g=2:
#     chronological snapshots:   [step0 has 6 markers] [step2 has 4] [step4 has 2]
#     after adding one branch to every chronological panel:
#       [7] [5] [3]
#     after reversing for input:
#       left->right counts become [3, 5, 7]
#   The correct combo solver reads the strip as reverse-time, starts from the
#   leftmost visible panel, filters that panel down to the markers shared by all
#   snapshots, and then consumes the last two true markers. A single-variant
#   solver that only continues the raw leftmost panel hits the injected branch.
#
# b) Complexity calibration
# - The generator always uses 2-4 frames so both "reverse order" and "leftmost
#   start" are actually visible.
# - Marker counts are forced to rise strictly from left to right, giving humans a
#   repeated cue that time runs backward in the strip.
# - Only one extra branch dot is added per non-start panel. That is enough to
#   invalidate wrong start panels without making the pictures visually noisy.
#
# c) Implicit-assumption audit
# - Original `start = frames[-1]` is broken twice over: the strip order is
#   reversed, and every non-left panel is intentionally non-deterministic.
# - Original actor-color extraction assumed a single non-bg non-divider color.
#   That remains valid here; no extra semantic colors are introduced.
# - `_find_agent()` must still recover the same sprite despite extra branch dots.
#   The generator validates each frame after branch insertion.
# - `main()` now assumes the frame strip itself encodes the start panel: after
#   detecting reverse time, the solver should trust the leftmost frame and ignore
#   older ambiguous panels.
#
# d) Placement legality
# - Every true path still obeys the source task constraints: no overlap with the
#   sprite body, no clipping, no 180-degree reversals, and no multi-direction
#   ambiguity on the actual continued panel.
# - Added branch markers are placed only on a different reachable half-line from
#   the true next step, never touching the sprite, and only in panels that are
#   not meant to be simulated from.
# - The generator rejects any frame containing an internal full-height solid
#   non-background column, so divider-based frame splitting remains valid.
#
# e) Visual observability
# - A human can see that the actor is closest to completion on the left, with
#   more unresolved dots appearing as one scans rightward. Across examples, the
#   leftmost panel consistently looks like the state nearest to the output.
# - The output is the continuation of that leftmost panel, not of the rightmost
#   one, because the right side panels contain extra branch choices that do not
#   appear in the final continuation.
#
# f) Visual cleanliness
# - The branch perturbation is minimal: one extra dot in each older panel. The
#   sprite silhouette and the main marker suffix stay easy to read.
# - No legend, overlay, recoloring, or stacked annotations are added; the strip
#   still looks like one clean family of snapshots.
#
# g) Change inferability
# - No hidden color-id trick is used. Reverse time is signaled by marker-count
#   progression and by the actor getting closer to the final output on the left.
# - The two dimensions fuse into a single inference chain: recognize that the
#   strip runs backward, then continue from its left edge because that is the
#   designated usable snapshot under the combo rule.
#
# New solver assumptions and generator guarantees
# - The leftmost frame is the designated start panel, but it is not locally
#   deterministic on its own; the generator adds one unique branch marker there.
# - Marker counts of detectable frames rise strictly left-to-right, and the true
#   continuation markers are exactly the intersection of the per-frame marker
#   sets. `main()` checks that reverse-time ordering before trusting the left
#   edge and shared-marker suffix.
# - Every frame contains one detectable sprite in the same actor color.
# - Generated samples are rejected when the original source solver would produce
#   the same output, preventing collapse back to the base task.

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left

_ORIGINAL_5545F144 = None


def _original_matches(inp: np.ndarray, out: np.ndarray) -> bool:
    global _ORIGINAL_5545F144
    if _ORIGINAL_5545F144 is None:
        spec = importlib.util.spec_from_file_location(
            "orig_5545f144", "/home/lixinyu/snap/new/arc2_solver_select/5545f144.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        _ORIGINAL_5545F144 = module
    try:
        original_out = np.asarray(_ORIGINAL_5545F144.main(np.asarray(inp, dtype=int)), dtype=int)
    except Exception:
        return False
    return np.array_equal(original_out, np.asarray(out, dtype=int))


def _leftmost_only_matches(inp: np.ndarray, out: np.ndarray) -> bool:
    g = np.asarray(inp, dtype=int)
    frames, bg = _split_frames(g)
    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    actor = _detect_actor(frames, bg, div_colors)
    try:
        naive_out = _simulate_to_end(frames[0], actor)
    except Exception:
        return False
    return np.array_equal(naive_out, np.asarray(out, dtype=int))


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

    best = None
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


def _detect_actor(frames: list[np.ndarray], bg: int, div_colors: set[int]) -> int:
    colors = set(map(int, np.concatenate([frame.ravel() for frame in frames])))
    best: tuple[tuple[int, int], int] | None = None
    for color in sorted(colors - {bg} - div_colors):
        detections = [_find_agent(frame, color) for frame in frames]
        count = sum(found is not None for found in detections)
        if count == 0:
            continue
        marker_total = sum(len(found[4]) for found in detections if found is not None)
        score = (count, marker_total)
        if best is None or score > best[0]:
            best = (score, color)
    assert best is not None, "No actor-colored sprite detected"
    return best[1]


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    offs: list[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
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

    dirs_with_candidates = set(dd for _, dd, _ in reachable)
    assert len(dirs_with_candidates) == 1, (
        f"head={head} dir={d}: markers reachable in multiple directions "
        f"{dirs_with_candidates}, expected exactly one"
    )

    reachable.sort()
    _, dd, pt = reachable[0]

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

    tname, d, head, agent, markers = found
    agent = set((int(y), int(x)) for y, x in agent)
    markers = set((int(y), int(x)) for y, x in markers)

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
        agent = {(ty + oy, tx + ox) for oy, ox in TEMPLATES[tname][d]}
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def _marker_counts(frames: list[np.ndarray], actor: int) -> list[int]:
    counts: list[int] = []
    for frame in frames:
        found = _find_agent(frame, actor)
        assert found is not None, "Every frame must contain a detectable sprite"
        counts.append(len(found[4]))
    return counts


def _shared_markers(frames: list[np.ndarray], actor: int) -> set[tuple[int, int]]:
    shared: set[tuple[int, int]] | None = None
    for frame in frames:
        found = _find_agent(frame, actor)
        assert found is not None, "Every frame must contain a detectable sprite"
        markers = {tuple(map(int, pt)) for pt in found[4]}
        shared = markers if shared is None else shared & markers
    assert shared is not None
    return shared


def _frame_with_markers(
    frame: np.ndarray, actor: int, markers: set[tuple[int, int]]
) -> np.ndarray:
    bg = _bg_color(frame)
    found = _find_agent(frame, actor)
    assert found is not None, "Start frame must contain a detectable sprite"
    _tname, _d, _head, agent, _orig_markers = found

    out = np.full(frame.shape, bg, dtype=int)
    for y, x in agent:
        out[int(y), int(x)] = actor
    for y, x in markers:
        out[int(y), int(x)] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue a triangle sprite from the left edge of a reverse-time snapshot strip.

    Input structure:
    - Equal-sized frames are concatenated horizontally, with optional full-height divider columns between them.
    - Each frame contains the same triangular sprite template plus actor-colored single-pixel dots.
    - The strip is presented in reverse time: later snapshots are on the left and older snapshots are on the right.

    Rule:
    - Detect the actor color and read the frames as a reversed movie whose marker counts increase from left to right.
    - Use the leftmost frame as the designated forward-start panel.
    - Recover its true pending markers by intersecting marker sets across the full reverse-time strip; those shared dots are the ones still present in every older snapshot.
    - From that frame, repeatedly move the head to the nearest reachable marker on the unique visible half-line, consume it, and reorient the sprite to that direction.
    - Stop when no markers remain and output only the final sprite on clean background.

    Key insight: Reverse order plus one injected branch per panel means the left edge is the start state, but only the markers shared across all frames belong to its true continuation.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)
    assert len(frames) >= 2, "Combo variant expects a multi-frame strip"

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    actor = _detect_actor(frames, bg, div_colors)

    counts = _marker_counts(frames, actor)
    assert counts == sorted(counts) and len(set(counts)) == len(counts), (
        f"Expected reverse-time strip with strictly increasing marker counts, got {counts}"
    )
    shared_markers = _shared_markers(frames, actor)
    left_found = _find_agent(frames[0], actor)
    assert left_found is not None
    left_markers = {tuple(map(int, pt)) for pt in left_found[4]}
    assert shared_markers < left_markers, (
        "Expected the leftmost frame to contain extra branch markers beyond the shared suffix"
    )

    start_frame = _frame_with_markers(frames[0], actor, shared_markers)
    return _simulate_to_end(start_frame, actor)


# --------------- generator (combo D9/2 + D19/0) ---------------

N_FRAMES_RANGE = (2, 5)
FRAME_MAX = 16
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 6, 15

STEPS_RANGE = (4, 13)
STEP_DIST_RANGE = (2, 8)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

MAX_ATTEMPTS = 3000

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


def _fits_at(
    head: tuple[int, int], target: tuple[int, int], tname: str, d: int, h: int, w: int
) -> bool:
    hy, hx = head
    ty, tx = target
    dy, dx = ty - hy, tx - hx
    for oy, ox in TEMPLATES[tname][d]:
        ny = hy + oy + dy
        nx = hx + ox + dx
        if not (0 <= ny < h and 0 <= nx < w):
            return False
    return True


def _add_branch_marker(
    rng: np.random.Generator,
    frame: np.ndarray,
    actor: int,
    head: tuple[int, int],
    d: int,
    true_dir: int,
    tname: str,
    forbidden: set[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    h, w = frame.shape
    occupied = set(map(tuple, np.argwhere(frame == actor)))
    forbidden = set() if forbidden is None else {tuple(map(int, pt)) for pt in forbidden}
    dirs = [dd for dd in range(4) if dd != true_dir]
    dirs = [dirs[i] for i in rng.permutation(len(dirs))]
    dist_choices = [int(v) for v in rng.permutation(np.arange(2, max(h, w)))]

    for dd in dirs:
        dr, dc = DIRS4[dd]
        for dist in dist_choices:
            pt = (head[0] + dist * dr, head[1] + dist * dc)
            if not (0 <= pt[0] < h and 0 <= pt[1] < w):
                continue
            if pt in forbidden:
                continue
            if pt in occupied:
                continue
            if any(_touching(pt, cell) for cell in occupied):
                continue
            if not _fits_at(head, pt, tname, d, h, w):
                continue

            out = frame.copy()
            out[pt] = actor
            return out, pt
    return None


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
) -> tuple[int, int, int, int, int, int, int, int | None, str]:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0]
    weights = np.array([(n**3) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
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


def _has_divider_like_column(frame: np.ndarray, bg: int) -> bool:
    for c in range(frame.shape[1]):
        if len(set(map(int, frame[:, c]))) == 1 and int(frame[0, c]) != bg:
            return True
    return False


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

    chron_frames: list[np.ndarray] = []
    chron_marker_sets: list[set[tuple[int, int]]] = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        markers = heads[si + 1 :]
        chron_frames.append(_render_frame(h, w, bg, actor, head, d, tname, markers))
        chron_marker_sets.append(set(markers))

    branched_chron_frames: list[np.ndarray] = []
    branch_points: list[tuple[int, int]] = []
    branch_forbidden = set(heads[1:])
    for fi, frame in enumerate(chron_frames):
        si = fi * gap
        head, d = placements[si]
        branch = _add_branch_marker(
            rng, frame, actor, head, d, step_dirs[si], tname, forbidden=branch_forbidden
        )
        if branch is None:
            return None
        branched_frame, branch_pt = branch
        branched_chron_frames.append(branched_frame)
        branch_points.append(branch_pt)
        branch_forbidden.add(branch_pt)

    if any(_has_divider_like_column(frame, bg) for frame in branched_chron_frames):
        return None

    frames = list(reversed(branched_chron_frames))
    out_frame = _render_frame(h, w, bg, actor, placements[-1][0], placements[-1][1], tname, [])
    inp = _assemble_input(frames, bg, divider)

    for fi, frame in enumerate(branched_chron_frames):
        si = fi * gap
        found = _find_agent(frame, actor)
        if found is None:
            return None
        f_tname, f_dir, f_head, _f_agent, f_markers = found
        if (f_tname, f_dir, f_head) != (tname, placements[si][1], placements[si][0]):
            return None
        expected_markers = set(chron_marker_sets[fi])
        expected_markers.add(branch_points[fi])
        if f_markers != expected_markers:
            return None

    reverse_counts = _marker_counts(frames, actor)
    if reverse_counts != sorted(reverse_counts):
        return None
    if len(set(reverse_counts)) != len(reverse_counts):
        return None
    if reverse_counts[0] >= reverse_counts[1]:
        return None
    shared_markers = _shared_markers(frames, actor)
    latest_true_markers = set(chron_marker_sets[-1])
    if shared_markers != latest_true_markers:
        return None
    if shared_markers == set(_find_agent(frames[0], actor)[4]):
        return None

    try:
        combo_out = main(inp)
    except Exception:
        return None
    if not np.array_equal(combo_out, out_frame):
        return None
    if _original_matches(inp, out_frame):
        return None
    if _leftmost_only_matches(inp, out_frame):
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
            "frame_order": "reverse_time",
            "start_frame_index": 0,
            "branch_panels": n_frames,
            "marker_counts_left_to_right": list(reverse_counts),
            "branch_points_chronological": list(branch_points),
            "original_solver_matches": False,
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
