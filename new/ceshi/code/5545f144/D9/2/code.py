# Variant of 5545f144
# Dimension: D9 — 多步序列：帧快照策略 (multi_step)
# Change: 帧是等间隔快照——从第0步开始每隔gap步取一帧，gap = total_steps / n_frames；最后一帧有gap个剩余标记 → 倒序帧（最新输入帧在左，初始帧在右）

import importlib.util
import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

# Design analysis for the D9 reverse-frame variant
# ------------------------------------------------
# a) Concrete behavior and trace
# - The objects stay the same: one arrow-like triangle plus single-pixel markers
#   in the same actor color, with full-height divider columns between equal-sized
#   frames.
# - The only intended dimension shift is temporal presentation. Base task order is:
#   [step 0][step g][step 2g]...[step (F-1)g]. This variant reverses the strip to:
#   [step (F-1)g]...[step 2g][step g][step 0].
# - One minimal support change is necessary to keep that reversal operative: every
#   older panel may contain one extra branch marker that has not yet been resolved
#   in the coarse snapshot strip. The latest panel stays locally deterministic.
# - Small trace: if F=3 and g=2, the chronological snapshots contain marker counts
#   [6, 4, 2]. In the variant input they appear left-to-right as [2, 5, 7] after
#   restoring one historical branch marker into the older panels. The old solver
#   would wrongly start from the rightmost frame and hit a multi-direction ambiguity,
#   while the variant solver must detect that the sparsest-marker frame is the
#   latest usable snapshot and continue from there.
# - The output is unchanged: simulate forward from the latest input snapshot until
#   every marker is consumed, then render only the final sprite body.
#
# b) Complexity calibration
# - To keep the D9 change visible, the generator now always uses 2-4 frames.
#   Allowing a single frame would erase the reversal entirely.
# - Marker counts still grow monotonically from left to right, so the strip provides
#   repeated evidence of reverse time without adding any new object type or color
#   mapping.
# - The path, turns, templates, and spacing rules remain untouched to avoid adding
#   unrelated degrees of freedom; the only added visual element is a single branch
#   marker in non-latest panels so the reversed starting point actually matters.
#
# c) Implicit-assumption audit of the original main()
# - `_split_frames` assumes horizontal frames separated by solid divider columns.
#   This remains valid.
# - Actor-color extraction assumes exactly one non-bg, non-divider color. This
#   remains valid.
# - `start = frames[-1]` relied on "latest snapshot is rightmost". That is the
#   broken assumption under this variant and is the main solver change.
# - A new solver that scans all frames implicitly assumes `_find_agent` works on
#   every frame, not only the latest one. The generator therefore now validates
#   every frame, not just one endpoint, even after branch markers are added.
#
# d) Placement legality
# - Marker placement and sprite stamping keep the original no-overlap and no-clipping
#   constraints.
# - Unique-half-line determinism is only guaranteed in the latest frame, because
#   that is the only frame the solver is meant to continue from. Older frames are
#   allowed one extra reachable branch marker, but they must still be independently
#   parseable by `_find_agent`.
#
# e) Visual observability
# - Humans can see the reverse ordering because the number of remaining markers
#   grows from left to right, and the right-side panels are visibly less resolved
#   than the leftmost panel. This is directly visible in 2-4 examples without
#   relying on color ids or hidden metadata.
#
# f) Visual cleanliness
# - No new overlays are introduced. Each panel still contains one sprite and sparse
#   markers, so reversing the strip does not make the canvas denser or noisier.
#
# g) Color reasoning
# - No color remapping is added. Background, actor, and divider roles stay exactly
#   the same, so the new inference burden is purely temporal rather than chromatic.

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left
DISTRACTOR_COLOR = 8

_ORIGINAL_5545F144 = None


def _matches_original(inp: np.ndarray, out: np.ndarray) -> bool:
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

    # Collect all reachable markers (on any half-line, fitting in grid).
    reachable: list[tuple[int, int, tuple[int, int]]] = []  # (dist, dir, target)
    for dd in range(4):
        for dist, pt in candidates(dd):
            if fits(pt):
                reachable.append((dist, dd, pt))

    if not reachable:
        return None

    # At most one direction should have candidates.
    dirs_with_candidates = set(dd for _, dd, _ in reachable)
    assert len(dirs_with_candidates) == 1, (
        f"head={head} dir={d}: markers reachable in multiple directions "
        f"{dirs_with_candidates}, expected exactly one"
    )

    # Pick the nearest on that direction.
    reachable.sort()
    _, dd, pt = reachable[0]

    # No 180° reversals observed in eval.
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


def _latest_frame_index(frames: list[np.ndarray], actor: int) -> int:
    """Pick the latest snapshot as the detectable frame with the fewest markers."""
    scored: list[tuple[int, int]] = []
    for idx, frame in enumerate(frames):
        found = _find_agent(frame, actor)
        if found is None:
            continue
        scored.append((len(found[4]), idx))

    assert scored, "No frame contains a detectable sprite"
    scored.sort()
    return scored[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue an arrow sprite from the latest usable reversed snapshot until all markers vanish.

    Input structure:
    - Equal-sized frames are concatenated horizontally and may be separated by full-height divider columns.
    - Every frame shows the same triangular sprite template plus remaining single-pixel markers in one shared actor color.
    - The snapshot cadence is uniform, but the strip is time-reversed: the latest input snapshot is on the left and the earliest is on the right.
    - Older panels can still contain one extra branch marker, so they are historical context rather than safe forward-start states.

    Rule:
    - Identify the latest usable snapshot by finding the frame with the fewest leftover markers.
    - From that frame, the head can see reachable markers on exactly one cardinal half-line; move to the nearest marker on that line.
    - Landing on a marker consumes it and reorients the sprite to the travel direction.
    - Keep advancing until no markers remain, then output only the final sprite on background.

    Key insight: The movie strip runs right-to-left in time, and only the sparsest panel is fully resolved enough to continue deterministically.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    # Actor color is the non-bg, non-divider color.
    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    actor = _detect_actor(frames, bg, div_colors)

    # Reverse-order strips place the latest state in the sparsest-marker frame.
    start = frames[_latest_frame_index(frames, actor)]
    return _simulate_to_end(start, actor)

# --------------- generator (codex gpt-5.4 (retry, turn-sign randomized)) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (2, 5)          # rng.integers low/high+1 -> 2..4 frames
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

# --- CFG: template selection ---
# Weights applied when n_steps >= TEMPLATE_HEAVY_THRESHOLD, in TEMPLATES order:
# (hollow3, filled4, stem4, midfilled6). Otherwise uniform.
TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 3000

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
    if ac == bc and br < ar: return 0
    if ar == br and bc > ac: return 1
    if ac == bc and br > ar: return 2
    if ar == br and bc < ac: return 3
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
) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Add one extra reachable marker on a different half-line to older panels."""
    h, w = frame.shape
    occupied = set(map(tuple, np.argwhere(frame == actor)))
    dirs = [dd for dd in range(4) if dd != true_dir]
    dirs = [dirs[i] for i in rng.permutation(len(dirs))]
    dist_choices = [int(v) for v in rng.permutation(np.arange(2, max(h, w)))]

    for dd in dirs:
        dr, dc = DIRS4[dd]
        for dist in dist_choices:
            pt = (head[0] + dist * dr, head[1] + dist * dc)
            if not (0 <= pt[0] < h and 0 <= pt[1] < w):
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


def _sample_params(rng: np.random.Generator) -> tuple[int, int, int, int, int, int, int, int | None, str]:
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

    palette = [int(c) for c in rng.permutation(10) if int(c) != DISTRACTOR_COLOR]
    bg, actor = palette[0], palette[1]
    divider = palette[2] if n_frames > 1 else None

    tw = np.array(TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1), dtype=float)
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
            return None  # 180 reversal — caller should resample step_dirs

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
    """Validate candidate marker against all previously-placed heads.

    Ensures: no sprite-body adjacency, no alternative half-line from a past head
    (would create ambiguity at that step), and no nearer marker on that half-line
    (would be consumed first).
    """
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
    h: int, w: int, bg: int, actor: int,
    head: tuple[int, int], d: int, tname: str,
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


def _add_distractors(
    rng: np.random.Generator, frame: np.ndarray, bg: int, color: int, count: int = 1
) -> np.ndarray | None:
    out = frame.copy()
    occupied = set(map(tuple, np.argwhere(frame != bg)))
    candidates = [
        (r, c)
        for r in range(frame.shape[0])
        for c in range(frame.shape[1])
        if frame[r, c] == bg and not any(_touching((r, c), cell) for cell in occupied)
    ]
    if len(candidates) < count:
        return None
    picks = rng.choice(len(candidates), size=count, replace=False)
    for idx in np.atleast_1d(picks):
        r, c = candidates[int(idx)]
        out[r, c] = color
    return out


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

    # Build chronological snapshots first, then reverse them for the variant.
    chron_frames = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        chron_frames.append(_render_frame(h, w, bg, actor, head, d, tname, heads[si + 1:]))
    frames = list(reversed(chron_frames))
    noisy_frames = []
    for frame in frames:
        noisy = _add_distractors(rng, frame, bg, DISTRACTOR_COLOR, count=1)
        if noisy is None:
            return None
        noisy_frames.append(noisy)
    frames = noisy_frames
    # Final output: just the sprite body, no markers.
    fhead, fdir = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, fhead, fdir, tname, [])
    inp = _assemble_input(frames, bg, divider)

    # Oracle checks: every frame must be parseable, reversed marker counts must
    # increase left-to-right, the latest snapshot must be identifiable, and the
    # solver must still land on the correct final sprite.
    for fi, frame in enumerate(chron_frames):
        si = fi * gap
        found = _find_agent(frame, actor)
        if found is None:
            return None
        f_tname, f_dir, f_head, _f_agent, f_markers = found
        if (f_tname, f_dir, f_head) != (tname, placements[si][1], placements[si][0]):
            return None
        if f_markers != set(heads[si + 1:]):
            return None

    reverse_counts = [len(_find_agent(frame, actor)[4]) for frame in frames]
    if reverse_counts != sorted(reverse_counts):
        return None
    if len(set(reverse_counts)) != len(reverse_counts):
        return None
    if _latest_frame_index(frames, actor) != 0:
        return None
    if not np.array_equal(main(inp), out_frame):
        return None

    return {
        "input": inp,
        "output": out_frame,
        "meta": {
            "frame_height": h, "frame_width": w, "n_frames": n_frames,
            "template": tname, "bg": bg, "actor": actor, "divider": divider,
            "total_steps": n_steps, "gap": gap,
            "turns": turns, "turn_ratio": turns / n_steps,
            "direction_diversity": len(set(step_dirs)),
            "step_dirs": list(step_dirs), "state_dirs": list(state_dirs),
            "heads": list(heads), "step_distances": dists,
            "frame_order": "reverse_time",
            "marker_counts_left_to_right": list(reverse_counts),
            "count_in_leftmost_frame": gap,
            "count_in_rightmost_frame": n_steps,
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
