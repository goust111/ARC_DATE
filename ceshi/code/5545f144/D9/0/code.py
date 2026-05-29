# Variant of 5545f144
# Dimension: D9 — 多步序列：帧快照策略 (multi_step)
# Change: 帧是等间隔快照——从第0步开始每隔gap步取一帧，gap = total_steps / n_frames；最后一帧有gap个剩余标记。→ 不等间隔快照

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left

# ---------------------------------------------------------------------------
# Design analysis for the D9 variant
#
# a) Concrete behavior and trace
#    - The grid still shows horizontal snapshots of one triangular sprite moving
#      through same-color single-pixel markers.
#    - Each frame still renders the current sprite plus every marker that will be
#      visited later along the path, so the marker count in a frame equals the
#      number of future movement steps remaining from that snapshot.
#    - The old task used evenly spaced snapshots, but that cadence never mattered
#      computationally because the solver could ignore earlier frames and simply
#      run the rightmost frame to exhaustion.
#    - To make D9 observable and necessary, this variant asks for the NEXT hidden
#      snapshot instead of the exhaustion state. The visible frame-to-frame marker
#      count drops are forced to be an arithmetic progression with increment +1.
#      Example: counts [10, 8, 5] imply gaps [2, 3], so the hidden next gap is 4.
#      Starting from the rightmost frame, the solver must consume exactly 4 more
#      markers and stop there.
#
# b) Complexity calibration
#    - Visible frames are restricted to 3 or 4 so every instance shows at least
#      two count differences; this keeps the arithmetic pattern learnable from a
#      single example and obvious across a small ARC training set.
#    - Base gaps vary over a small range, producing repeated support for the
#      "difference grows by one" rule without introducing many sub-cases.
#    - One extra tail marker is always left after the hidden output snapshot so
#      "simulate to the end" is guaranteed to disagree with the intended answer.
#
# c) Implicit-assumption audit for main()
#    - _split_frames assumes left-to-right chronological order and full-height
#      divider columns. The generator preserves both.
#    - _find_agent assumes the sprite template can be isolated from same-color
#      markers. The generator keeps the original body/marker separation rules and
#      re-validates recovery on every visible frame.
#    - _infer_next_gap assumes frame marker counts strictly decrease and the drops
#      form consecutive integers. The generator constructs exactly that pattern.
#    - _simulate_steps assumes the next intended marker is always the unique
#      reachable half-line choice, even when future markers beyond the output
#      snapshot remain visible. The original path legality checks still guarantee
#      this.
#
# d) Placement legality
#    - The generator remains path-first. It samples a full path of distinct heads,
#      checks no 180-degree reversals, prevents ambiguity from any past head, and
#      ensures every stamped sprite fits in bounds after one global translation.
#    - Because output stops before the path ends, the generator additionally checks
#      that the output head differs from the exhaustion head.
#
# e) Visual observability
#    - Humans can read the non-uniform cadence directly from how many markers
#      disappear between adjacent frames.
#    - The output sprite lands on an intermediate future marker rather than at the
#      very end of the chain, so earlier frames materially affect the answer.
#
# f) Visual cleanliness
#    - No new visual glyphs are added. The scene stays as sprite + isolated dots +
#      divider columns, which keeps the irregular cadence legible rather than busy.
#
# g) Color inference
#    - Colors keep the original semantics: one background, one actor color shared
#      by sprite and markers, and one divider color. No invisible color-ID logic
#      is used for ordering or for any rule branch.
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


def _simulate_steps(frame: np.ndarray, actor: int, steps: int | None) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, agent, markers = found
    agent = set((int(y), int(x)) for y, x in agent)
    markers = set((int(y), int(x)) for y, x in markers)

    if steps is None:
        target_steps = len(markers)
    else:
        target_steps = int(steps)
        assert 0 <= target_steps <= len(markers), (
            f"Requested {target_steps} steps but only {len(markers)} markers remain"
        )

    for _ in range(target_steps):
        offs = TEMPLATES[tname][d]
        choice = _choose_target(head, d, markers, offs, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} visible markers at head={head} dir={d}"
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


def _detect_actor(g: np.ndarray, bg: int) -> int:
    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    return next(iter(colors - {bg} - div_colors))


def _frame_marker_counts(frames: list[np.ndarray], actor: int) -> list[int]:
    counts: list[int] = []
    for i, frame in enumerate(frames):
        found = _find_agent(frame, actor)
        assert found is not None, f"Could not recover sprite from frame {i}"
        counts.append(len(found[-1]))
    return counts


def _infer_next_gap(frames: list[np.ndarray], actor: int) -> int:
    counts = _frame_marker_counts(frames, actor)
    assert len(counts) >= 2, "Need at least two visible frames"
    diffs = [counts[i] - counts[i + 1] for i in range(len(counts) - 1)]
    assert all(d > 0 for d in diffs), f"Marker counts must strictly decrease: {counts}"
    if len(diffs) >= 2:
        step = diffs[1] - diffs[0]
        assert step == 1, f"Expected gap increments of +1, got diffs={diffs}"
        assert all(diffs[i + 1] - diffs[i] == step for i in range(len(diffs) - 1)), (
            f"Gap pattern is inconsistent: diffs={diffs}"
        )
    return diffs[-1] + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Predict the triangular sprite's next hidden snapshot from an irregular frame sequence.

    Input structure:
    - Frames are concatenated left-to-right with solid divider columns; the background is the most common color.
    - Each frame contains the same-color triangular sprite plus all future single-pixel markers that still remain from that snapshot onward.

    Rule:
    - Count the markers in each visible frame. The drop from one frame to the next is the number of movement steps between those snapshots.
    - Those snapshot gaps are not uniform: they grow by exactly one each time, so the hidden next gap is one larger than the final visible drop.
    - Start from the rightmost visible frame and apply the original movement rule for exactly that many marker-consumption steps: at each step the head moves to the nearest reachable marker on its unique aligned half-line and the sprite re-faces that direction.
    - Output only the sprite at that next hidden snapshot.

    Key insight:
    - The earlier frames are needed only through their marker-count differences: the irregular cadence tells you how far past the last visible state to simulate.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)
    actor = _detect_actor(g, bg)
    next_gap = _infer_next_gap(frames, actor)
    return _simulate_steps(frames[-1], actor, next_gap)


# --------------- generator (D9 arithmetic-gap next-snapshot variant) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (3, 5)         # 3..4 visible frames
FRAME_MAX = 18                  # max frame height/width inclusive
H_FLOOR, H_CAP = 8, 18
W_FLOOR, W_CAP = 6, 18

# --- CFG: snapshot cadence ---
BASE_GAP_RANGE = (1, 4)         # candidate starting gaps 1..3 (filtered per n_frames)
TAIL_STEPS = 1                  # leave one extra marker after the hidden output snapshot
MAX_TOTAL_STEPS = 15

# --- CFG: path complexity ---
STEP_DIST_RANGE = (2, 7)        # per-step manhattan distances 2..6 inclusive
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 11
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 500


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
) -> tuple[int, int, list[int], int, int, int, int, int | None, str]:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    base_choices = [
        a
        for a in range(*BASE_GAP_RANGE)
        if sum(a + i for i in range(n_frames)) + TAIL_STEPS <= MAX_TOTAL_STEPS
    ]
    assert base_choices, f"No valid base gap choices for n_frames={n_frames}"
    base_gap = int(rng.choice(base_choices))
    visible_gaps = [base_gap + i for i in range(n_frames - 1)]
    total_steps = sum(base_gap + i for i in range(n_frames)) + TAIL_STEPS

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (total_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (total_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX + 1))
    w = int(rng.integers(w_lo, FRAME_MAX + 1))

    palette = rng.permutation(10)
    bg, actor = int(palette[0]), int(palette[1])
    divider = int(palette[2]) if n_frames > 1 else None

    weights = np.array(
        TEMPLATE_WEIGHTS_HEAVY if total_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    weights /= weights.sum()
    tname = str(rng.choice(list(TEMPLATES), p=weights))
    return n_frames, total_steps, visible_gaps, h, w, bg, actor, divider, tname


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


def _snapshot_steps(visible_gaps: list[int]) -> list[int]:
    steps = [0]
    cur = 0
    for gap in visible_gaps:
        cur += gap
        steps.append(cur)
    return steps


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, visible_gaps, h, w, bg, actor, divider, tname = _sample_params(rng)
    next_gap = visible_gaps[-1] + 1
    snapshot_steps = _snapshot_steps(visible_gaps)
    output_step = snapshot_steps[-1] + next_gap
    assert output_step < n_steps, "TAIL_STEPS must keep output strictly before exhaustion"

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

    frames = []
    for si in snapshot_steps:
        head, d = placements[si]
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, heads[si + 1 :]))

    ohead, odir = placements[output_step]
    out_frame = _render_frame(h, w, bg, actor, ohead, odir, tname, [])
    inp = _assemble_input(frames, bg, divider)

    counts = _frame_marker_counts(frames, actor)
    if counts != [n_steps - si for si in snapshot_steps]:
        return None
    if [counts[i] - counts[i + 1] for i in range(len(counts) - 1)] != visible_gaps:
        return None
    if _infer_next_gap(frames, actor) != next_gap:
        return None

    for fi, si in enumerate(snapshot_steps):
        found = _find_agent(frames[fi], actor)
        if found is None:
            return None
        f_tname, f_dir, f_head, _f_agent, f_markers = found
        if (f_tname, f_dir, f_head) != (tname, placements[si][1], placements[si][0]):
            return None
        if f_markers != set(heads[si + 1 :]):
            return None

    solved = main(inp)
    if not np.array_equal(solved, out_frame):
        return None

    exhaustion = _simulate_steps(frames[-1], actor, None)
    if np.array_equal(exhaustion, out_frame):
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
            "visible_gaps": list(visible_gaps),
            "next_gap": next_gap,
            "tail_steps": TAIL_STEPS,
            "snapshot_steps": list(snapshot_steps),
            "output_step": output_step,
            "turns": turns,
            "turn_ratio": turns / n_steps,
            "direction_diversity": len(set(step_dirs)),
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "step_distances": dists,
            "marker_counts": counts,
            "count_in_last_input_frame": counts[-1],
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
