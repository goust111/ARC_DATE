# Combo variant of 5545f144 (index 17)
# combo_spec: D8/0+D9/2+D20/2
# Name: 垂直帧拼接+倒序帧+无显式分隔线

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left


# Design analysis for the fused combo
# a) Concrete behavior and trace:
#    The base task shows equal-cadence snapshots of one triangular sprite and its
#    remaining one-pixel markers. This combo changes three linked parts at once:
#    the snapshots are stacked vertically, the stack runs in reverse time, and
#    no divider rows are drawn between frames. A 3-frame example that used to be
#    [step0 | step3 | step6] becomes a tight top-to-bottom strip
#    [step6 | step3 | step0]. The reverse-time cue is purely structural:
#    marker counts increase from top to bottom in a fixed arithmetic progression,
#    so the sparsest top slice is visibly the latest snapshot.
# b) Complexity calibration:
#    The generator always emits 2..4 frames. Allowing one frame would erase both
#    the vertical-stacking change and the reverse-order clue. Equal-cadence
#    snapshots keep that progression obvious without introducing extra objects.
# c) Hidden assumptions broken from the original main():
#    - Horizontal divider splitting is invalid: there are no separators at all.
#    - "Latest frame is the rightmost one" is invalid: time now runs top-down in
#      reverse, so the latest usable state is the sparsest slice near the top.
#    - Actor detection can no longer subtract divider colors because the combo
#      intentionally removes divider rows/columns.
# d) Placement legality:
#    All path markers remain pairwise isolated and non-touching to the sprite,
#    as in the base task, so each hidden frame stays legible.
# e) Visual observability:
#    Humans can spot repeated same-width panels in a tall strip because the actor
#    template reappears every frame and marker counts grow from top to bottom at
#    a constant rate.
#    The lack of divider rows is therefore visible but still inferable from the
#    repeated panel rhythm. The top panel is consistently the most resolved one.
# f) Visual tidiness:
#    No new colors are introduced. Every panel still contains one sprite plus
#    sparse single-pixel markers, so the canvas stays readable despite the
#    missing separators.
# g) Change inferability:
#    The only non-background color is still the actor color used by both sprite
#    and markers. The new reasoning burden is structural: infer equal-height
#    vertical frames, notice reverse-time marker growth, and choose the top
#    slice as the latest snapshot before simulating.
# h) New solver assumptions and generator guarantees:
#    - Assumption: some equal-height split into 2..4 vertical slices exists where
#      every slice contains the same sprite template and isolated singleton
#      leftovers. Guarantee: `_build_case` constructs exactly that.
#    - Assumption: in the intended split, leftover-marker counts from top to
#      bottom are [gap, 2*gap, ..., n_frames*gap]. Guarantee: `_build_case`
#      samples equal-cadence snapshots and reverses their order before stacking.
# i) Anti-degeneracy note:
#    Reverse ordering does not alter the motion rule itself, so the examples make
#    the ordering legible through marker-count progression rather than extra
#    hidden objects.


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


def _touching(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


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


def _reachable_targets(
    head: tuple[int, int],
    markers: set[tuple[int, int]],
    offs: list[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int, tuple[int, int]]]:
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

    reachable: list[tuple[int, int, tuple[int, int]]] = []
    for y, x in markers:
        if x == hx and y < hy and fits((y, x)):
            reachable.append((hy - y, 0, (y, x)))
        elif y == hy and x > hx and fits((y, x)):
            reachable.append((x - hx, 1, (y, x)))
        elif x == hx and y > hy and fits((y, x)):
            reachable.append((y - hy, 2, (y, x)))
        elif y == hy and x < hx and fits((y, x)):
            reachable.append((hx - x, 3, (y, x)))
    reachable.sort()
    return reachable


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    offs: list[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    reachable = _reachable_targets(head, markers, offs, h, w)
    if not reachable:
        return None

    dirs_with_candidates = {dd for _, dd, _ in reachable}
    assert len(dirs_with_candidates) == 1, (
        f"head={head} dir={d}: markers reachable in multiple directions "
        f"{dirs_with_candidates}, expected exactly one"
    )

    _, dd, pt = reachable[0]
    assert dd != (d + 2) % 4, (
        f"head={head} dir={d}: would reverse to dir={dd}, not allowed"
    )
    return dd, pt


def _markers_are_isolated(
    markers: set[tuple[int, int]], agent: set[tuple[int, int]]
) -> bool:
    marker_list = list(markers)
    for i, pt in enumerate(marker_list):
        if any(_touching(pt, cell) for cell in agent):
            return False
        for other in marker_list[i + 1:]:
            if _touching(pt, other):
                return False
    return True


def _frame_profile(frame: np.ndarray, actor: int) -> dict | None:
    found = _find_agent(frame, actor)
    if found is None:
        return None
    tname, d, head, agent, markers = found
    if not _markers_are_isolated(markers, agent):
        return None
    return {
        "tname": tname,
        "dir": d,
        "head": head,
        "agent": agent,
        "markers": markers,
    }


def _valid_segmentations(g: np.ndarray, actor: int) -> list[dict]:
    total_h, w = g.shape
    candidates: list[dict] = []
    for n_frames in range(2, 5):
        if total_h % n_frames != 0:
            continue
        frame_h = total_h // n_frames
        frames = [g[i * frame_h:(i + 1) * frame_h, :].copy() for i in range(n_frames)]
        profiles = [_frame_profile(frame, actor) for frame in frames]
        if any(profile is None for profile in profiles):
            continue

        profiles = list(profiles)
        tnames = {profile["tname"] for profile in profiles}
        if len(tnames) != 1:
            continue

        counts = [len(profile["markers"]) for profile in profiles]
        if len(set(counts)) != n_frames:
            continue
        gap = counts[0]
        if gap <= 0:
            continue
        if counts != [gap * (i + 1) for i in range(n_frames)]:
            continue

        candidates.append(
            {
                "n_frames": n_frames,
                "frame_h": frame_h,
                "frames": frames,
                "profiles": profiles,
                "counts": counts,
                "gap": gap,
                "width": w,
            }
        )
    return candidates


def _split_frames(g: np.ndarray, actor: int) -> tuple[list[np.ndarray], dict | None]:
    candidates = _valid_segmentations(g, actor)
    if candidates:
        candidates.sort(key=lambda item: (item["counts"][0], -item["n_frames"]))
        best = candidates[0]
        return best["frames"], best
    return [g.copy()], None


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue a triangular sprite from the latest hidden snapshot in a tight vertical strip until every marker is consumed.

    Input structure:
    - Several equal-width frames are stacked directly top-to-bottom with no divider rows; the background is the most common color.
    - Every frame contains the same triangular sprite template and actor-colored singleton dots.
    - Time is reversed: the top slice is the latest snapshot and lower slices are older.

    Rule:
    - Infer the frame height by finding a 2..4-way vertical split where every slice parses into the same sprite template and isolated leftover dots.
    - In the intended split, marker counts from top to bottom form an arithmetic progression `[gap, 2*gap, ..., n*gap]`, revealing equal-cadence snapshots in reverse order.
    - Use the top slice as the actual starting state.
    - From that frame, advance the sprite to the nearest reachable marker, consume it, and reorient to the movement direction.
    - Repeat until no markers remain, then output one clean frame with only the final sprite.

    Key insight: The hard part is not the motion rule but discovering the hidden panel boundaries and noticing that the sparsest top slice is the latest state because time runs downward in reverse.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    colors = sorted(set(map(int, g.ravel())) - {bg})
    assert colors, "expected a non-background actor color"
    actor = colors[0]

    frames, _ = _split_frames(g, actor)
    start = frames[0]
    return _simulate_to_end(start, actor)


# --------------- generator (fused vertical-reverse-no-divider variant) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (2, 5)          # 2..4 frames; the fusion is invisible at 1 frame
FRAME_MAX = 16                   # rng.integers high for H, W -> 8..15 / 6..15
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

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 4000

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(2, 5)) for n in range(*STEPS_RANGE)}


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
) -> tuple[int, int, int, int, int, int, int, str]:
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

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames, n_steps, gap, h, w, bg, actor, tname


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
    for marker in markers:
        frame[marker] = actor
    return frame


def _assemble_input(frames: list[np.ndarray], bg: int) -> np.ndarray:
    n_frames = len(frames)
    h, w = frames[0].shape
    inp = np.full((n_frames * h, w), bg, dtype=int)
    for i, frame in enumerate(frames):
        paste(inp, frame, i * h, 0)
    return inp


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, gap, h, w, bg, actor, tname = _sample_params(rng)
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
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        chron_frames.append(_render_frame(h, w, bg, actor, head, d, tname, heads[si + 1:]))

    frames = list(reversed(chron_frames))
    reverse_counts: list[int] = []
    for frame in frames:
        profile = _frame_profile(frame, actor)
        if profile is None:
            return None
        reverse_counts.append(len(profile["markers"]))

    if reverse_counts != [gap * (i + 1) for i in range(n_frames)]:
        return None

    fhead, fdir = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, fhead, fdir, tname, [])
    inp = _assemble_input(frames, bg)

    candidates = _valid_segmentations(inp, actor)
    if len(candidates) != 1:
        return None
    inferred = candidates[0]
    if inferred["n_frames"] != n_frames or inferred["frame_h"] != h:
        return None
    if inferred["counts"] != reverse_counts or inferred["gap"] != gap:
        return None

    for fi, frame in enumerate(chron_frames):
        si = fi * gap
        found = _find_agent(frame, actor)
        if found is None:
            return None
        f_tname, f_dir, f_head, _agent, f_markers = found
        if (f_tname, f_dir, f_head) != (tname, placements[si][1], placements[si][0]):
            return None
        if f_markers != set(heads[si + 1:]):
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
            "total_steps": n_steps,
            "gap": gap,
            "turns": turns,
            "turn_ratio": turns / n_steps,
            "direction_diversity": len(set(step_dirs)),
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "step_distances": dists,
            "frame_order": "reverse_time_vertical",
            "has_dividers": False,
            "marker_counts_top_to_bottom": list(reverse_counts),
            "inferred_frame_h": inferred["frame_h"],
            "inferred_n_frames": inferred["n_frames"],
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
