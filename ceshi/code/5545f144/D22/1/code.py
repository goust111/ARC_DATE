# Variant of 5545f144
# Dimension: D22 — 消耗完成判定 (path)
# Change: 所有标记必须被消耗，模拟在标记数为0时终止；不允许有孤立不可达标记。... → 消耗N个后停止
#
# Design analysis for the D22 variant
# - concretization:
#   - Keep the original visual language: a single-color triangular sprite and
#     single-pixel markers inside each frame, with full-height divider columns.
#   - The new visible counter is the number of frames F. The rightmost frame is
#     still the starting snapshot, but the sprite now performs exactly F marker
#     consumptions from that snapshot and then stops, even when markers remain.
#   - Example trace: if the last frame contains a forced future path
#     m1 -> m2 -> m3 -> m4 -> m5 -> m6 and the input has 3 frames, the old
#     task outputs the sprite at m6 on a clean background; this variant outputs
#     the sprite at m3 and keeps markers m4, m5, m6.
# - complexity calibration:
#   - F is limited to 2..4 so the stop count is easy to read by eye.
#   - The generator forces 3..4 markers to survive after the stop, so the new
#     termination rule is always visible in the output and cannot collapse back
#     to the original "consume all" behavior.
#   - Earlier frames are still genuine path snapshots, so only D22 changes.
# - implicit assumption review:
#   - `_split_frames` assumes solid full-height dividers; the generator keeps
#     this unchanged.
#   - Actor detection assumes one actor color plus one optional divider color;
#     the generator still uses distinct palette entries for bg / actor / divider.
#   - `_find_agent` assumes all extra actor pixels are markers rather than a
#     second sprite; the generator still places exactly one sprite.
#   - `_choose_target` assumes every step has reachable markers on at most one
#     half-line and forbids 180-degree reversals; the path constructor still
#     enforces both constraints.
#   - The new `main()` assumes the last frame contains at least F remaining
#     markers because it must consume exactly F of them; the generator enforces
#     `last_frame_markers = F + leftover`, with `leftover` in 3..4.
#   - Rendering remaining markers after stopping is safe because future markers
#     are generated so they never touch any earlier sprite body.
# - placement legality:
#   - Path rollout keeps every landing head unique, disallows marker adjacency to
#     existing sprite bodies, and checks all stamped template placements can be
#     translated fully inside the frame.
#   - Each rendered snapshot uses future heads as markers, so markers never
#     overlap the sprite of the state being shown.
# - visual observability and cleanliness:
#   - The frame count is a directly visible tally and the output keeps several
#     leftover markers, making the new stop rule inferable from a few examples.
#   - No extra symbols or colors are introduced beyond bg / actor / divider, so
#     the grid stays close to the source task and remains visually clean.
# - color inference:
#   - No rule depends on numeric color ids or on sorting colors by id. Colors are
#     only used as visible roles.
# - audit of new `main()` assumptions:
#   - Loop count = number of frames.
#   - Each loop iteration assumes a unique next target exists; `_rollout_path`
#     and `_marker_compatible_with_existing` guarantee this for generated cases.
#   - The stop state may still contain markers, so the output renderer must keep
#     those pixels instead of clearing the frame.

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L


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

    dirs_with_candidates = {dd for _, dd, _ in reachable}
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


def _simulate_path(frame: np.ndarray, actor: int, stop_after: int | None) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, agent, markers = found
    agent = {(int(y), int(x)) for y, x in agent}
    markers = {(int(y), int(x)) for y, x in markers}

    if stop_after is not None:
        assert len(markers) >= stop_after, (
            f"need at least {stop_after} markers in last frame, found {len(markers)}"
        )

    max_steps = h * w * 4
    consumed = 0
    for _ in range(max_steps):
        if stop_after is not None and consumed >= stop_after:
            break
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
        consumed += 1

    out = np.full((h, w), bg, dtype=int)
    for y, x in markers:
        out[y, x] = actor
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a triangular sprite along its forced marker path for a frame-count quota.

    Input structure:
    - Several same-height frames are concatenated left-to-right and separated by
      solid divider columns; the most common color is background.
    - Each frame contains one triangular sprite plus isolated one-pixel markers,
      all drawn in the same actor color.

    Rule:
    - Read the rightmost frame as the current state and let N be the number of
      frames in the input.
    - Repeatedly follow the unique reachable marker half-line from the sprite
      head, jumping to the nearest marker and re-facing that direction.
    - After exactly N marker consumptions, stop even if markers remain, and
      output the stopped sprite together with every still-unconsumed marker.

    Key insight:
    - The triangle tip still reveals the current facing, but the stopping count
      is no longer "until empty" — it is encoded by how many frames are shown.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    start = frames[-1]
    stop_after = len(frames)
    return _simulate_path(start, actor, stop_after=stop_after)


# --------------- generator (D22 variant: frame-count stop quota) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (2, 5)          # 2..4 frames; frame count is the stop quota
FRAME_MAX = 18                   # rng.integers high for H, W -> 8..17 / 6..17
H_FLOOR, H_CAP = 8, 17
W_FLOOR, W_CAP = 6, 17

# --- CFG: path cadence ---
SNAP_GAP_RANGE = (1, 3)          # 1..2 steps between consecutive input snapshots
LEFTOVER_RANGE = (3, 5)          # 3..4 markers survive after the forced stop

# --- CFG: path complexity ---
STEP_DIST_RANGE = (2, 6)         # 2..5 inclusive
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 11
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 400


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
    """Interval of translations that keep every placement's template in-grid."""
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
) -> tuple[int, int, int, int, int, int, int, int | None, str, int, int]:
    """Sample path/snapshot parameters plus frame geometry and colors."""
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    stop_after = n_frames
    snap_gap = int(rng.integers(*SNAP_GAP_RANGE))
    leftover = int(rng.integers(*LEFTOVER_RANGE))
    tail_steps = stop_after + leftover
    n_steps = (n_frames - 1) * snap_gap + tail_steps

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = rng.permutation(10)
    bg, actor, divider = int(palette[0]), int(palette[1]), int(palette[2])

    weights = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    weights /= weights.sum()
    tname = str(rng.choice(list(TEMPLATES), p=weights))
    return n_frames, stop_after, snap_gap, tail_steps, h, w, bg, actor, divider, tname, n_steps, leftover


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
    """
    Validate a candidate future marker against all already-fixed path states.

    This preserves the source task's "forced next half-line" invariant and also
    keeps future markers visually detached from any earlier sprite body.
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


def _rollout_path(
    rng: np.random.Generator, tname: str, step_dirs: list[int], h: int, w: int
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]] | None:
    """Greedy path rollout. Returns (heads, state_dirs, fit_placements) or None."""
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


def _build_case(rng: np.random.Generator) -> dict | None:
    (
        n_frames,
        stop_after,
        snap_gap,
        tail_steps,
        h,
        w,
        bg,
        actor,
        divider,
        tname,
        n_steps,
        leftover,
    ) = _sample_params(rng)
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

    frames: list[np.ndarray] = []
    for fi in range(n_frames):
        si = fi * snap_gap
        head, d = placements[si]
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, heads[si + 1 :]))

    last_idx = (n_frames - 1) * snap_gap
    stop_idx = last_idx + stop_after
    stop_head, stop_dir = placements[stop_idx]
    remaining_markers = heads[stop_idx + 1 :]
    out_frame = _render_frame(h, w, bg, actor, stop_head, stop_dir, tname, remaining_markers)
    inp = _assemble_input(frames, bg, divider)

    found_last = _find_agent(frames[-1], actor)
    if found_last is None:
        return None
    f_tname, f_dir, f_head, _f_agent, f_markers = found_last
    if (f_tname, f_dir, f_head) != (tname, placements[last_idx][1], placements[last_idx][0]):
        return None
    if f_markers != set(heads[last_idx + 1 :]):
        return None

    found_out = _find_agent(out_frame, actor)
    if found_out is None:
        return None
    o_tname, o_dir, o_head, _o_agent, o_markers = found_out
    if (o_tname, o_dir, o_head) != (tname, stop_dir, stop_head):
        return None
    if o_markers != set(remaining_markers):
        return None

    solver_out = main(inp)
    if not np.array_equal(solver_out, out_frame):
        return None

    full_out = _simulate_path(frames[-1], actor, stop_after=None)
    if np.array_equal(full_out, out_frame):
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
            "snapshot_gap": snap_gap,
            "stop_after": stop_after,
            "last_frame_markers": tail_steps,
            "leftover_after_stop": leftover,
            "turns": turns,
            "turn_ratio": turns / n_steps,
            "direction_diversity": len(set(step_dirs)),
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "step_distances": dists,
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
