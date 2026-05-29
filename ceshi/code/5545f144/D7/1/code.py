# Variant of 5545f144
# Dimension: D7 — 变换：精灵朝向更新 (transform)
# Change: 消耗标记后精灵重新朝向该步的移动方向，模板按新方向旋转重新戳印。... → 朝向为下一个标记方向

import importlib.util
import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left

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

# Design analysis for the D7 variant
# a) Concrete behavior:
#    A state still contains one arrow sprite plus same-color single-pixel markers.
#    The head moves to the nearest marker on its facing half-line, but once it lands
#    the stamped orientation becomes a lookahead cue: it points toward the next
#    marker that will be eaten from the new head position.
#    Example with heads H0 -> H1 -> H2 and a 90-degree turn at H1:
#    - original state at H1 faces H0->H1;
#    - variant state at H1 faces H1->H2.
#    The consumed marker set is unchanged; only which rotated template is visible at
#    intermediate snapshots changes.
# b) Complexity calibration:
#    If n_frames were allowed to be 1, the only input snapshot would be the initial
#    state, where original and variant orientations coincide. To keep D7 observable,
#    generation requires 2..4 frames and also forces each snapshot boundary after the
#    first to occur right after a turn, so the displayed arrow differs from the
#    original rule on visible states.
# c) Hidden assumptions in the old solver:
#    - _simulate_to_end previously set d = moved_direction after landing. That is
#      exactly the old invariant and must change.
#    - The old _choose_target tolerated facing as a soft hint and only rejected 180°
#      reversals. In the variant, the current arrow already points at the immediate
#      next marker, so the chosen direction must equal the detected facing.
#    - The old generator's state_dirs duplicated the previous step direction at each
#      landing state. The new generator must shift those directions forward by one
#      step, with the final state falling back to the last travel direction because
#      there is no later marker to preview.
# d) Placement legality:
#    We still forbid overlap, touching-to-body ambiguity, out-of-bounds stamping, and
#    multi-half-line ambiguity. For every step we now reserve space for two stamps at
#    the landing head: the translated pre-turn sprite and the lookahead-facing sprite.
# e) Visual observability:
#    The variant uses only visible geometry. No color ids or invisible metadata drive
#    the rule. Every later input frame shows a turned arrow whose head location is the
#    same as before, but whose orientation previews the next leg.
# f) Visual cleanliness:
#    Markers remain isolated single pixels, dividers remain solid full-height columns,
#    and only one sprite is present per frame. This keeps the rightmost frame easy to
#    parse even though sprite and markers share one color.
# g) Color inference:
#    No new color logic is introduced; the actor and markers still share one color and
#    the divider stays a separate solid color when present.


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


def _aligned_candidates(
    head: tuple[int, int], markers: set[tuple[int, int]], dir_idx: int
) -> list[tuple[int, tuple[int, int]]]:
    hy, hx = head
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


def _shape_fits_at(
    head: tuple[int, int], offs: list[tuple[int, int]], h: int, w: int
) -> bool:
    hy, hx = head
    for oy, ox in offs:
        ny = hy + oy
        nx = hx + ox
        if not (0 <= ny < h and 0 <= nx < w):
            return False
    return True


def _translated_shape_fits(
    head: tuple[int, int],
    target: tuple[int, int],
    offs: list[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    hy, hx = head
    ty, tx = target
    dy, dx = ty - hy, tx - hx
    for oy, ox in offs:
        ny = hy + oy + dy
        nx = hx + ox + dx
        if not (0 <= ny < h and 0 <= nx < w):
            return False
    return True


def _unique_marker_direction(head: tuple[int, int], markers: set[tuple[int, int]]) -> tuple[int, tuple[int, int]] | None:
    reachable: list[tuple[int, int, tuple[int, int]]] = []
    for dd in range(4):
        for dist, pt in _aligned_candidates(head, markers, dd):
            reachable.append((dist, dd, pt))

    if not reachable:
        return None

    dirs_with_candidates = {dd for _, dd, _ in reachable}
    assert len(dirs_with_candidates) == 1, (
        f"head={head}: markers visible on multiple half-lines {dirs_with_candidates}"
    )

    reachable.sort()
    _, dd, pt = reachable[0]
    return dd, pt


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    offs: list[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    choice = _unique_marker_direction(head, markers)
    if choice is None:
        return None

    dd, pt = choice
    assert dd == d, f"head={head}: facing {d} does not point to the next marker direction {dd}"
    assert _translated_shape_fits(head, pt, offs, h, w), (
        f"head={head} dir={d}: translated sprite would leave the frame at target={pt}"
    )
    return dd, pt


def _lookahead_direction(
    head: tuple[int, int],
    move_dir: int,
    markers: set[tuple[int, int]],
    tname: str,
    h: int,
    w: int,
) -> int:
    if not markers:
        return move_dir

    choice = _unique_marker_direction(head, markers)
    assert choice is not None
    next_dir, _ = choice
    assert next_dir != (move_dir + 2) % 4, (
        f"head={head} dir={move_dir}: lookahead would reverse to {next_dir}"
    )
    assert _shape_fits_at(head, TEMPLATES[tname][next_dir], h, w), (
        f"head={head}: lookahead-facing sprite {next_dir} would not fit in-frame"
    )
    return next_dir


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

        offs = TEMPLATES[tname][d]
        move_dir, target = _choose_target(head, d, markers, offs, h, w)
        assert target is not None, f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"

        head = target
        markers.discard(head)
        d = _lookahead_direction(head, move_dir, markers, tname, h, w)
        agent = {(head[0] + oy, head[1] + ox) for oy, ox in TEMPLATES[tname][d]}

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue the arrow-sprite simulation from the last snapshot and render the end state.

    Input structure:
    - Several equally sized frames may be concatenated left-to-right, with optional
      full-height divider columns in a third color.
    - Inside each frame, one actor color is reused for both the triangular sprite and
      the remaining single-pixel markers.

    Rule:
    - Read only the rightmost frame as the current state.
    - The sprite head advances to the nearest marker on the half-line indicated by its
      current facing, and that landing marker disappears.
    - After each landing, the sprite does not keep the travel direction unless that was
      also the next target direction; instead it immediately rotates to face the next
      remaining marker that will be consumed from the new head position.
    - When no markers remain, output just the final sprite on clean background.

    Key insight: Each visible orientation is a one-step preview of the future path, so
    solving requires consuming the current target and then recomputing the facing from
    the updated marker set.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    start = frames[-1]
    return _simulate_to_end(start, actor)


# --------------- generator ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (2, 5)          # 2..4 frames; 1 frame would hide the D7 change
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

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 600

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(2, 5)) for n in range(*STEPS_RANGE)}


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
) -> tuple[int, int, int, int, int, int, int, int | None, str]:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0]
    weights = np.array([(n ** 3) / max(1, _STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = rng.permutation(10)
    bg, actor = int(palette[0]), int(palette[1])
    divider = int(palette[2])

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


def _state_dirs_from_steps(step_dirs: list[int]) -> list[int]:
    return list(step_dirs) + [step_dirs[-1]]


def _marker_compatible_with_existing(
    nxt: tuple[int, int],
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
) -> bool:
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, state_dirs[j])
        if any(_touching(nxt, cell) for cell in sprite):
            return False
        rel_dir = _direction(old_head, nxt)
        if rel_dir is None:
            continue
        if rel_dir != state_dirs[j]:
            return False
        if j < len(heads) - 1 and _dist(old_head, nxt) <= _dist(old_head, heads[j + 1]):
            return False
    return True


def _rollout_path(
    rng: np.random.Generator, tname: str, step_dirs: list[int], h: int, w: int
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]] | None:
    n_steps = len(step_dirs)
    state_dirs = _state_dirs_from_steps(step_dirs)
    heads: list[tuple[int, int]] = [(0, 0)]
    fit_placements: list[tuple[tuple[int, int], int]] = [((0, 0), state_dirs[0])]

    for i in range(n_steps):
        cur = heads[-1]
        cur_dir = state_dirs[i]
        move_dir = step_dirs[i]
        assert cur_dir == move_dir

        dr, dc = DIRS4[move_dir]
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
            post_dir = state_dirs[i + 1]
            trial_fit = fit_placements + [(nxt, move_dir), (nxt, post_dir)]
            if _shift_interval(trial_fit, h, w, tname) is None:
                continue
            chosen = nxt
            break

        if chosen is None:
            return None

        heads.append(chosen)
        fit_placements.extend([(chosen, move_dir), (chosen, state_dirs[i + 1])])

    return heads, state_dirs, fit_placements


def _validate_path(
    step_dirs: list[int], heads: list[tuple[int, int]], gap: int, n_frames: int
) -> tuple[bool, int, list[int], int]:
    n_steps = len(step_dirs)
    dir_set = set(step_dirs)
    if len(dir_set) < MIN_DIR_DIVERSITY:
        return False, 0, [], 0
    if not (dir_set & {0, 2} and dir_set & {1, 3}):
        return False, 0, [], 0
    turns = sum(step_dirs[i] != step_dirs[i - 1] for i in range(1, n_steps))
    if turns < MIN_TURNS_ACCEPT or not (TURN_RATIO_LO <= turns / n_steps <= TURN_RATIO_HI):
        return False, turns, [], 0
    dists = [_dist(heads[i], heads[i + 1]) for i in range(n_steps)]
    if len(set(dists)) < MIN_DIST_VARIETY:
        return False, turns, dists, 0

    boundary_turns = 0
    for fi in range(1, n_frames):
        si = fi * gap
        if si >= n_steps:
            return False, turns, dists, boundary_turns
        if step_dirs[si] == step_dirs[si - 1]:
            return False, turns, dists, boundary_turns
        boundary_turns += 1
    return True, turns, dists, boundary_turns


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
    n_frames, n_steps, gap, h, w, bg, actor, divider, tname = _sample_params(rng)
    step_dirs = _make_step_dirs(rng, n_steps)

    rollout = _rollout_path(rng, tname, step_dirs, h, w)
    if rollout is None:
        return None
    heads, state_dirs, fit_placements = rollout

    ok, turns, dists, boundary_turns = _validate_path(step_dirs, heads, gap, n_frames)
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
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, heads[si + 1:]))

    fhead, fdir = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, fhead, fdir, tname, [])
    inp = _assemble_input(frames, bg, divider)

    last_idx = (n_frames - 1) * gap
    found = _find_agent(frames[-1], actor)
    if found is None:
        return None
    f_tname, f_dir, f_head, _f_agent, f_markers = found
    if (f_tname, f_dir, f_head) != (tname, placements[last_idx][1], placements[last_idx][0]):
        return None
    if f_markers != set(heads[last_idx + 1:]):
        return None
    if not np.array_equal(main(inp), out_frame):
        return None
    if _matches_original(inp, out_frame):
        return None

    old_like_state_dirs = [step_dirs[0]] + list(step_dirs)
    visible_diff_frames = sum(
        state_dirs[fi * gap] != old_like_state_dirs[fi * gap] for fi in range(1, n_frames)
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
            "turns": turns,
            "turn_ratio": turns / n_steps,
            "direction_diversity": len(set(step_dirs)),
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "step_distances": dists,
            "count_in_first_frame": n_steps,
            "count_in_last_input_frame": gap,
            "boundary_turns": boundary_turns,
            "visible_diff_frames": visible_diff_frames,
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
