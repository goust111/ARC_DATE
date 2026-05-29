# Variant of 5545f144
# Dimension: D6 — 轨迹：移动痕迹 (trajectory)
# Change: 精灵移动无轨迹——旧位置恢复为背景色，只保留当前位置的精灵体。→ 轨迹编码移动方向：
# 向上=红(2)、向右=蓝(1)、向下=绿(3)、向左=黄(4)；输入快照与输出都会保留这条轨迹。

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

# -----------------------------------------------------------------------------
# Design analysis for the D6 variant
#
# 1) Concrete behavior:
#    - The base task already defines a head-anchored triangular sprite that jumps
#      between axis-aligned single-pixel markers. The D6 change is localized to
#      what remains behind after each move.
#    - To keep the visual rule legible, the trajectory is defined on the HEAD
#      path rather than the full swept sprite body. For a move from A to B:
#      every head-path cell from A up to but not including B is recolored by the
#      movement direction. The target head cell is left free for the sprite.
#    - Example: if the head moves right from (4,2) to (4,6), cells (4,2..5) are
#      painted blue; if the next move goes up to (1,6), cells (4,6),(3,6),(2,6)
#      are painted red. The final sprite still uses the actor color and sits on
#      top of the last head position.
#
# 2) Whole-pipeline impact:
#    - Original task: old positions vanished, so the latest frame alone was the
#      complete state.
#    - Variant: trajectory persists, so EVERY snapshot frame must already show
#      all past trajectory painted so far. This keeps the motion rule visible in
#      the input, and the rightmost frame is still sufficient for continuation.
#    - Solver change therefore happens exactly at simulation/render time:
#      preserve existing trail colors from the rightmost frame, continue the
#      remaining marker consumption, and add new direction-colored trail cells.
#
# 3) Complexity calibration:
#    - Trail colors are fixed to four canonical ARC colors, one per direction.
#      This is a small, repeated vocabulary with direct geometric meaning.
#    - The actor/bg/divider colors are sampled only from NON_TRAIL_COLORS, so
#      the new trail never aliases with the sprite, markers, background, or the
#      divider. Without this restriction `_find_agent` and human perception
#      would both become brittle.
#    - The generator also enforces at least two remaining steps in the final
#      input frame, so the output is not reduced to a trivial one-segment trail.
#
# 4) Hidden assumptions from the original main():
#    - `_split_frames` used to assume every full-height solid non-bg column was a
#      divider. In this variant a full-height TRAIL column must NOT be treated as
#      a divider, so divider detection ignores the four trajectory colors.
#    - Actor extraction used to take "the only non-bg non-divider color". That no
#      longer holds because trail colors are now present in the frames. The new
#      actor extractor removes divider colors and all trail colors first.
#    - Returning a clean background at the end used to be harmless. Here it would
#      erase valid state, so the solver must explicitly preserve existing trail
#      and extend it.
#
# 5) Placement legality guaranteed by the generator:
#    - Future marker heads are never placed on existing trail cells.
#    - New trail segments never cross existing trail cells.
#    - Snapshot sprite placements never overlap existing trail cells, so no trail
#      is hidden underneath the actor body. That keeps the rightmost frame fully
#      informative for the solver.
#
# 6) Visual observability / cleanliness:
#    - The input frames visibly accumulate red/blue/green/yellow axis-aligned
#      traces, so the D6 change is observable before the solver ever computes the
#      output.
#    - Using head-width-1 trails avoids the noisy thick blobs that full-body
#      sweep rendering would create around turns and large templates.
# -----------------------------------------------------------------------------

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
TRAIL_BY_DIR = {0: 2, 1: 1, 2: 3, 3: 4}  # U->red, R->blue, D->green, L->yellow
TRAIL_COLORS = set(TRAIL_BY_DIR.values())
NON_TRAIL_COLORS = (0, 5, 6, 7, 8, 9)
ACTOR_COLORS = (5, 6, 7, 8, 9)


def _split_frames(g: np.ndarray) -> tuple[list[np.ndarray], int]:
    """
    Input is multiple frames concatenated horizontally, separated by full-height
    divider columns. Divider columns never use the four trajectory colors.
    """
    h, w = g.shape
    bg = _bg_color(g)

    solid_cols = [c for c in range(w) if len(set(map(int, g[:, c]))) == 1]
    div_cols = [
        c for c in solid_cols
        if int(g[0, c]) != bg and int(g[0, c]) not in TRAIL_COLORS
    ]

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


def _segment_cells(
    a: tuple[int, int], b: tuple[int, int], *, include_end: bool
) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    dr = 0 if br == ar else (1 if br > ar else -1)
    dc = 0 if bc == ac else (1 if bc > ac else -1)
    dist = abs(br - ar) + abs(bc - ac)
    stop = dist + 1 if include_end else dist
    return [(ar + dr * k, ac + dc * k) for k in range(stop)]


def _extract_trail(frame: np.ndarray) -> dict[tuple[int, int], int]:
    trail: dict[tuple[int, int], int] = {}
    for y, x in np.argwhere(np.isin(frame, list(TRAIL_COLORS))):
        trail[(int(y), int(x))] = int(frame[y, x])
    return trail


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)
    trail = _extract_trail(frame)

    found = _find_agent(frame, actor)
    if found is None:
        out = np.full((h, w), bg, dtype=int)
        for (y, x), color in trail.items():
            out[y, x] = color
        return out

    tname, d, head, _agent, markers = found
    markers = set((int(y), int(x)) for y, x in markers)

    max_steps = h * w * 4
    agent = _sprite_cells(head, tname, d)
    for _ in range(max_steps):
        if not markers:
            break

        offs = TEMPLATES[tname][d]
        choice = _choose_target(head, d, markers, offs, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        nd, target = choice

        for cell in _segment_cells(head, target, include_end=False):
            trail[cell] = TRAIL_BY_DIR[nd]

        head = target
        d = nd
        agent = _sprite_cells(head, tname, d)
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for (y, x), color in trail.items():
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = color
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue a triangular sprite simulation and preserve its direction-colored trail.

    Input structure:
    - Frames are concatenated left-to-right with optional solid divider columns; the background is the most common color.
    - Each frame contains the actor-colored triangular sprite, same-colored remaining markers, and any already-painted trajectory cells:
      red for upward motion, blue for rightward motion, green for downward motion, yellow for leftward motion.

    Rule:
    - Start from the rightmost frame, which is the latest snapshot.
    - Preserve the trail already present there. Repeatedly find the unique reachable marker direction, move the sprite to the nearest marker on that half-line,
      and color every vacated head-path cell by that movement's fixed direction color.
    - Consume markers until none remain, then output the final frame with the accumulated trail and the final sprite body.

    Key insight: the latest snapshot already stores the past trajectory, so solving only requires detecting the current arrow, reading the remaining markers, and extending the trail with the same direction-to-color mapping.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {
        int(g[0, c]) for c in solid_cols
        if int(g[0, c]) != bg and int(g[0, c]) not in TRAIL_COLORS
    }
    colors = set(map(int, g.ravel()))
    actor_candidates = colors - {bg} - div_colors - TRAIL_COLORS
    assert len(actor_candidates) == 1, f"expected 1 actor color, got {actor_candidates}"
    actor = next(iter(actor_candidates))

    start = frames[-1]
    return _simulate_to_end(start, actor)


# --------------- generator (trajectory-color variant) ---------------

N_FRAMES_RANGE = (1, 5)
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
MIN_SUFFIX_STEPS = 2
MIN_SUFFIX_DIR_DIVERSITY = 2

TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

MAX_ATTEMPTS = 800

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
    multiples = [
        n for n in range(*STEPS_RANGE)
        if n % n_frames == 0 and n // n_frames >= MIN_SUFFIX_STEPS
    ]
    weights = np.array([(n ** 3) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    bg = int(rng.choice(NON_TRAIL_COLORS))
    actor = int(rng.choice([c for c in ACTOR_COLORS if c != bg]))
    divider = None
    if n_frames > 1:
        divider_choices = [c for c in NON_TRAIL_COLORS if c not in {bg, actor}]
        divider = int(rng.choice(divider_choices))

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
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]], set[tuple[int, int]]] | None:
    n_steps = len(step_dirs)
    d0 = step_dirs[0]
    heads: list[tuple[int, int]] = [(0, 0)]
    state_dirs: list[int] = [d0]
    fit_placements: list[tuple[tuple[int, int], int]] = [((0, 0), d0)]
    used_trace: set[tuple[int, int]] = set()

    for i in range(n_steps):
        cur = heads[-1]
        cur_dir = state_dirs[-1]
        next_dir = step_dirs[i]
        if next_dir == (cur_dir + 2) % 4:
            return None

        dr, dc = DIRS4[next_dir]
        dist_order = list(rng.permutation(np.arange(*STEP_DIST_RANGE)))
        chosen = None
        chosen_fit = None
        chosen_trace = None
        for dist in dist_order:
            nxt = (cur[0] + int(dist) * dr, cur[1] + int(dist) * dc)
            if nxt in heads or nxt in used_trace:
                continue
            if any(_touching(nxt, m) for m in heads[1:]):
                continue
            if not _marker_compatible_with_existing(nxt, heads, state_dirs, step_dirs, tname):
                continue

            seg_trace = _segment_cells(cur, nxt, include_end=False)
            if any(cell in used_trace for cell in seg_trace):
                continue
            if any(cell in heads[:-1] for cell in seg_trace[1:]):
                continue

            trial_fit = fit_placements + [(nxt, cur_dir), (nxt, next_dir)]
            if _shift_interval(trial_fit, h, w, tname) is None:
                continue

            nxt_sprite = _sprite_cells(nxt, tname, next_dir)
            if nxt_sprite & used_trace:
                continue

            chosen = nxt
            chosen_fit = trial_fit
            chosen_trace = seg_trace
            break

        if chosen is None or chosen_fit is None or chosen_trace is None:
            return None

        heads.append(chosen)
        state_dirs.append(next_dir)
        fit_placements = chosen_fit
        used_trace.update(chosen_trace)

    return heads, state_dirs, fit_placements, used_trace


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


def _trail_map(
    heads: list[tuple[int, int]], step_dirs: list[int], upto_steps: int
) -> dict[tuple[int, int], int]:
    trail: dict[tuple[int, int], int] = {}
    for i in range(upto_steps):
        for cell in _segment_cells(heads[i], heads[i + 1], include_end=False):
            trail[cell] = TRAIL_BY_DIR[step_dirs[i]]
    return trail


def _render_frame(
    h: int,
    w: int,
    bg: int,
    actor: int,
    head: tuple[int, int],
    d: int,
    tname: str,
    markers: list[tuple[int, int]],
    trail: dict[tuple[int, int], int],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    for (y, x), color in trail.items():
        frame[y, x] = color
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


def _has_suffix_variety(step_dirs: list[int], gap: int) -> bool:
    suffix = step_dirs[-gap:]
    return len(suffix) >= MIN_SUFFIX_STEPS and len(set(suffix)) >= MIN_SUFFIX_DIR_DIVERSITY


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, gap, h, w, bg, actor, divider, tname = _sample_params(rng)
    step_dirs = _make_step_dirs(rng, n_steps)
    if not _has_suffix_variety(step_dirs, gap):
        return None

    rollout = _rollout_path(rng, tname, step_dirs, h, w)
    if rollout is None:
        return None
    heads, state_dirs, fit_placements, _used_trace = rollout

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
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        trail = _trail_map(heads, step_dirs, si)
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, heads[si + 1:], trail))

    fhead, fdir = placements[-1]
    out_trail = _trail_map(heads, step_dirs, n_steps)
    out_frame = _render_frame(h, w, bg, actor, fhead, fdir, tname, [], out_trail)
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
    expected_last_trail = _trail_map(heads, step_dirs, last_idx)
    if _extract_trail(frames[-1]) != expected_last_trail:
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
            "trail_cells_last_input_frame": len(expected_last_trail),
            "trail_cells_output": len(out_trail),
            "suffix_dir_diversity": len(set(step_dirs[-gap:])),
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
