# Variant of 5545f144
# Dimension: D6 — 轨迹：移动痕迹 (trajectory)
# Change: 精灵移动无轨迹——旧位置恢复为背景色，只保留当前位置的精灵体。... → 连续轨迹涂色

import numpy as np

from arc_utils import Color
from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
TRAIL_COLOR = int(Color.GREY)

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left

# Design analysis for the D6 variant
# spec:
#   change_focus:
#     old_behavior: Only the current triangle was visible; vacated cells returned to bg.
#     new_behavior: Every cell occupied while the sprite slides toward markers becomes
#       a fixed grey trail; the final frame/output shows the whole traversed footprint
#       plus the live sprite at the endpoint.
#   concretization:
#     input_frames:
#       - Frames remain horizontal snapshots with divider columns.
#       - Each snapshot now contains three visible layers inside the frame:
#         current actor-colored sprite, remaining actor-colored marker dots, and the
#         accumulated grey trail from earlier moves.
#     output:
#       - Single frame on bg containing the full grey trail and the final actor sprite.
#     micro_example:
#       - If a filled triangle at (5,2) slides three cells right while facing up, the
#         old task only kept the landing sprite.
#       - This variant paints the start placement and the two intermediate translated
#         placements grey; only the landing placement stays actor-colored.
#   complexity_calibration:
#     - The path logic, template set, and marker-consumption rule stay unchanged.
#     - The trail color is fixed grey across all samples, keeping the new behavior
#       visually stable instead of introducing an extra latent color mapping problem.
#     - At least two frames are generated so the trail is visible in the input, not
#       only inferred from train outputs.
#   implicit_assumption_review:
#     - Original main() assumed every non-bg, non-divider pixel belonged to the actor.
#       That is no longer valid once trail pixels exist, so actor extraction must
#       explicitly exclude grey trail pixels.
#     - Original _simulate_to_end() rebuilt a clean bg canvas and stamped only the
#       final sprite. In the variant it must preserve any trail already present in the
#       last snapshot, add the trail of each subsequent slide, and then stamp the
#       final sprite on top so actor beats trail on overlapping cells.
#     - _find_agent() still assumes actor and markers share one color; generator keeps
#       trail on a separate color to preserve this invariant.
#   placement_legality:
#     - Future marker dots must never sit on already-painted trail cells in any input
#       snapshot, otherwise a marker would be hidden and the solver would read the
#       wrong remaining-marker set.
#     - All sprite placements at marker states are still kept in-bounds exactly as in
#       the base task; translating between two in-bounds states on one axis keeps all
#       intermediate translated placements in-bounds too.
#   visual_observability:
#     - The grey trail is a persistent, high-contrast cue present in later snapshots
#       and in the output, so the D6 change is readable from a few examples.
#     - Divider, actor, and trail always use distinct colors.
#   cleanliness:
#     - Generator rejects cases where frame splitting breaks, bg ceases to be the
#       most common color, or future markers are covered by historic trail.


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


def _template_bounds(tname: str, d: int) -> tuple[int, int, int, int]:
    offs = TEMPLATES[tname][d]
    rs = [dr for dr, _ in offs]
    cs = [dc for _, dc in offs]
    return min(rs), max(rs), min(cs), max(cs)


def _template_patch(tname: str, d: int, color: int, bg: int) -> tuple[np.ndarray, int, int]:
    r0, r1, c0, c1 = _template_bounds(tname, d)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for dr, dc in TEMPLATES[tname][d]:
        patch[dr - r0, dc - c0] = color
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


def _segment_trail_cells(
    start: tuple[int, int],
    end: tuple[int, int],
    tname: str,
    d: int,
) -> set[tuple[int, int]]:
    move_dir = _direction(start, end)
    assert move_dir is not None, f"non-cardinal movement from {start} to {end}"
    dist = _dist(start, end)
    dr, dc = DIRS4[move_dir]

    trail: set[tuple[int, int]] = set()
    for step in range(dist):
        head = (start[0] + step * dr, start[1] + step * dc)
        trail |= _sprite_cells(head, tname, d)
    trail -= _sprite_cells(end, tname, d)
    return trail


def _render_state(
    h: int,
    w: int,
    bg: int,
    actor: int,
    trail_color: int,
    head: tuple[int, int],
    d: int,
    tname: str,
    markers: list[tuple[int, int]],
    trail_cells: set[tuple[int, int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    for y, x in trail_cells:
        if 0 <= y < h and 0 <= x < w:
            frame[y, x] = trail_color
    patch, ar, ac = _template_patch(tname, d, actor, bg)
    paste(frame, patch, head[0] - ar, head[1] - ac, transparent=bg)
    for m in markers:
        frame[m] = actor
    return frame


def _simulate_to_end(frame: np.ndarray, actor: int, trail_color: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, agent, markers = found
    markers = set((int(y), int(x)) for y, x in markers)
    trail_cells = set(map(tuple, np.argwhere(frame == trail_color)))

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        offs = TEMPLATES[tname][d]
        choice = _choose_target(head, d, markers, offs, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        nd, target = choice

        trail_cells |= _segment_trail_cells(head, target, tname, d)

        head = target
        d = nd
        agent = _sprite_cells(head, tname, d)
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in trail_cells:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = trail_color
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance a triangular sprite through aligned markers while preserving its path as a trail.

    Input structure:
    - Frames are concatenated horizontally and split by full-height divider columns; bg is the most common color.
    - Each frame shows the actor-colored sprite, the remaining actor-colored marker dots, and any grey trail already left behind.

    Rule:
    - Read the rightmost frame as the current state.
    - From the sprite head, exactly one cardinal half-line contains reachable markers; move to the nearest one without 180-degree reversal.
    - As the sprite slides away, every vacated intermediate placement is painted grey; on arrival the marker is consumed and the sprite faces the travel direction.
    - Repeat until no markers remain, then output the final sprite together with the full accumulated grey trail.

    Key insight: The triangle tip still determines which marker is consumed next, but the output now requires replaying the whole remaining motion to color the traversed cells.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor_colors = colors - {bg, TRAIL_COLOR} - div_colors
    assert len(actor_colors) == 1, f"expected one actor color, got {actor_colors}"
    actor = next(iter(actor_colors))

    start = frames[-1]
    return _simulate_to_end(start, actor, TRAIL_COLOR)


# --------------- generator (D6 continuous trail variant) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (2, 5)          # 2..4 frames so the input visibly contains trail
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
MIN_TRAIL_CELLS = 8

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 800

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
) -> tuple[int, int, int, int, int, int, int, int, str]:
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

    palette = [c for c in range(10) if c != TRAIL_COLOR]
    rng.shuffle(palette)
    bg, actor, divider = map(int, palette[:3])

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
    trail_so_far: set[tuple[int, int]],
) -> bool:
    if nxt in trail_so_far:
        return False
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
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]], list[set[tuple[int, int]]]] | None:
    n_steps = len(step_dirs)
    d0 = step_dirs[0]
    heads: list[tuple[int, int]] = [(0, 0)]
    state_dirs: list[int] = [d0]
    fit_placements: list[tuple[tuple[int, int], int]] = [((0, 0), d0)]
    segment_trails: list[set[tuple[int, int]]] = []
    trail_so_far: set[tuple[int, int]] = set()

    for i in range(n_steps):
        cur = heads[-1]
        cur_dir = state_dirs[-1]
        next_dir = step_dirs[i]
        if next_dir == (cur_dir + 2) % 4:
            return None

        dr, dc = DIRS4[next_dir]
        dist_order = list(rng.permutation(np.arange(*STEP_DIST_RANGE)))
        chosen = None
        chosen_trail = None
        for dist in dist_order:
            nxt = (cur[0] + int(dist) * dr, cur[1] + int(dist) * dc)
            if nxt in heads:
                continue
            if any(_touching(nxt, m) for m in heads[1:]):
                continue
            if not _marker_compatible_with_existing(nxt, heads, state_dirs, step_dirs, tname, trail_so_far):
                continue
            seg_trail = _segment_trail_cells(cur, nxt, tname, cur_dir)
            trial_fit = fit_placements + [(nxt, cur_dir), (nxt, next_dir)]
            if _shift_interval(trial_fit, h, w, tname) is None:
                continue
            chosen = nxt
            chosen_trail = seg_trail
            break

        if chosen is None or chosen_trail is None:
            return None

        heads.append(chosen)
        state_dirs.append(next_dir)
        fit_placements.extend([(chosen, cur_dir), (chosen, next_dir)])
        segment_trails.append(chosen_trail)
        trail_so_far |= chosen_trail

    return heads, state_dirs, fit_placements, segment_trails


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


def _trail_prefixes(segment_trails: list[set[tuple[int, int]]]) -> list[set[tuple[int, int]]]:
    out: list[set[tuple[int, int]]] = [set()]
    acc: set[tuple[int, int]] = set()
    for seg in segment_trails:
        acc = acc | seg
        out.append(set(acc))
    return out


def _trail_is_compatible(
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
    segment_trails: list[set[tuple[int, int]]],
) -> bool:
    prefixes = _trail_prefixes(segment_trails)
    n_steps = len(segment_trails)
    for si in range(n_steps + 1):
        trail = prefixes[si]
        future_markers = heads[si + 1 :]
        if any(m in trail for m in future_markers):
            return False
        if si < len(heads):
            sprite = _sprite_cells(heads[si], tname, state_dirs[si])
            if any(m in sprite for m in future_markers):
                return False
    return True


def _assemble_input(frames: list[np.ndarray], bg: int, divider: int) -> np.ndarray:
    n_frames = len(frames)
    h, w = frames[0].shape
    total_w = n_frames * w + (n_frames - 1)
    inp = np.full((h, total_w), bg, dtype=int)
    for i, fr in enumerate(frames):
        c0 = i * (w + 1)
        paste(inp, fr, 0, c0)
        if i < n_frames - 1:
            inp[:, c0 + w] = divider
    return inp


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, gap, h, w, bg, actor, divider, tname = _sample_params(rng)
    step_dirs = _make_step_dirs(rng, n_steps)

    rollout = _rollout_path(rng, tname, step_dirs, h, w)
    if rollout is None:
        return None
    heads, state_dirs, fit_placements, segment_trails = rollout

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

    segment_trails = [
        {(r + tr, c + tc) for r, c in seg}
        for seg in segment_trails
    ]
    if not _trail_is_compatible(heads, state_dirs, tname, segment_trails):
        return None

    trail_prefixes = _trail_prefixes(segment_trails)
    full_trail = trail_prefixes[-1]
    if len(full_trail) < MIN_TRAIL_CELLS:
        return None

    frames = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        frames.append(
            _render_state(
                h,
                w,
                bg,
                actor,
                TRAIL_COLOR,
                head,
                d,
                tname,
                heads[si + 1 :],
                trail_prefixes[si],
            )
        )

    fhead, fdir = placements[-1]
    out_frame = _render_state(
        h,
        w,
        bg,
        actor,
        TRAIL_COLOR,
        fhead,
        fdir,
        tname,
        [],
        full_trail,
    )
    inp = _assemble_input(frames, bg, divider)

    split_frames, split_bg = _split_frames(inp)
    if split_bg != bg or len(split_frames) != n_frames or any(fr.shape != (h, w) for fr in split_frames):
        return None
    if _bg_color(inp) != bg:
        return None

    last_idx = (n_frames - 1) * gap
    found = _find_agent(frames[-1], actor)
    if found is None:
        return None
    f_tname, f_dir, f_head, _f_agent, f_markers = found
    if (f_tname, f_dir, f_head) != (tname, placements[last_idx][1], placements[last_idx][0]):
        return None
    if f_markers != set(heads[last_idx + 1 :]):
        return None
    if set(map(tuple, np.argwhere(frames[-1] == TRAIL_COLOR))) != trail_prefixes[last_idx]:
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
            "trail": TRAIL_COLOR,
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
            "trail_cells_final": len(full_trail),
            "trail_cells_last_input": len(trail_prefixes[last_idx]),
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
