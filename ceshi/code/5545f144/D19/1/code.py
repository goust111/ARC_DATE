# Variant of 5545f144
# Dimension: D19 — 模拟起始帧选择 (multi_step)
# Change: 从最右帧（最后一帧）开始模拟到标记全部消耗完毕。... → 从标记最少的帧开始

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left

# -----------------------------------------------------------------------------
# Design analysis for the D19 variant
#
# a) Concrete behavior and trace
# - Visual objects stay the same as the source task: each frame contains one
#   triangular sprite and several single-pixel markers, all in the actor color.
#   Frames are still concatenated horizontally with full-height divider columns.
# - The D19 change is applied at the frame-selection stage before simulation.
#   The solver must inspect every frame, identify the sprite body, count the
#   leftover actor pixels that are actually markers, and choose the frame with
#   the fewest markers.
# - To make that choice observable, frames can no longer be a mere permutation
#   of one shared timeline: if every frame belonged to the same path, starting
#   from any snapshot would reach the same final sprite, so the new rule would
#   be visually invisible. Therefore the generator turns the horizontal strip
#   into several independent candidate states that all obey the same movement
#   rule. Only the least-marker frame should be continued to the end.
# - Small trace:
#   frame A has 8 markers, frame B has 4 markers, frame C has 6 markers.
#   Old logic: start from frame C because it is rightmost.
#   New logic: start from frame B because 4 is the unique minimum.
#   Because A/B/C are independent candidate paths, simulating B ends at a
#   different final position than simulating C, so D19 changes the answer.
#
# b) Complexity calibration
# - Marker counts are sampled as distinct values, with the minimum count kept at
#   least 2 below the second-smallest count. This makes "least markers" easy to
#   read by eye and avoids near-ties.
# - The chosen frame is forced not to be the rightmost frame, and its final
#   output is forced to differ from the rightmost frame's final output. This
#   prevents silent collapse back to the source behavior.
# - The sprite template stays shared across frames so the task still looks like
#   one coherent family rather than a bag of unrelated shapes.
#
# c) Old assumptions that break under D19
# - In source `main()`, `start = frames[-1]` assumes the rightmost frame is the
#   intended state. That assumption is no longer valid and must be replaced by
#   per-frame marker counting.
# - The source generator assumes all frames belong to one path, so the last
#   frame's markers are the smallest suffix. Under D19 that would be harmless
#   for solving but unobservable for humans, so the generator must instead
#   produce multiple valid candidate states.
# - `_simulate_to_end()` originally relied on there being at least one marker so
#   `agent` would be assigned inside the loop. The variant still generates at
#   least 4 markers, but the function is hardened by initializing `agent`
#   immediately from the detected sprite.
#
# d) Placement legality
# - Every candidate frame is generated with the original path-first legality
#   checks: each future landing head and full rotated template must fit inside
#   the frame; markers cannot touch stamped sprite cells; no step may see
#   reachable markers on multiple half-lines; and no 180-degree reversal is
#   allowed.
# - Because the top-level input uses divider columns to split frames, the
#   generator also rejects any frame that accidentally contains its own internal
#   full-height solid non-background column, which would confuse frame splitting.
#
# e) Visual observability
# - The deciding signal is purely visible: count the marker dots left in each
#   frame after recognizing the triangle body.
# - The minimum-marker frame appears in varying positions, never always on the
#   right, so the solver cannot use a fixed spatial heuristic.
# - Independent candidate outputs differ, so examples can demonstrate that the
#   chosen frame really matters.
#
# f) Visual cleanliness
# - Each frame keeps the original sparse layout and single actor color, so there
#   is no added overlap noise.
# - Divider columns remain full-height solid lines, which cleanly separate
#   candidate states.
#
# g) Color inference
# - No color remapping is introduced. Background, actor, and divider keep their
#   original roles, so all new reasoning is spatial and count-based.
#
# Main assumption audit for the new solver
# - `_split_frames()` assumes divider columns are full-height and non-background:
#   generator enforces that and rejects divider-like actor columns inside frames.
# - Actor-color detection assumes there is exactly one non-bg, non-divider color:
#   generator uses one actor color globally.
# - `_select_start_frame()` assumes every frame contains one detectable sprite
#   plus leftover marker pixels: each generated frame is validated with
#   `_find_agent()` before it is accepted.
# - `_simulate_to_end()` assumes the chosen frame is internally consistent with
#   the movement rule: generator verifies each frame with `_simulate_to_end()`.
# -----------------------------------------------------------------------------


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
        choice = _choose_target(head, d, markers, offs, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        nd, (ty, tx) = choice

        head = (ty, tx)
        d = nd
        offs = TEMPLATES[tname][d]
        agent = {(ty + oy, tx + ox) for oy, ox in offs}
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def _frame_marker_count(frame: np.ndarray, actor: int) -> int | None:
    found = _find_agent(frame, actor)
    if found is None:
        return None
    return len(found[4])


def _select_start_frame(frames: list[np.ndarray], actor: int) -> tuple[int, np.ndarray]:
    ranked: list[tuple[int, int]] = []
    for i, frame in enumerate(frames):
        marker_count = _frame_marker_count(frame, actor)
        if marker_count is not None:
            ranked.append((marker_count, i))
    if not ranked:
        return len(frames) - 1, frames[-1]
    _, idx = min(ranked)
    return idx, frames[idx]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue the arrow simulation from the candidate frame with the fewest markers.

    Input structure:
    - Frames are concatenated horizontally with full-height divider columns; the
      background is the most common color across the full grid.
    - Every frame is a self-contained candidate state containing one triangular
      sprite plus single-pixel markers in the same actor color.

    Rule:
    - Detect the sprite separately inside each frame, then count how many actor
      pixels remain as markers; choose the frame with the smallest marker count.
    - Starting from that chosen frame, repeatedly move the sprite head to the
      nearest reachable marker on its unique aligned half-line and turn to face
      that direction when the marker is consumed.
    - Ignore the other frames after selection, and stop when the chosen frame
      has no markers left; output only the final sprite on a clean background.

    Key insight:
    - The triangle body can be peeled off from each frame, so the deciding clue
      is not frame position but which frame visibly has the fewest leftover dots.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    _, start = _select_start_frame(frames, actor)
    return _simulate_to_end(start, actor)


# --------------- generator ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (2, 5)          # 2..4 frames; D19 must be visible
FRAME_MAX = 16                   # rng.integers high for H, W -> 8..15 / 6..15
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 6, 15

# --- CFG: path complexity ---
STEPS_RANGE = (4, 13)            # 4..12 inclusive
STEP_DIST_RANGE = (2, 8)         # 2..7 inclusive
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 400
FRAME_ATTEMPTS = 120


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


def _sample_shared_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int | None, str, list[int], int] | None:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))

    raw_counts = list(map(int, rng.choice(np.arange(*STEPS_RANGE), size=n_frames, replace=False)))
    raw_counts.sort()
    if n_frames >= 2 and raw_counts[1] - raw_counts[0] < 2:
        return None
    if raw_counts[-1] - raw_counts[0] < 3:
        return None

    max_steps = raw_counts[-1]
    h_lo = min(H_CAP, max(H_FLOOR, 6 + (max_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (max_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = rng.permutation(10)
    bg, actor = int(palette[0]), int(palette[1])
    divider = int(palette[2])

    weights = np.array(
        TEMPLATE_WEIGHTS_HEAVY if max_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    weights /= weights.sum()
    tname = str(rng.choice(list(TEMPLATES), p=weights))

    start_pos = int(rng.integers(0, n_frames - 1))  # never the rightmost frame
    ordered_counts = [None] * n_frames
    ordered_counts[start_pos] = raw_counts[0]
    other_counts = raw_counts[1:]
    rng.shuffle(other_counts)
    other_positions = [i for i in range(n_frames) if i != start_pos]
    for pos, count in zip(other_positions, other_counts):
        ordered_counts[pos] = int(count)

    return n_frames, h, w, bg, actor, divider, tname, list(map(int, ordered_counts)), start_pos


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


def _has_internal_divider_like_column(frame: np.ndarray, bg: int) -> bool:
    _, w = frame.shape
    for c in range(w):
        if len(set(map(int, frame[:, c]))) == 1 and int(frame[0, c]) != bg:
            return True
    return False


def _build_single_frame_case(
    rng: np.random.Generator,
    h: int,
    w: int,
    bg: int,
    actor: int,
    tname: str,
    n_steps: int,
) -> dict | None:
    for _ in range(FRAME_ATTEMPTS):
        step_dirs = _make_step_dirs(rng, n_steps)
        rollout = _rollout_path(rng, tname, step_dirs, h, w)
        if rollout is None:
            continue
        heads, state_dirs, fit_placements = rollout

        ok, turns, dists = _validate_path(step_dirs, heads)
        if not ok:
            continue

        shift = _shift_interval(fit_placements, h, w, tname)
        if shift is None:
            continue
        tr_lo, tr_hi, tc_lo, tc_hi = shift
        tr = int(rng.integers(tr_lo, tr_hi + 1))
        tc = int(rng.integers(tc_lo, tc_hi + 1))
        heads = [(r + tr, c + tc) for r, c in heads]
        placements = [(heads[i], state_dirs[i]) for i in range(n_steps + 1)]

        frame = _render_frame(h, w, bg, actor, placements[0][0], placements[0][1], tname, heads[1:])
        out_frame = _render_frame(h, w, bg, actor, placements[-1][0], placements[-1][1], tname, [])

        if _has_internal_divider_like_column(frame, bg):
            continue

        found = _find_agent(frame, actor)
        if found is None:
            continue
        f_tname, f_dir, f_head, _f_agent, f_markers = found
        if (f_tname, f_dir, f_head) != (tname, placements[0][1], placements[0][0]):
            continue
        if f_markers != set(heads[1:]):
            continue
        if not np.array_equal(_simulate_to_end(frame, actor), out_frame):
            continue

        return {
            "frame": frame,
            "output": out_frame,
            "meta": {
                "marker_count": n_steps,
                "turns": turns,
                "turn_ratio": turns / n_steps,
                "direction_diversity": len(set(step_dirs)),
                "step_dirs": list(step_dirs),
                "state_dirs": list(state_dirs),
                "heads": list(heads),
                "step_distances": dists,
            },
        }
    return None


def _build_case(rng: np.random.Generator) -> dict | None:
    shared = _sample_shared_params(rng)
    if shared is None:
        return None
    n_frames, h, w, bg, actor, divider, tname, counts, intended_start_idx = shared

    frames: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    metas: list[dict] = []
    for count in counts:
        built = _build_single_frame_case(rng, h, w, bg, actor, tname, count)
        if built is None:
            return None
        frames.append(built["frame"])
        outputs.append(built["output"])
        metas.append(built["meta"])

    inp = _assemble_input(frames, bg, divider)
    out_frame = outputs[intended_start_idx]

    selected_idx, _ = _select_start_frame(frames, actor)
    if selected_idx != intended_start_idx:
        return None
    if selected_idx == n_frames - 1:
        return None
    if any(np.array_equal(out_frame, outputs[i]) for i in range(n_frames) if i != selected_idx):
        return None
    if np.array_equal(out_frame, outputs[-1]):
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
            "marker_counts": list(counts),
            "selected_frame_index": selected_idx,
            "rightmost_frame_index": n_frames - 1,
            "rightmost_matches_selected_output": False,
            "frames": metas,
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
