# Variant of 5545f144
# Dimension: D18 — 步距约束 (meta)
# Change: 每步移动的曼哈顿距离为2-7格，且整条路径至少有2种不同步距。... → 固定步距

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left

# Design analysis (D18 fixed stride)
# spec:
#   concretization:
#     - Frames, divider columns, colors, and the four triangle templates stay the same as 5545f144.
#     - The only rule change is geometric: every head-to-marker jump uses one shared Manhattan
#       distance K, with K chosen from a small visible set {2, 3, 4}.
#     - Markers still lie on cardinal half-lines from the head, and only the head landing consumes them.
#   behavior_trace:
#     - Example: if the rightmost frame shows head=(5,3) facing right and remaining markers
#       at (5,6), (8,6), (8,3), then K=3 and the sprite must jump R3, D3, L3.
#     - In the original task the same turning pattern could have used mixed distances such as
#       R2, D5, L3; the solver only needed "nearest on the unique half-line".
#     - In the variant the first legal jump reveals K, and every later jump must land exactly K cells away.
#   complexity_calibration:
#     - The task still keeps multiple frames, shared actor/marker color, >=2 turns, and >=3 used
#       directions, so the triangle orientation and marker ordering still matter.
#     - Restricting K to {2,3,4} keeps the equal spacing visible without making 8..15 sized frames too sparse.
#     - The generator forces the last input frame to keep at least 2 markers, so repeated spacing is usually visible.
#   main_hidden_assumptions:
#     - _find_agent still depends on markers not masquerading as part of another triangle placement.
#     - _choose_target can no longer accept any distance; it infers K from the first jump and then
#       requires an exact-K landing on every later step.
#     - The landing-fit check still uses the current orientation at the landing cell before any turn,
#       so the generator must ensure both "arrive as-is" and "arrive then rotate" placements fit in-grid.
#   placement_legality:
#     - The generator builds the path first, forbids head reuse, forbids marker adjacency to stamped
#       sprite cells, and rejects any partial path whose translated placements cannot fit a frame.
#     - In the original task any farther aligned marker could matter because the solver searched a full half-line.
#       In this variant only markers exactly K away matter after K is inferred, so a farther aligned marker is harmless.
#     - The generator therefore only forbids exact-K alternative markers from older heads; this is the necessary
#       compatibility adjustment that makes fixed-stride paths feasible without changing the sprite vocabulary.
#   visual_observability:
#     - The new cue is repeated equal spacing between consecutive landings; quantitatively every valid
#       sample must satisfy unique(step_distances) == 1, unlike the original generator which required variety.
#   visual_cleanliness:
#     - K>=2 plus the existing touching checks keeps single-pixel markers detached from the triangle body.


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


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    offs: list[tuple[int, int]],
    h: int,
    w: int,
    step_dist: int | None,
) -> tuple[int, tuple[int, int], int] | None:
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

    reachable: list[tuple[int, int, tuple[int, int]]] = []  # (dist, dir, target)
    for dd in range(4):
        for dist, pt in candidates(dd):
            if fits(pt):
                reachable.append((dist, dd, pt))

    if not reachable:
        return None

    if step_dist is None:
        reachable.sort()
        dist = reachable[0][0]
        exact = [(cand_dist, dd, pt) for cand_dist, dd, pt in reachable if cand_dist == dist]
    else:
        dist = step_dist
        exact = [(cand_dist, dd, pt) for cand_dist, dd, pt in reachable if cand_dist == step_dist]
        assert exact, (
            f"head={head} dir={d}: no marker at fixed step {step_dist}, "
            f"reachable distances are {[dist for dist, _, _ in reachable]}"
        )

    dirs_with_candidates = {dd for _, dd, _ in exact}
    assert len(dirs_with_candidates) == 1, (
        f"head={head} dir={d}: markers at distance {dist} appear in multiple directions "
        f"{dirs_with_candidates}, expected exactly one"
    )

    exact.sort()
    dist, dd, pt = exact[0]

    assert dd != (d + 2) % 4, (
        f"head={head} dir={d}: would reverse to dir={dd}, not allowed"
    )

    return dd, pt, dist


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, agent, markers = found
    markers = {(int(y), int(x)) for y, x in markers}
    agent = {(int(y), int(x)) for y, x in agent}
    step_dist: int | None = None

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        offs = TEMPLATES[tname][d]
        choice = _choose_target(head, d, markers, offs, h, w, step_dist)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        nd, (ty, tx), dist = choice

        if step_dist is None:
            step_dist = dist
        else:
            assert dist == step_dist, (
                f"head={head} dir={d}: expected fixed step {step_dist}, got {dist}"
            )

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
    Advance a triangular sprite through equal-length cardinal jumps until every marker is consumed.

    Input structure:
    - Frames are concatenated horizontally and separated by solid divider columns; the background is the most common color.
    - Each frame contains one triangular sprite plus the still-unvisited single-pixel markers, all in the same actor color.

    Rule:
    - Read the rightmost frame as the current state.
    - Exactly one cardinal half-line from the head has reachable markers; the nearest such marker reveals the common jump length.
    - Every later move must land exactly that same distance away on the unique reachable half-line, consuming the landed marker and re-facing the sprite.
    - When no markers remain, output only the final sprite on a clean background.

    Key insight: The remaining markers are not just ordered by direction; they also form an equally spaced chain, so the first legal jump fixes the stride for the rest of the simulation.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    start = frames[-1]
    return _simulate_to_end(start, actor)


# --------------- generator (fixed-step D18 variant) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (1, 5)          # rng.integers low/high+1 -> 1..4 frames
FRAME_MAX = 16                   # rng.integers high for H, W -> 8..15 / 6..15
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 6, 15

# --- CFG: path complexity ---
STEPS_RANGE = (4, 13)            # 4..12 inclusive
FIXED_STEP_CHOICES = (2, 3, 4)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_LAST_FRAME_MARKERS = 2

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 300
SEARCH_NODE_BUDGET = 12000

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


def _step_options(n_steps: int) -> tuple[int, ...]:
    if n_steps >= 10:
        return (2,)
    if n_steps >= 7:
        return (2, 3)
    return FIXED_STEP_CHOICES


def _sample_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int, int, int | None, str, int]:
    """Sample (n_frames, n_steps, gap, h, w, bg, actor, divider, tname, step_dist)."""
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [
        n
        for n in range(*STEPS_RANGE)
        if n % n_frames == 0 and n // n_frames >= MIN_LAST_FRAME_MARKERS
    ]
    weights = np.array([(n ** 2) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    step_dist = int(rng.choice(_step_options(n_steps)))
    approx_span = step_dist * max(2, min(4, (n_steps + 1) // 3))
    h_lo = min(H_CAP, max(H_FLOOR, 5 + approx_span))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + approx_span))
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
    return n_frames, n_steps, gap, h, w, bg, actor, divider, tname, step_dist


def _marker_compatible_with_existing(
    nxt: tuple[int, int],
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
    step_dist: int,
) -> bool:
    """Validate candidate marker against all previously-placed heads."""
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, state_dirs[j])
        if any(_touching(nxt, cell) for cell in sprite):
            return False

        rel_dir = _direction(old_head, nxt)
        if rel_dir is None or j == len(heads) - 1:
            continue

        if _dist(old_head, nxt) != step_dist:
            continue
        if rel_dir != state_dirs[j + 1]:
            return False
    return True


def _build_path(
    rng: np.random.Generator, tname: str, n_steps: int, step_dist: int, h: int, w: int
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]] | None:
    """Backtrack a path whose every move has the same Manhattan distance."""
    min_turns = max(MIN_TURNS_ACCEPT, int(np.ceil(TURN_RATIO_LO * n_steps)))
    max_turns = min(n_steps - 1, int(np.floor(TURN_RATIO_HI * n_steps)))
    budget = [SEARCH_NODE_BUDGET]

    def dfs(
        step_idx: int,
        heads: list[tuple[int, int]],
        state_dirs: list[int],
        fit_placements: list[tuple[tuple[int, int], int]],
        turns: int,
    ) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]] | None:
        budget[0] -= 1
        if budget[0] < 0:
            return None
        if step_idx == n_steps:
            step_dirs = state_dirs[1:]
            if _validate_path(step_dirs, heads, step_dist)[0]:
                return heads, state_dirs, fit_placements
            return None

        cur = heads[-1]
        cur_dir = state_dirs[-1]
        if step_idx == 0:
            dir_choices = [cur_dir]
        else:
            dir_choices = [cur_dir, (cur_dir + 1) % 4, (cur_dir - 1) % 4]
            rng.shuffle(dir_choices)

        for next_dir in dir_choices:
            next_turns = turns + int(step_idx > 0 and next_dir != cur_dir)
            remaining_after = n_steps - step_idx - 1
            if next_turns > max_turns or next_turns + remaining_after < min_turns:
                continue

            dr, dc = DIRS4[next_dir]
            nxt = (cur[0] + step_dist * dr, cur[1] + step_dist * dc)
            if nxt in heads:
                continue
            if any(_touching(nxt, marker) for marker in heads[1:]):
                continue
            if not _marker_compatible_with_existing(nxt, heads, state_dirs, tname, step_dist):
                continue

            new_places = [(nxt, cur_dir)]
            if next_dir != cur_dir:
                new_places.append((nxt, next_dir))
            trial_fit = fit_placements + new_places
            if _shift_interval(trial_fit, h, w, tname) is None:
                continue

            out = dfs(
                step_idx + 1,
                heads + [nxt],
                state_dirs + [next_dir],
                trial_fit,
                next_turns,
            )
            if out is not None:
                return out
        return None

    for d0 in map(int, rng.permutation(4)):
        out = dfs(0, [(0, 0)], [d0], [((0, 0), d0)], 0)
        if out is not None:
            return out
    return None


def _validate_path(
    step_dirs: list[int], heads: list[tuple[int, int]], step_dist: int
) -> tuple[bool, int, list[int]]:
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
    if any(dist != step_dist for dist in dists):
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
    n_frames, n_steps, gap, h, w, bg, actor, divider, tname, step_dist = _sample_params(rng)

    rollout = _build_path(rng, tname, n_steps, step_dist, h, w)
    if rollout is None:
        return None
    heads, state_dirs, fit_placements = rollout
    step_dirs = state_dirs[1:]

    ok, turns, dists = _validate_path(step_dirs, heads, step_dist)
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
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, heads[si + 1 :]))

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
    if f_markers != set(heads[last_idx + 1 :]):
        return None
    if len(f_markers) < MIN_LAST_FRAME_MARKERS:
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
            "step_distance": step_dist,
            "step_distances": dists,
            "step_distance_unique_count": len(set(dists)),
            "count_in_first_frame": n_steps,
            "count_in_last_input_frame": gap,
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
