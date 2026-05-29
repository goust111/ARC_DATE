# Combo variant of 5545f144 (index 11)
# combo_spec: D4/2+D18/0
# Name: 对角线移动+固定步距

"""
Design analysis for combo D4/2 + D18/0.

a) Concrete behavior and trace
- The strip-of-frames input format stays the same, but the arrow sprite now uses
  diagonal orientations and moves only along 45-degree rays. Markers are still
  single pixels of the same color as the sprite.
- Original task trace: from head H, inspect row/column half-lines, jump to the
  nearest marker on the unique active cardinal half-line, then allow only
  forward or 90-degree turns.
- Fused trace: from head H, inspect the four diagonal half-lines instead. The
  first reachable marker fixes a diagonal stride K, and every later move must
  land exactly K cells away on the unique active diagonal ray. Example:
  head=(6,4), remaining markers at (4,6),(2,8),(4,10). The fused rule reads
  NE distance 2 first, so K=2; the sprite goes to (4,6), then must keep making
  diagonal jumps of length 2, such as NE to (2,8) or SE to (6,8) when a turn is
  required. A farther dot like (2,8) is not chosen immediately because the
  nearer equal-ray dot at distance 2 comes first.
- The visible difference from the source task is therefore twofold at once:
  marker chains lie on diagonals, and consecutive head landings are equally
  spaced rather than mixing different lengths.

b) Complexity calibration
- The vocabulary stays small: one actor color, one optional divider color, and
  the same four triangle families as the base task. The only geometric changes
  are diagonal orientation plus a shared jump length.
- Fixed stride is intentionally limited to a short visible set {2,3,4}; larger
  values made 8..15 sized frames too sparse once diagonal turns were required.
- The generator also forces the last input frame to retain at least two markers,
  so the equal spacing is usually visible in the actual state solved by `main()`
  instead of being hidden in earlier snapshots.

c) Implicit-assumption audit of the original logic
- Original `_choose_target` assumed row/column alignment. That breaks under D4/2,
  so reachability must be rewritten in terms of diagonal rays and Chebyshev-like
  diagonal distance.
- Original landing-fit code translated the current template to the target. That
  was acceptable only because the source task stayed cardinal; here the solver
  must check the template after re-facing to the candidate diagonal direction.
- Original logic accepted any nearest aligned marker. D18/0 invalidates that:
  once the first jump reveals K, later steps must reject all distances except K.
- Original `_simulate_to_end` implicitly relied on `agent` being assigned inside
  the loop. With a fixed-step variant, zero remaining markers is common at loop
  entry for some tests, so the current body is initialized before stepping.

d) Placement legality
- The generator constructs a path of future head positions first. Every move is
  exactly K cells on one diagonal direction, with only straight or 90-degree
  turns and no 180-degree reversals.
- Candidate markers are rejected if they reuse a head, touch any stamped sprite
  body, or create a closer diagonal alternative from an earlier head. This last
  check is what keeps each snapshot consistent with "first visible jump fixes K".
- A global translation is applied only after all placements are sampled, and it
  is accepted only if every stamped sprite placement fits inside the frame.

e) Visual observability
- Humans can see the fusion from a few examples because the sprite itself points
  diagonally and the remaining dots form repeated equal diagonal gaps.
- The rule does not depend on numeric color ids, frame ordering quirks, or any
  hidden metadata. Only visible geometry matters: diagonal alignment, nearest
  marker for the first step, then repeated equal spacing.
- The output keeps only the final sprite, so the training pairs directly support
  "continue the same diagonal stride pattern until the dots are exhausted".

f) Visual cleanliness
- Markers remain isolated single pixels, and the generator enforces at least a
  Chebyshev gap of 2 from earlier markers and sprite cells, which keeps the
  actor-colored dots from blending into the diagonal triangle body.
- Frame sizes are biased upward when more steps or larger stride are used, so
  the diagonal route is readable rather than cramped against borders.
- Because the same actor color is shared by sprite and markers, shape separation
  must come from spacing; the generator explicitly preserves that spacing.

g) Change inferability
- There is no new semantic color mapping to learn. The only new latent variable
- is K, the common stride, and it is visually supported by repeated equal
  marker gaps.
- The diagonal orientation is also repeated structurally: both the sprite tip
  and the marker chain agree on the same four 45-degree movement directions.
- Oracle checks ensure samples do not silently collapse back to the original
  cardinal or variable-stride behavior.

h) New solver assumptions and generator guarantees
- `main()` still assumes the rightmost frame is the latest snapshot; the
  generator preserves the original evenly spaced cadence.
- `_find_agent` assumes one diagonal template is the largest coherent actor
  cluster in the active frame; the generator keeps future markers isolated so
  they cannot masquerade as a better template placement.
- `_choose_target` assumes the first legal move determines a unique stride and
  every later legal move exists at exactly that stride on one non-reverse
  diagonal ray; path construction enforces those uniqueness constraints.
"""

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 1), (1, 1), (1, -1), (-1, -1)]  # NE, SE, SW, NW

# Direction indices: 0=NE, 1=SE, 2=SW, 3=NW


def _rot90_cw(pt: tuple[int, int]) -> tuple[int, int]:
    r, c = pt
    return c, -r


def _rotated(offs: list[tuple[int, int]], turns: int) -> list[tuple[int, int]]:
    out = list(offs)
    for _ in range(turns % 4):
        out = [_rot90_cw(pt) for pt in out]
    return out


_CANONICAL_NE: dict[str, list[tuple[int, int]]] = {
    "hollow3": [(0, 0), (1, -1), (1, 0)],
    "filled4": [(0, 0), (1, -1), (1, 0), (2, -1)],
    "stem4": [(0, 0), (1, -1), (2, -2), (2, -1)],
    "midfilled6": [(0, 0), (1, -1), (1, 0), (2, -2), (2, -1), (2, 0)],
}


def _make_templates() -> dict[str, dict[int, list[tuple[int, int]]]]:
    templates: dict[str, dict[int, list[tuple[int, int]]]] = {}
    for tname, base in _CANONICAL_NE.items():
        dirs: dict[int, list[tuple[int, int]]] = {}
        for d in range(4):
            dirs[d] = _rotated(base, d)
        templates[tname] = dirs
    return templates


TEMPLATES = _make_templates()


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


def _diag_direction(a: tuple[int, int], b: tuple[int, int]) -> int | None:
    ar, ac = a
    br, bc = b
    dr, dc = br - ar, bc - ac
    if dr == 0 or abs(dr) != abs(dc):
        return None
    if dr < 0 and dc > 0:
        return 0
    if dr > 0 and dc > 0:
        return 1
    if dr > 0 and dc < 0:
        return 2
    if dr < 0 and dc < 0:
        return 3
    return None


def _diag_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


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


def _fits_template(head: tuple[int, int], tname: str, d: int, h: int, w: int) -> bool:
    hy, hx = head
    for oy, ox in TEMPLATES[tname][d]:
        ny = hy + oy
        nx = hx + ox
        if not (0 <= ny < h and 0 <= nx < w):
            return False
    return True


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    tname: str,
    h: int,
    w: int,
    step_dist: int | None,
) -> tuple[int, tuple[int, int], int] | None:
    def candidates(dir_idx: int) -> list[tuple[int, tuple[int, int]]]:
        out: list[tuple[int, tuple[int, int]]] = []
        for pt in markers:
            if _diag_direction(head, pt) == dir_idx:
                out.append((_diag_distance(head, pt), pt))
        out.sort()
        return out

    reachable: list[tuple[int, int, tuple[int, int]]] = []
    for dd in range(4):
        for dist, pt in candidates(dd):
            if _fits_template(pt, tname, dd, h, w):
                reachable.append((dist, dd, pt))

    if not reachable:
        return None

    if step_dist is None:
        dist = min(dist for dist, _, _ in reachable)
        exact = [(cand_dist, dd, pt) for cand_dist, dd, pt in reachable if cand_dist == dist]
    else:
        dist = step_dist
        exact = [(cand_dist, dd, pt) for cand_dist, dd, pt in reachable if cand_dist == step_dist]
        assert exact, (
            f"head={head} dir={d}: no marker at fixed diagonal step {step_dist}, "
            f"reachable distances are {[cand_dist for cand_dist, _, _ in reachable]}"
        )

    dirs_with_candidates = {dd for _, dd, _ in exact}
    assert len(dirs_with_candidates) == 1, (
        f"head={head} dir={d}: markers at distance {dist} appear in multiple diagonal "
        f"directions {dirs_with_candidates}, expected exactly one"
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

    tname, d, head, _agent, markers = found
    markers = {(int(y), int(x)) for y, x in markers}
    agent = _sprite_cells(head, tname, d)
    step_dist: int | None = None

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        choice = _choose_target(head, d, markers, tname, h, w, step_dist)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        d, head, dist = choice

        if step_dist is None:
            step_dist = dist
        else:
            assert dist == step_dist, (
                f"head={head} dir={d}: expected fixed diagonal step {step_dist}, got {dist}"
            )

        agent = _sprite_cells(head, tname, d)
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue a diagonal-moving triangular sprite through equal-length jumps until every marker is consumed.

    Input structure:
    - Frames are concatenated left-to-right, optionally separated by full-height divider columns.
    - Each frame shows one diagonal arrow sprite and the still-unvisited single-pixel markers in the same actor color.

    Rule:
    - Read the rightmost frame as the latest snapshot.
    - From the head, consider only the four diagonal half-lines (NE, SE, SW, NW). The nearest reachable marker fixes the common jump length.
    - Every later move must land exactly that same diagonal distance away on the unique non-reverse diagonal ray that still has a reachable marker.
    - The head lands on the marker, the marker disappears, the sprite re-faces that diagonal direction, and the process repeats until no markers remain.

    Key insight: The remaining dots form a diagonal chain with one shared spacing, so the first legal jump reveals both the 45-degree movement family and the fixed stride used for the rest of the replay.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    start = frames[-1]
    return _simulate_to_end(start, actor)


# --------------- generator (diagonal + fixed-stride combo) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (2, 5)  # 2..4 frames
FRAME_MAX = 16
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 8, 15

# --- CFG: path complexity ---
STEPS_RANGE = (4, 11)  # 4..10 inclusive
FIXED_STEP_CHOICES = (2, 3, 4)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_LAST_FRAME_MARKERS = 2

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 8
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 500
SEARCH_NODE_BUDGET = 20000

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(1, 5)) for n in range(*STEPS_RANGE)}


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


def _step_options(n_steps: int) -> tuple[int, ...]:
    if n_steps >= 9:
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
    h_lo = min(H_CAP, max(H_FLOOR, 6 + approx_span))
    w_lo = min(W_CAP, max(W_FLOOR, 6 + approx_span))
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
    """Reject collisions and diagonal alternatives that would change the inferred stride/path."""
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, state_dirs[j])
        if any(_touching(nxt, cell) for cell in sprite):
            return False

        rel_dir = _diag_direction(old_head, nxt)
        if rel_dir is None or j == len(heads) - 1:
            continue

        dist = _diag_distance(old_head, nxt)
        if dist < step_dist:
            return False
        if dist == step_dist and rel_dir != state_dirs[j + 1]:
            return False
    return True


def _build_path(
    rng: np.random.Generator, tname: str, n_steps: int, step_dist: int, h: int, w: int
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]] | None:
    """Backtrack a diagonal path whose every move has the same stride."""
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

            trial_fit = fit_placements + [(nxt, next_dir)]
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

    row_signs = {1 if DIRS4[d][0] > 0 else -1 for d in dir_set}
    col_signs = {1 if DIRS4[d][1] > 0 else -1 for d in dir_set}
    if row_signs != {-1, 1} or col_signs != {-1, 1}:
        return False, 0, []

    turns = sum(step_dirs[i] != step_dirs[i - 1] for i in range(1, n_steps))
    if turns < MIN_TURNS_ACCEPT or not (TURN_RATIO_LO <= turns / n_steps <= TURN_RATIO_HI):
        return False, turns, []

    dists = [_diag_distance(heads[i], heads[i + 1]) for i in range(n_steps)]
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
