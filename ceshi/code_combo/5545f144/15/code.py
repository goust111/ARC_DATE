# Combo variant of 5545f144 (index 15)
# combo_spec: D1/1+D18/0+D22/0
# Name: 最近邻贪心+固定步距+部分消耗停止
#
# Design analysis for the combo variant
# =====================================
# a) Concretization and behavior trace
# - Visible objects stay the same: horizontally concatenated frames, one triangle
#   sprite, and actor-colored single-pixel dots.
# - The fused rule changes three linked parts at once.
#   1. D1/1: the next target is chosen by global Manhattan nearest-neighbor, not
#      by the current facing ray.
#   2. D18/0: all consumed jumps share one fixed stride K.
#   3. D22/0: once no marker remains at exactly stride K, the simulation stops
#      and the leftover dots stay in the output.
# - Small trace:
#   current head H faces Right. Dots are A=(H up by K), B=(H right by 2K),
#   plus another future path dot farther away from A. Original task would either
#   force the right-facing ray or reject the multi-direction layout. The combo
#   task instead jumps to A because A is the unique nearest marker, then keeps
#   using the same K. After the consumable chain ends, B survives because it is
#   still farther than K from the final head.
#
# b) Complexity calibration
# - Fixed stride is restricted to a small visible set {2,3,4}; this keeps the
#   equal spacing legible without making 8..15 sized frames too sparse.
# - The generated continuation in the rightmost frame is required to be
#   non-facing: the next step direction differs from the sprite's current
#   orientation. A stranded dot is also placed on the current facing ray, so the
#   ignore-facing behavior is visible instead of collapsing back to the base task.
# - Last input frames keep at least two path markers before adding stranded dots,
#   so both the chain and the stop condition are observable.
#
# c) Hidden assumptions broken from the original main()
# - Original `_choose_target()` assumed at most one active half-line. That is
#   false here because dots may remain in multiple directions; only distance and
#   the fixed stride matter.
# - Original solver ended only when marker count reached zero. Here stopping on
#   `no marker at distance K` is normal, so leftovers must be preserved.
# - Original code implicitly assumed `agent` is defined after at least one move.
#   Immediate stopping is legal now, so the detected starting sprite is kept.
# - Fixed stride means the solver now assumes later leftover dots are all farther
#   than K from every visited head; the generator explicitly enforces that.
#
# d) Placement legality
# - Every sprite state must fit in-grid.
# - Path markers never reuse a head, never touch any sprite state, and for every
#   earlier head are strictly farther than K unless they are that head's intended
#   immediate successor. This guarantees a unique nearest consumable winner.
# - Stranded dots also avoid all sprite states, avoid touching other markers, and
#   are placed at distance > K from every visited head.
#
# e) Visual observability
# - Human-facing cue 1: in the rightmost frame the arrow points one way while the
#   continuation goes to a closer dot elsewhere.
# - Human-facing cue 2: one leftover dot sits on that facing ray but at the wrong
#   distance, so the final output shows that the sprite stops because stride K is
#   exhausted, not because dots disappeared.
#
# f) Visual cleanliness
# - Single-pixel dots remain isolated.
# - No touching between dots and stamped sprite cells is allowed, so `_find_agent`
#   still recovers the triangle cleanly.
#
# g) Color inference
# - Colors follow the base task exactly: background, one actor color shared by
#   sprite and dots, plus an optional divider color. No color-id logic is used.
#
# h) New solver assumptions and generator guarantees
# - Solver assumption: the initial nearest marker is unique and cardinally
#   aligned, which reveals stride K.
# - Solver assumption: after K is fixed, any surviving dot closer than K would be
#   invalid input. The generator forbids that by construction.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

for _extra in (
    Path("/home/lixinyu/snap/new/solvers"),
    Path("/home/lixinyu/snap/new/arc2_solver_select"),
):
    _s = str(_extra)
    if _s not in sys.path:
        sys.path.append(_s)

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

LAST_CASE_META: dict | None = None


def _split_frames(g: np.ndarray) -> tuple[list[np.ndarray], int]:
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


def _sprite_cells(head: tuple[int, int], tname: str, d: int) -> set[tuple[int, int]]:
    hr, hc = head
    return {(hr + dr, hc + dc) for dr, dc in TEMPLATES[tname][d]}


def _head_fits(head: tuple[int, int], tname: str, d: int, h: int, w: int) -> bool:
    hr, hc = head
    for dr, dc in TEMPLATES[tname][d]:
        nr = hr + dr
        nc = hc + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return False
    return True


def _choose_target(
    head: tuple[int, int],
    markers: set[tuple[int, int]],
    tname: str,
    h: int,
    w: int,
    step_dist: int | None,
) -> tuple[int, tuple[int, int], int] | None:
    if not markers:
        return None

    ranked = sorted((_dist(head, pt), int(pt[0]), int(pt[1]), pt) for pt in markers)
    if step_dist is None:
        step_dist = ranked[0][0]
        nearest = [pt for dist, _y, _x, pt in ranked if dist == step_dist]
        assert len(nearest) == 1, f"head={head}: nearest marker tie at distance {step_dist}"
        nd = _direction(head, nearest[0])
        assert nd is not None, f"head={head}: nearest marker {nearest[0]} is not cardinally aligned"
        assert _head_fits(nearest[0], tname, nd, h, w), (
            f"head={head}: landing at {nearest[0]} with dir={nd} would place sprite out of bounds"
        )

    min_dist = ranked[0][0]
    assert min_dist >= step_dist, (
        f"head={head}: found marker closer than fixed stride {step_dist}: distance {min_dist}"
    )
    if min_dist > step_dist:
        return None

    exact: list[tuple[int, int, tuple[int, int]]] = []
    for pt in markers:
        if _dist(head, pt) != step_dist:
            continue
        nd = _direction(head, pt)
        assert nd is not None, f"head={head}: marker {pt} matches stride {step_dist} but is diagonal"
        assert _head_fits(pt, tname, nd, h, w), (
            f"head={head}: landing at {pt} with dir={nd} would place sprite out of bounds"
        )
        exact.append((int(pt[0]), int(pt[1]), pt))

    assert len(exact) == 1, (
        f"head={head}: expected one stride-{step_dist} target, got {[pt for _, _, pt in exact]}"
    )
    pt = exact[0][2]
    nd = _direction(head, pt)
    assert nd is not None
    return nd, pt, step_dist


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, agent, markers = found
    agent = {(int(y), int(x)) for y, x in agent}
    markers = {(int(y), int(x)) for y, x in markers}
    step_dist: int | None = None

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        choice = _choose_target(head, markers, tname, h, w, step_dist)
        if choice is None:
            break
        d, head, step_dist = choice
        agent = _sprite_cells(head, tname, d)
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        out[y, x] = actor
    for y, x in markers:
        out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue a triangular sprite along a fixed-stride nearest-dot route until that stride can no longer be used.

    Input structure:
    - Several equal-height frames are concatenated horizontally, with optional
      solid divider columns between them; the most common color is background.
    - Each frame contains one triangular sprite plus same-colored one-pixel dots,
      so the sprite must be recovered from its shape rather than from color alone.

    Rule:
    - Read only the rightmost frame as the current state.
    - Ignore the current facing when choosing the next target: the unique nearest
      remaining marker fixes the common Manhattan jump length K.
    - Each later move must land on the unique marker exactly K away; the sprite
      re-faces the travel direction after consuming that marker.
    - If no marker remains at distance K, stop immediately even if farther dots
      still exist, and keep those leftover dots in the output.

    Key insight: One nearby dot reveals the stride, but the arrow tip no longer
    dictates priority; after that, the sprite follows an equally spaced nearest
    chain and any wrong-distance dots are permanent leftovers.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    return _simulate_to_end(frames[-1], actor)


N_FRAMES_RANGE = (1, 5)
FRAME_MAX = 16
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 6, 15

STEPS_RANGE = (4, 13)
FIXED_STEP_CHOICES = (2, 3, 4)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_LAST_FRAME_PATH_MARKERS = 2

TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

MAX_ATTEMPTS = 1600
SEARCH_NODE_BUDGET = 24000
STRANDED_MARKERS_RANGE = (1, 4)

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
    if n_steps >= 10:
        return (2,)
    if n_steps >= 7:
        return (2, 3)
    return FIXED_STEP_CHOICES


def _compatible_multiples(n_frames: int, step_dist: int) -> list[int]:
    return [
        n
        for n in range(*STEPS_RANGE)
        if n % n_frames == 0
        and n // n_frames >= MIN_LAST_FRAME_PATH_MARKERS
        and step_dist in _step_options(n)
    ]


def _sample_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int, int, int | None, str, int]:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    step_candidates = [sd for sd in FIXED_STEP_CHOICES if _compatible_multiples(n_frames, sd)]
    step_weights = np.array([1.0 for _ in step_candidates], dtype=float)
    step_weights /= step_weights.sum()
    step_dist = int(rng.choice(step_candidates, p=step_weights))

    multiples = _compatible_multiples(n_frames, step_dist)
    weights = np.array([n / _STEP_AVAILABILITY[n] for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

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


def _candidate_compatible(
    nxt: tuple[int, int],
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
    step_dist: int,
) -> bool:
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, state_dirs[j])
        if any(_touching(nxt, cell) for cell in sprite):
            return False
        if j < len(heads) - 1 and _dist(old_head, nxt) <= step_dist:
            return False
    return True


def _visible_metrics(
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    n_frames: int,
    gap: int,
    step_dist: int,
) -> dict[str, int | bool]:
    last_idx = (n_frames - 1) * gap
    cur = heads[last_idx]
    facing = state_dirs[last_idx]
    next_dir = state_dirs[last_idx + 1]

    branch_dirs: set[int] = set()
    for pt in heads[last_idx + 1 :]:
        dd = _direction(cur, pt)
        if dd is not None:
            branch_dirs.add(dd)

    return {
        "last_idx": last_idx,
        "visible_nonfacing": next_dir != facing,
        "visible_branch_dirs": len(branch_dirs),
        "visible_reverse": next_dir == (facing + 2) % 4,
        "next_step_distance": _dist(cur, heads[last_idx + 1]),
        "step_distance": step_dist,
    }


def _validate_path(
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    n_frames: int,
    gap: int,
    step_dist: int,
) -> tuple[bool, int, list[int], dict[str, int | bool]]:
    step_dirs = state_dirs[1:]
    n_steps = len(step_dirs)
    dir_set = set(step_dirs)
    if len(dir_set) < MIN_DIR_DIVERSITY:
        return False, 0, [], {}
    if not (dir_set & {0, 2} and dir_set & {1, 3}):
        return False, 0, [], {}

    turns = sum(step_dirs[i] != step_dirs[i - 1] for i in range(1, n_steps))
    if turns < MIN_TURNS_ACCEPT or not (TURN_RATIO_LO <= turns / n_steps <= TURN_RATIO_HI):
        return False, turns, [], {}

    dists = [_dist(heads[i], heads[i + 1]) for i in range(n_steps)]
    if any(dist != step_dist for dist in dists):
        return False, turns, dists, {}

    visible = _visible_metrics(heads, state_dirs, n_frames, gap, step_dist)
    if not visible["visible_nonfacing"]:
        return False, turns, dists, visible

    return True, turns, dists, visible


def _build_path(
    rng: np.random.Generator,
    tname: str,
    n_steps: int,
    step_dist: int,
    h: int,
    w: int,
    n_frames: int,
    gap: int,
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]], dict[str, int | bool]] | None:
    min_turns = max(MIN_TURNS_ACCEPT, int(np.ceil(TURN_RATIO_LO * n_steps)))
    max_turns = min(n_steps - 1, int(np.floor(TURN_RATIO_HI * n_steps)))
    budget = [SEARCH_NODE_BUDGET]

    def dfs(
        step_idx: int,
        heads: list[tuple[int, int]],
        state_dirs: list[int],
        placements: list[tuple[tuple[int, int], int]],
        turns: int,
    ) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]], dict[str, int | bool]] | None:
        budget[0] -= 1
        if budget[0] < 0:
            return None

        if step_idx == n_steps:
            ok, _turns, _dists, visible = _validate_path(heads, state_dirs, n_frames, gap, step_dist)
            if ok:
                return heads, state_dirs, placements, visible
            return None

        cur = heads[-1]
        prev_dir = state_dirs[-1]
        dir_choices = list(map(int, rng.permutation(4)))
        for next_dir in dir_choices:
            next_turns = turns + int(step_idx > 0 and next_dir != prev_dir)
            remaining_after = n_steps - step_idx - 1
            if next_turns > max_turns or next_turns + remaining_after < min_turns:
                continue

            dr, dc = DIRS4[next_dir]
            nxt = (cur[0] + step_dist * dr, cur[1] + step_dist * dc)
            if nxt in heads:
                continue
            if any(_touching(nxt, marker) for marker in heads[1:]):
                continue
            if not _candidate_compatible(nxt, heads, state_dirs, tname, step_dist):
                continue

            trial = placements + [(nxt, next_dir)]
            if _shift_interval(trial, h, w, tname) is None:
                continue

            out = dfs(
                step_idx + 1,
                heads + [nxt],
                state_dirs + [next_dir],
                trial,
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


def _place_stranded_markers(
    rng: np.random.Generator,
    h: int,
    w: int,
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
    step_dist: int,
    last_idx: int,
    requested: int,
) -> list[tuple[int, int]] | None:
    path_markers = set(heads[1:])
    sprite_states = [_sprite_cells(head, tname, d) for head, d in zip(heads, state_dirs)]
    sprite_cells = set().union(*sprite_states)
    chosen: list[tuple[int, int]] = []

    cur = heads[last_idx]
    facing = state_dirs[last_idx]
    dr, dc = DIRS4[facing]
    ray_candidates: list[tuple[int, int]] = []
    for dist in range(step_dist + 1, max(h, w) + 1):
        pt = (cur[0] + dist * dr, cur[1] + dist * dc)
        if 0 <= pt[0] < h and 0 <= pt[1] < w:
            ray_candidates.append(pt)
    rng.shuffle(ray_candidates)

    def ok(pt: tuple[int, int]) -> bool:
        if pt in path_markers or pt in sprite_cells:
            return False
        if any(_dist(pt, head) <= step_dist for head in heads):
            return False
        if any(_touching(pt, other) for other in path_markers):
            return False
        if any(_touching(pt, other) for other in chosen):
            return False
        if any(_touching(pt, cell) for cell in sprite_cells):
            return False
        return True

    primary = next((pt for pt in ray_candidates if ok(pt)), None)
    if primary is None:
        return None
    chosen.append(primary)
    if requested == 1:
        return chosen

    cells = [(int(r), int(c)) for r in range(h) for c in range(w)]
    rng.shuffle(cells)
    for pt in cells:
        if ok(pt):
            chosen.append(pt)
            if len(chosen) == requested:
                return chosen

    return chosen if chosen else None


def _build_case(rng: np.random.Generator) -> dict | None:
    params = _sample_params(rng)
    n_frames, n_steps, gap, h, w, bg, actor, divider, tname, step_dist = params

    rollout = _build_path(rng, tname, n_steps, step_dist, h, w, n_frames, gap)
    if rollout is None:
        return None
    heads0, state_dirs0, placements0, visible0 = rollout

    shift = _shift_interval(placements0, h, w, tname)
    if shift is None:
        return None
    tr_lo, tr_hi, tc_lo, tc_hi = shift
    translations = [(tr, tc) for tr in range(tr_lo, tr_hi + 1) for tc in range(tc_lo, tc_hi + 1)]
    rng.shuffle(translations)

    requested_stranded = int(rng.integers(*STRANDED_MARKERS_RANGE))
    for tr, tc in translations:
        heads = [(r + tr, c + tc) for r, c in heads0]
        state_dirs = list(state_dirs0)
        placements = [(heads[i], state_dirs[i]) for i in range(n_steps + 1)]
        last_idx = (n_frames - 1) * gap

        stranded = _place_stranded_markers(
            rng, h, w, heads, state_dirs, tname, step_dist, last_idx, requested_stranded
        )
        if stranded is None or len(stranded) != requested_stranded:
            continue

        frames: list[np.ndarray] = []
        for fi in range(n_frames):
            si = fi * gap
            frames.append(
                _render_frame(
                    h,
                    w,
                    bg,
                    actor,
                    heads[si],
                    state_dirs[si],
                    tname,
                    list(heads[si + 1 :]) + stranded,
                )
            )

        out_frame = _render_frame(h, w, bg, actor, heads[-1], state_dirs[-1], tname, stranded)
        inp = _assemble_input(frames, bg, divider)

        found = _find_agent(frames[-1], actor)
        if found is None:
            continue
        f_tname, f_dir, f_head, _f_agent, f_markers = found
        if (f_tname, f_dir, f_head) != (tname, state_dirs[last_idx], heads[last_idx]):
            continue
        expected_markers = set(heads[last_idx + 1 :]) | set(stranded)
        if f_markers != expected_markers:
            continue
        if not np.array_equal(main(inp), out_frame):
            continue

        last_metrics = _visible_metrics(heads, state_dirs, n_frames, gap, step_dist)
        branch_dirs = set()
        cur = heads[last_idx]
        for pt in list(heads[last_idx + 1 :]) + stranded:
            dd = _direction(cur, pt)
            if dd is not None:
                branch_dirs.add(dd)

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
                "heads": list(heads),
                "state_dirs": list(state_dirs),
                "step_dirs": list(state_dirs[1:]),
                "step_distance": step_dist,
                "step_distances": [step_dist] * n_steps,
                "step_distance_unique_count": 1,
                "count_in_first_frame": n_steps + len(stranded),
                "count_in_last_input_frame": gap + len(stranded),
                "consumed_steps": n_steps,
                "stranded_markers": list(stranded),
                "stranded_count": len(stranded),
                "visible_nonfacing_last": bool(last_metrics["visible_nonfacing"]),
                "visible_reverse_last": bool(last_metrics["visible_reverse"]),
                "visible_branch_dirs_last_path_only": int(visible0["visible_branch_dirs"]),
                "visible_branch_dirs_last_total": len(branch_dirs),
                "primary_stranded_on_facing_ray": True,
            },
        }

    return None


def generate_input() -> np.ndarray:
    global LAST_CASE_META

    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            LAST_CASE_META = case["meta"]
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
