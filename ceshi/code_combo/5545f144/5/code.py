# Combo variant of 5545f144 (index 5)
# combo_spec: D5/0+D6/2
# Name: 标记转化+转弯处留标记

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg as _bg_color
from arc_utils import paste

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
# a) Concrete behavior:
#    The movement path is still the original one: from the current head, there is
#    exactly one valid half-line of future targets, and the sprite jumps to the
#    nearest target on it. The fusion changes what remains visible afterward.
#    Every consumed target persists as history instead of disappearing, but the
#    history splits into two colors:
#    - ordinary visited stops -> `trail`
#    - visited stops that were later departed via a 90-degree turn -> `turn`
#    Example fragment with head targets A -> B -> C:
#    - If A->B and B->C use the same direction, then once the sprite leaves B,
#      B becomes a `trail` pixel.
#    - If A->B is rightward and B->C is upward, then once the sprite leaves B,
#      B becomes a `turn` pixel.
#    This resolves the single-cell collision between "recolored visited marker"
#    and "turn marker" by making the turn marker the special recolored state of
#    that visited stop.
#
# b) Complexity calibration:
#    The variant uses exactly two extra foreground colors beyond the actor:
#    one for ordinary visited chain cells and one for corner cells. That keeps
#    the visual vocabulary small while still making the fused rule unmistakable.
#    Generation requires both trace colors to be visible in the last input frame
#    and in the output, preventing collapse to either single-variant behavior.
#
# c) Implicit assumption audit:
#    - Original `main()` assumed there was exactly one semantic foreground color.
#      That breaks here: actor, visited-chain, and turn-corner roles must be
#      separated.
#    - Original solver could start from the last frame. That is no longer enough
#      to identify which extra color means "ordinary visited" versus "turn".
#      The new solver therefore reconstructs the whole path from frame 0, then
#      tests the two possible color-role assignments against all snapshots.
#    - Original simulation deleted targets. Here consumed targets remain in a
#      persistent visited set, and their final color depends on the next move
#      taken from that stop.
#
# d) Placement legality:
#    The generator still ensures unique reachable half-lines and in-grid sprite
#    placements. It additionally enforces that no later sprite pose covers any
#    previously visited head, because every visited head must remain visible as
#    either `trail` or `turn` after the sprite moves on.
#
# e) Visual observability:
#    Humans can infer the rule from frame progression alone:
#    actor-colored future targets shrink, while two non-actor colors accumulate
#    exactly on already-consumed target cells. Corner-colored cells appear only
#    at bends in the visited polyline, so the split between "all visited" and
#    "turn-specific visited" is structurally readable.
#
# f) Visual cleanliness:
#    All traces are still isolated single pixels at original marker locations.
#    No offset decorations or stacked multi-cell symbols are introduced, so the
#    grid stays sparse and legible. The current head is the only visited stop
#    hidden at any given snapshot, because the sprite sits on top of it.
#
# g) Change inferability:
#    No hidden use of color ids is needed for the puzzle itself. The first frame
#    identifies the actor color by being the only frame without history marks.
#    Later frames show the same visited path split into two structurally distinct
#    subsets, allowing both humans and the solver to infer the color roles.

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left


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


def _turn_head_indices(state_dirs: list[int]) -> set[int]:
    return {
        i
        for i in range(1, len(state_dirs) - 1)
        if state_dirs[i + 1] != state_dirs[i]
    }


def _simulate_path_from_frame(
    frame: np.ndarray, actor: int
) -> tuple[str, list[tuple[int, int]], list[int]]:
    h, w = frame.shape
    found = _find_agent(frame, actor)
    assert found is not None, "failed to detect actor in the starting frame"
    tname, d, head, _agent, markers = found
    markers = {(int(y), int(x)) for y, x in markers}

    heads = [tuple(map(int, head))]
    state_dirs = [int(d)]
    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break
        offs = TEMPLATES[tname][state_dirs[-1]]
        choice = _choose_target(heads[-1], state_dirs[-1], markers, offs, h, w)
        assert choice is not None, "path simulation got stuck"
        nd, nxt = choice
        heads.append(tuple(map(int, nxt)))
        state_dirs.append(int(nd))
        markers.discard(heads[-1])

    assert not markers, "starting frame contains stranded future markers"
    return tname, heads, state_dirs


def _render_state(
    h: int,
    w: int,
    bg: int,
    actor: int,
    trail: int,
    turn: int,
    head: tuple[int, int],
    d: int,
    tname: str,
    future_markers: list[tuple[int, int]],
    trail_marks: list[tuple[int, int]],
    turn_marks: list[tuple[int, int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    for y, x in trail_marks:
        frame[y, x] = trail
    for y, x in turn_marks:
        frame[y, x] = turn
    for y, x in future_markers:
        frame[y, x] = actor
    patch, ar, ac = _template_patch(tname, d, actor, bg)
    paste(frame, patch, head[0] - ar, head[1] - ac, transparent=bg)
    return frame


def _render_snapshot_from_path(
    h: int,
    w: int,
    bg: int,
    actor: int,
    trail: int,
    turn: int,
    tname: str,
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    step_idx: int,
) -> np.ndarray:
    turn_idxs = _turn_head_indices(state_dirs)
    trail_marks = [
        heads[i] for i in range(1, step_idx + 1) if i not in turn_idxs
    ]
    turn_marks = [
        heads[i] for i in range(1, step_idx + 1) if i in turn_idxs
    ]
    return _render_state(
        h=h,
        w=w,
        bg=bg,
        actor=actor,
        trail=trail,
        turn=turn,
        head=heads[step_idx],
        d=state_dirs[step_idx],
        tname=tname,
        future_markers=heads[step_idx + 1 :],
        trail_marks=trail_marks,
        turn_marks=turn_marks,
    )


def _infer_actor_and_extras(
    g: np.ndarray, frames: list[np.ndarray], bg: int
) -> tuple[int, tuple[int, int]]:
    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}

    first_colors = set(map(int, frames[0].ravel())) - {bg}
    assert len(first_colors) == 1, f"expected one non-bg color in frame 0, got {first_colors}"
    actor = next(iter(first_colors))

    colors = set(map(int, g.ravel()))
    extras = tuple(sorted(colors - {bg, actor} - div_colors))
    assert len(extras) == 2, f"expected two history colors, got {extras}"
    return actor, extras


def _match_snapshot_sequence(
    frames: list[np.ndarray],
    actor: int,
    trail: int,
    turn: int,
) -> tuple[str, list[tuple[int, int]], list[int], int] | None:
    start = frames[0]
    h, w = start.shape
    bg = _bg_color(start)

    tname, heads, state_dirs = _simulate_path_from_frame(start, actor)
    n_frames = len(frames)
    n_steps = len(heads) - 1
    if n_steps % n_frames != 0:
        return None
    gap = n_steps // n_frames

    for fi, frame in enumerate(frames):
        step_idx = fi * gap
        expected = _render_snapshot_from_path(
            h=h,
            w=w,
            bg=bg,
            actor=actor,
            trail=trail,
            turn=turn,
            tname=tname,
            heads=heads,
            state_dirs=state_dirs,
            step_idx=step_idx,
        )
        if not np.array_equal(frame, expected):
            return None
    return tname, heads, state_dirs, gap


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance the triangular sprite to the end while preserving two kinds of visited markers.

    Input structure:
    - Frames are concatenated left-to-right with solid divider columns; the background is the most common color.
    - Frame 0 contains only the actor-colored sprite and future target dots; later frames add two history colors: one for ordinary visited stops and one for visited stops that were departed via a 90-degree turn.

    Rule:
    - Reconstruct the movement path from the leftmost frame, where all future targets still share the actor color.
    - Test the two possible assignments of the extra colors against the entire snapshot sequence; the correct assignment is the one whose rendered snapshots match every frame exactly.
    - Continue the same nearest-marker half-line chase to the end: every consumed target remains visible, colored `trail` after a straight departure and `turn` after a turning departure, while the current head is hidden under the sprite.

    Key insight:
    - The path geometry itself is unchanged from the original task; the new information is the split of already-consumed targets into straight-history versus corner-history, and the multiple snapshots make that split identifiable.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)
    actor, extras = _infer_actor_and_extras(g, frames, bg)

    matches: list[tuple[int, int, str, list[tuple[int, int]], list[int], int]] = []
    for trail, turn in itertools.permutations(extras, 2):
        path = _match_snapshot_sequence(frames, actor, trail, turn)
        if path is not None:
            tname, heads, state_dirs, gap = path
            matches.append((trail, turn, tname, heads, state_dirs, gap))

    assert len(matches) == 1, f"expected a unique color-role assignment, got {len(matches)}"
    trail, turn, tname, heads, state_dirs, _gap = matches[0]

    h, w = frames[0].shape
    return _render_snapshot_from_path(
        h=h,
        w=w,
        bg=bg,
        actor=actor,
        trail=trail,
        turn=turn,
        tname=tname,
        heads=heads,
        state_dirs=state_dirs,
        step_idx=len(heads) - 1,
    )


# --------------- generator (combo D5/0 + D6/2) ---------------

N_FRAMES_RANGE = (2, 5)          # 2..4 frames; history must already be visible
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

TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

MAX_ATTEMPTS = 700

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(2, 5)) for n in range(*STEPS_RANGE)}


def _visited_clear_of_pose(
    visited_heads: list[tuple[int, int]],
    head: tuple[int, int],
    d: int,
    tname: str,
) -> bool:
    if not visited_heads:
        return True
    sprite = _sprite_cells(head, tname, d)
    return sprite.isdisjoint(set(visited_heads))


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


def _validate_path(
    step_dirs: list[int], heads: list[tuple[int, int]]
) -> tuple[bool, int, list[int]]:
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


def _visited_clear_of_all_future_sprites(
    heads: list[tuple[int, int]], state_dirs: list[int], tname: str
) -> bool:
    for i in range(1, len(heads)):
        pt = heads[i]
        for j in range(i + 1, len(heads)):
            if pt in _sprite_cells(heads[j], tname, state_dirs[j]):
                return False
    return True


def _sample_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int, int, int, int, int, str]:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0]
    weights = np.array([(n**3) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = rng.permutation(10)
    bg, actor, trail, turn, divider = map(int, palette[:5])

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames, n_steps, gap, h, w, bg, actor, trail, turn, divider, tname


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
            if not _visited_clear_of_pose(heads[1:], nxt, next_dir, tname):
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
    n_frames, n_steps, gap, h, w, bg, actor, trail, turn, divider, tname = _sample_params(rng)
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
    state_dirs = list(state_dirs)

    if not _visited_clear_of_all_future_sprites(heads, state_dirs, tname):
        return None

    frames: list[np.ndarray] = []
    for fi in range(n_frames):
        si = fi * gap
        frames.append(
            _render_snapshot_from_path(
                h=h,
                w=w,
                bg=bg,
                actor=actor,
                trail=trail,
                turn=turn,
                tname=tname,
                heads=heads,
                state_dirs=state_dirs,
                step_idx=si,
            )
        )

    out_frame = _render_snapshot_from_path(
        h=h,
        w=w,
        bg=bg,
        actor=actor,
        trail=trail,
        turn=turn,
        tname=tname,
        heads=heads,
        state_dirs=state_dirs,
        step_idx=n_steps,
    )
    inp = _assemble_input(frames, bg, divider)

    first_colors = set(map(int, frames[0].ravel())) - {bg}
    if first_colors != {actor}:
        return None
    if np.count_nonzero(frames[-1] == trail) <= 0 or np.count_nonzero(frames[-1] == turn) <= 0:
        return None
    if np.count_nonzero(out_frame == trail) <= 0 or np.count_nonzero(out_frame == turn) <= 0:
        return None

    found = _find_agent(frames[0], actor)
    if found is None:
        return None
    f_tname, f_dir, f_head, _f_agent, f_markers = found
    if (f_tname, f_dir, f_head) != (tname, state_dirs[0], heads[0]):
        return None
    if f_markers != set(heads[1:]):
        return None

    matches = []
    for trail_c, turn_c in itertools.permutations((trail, turn), 2):
        path = _match_snapshot_sequence(frames, actor, trail_c, turn_c)
        if path is not None:
            matches.append((trail_c, turn_c, path))
    if len(matches) != 1 or matches[0][0] != trail or matches[0][1] != turn:
        return None

    if not np.array_equal(main(inp), out_frame):
        return None

    turn_idxs = _turn_head_indices(state_dirs)
    last_idx = (n_frames - 1) * gap
    visible_trail_last = sum(frames[-1].ravel() == trail)
    visible_turn_last = sum(frames[-1].ravel() == turn)

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
            "trail": trail,
            "turn": turn,
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
            "turn_head_indices": sorted(turn_idxs),
            "visible_trail_last_frame": int(visible_trail_last),
            "visible_turn_last_frame": int(visible_turn_last),
            "visible_trail_output": int(np.count_nonzero(out_frame == trail)),
            "visible_turn_output": int(np.count_nonzero(out_frame == turn)),
            "visited_before_last_frame": last_idx,
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
