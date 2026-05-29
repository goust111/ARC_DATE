# Combo variant of 5545f144 (index 12)
# combo_spec: D7/3+D15/1
# Name: 消耗后模板变形+帧间模板差异编码

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg as _bg_color
from arc_utils import paste

# -----------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concretization and behavior trace
# - The visible vocabulary stays the same: each frame has one actor-colored arrow
#   sprite plus actor-colored single-pixel markers; frames are concatenated
#   horizontally and may be separated by a solid divider color.
# - The fused rule combines two changes. First, every consumed marker advances the
#   sprite to the next template in the fixed four-phase cycle
#   `hollow3 -> filled4 -> stem4 -> midfilled6 -> ...`. Second, the input frames
#   are chosen so those template phases are visibly different across snapshots,
#   making the frame-to-frame silhouette itself encode how many consumptions have
#   happened between snapshots.
# - Concrete trace: suppose frame 0 shows `hollow3`, frame 1 shows `midfilled6`,
#   frame 2 shows `stem4`, and each later frame has exactly three fewer markers.
#   Then the viewer can read that every snapshot is three consumptions later than
#   the previous one, because the template index also advances by +3 mod 4 each
#   time. From the last frame, three markers remain, so the final template must be
#   three more phase steps ahead of that last visible silhouette. The base task
#   would keep one template forever, and the D7-only variant would morph but would
#   not force the multi-frame phase encoding to be visible.
#
# b) Complexity calibration
# - The cycle is kept small and fixed at exactly four templates. That makes the
#   morphology obvious without introducing a large search space or hidden mapping.
# - To keep the frame-encoding visually strong, the generator only uses 3 or 4
#   input frames and fixes the snapshot cadence gap to 3 steps. Therefore adjacent
#   frames always show different templates, and the visible snapshot templates are
#   all distinct (`3` distinct silhouettes for 3 frames, `4` for 4 frames).
# - This avoids collapsing into the simpler D15 single-pair alternation pattern or
#   into the original task where every frame would show the same silhouette.
#
# c) Implicit assumption audit
# - Original `_simulate_to_end` assumed the detected template never changes. That
#   breaks immediately: after every landing the solver must update both direction
#   and template phase.
# - Original `main()` effectively ignored all earlier frames and trusted the last
#   frame alone. In the fused version, `main()` now also checks the whole snapshot
#   sequence: marker counts must decrease by a constant gap, and template indices
#   must advance by that same gap modulo 4.
# - Original target selection only needed the current template. Here a move is
#   valid only when the translated current body fits and the post-consumption next
#   template also fits at the landing head.
# - Generator-side placement and shift logic can no longer assume one fixed body
#   shape; every state must carry its own `(head, direction, template)` triple.
#
# d) Placement legality
# - Each future marker is still a future head location. The generator rejects any
#   candidate marker that touches an earlier sprite body, duplicates an earlier
#   head, or creates a second reachable half-line from an earlier state.
# - Fit checks are done against the actual template for every state in the rollout,
#   including the morphed landing state after each consumption, so there is no
#   clipping and no off-grid stamping.
#
# e) Visual observability
# - Humans can infer the rule from shape alone: the four templates have clearly
#   different silhouettes and use the same color, so the relevant information is
#   entirely geometric and visible.
# - Because the cadence gap is always 3, consecutive frames visibly march through
#   different phases of the same four-template cycle while the marker counts drop
#   by 3 each time. That repeated agreement is enough for a human to infer that
#   frame silhouette encodes the snapshot step count modulo 4.
#
# f) Visual cleanliness
# - Markers remain isolated single pixels and are kept away from the sprite body by
#   the same non-touching checks used in the base task family, so the silhouette of
#   each template stays readable.
# - Divider columns stay solid and use a distinct color, preventing any ambiguity
#   about frame boundaries.
#
# g) Change inferability
# - No color remapping is introduced. The only changing attributes are direction,
#   position, remaining marker count, and template phase, all of which are directly
#   visible in the grid.
# - The new solver assumes every frame contains a uniquely detectable sprite and
#   that the snapshot sequence is consistent with one constant gap and one fixed
#   four-template cycle. The generator verifies these assumptions frame by frame
#   before accepting a case.
# -----------------------------------------------------------------------------

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
TEMPLATE_ORDER = ("hollow3", "filled4", "stem4", "midfilled6")
TEMPLATE_INDEX = {name: i for i, name in enumerate(TEMPLATE_ORDER)}


def _next_template(tname: str, steps: int = 1) -> str:
    return TEMPLATE_ORDER[(TEMPLATE_INDEX[tname] + steps) % len(TEMPLATE_ORDER)]


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


def _sprite_cells(head: tuple[int, int], tname: str, d: int) -> set[tuple[int, int]]:
    hy, hx = head
    return {(hy + oy, hx + ox) for oy, ox in TEMPLATES[tname][d]}


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


def _fits_translated(
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


def _fits_state(head: tuple[int, int], tname: str, d: int, h: int, w: int) -> bool:
    return all(0 <= y < h and 0 <= x < w for y, x in _sprite_cells(head, tname, d))


def _choose_target(
    head: tuple[int, int],
    d: int,
    tname: str,
    markers: set[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    hy, hx = head
    offs = TEMPLATES[tname][d]
    next_tname = _next_template(tname)

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

    reachable: list[tuple[int, int, tuple[int, int]]] = []
    for dd in range(4):
        if dd == (d + 2) % 4:
            continue
        for dist, pt in candidates(dd):
            if not _fits_translated(head, pt, offs, h, w):
                continue
            if not _fits_state(pt, next_tname, dd, h, w):
                continue
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
    return dd, pt


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, _agent, markers = found
    markers = {(int(y), int(x)) for y, x in markers}
    agent = _sprite_cells(head, tname, d)

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        choice = _choose_target(head, d, tname, markers, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d} "
            f"template={tname}"
        )
        nd, (ty, tx) = choice

        head = (ty, tx)
        d = nd
        tname = _next_template(tname)
        agent = _sprite_cells(head, tname, d)
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        out[y, x] = actor
    return out


def _validate_frame_encoding(
    frame_states: list[tuple[str, int, tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]]
) -> int | None:
    if len(frame_states) < 2:
        return None

    marker_counts = [len(state[4]) for state in frame_states]
    gaps = [marker_counts[i] - marker_counts[i + 1] for i in range(len(marker_counts) - 1)]
    gap = gaps[0]
    assert gap > 0 and all(g == gap for g in gaps), f"inconsistent snapshot gaps: {marker_counts}"

    template_idx = [TEMPLATE_INDEX[state[0]] for state in frame_states]
    phase_steps = [
        (template_idx[i + 1] - template_idx[i]) % len(TEMPLATE_ORDER)
        for i in range(len(template_idx) - 1)
    ]
    expected = gap % len(TEMPLATE_ORDER)
    assert expected != 0, f"template phase would not change across frames: gap={gap}"
    assert all(step == expected for step in phase_steps), (
        f"template steps {phase_steps} disagree with gap={gap} (expected mod-4 step {expected})"
    )
    if len(frame_states) >= 3:
        assert len(set(template_idx)) >= 3, (
            f"snapshot templates are not visibly diverse enough: {template_idx}"
        )
    return gap


def _find_actor_color(g: np.ndarray, bg: int) -> int:
    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    return next(iter(colors - {bg} - div_colors))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue a multi-frame marker-consuming arrow simulation whose body phase changes after every landing.

    Input structure:
    - The input is 3 or 4 same-height frames concatenated horizontally, optionally split by solid divider columns; the most common color is background.
    - In each frame, one actor-colored triangular sprite shares its color with the remaining one-pixel markers.
    - The sprite template may differ from frame to frame, because template type now records which consumption phase that snapshot belongs to.

    Rule:
    - Treat the frames as equally spaced checkpoints from one run. Detect the sprite in every frame, verify that the marker counts drop by a constant gap, and verify that the visible template indices advance by that same gap modulo the four-template cycle.
    - Start from the rightmost frame. From the current head, exactly one non-reversing cardinal half-line has reachable markers; move to the nearest one.
    - Each landing consumes that marker, turns the sprite to the travel direction, and advances the body to the next template in the cycle `hollow3 -> filled4 -> stem4 -> midfilled6 -> ...`.
    - Continue until no markers remain, then output only the final sprite on a clean background.

    Key insight: the earlier frames are not redundant decoration; their silhouette progression and marker-count progression jointly reveal the snapshot step cadence, so template type is a visible encoding of consumed-step count.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)
    actor = _find_actor_color(g, bg)

    frame_states: list[
        tuple[str, int, tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]
    ] = []
    for frame in frames:
        found = _find_agent(frame, actor)
        assert found is not None, "every frame must contain a detectable sprite"
        frame_states.append(found)

    _validate_frame_encoding(frame_states)
    return _simulate_to_end(frames[-1], actor)


# --------------- generator ---------------

N_FRAMES_RANGE = (3, 5)  # 3..4 frames
FRAME_MAX = 16
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 6, 15

STEPS_RANGE = (8, 13)  # 8..12 inclusive; valid choices under constraints become 9 or 12
STEP_DIST_RANGE = (2, 8)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

MAX_ATTEMPTS = 500

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(3, 5)) for n in range(*STEPS_RANGE)}
_PATH_LIBRARY: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    3: (
        (
            (0, 0),
            (2, 0),
            (4, 0),
            (4, -6),
            (1, -6),
            (1, -9),
            (1, -12),
            (-4, -12),
            (-8, -12),
            (-8, -8),
        ),
        (
            (0, 0),
            (0, 3),
            (-5, 3),
            (-7, 3),
            (-10, 3),
            (-10, -1),
            (-6, -1),
            (-3, -1),
            (-3, -7),
            (-3, -10),
        ),
    ),
    4: (
        (
            (0, 0),
            (6, 0),
            (6, 4),
            (10, 4),
            (10, 11),
            (5, 11),
            (5, 8),
            (1, 8),
            (1, 12),
            (-2, 12),
            (-2, 5),
            (-2, 3),
            (3, 3),
        ),
        (
            (0, 0),
            (0, -6),
            (5, -6),
            (10, -6),
            (12, -6),
            (12, -2),
            (12, 4),
            (12, 6),
            (7, 6),
            (7, -1),
            (7, -3),
            (3, -3),
            (3, 1),
        ),
    ),
}


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


def _state_templates(start_tname: str, n_steps: int) -> list[str]:
    return [_next_template(start_tname, i) for i in range(n_steps + 1)]


def _shift_interval(
    placements: list[tuple[tuple[int, int], int, str]],
    h: int,
    w: int,
) -> tuple[int, int, int, int] | None:
    tr_lo, tr_hi = -10**9, 10**9
    tc_lo, tc_hi = -10**9, 10**9
    for (hr, hc), d, tname in placements:
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
) -> tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int | None,
    str,
    tuple[tuple[int, int], ...],
    int,
]:
    n_frames = 3 + int(rng.integers(0, 2))
    n_steps = n_frames * 3
    gap = 3
    # The path templates below are pre-validated on a 15x15 frame. Fixing the
    # frame size keeps both 3-frame and 4-frame branches reliably generatable.
    h = 15
    w = 15
    palette = rng.permutation(10)
    bg, actor = int(palette[0]), int(palette[1])
    divider = int(palette[2]) if n_frames > 1 else None
    start_tname = str(rng.choice(TEMPLATE_ORDER))
    rel_heads = _PATH_LIBRARY[n_frames][int(rng.integers(0, len(_PATH_LIBRARY[n_frames])))]
    rotation = int(rng.integers(0, 4))
    return n_frames, n_steps, gap, h, w, bg, actor, divider, start_tname, rel_heads, rotation


def _rotate_point(pt: tuple[int, int], quarter_turns: int) -> tuple[int, int]:
    y, x = pt
    for _ in range(quarter_turns % 4):
        y, x = x, -y
    return y, x


def _rotate_dir(d: int, quarter_turns: int) -> int:
    return (d + quarter_turns) % 4


def _heads_to_step_dirs(heads: list[tuple[int, int]]) -> list[int]:
    step_dirs: list[int] = []
    for i in range(len(heads) - 1):
        d = _direction(heads[i], heads[i + 1])
        assert d is not None, f"non-cardinal step in abstract path: {heads[i]} -> {heads[i + 1]}"
        step_dirs.append(d)
    return step_dirs


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
    state_templates: list[str],
    step_dirs: list[int],
) -> bool:
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, state_templates[j], state_dirs[j])
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
    start_tname: str,
    rel_heads: tuple[tuple[int, int], ...],
    rotation: int,
    h: int,
    w: int,
) -> tuple[
    list[tuple[int, int]],
    list[int],
    list[str],
    list[tuple[tuple[int, int], int, str]],
    list[int],
] | None:
    heads = [_rotate_point(head, rotation) for head in rel_heads]
    step_dirs = [_rotate_dir(d, rotation) for d in _heads_to_step_dirs(list(rel_heads))]
    n_steps = len(step_dirs)
    templates = _state_templates(start_tname, n_steps)
    state_dirs: list[int] = [step_dirs[0]]
    fit_placements: list[tuple[tuple[int, int], int, str]] = [(heads[0], step_dirs[0], templates[0])]

    for i in range(n_steps):
        cur = heads[i]
        cur_dir = state_dirs[i]
        next_dir = step_dirs[i]
        next_tname = templates[i + 1]
        if next_dir == (cur_dir + 2) % 4:
            return None
        nxt = heads[i + 1]
        if nxt in heads[: i + 1]:
            return None
        if any(_touching(nxt, m) for m in heads[1 : i + 1]):
            return None
        if not _marker_compatible_with_existing(
            nxt, heads[: i + 1], state_dirs, templates, step_dirs
        ):
            return None
        trial_fit = fit_placements + [
            (nxt, cur_dir, templates[i]),
            (nxt, next_dir, next_tname),
        ]
        if _shift_interval(trial_fit, h, w) is None:
            return None
        state_dirs.append(next_dir)
        fit_placements = trial_fit

    return heads, state_dirs, templates, fit_placements, step_dirs


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


def _simulate_constant_template(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, _agent, markers = found
    markers = {(int(y), int(x)) for y, x in markers}
    agent = _sprite_cells(head, tname, d)

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

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

        reachable: list[tuple[int, int, tuple[int, int]]] = []
        for dd in range(4):
            if dd == (d + 2) % 4:
                continue
            for dist, pt in candidates(dd):
                if _fits_translated(head, pt, TEMPLATES[tname][d], h, w):
                    reachable.append((dist, dd, pt))

        if not reachable:
            break

        dirs_with_candidates = {dd for _, dd, _ in reachable}
        if len(dirs_with_candidates) != 1:
            break
        reachable.sort()
        _, nd, head = reachable[0]
        d = nd
        agent = _sprite_cells(head, tname, d)
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        out[y, x] = actor
    return out


def _build_case(rng: np.random.Generator) -> dict | None:
    (
        n_frames,
        n_steps,
        gap,
        h,
        w,
        bg,
        actor,
        divider,
        start_tname,
        rel_heads,
        rotation,
    ) = _sample_params(rng)

    rollout = _rollout_path(start_tname, rel_heads, rotation, h, w)
    if rollout is None:
        return None
    heads, state_dirs, state_templates, placements, step_dirs = rollout

    ok, turns, dists = _validate_path(step_dirs, heads)
    if not ok:
        return None

    shift = _shift_interval(placements, h, w)
    if shift is None:
        return None
    tr_lo, tr_hi, tc_lo, tc_hi = shift
    tr = int(rng.integers(tr_lo, tr_hi + 1))
    tc = int(rng.integers(tc_lo, tc_hi + 1))
    heads = [(r + tr, c + tc) for r, c in heads]

    frames: list[np.ndarray] = []
    snapshot_templates: list[str] = []
    snapshot_marker_counts: list[int] = []
    for fi in range(n_frames):
        si = fi * gap
        snapshot_templates.append(state_templates[si])
        snapshot_marker_counts.append(len(heads[si + 1 :]))
        frames.append(
            _render_frame(
                h,
                w,
                bg,
                actor,
                heads[si],
                state_dirs[si],
                state_templates[si],
                heads[si + 1 :],
            )
        )

    out_frame = _render_frame(
        h, w, bg, actor, heads[-1], state_dirs[-1], state_templates[-1], []
    )
    inp = _assemble_input(frames, bg, divider)

    if gap != 3:
        return None
    if len(set(snapshot_templates)) != n_frames:
        return None
    if snapshot_marker_counts != [n_steps - i * gap for i in range(n_frames)]:
        return None

    for fi, frame in enumerate(frames):
        found = _find_agent(frame, actor)
        if found is None:
            return None
        f_tname, f_dir, f_head, _f_agent, f_markers = found
        si = fi * gap
        if (f_tname, f_dir, f_head) != (state_templates[si], state_dirs[si], heads[si]):
            return None
        if f_markers != set(heads[si + 1 :]):
            return None

    frame_states = [_find_agent(frame, actor) for frame in frames]
    if any(state is None for state in frame_states):
        return None
    assert all(state is not None for state in frame_states)
    inferred_gap = _validate_frame_encoding(frame_states)  # type: ignore[arg-type]
    if inferred_gap != gap:
        return None

    if np.array_equal(out_frame, frames[-1]):
        return None
    if np.array_equal(_simulate_constant_template(frames[-1], actor), out_frame):
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
            "start_template": start_tname,
            "snapshot_templates": list(snapshot_templates),
            "snapshot_marker_counts": list(snapshot_marker_counts),
            "bg": bg,
            "actor": actor,
            "divider": divider,
            "total_steps": n_steps,
            "gap": gap,
            "phase_stride_mod_4": gap % 4,
            "turns": turns,
            "turn_ratio": turns / n_steps,
            "direction_diversity": len(set(step_dirs)),
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "state_templates": list(state_templates),
            "heads": list(heads),
            "step_distances": dists,
            "count_in_first_frame": n_steps,
            "count_in_last_input_frame": gap,
            "visible_template_count": len(set(snapshot_templates)),
            "final_template_differs_from_last_frame": state_templates[-1] != snapshot_templates[-1],
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
