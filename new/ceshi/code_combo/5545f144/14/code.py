# Combo variant of 5545f144 (index 14)
# combo_spec: D9/0+D20/0
# Name: 不等间隔快照+分隔线颜色编码帧间关系

from __future__ import annotations

from itertools import permutations

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left

# Divider colors are fixed at the task level, but their meaning is not.
# Each sample randomizes which of these colors means 1/2/3 consumed markers.
DIVIDER_COLORS = (1, 2, 4)  # blue, red, yellow

# ---------------------------------------------------------------------------
# Design analysis for the D9/0 + D20/0 combo
#
# a) Concrete behavior and trace
#    - The canvas still contains one moving triangle sprite and same-colored
#      single-pixel markers, arranged as horizontal snapshots.
#    - Visible snapshots are no longer evenly spaced. We show exactly four visible
#      frames at steps [0, g1, g1+g2, g1+g2+g3], where {g1, g2, g3} is a
#      permutation of {1, 2, 3}. So the marker counts drop by three different
#      amounts across the visible sequence.
#    - Each full-height divider column now carries information. The first three
#      dividers sit between visible frames and their colors encode g1/g2/g3.
#      A fourth trailing divider sits to the right of the last visible frame and
#      encodes the hidden gap from that last frame to the desired output.
#    - Example: visible marker counts [9, 8, 6, 3] imply visible gaps [1, 2, 3].
#      If the divider colors are [yellow, blue, red, blue], then within this
#      sample yellow=1, blue=2, red=3, so the trailing blue divider means:
#      starting from the rightmost visible frame, consume 2 more markers and stop.
#
# b) Complexity calibration
#    - The visible gaps are restricted to the small set {1, 2, 3}. This keeps the
#      color alphabet small and gives a complete in-sample legend: all three gap
#      sizes appear once among the visible dividers.
#    - The hidden gap is forced to differ from the last visible gap, so the task
#      cannot degenerate into "repeat the last interval" or "ignore the divider".
#    - A positive hidden tail remains after the output snapshot, so running the
#      last visible frame to exhaustion is also incorrect.
#
# c) Implicit-assumption audit for main()
#    - `_split_frames` can no longer assume every divider lies between two visible
#      frames; the final divider is followed by the hidden output rather than by a
#      visible frame. The parser therefore returns all divider colors and simply
#      ignores the empty segment after the trailing divider.
#    - Actor detection cannot treat divider colors as potential sprite colors.
#      The generator reserves actor/bg colors outside `DIVIDER_COLORS`.
#    - `main()` cannot assume the last visible marker count already determines how
#      far to simulate. It must first recover the per-sample color->gap mapping
#      from visible count differences, then read the trailing divider.
#    - `_simulate_steps` assumes the requested hidden gap is no larger than the
#      remaining marker count; the generator enforces `hidden_tail >= 1`.
#
# d) Placement legality
#    - Generation remains path-first. It creates a full legal head path, checks
#      no 180-degree reversals, prevents ambiguous half-line choices from any past
#      head, and ensures every rotated sprite placement fits after one translation.
#    - Every visible frame is rendered from an actual rollout state, with markers
#      equal to the future heads still to be consumed. The output is another real
#      intermediate state on the same path.
#
# e) Visual observability
#    - Humans can read the visible gaps directly from marker-count differences.
#      Because each visible divider color is paired with one of those differences,
#      the sample itself teaches the mapping from color to consumed-step count.
#    - The trailing divider then reuses one of those colors to request the hidden
#      extrapolation interval. Earlier frames are therefore necessary: without the
#      visible gaps, the final divider's color has no meaning.
#
# f) Visual cleanliness
#    - The only new element is one extra full-height divider at the far right.
#      It does not overlap the sprite or markers, and the frame interiors keep the
#      original clean vocabulary of one triangle plus isolated dots.
#    - Restricting gaps to 1/2/3 keeps divider colors and count differences easy
#      to compare at a glance instead of making the strip visually noisy.
#
# g) Color inference
#    - No rule depends on numeric color ids. The mapping from divider color to gap
#      is randomized per sample, inferred only from structure: each visible divider
#      color corresponds to the drop in future-marker count across its two frames.
#    - The generator guarantees that all three colors appear among visible dividers
#      before one of them is reused by the trailing divider, so the hidden divider
#      never introduces an unseen color category.
#
# h) New main() assumptions and generator guarantees
#    - Assumption: there are four visible frames and four divider columns. The
#      generator always emits that layout and verifies `_split_frames` recovers it.
#    - Assumption: visible frame marker counts strictly decrease by {1,2,3} in
#      some order and uniquely determine a color->gap legend. The generator uses
#      exactly one instance of each visible gap.
#    - Assumption: simulating `hidden_gap` steps from the last visible frame gives
#      a different state than either simulating `last_visible_gap` steps or going
#      to exhaustion. The generator rejects degenerate cases with both checks.
# ---------------------------------------------------------------------------


def _split_frames(g: np.ndarray) -> tuple[list[np.ndarray], list[int], int]:
    """
    Split a horizontally concatenated strip into visible frames and divider colors.

    The final divider is allowed to trail the last visible frame, so there may be
    one more divider than there are gaps between visible frames.
    """
    h, w = g.shape
    bg = _bg_color(g)

    frames: list[np.ndarray] = []
    divider_colors: list[int] = []
    start = 0
    for c in range(w):
        col = g[:, c]
        if len(set(map(int, col))) == 1 and int(col[0]) in DIVIDER_COLORS:
            if c > start:
                frames.append(g[:, start:c])
            divider_colors.append(int(col[0]))
            start = c + 1
    if start < w:
        frames.append(g[:, start:w])

    return frames, divider_colors, bg


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


def _simulate_steps(frame: np.ndarray, actor: int, steps: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, agent, markers = found
    agent = {(int(y), int(x)) for y, x in agent}
    markers = {(int(y), int(x)) for y, x in markers}

    assert 0 <= steps <= len(markers), (
        f"requested {steps} steps, but only {len(markers)} markers remain"
    )

    for _ in range(int(steps)):
        offs = TEMPLATES[tname][d]
        choice = _choose_target(head, d, markers, offs, h, w)
        assert choice is not None, (
            f"stuck with {len(markers)} markers at head={head} dir={d}"
        )
        nd, (ty, tx) = choice
        head = (ty, tx)
        d = nd
        agent = {(ty + oy, tx + ox) for oy, ox in TEMPLATES[tname][d]}
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()
    return _simulate_steps(frame, actor, len(found[4]))


def _detect_actor(g: np.ndarray, bg: int, divider_colors: list[int]) -> int:
    colors = set(map(int, g.ravel()))
    actor_colors = colors - {bg} - set(divider_colors)
    assert len(actor_colors) == 1, f"expected exactly one actor color, got {actor_colors}"
    return next(iter(actor_colors))


def _frame_marker_counts(frames: list[np.ndarray], actor: int) -> list[int]:
    counts: list[int] = []
    for i, frame in enumerate(frames):
        found = _find_agent(frame, actor)
        assert found is not None, f"could not recover sprite from frame {i}"
        counts.append(len(found[-1]))
    return counts


def _infer_hidden_gap(frames: list[np.ndarray], divider_colors: list[int], actor: int) -> int:
    assert len(frames) >= 2, "need at least two visible frames"
    assert len(divider_colors) == len(frames), (
        "expected one trailing divider in addition to the visible inter-frame dividers"
    )

    counts = _frame_marker_counts(frames, actor)
    visible_gaps = [counts[i] - counts[i + 1] for i in range(len(counts) - 1)]
    assert all(gap > 0 for gap in visible_gaps), f"marker counts must decrease: {counts}"

    mapping: dict[int, int] = {}
    for color, gap in zip(divider_colors[:-1], visible_gaps):
        prev = mapping.get(int(color))
        if prev is None:
            mapping[int(color)] = int(gap)
        else:
            assert prev == gap, f"inconsistent divider mapping for color {color}: {prev} vs {gap}"

    hidden_color = int(divider_colors[-1])
    assert hidden_color in mapping, f"hidden divider color {hidden_color} was not taught visibly"
    return mapping[hidden_color]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Predict the next hidden triangular-sprite snapshot from irregular frames plus a color legend.

    Input structure:
    - Four visible frames are concatenated left-to-right, separated by full-height one-column dividers, and one extra divider trails the last visible frame.
    - Each visible frame contains the same actor-colored triangle and every future actor-colored marker that still remains from that snapshot onward.
    - Divider colors are not decorative: the first three encode the visible frame-to-frame marker-consumption gaps, and the trailing divider encodes the hidden gap from the last visible frame to the output.

    Rule:
    - Recover the sprite and count the remaining markers in each visible frame; the count drops across adjacent visible frames reveal how many markers were consumed in each irregular interval.
    - Match those visible drops to the visible divider colors to infer this sample's color->gap mapping.
    - Read the trailing divider color, convert it into a hidden step count via that mapping, and then start from the rightmost visible frame.
    - Advance the sprite by exactly that many marker-consumption moves, always following the original motion rule: move to the nearest reachable marker on the unique aligned half-line and re-face that direction.

    Key insight:
    - Earlier frames act as a legend for the divider colors. The rightmost visible frame gives the local geometry, but only the visible marker-count drops tell you how many moves the final divider is asking you to simulate.
    """
    g = np.asarray(grid, dtype=int)
    frames, divider_colors, bg = _split_frames(g)
    actor = _detect_actor(g, bg, divider_colors)
    hidden_gap = _infer_hidden_gap(frames, divider_colors, actor)
    return _simulate_steps(frames[-1], actor, hidden_gap)


# --------------- generator ---------------

N_VISIBLE_FRAMES = 4
FRAME_MAX = 16
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 6, 15

STEP_DIST_RANGE = (2, 8)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 2
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

VISIBLE_GAP_COUNTS = (1, 2, 3)
ALL_VISIBLE_GAP_ORDERS = tuple(tuple(order) for order in permutations(VISIBLE_GAP_COUNTS))
HIDDEN_TAIL_RANGE = (1, 4)  # 1..3 markers remain after the hidden output snapshot
MAX_ATTEMPTS = 500


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
    target_visible_gaps: tuple[int, int, int] | None = None,
    target_hidden_gap: int | None = None,
    target_template: str | None = None,
) -> tuple[int, int, int, int, int, int, str, list[int], list[int], dict[int, int], int]:
    if target_visible_gaps is None:
        visible_gaps = list(map(int, rng.permutation(np.array(VISIBLE_GAP_COUNTS))))
    else:
        visible_gaps = [int(gap) for gap in target_visible_gaps]
        assert tuple(visible_gaps) in ALL_VISIBLE_GAP_ORDERS

    hidden_gap_choices = [gap for gap in VISIBLE_GAP_COUNTS if gap != visible_gaps[-1]]
    if target_hidden_gap is None:
        hidden_gap = int(rng.choice(hidden_gap_choices))
    else:
        assert int(target_hidden_gap) in hidden_gap_choices
        hidden_gap = int(target_hidden_gap)
    hidden_tail = int(rng.integers(*HIDDEN_TAIL_RANGE))
    n_steps = sum(visible_gaps) + hidden_gap + hidden_tail

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    free_palette = [c for c in range(10) if c not in DIVIDER_COLORS]
    palette = list(map(int, rng.permutation(free_palette)))
    bg, actor = palette[0], palette[1]

    if target_template is None:
        weights = np.array(
            TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
            dtype=float,
        )
        weights /= weights.sum()
        tname = str(rng.choice(list(TEMPLATES), p=weights))
    else:
        assert target_template in TEMPLATES
        tname = str(target_template)

    permuted_dividers = list(map(int, rng.permutation(np.array(DIVIDER_COLORS))))
    color_by_gap = {gap: permuted_dividers[i] for i, gap in enumerate(VISIBLE_GAP_COUNTS)}
    divider_colors = [color_by_gap[gap] for gap in visible_gaps] + [color_by_gap[hidden_gap]]

    snapshot_steps = [0]
    cur = 0
    for gap in visible_gaps:
        cur += gap
        snapshot_steps.append(cur)

    output_step = snapshot_steps[-1] + hidden_gap
    return n_steps, h, w, bg, actor, hidden_tail, tname, snapshot_steps, divider_colors, color_by_gap, output_step


def _make_step_dirs(rng: np.random.Generator, n_steps: int) -> list[int]:
    turn_min = max(TURN_MIN_ABS, int(np.ceil(TURN_RATIO_LO * n_steps)))
    turn_max = min(n_steps - 1, int(np.floor(TURN_RATIO_HI * n_steps)))
    if turn_min > turn_max:
        return []
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
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]] | None:
    n_steps = len(step_dirs)
    if n_steps == 0:
        return None

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


def _assemble_input(frames: list[np.ndarray], bg: int, divider_colors: list[int]) -> np.ndarray:
    h, w = frames[0].shape
    total_w = len(frames) * w + len(divider_colors)
    inp = np.full((h, total_w), bg, dtype=int)

    c0 = 0
    for i, fr in enumerate(frames):
        paste(inp, fr, 0, c0)
        c0 += w
        if i < len(divider_colors):
            inp[:, c0] = divider_colors[i]
            c0 += 1
    return inp


def _build_case(
    rng: np.random.Generator,
    target_visible_gaps: tuple[int, int, int] | None = None,
    target_hidden_gap: int | None = None,
    target_template: str | None = None,
) -> dict | None:
    (
        n_steps,
        h,
        w,
        bg,
        actor,
        hidden_tail,
        tname,
        snapshot_steps,
        divider_colors,
        color_by_gap,
        output_step,
    ) = _sample_params(
        rng,
        target_visible_gaps=target_visible_gaps,
        target_hidden_gap=target_hidden_gap,
        target_template=target_template,
    )

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
    for si in snapshot_steps:
        head, d = placements[si]
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, heads[si + 1 :]))

    out_head, out_dir = placements[output_step]
    out_frame = _render_frame(h, w, bg, actor, out_head, out_dir, tname, [])
    inp = _assemble_input(frames, bg, divider_colors)

    parsed_frames, parsed_dividers, parsed_bg = _split_frames(inp)
    if parsed_bg != bg or parsed_dividers != divider_colors or len(parsed_frames) != N_VISIBLE_FRAMES:
        return None
    if any(frame.shape != (h, w) for frame in parsed_frames):
        return None
    if any(not np.array_equal(parsed_frames[i], frames[i]) for i in range(N_VISIBLE_FRAMES)):
        return None

    visible_counts = _frame_marker_counts(frames, actor)
    visible_gaps = [visible_counts[i] - visible_counts[i + 1] for i in range(len(visible_counts) - 1)]
    if sorted(visible_gaps) != [1, 2, 3]:
        return None

    hidden_gap = _infer_hidden_gap(parsed_frames, parsed_dividers, actor)
    if output_step + hidden_tail != n_steps:
        return None
    if hidden_gap == visible_gaps[-1]:
        return None

    for fi, si in enumerate(snapshot_steps):
        found = _find_agent(frames[fi], actor)
        if found is None:
            return None
        f_tname, f_dir, f_head, _f_agent, f_markers = found
        if (f_tname, f_dir, f_head) != (tname, placements[si][1], placements[si][0]):
            return None
        if f_markers != set(heads[si + 1 :]):
            return None

    solved = main(inp)
    if not np.array_equal(solved, out_frame):
        return None

    if np.array_equal(_simulate_to_end(frames[-1], actor), out_frame):
        return None
    if visible_gaps[-1] <= visible_counts[-1]:
        if np.array_equal(_simulate_steps(frames[-1], actor, visible_gaps[-1]), out_frame):
            return None

    return {
        "input": inp,
        "output": out_frame,
        "meta": {
            "frame_height": h,
            "frame_width": w,
            "n_visible_frames": N_VISIBLE_FRAMES,
            "template": tname,
            "bg": bg,
            "actor": actor,
            "divider_colors": list(divider_colors),
            "color_by_gap": dict(color_by_gap),
            "visible_gaps": list(visible_gaps),
            "hidden_gap": hidden_gap,
            "hidden_tail": hidden_tail,
            "snapshot_steps": list(snapshot_steps),
            "output_step": output_step,
            "total_steps": n_steps,
            "turns": turns,
            "turn_ratio": turns / n_steps,
            "direction_diversity": len(set(step_dirs)),
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "step_distances": dists,
            "visible_marker_counts": visible_counts,
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    target_hidden_gap = int(rng.choice(np.array(VISIBLE_GAP_COUNTS)))
    valid_visible_orders = [
        order for order in ALL_VISIBLE_GAP_ORDERS if int(order[-1]) != target_hidden_gap
    ]
    target_visible_gaps = tuple(valid_visible_orders[int(rng.integers(0, len(valid_visible_orders)))])
    target_template = str(rng.choice(list(TEMPLATES)))
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(
            rng,
            target_visible_gaps=target_visible_gaps,
            target_hidden_gap=target_hidden_gap,
            target_template=target_template,
        )
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
