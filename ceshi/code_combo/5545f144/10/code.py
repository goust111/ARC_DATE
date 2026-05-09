# Combo variant of 5545f144 (index 10)
# combo_spec: D3/0+D10/2
# Name: 多色标记消耗顺序+精灵颜色随消耗变化

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left

# Fixed visible cycle shared by markers and sprite state:
# red -> blue -> green -> red -> ...
MARKER_CYCLE = (2, 1, 3)
ACTIVE_COLORS = set(MARKER_CYCLE)
NON_ACTIVE_PALETTE = [0, 4, 5, 6, 7, 8, 9]


# ---------------------------------------------------------------------------
# Design analysis for the combo variant D3/0 + D10/2
#
# concrete_pixel_rule:
#   - Keep the original 5545f144 geometry: each frame contains one triangular
#     sprite and isolated one-pixel markers; the input is still horizontal
#     snapshots with optional solid divider columns.
#   - Replace the old single actor color with a shared three-color state cycle:
#       red(2) -> blue(1) -> green(3) -> red -> ...
#   - At any moment the sprite itself is painted in its current state color.
#     Only markers of that SAME color are edible. After the sprite consumes one,
#     the sprite changes to the next color in the cycle.
#
# behavior_trace_with_small_example:
#   - Original task from a head H:
#       if markers exist north and east, generation had to forbid that because
#       any aligned marker was edible and the move would be ambiguous.
#   - Fused task from a red sprite at H:
#       a red marker may sit east, a blue marker north, a green marker west.
#       Only the red one is eligible now, so the sprite must move east. After
#       landing, the sprite becomes blue; now the blue marker north may become
#       the next forced move.
#   - The difference is twofold and coupled:
#       1. marker colors encode a temporal order;
#       2. the sprite color is the live state that tells which color in that
#          order is active right now.
#
# complexity_calibration:
#   - Use exactly three active colors, not an unconstrained palette. This is
#     enough to create repeated evidence for a cycle without making the task a
#     "guess arbitrary color mapping" problem.
#   - Keep the cycle fixed and visible across all generated cases so multiple
#     samples reinforce the same order instead of scattering into many possible
#     permutations.
#   - Require at least two input frames and choose frame gaps not divisible by 3,
#     so adjacent snapshots show different sprite colors and humans can observe
#     that the sprite state advances over time.
#
# implicit_assumption_audit_against_original_main:
#   - Original actor-color extraction assumed one non-bg, non-divider color.
#     That is false now because active content uses three colors.
#   - Original `_find_agent(frame, actor)` assumed all pixels of the actor color
#     belonged to either the sprite or same-color markers. That is false now
#     because the sprite is only one multi-pixel monochrome component among
#     three marker colors, so detection must search connected components.
#   - Original target selection considered all markers. That is false now:
#     reachable candidates must first be filtered to the sprite's current color.
#   - Original output painter used a fixed actor color. That is false now
#     because the final sprite color depends on how many consumptions remain.
#
# placement_legality:
#   - The generator still rolls out a geometric path first, then renders frames.
#   - Every future head is also a marker cell; each one must stay in-bounds and
#     every stamped template at every state must fit in-grid.
#   - Markers remain isolated single pixels and are rejected if they 8-touch any
#     previously visible sprite cell. This preserves a unique monochrome sprite
#     component for `_find_agent`.
#   - For a past state j, only future markers whose color equals the sprite color
#     at state j are allowed to constrain the unique next half-line. Wrong-color
#     markers may align elsewhere and act as deliberate distractors.
#
# visual_observability_and_neatness:
#   - All three active colors are semantic, visible categories; no rule depends
#     on numeric ordering or hidden metadata.
#   - The generator rejects cases where ignoring marker colors would yield the
#     same final answer. This prevents silent collapse back to the original task.
#   - It also enforces color-gated states, including a wrong-color reachable
#     distractor from the last input frame, so the fusion behavior is visible in
#     the actual frame the solver starts from.
# ---------------------------------------------------------------------------


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


def _cycle_index(color: int) -> int:
    idx = MARKER_CYCLE.index(int(color))
    return int(idx)


def _next_cycle_color(color: int) -> int:
    idx = _cycle_index(color)
    return int(MARKER_CYCLE[(idx + 1) % len(MARKER_CYCLE)])


def _components8(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            comp: set[tuple[int, int]] = set()
            while stack:
                cy, cx = stack.pop()
                comp.add((int(cy), int(cx)))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny = cy + dy
                        nx = cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            comps.append(comp)
    return comps


def _match_template(
    pts: set[tuple[int, int]]
) -> tuple[str, int, tuple[int, int], set[tuple[int, int]]] | None:
    for head in pts:
        hy, hx = int(head[0]), int(head[1])
        for tname, dirs in TEMPLATES.items():
            for d, offs in dirs.items():
                placed = {(hy + dy, hx + dx) for dy, dx in offs}
                if placed == pts:
                    return tname, d, (hy, hx), placed
    return None


def _find_agent(
    frame: np.ndarray, bg: int
) -> tuple[str, int, tuple[int, int], int, set[tuple[int, int]], dict[tuple[int, int], int]] | None:
    colors = sorted({int(v) for v in frame.ravel() if int(v) != bg})
    candidates = []
    for color in colors:
        comps = _components8(frame == color)
        for comp in comps:
            if len(comp) < 3:
                continue
            matched = _match_template(comp)
            if matched is not None:
                tname, d, head, placed = matched
                candidates.append((len(placed), tname, d, head, int(color), placed))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    _, tname, d, head, color, placed = candidates[0]
    assert color in ACTIVE_COLORS, f"sprite color {color} not in active cycle {MARKER_CYCLE}"

    markers: dict[tuple[int, int], int] = {}
    ys, xs = np.where(frame != bg)
    for y, x in zip(ys.tolist(), xs.tolist()):
        pt = (int(y), int(x))
        if pt in placed:
            continue
        markers[pt] = int(frame[y, x])
    return tname, d, head, int(color), set(map(tuple, placed)), markers


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


def _fits_target(
    head: tuple[int, int],
    offs: list[tuple[int, int]],
    target: tuple[int, int],
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


def _reachable_targets(
    head: tuple[int, int],
    points: set[tuple[int, int]],
    offs: list[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int, tuple[int, int]]]:
    hy, hx = head
    reachable: list[tuple[int, int, tuple[int, int]]] = []
    for y, x in points:
        if x == hx and y < hy:
            dd = 0
            dist = hy - y
        elif y == hy and x > hx:
            dd = 1
            dist = x - hx
        elif x == hx and y > hy:
            dd = 2
            dist = y - hy
        elif y == hy and x < hx:
            dd = 3
            dist = hx - x
        else:
            continue
        if _fits_target(head, offs, (y, x), h, w):
            reachable.append((int(dist), int(dd), (int(y), int(x))))
    reachable.sort()
    return reachable


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    offs: list[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    reachable = _reachable_targets(head, markers, offs, h, w)
    if not reachable:
        return None

    dirs_with_candidates = {dd for _, dd, _ in reachable}
    assert len(dirs_with_candidates) == 1, (
        f"head={head} dir={d}: reachable matching-color markers in multiple directions "
        f"{dirs_with_candidates}"
    )

    _, dd, pt = reachable[0]
    assert dd != (d + 2) % 4, (
        f"head={head} dir={d}: would reverse to dir={dd}, not allowed"
    )
    return dd, pt


def _simulate_to_end(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, bg)
    if found is None:
        return frame.copy()

    tname, d, head, current_color, agent, markers = found
    if not markers:
        out = np.full((h, w), bg, dtype=int)
        for y, x in agent:
            out[y, x] = int(current_color)
        return out

    assert set(markers.values()).issubset(ACTIVE_COLORS), (
        f"unexpected marker colors {set(markers.values()) - ACTIVE_COLORS}"
    )

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        offs = TEMPLATES[tname][d]
        matching = {pt for pt, color in markers.items() if color == current_color}
        choice = _choose_target(head, d, matching, offs, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} markers at head={head} dir={d} color={current_color}"
        )
        nd, (ty, tx) = choice

        head = (ty, tx)
        d = nd
        agent = {(ty + oy, tx + ox) for oy, ox in TEMPLATES[tname][d]}

        landed_color = markers.pop(head, None)
        assert landed_color == current_color, (
            f"landed on marker color {landed_color}, expected current color {current_color}"
        )
        current_color = _next_cycle_color(current_color)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = int(current_color)
    return out


def _simulate_ignore_colors(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, bg)
    if found is None:
        return frame.copy()

    tname, d, head, current_color, agent, markers = found
    marker_positions = set(markers)

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not marker_positions:
            break
        offs = TEMPLATES[tname][d]
        choice = _choose_target(head, d, marker_positions, offs, h, w)
        assert choice is not None
        nd, (ty, tx) = choice
        head = (ty, tx)
        d = nd
        agent = {(ty + oy, tx + ox) for oy, ox in TEMPLATES[tname][d]}
        marker_positions.discard(head)
        current_color = _next_cycle_color(current_color)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = int(current_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance a triangular sprite through a color-locked marker sequence until the board is cleared.

    Input structure:
    - Frames are concatenated left-to-right and optional divider columns are solid, full-height, and non-background.
    - Each frame contains one monochrome triangular sprite plus isolated single-pixel markers in the visible cycle
      colors red, blue, and green.

    Rule:
    - Start from the rightmost frame.
    - The sprite may only eat markers whose color matches its current body color.
    - Among those matching-color markers, exactly one cardinal half-line from the head contains reachable candidates;
      move to the nearest one on that half-line, consume it, re-face that travel direction, and recolor the sprite to
      the next color in the repeating cycle red -> blue -> green -> red -> ...
    - Ignore wrong-color markers until the sprite cycles to their color, and continue until no markers remain; output
      only the final sprite on a clean background.

    Key insight: the triangle still determines motion geometry, but the sprite's present color is now a live state
    variable telling you which visible marker color is currently edible and which color will become active next.
    """
    g = np.asarray(grid, dtype=int)
    frames, _bg = _split_frames(g)

    start = frames[-1]
    return _simulate_to_end(start)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

FRAME_CONFIGS = (
    (2, 4),  # compact two-frame sample; rightmost frame starts on blue
    (3, 2),  # three-frame sample; visibly shows red -> green -> blue
)
FRAME_MAX = 16
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 6, 15

TOTAL_STEPS = 8
STEP_DIST_RANGE = (2, 8)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2
MIN_COLOR_GATE_STATES = 1

TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

MAX_ATTEMPTS = 500


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
) -> tuple[int, int, int, int, int, int, int, str]:
    n_frames, gap = FRAME_CONFIGS[int(rng.integers(0, len(FRAME_CONFIGS)))]
    n_steps = int(TOTAL_STEPS)
    gap = int(gap)

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(max(h_lo, 10), FRAME_MAX))
    w = int(rng.integers(max(w_lo, 8), FRAME_MAX))

    palette = list(rng.permutation(NON_ACTIVE_PALETTE))
    bg = int(palette[0])
    divider = int(palette[1])

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames, n_steps, gap, h, w, bg, divider, tname


def _choose_block_dirs(rng: np.random.Generator, prev_dir: int | None) -> list[int] | None:
    for _ in range(40):
        d1 = int(rng.integers(0, 4))
        if prev_dir is not None and d1 == (prev_dir + 2) % 4:
            continue
        turn = int(rng.choice([1, -1]))
        d2 = int((d1 + turn) % 4)
        d3 = int((d1 + 2) % 4)
        return [d1, d2, d3, d3]
    return None


def _construct_two_block_path(
    rng: np.random.Generator, tname: str, h: int, w: int
) -> tuple[list[int], list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]] | None:
    block1 = _choose_block_dirs(rng, None)
    if block1 is None:
        return None
    block2 = _choose_block_dirs(rng, block1[-1])
    if block2 is None:
        return None

    step_dirs = block1 + block2
    state_dirs: list[int] = [step_dirs[0]]
    heads: list[tuple[int, int]] = [(0, 0)]
    fit_placements: list[tuple[tuple[int, int], int]] = [((0, 0), step_dirs[0])]

    dist_values = list(np.arange(*STEP_DIST_RANGE))
    for _ in range(500):
        a, b, c, d, e, f = [int(rng.choice(dist_values)) for _ in range(6)]
        step_dists = [a, b, a, c, d, e, d, f]

        tmp_heads = [(0, 0)]
        tmp_state = [step_dirs[0]]
        tmp_fit = [((0, 0), step_dirs[0])]
        ok = True
        for i, (next_dir, dist) in enumerate(zip(step_dirs, step_dists)):
            cur = tmp_heads[-1]
            cur_dir = tmp_state[-1]
            if next_dir == (cur_dir + 2) % 4:
                ok = False
                break
            dr, dc = DIRS4[next_dir]
            nxt = (cur[0] + dist * dr, cur[1] + dist * dc)
            if nxt in tmp_heads:
                ok = False
                break
            if any(_touching(nxt, m) for m in tmp_heads[1:]):
                ok = False
                break
            marker_color = _step_color(i)
            if not _marker_compatible_with_existing(
                nxt, marker_color, tmp_heads, tmp_state, step_dirs, tname
            ):
                ok = False
                break
            trial_fit = tmp_fit + [(nxt, cur_dir), (nxt, next_dir)]
            if _shift_interval(trial_fit, h, w, tname) is None:
                ok = False
                break
            tmp_heads.append(nxt)
            tmp_state.append(next_dir)
            tmp_fit.extend([(nxt, cur_dir), (nxt, next_dir)])

        if not ok:
            continue

        heads = tmp_heads
        state_dirs = tmp_state
        fit_placements = tmp_fit
        return step_dirs, heads, state_dirs, fit_placements

    return None


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


def _step_color(step_idx: int) -> int:
    return int(MARKER_CYCLE[step_idx % len(MARKER_CYCLE)])


def _marker_compatible_with_existing(
    nxt: tuple[int, int],
    marker_color: int,
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

        if marker_color != _step_color(j):
            continue

        if rel_dir != step_dirs[j]:
            return False
        if j < len(heads) - 1 and _dist(old_head, nxt) <= _dist(old_head, heads[j + 1]):
            return False
    return True


def _candidate_gate_score(
    nxt: tuple[int, int],
    marker_color: int,
    heads: list[tuple[int, int]],
    step_dirs: list[int],
    focus_state: int | None = None,
) -> int:
    score = 0
    for j in range(len(heads) - 1):
        need_color = _step_color(j)
        if marker_color == need_color:
            continue
        rel_dir = _direction(heads[j], nxt)
        if rel_dir is None:
            continue
        actual_dir = step_dirs[j]
        actual_dist = _dist(heads[j], heads[j + 1])
        cand_dist = _dist(heads[j], nxt)
        if rel_dir != actual_dir:
            score += 3
        elif cand_dist < actual_dist:
            score += 2
        else:
            score += 1
    if focus_state is not None and focus_state < len(heads) - 1:
        need_color = _step_color(focus_state)
        if marker_color != need_color:
            rel_dir = _direction(heads[focus_state], nxt)
            if rel_dir is not None:
                actual_dir = step_dirs[focus_state]
                actual_dist = _dist(heads[focus_state], heads[focus_state + 1])
                cand_dist = _dist(heads[focus_state], nxt)
                if rel_dir != actual_dir:
                    score += 12
                elif cand_dist < actual_dist:
                    score += 10
                else:
                    score += 1
    return score


def _rollout_path(
    rng: np.random.Generator,
    tname: str,
    step_dirs: list[int],
    h: int,
    w: int,
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]] | None:
    n_steps = len(step_dirs)
    d0 = step_dirs[0]
    heads: list[tuple[int, int]] = [(0, 0)]
    state_dirs: list[int] = [d0]
    fit_placements: list[tuple[tuple[int, int], int]] = [((0, 0), d0)]
    node_budget = 5000

    def search(i: int) -> bool:
        nonlocal node_budget
        node_budget -= 1
        if node_budget < 0:
            return False
        if i == n_steps:
            return True

        cur = heads[-1]
        cur_dir = state_dirs[-1]
        next_dir = step_dirs[i]
        marker_color = _step_color(i)
        if next_dir == (cur_dir + 2) % 4:
            return False

        dr, dc = DIRS4[next_dir]
        dist_values = np.arange(*STEP_DIST_RANGE)
        dist_weights = 1 / dist_values.astype(float)
        dist_weights /= dist_weights.sum()
        dist_order = list(
            rng.choice(dist_values, size=len(dist_values), replace=False, p=dist_weights)
        )

        feasible: list[tuple[int, float, tuple[int, int]]] = []
        for dist in dist_order:
            nxt = (cur[0] + int(dist) * dr, cur[1] + int(dist) * dc)
            if nxt in heads:
                continue
            if any(_touching(nxt, m) for m in heads[1:]):
                continue
            if not _marker_compatible_with_existing(
                nxt, marker_color, heads, state_dirs, step_dirs, tname
            ):
                continue
            trial_fit = fit_placements + [(nxt, cur_dir), (nxt, next_dir)]
            if _shift_interval(trial_fit, h, w, tname) is None:
                continue
            gate_score = _candidate_gate_score(nxt, marker_color, heads, step_dirs)
            feasible.append((int(gate_score), float(rng.random()), nxt))

        feasible.sort(reverse=True)
        for _, _, chosen in feasible:
            heads.append(chosen)
            state_dirs.append(next_dir)
            fit_placements.extend([(chosen, cur_dir), (chosen, next_dir)])
            if search(i + 1):
                return True
            fit_placements.pop()
            fit_placements.pop()
            state_dirs.pop()
            heads.pop()
        return False

    if not search(0):
        return None
    return heads, state_dirs, fit_placements


def _append_gated_suffix(
    rng: np.random.Generator,
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    fit_placements: list[tuple[tuple[int, int], int]],
    prefix_dirs: list[int],
    tname: str,
    h: int,
    w: int,
    start_idx: int,
) -> list[int] | None:
    h0 = heads[-1]
    dist_values = list(np.arange(*STEP_DIST_RANGE))
    current_dir = state_dirs[-1]

    d1_options = list(rng.permutation(np.arange(4)))
    for d1 in d1_options:
        d1 = int(d1)
        if d1 == (current_dir + 2) % 4:
            continue
        for turn in rng.permutation(np.array([1, -1], dtype=int)).tolist():
            d2 = int((d1 + int(turn)) % 4)
            d3 = int((d1 + 2) % 4)
            d4_options = [d2, int((d2 + 2) % 4)]
            d4_options = [int(d4_options[i]) for i in rng.permutation(len(d4_options))]
            for d4 in d4_options:
                suffix_dirs = [d1, d2, d3, d4]
                step_dirs = prefix_dirs + suffix_dirs
                orders = [
                    (int(a), int(b), int(c))
                    for a in rng.permutation(dist_values)
                    for b in rng.permutation(dist_values)
                    for c in rng.permutation(dist_values)
                ]
                for a, b, c in orders:
                    candidates = []
                    cur = h0
                    for dist, dcur in [(a, d1), (b, d2), (a, d3), (c, d4)]:
                        dr, dc = DIRS4[dcur]
                        nxt = (cur[0] + dist * dr, cur[1] + dist * dc)
                        candidates.append(nxt)
                        cur = nxt

                    if len(set(candidates)) != len(candidates):
                        continue
                    if any(pt in heads for pt in candidates):
                        continue
                    existing = heads[1:]
                    if any(any(_touching(pt, other) for other in existing) for pt in candidates):
                        continue
                    if any(
                        _touching(candidates[i], candidates[j])
                        for i in range(len(candidates))
                        for j in range(i + 1, len(candidates))
                    ):
                        continue

                    tmp_heads = list(heads)
                    tmp_state_dirs = list(state_dirs)
                    tmp_fit = list(fit_placements)
                    ok = True
                    for step_offset, nxt in enumerate(candidates):
                        i = start_idx + step_offset
                        cur_dir = tmp_state_dirs[-1]
                        next_dir = step_dirs[i]
                        marker_color = _step_color(i)
                        if not _marker_compatible_with_existing(
                            nxt, marker_color, tmp_heads, tmp_state_dirs, step_dirs, tname
                        ):
                            ok = False
                            break
                        trial_fit = tmp_fit + [(nxt, cur_dir), (nxt, next_dir)]
                        if _shift_interval(trial_fit, h, w, tname) is None:
                            ok = False
                            break
                        tmp_heads.append(nxt)
                        tmp_state_dirs.append(next_dir)
                        tmp_fit.extend([(nxt, cur_dir), (nxt, next_dir)])

                    if not ok:
                        continue

                    heads[:] = tmp_heads
                    state_dirs[:] = tmp_state_dirs
                    fit_placements[:] = tmp_fit
                    return step_dirs
    return None


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
    sprite_color: int,
    head: tuple[int, int],
    d: int,
    tname: str,
    markers: list[tuple[tuple[int, int], int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    patch, ar, ac = _template_patch(tname, d, sprite_color, bg)
    paste(frame, patch, head[0] - ar, head[1] - ac, transparent=bg)
    for (mr, mc), color in markers:
        frame[mr, mc] = int(color)
    return frame


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


def _color_gate_metrics(
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    step_dirs: list[int],
    tname: str,
    h: int,
    w: int,
) -> tuple[int, int]:
    n_steps = len(heads) - 1
    gate_states = 0
    distractor_states = 0
    head_to_step = {heads[k]: k - 1 for k in range(1, len(heads))}

    for j in range(n_steps):
        head = heads[j]
        d = state_dirs[j]
        offs = TEMPLATES[tname][d]
        need_color = _step_color(j)

        aware_points = {
            heads[k] for k in range(j + 1, n_steps + 1) if _step_color(k - 1) == need_color
        }
        aware_choice = _choose_target(head, d, aware_points, offs, h, w)
        assert aware_choice == (step_dirs[j], heads[j + 1])

        blind_points = {heads[k] for k in range(j + 1, n_steps + 1)}
        blind_reachable = _reachable_targets(head, blind_points, offs, h, w)
        wrong_reachable = [
            item for item in blind_reachable if _step_color(head_to_step[item[2]]) != need_color
        ]
        if wrong_reachable:
            distractor_states += 1

        blind_dirs = {dd for _, dd, _ in blind_reachable}
        if len(blind_dirs) != 1:
            gate_states += 1
            continue
        _, blind_dir, blind_pt = blind_reachable[0]
        if blind_dir == (d + 2) % 4:
            gate_states += 1
            continue
        if (blind_dir, blind_pt) != aware_choice:
            gate_states += 1

    return gate_states, distractor_states


def _last_frame_is_gate_state(
    last_idx: int,
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    step_dirs: list[int],
    tname: str,
    h: int,
    w: int,
) -> bool:
    head = heads[last_idx]
    d = state_dirs[last_idx]
    offs = TEMPLATES[tname][d]
    need_color = _step_color(last_idx)
    aware_points = {
        heads[k] for k in range(last_idx + 1, len(heads)) if _step_color(k - 1) == need_color
    }
    aware_choice = _choose_target(head, d, aware_points, offs, h, w)
    assert aware_choice == (step_dirs[last_idx], heads[last_idx + 1])
    blind_points = {heads[k] for k in range(last_idx + 1, len(heads))}
    blind_reachable = _reachable_targets(head, blind_points, offs, h, w)
    blind_dirs = {dd for _, dd, _ in blind_reachable}
    if len(blind_dirs) != 1:
        return True
    _, blind_dir, blind_pt = blind_reachable[0]
    if blind_dir == (d + 2) % 4:
        return True
    return (blind_dir, blind_pt) != aware_choice


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, gap, h, w, bg, divider, tname = _sample_params(rng)
    last_idx = (n_frames - 1) * gap
    if last_idx >= n_steps:
        return None
    remaining_from_last = n_steps - last_idx
    built = _construct_two_block_path(rng, tname, h, w)
    if built is None:
        return None
    step_dirs, heads, state_dirs, fit_placements = built

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

    placements = [(heads[i], state_dirs[i], _step_color(i)) for i in range(n_steps + 1)]
    marker_colors = {heads[i]: _step_color(i - 1) for i in range(1, n_steps + 1)}

    gate_states, distractor_states = _color_gate_metrics(
        heads, state_dirs, step_dirs, tname, h, w
    )
    if gate_states < MIN_COLOR_GATE_STATES:
        return None

    if not _last_frame_is_gate_state(last_idx, heads, state_dirs, step_dirs, tname, h, w):
        return None

    sprite_colors_in_frames = [_step_color(fi * gap) for fi in range(n_frames)]
    if len(set(sprite_colors_in_frames)) < 2:
        return None
    if n_frames >= 3 and len(set(sprite_colors_in_frames)) < 3:
        return None

    frames = []
    for fi in range(n_frames):
        si = fi * gap
        head, d, sprite_color = placements[si]
        remain = [(heads[k], marker_colors[heads[k]]) for k in range(si + 1, n_steps + 1)]
        frames.append(_render_frame(h, w, bg, sprite_color, head, d, tname, remain))

    fhead, fdir, fcolor = placements[-1]
    out_frame = _render_frame(h, w, bg, fcolor, fhead, fdir, tname, [])
    inp = _assemble_input(frames, bg, divider)

    found = _find_agent(frames[-1], bg)
    if found is None:
        return None
    f_tname, f_dir, f_head, f_sprite_color, _f_agent, f_markers = found
    if (f_tname, f_dir, f_head, f_sprite_color) != (
        tname,
        placements[last_idx][1],
        placements[last_idx][0],
        placements[last_idx][2],
    ):
        return None
    expected_markers = {heads[k]: marker_colors[heads[k]] for k in range(last_idx + 1, n_steps + 1)}
    if f_markers != expected_markers:
        return None
    if not np.array_equal(main(inp), out_frame):
        return None
    try:
        if np.array_equal(_simulate_ignore_colors(frames[-1]), out_frame):
            return None
    except Exception:
        pass

    return {
        "input": inp,
        "output": out_frame,
        "meta": {
            "frame_height": h,
            "frame_width": w,
            "n_frames": n_frames,
            "template": tname,
            "bg": bg,
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
            "count_in_last_input_frame": remaining_from_last,
            "color_cycle": list(MARKER_CYCLE),
            "color_gate_states": gate_states,
            "wrong_color_distractor_states": distractor_states,
            "last_frame_is_gate_state": True,
            "sprite_colors_in_frames": list(sprite_colors_in_frames),
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
