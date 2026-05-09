# Variant of 5545f144
# Dimension: D10 — 颜色使用策略 (meta)
# Change: 原题只有背景色 / actor 色 / 分隔线色，精灵与标记共用同一颜色；
# 本变体改为双色循环：精灵每吃掉一个匹配色标记后就在两种活动颜色之间翻转，
# 并且只有与精灵当前颜色相同的标记才可被消耗。

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left

# ---------------------------------------------------------------------------
# Design analysis for the D10 variant
# ---------------------------------------------------------------------------
#
# concrete_rule:
#   - Keep the original geometry: a triangular arrow sprite moves from marker to
#     marker along cardinal half-lines, re-facing the direction of travel.
#   - Replace the old "one shared actor color" rule with an alternating
#     two-color state machine.
#   - If the sprite is currently color A, only A-colored markers are edible.
#     When the head lands on such a marker, that marker disappears and the
#     sprite flips to color B. The next step can only eat a B marker, then the
#     sprite flips back to A, and so on.
#   - Markers of the wrong color remain visible and may lie on tempting
#     half-lines; they are ignored until the sprite flips to that color.
#
# behavior_trace:
#   - Original step logic from a head H:
#       "Look at every remaining marker; exactly one direction is populated; move
#       to the nearest marker on that direction."
#   - Variant step logic from a head H with current color A:
#       "Filter remaining markers to color A first; among those, exactly one
#       direction is populated; move to the nearest A marker on that direction.
#       Wrong-color markers can sit on other directions, or even closer on the
#       same direction, without being edible yet."
#   - Example:
#       sprite at H is red, a blue marker is 2 cells north, a red marker is
#       4 cells east. The original solver would see ambiguity; the variant must
#       ignore the blue dot, move east to the red dot, then recolor blue.
#
# complexity_calibration:
#   - Use exactly two activity colors. This is enough to create genuine
#     state-dependence, but still easy for a human to infer from 2-4 examples.
#   - Force the frame gap to be 3 steps and keep at least 2 input frames.
#     Because the gap is odd, adjacent snapshots always show the sprite in the
#     opposite color, so the alternation is visible instead of hidden by parity.
#   - Require at least a couple of "color-gated" states during generation:
#     states where a color-blind version of the rule would choose a different
#     target or become ambiguous. This prevents the colored dots from becoming
#     decorative noise.
#
# implicit_assumption_audit:
#   - The original `main()` assumes a single actor color can be extracted from
#     the whole input. That is false here: active content now uses two colors.
#   - The original `_find_agent(frame, actor)` assumes all actor-colored pixels
#     belong either to the sprite or to markers of the same color. Here the
#     sprite is only one monochrome component among two marker colors, so the
#     detector must instead find the unique multi-pixel sprite component.
#   - The original `_choose_target` inspects all markers. That would wrongly let
#     wrong-color markers create ambiguity or become consumed too early.
#   - The original output painter stamps the fixed actor color. In the variant
#     the final sprite color depends on the number of remaining consumptions, so
#     the solver must carry color state through the simulation.
#
# placement_legality:
#   - Keep the original no-overlap and in-bounds guarantees for every stamped
#     sprite position.
#   - Keep all markers as isolated single pixels (no touching each other and no
#     touching any stamped sprite cell in any earlier state). This ensures the
#     sprite remains the only 8-connected monochrome multi-pixel component, so
#     `_find_agent` is stable.
#   - Only same-color future markers constrain the current move direction; wrong
#     colors may align freely and act as distractors.
#
# visual_observability:
#   - Every generated input has multiple temporal snapshots, so the sprite's
#     color alternation is directly visible across frames.
#   - The rightmost frame always retains exactly 3 markers, which guarantees
#     both colors remain visible in the starting state used by the solver.
#   - No color-id arithmetic is used to define the cycle order; the two active
#     colors are sampled, and the alternation is revealed by repeated spatial
#     behavior in the snapshots themselves.
#
# new_solver_assumptions_and_generator_guards:
#   - Solver assumption: each frame contains exactly one sprite component whose
#     cell set matches one of the triangle templates.
#     Generator guard: markers are isolated singletons, so no extra multi-pixel
#     monochrome component can form.
#   - Solver assumption: the frame uses exactly two active colors for sprite and
#     markers.
#     Generator guard: it always samples exactly two such colors.
#   - Solver assumption: for the current sprite color, reachable matching
#     markers exist on exactly one half-line and do not force a 180-degree move.
#     Generator guard: path rollout enforces this color-aware uniqueness.


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
                candidates.append((len(placed), tname, d, head, color, placed))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    _, tname, d, head, color, placed = candidates[0]
    markers: dict[tuple[int, int], int] = {}
    ys, xs = np.where(frame != bg)
    for y, x in zip(ys.tolist(), xs.tolist()):
        pt = (int(y), int(x))
        if pt not in placed:
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
    """Cardinal direction from a to b, or None if not axis-aligned."""
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
        f"head={head} dir={d}: matching-color markers reachable in multiple directions "
        f"{dirs_with_candidates}, expected exactly one"
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
    active_colors = sorted(set(markers.values()) | {int(current_color)})
    if not markers:
        out = np.full((h, w), bg, dtype=int)
        for y, x in agent:
            out[y, x] = int(current_color)
        return out

    assert len(active_colors) == 2, f"expected exactly two active colors, got {active_colors}"
    other_color = active_colors[0] if active_colors[1] == current_color else active_colors[1]

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        offs = TEMPLATES[tname][d]
        matching = {pt for pt, color in markers.items() if color == current_color}
        choice = _choose_target(head, d, matching, offs, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d} color={current_color}"
        )
        nd, (ty, tx) = choice

        head = (ty, tx)
        d = nd
        agent = {(ty + oy, tx + ox) for oy, ox in TEMPLATES[tname][d]}

        landed_color = markers.pop(head, None)
        assert landed_color == current_color, (
            f"landed on marker color {landed_color}, expected current color {current_color}"
        )
        current_color, other_color = other_color, current_color

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = int(current_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a triangular sprite across the grid while alternating between two edible colors.

    Input structure:
    - Frames are concatenated horizontally and split by solid divider columns; the most common color is the background.
    - Each frame contains one monochrome triangular sprite plus single-pixel markers in exactly two active colors.

    Rule:
    - Start from the rightmost frame.
    - The sprite may only consume markers whose color matches its current body color.
    - Among matching-color markers, exactly one cardinal half-line from the head contains reachable candidates; move to the nearest one on that half-line, consume it, re-face that travel direction, and flip the sprite to the other active color.
    - Ignore wrong-color markers until the sprite's color flips to match them, and continue until every marker is gone; output only the final sprite.

    Key insight: The rightmost snapshot already tells you both the sprite's facing direction and its current color state, so the remaining path is a color-filtered version of the original half-line chase.
    """
    g = np.asarray(grid, dtype=int)
    frames, _bg = _split_frames(g)

    # Use the latest frame (rightmost) as the starting state.
    start = frames[-1]
    return _simulate_to_end(start)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (2, 3)          # rng.integers low/high+1 -> always 2 frames
FRAME_MAX = 16                   # rng.integers high for H, W -> 8..15 / 6..15
H_FLOOR, H_CAP = 8, 15           # clamps for computed H lower bound
W_FLOOR, W_CAP = 6, 15           # clamps for computed W lower bound

# --- CFG: path complexity ---
GAP = 3                          # odd gap keeps consecutive snapshot colors flipped
STEP_DIST_RANGE = (2, 8)         # per-step manhattan distance choices (2..7 inclusive)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3                 # absolute floor on #turns used during segment sampling
MIN_TURNS_ACCEPT = 2             # post-hoc minimum for accepted paths
MIN_DIR_DIVERSITY = 3            # >= 3 of {U,R,D,L} must appear
MIN_DIST_VARIETY = 2             # >= 2 distinct step distances
MIN_COLOR_GATE_STATES = 1        # states where color-blind movement would diverge

# --- CFG: template selection ---
# Weights applied when n_steps >= TEMPLATE_HEAVY_THRESHOLD, in TEMPLATES order:
# (hollow3, filled4, stem4, midfilled6). Otherwise uniform.
TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 400


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


def _sample_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int, tuple[int, int], int, str]:
    """Sample (n_frames, n_steps, gap, h, w, bg, cycle_colors, divider, tname)."""
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_steps = int(n_frames * GAP)
    gap = int(GAP)

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = rng.permutation(10)
    bg = int(palette[0])
    cycle_colors = (int(palette[1]), int(palette[2]))
    divider = int(palette[3])

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames, n_steps, gap, h, w, bg, cycle_colors, divider, tname


def _make_step_dirs(rng: np.random.Generator, n_steps: int) -> list[int]:
    """Produce a length-n sequence of step directions with turn ratio in [LO, HI]."""
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


def _marker_color_index(step_idx: int) -> int:
    return int(step_idx % 2)


def _marker_compatible_with_existing(
    nxt: tuple[int, int],
    marker_color_idx: int,
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    step_dirs: list[int],
    tname: str,
) -> bool:
    """Validate candidate marker against previously placed heads with color-aware rules."""
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, state_dirs[j])
        if any(_touching(nxt, cell) for cell in sprite):
            return False

        rel_dir = _direction(old_head, nxt)
        if rel_dir is None:
            continue

        state_color_idx = j % 2
        if marker_color_idx != state_color_idx:
            continue

        if rel_dir != step_dirs[j]:
            return False
        if j < len(heads) - 1 and _dist(old_head, nxt) <= _dist(old_head, heads[j + 1]):
            return False
    return True


def _candidate_gate_score(
    nxt: tuple[int, int],
    marker_color_idx: int,
    heads: list[tuple[int, int]],
    step_dirs: list[int],
) -> int:
    """
    Prefer future markers that become wrong-color distractors for earlier states.

    Higher scores correspond to stronger color dependence:
    - different direction from an earlier state's true next move
    - or a nearer wrong-color marker on the same direction
    """
    score = 0
    for j in range(len(heads) - 1):
        if marker_color_idx == j % 2:
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
    return score


def _rollout_path(
    rng: np.random.Generator, tname: str, step_dirs: list[int], h: int, w: int
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]] | None:
    """Greedy color-aware path rollout. Returns (heads, state_dirs, fit_placements)."""
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
            return None  # 180 reversal — caller should resample step_dirs

        dr, dc = DIRS4[next_dir]
        marker_color_idx = _marker_color_index(i)
        dist_order = list(rng.permutation(np.arange(*STEP_DIST_RANGE)))
        feasible: list[tuple[int, tuple[int, int]]] = []
        for dist in dist_order:
            nxt = (cur[0] + int(dist) * dr, cur[1] + int(dist) * dc)
            if nxt in heads:
                continue
            if any(_touching(nxt, m) for m in heads[1:]):
                continue
            if not _marker_compatible_with_existing(
                nxt, marker_color_idx, heads, state_dirs, step_dirs, tname
            ):
                continue
            trial_fit = fit_placements + [(nxt, cur_dir), (nxt, next_dir)]
            if _shift_interval(trial_fit, h, w, tname) is None:
                continue
            gate_score = _candidate_gate_score(nxt, marker_color_idx, heads, step_dirs)
            feasible.append((int(gate_score), nxt))

        if not feasible:
            return None

        best_gate = max(score for score, _ in feasible)
        best = [nxt for score, nxt in feasible if score == best_gate]
        chosen = best[int(rng.integers(0, len(best)))]

        heads.append(chosen)
        state_dirs.append(next_dir)
        fit_placements.extend([(chosen, cur_dir), (chosen, next_dir)])

    return heads, state_dirs, fit_placements


def _validate_path(step_dirs: list[int], heads: list[tuple[int, int]]) -> tuple[bool, int, list[int]]:
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
    tname: str,
    h: int,
    w: int,
) -> tuple[int, int]:
    n_steps = len(heads) - 1
    gate_states = 0
    distractor_states = 0

    for j in range(n_steps):
        head = heads[j]
        d = state_dirs[j]
        offs = TEMPLATES[tname][d]

        aware_points = {heads[k] for k in range(j + 1, n_steps + 1) if (k - 1) % 2 == j % 2}
        blind_points = {heads[k] for k in range(j + 1, n_steps + 1)}

        aware_choice = _choose_target(head, d, aware_points, offs, h, w)
        assert aware_choice == (step_dirs := (_direction(heads[j], heads[j + 1])), heads[j + 1])

        blind_reachable = _reachable_targets(head, blind_points, offs, h, w)
        wrong_reachable = [item for item in blind_reachable if (heads.index(item[2]) - 1) % 2 != j % 2]
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


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, gap, h, w, bg, cycle_colors, divider, tname = _sample_params(rng)
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
    placements = [(heads[i], state_dirs[i], cycle_colors[i % 2]) for i in range(n_steps + 1)]
    marker_colors = {heads[i]: cycle_colors[(i - 1) % 2] for i in range(1, n_steps + 1)}

    gate_states, distractor_states = _color_gate_metrics(heads, state_dirs, tname, h, w)
    if gate_states < MIN_COLOR_GATE_STATES:
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

    last_idx = (n_frames - 1) * gap
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

    return {
        "input": inp,
        "output": out_frame,
        "meta": {
            "frame_height": h,
            "frame_width": w,
            "n_frames": n_frames,
            "template": tname,
            "bg": bg,
            "cycle_colors": list(cycle_colors),
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
            "color_gate_states": gate_states,
            "wrong_color_distractor_states": distractor_states,
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
