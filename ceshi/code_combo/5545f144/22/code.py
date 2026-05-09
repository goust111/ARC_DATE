# Combo variant of 5545f144 (index 22)
# combo_spec: D1/0+D5/0+D10/2
# Name: 颜色匹配消耗+标记转化+精灵变色循环

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
#
# a) Concrete behavior and trace:
#    Keep the original frame strip and the original arrow geometry, but fuse
#    three changes into one stateful rule:
#    - future markers are single pixels in two alternating activity colors A/B;
#    - the sprite may consume only markers whose color matches its current body;
#    - each consumed marker does not disappear: it changes to one fixed trail
#      color T and stays on the board; after each consumption the sprite flips
#      A <-> B.
#    Small trace:
#    - Original state from head H: all remaining dots share one color, so the
#      solver inspects every dot and moves to the nearest reachable one on the
#      only populated half-line.
#    - Combo state from head H with sprite color A:
#      1. Ignore every future B-colored dot for target selection.
#      2. Among only A-colored future dots, exactly one half-line is legal.
#      3. Move to the nearest such A dot, face that move direction, recolor the
#         old stop into T, and flip the sprite to B.
#      4. Repeat symmetrically for B.
#    Difference from the source task: geometry still matters, but only after a
#    color gate has filtered the candidate set; consumed points also remain as a
#    visible trail instead of vanishing.
#
# b) Complexity calibration:
#    Use exactly two activity colors plus one trail color. Two active colors are
#    enough to create state dependence without making the palette arbitrary, and
#    one dedicated trail color keeps the recolor rule legible. The frame gap is
#    fixed to 3, so the last input frame always shows non-empty trail evidence
#    and the sprite color flips between adjacent snapshots.
#
# c) Implicit assumption audit against the original main():
#    - "There is one global actor color" is no longer true. The fused solver
#      must infer two activity colors from the first frame and detect the sprite
#      as a template-shaped monochrome component rather than by color exclusion.
#    - "All non-sprite actor pixels are future markers" is no longer true.
#      Later frames also contain trail-colored visited stops, so the solver must
#      separate future markers from visited trail by color role.
#    - "Every visible marker constrains the next move" is false. Wrong-color
#      markers must be ignored until the sprite flips into that color.
#    - "Consumed markers disappear" is false. The current head in the rightmost
#      frame is itself a consumed stop hidden under the sprite, so simulation
#      must seed visited = visible_trail U {current_head}.
#
# d) Placement legality:
#    The generator enforces all original in-bounds and no-overlap constraints,
#    plus three fused guards:
#    - isolated singleton markers/trail points so the sprite remains the only
#      multi-pixel monochrome component;
#    - color-aware uniqueness, meaning only same-color future markers can create
#      half-line ambiguity for a given state;
#    - later sprite poses may not cover older visible trail points, so the kept
#      trail remains observable in late snapshots and in the output.
#
# e) Visual observability:
#    A human can infer the rule from repeated cues across samples:
#    - the sprite body alternates between two colors across time snapshots;
#    - future dots of both colors stay visible in the same frame, but only one
#      color is "live" at a time because the path follows only that color;
#    - visited stops accumulate in a third color and align with earlier landing
#      sites. No hidden use of numeric color ids is needed.
#
# f) Visual cleanliness:
#    All non-sprite objects stay as isolated single pixels. The trail uses one
#    dedicated color and future markers use only two colors, so the board stays
#    readable even after fusion. The only intentional overlap is that the sprite
#    covers its current head, hiding the newest trail point until it moves away.
#
# g) Change inferability:
#    The color cycle is structural rather than arbitrary: after every move the
#    body flips between the only two activity colors visible from frame 0, and
#    consumed points always settle into the sole extra trail color. The solver's
#    new assumptions are therefore:
#    - first frame uses exactly two non-bg/non-divider colors and no trail;
#    - all later extra color usage belongs to a single trail color;
#    - for the current sprite color, reachable matching markers exist on exactly
#      one half-line and never force a 180-degree reversal.
#    The generator explicitly constructs cases that satisfy these assumptions.


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


def _divider_colors(g: np.ndarray, bg: int) -> set[int]:
    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    return {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}


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
    candidates: list[tuple[int, str, int, tuple[int, int], int, set[tuple[int, int]]]] = []
    for color in colors:
        for comp in _components8(frame == color):
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
    markers: dict[tuple[int, int], int] = {}
    ys, xs = np.where(frame != bg)
    for y, x in zip(ys.tolist(), xs.tolist()):
        pt = (int(y), int(x))
        if pt not in placed:
            markers[pt] = int(frame[y, x])
    return tname, d, head, color, set(map(tuple, placed)), markers


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
        f"head={head} dir={d}: matching-color markers reachable in multiple directions "
        f"{dirs_with_candidates}, expected exactly one"
    )

    _, dd, pt = reachable[0]
    assert dd != (d + 2) % 4, (
        f"head={head} dir={d}: would reverse to dir={dd}, not allowed"
    )
    return dd, pt


def _render_state(
    h: int,
    w: int,
    bg: int,
    sprite_color: int,
    trail_color: int,
    head: tuple[int, int],
    d: int,
    tname: str,
    future_markers: dict[tuple[int, int], int],
    visited_markers: list[tuple[int, int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    for y, x in visited_markers:
        frame[y, x] = trail_color
    for (y, x), color in future_markers.items():
        frame[y, x] = int(color)
    patch, ar, ac = _template_patch(tname, d, sprite_color, bg)
    paste(frame, patch, head[0] - ar, head[1] - ac, transparent=bg)
    return frame


def _other_color(current_color: int, cycle_colors: tuple[int, int]) -> int:
    a, b = cycle_colors
    assert current_color in cycle_colors, f"sprite color {current_color} not in cycle {cycle_colors}"
    return b if current_color == a else a


def _simulate_to_end(frame: np.ndarray, cycle_colors: tuple[int, int], trail_color: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, bg)
    if found is None:
        return frame.copy()

    tname, d, head, current_color, _agent, markers = found
    assert current_color in cycle_colors, (
        f"sprite color {current_color} not among inferred activity colors {cycle_colors}"
    )

    future_markers = {
        pt: color for pt, color in markers.items() if color in cycle_colors
    }
    visited = {pt for pt, color in markers.items() if color == trail_color}
    other_colors = set(markers.values()) - set(cycle_colors) - {trail_color}
    assert not other_colors, f"unexpected extra colors in frame: {other_colors}"
    visited.add(head)

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not future_markers:
            break
        offs = TEMPLATES[tname][d]
        matching = {pt for pt, color in future_markers.items() if color == current_color}
        choice = _choose_target(head, d, matching, offs, h, w)
        assert choice is not None, (
            f"stuck with {len(future_markers)} future markers at head={head} dir={d} color={current_color}"
        )
        nd, target = choice
        landed_color = future_markers.pop(target, None)
        assert landed_color == current_color, (
            f"landed on marker color {landed_color}, expected current color {current_color}"
        )
        head = target
        d = nd
        visited.add(head)
        current_color = _other_color(current_color, cycle_colors)

    return _render_state(
        h=h,
        w=w,
        bg=bg,
        sprite_color=current_color,
        trail_color=trail_color,
        head=head,
        d=d,
        tname=tname,
        future_markers={},
        visited_markers=sorted(visited),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a triangular sprite across the grid by color-gated half-line chasing, while keeping a recolored trail.

    Input structure:
    - Frames are concatenated left-to-right and may be split by solid divider columns; the most common color is the background.
    - Each frame contains one monochrome triangular sprite, future single-pixel markers in exactly two activity colors, and in later frames a third color marking already-visited stops.

    Rule:
    - Read the rightmost frame as the current simulation state.
    - The sprite may consume only future markers whose color matches its current body color.
    - Among those matching-color markers, exactly one cardinal half-line from the head contains reachable candidates; move to the nearest one on that half-line, re-face that travel direction, recolor the consumed stop into the fixed trail color, and flip the sprite to the other activity color.
    - Repeat until no future markers remain, then output the final sprite together with the full recolored trail.

    Key insight: The puzzle is no longer solved by geometric alignment alone; you must first identify the sprite's current color state, use it to filter which dots are edible, and notice that every eaten dot persists in a separate trail color instead of disappearing.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)
    div_colors = _divider_colors(g, bg)

    first_colors = sorted({int(v) for v in frames[0].ravel() if int(v) not in {bg} | div_colors})
    assert len(first_colors) == 2, f"expected two activity colors in first frame, got {first_colors}"

    all_colors = {int(v) for v in g.ravel()}
    trail_candidates = sorted(all_colors - {bg} - div_colors - set(first_colors))
    assert len(trail_candidates) == 1, f"expected one trail color, got {trail_candidates}"
    trail_color = trail_candidates[0]

    start = frames[-1]
    return _simulate_to_end(start, (first_colors[0], first_colors[1]), trail_color)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

N_FRAMES_RANGE = (2, 5)  # 2..4 frames; trail must be visible
FRAME_MAX = 16
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 6, 15

GAP = 3  # odd gap makes adjacent snapshots flip sprite color
STEP_DIST_RANGE = (2, 8)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2
MIN_COLOR_GATE_STATES = 1
MIN_VISIBLE_GATE_STATES = 1

TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

MAX_ATTEMPTS = 1200


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
) -> tuple[int, int, int, int, int, int, tuple[int, int], int, int, str]:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_steps = int(n_frames * GAP)
    gap = int(GAP)

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = list(map(int, rng.permutation(10)))
    bg = palette[0]
    cycle_colors = (palette[1], palette[2])
    trail_color = palette[3]
    divider = palette[4]

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames, n_steps, gap, h, w, bg, cycle_colors, trail_color, divider, tname


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


def _trail_clear_of_pose(
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
    marker_color_idx: int,
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
        marker_color_idx = i % 2
        dist_order = list(rng.permutation(np.arange(*STEP_DIST_RANGE)))
        feasible: list[tuple[int, tuple[int, int]]] = []
        for dist in dist_order:
            nxt = (cur[0] + int(dist) * dr, cur[1] + int(dist) * dc)
            if nxt in heads:
                continue
            if any(_touching(nxt, pt) for pt in heads[1:]):
                continue
            if not _marker_compatible_with_existing(
                nxt, marker_color_idx, heads, state_dirs, step_dirs, tname
            ):
                continue
            if not _trail_clear_of_pose(heads[1:], nxt, next_dir, tname):
                continue
            trial_fit = fit_placements + [(nxt, cur_dir), (nxt, next_dir)]
            if _shift_interval(trial_fit, h, w, tname) is None:
                continue
            gate_score = _candidate_gate_score(nxt, marker_color_idx, heads, step_dirs)
            feasible.append((gate_score, nxt))

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


def _gate_metrics_on_indices(
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
    h: int,
    w: int,
    state_indices: list[int] | range,
) -> tuple[int, int]:
    n_steps = len(heads) - 1
    gate_states = 0
    distractor_states = 0
    for j in state_indices:
        head = heads[j]
        d = state_dirs[j]
        offs = TEMPLATES[tname][d]

        aware_points = {heads[k] for k in range(j + 1, n_steps + 1) if (k - 1) % 2 == j % 2}
        aware_choice = _choose_target(head, d, aware_points, offs, h, w)
        actual_choice = (_direction(heads[j], heads[j + 1]), heads[j + 1])
        assert aware_choice == actual_choice

        blind_points = {heads[k] for k in range(j + 1, n_steps + 1)}
        blind_reachable = _reachable_targets(head, blind_points, offs, h, w)
        wrong_reachable = [
            item for item in blind_reachable if (heads.index(item[2]) - 1) % 2 != j % 2
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


def _color_gate_metrics(
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
    h: int,
    w: int,
) -> tuple[int, int]:
    return _gate_metrics_on_indices(heads, state_dirs, tname, h, w, range(len(heads) - 1))


def _visible_color_gate_metrics(
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
    h: int,
    w: int,
    gap: int,
) -> tuple[int, int]:
    return _gate_metrics_on_indices(heads, state_dirs, tname, h, w, range(0, len(heads) - 1, gap))


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, gap, h, w, bg, cycle_colors, trail_color, divider, tname = _sample_params(rng)
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
    future_marker_colors = {
        heads[i]: cycle_colors[(i - 1) % 2] for i in range(1, n_steps + 1)
    }

    gate_states, distractor_states = _color_gate_metrics(heads, state_dirs, tname, h, w)
    if gate_states < MIN_COLOR_GATE_STATES:
        return None
    visible_gate_states, visible_distractor_states = _visible_color_gate_metrics(
        heads, state_dirs, tname, h, w, gap
    )
    if visible_gate_states < MIN_VISIBLE_GATE_STATES:
        return None

    frames: list[np.ndarray] = []
    for fi in range(n_frames):
        si = fi * gap
        head, d, sprite_color = placements[si]
        future = {heads[k]: future_marker_colors[heads[k]] for k in range(si + 1, n_steps + 1)}
        visited = heads[1:si]
        frames.append(
            _render_state(
                h=h,
                w=w,
                bg=bg,
                sprite_color=sprite_color,
                trail_color=trail_color,
                head=head,
                d=d,
                tname=tname,
                future_markers=future,
                visited_markers=visited,
            )
        )

    fhead, fdir, fcolor = placements[-1]
    out_frame = _render_state(
        h=h,
        w=w,
        bg=bg,
        sprite_color=fcolor,
        trail_color=trail_color,
        head=fhead,
        d=fdir,
        tname=tname,
        future_markers={},
        visited_markers=heads[1:],
    )
    inp = _assemble_input(frames, bg, divider)

    first_colors = {int(v) for v in frames[0].ravel() if int(v) != bg}
    if first_colors != set(cycle_colors):
        return None

    last_idx = (n_frames - 1) * gap
    found = _find_agent(frames[-1], bg)
    if found is None:
        return None
    f_tname, f_dir, f_head, f_sprite_color, f_agent, f_markers = found
    if (f_tname, f_dir, f_head, f_sprite_color) != (
        tname,
        placements[last_idx][1],
        placements[last_idx][0],
        placements[last_idx][2],
    ):
        return None

    visible_trail = {pt for pt, color in f_markers.items() if color == trail_color}
    expected_visible_trail = set(heads[1:last_idx])
    if visible_trail != expected_visible_trail:
        return None

    expected_future = {heads[k]: future_marker_colors[heads[k]] for k in range(last_idx + 1, n_steps + 1)}
    parsed_future = {pt: color for pt, color in f_markers.items() if color in cycle_colors}
    if parsed_future != expected_future:
        return None

    last_frame_colors = {int(v) for v in frames[-1].ravel() if int(v) != bg}
    if not set(cycle_colors).issubset(last_frame_colors):
        return None
    if trail_color not in last_frame_colors:
        return None
    if np.count_nonzero(frames[-1] == trail_color) <= 0:
        return None
    if np.count_nonzero(out_frame == trail_color) <= 0:
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
            "trail_color": trail_color,
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
            "count_in_last_input_frame": len(expected_future),
            "visible_trail_in_last_frame": len(expected_visible_trail),
            "color_gate_states": gate_states,
            "wrong_color_distractor_states": distractor_states,
            "visible_color_gate_states": visible_gate_states,
            "visible_wrong_color_distractor_states": visible_distractor_states,
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
