# Variant of 5545f144
# Dimension: D7 — 变换：精灵朝向更新 (transform)
# Change: 消耗标记后精灵重新朝向该步的移动方向，模板按新方向旋转重新戳印。... → 消耗后精灵模板变形

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

# -----------------------------------------------------------------------------
# Design analysis for the D7 variant
#
# a) Concretization and behavior trace
# - The visible objects stay the same kinds of arrow-like triangles plus single-pixel
#   markers, all in one actor color, with optional solid divider columns between
#   horizontally concatenated frames.
# - The changed behavior happens only when a marker is consumed. The sprite still
#   moves to the nearest reachable marker on the unique visible half-line, and its
#   facing still updates to that movement direction. The new part is that the body
#   shape also advances one phase in a fixed cycle:
#   hollow3 -> filled4 -> stem4 -> midfilled6 -> hollow3 -> ...
# - Concrete trace: if the last input frame shows a right-facing `filled4` sprite
#   with three markers left, then the next three landings yield:
#   step 1 -> right/up/down/left according to the chosen marker, body becomes
#   `stem4`; step 2 -> body becomes `midfilled6`; step 3 -> body becomes
#   `hollow3`. The original task would have stayed `filled4` throughout. Here the
#   solver must track both direction and template phase.
#
# b) Complexity calibration
# - The cycle has exactly four reusable states. That is enough to make the variant
#   visually obvious without exploding the search space.
# - The four templates already differ in area and silhouette, so the morph is
#   visible even when the direction is unchanged across consecutive steps.
# - To avoid invisible degeneration, the generator rejects cases where the frame
#   cadence gap is a multiple of 4; otherwise all snapshot frames would show the
#   same template phase.
#
# c) Old implicit assumptions that break under the variant
# - `_simulate_to_end` originally treated `tname` as constant. That is now false:
#   every successful landing must advance it.
# - `_choose_target` originally only reasoned about the current template. Under the
#   new rule, a step is only legal if the current translated body fits during the
#   move and the post-consumption next template also fits at the landing cell.
# - The generator's `_shift_interval`, `_rollout_path`, and marker-compatibility
#   logic originally assumed a fixed template for all states. They now need the
#   per-step template sequence, otherwise a path may validate for one body shape
#   but fail for another later phase.
#
# d) Placement legality
# - Each future marker is still a future head position. The generator ensures every
#   such pixel is isolated from every earlier stamped sprite body, does not collide
#   with another marker, and only lies on the intended half-line for each prior
#   state.
# - Translation fitting is checked against both the moving body and the morphed
#   landing body, so the solver never needs to clip or silently drop pixels.
#
# e) Visual observability
# - The rule is inferable from 2-4 examples because the body changes among four
#   visibly different silhouettes while the actor color stays constant. No hidden
#   quantity such as color id is used for sorting or branching.
# - Rejecting `gap % 4 == 0` ensures that at least one visible frame transition, or
#   the final output versus the last input frame, exhibits a phase change.
#
# f) Visual cleanliness
# - Markers remain isolated single pixels, away from the sprite body, so the sprite
#   silhouette is readable in every frame.
# - Divider columns still use a distinct solid color and never overlap actor cells.
#
# g) Color inferability
# - Colors do not change in this variant. The only changing attribute is the sprite
#   shape phase plus its direction, which is directly visible from geometry.
# -----------------------------------------------------------------------------

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
TEMPLATE_ORDER = ("hollow3", "filled4", "stem4", "midfilled6")
TEMPLATE_INDEX = {name: i for i, name in enumerate(TEMPLATE_ORDER)}


def _next_template(tname: str, steps: int = 1) -> str:
    idx = (TEMPLATE_INDEX[tname] + steps) % len(TEMPLATE_ORDER)
    return TEMPLATE_ORDER[idx]


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


def _sprite_cells(head: tuple[int, int], tname: str, d: int) -> set[tuple[int, int]]:
    hy, hx = head
    return {(hy + oy, hx + ox) for oy, ox in TEMPLATES[tname][d]}


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
    markers: set[tuple[int, int]],
    offs: list[tuple[int, int]],
    next_tname: str,
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

    dirs_with_candidates = set(dd for _, dd, _ in reachable)
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

        offs = TEMPLATES[tname][d]
        next_tname = _next_template(tname)
        choice = _choose_target(head, d, markers, offs, next_tname, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d} "
            f"template={tname}"
        )
        nd, (ty, tx) = choice

        head = (ty, tx)
        d = nd
        tname = next_tname
        agent = _sprite_cells(head, tname, d)
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance a triangular sprite through the remaining aligned markers until none are left.

    Input structure:
    - Several same-height frames may be concatenated horizontally, with optional
      full-height divider columns in a third color; the background is the most
      common color overall.
    - Inside each frame, the actor color is shared by one oriented triangle-like
      sprite and by isolated one-pixel markers.

    Rule:
    - Read only the rightmost frame as the current state.
    - From the sprite's head, exactly one cardinal half-line has reachable markers;
      the sprite moves to the nearest such marker without ever reversing 180°.
    - Landing on a marker consumes it, updates the facing to that travel direction,
      and advances the sprite body to the next template in the fixed cycle
      `hollow3 -> filled4 -> stem4 -> midfilled6 -> ...`.
    - Repeat until all markers are consumed, then output only the final sprite on
      a clean background.

    Key insight: the visible sprite state is two-dimensional: the tip gives the
    current direction, and the current silhouette gives the current phase in the
    four-shape morph cycle that advances after every consumption.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    start = frames[-1]
    return _simulate_to_end(start, actor)


# --------------- generator (D7 template-cycle variant) ---------------

N_FRAMES_RANGE = (1, 5)
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

MAX_ATTEMPTS = 400

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
) -> tuple[int, int, int, int, int, int, int, int | None, str]:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0 and (n // n_frames) % 4 != 0]
    weights = np.array([(n**3) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = rng.permutation(10)
    bg, actor = int(palette[0]), int(palette[1])
    divider = int(palette[2]) if n_frames > 1 else None
    start_tname = str(rng.choice(TEMPLATE_ORDER))
    return n_frames, n_steps, gap, h, w, bg, actor, divider, start_tname


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
    rng: np.random.Generator,
    start_tname: str,
    step_dirs: list[int],
    h: int,
    w: int,
) -> tuple[list[tuple[int, int]], list[int], list[str], list[tuple[tuple[int, int], int, str]]] | None:
    n_steps = len(step_dirs)
    d0 = step_dirs[0]
    heads: list[tuple[int, int]] = [(0, 0)]
    state_dirs: list[int] = [d0]
    state_templates: list[str] = [start_tname]
    fit_placements: list[tuple[tuple[int, int], int, str]] = [((0, 0), d0, start_tname)]

    for i in range(n_steps):
        cur = heads[-1]
        cur_dir = state_dirs[-1]
        cur_tname = state_templates[-1]
        next_dir = step_dirs[i]
        next_tname = _next_template(cur_tname)
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
            if not _marker_compatible_with_existing(nxt, heads, state_dirs, state_templates, step_dirs):
                continue
            trial_fit = fit_placements + [(nxt, cur_dir, cur_tname), (nxt, next_dir, next_tname)]
            if _shift_interval(trial_fit, h, w) is None:
                continue
            chosen = nxt
            break

        if chosen is None:
            return None

        heads.append(chosen)
        state_dirs.append(next_dir)
        state_templates.append(next_tname)
        fit_placements.extend([(chosen, cur_dir, cur_tname), (chosen, next_dir, next_tname)])

    return heads, state_dirs, state_templates, fit_placements


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


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, gap, h, w, bg, actor, divider, start_tname = _sample_params(rng)
    step_dirs = _make_step_dirs(rng, n_steps)

    rollout = _rollout_path(rng, start_tname, step_dirs, h, w)
    if rollout is None:
        return None
    heads, state_dirs, state_templates, fit_placements = rollout

    ok, turns, dists = _validate_path(step_dirs, heads)
    if not ok:
        return None

    shift = _shift_interval(fit_placements, h, w)
    if shift is None:
        return None
    tr_lo, tr_hi, tc_lo, tc_hi = shift
    tr = int(rng.integers(tr_lo, tr_hi + 1))
    tc = int(rng.integers(tc_lo, tc_hi + 1))
    heads = [(r + tr, c + tc) for r, c in heads]
    placements = [(heads[i], state_dirs[i], state_templates[i]) for i in range(n_steps + 1)]

    frames = []
    for fi in range(n_frames):
        si = fi * gap
        head, d, tname = placements[si]
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, heads[si + 1:]))

    fhead, fdir, ftname = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, fhead, fdir, ftname, [])
    inp = _assemble_input(frames, bg, divider)

    last_idx = (n_frames - 1) * gap
    found = _find_agent(frames[-1], actor)
    if found is None:
        return None
    f_tname, f_dir, f_head, _f_agent, f_markers = found
    if (f_tname, f_dir, f_head) != (
        placements[last_idx][2],
        placements[last_idx][1],
        placements[last_idx][0],
    ):
        return None
    if f_markers != set(heads[last_idx + 1:]):
        return None
    if np.array_equal(out_frame, frames[-1]):
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
            "bg": bg,
            "actor": actor,
            "divider": divider,
            "total_steps": n_steps,
            "gap": gap,
            "gap_mod_4": gap % 4,
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
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
