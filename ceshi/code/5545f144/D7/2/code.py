# Variant of 5545f144
# Dimension: D7 — 变换：精灵朝向更新 (transform)
# Change: 消耗标记后精灵重新朝向该步的移动方向，模板按新方向旋转重新戳印。... → 累积旋转

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

# Design analysis for the D7 variant
# ----------------------------------
# spec:
#   task: the same triangular sprite consumes aligned one-pixel markers across
#     horizontally concatenated snapshots, but its displayed facing now keeps an
#     accumulated extra clockwise quarter-turn whenever the movement path turns.
#   visible_effect:
#     - Movement is still constrained by marker alignment on half-lines.
#     - The sprite shape is still one monochrome triangle template.
#     - The difference is purely in the stamped orientation after each landing:
#       old facing = move_dir; new facing = move_dir + turn_count_so_far (mod 4).
#   concrete_trace:
#     - Suppose move directions are [Up, Up, Right, Down].
#     - Old task facings by head state are [Up, Up, Up, Right, Down].
#     - New task facings are        [Up, Up, Up, Down, Left]:
#         step 0: no earlier turn, face Up
#         step 1: still straight, face Up
#         step 2: path turns once, so Right + 1 cw = Down
#         step 3: path turns twice, so Down + 2 cw = Left
#     - This is the exact D7 change used by both solver and generator.
#   complexity_calibration:
#     - The core path logic, colors, frame layout, and marker semantics stay the
#       same as the source task; only orientation updating changes.
#     - To keep the twist visible, the generator rejects cases where the final
#       accumulated turn offset is 0 mod 4; otherwise the output could collapse
#       back to the original orientation rule.
#   hidden_assumptions_review:
#     - Original main() starts from the rightmost frame because displayed facing
#       also equals the previous move direction. That no longer holds after D7,
#       so the new solver must simulate forward from the first frame where the
#       initial facing is still untwisted and directly readable.
#     - Original _choose_target() checks legality using the current template
#       orientation. Under D7, target legality depends on the post-landing
#       facing after applying the cumulative rotation, so that fit check must be
#       recomputed per candidate direction.
#     - Original 180-degree reversal logic compares against displayed facing.
#       Under D7 that would change the task itself. To preserve the original
#       movement rule, reversal is checked against the previous move direction.
#   placement_legality:
#     - Every snapshot uses the same template family, but the stamped direction
#       at head i is the derived facing for that state.
#     - Candidate markers are rejected if they touch any previously stamped
#       sprite cell, create a second half-line option from an earlier head, or
#       would place the newly twisted template out of bounds.
#     - Generator oracle-checks every visible frame with _find_agent(), because
#       the new solver relies on the first frame rather than only the last one.
#   visual_observability:
#     - No color remapping is introduced; the only new signal is orientation.
#     - The existing templates are rotationally asymmetric, so the cumulative
#       twist is visible from 2-3 examples.
#     - Rejecting final offset 0 ensures new_main(input) usually differs from
#       the original rule on the same input instead of silently degenerating.
#   cleanliness:
#     - Markers remain isolated single pixels, share one actor color, and never
#       overlap the sprite body at the snapshot where they appear.
#     - Divider columns and frame cadence stay unchanged, so the grid remains as
#       readable as the original task.

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L


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


def _sprite_cells(head: tuple[int, int], tname: str, d: int) -> set[tuple[int, int]]:
    hy, hx = head
    return {(hy + oy, hx + ox) for oy, ox in TEMPLATES[tname][d]}


def _derive_facings(step_dirs: list[int]) -> tuple[list[int], list[int]]:
    """Return facing per head-state and cumulative turn offset per head-state."""
    facings = [int(step_dirs[0])]
    spins = [0]
    spin = 0
    prev_move = int(step_dirs[0])
    for i, move_dir in enumerate(step_dirs):
        move_dir = int(move_dir)
        if i > 0 and move_dir != prev_move:
            spin = (spin + 1) % 4
        facings.append(int((move_dir + spin) % 4))
        spins.append(int(spin))
        prev_move = move_dir
    return facings, spins


def _choose_target(
    head: tuple[int, int],
    prev_move_dir: int,
    spin: int,
    step_idx: int,
    markers: set[tuple[int, int]],
    tname: str,
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
    for move_dir in range(4):
        if step_idx > 0 and move_dir == (prev_move_dir + 2) % 4:
            continue
        next_spin = (spin + int(step_idx > 0 and move_dir != prev_move_dir)) % 4
        facing = int((move_dir + next_spin) % 4)
        offs = TEMPLATES[tname][facing]
        for dist, pt in candidates(move_dir):
            ty, tx = pt
            if all(0 <= ty + oy < h and 0 <= tx + ox < w for oy, ox in offs):
                reachable.append((dist, move_dir, pt))

    if not reachable:
        return None

    dirs_with_candidates = {move_dir for _, move_dir, _ in reachable}
    assert len(dirs_with_candidates) == 1, (
        f"head={head} prev_move={prev_move_dir}: markers reachable in multiple "
        f"directions {dirs_with_candidates}, expected exactly one"
    )

    reachable.sort()
    _, move_dir, pt = reachable[0]
    if step_idx > 0:
        assert move_dir != (prev_move_dir + 2) % 4, (
            f"head={head} prev_move={prev_move_dir}: would reverse to {move_dir}"
        )
    return move_dir, pt


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, facing, head, agent, markers = found
    agent = set((int(y), int(x)) for y, x in agent)
    markers = set((int(y), int(x)) for y, x in markers)

    prev_move_dir = int(facing)
    spin = 0
    step_idx = 0
    max_steps = h * w * 4

    for _ in range(max_steps):
        if not markers:
            break

        choice = _choose_target(head, prev_move_dir, spin, step_idx, markers, tname, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} "
            f"prev_move={prev_move_dir} spin={spin}"
        )
        move_dir, new_head = choice

        if step_idx > 0 and move_dir != prev_move_dir:
            spin = (spin + 1) % 4
        facing = int((move_dir + spin) % 4)
        head = (int(new_head[0]), int(new_head[1]))
        prev_move_dir = int(move_dir)
        agent = _sprite_cells(head, tname, facing)
        markers.discard(head)
        step_idx += 1

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a triangular sprite across the grid while its facing accumulates twists.

    Input structure:
    - Frames are concatenated left-to-right with full-height divider columns; the
      background is the most common color.
    - Each frame contains one triangular sprite plus single-pixel markers, all
      in one actor color, using the same template family across snapshots.

    Rule:
    - Start from the leftmost frame, where the sprite still points along its
      first movement direction.
    - Repeatedly move the head to the nearest remaining marker on the unique
      reachable half-line (same row or column, one side only).
    - When the path turns relative to the previous move, add one extra clockwise
      quarter-turn to an accumulated rotation offset; stamp the sprite at the
      landing cell with facing = move direction + accumulated offset.
    - Continue until all markers are consumed; output the final sprite alone on
      a clean background.

    Key insight: The markers still determine where the head travels, but only
    the earliest snapshot reveals the untwisted starting direction; afterwards
    the visible triangle orientation drifts away from the movement direction by
    the cumulative number of turns.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    start = frames[0]
    return _simulate_to_end(start, actor)


# --------------- generator (D7 cumulative-turn variant) ---------------

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

TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

MAX_ATTEMPTS = 500

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
) -> tuple[int, int, int, int, int, int, int, int | None, str]:
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
    bg, actor = int(palette[0]), int(palette[1])
    divider = int(palette[2]) if n_frames > 1 else None

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames, n_steps, gap, h, w, bg, actor, divider, tname


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
        out.extend([int(d)] * int(ln))
    return out


def _marker_compatible_with_existing(
    nxt: tuple[int, int],
    heads: list[tuple[int, int]],
    face_dirs: list[int],
    step_dirs: list[int],
    tname: str,
) -> bool:
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, face_dirs[j])
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
) -> tuple[list[tuple[int, int]], list[int], list[int], list[tuple[tuple[int, int], int]]] | None:
    n_steps = len(step_dirs)
    face_dirs, spins = _derive_facings(step_dirs)
    heads: list[tuple[int, int]] = [(0, 0)]
    fit_placements: list[tuple[tuple[int, int], int]] = [((0, 0), face_dirs[0])]

    for i in range(n_steps):
        cur = heads[-1]
        next_dir = int(step_dirs[i])
        dr, dc = DIRS4[next_dir]
        next_face = int(face_dirs[i + 1])

        dist_order = list(rng.permutation(np.arange(*STEP_DIST_RANGE)))
        chosen = None
        for dist in dist_order:
            nxt = (cur[0] + int(dist) * dr, cur[1] + int(dist) * dc)
            if nxt in heads:
                continue
            if any(_touching(nxt, m) for m in heads[1:]):
                continue
            if not _marker_compatible_with_existing(nxt, heads, face_dirs, step_dirs, tname):
                continue
            trial_fit = fit_placements + [(nxt, next_face)]
            if _shift_interval(trial_fit, h, w, tname) is None:
                continue
            chosen = nxt
            break

        if chosen is None:
            return None

        heads.append(chosen)
        fit_placements.append((chosen, next_face))

    return heads, face_dirs, spins, fit_placements


def _validate_path(
    step_dirs: list[int], heads: list[tuple[int, int]], spins: list[int]
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
    if spins[-1] == 0:
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
    n_frames, n_steps, gap, h, w, bg, actor, divider, tname = _sample_params(rng)
    step_dirs = _make_step_dirs(rng, n_steps)

    rollout = _rollout_path(rng, tname, step_dirs, h, w)
    if rollout is None:
        return None
    heads, face_dirs, spins, fit_placements = rollout

    ok, turns, dists = _validate_path(step_dirs, heads, spins)
    if not ok:
        return None

    shift = _shift_interval(fit_placements, h, w, tname)
    if shift is None:
        return None
    tr_lo, tr_hi, tc_lo, tc_hi = shift
    tr = int(rng.integers(tr_lo, tr_hi + 1))
    tc = int(rng.integers(tc_lo, tc_hi + 1))
    heads = [(r + tr, c + tc) for r, c in heads]
    placements = [(heads[i], face_dirs[i]) for i in range(n_steps + 1)]

    frames = []
    for fi in range(n_frames):
        si = fi * gap
        head, facing = placements[si]
        frames.append(_render_frame(h, w, bg, actor, head, facing, tname, heads[si + 1 :]))

    fhead, ffacing = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, fhead, ffacing, tname, [])
    inp = _assemble_input(frames, bg, divider)

    for fi, frame in enumerate(frames):
        found = _find_agent(frame, actor)
        if found is None:
            return None
        f_tname, f_dir, f_head, _f_agent, f_markers = found
        si = fi * gap
        if (f_tname, f_dir, f_head) != (tname, placements[si][1], placements[si][0]):
            return None
        if f_markers != set(heads[si + 1 :]):
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
            "face_dirs": list(face_dirs),
            "turn_offsets": list(spins),
            "heads": list(heads),
            "step_distances": dists,
            "count_in_first_frame": n_steps,
            "count_in_last_input_frame": gap,
            "final_offset": int(spins[-1]),
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
