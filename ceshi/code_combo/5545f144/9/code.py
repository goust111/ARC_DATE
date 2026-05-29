# Combo variant of 5545f144 (index 9)
# combo_spec: D8/3+D16/0
# Name: 紧凑裁剪输出+消耗顺序编码

import numpy as np

from arc_utils import bbox_coords
from arc_utils import get_bg as _bg_color
from arc_utils import paste

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
# spec_combo:
#   core_change:
#     old:
#       - Run the triangle-vs-markers simulation from the latest snapshot.
#       - Return the final sprite on a full-size clean frame.
#     new:
#       - Recover the full marker-consumption path from the earliest snapshot,
#         because only the first frame still contains every original marker.
#       - Paint each original marker location by its consumption rank:
#         first -> color 1, second -> color 2, ... , last -> color N.
#       - Remove the sprite entirely and tight-crop to the bbox of those
#         ranked marker locations.
#   concretization:
#     input_visuals:
#       - Input stays a left-to-right strip of simulation snapshots with optional
#         full-height divider columns.
#       - The leftmost frame is step 0 and shows the sprite plus all markers.
#       - Later frames show suffixes of the same path, so they confirm that the
#         dots disappear in a specific order even though the solver only needs
#         the first frame to reconstruct the whole sequence.
#     output_visuals:
#       - Output is no longer a sprite frame.
#       - It is a compact patch whose foreground pixels sit exactly at the
#         original marker coordinates, recolored 1..N by visit order, on bg 0.
#       - Empty rows/cols outside the marker bbox are removed; interior gaps stay.
#   behavior_trace:
#     example:
#       - Suppose frame 0 contains markers at A,B,C,D and the sprite moves
#         A -> B -> C -> D under the original half-line rule.
#       - Source task output: only the final triangle after landing on D.
#       - D8-only output: the same final triangle, but tightly cropped.
#       - This combo output: a cropped rectangle spanning {A,B,C,D}, with A=1,
#         B=2, C=3, D=4, and no triangle at all.
#       - Difference is therefore twofold and coupled: the simulation target
#         changes from "where the sprite ends" to "which dot was consumed when",
#         and the geometry changes from full-frame to marker-bbox crop.
#   complexity_calibration:
#     - Order colors are literal palette ids 1..N, so we cap total steps at 7.
#       That preserves visible variety without colliding with the chosen input
#       actor/divider colors 8/9 or the fixed background 0.
#     - Path logic, templates, turn constraints, and multi-frame cadence remain
#       from the source task, so the puzzle is still about inferring directed
#       motion rather than merely reading off colors.
#     - Tight crop is applied to the ranked marker map, not to a sprite, so the
#       D8 and D16 changes are fused rather than appended independently.
#   implicit_assumption_audit:
#     main_related_lines:
#       - _split_frames still assumes dividers are full-height solid non-bg
#         columns. Generator keeps that unchanged.
#       - actor-color detection still assumes exactly one non-bg non-divider
#         color is shared by sprite and markers. Generator uses one actor color.
#       - _find_agent assumes one template match explains the sprite while all
#         leftover actor pixels are isolated markers. Generator validates this
#         on the first frame, which now matters for solving.
#       - _simulate_consumption_order assumes the first frame is the earliest
#         snapshot with the full marker set. Generator preserves the original
#         cadence ordering left-to-right.
#       - _render_order_crop assumes bg is not one of the rank colors 1..N and
#         N <= 9. Generator enforces bg=0 and N in 4..7.
#     merged_change_risks:
#       - Starting from the rightmost frame would lose already-consumed markers,
#         so the source-task shortcut is no longer valid.
#       - Cropping by the final sprite bbox would ignore the D16 order map, so
#         the crop must be computed from marker coordinates instead.
#   placement_legality:
#     - We keep the source path-first rollout constraints: no marker overlaps
#       any sprite body, no out-of-bounds template placement, no ambiguous
#       half-lines, and no 180-degree reversals.
#     - Because output uses only original marker coordinates, every marker must
#       remain unique; the rollout already forbids revisiting a head position.
#     - Generator rejects cases whose marker bbox would equal the whole frame in
#       either dimension, ensuring the compact crop is genuinely visible.
#   visual_observability:
#     - From 2-4 examples, humans can see that the output keeps the marker
#       geometry from the earliest frame, discards the triangle, assigns a
#       left-to-right visible palette progression 1..N, and trims outer padding.
#     - The rule does not depend on hidden numeric ordering of arbitrary colors;
#       the ordered palette is explicit and repeated across all samples.
#   cleanliness:
#     - Sprite and order labels never overlap in the output because the sprite
#       is removed entirely before rendering the cropped rank map.
#     - Output colors are limited and structurally tied to path order, which is
#       much cleaner than introducing unrelated decorative recolorings.
#   color_reasoning:
#     - Input uses only bg 0 plus actor/divider in {8,9}; output uses 1..N.
#       This separation makes the order encoding easy to notice and prevents the
#       ranked markers from being confused with sprite or divider pixels.

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

    dirs_with_candidates = set(dd for _, dd, _ in reachable)
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


def _simulate_consumption_order(frame: np.ndarray, actor: int) -> list[tuple[int, int]]:
    h, w = frame.shape
    found = _find_agent(frame, actor)
    if found is None:
        return []

    tname, d, head, _agent, markers = found
    markers = {(int(y), int(x)) for y, x in markers}

    order: list[tuple[int, int]] = []
    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break
        offs = TEMPLATES[tname][d]
        choice = _choose_target(head, d, markers, offs, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        nd, target = choice
        head = (int(target[0]), int(target[1]))
        d = int(nd)
        order.append(head)
        markers.discard(head)

    assert not markers, f"Unconsumed markers remain: {markers}"
    return order


def _render_order_crop(order: list[tuple[int, int]], bg: int) -> np.ndarray:
    if not order:
        return np.full((0, 0), bg, dtype=int)
    assert len(order) <= 9, "rank colors are defined only for orders 1..9"
    assert bg not in set(range(1, len(order) + 1)), "background collides with rank colors"

    pts = np.array(order, dtype=int)
    r0, c0, r1, c1 = bbox_coords(pts)
    out = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), bg, dtype=int)
    for idx, (r, c) in enumerate(order, start=1):
        out[int(r - r0), int(c - c0)] = int(idx)
    return out


def _order_bbox_shape(order: list[tuple[int, int]]) -> tuple[int, int]:
    pts = np.array(order, dtype=int)
    r0, c0, r1, c1 = bbox_coords(pts)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full consumption path and emit a ranked, tightly cropped marker map.

    Input structure:
    - The input is still a strip of equal-size frames laid out left-to-right, with solid divider columns between frames when more than one snapshot is present.
    - Each frame contains one triangular sprite and same-colored single-pixel markers on a uniform background.
    - The leftmost frame is the earliest snapshot and therefore the only frame that still shows every original marker location.

    Rule:
    - Detect the sprite in the first frame and simulate the original half-line consumption process from that earliest state.
    - Record the marker coordinates in the exact order they are consumed; color the 1st consumed marker with color 1, the 2nd with color 2, and so on.
    - Remove the sprite entirely and crop the output to the minimal bounding rectangle containing those ranked marker positions.

    Key insight: The later snapshots are evidence for the same motion rule, but the answer is no longer the final sprite state; it is the earliest frame's full marker set rewritten as an ordered color trace and then tightly cropped.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    order = _simulate_consumption_order(frames[0], actor)
    return _render_order_crop(order, bg)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

N_FRAMES_RANGE = (2, 5)
FRAME_MAX = 16
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 6, 15

# Rank colors are literal 1..N, so keep N <= 7 and reserve 8/9 for input-only colors.
STEPS_RANGE = (4, 8)  # 4..7 inclusive
STEP_DIST_RANGE = (2, 8)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

FRAME_COUNT_VALUES = np.array(range(*N_FRAMES_RANGE), dtype=int)
FRAME_COUNT_WEIGHTS = np.array([0.48, 0.34, 0.18], dtype=float)
STEP_COUNT_VALUES = np.array(range(*STEPS_RANGE), dtype=int)
STEP_COUNT_WEIGHTS = np.array([1.0, 1.35, 1.0, 1.35], dtype=float)

TEMPLATE_HEAVY_THRESHOLD = 7
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

MAX_ATTEMPTS = 500

BG_COLOR = 0
INPUT_ONLY_COLORS = (8, 9)


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
) -> tuple[int, int, int, int, int, int, int | None, str]:
    n_frames = int(rng.choice(FRAME_COUNT_VALUES, p=FRAME_COUNT_WEIGHTS / FRAME_COUNT_WEIGHTS.sum()))
    n_steps = int(rng.choice(STEP_COUNT_VALUES, p=STEP_COUNT_WEIGHTS / STEP_COUNT_WEIGHTS.sum()))
    max_gap = (n_steps - 1) // (n_frames - 1)
    gap_candidates = np.arange(1, max_gap + 1, dtype=int)
    gap_weights = gap_candidates.astype(float)
    gap = int(rng.choice(gap_candidates, p=gap_weights / gap_weights.sum()))

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    actor = int(rng.choice(INPUT_ONLY_COLORS))
    divider = int(next(c for c in INPUT_ONLY_COLORS if c != actor)) if n_frames > 1 else None

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames, n_steps, gap, h, w, actor, divider, tname


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
    n_frames, n_steps, gap, h, w, actor, divider, tname = _sample_params(rng)
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

    frames = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        frames.append(_render_frame(h, w, BG_COLOR, actor, head, d, tname, heads[si + 1 :]))

    order = list(heads[1:])
    out_crop = _render_order_crop(order, BG_COLOR)
    out_h, out_w = map(int, out_crop.shape)
    if out_h >= h or out_w >= w:
        return None

    inp = _assemble_input(frames, BG_COLOR, divider)

    first_found = _find_agent(frames[0], actor)
    if first_found is None:
        return None
    s_tname, s_dir, s_head, _s_agent, s_markers = first_found
    if (s_tname, s_dir, s_head) != (tname, placements[0][1], placements[0][0]):
        return None
    if s_markers != set(order):
        return None

    last_idx = (n_frames - 1) * gap
    last_remaining = n_steps - last_idx
    last_found = _find_agent(frames[-1], actor)
    if last_found is None:
        return None
    f_tname, f_dir, f_head, _f_agent, f_markers = last_found
    if (f_tname, f_dir, f_head) != (tname, placements[last_idx][1], placements[last_idx][0]):
        return None
    if f_markers != set(heads[last_idx + 1 :]):
        return None

    if not np.array_equal(main(inp), out_crop):
        return None

    non_bg = sorted(set(map(int, out_crop.ravel())) - {BG_COLOR})
    if non_bg != list(range(1, n_steps + 1)):
        return None

    marker_box_h, marker_box_w = _order_bbox_shape(order)
    return {
        "input": inp,
        "output": out_crop,
        "meta": {
            "frame_height": h,
            "frame_width": w,
            "n_frames": n_frames,
            "template": tname,
            "bg": BG_COLOR,
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
            "step_distances": dists,
            "count_in_first_frame": n_steps,
            "count_in_last_input_frame": last_remaining,
            "output_shape": (out_h, out_w),
            "output_colors": non_bg,
            "marker_bbox_shape": (marker_box_h, marker_box_w),
            "crop_margin_rows": int(h - out_h),
            "crop_margin_cols": int(w - out_w),
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
