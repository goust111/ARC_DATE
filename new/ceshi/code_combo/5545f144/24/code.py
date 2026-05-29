# Combo variant of 5545f144 (index 24)
# combo_spec: D9/3+D22/1
# Name: 输出为下一帧+消耗N个后停止
#
# Design analysis for the combo variant
# -------------------------------------
# concretization and behavior trace:
# - The visual language stays the same as the source task: horizontally
#   concatenated frames, solid divider columns, one triangular sprite, and
#   same-colored single-pixel markers that represent future landing heads.
# - Two source changes must interact rather than run independently:
#   1) D9/3 says the output is the NEXT evenly spaced snapshot after the
#      rightmost input frame, not the fully exhausted end state.
#   2) D22/1 says the sprite stops after a fixed quota N consumptions, where N
#      is encoded by the input rather than by "until no markers remain".
# - The fusion makes these two numbers the same visible quantity:
#   if the input has F frames, then the snapshot gap is g = F and the stop
#   quota is also N = F. So the rightmost frame is exactly one quota before the
#   output snapshot.
# - Concrete trace: with F=3 and leftover=4, the hidden path has 13 future
#   heads. Input frames are sampled at steps 0, 3, 6, so the visible marker
#   counts are 13, 10, 7. The output is the next snapshot at step 9, after
#   exactly 3 more consumptions from the last input frame, leaving 4 markers.
#   The original task would have continued from 7 markers down to 0 instead.
#
# complexity calibration:
# - F is restricted to 2..4 so the frame-count encoding is easy to read, but
#   the generator biases toward 3 or 4 frames so the constant-gap pattern is
#   usually supported by multiple differences.
# - Leftover markers after the forced stop are kept at 3..4, which makes the
#   output visibly non-terminal and prevents collapse back to the source task.
# - No extra colors, glyphs, or counters are introduced; the only new logic is
#   the arithmetic relation between frame count and marker-count cadence.
#
# implicit assumption audit around main():
# - `_split_frames()` still assumes full-height divider columns; generation
#   preserves that layout exactly.
# - Actor-color detection still assumes there is one actor color plus one
#   optional divider color; generation keeps this unchanged.
# - `_find_agent()` still assumes all extra actor pixels belong to markers
#   rather than a second sprite; generation still renders exactly one sprite.
# - The source `main()` only needed the rightmost frame. That would be too weak
#   here because it would collapse to the plain D22 rule and ignore the snapshot
#   structure. The combo solver therefore inspects every frame, infers the
#   constant marker-count drop `g`, and checks that it matches the frame-count
#   quota `N`.
# - `_simulate_path(..., stop_after=g)` assumes the last frame contains at least
#   `g` future markers and that each of those steps has a unique legal target.
#   Generation guarantees `last_frame_markers = g + leftover` and builds the
#   path with the original uniqueness / non-reversal constraints.
#
# placement legality:
# - Path rollout is still forward-constructed from future head positions.
# - Each candidate future head must fit the rotated template in-grid, remain
#   disjoint from every earlier sprite body, avoid touching earlier markers, and
#   avoid creating multiple reachable half-lines from any earlier head.
# - Because rendered markers are literal future head locations, the output can
#   safely show the stopped sprite plus leftover markers without overlap.
#
# visual observability and cleanliness:
# - The input frames now show an arithmetic progression in marker counts, and
#   the amount removed per frame equals the number of frames themselves.
# - The output continues that cadence once more while still leaving markers, so
#   both parts of the fusion are visually present in ordinary grid content.
# - No hidden dependence on numeric color ids is used; only countable objects
#   and spatial sprite orientation matter.

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

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


def _render_state(
    h: int,
    w: int,
    bg: int,
    actor: int,
    head: tuple[int, int],
    d: int,
    tname: str,
    markers: set[tuple[int, int]] | list[tuple[int, int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    agent = {(head[0] + oy, head[1] + ox) for oy, ox in TEMPLATES[tname][d]}
    for y, x in agent:
        frame[y, x] = actor
    for y, x in markers:
        frame[y, x] = actor
    return frame


def _simulate_path(frame: np.ndarray, actor: int, stop_after: int | None) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, _agent, markers = found
    markers = {(int(y), int(x)) for y, x in markers}

    if stop_after is not None:
        assert len(markers) >= stop_after, (
            f"need at least {stop_after} markers in last frame, found {len(markers)}"
        )

    consumed = 0
    max_steps = h * w * 4
    for _ in range(max_steps):
        if stop_after is not None and consumed >= stop_after:
            break
        if not markers:
            break

        offs = TEMPLATES[tname][d]
        choice = _choose_target(head, d, markers, offs, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        nd, target = choice
        head = target
        d = nd
        markers.discard(head)
        consumed += 1

    return _render_state(h, w, bg, actor, head, d, tname, markers)


def _infer_gap_and_quota(
    frames: list[np.ndarray], actor: int
) -> tuple[int, int, list[int]]:
    assert len(frames) >= 2, "combo requires at least two input frames"

    counts: list[int] = []
    for i, frame in enumerate(frames):
        found = _find_agent(frame, actor)
        assert found is not None, f"frame {i} does not contain a detectable sprite"
        counts.append(len(found[-1]))

    diffs = [counts[i] - counts[i + 1] for i in range(len(counts) - 1)]
    gap = diffs[0]
    assert gap > 0, f"snapshot cadence must be positive, got {gap}"
    assert all(diff == gap for diff in diffs), (
        f"marker-count drops are not uniform: counts={counts}, diffs={diffs}"
    )

    quota = len(frames)
    assert gap == quota, (
        f"expected snapshot gap to equal frame-count quota, got gap={gap}, quota={quota}"
    )
    assert counts[-1] > gap, (
        f"last frame must still have leftover markers after one more snapshot, got counts={counts}"
    )
    return gap, quota, counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance the triangular sprite by one more encoded snapshot and stop there.

    Input structure:
    - Several same-height frames are concatenated left-to-right, separated by
      solid divider columns, with one sprite plus same-colored one-pixel markers
      in each frame.
    - Across the sequence, the sprite template stays the same while orientation
      and the number of remaining future markers change.

    Rule:
    - Read every frame, count how many markers remain, and infer the constant
      snapshot gap from the repeated drop between consecutive frames.
    - Let N be the number of input frames. In this fused variant, that frame
      count is also the stop quota, so the next snapshot is exactly N forced
      marker consumptions after the rightmost frame.
    - Starting from the rightmost frame, repeatedly jump to the nearest marker
      on the unique reachable half-line, re-face that direction, and stop after
      those N moves even if markers still remain.

    Key insight:
    - The earlier frames are necessary: they reveal an arithmetic cadence in the
      marker counts, and that cadence matches the visible frame count. So the
      output is one more equally spaced snapshot rather than the terminal board.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    gap, quota, _counts = _infer_gap_and_quota(frames, actor)
    assert gap == quota
    return _simulate_path(frames[-1], actor, stop_after=gap)


# --------------- generator (combo: next snapshot with frame-count cadence) ---------------

# --- CFG: frame count and cadence ---
FRAME_COUNT_OPTIONS = (2, 3, 4)
FRAME_COUNT_WEIGHTS = (1, 2, 2)
LEFTOVER_RANGE = (3, 5)         # 3..4 markers survive after the output snapshot

# --- CFG: canvas / frame bounds ---
FRAME_MAX = 21                  # rng.integers high for H, W -> 9..20 / 7..20
H_FLOOR, H_CAP = 9, 20
W_FLOOR, W_CAP = 7, 20
LONG_FRAME_COUNT = 4
LONG_FRAME_MAX = 25             # 4-frame paths need a larger search box to stay reachable
LONG_H_CAP, LONG_W_CAP = 24, 24

# --- CFG: path complexity ---
STEP_DIST_RANGE = (2, 6)        # 2..5 inclusive
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 15
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 500
FRAME_TARGET_ATTEMPTS = {2: 250, 3: 400, 4: 1200}
FRAME_TARGET_PRIORITY = {3: 0, 4: 1, 2: 2}
_FRAME_USAGE = {2: 0, 3: 0, 4: 0}


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
    rng: np.random.Generator, target_n_frames: int | None = None
) -> tuple[int, int, int, int, int, int, int, int, int, str]:
    if target_n_frames is None:
        weights = np.array(FRAME_COUNT_WEIGHTS, dtype=float)
        weights /= weights.sum()
        n_frames = int(rng.choice(FRAME_COUNT_OPTIONS, p=weights))
    else:
        n_frames = int(target_n_frames)
    gap = n_frames
    leftover = int(rng.integers(*LEFTOVER_RANGE))
    n_steps = n_frames * gap + leftover

    if n_frames >= LONG_FRAME_COUNT:
        h_cap = LONG_H_CAP
        w_cap = LONG_W_CAP
        frame_max = LONG_FRAME_MAX
    else:
        h_cap = H_CAP
        w_cap = W_CAP
        frame_max = FRAME_MAX

    h_lo = min(h_cap, max(H_FLOOR, 7 + (n_steps + 1) // 2))
    w_lo = min(w_cap, max(W_FLOOR, 5 + (n_steps + 1) // 2))
    h = int(rng.integers(h_lo, frame_max))
    w = int(rng.integers(w_lo, frame_max))

    palette = rng.permutation(10)
    bg, actor, divider = int(palette[0]), int(palette[1]), int(palette[2])

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames, gap, leftover, n_steps, h, w, bg, actor, divider, tname


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


def _build_case(rng: np.random.Generator, target_n_frames: int | None = None) -> dict | None:
    n_frames, gap, leftover, n_steps, h, w, bg, actor, divider, tname = _sample_params(
        rng, target_n_frames=target_n_frames
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
    marker_counts: list[int] = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        future_markers = heads[si + 1 :]
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, future_markers))
        marker_counts.append(len(future_markers))

    stop_idx = n_frames * gap
    stop_head, stop_dir = placements[stop_idx]
    remaining_markers = heads[stop_idx + 1 :]
    out_frame = _render_frame(h, w, bg, actor, stop_head, stop_dir, tname, remaining_markers)
    inp = _assemble_input(frames, bg, divider)

    if marker_counts[-1] != gap + leftover:
        return None
    if any(marker_counts[i] - marker_counts[i + 1] != gap for i in range(n_frames - 1)):
        return None
    if len(remaining_markers) != leftover:
        return None

    found_last = _find_agent(frames[-1], actor)
    if found_last is None:
        return None
    f_tname, f_dir, f_head, _f_agent, f_markers = found_last
    last_idx = (n_frames - 1) * gap
    if (f_tname, f_dir, f_head) != (tname, placements[last_idx][1], placements[last_idx][0]):
        return None
    if f_markers != set(heads[last_idx + 1 :]):
        return None

    found_out = _find_agent(out_frame, actor)
    if found_out is None:
        return None
    o_tname, o_dir, o_head, _o_agent, o_markers = found_out
    if (o_tname, o_dir, o_head) != (tname, stop_dir, stop_head):
        return None
    if o_markers != set(remaining_markers):
        return None

    inferred_gap, inferred_quota, inferred_counts = _infer_gap_and_quota(frames, actor)
    if (inferred_gap, inferred_quota, inferred_counts) != (gap, n_frames, marker_counts):
        return None

    solver_out = main(inp)
    if not np.array_equal(solver_out, out_frame):
        return None

    terminal_out = _simulate_path(frames[-1], actor, stop_after=None)
    if np.array_equal(terminal_out, out_frame):
        return None

    return {
        "input": inp,
        "output": out_frame,
        "meta": {
            "frame_height": h,
            "frame_width": w,
            "n_frames": n_frames,
            "gap": gap,
            "stop_after": gap,
            "leftover_after_output": leftover,
            "total_future_markers": n_steps,
            "marker_counts": list(marker_counts),
            "template": tname,
            "bg": bg,
            "actor": actor,
            "divider": divider,
            "turns": turns,
            "turn_ratio": turns / n_steps,
            "direction_diversity": len(set(step_dirs)),
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "step_distances": dists,
        },
    }


def _preferred_frame_targets() -> list[int]:
    min_used = min(_FRAME_USAGE.values())
    return sorted(
        FRAME_COUNT_OPTIONS,
        key=lambda n: (_FRAME_USAGE[n] != min_used, FRAME_TARGET_PRIORITY[n], _FRAME_USAGE[n]),
    )


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for n_frames in _preferred_frame_targets():
        for _ in range(FRAME_TARGET_ATTEMPTS[n_frames]):
            case = _build_case(rng, target_n_frames=n_frames)
            if case is not None:
                _FRAME_USAGE[n_frames] += 1
                return case["input"]
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            _FRAME_USAGE[case["meta"]["n_frames"]] += 1
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
