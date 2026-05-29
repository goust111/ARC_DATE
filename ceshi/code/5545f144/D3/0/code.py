# Variant of 5545f144
# Dimension: D3 — 目标对象：标记类型 (dst)
# Change: 散布的单像素标记点，与精灵共用同一颜色（actor色），被消耗后消失。... → 多色标记点对应消耗顺序

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left

# Visible, fixed marker cycle used across the whole task family:
# red -> blue -> green -> red -> ...
MARKER_CYCLE = (2, 1, 3)
MARKER_COLORS = set(MARKER_CYCLE)
NON_MARKER_PALETTE = [0, 4, 5, 6, 7, 8, 9]


# ---------------------------------------------------------------------------
# Design analysis for the D3 variant
#
# spec_variant:
#   task: triangular sprite follows only the next marker whose COLOR matches a
#     global visible cycle; wrong-color markers remain visible decoys
#   colors:
#     background: most common color
#     actor: one non-bg color never reused by markers
#     markers: exactly the three visible colors in MARKER_CYCLE = red, blue, green
#     divider: optional full-height solid column color, distinct from the above
#   concrete appearance:
#     - The sprite keeps the original triangle templates and still uses a single color.
#     - Markers remain isolated one-pixel dots, but are now split across three colors.
#     - Step i consumes the marker at path position i with color cycle[i mod 3].
#   behavior trace:
#     - Original: from the current head, any aligned marker could be the next target,
#       so generation had to forbid markers on multiple half-lines.
#     - Variant: from the current head, only aligned markers of the NEXT cycle color
#       are eligible. Wrong-color aligned markers may coexist and serve as visible decoys.
#     - Example:
#         head facing right, reachable red above, blue right. If the next cycle color
#         is blue, the sprite must go right; the red dot stays for later.
#       This is the core difference from the original same-color rule.
#   complexity calibration:
#     - The cycle length is fixed at 3. That is enough to force sequence reasoning,
#       but still short enough to be repeated several times within 4..12 steps.
#     - Marker colors are fixed visible categories, not derived from numeric color ids.
#     - We keep the original path geometry so the only new reasoning burden is color order.
#   implicit-assumption review for main():
#     - Original actor detection assumed "all non-bg, non-divider cells share one color".
#       This is false now, so actor must instead be the unique non-marker, non-divider color.
#     - Original _find_agent treated leftover actor-colored cells as markers.
#       This is false now; markers are collected from MARKER_COLORS only.
#     - Original target selection used all markers and asserted one direction globally.
#       This is false now; the assertion must apply only to markers of the required color.
#     - Original start state needed no step index. This variant does: the last frame is
#       step (F-1)*gap, where gap = remaining_marker_count and total_steps = F*gap, so
#       the next cycle phase is inferred as start_step = (F-1)*gap.
#   placement legality:
#     - The generator still builds a path first, then renders snapshots.
#     - Every landing head and stamped sprite must fit the frame.
#     - Markers never overlap sprite cells or each other.
#     - For each past state j, only markers whose color equals cycle[j] are constrained
#       to the unique next half-line; markers of other colors may align elsewhere.
#   visual observability:
#     - The generator rejects cases where ignoring marker colors would produce the same
#       final output. This prevents silent fallback to the original task.
#     - It also requires a wrong-color decoy aligned from the last input frame, so the
#       color-order rule is visible directly in the solving frame.
#   neatness:
#     - Markers remain isolated dots with the original spacing constraints.
#     - The actor color is disjoint from marker colors, so the sprite is easy to locate.
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


def _find_agent(
    frame: np.ndarray, actor: int
) -> tuple[str, int, tuple[int, int], set[tuple[int, int]]] | None:
    pts = set(map(tuple, np.argwhere(frame == actor)))
    if not pts:
        return None

    for head in pts:
        hy, hx = int(head[0]), int(head[1])
        for tname, dirs in TEMPLATES.items():
            for d, offs in dirs.items():
                placed = {(hy + dy, hx + dx) for dy, dx in offs}
                if placed == pts:
                    return tname, d, (hy, hx), placed
    return None


def _marker_map(frame: np.ndarray) -> dict[tuple[int, int], int]:
    markers: dict[tuple[int, int], int] = {}
    for color in MARKER_CYCLE:
        for y, x in np.argwhere(frame == color):
            markers[(int(y), int(x))] = color
    return markers


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


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: dict[tuple[int, int], int],
    required_color: int,
    offs: list[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    hy, hx = head

    def candidates(dir_idx: int) -> list[tuple[int, tuple[int, int]]]:
        out: list[tuple[int, tuple[int, int]]] = []
        for (y, x), color in markers.items():
            if color != required_color:
                continue
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
        f"head={head} dir={d}: required color {required_color} reachable in "
        f"multiple directions {dirs_with_candidates}"
    )

    reachable.sort()
    _, dd, pt = reachable[0]

    assert dd != (d + 2) % 4, (
        f"head={head} dir={d}: would reverse to dir={dd}, not allowed"
    )
    return dd, pt


def _choose_target_ignore_colors(
    head: tuple[int, int],
    d: int,
    marker_positions: set[tuple[int, int]],
    offs: list[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    all_markers = {pt: MARKER_CYCLE[0] for pt in marker_positions}
    return _choose_target(head, d, all_markers, MARKER_CYCLE[0], offs, h, w)


def _simulate_to_end(frame: np.ndarray, actor: int, start_step: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, _agent = found
    markers = _marker_map(frame)
    agent = set(_agent)

    max_steps = h * w * 4
    step = start_step
    for _ in range(max_steps):
        if not markers:
            break

        required_color = MARKER_CYCLE[step % len(MARKER_CYCLE)]
        offs = TEMPLATES[tname][d]
        choice = _choose_target(head, d, markers, required_color, offs, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} markers at head={head} dir={d}; "
            f"need color {required_color}"
        )
        nd, (ty, tx) = choice

        head = (ty, tx)
        d = nd
        offs = TEMPLATES[tname][d]
        agent = {(ty + oy, tx + ox) for oy, ox in offs}
        markers.pop(head)
        step += 1

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def _simulate_ignore_colors(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, _agent = found
    markers = set(_marker_map(frame))
    agent = set(_agent)

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break
        offs = TEMPLATES[tname][d]
        choice = _choose_target_ignore_colors(head, d, markers, offs, h, w)
        assert choice is not None
        nd, (ty, tx) = choice
        head = (ty, tx)
        d = nd
        offs = TEMPLATES[tname][d]
        agent = {(ty + oy, tx + ox) for oy, ox in offs}
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a triangular sprite across the grid, consuming aligned marker pixels in a color cycle.

    Input structure:
    - Frames are concatenated horizontally and optional divider columns are solid, full-height, and non-background.
    - Each frame contains one single-color triangular sprite plus isolated one-pixel markers in three fixed colors:
      red, blue, and green.

    Rule:
    - Start from the rightmost frame.
    - Marker consumption follows a repeating color order: red -> blue -> green -> red -> ...
    - From the current head, only markers of the next required color are eligible; among those, the sprite moves to
      the nearest reachable marker on a row/column half-line and re-faces that direction.
    - The rightmost frame is a snapshot taken after some steps, so recover the cycle phase using:
      remaining_markers = gap, consumed_steps = (number_of_frames - 1) * gap.
    - Repeat until all markers are consumed; output only the final sprite on a clean background.

    Key insight: the triangle still gives direction, but the decisive filter is the visible red-blue-green cycle, and
    the current phase is recoverable from the frame cadence.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor_candidates = colors - {bg} - div_colors - MARKER_COLORS
    assert len(actor_candidates) == 1, f"expected one actor color, got {actor_candidates}"
    actor = next(iter(actor_candidates))

    start = frames[-1]
    remaining_markers = len(_marker_map(start))
    start_step = (len(frames) - 1) * remaining_markers
    return _simulate_to_end(start, actor, start_step)


# --------------- generator ---------------

N_FRAME_VALUES = np.array([1, 2, 3, 4], dtype=int)
N_FRAME_WEIGHTS = np.array([0.58, 0.24, 0.12, 0.06], dtype=float)
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
MIN_GAP = 2

TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

MAX_ATTEMPTS = 500

_STEP_AVAILABILITY = {
    n: sum(n % f == 0 and n // f >= MIN_GAP for f in range(1, 5))
    for n in range(*STEPS_RANGE)
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


def _sprite_cells(head: tuple[int, int], tname: str, d: int) -> set[tuple[int, int]]:
    hr, hc = head
    return {(hr + dr, hc + dc) for dr, dc in TEMPLATES[tname][d]}


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
    n_frames = int(rng.choice(N_FRAME_VALUES, p=N_FRAME_WEIGHTS))
    multiples = [
        n for n in range(*STEPS_RANGE)
        if n % n_frames == 0 and n // n_frames >= MIN_GAP
    ]
    weights = np.array([1 / (n ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(max(h_lo, 10), FRAME_MAX))
    w = int(rng.integers(max(w_lo, 8), FRAME_MAX))

    palette = list(rng.permutation(NON_MARKER_PALETTE))
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
        out.extend([d] * int(ln))
    return out


def _marker_compatible_with_existing(
    nxt: tuple[int, int],
    nxt_color: int,
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    step_dirs: list[int],
    step_colors: list[int],
    tname: str,
) -> bool:
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, state_dirs[j])
        if any(_touching(nxt, cell) for cell in sprite):
            return False
        rel_dir = _direction(old_head, nxt)
        if rel_dir is None:
            continue
        if nxt_color != step_colors[j]:
            continue
        if rel_dir != step_dirs[j]:
            return False
        if j < len(heads) - 1 and _dist(old_head, nxt) <= _dist(old_head, heads[j + 1]):
            return False
    return True


def _rollout_path(
    rng: np.random.Generator,
    tname: str,
    step_dirs: list[int],
    step_colors: list[int],
    h: int,
    w: int,
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
        next_color = step_colors[i]
        if next_dir == (cur_dir + 2) % 4:
            return None

        dr, dc = DIRS4[next_dir]
        dist_values = np.arange(*STEP_DIST_RANGE)
        dist_weights = 1 / dist_values.astype(float)
        dist_weights /= dist_weights.sum()
        dist_order = list(rng.choice(dist_values, size=len(dist_values), replace=False, p=dist_weights))
        chosen = None
        for dist in dist_order:
            nxt = (cur[0] + int(dist) * dr, cur[1] + int(dist) * dc)
            if nxt in heads:
                continue
            if any(_touching(nxt, m) for m in heads[1:]):
                continue
            if not _marker_compatible_with_existing(
                nxt, next_color, heads, state_dirs, step_dirs, step_colors, tname
            ):
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
    markers: list[tuple[tuple[int, int], int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    patch, ar, ac = _template_patch(tname, d, actor, bg)
    paste(frame, patch, head[0] - ar, head[1] - ac, transparent=bg)
    for (mr, mc), color in markers:
        frame[mr, mc] = color
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


def _snapshot_has_wrong_color_decoy(
    step_idx: int,
    heads: list[tuple[int, int]],
    step_dirs: list[int],
    step_colors: list[int],
) -> bool:
    cur = heads[step_idx]
    need_dir = step_dirs[step_idx]
    need_color = step_colors[step_idx]
    for later in range(step_idx, len(step_dirs)):
        pos = heads[later + 1]
        color = step_colors[later]
        rel = _direction(cur, pos)
        if rel is None:
            continue
        if color != need_color and rel != need_dir:
            return True
    return False


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, gap, h, w, bg, actor, divider, tname = _sample_params(rng)
    step_dirs = _make_step_dirs(rng, n_steps)
    step_colors = [MARKER_CYCLE[i % len(MARKER_CYCLE)] for i in range(n_steps)]

    rollout = _rollout_path(rng, tname, step_dirs, step_colors, h, w)
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
        markers = [(heads[j + 1], step_colors[j]) for j in range(si, n_steps)]
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, markers))

    fhead, fdir = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, fhead, fdir, tname, [])
    inp = _assemble_input(frames, bg, divider)

    start = frames[-1]
    if len(_marker_map(start)) != gap:
        return None
    if not np.array_equal(main(inp), out_frame):
        return None
    try:
        if np.array_equal(_simulate_ignore_colors(start, actor), out_frame):
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
            "actor": actor,
            "divider": divider,
            "total_steps": n_steps,
            "gap": gap,
            "turns": turns,
            "turn_ratio": turns / n_steps,
            "direction_diversity": len(set(step_dirs)),
            "step_dirs": list(step_dirs),
            "step_colors": list(step_colors),
            "heads": list(heads),
            "step_distances": dists,
            "last_frame_has_wrong_color_decoy": _snapshot_has_wrong_color_decoy(
                (n_frames - 1) * gap, heads, step_dirs, step_colors
            ),
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
