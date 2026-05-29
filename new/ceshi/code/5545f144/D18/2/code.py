# Variant of 5545f144
# Dimension: D18 — 步距约束 (meta)
# Change: 每步移动的曼哈顿距离为2-7格，且整条路径至少有2种不同步距。... → 步距由标记颜色编码

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

# -----------------------------------------------------------------------------
# Design analysis for the D18 variant
#
# Concrete realization:
# - The triangle sprite stays single-colored and keeps the original four
#   templates plus the multi-frame horizontal snapshot layout.
# - The consumed dots are no longer the actor color. Instead, three fixed marker
#   colors encode three visible distance bands:
#     color 1 -> step distance 2 or 3
#     color 2 -> step distance 4 or 5
#     color 3 -> step distance 6 or 7
# - A marker is reachable only when BOTH conditions hold:
#   1. it lies on a cardinal half-line from the current head;
#   2. its distance from the head falls inside the band assigned to its color.
# - This changes the critical step-selection site. Example:
#   head at (5, 5), next valid move is to a color-2 marker at (5, 9), distance 4.
#   Another marker may already sit at (2, 5), distance 3, but if it is color 2
#   then it is geometrically aligned yet invalid for that state. The original
#   solver would treat both as ordinary dots and become ambiguous; the variant
#   solver must ignore the color-mismatched one.
#
# Complexity calibration:
# - Exactly three bands cover the old 2..7 range without exploding the number of
#   colors. Two-color banding was too coarse; six exact-distance colors would be
#   harder to infer from a few examples.
# - Generated cases must use at least two marker colors overall and at least two
#   marker colors in the last input frame, so the new rule is visible where the
#   solver actually starts.
# - The generator also requires at least one "color-only conflict" in the last
#   input frame: a future marker that is geometrically tempting from the current
#   head but invalid because its color does not authorize that distance. This
#   prevents silent regression back to the original geometry-only behavior.
#
# Hidden assumptions in the original main() that no longer hold:
# - "Actor color is the only non-bg non-divider color" is false now; markers have
#   their own colors, so actor detection must use shape/connectivity instead.
# - "Markers are leftover pixels of the actor color after template matching" is
#   false; actor detection and marker extraction must be separated.
# - "_choose_target uses the nearest aligned marker regardless of color" is the
#   exact rule being changed; candidate filtering must happen before direction
#   disambiguation.
#
# Placement legality and cleanliness:
# - Markers remain single pixels and are kept 8-separated from each other and
#   from every stamped sprite body seen along the path. This keeps the actor as
#   the only multi-pixel connected color, which the solver relies on.
# - The generator still enforces in-grid placements, no 180-degree reversals,
#   evenly spaced frame cadence, and end-to-end oracle verification with main().
# -----------------------------------------------------------------------------

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left

DIST_COLOR_TO_STEPS: dict[int, frozenset[int]] = {
    1: frozenset({2, 3}),
    2: frozenset({4, 5}),
    3: frozenset({6, 7}),
}
STEP_TO_DIST_COLOR = {
    dist: color
    for color, dists in DIST_COLOR_TO_STEPS.items()
    for dist in dists
}
MARKER_COLORS = tuple(DIST_COLOR_TO_STEPS)


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


def _distance_color(dist: int) -> int:
    return STEP_TO_DIST_COLOR[int(dist)]


def _distance_matches_color(color: int, dist: int) -> bool:
    return int(dist) in DIST_COLOR_TO_STEPS.get(int(color), frozenset())


def _find_agent(
    frame: np.ndarray, actor: int
) -> tuple[str, int, tuple[int, int], set[tuple[int, int]]] | None:
    pts = set(map(tuple, np.argwhere(frame == actor)))
    if not pts:
        return None

    best = None  # (score, template_name, dir, head, placed)
    for head in pts:
        hy, hx = int(head[0]), int(head[1])
        for tname, dirs in TEMPLATES.items():
            for d, offs in dirs.items():
                placed = {(hy + dy, hx + dx) for dy, dx in offs}
                if placed == pts:
                    score = (2, len(placed), 0)
                elif placed.issubset(pts):
                    leftover = len(pts - placed)
                    score = (1, len(placed), -leftover)
                else:
                    continue
                if best is None or score > best[0]:
                    best = (score, tname, d, (hy, hx), placed)

    if best is None:
        return None
    _, tname, d, head, placed = best
    return tname, d, head, set(map(tuple, placed))


def _has_same_color_neighbor(frame: np.ndarray, color: int) -> bool:
    pts = {(int(y), int(x)) for y, x in np.argwhere(frame == color)}
    for y, x in pts:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                if (y + dy, x + dx) in pts:
                    return True
    return False


def _find_actor_color(frame: np.ndarray, bg: int) -> int:
    colors = sorted(set(map(int, frame.ravel())) - {bg})
    candidates = [color for color in colors if _has_same_color_neighbor(frame, color)]
    if len(candidates) == 1:
        return int(candidates[0])

    found = []
    for color in colors:
        agent = _find_agent(frame, int(color))
        if agent is not None:
            found.append(int(color))
    assert len(found) == 1, f"expected one actor color, found {found}"
    return int(found[0])


def _extract_markers(
    frame: np.ndarray, bg: int, actor: int
) -> dict[tuple[int, int], int]:
    markers: dict[tuple[int, int], int] = {}
    ys, xs = np.where((frame != bg) & (frame != actor))
    for y, x in zip(ys, xs):
        markers[(int(y), int(x))] = int(frame[y, x])
    return markers


def _template_bounds(tname: str, d: int) -> tuple[int, int, int, int]:
    offs = TEMPLATES[tname][d]
    rs = [dr for dr, _ in offs]
    cs = [dc for _, dc in offs]
    return min(rs), max(rs), min(cs), max(cs)


def _fits_head(tname: str, d: int, head: tuple[int, int], h: int, w: int) -> bool:
    hy, hx = head
    r0, r1, c0, c1 = _template_bounds(tname, d)
    return 0 <= hy + r0 and hy + r1 < h and 0 <= hx + c0 and hx + c1 < w


def _choose_target(
    head: tuple[int, int],
    d: int,
    tname: str,
    markers: dict[tuple[int, int], int],
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    hy, hx = head

    reachable: list[tuple[int, int, tuple[int, int]]] = []  # (dist, dir, target)
    for (y, x), color in markers.items():
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
        if not _fits_head(tname, dd, (y, x), h, w):
            continue
        if _distance_matches_color(int(color), int(dist)):
            reachable.append((int(dist), dd, (int(y), int(x))))

    if not reachable:
        return None

    dirs_with_candidates = {dd for _, dd, _ in reachable}
    assert len(dirs_with_candidates) == 1, (
        f"head={head} dir={d}: valid markers appear in multiple directions "
        f"{dirs_with_candidates}, expected exactly one"
    )

    reachable.sort(key=lambda item: (item[0], item[1], item[2][0], item[2][1]))
    _, dd, pt = reachable[0]

    assert dd != (d + 2) % 4, (
        f"head={head} dir={d}: would reverse to dir={dd}, not allowed"
    )

    return dd, pt


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, agent = found
    markers = _extract_markers(frame, bg, actor)

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        choice = _choose_target(head, d, tname, markers, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        nd, (ty, tx) = choice

        head = (ty, tx)
        d = nd
        agent = {(ty + oy, tx + ox) for oy, ox in TEMPLATES[tname][d]}
        markers.pop(head, None)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a triangular sprite across the grid by consuming color-coded marker pixels.

    Input structure:
    - Frames are concatenated left-to-right, optionally separated by solid divider columns; the background is the dominant color.
    - Each frame contains one single-color triangular sprite plus isolated one-pixel markers in three fixed colors that encode short, medium, and long step bands.

    Rule:
    - Start from the rightmost frame.
    - At each step, inspect all aligned markers on the four cardinal half-lines from the head, but only treat a marker as reachable when its color matches its distance band from the head.
    - Among the reachable markers there is exactly one valid direction; move the head onto the nearest valid marker on that direction, consume it, and re-orient the sprite to that movement direction. Repeat until every marker is consumed, then output only the final sprite.

    Key insight: Geometry alone is no longer enough — marker color tells you which distances are legal, so some aligned dots must be ignored even though they look reachable spatially.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    start = frames[-1]
    actor = _find_actor_color(start, bg)
    return _simulate_to_end(start, actor)


# --------------- generator (D18 variant: color-encoded distance bands) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (1, 5)          # 1..4 frames
FRAME_MAX = 16                   # H, W sampled in [lo, 15]
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 6, 15

# --- CFG: path complexity ---
STEPS_RANGE = (4, 13)            # total steps 4..12 inclusive
STEP_DIST_RANGE = (2, 8)         # actual distances 2..7 inclusive
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_BUCKET_VARIETY = 2
MIN_LAST_FRAME_BUCKET_VARIETY = 2
MIN_GAP = 3

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 700

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(1, 5)) for n in range(*STEPS_RANGE)}


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
) -> tuple[int, int, int, int, int, int, int, int | None, str]:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [
        n for n in range(*STEPS_RANGE)
        if n % n_frames == 0 and n // n_frames >= MIN_GAP
    ]
    weights = np.array([(n ** 3) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = list(rng.permutation([c for c in range(10) if c not in DIST_COLOR_TO_STEPS]))
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
    marker_color: int,
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    step_dirs: list[int],
    tname: str,
) -> bool:
    """
    Variant-aware future-marker check.

    A future marker may be aligned with an old head as long as it is NOT a valid
    move from that old state under the color-distance rule. This is the precise
    relaxation relative to the original generator.
    """
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, state_dirs[j])
        if any(_touching(nxt, cell) for cell in sprite):
            return False

        rel_dir = _direction(old_head, nxt)
        if rel_dir is None:
            continue

        dist = _dist(old_head, nxt)
        if not _distance_matches_color(marker_color, dist):
            continue

        if rel_dir != step_dirs[j]:
            return False
        if j < len(heads) - 1 and dist <= _dist(old_head, heads[j + 1]):
            return False
    return True


def _rollout_path(
    rng: np.random.Generator, tname: str, step_dirs: list[int], h: int, w: int
) -> tuple[
    list[tuple[int, int]],
    list[int],
    list[tuple[tuple[int, int], int]],
    list[int],
    list[int],
] | None:
    n_steps = len(step_dirs)
    d0 = step_dirs[0]
    heads: list[tuple[int, int]] = [(0, 0)]
    state_dirs: list[int] = [d0]
    fit_placements: list[tuple[tuple[int, int], int]] = [((0, 0), d0)]
    step_dists: list[int] = []
    step_colors: list[int] = []

    for i in range(n_steps):
        cur = heads[-1]
        cur_dir = state_dirs[-1]
        next_dir = step_dirs[i]
        if next_dir == (cur_dir + 2) % 4:
            return None

        dr, dc = DIRS4[next_dir]
        candidates = []
        for dist in list(rng.permutation(np.arange(*STEP_DIST_RANGE))):
            dist = int(dist)
            nxt = (cur[0] + dist * dr, cur[1] + dist * dc)
            marker_color = _distance_color(dist)
            if nxt in heads:
                continue
            if any(_touching(nxt, m) for m in heads[1:]):
                continue
            if not _marker_compatible_with_existing(
                nxt, marker_color, heads, state_dirs, step_dirs, tname
            ):
                continue
            trial_fit = fit_placements + [(nxt, cur_dir), (nxt, next_dir)]
            interval = _shift_interval(trial_fit, h, w, tname)
            if interval is None:
                continue
            tr_lo, tr_hi, tc_lo, tc_hi = interval
            area = (tr_hi - tr_lo + 1) * (tc_hi - tc_lo + 1)
            compactness = abs(nxt[0]) + abs(nxt[1])
            candidates.append((int(area), int(compactness), float(rng.random()), nxt, dist, marker_color))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        _, _, _, nxt, dist, marker_color = candidates[0]
        heads.append(nxt)
        state_dirs.append(next_dir)
        step_dists.append(int(dist))
        step_colors.append(int(marker_color))
        fit_placements.extend([(nxt, cur_dir), (nxt, next_dir)])

    return heads, state_dirs, fit_placements, step_dists, step_colors


def _validate_path(
    step_dirs: list[int],
    heads: list[tuple[int, int]],
    step_dists: list[int],
    last_idx: int,
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

    if len(set(step_dists)) < 2:
        return False, turns, step_dists

    step_colors = [_distance_color(dist) for dist in step_dists]
    if len(set(step_colors)) < MIN_BUCKET_VARIETY:
        return False, turns, step_dists
    if len(set(step_colors[last_idx:])) < MIN_LAST_FRAME_BUCKET_VARIETY:
        return False, turns, step_dists

    if len(heads[last_idx + 1:]) < MIN_GAP:
        return False, turns, step_dists
    return True, turns, step_dists


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
        frame[mr, mc] = int(color)
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


def _remaining_markers(
    heads: list[tuple[int, int]], step_colors: list[int], start_step: int
) -> dict[tuple[int, int], int]:
    return {
        tuple(heads[k + 1]): int(step_colors[k])
        for k in range(start_step, len(step_colors))
    }


def _state_variant_metrics(
    head: tuple[int, int],
    next_dir: int,
    actual_next: tuple[int, int],
    remaining: dict[tuple[int, int], int],
    tname: str,
    h: int,
    w: int,
) -> dict[str, int]:
    actual_dist = _dist(head, actual_next)
    geometric_dirs: set[int] = set()
    valid_dirs: set[int] = set()
    cross_dir_blocked = 0
    nearer_same_dir_blocked = 0

    for pt, color in remaining.items():
        dd = _direction(head, pt)
        if dd is None or not _fits_head(tname, dd, pt, h, w):
            continue
        dist = _dist(head, pt)
        geometric_dirs.add(dd)
        if _distance_matches_color(color, dist):
            valid_dirs.add(dd)
            continue
        if dd != next_dir:
            cross_dir_blocked += 1
        elif dist < actual_dist:
            nearer_same_dir_blocked += 1

    return {
        "geometric_dirs": len(geometric_dirs),
        "valid_dirs": len(valid_dirs),
        "cross_dir_blocked": cross_dir_blocked,
        "nearer_same_dir_blocked": nearer_same_dir_blocked,
        "blocked_total": cross_dir_blocked + nearer_same_dir_blocked,
    }


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, gap, h, w, bg, actor, divider, tname = _sample_params(rng)
    last_idx = (n_frames - 1) * gap
    step_dirs = _make_step_dirs(rng, n_steps)

    rollout = _rollout_path(rng, tname, step_dirs, h, w)
    if rollout is None:
        return None
    heads, state_dirs, fit_placements, step_dists, step_colors = rollout

    ok, turns, dists = _validate_path(step_dirs, heads, step_dists, last_idx)
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

    start_remaining = _remaining_markers(heads, step_colors, last_idx)
    start_metrics = _state_variant_metrics(
        heads[last_idx],
        step_dirs[last_idx],
        heads[last_idx + 1],
        start_remaining,
        tname,
        h,
        w,
    )
    if start_metrics["blocked_total"] < 1:
        return None
    if start_metrics["valid_dirs"] != 1:
        return None

    frames: list[np.ndarray] = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        marker_list = [
            (tuple(heads[k + 1]), int(step_colors[k]))
            for k in range(si, n_steps)
        ]
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, marker_list))

    fhead, fdir = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, fhead, fdir, tname, [])
    inp = _assemble_input(frames, bg, divider)

    found = _find_agent(frames[-1], actor)
    if found is None:
        return None
    f_tname, f_dir, f_head, _f_agent = found
    if (f_tname, f_dir, f_head) != (tname, placements[last_idx][1], placements[last_idx][0]):
        return None

    remaining = _extract_markers(frames[-1], bg, actor)
    if remaining != start_remaining:
        return None

    if _find_actor_color(frames[-1], bg) != actor:
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
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "step_distances": list(dists),
            "step_colors": list(step_colors),
            "distance_bands": {int(k): sorted(v) for k, v in DIST_COLOR_TO_STEPS.items()},
            "count_in_first_frame": n_steps,
            "count_in_last_input_frame": gap,
            "last_frame_bucket_variety": len(set(step_colors[last_idx:])),
            "start_metrics": start_metrics,
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
