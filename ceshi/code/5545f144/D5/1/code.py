# Variant of 5545f144
# Dimension: D5 -- 路径规则：标记消耗机制 (path)
# Change: 精灵到达标记位置时只消耗该标记 -> 精灵到达标记位置时，会消耗该标记以及
# 其曼哈顿距离 2 以内的所有标记（区域消耗 / blast radius = 2）。

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
for _parent in _HERE.parents:
    _solvers = _parent / "solvers"
    if _solvers.is_dir():
        if str(_solvers) not in sys.path:
            sys.path.append(str(_solvers))
        break

from arc_utils import get_bg as _bg_color
from arc_utils import paste

# ---------------------------------------------------------------------------
# Design analysis for the D5 variant
#
# a) Concrete behavior trace
# - Visual objects stay the same as the source task: one monochrome triangular
#   sprite plus monochrome single-cell marker dots inside each frame.
# - The change happens only after the head reaches the chosen aligned marker.
#   Old rule: remove exactly that landing dot. New rule: remove every dot whose
#   Manhattan distance from the landing head is <= 2, so a small diamond-shaped
#   neighborhood disappears at once.
# - Example: if the head moves onto (5, 8) and there are markers at
#   (5, 8), (4, 8), (5, 10), (6, 9), then the old task would remove only
#   (5, 8), while the variant removes all four in one move.
#
# b) Complexity calibration
# - The path logic is intentionally not simplified: the next move is still
#   chosen by the nearest aligned marker on the unique reachable half-line.
# - To make the blast visible, the generator adds extra dots around some future
#   landing cells, so initial marker count is larger than the number of moves.
# - The number of extra dots is kept small (1-2 per selected landing) to avoid
#   excessive clutter and to preserve repeated evidence across frames.
#
# c) Hidden assumptions from the original solver
# - `_choose_target(...)` only depends on which markers are currently visible on
#   row/column half-lines, so it still works.
# - `_simulate_to_end(...)` originally assumed `markers.discard(head)` was
#   sufficient. That assumption breaks under area consumption and must be
#   replaced with radius-based removal.
# - The original generator assumed every future head remains present as a marker
#   until that exact step. Under blast removal this is false unless consecutive
#   heads are farther than radius 2, so step distances are raised to >= 3.
#
# d) Placement legality
# - Extra dots are attached to a specific future landing cluster. A candidate
#   extra dot is accepted only if the full forward simulation still chooses the
#   same primary landing sequence and only if no future marker gets destroyed
#   too early by an earlier blast.
# - Visible markers are also kept non-touching from the current sprite at every
#   simulated state to keep agent detection reliable.
#
# e) Visual observability
# - Each accepted instance must contain at least two multi-dot blasts, and at
#   least one of them must still be visible in the rightmost input frame.
# - This ensures the training/test examples visibly contain clusters that cannot
#   be explained by the original "consume exactly one dot per move" rule.
#
# f) Visual cleanliness
# - The sprite template set, colors, divider logic, and frame cadence remain
#   unchanged, so only marker grouping changes.
# - Extra dots are local to a landing site instead of being scattered globally,
#   which keeps the board structured rather than noisy.
#
# g) Color reasoning
# - No new color logic is introduced. Sprite and markers still share one color,
#   divider stays a separate solid color, and no ordering or rule depends on
#   color ids.
# ---------------------------------------------------------------------------

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
BLAST_RADIUS = 2
BLAST_OFFSETS = [
    (dr, dc)
    for dr in range(-BLAST_RADIUS, BLAST_RADIUS + 1)
    for dc in range(-BLAST_RADIUS, BLAST_RADIUS + 1)
    if 0 < abs(dr) + abs(dc) <= BLAST_RADIUS
]

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


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _consume_region(
    markers: set[tuple[int, int]], center: tuple[int, int], radius: int = BLAST_RADIUS
) -> set[tuple[int, int]]:
    cy, cx = center
    return {
        (y, x)
        for y, x in markers
        if abs(y - cy) + abs(x - cx) > radius
    }


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


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, agent, markers = found
    markers = {(int(y), int(x)) for y, x in markers}

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        offs = TEMPLATES[tname][d]
        choice = _choose_target(head, d, markers, offs, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        nd, (ty, tx) = choice

        head = (ty, tx)
        d = nd
        offs = TEMPLATES[tname][d]
        agent = {(ty + oy, tx + ox) for oy, ox in offs}

        # Variant rule: landing detonates a Manhattan-radius-2 cleanup.
        markers = _consume_region(markers, head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance a triangular sprite through aligned markers, with each landing clearing a local region.

    Input structure:
    - Several equal-height frames may be concatenated left-to-right, separated by solid divider columns.
    - Each frame contains one monochrome triangular sprite plus same-colored single-cell marker dots on the background.

    Rule:
    - Read the rightmost frame as the current state.
    - From the sprite head, exactly one cardinal half-line has reachable markers; move to the nearest marker on that line while keeping the rotated sprite in bounds.
    - After landing, re-face the sprite along that travel direction and erase every marker whose Manhattan distance from the landing head is at most 2.
    - Repeat until no markers remain, then output only the final sprite on a clean background.

    Key insight: the aligned nearest marker still determines the next move, but nearby off-axis dots disappear as collateral damage when that landing cell is reached.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    start = frames[-1]
    return _simulate_to_end(start, actor)


# --------------- generator (D5 regional-consumption variant) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (1, 5)
FRAME_MAX = 16
H_FLOOR, H_CAP = 9, 15
W_FLOOR, W_CAP = 7, 15

# --- CFG: path complexity ---
STEPS_RANGE = (4, 11)            # total moves: 4..10
STEP_DIST_RANGE = (3, 8)         # primary landings stay beyond blast radius
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

# --- CFG: cluster / blast observability ---
MULTI_STEPS_RANGE = (2, 5)       # choose 2..4 landings to receive extra dots
EXTRAS_PER_CLUSTER_RANGE = (1, 3)  # 1..2 extra dots
MIN_TOTAL_EXTRAS = 3
MIN_MULTI_STEPS = 2

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 8
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
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
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0]
    weights = np.array([(n ** 3) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    h_lo = min(H_CAP, max(H_FLOOR, 7 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 5 + (n_steps + 2) // 2))
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


def _union_markers(clusters: list[set[tuple[int, int]]], start: int = 0) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for cluster in clusters[start:]:
        out |= cluster
    return out


def _markers_clear_of_sprite(
    head: tuple[int, int],
    d: int,
    tname: str,
    markers: set[tuple[int, int]],
) -> bool:
    sprite = _sprite_cells(head, tname, d)
    return all(not _touching(marker, cell) for marker in markers for cell in sprite)


def _step_dirs_from_heads(heads: list[tuple[int, int]]) -> list[int]:
    return [int(_direction(heads[i], heads[i + 1])) for i in range(len(heads) - 1)]


def _validate_blast_plan(
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
    h: int,
    w: int,
    clusters: list[set[tuple[int, int]]],
) -> bool:
    step_dirs = _step_dirs_from_heads(heads)
    n_steps = len(step_dirs)
    for i in range(n_steps):
        visible = _union_markers(clusters, i)
        if not _markers_clear_of_sprite(heads[i], state_dirs[i], tname, visible):
            return False
        try:
            choice = _choose_target(
                heads[i], state_dirs[i], visible, TEMPLATES[tname][state_dirs[i]], h, w
            )
        except AssertionError:
            return False
        if choice != (step_dirs[i], heads[i + 1]):
            return False
        post = _consume_region(visible, heads[i + 1])
        if post != _union_markers(clusters, i + 1):
            return False
    return True


def _build_clusters(
    rng: np.random.Generator,
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
    h: int,
    w: int,
    last_idx: int,
) -> list[set[tuple[int, int]]] | None:
    n_steps = len(heads) - 1
    clusters = [{heads[i + 1]} for i in range(n_steps)]
    if not _validate_blast_plan(heads, state_dirs, tname, h, w, clusters):
        return None

    all_steps = list(range(n_steps))
    suffix_steps = [i for i in all_steps if i >= last_idx]
    if not suffix_steps:
        return None

    n_multi = min(n_steps, int(rng.integers(*MULTI_STEPS_RANGE)))
    chosen = {int(rng.choice(suffix_steps))}
    remaining = [i for i in all_steps if i not in chosen]
    rng.shuffle(remaining)
    for i in remaining:
        if len(chosen) >= n_multi:
            break
        chosen.add(int(i))
    chosen_steps = list(chosen)
    rng.shuffle(chosen_steps)

    used = set().union(*clusters)
    for step in chosen_steps:
        goal = int(rng.integers(*EXTRAS_PER_CLUSTER_RANGE))
        offsets = list(BLAST_OFFSETS)
        rng.shuffle(offsets)
        added = 0
        for dr, dc in offsets:
            pos = (heads[step + 1][0] + dr, heads[step + 1][1] + dc)
            if not (0 <= pos[0] < h and 0 <= pos[1] < w):
                continue
            if pos in used:
                continue
            trial = [set(cluster) for cluster in clusters]
            trial[step].add(pos)
            if _validate_blast_plan(heads, state_dirs, tname, h, w, trial):
                clusters = trial
                used.add(pos)
                added += 1
                if added >= goal:
                    break

    total_extras = sum(len(cluster) - 1 for cluster in clusters)
    multi_steps = sum(len(cluster) > 1 for cluster in clusters)
    suffix_multi = sum(len(clusters[i]) > 1 for i in suffix_steps)
    if total_extras < MIN_TOTAL_EXTRAS:
        return None
    if multi_steps < MIN_MULTI_STEPS or suffix_multi < 1:
        return None
    return clusters


def _render_frame(
    h: int,
    w: int,
    bg: int,
    actor: int,
    head: tuple[int, int],
    d: int,
    tname: str,
    markers: set[tuple[int, int]],
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

    last_idx = (n_frames - 1) * gap
    clusters = _build_clusters(rng, heads, state_dirs, tname, h, w, last_idx)
    if clusters is None:
        return None

    frames: list[np.ndarray] = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        visible = _union_markers(clusters, si)
        if not _markers_clear_of_sprite(head, d, tname, visible):
            return None
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, visible))

    fhead, fdir = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, fhead, fdir, tname, set())
    inp = _assemble_input(frames, bg, divider)

    for fi, frame in enumerate(frames):
        si = fi * gap
        found = _find_agent(frame, actor)
        if found is None:
            return None
        f_tname, f_dir, f_head, _f_agent, f_markers = found
        if (f_tname, f_dir, f_head) != (tname, placements[si][1], placements[si][0]):
            return None
        if f_markers != _union_markers(clusters, si):
            return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out_frame):
        return None

    total_markers = sum(len(cluster) for cluster in clusters)
    cluster_sizes = [len(cluster) for cluster in clusters]
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
            "total_moves": n_steps,
            "gap": gap,
            "turns": turns,
            "turn_ratio": turns / n_steps,
            "direction_diversity": len(set(step_dirs)),
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "step_distances": dists,
            "initial_marker_count": total_markers,
            "extra_marker_count": total_markers - n_steps,
            "cluster_sizes": cluster_sizes,
            "multi_blast_steps": sum(size > 1 for size in cluster_sizes),
            "last_frame_multi_steps": sum(size > 1 for size in cluster_sizes[last_idx:]),
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
