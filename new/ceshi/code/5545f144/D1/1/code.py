# Variant of 5545f144
# Dimension: D1 — 匹配条件：方向对齐 (match_cond)
# Change: 精灵头部方向与标记之间的射线对齐——标记必须位于头部所朝方向的同行或同列半线上（direction_matching），且该步仅一个方向有可达标记
#   → 最近邻贪心消耗（忽略方向）

# Design analysis for the D1 variant
# ==================================
# a) Concretization and behavior trace
# - The visible objects stay the same as the base task: one triangular sprite plus
#   several single-pixel dots of the same color, with horizontal frame snapshots
#   separated by full-height divider columns.
# - The changed behavior is only the target-selection rule. The sprite still jumps
#   head-to-marker and still re-stamps one of four cardinal rotations, so every
#   consumed marker is generated on the same row or column as the current head.
#   The difference is that the sprite no longer filters candidates by its current
#   facing direction; it compares all remaining markers by Manhattan distance and
#   consumes the unique nearest one.
# - Small trace:
#   state S: head at (5,5), facing Right, remaining markers at (5,9), (2,5), (5,12)
#   original rule: only the facing ray matters, so it must choose (5,9)
#   variant rule: distances are 4, 3, 7, so it chooses (2,5) instead, then faces Up
#   difference: the path is driven by global distance, not by the arrow tip.
# - Because the sprite only has 4 visible orientations, the generator guarantees
#   that the nearest winner at each step is axis-aligned with the head, even though
#   other non-winning markers may sit elsewhere.
#
# b) Complexity calibration
# - If every step still had exactly one reachable ray, the variant would collapse
#   back to the original visual logic. The generator therefore accepts only paths
#   with at least one "variant-forcing" event: a reverse move, or a state where a
#   non-facing marker wins while another future marker exists on the old facing ray,
#   or a state with axis-aligned future markers in multiple directions.
# - The same four templates are kept so the object class stays recognizable, while
#   path diversity and turn ratios remain similar to the base task.
#
# c) Hidden assumptions from the original main()
# - Original `_choose_target()` assumed all relevant markers lie on one half-line
#   and asserted that exactly one direction has candidates. That assumption is false
#   here and must be removed.
# - Original `_choose_target()` checked in-bounds by translating the current
#   orientation offsets, which is harmless only when the next orientation matches
#   the current one. In this variant turns happen more freely, so fit checking must
#   use the post-move orientation.
# - Original `_simulate_to_end()` relied on at least one step happening before
#   `agent` was defined. The variant keeps non-empty marker sets in practice, but
#   the implementation is hardened by initializing the current sprite body early.
#
# d) Placement legality
# - Every state head plus its rotated template must fit in-grid.
# - Future marker cells must not touch any already-materialized sprite body; this
#   keeps sprite detection stable and prevents marker/sprite blobs from merging.
# - For every earlier state `j`, any later marker must be strictly farther from
#   `head_j` than the intended next marker `head_{j+1}`. That guarantees the new
#   nearest-neighbor solver follows the constructed path with no distance ties.
#
# e) Visual observability
# - Humans can see the change because later frames often contain dots in several
#   directions from the current head, and the output corresponds to the nearest-dot
#   continuation rather than to the arrow tip's current direction.
# - The generator does not hide the variant behind color ids or metadata.
#
# f) Visual cleanliness
# - Markers remain isolated single pixels.
# - Markers and sprite share one color as in the base task, but no-touch spacing
#   keeps the sprite silhouette readable.
#
# g) Color inference
# - Colors are unchanged from the original design: background, actor, and optional
#   divider. No extra color logic is introduced.
#
# h) New solver assumptions and generator guarantees
# - Solver assumption: the unique nearest remaining marker is axis-aligned with the
#   head, so its movement direction maps to one of the four sprite rotations.
# - Generator guarantee: accepted paths enforce unique nearest winners and only use
#   cardinal jumps between consecutive head positions.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

for _extra in (
    Path("/home/lixinyu/snap/new/solvers"),
    Path("/home/lixinyu/snap/new/arc2_solver_select"),
):
    _s = str(_extra)
    if _s not in sys.path:
        sys.path.append(_s)

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L


def _split_frames(g: np.ndarray) -> tuple[list[np.ndarray], int]:
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


LAST_CASE_META: dict | None = None


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


def _sprite_cells(head: tuple[int, int], tname: str, d: int) -> set[tuple[int, int]]:
    hr, hc = head
    return {(hr + dr, hc + dc) for dr, dc in TEMPLATES[tname][d]}


def _head_fits(head: tuple[int, int], tname: str, d: int, h: int, w: int) -> bool:
    hr, hc = head
    for dr, dc in TEMPLATES[tname][d]:
        nr = hr + dr
        nc = hc + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return False
    return True


def _choose_target(
    head: tuple[int, int],
    markers: set[tuple[int, int]],
    tname: str,
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    if not markers:
        return None

    ranked = sorted((_dist(head, pt), int(pt[0]), int(pt[1]), pt) for pt in markers)
    best_dist = ranked[0][0]
    nearest = [pt for dist, _y, _x, pt in ranked if dist == best_dist]
    assert len(nearest) == 1, f"head={head}: nearest marker tie at distance {best_dist}"

    pt = nearest[0]
    nd = _direction(head, pt)
    assert nd is not None, f"head={head}: nearest marker {pt} is not cardinally aligned"
    assert _head_fits(pt, tname, nd, h, w), (
        f"head={head}: landing at {pt} with dir={nd} would place sprite out of bounds"
    )
    return nd, pt


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, _agent, markers = found
    markers = set((int(y), int(x)) for y, x in markers)
    agent = _sprite_cells(head, tname, d)

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        choice = _choose_target(head, markers, tname, h, w)
        assert choice is not None, f"Stuck with {len(markers)} unconsumed markers at head={head}"
        d, head = choice
        agent = _sprite_cells(head, tname, d)
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue a triangular sprite simulation until every marker has been consumed.

    Input structure:
    - Several equal-height frames are concatenated horizontally; non-background
      solid columns act as dividers.
    - Each frame contains one triangular sprite plus same-colored one-pixel dots.

    Rule:
    - Read only the rightmost frame as the current state.
    - Ignore the sprite's current facing when picking the next target: among all
      remaining markers, choose the unique nearest one by Manhattan distance.
    - The head jumps onto that marker, the marker disappears, and the sprite is
      re-stamped facing the move direction. Repeat until no markers remain.

    Key insight: The arrow tip still tells you how the sprite is drawn after each
    jump, but it no longer decides which dot must be eaten next; distance does.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    return _simulate_to_end(frames[-1], actor)


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
MIN_VARIANT_EVENTS = 1

TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

MAX_ATTEMPTS = 3000
REVERSE_SEGMENT_PROB = 0.35

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
        prev = segments[-1]
        if rng.random() < REVERSE_SEGMENT_PROB:
            nxt = (prev + 2) % 4
        else:
            nxt = int((prev + rng.choice([1, -1])) % 4)
        segments.append(nxt)

    lengths = np.ones(n_segments, dtype=int)
    for _ in range(n_steps - n_segments):
        lengths[int(rng.integers(0, n_segments))] += 1

    out: list[int] = []
    for d, ln in zip(segments, lengths):
        out.extend([d] * int(ln))
    return out


def _candidate_compatible(
    nxt: tuple[int, int],
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
) -> bool:
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, state_dirs[j])
        if any(_touching(nxt, cell) for cell in sprite):
            return False
        if j < len(heads) - 1 and _dist(old_head, nxt) <= _dist(old_head, heads[j + 1]):
            return False
    return True


def _rollout_path(
    rng: np.random.Generator, tname: str, step_dirs: list[int], h: int, w: int
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]] | None:
    heads: list[tuple[int, int]] = [(0, 0)]
    state_dirs: list[int] = [step_dirs[0]]
    placements: list[tuple[tuple[int, int], int]] = [((0, 0), step_dirs[0])]

    for next_dir in step_dirs:
        cur = heads[-1]
        dr, dc = DIRS4[next_dir]
        dist_order = list(rng.permutation(np.arange(*STEP_DIST_RANGE)))
        chosen = None
        for dist in dist_order:
            nxt = (cur[0] + int(dist) * dr, cur[1] + int(dist) * dc)
            if nxt in heads:
                continue
            if not _candidate_compatible(nxt, heads, state_dirs, tname):
                continue
            trial = placements + [(nxt, next_dir)]
            if _shift_interval(trial, h, w, tname) is None:
                continue
            chosen = nxt
            break

        if chosen is None:
            return None

        heads.append(chosen)
        state_dirs.append(next_dir)
        placements.append((chosen, next_dir))

    return heads, state_dirs, placements


def _variant_event_counts(
    heads: list[tuple[int, int]], state_dirs: list[int]
) -> dict[str, int]:
    reverse = 0
    forward_competition = 0
    axis_ambiguity = 0
    nonfacing = 0

    for i in range(len(heads) - 1):
        cur = heads[i]
        facing = state_dirs[i]
        intended = heads[i + 1]
        intended_dir = _direction(cur, intended)
        intended_dist = _dist(cur, intended)
        assert intended_dir is not None

        if intended_dir != facing:
            nonfacing += 1
        if intended_dir == (facing + 2) % 4:
            reverse += 1

        dir_to_dists: dict[int, list[int]] = {}
        for pt in heads[i + 1 :]:
            dd = _direction(cur, pt)
            if dd is None:
                continue
            dir_to_dists.setdefault(dd, []).append(_dist(cur, pt))

        forward_dists = dir_to_dists.get(facing, [])
        if forward_dists and intended_dir != facing and intended_dist < min(forward_dists):
            forward_competition += 1
        if intended_dir != facing and len(dir_to_dists) >= 2:
            axis_ambiguity += 1

    return {
        "reverse": reverse,
        "forward_competition": forward_competition,
        "axis_ambiguity": axis_ambiguity,
        "nonfacing": nonfacing,
    }


def _validate_path(
    step_dirs: list[int], heads: list[tuple[int, int]], state_dirs: list[int]
) -> tuple[bool, int, list[int], dict[str, int]]:
    n_steps = len(step_dirs)
    dir_set = set(step_dirs)
    if len(dir_set) < MIN_DIR_DIVERSITY:
        return False, 0, [], {}
    if not (dir_set & {0, 2} and dir_set & {1, 3}):
        return False, 0, [], {}

    turns = sum(step_dirs[i] != step_dirs[i - 1] for i in range(1, n_steps))
    if turns < MIN_TURNS_ACCEPT or not (TURN_RATIO_LO <= turns / n_steps <= TURN_RATIO_HI):
        return False, turns, [], {}

    dists = [_dist(heads[i], heads[i + 1]) for i in range(n_steps)]
    if len(set(dists)) < MIN_DIST_VARIETY:
        return False, turns, dists, {}

    events = _variant_event_counts(heads, state_dirs)
    if (
        events["reverse"] + events["forward_competition"] + events["axis_ambiguity"]
        < MIN_VARIANT_EVENTS
    ):
        return False, turns, dists, events

    return True, turns, dists, events


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
    heads, state_dirs, placements = rollout

    ok, turns, dists, events = _validate_path(step_dirs, heads, state_dirs)
    if not ok:
        return None

    shift = _shift_interval(placements, h, w, tname)
    if shift is None:
        return None
    tr_lo, tr_hi, tc_lo, tc_hi = shift
    tr = int(rng.integers(tr_lo, tr_hi + 1))
    tc = int(rng.integers(tc_lo, tc_hi + 1))
    heads = [(r + tr, c + tc) for r, c in heads]
    placements = [(heads[i], state_dirs[i]) for i in range(n_steps + 1)]

    frames: list[np.ndarray] = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, heads[si + 1 :]))

    fhead, fdir = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, fhead, fdir, tname, [])
    inp = _assemble_input(frames, bg, divider)

    last_idx = (n_frames - 1) * gap
    found = _find_agent(frames[-1], actor)
    if found is None:
        return None
    f_tname, f_dir, f_head, _f_agent, f_markers = found
    if (f_tname, f_dir, f_head) != (tname, placements[last_idx][1], placements[last_idx][0]):
        return None
    if f_markers != set(heads[last_idx + 1 :]):
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
            "step_distances": dists,
            "count_in_first_frame": n_steps,
            "count_in_last_input_frame": gap,
            "variant_events": dict(events),
        },
    }


def generate_input() -> np.ndarray:
    global LAST_CASE_META

    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            LAST_CASE_META = case["meta"]
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
