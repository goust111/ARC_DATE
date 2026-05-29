# Combo variant of 5545f144 (index 13)
# combo_spec: D6/3+D11/1
# Name: 轨迹颜色编码方向+螺旋路径约束

from __future__ import annotations

import itertools
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

_SOLVERS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVERS_DIR) not in sys.path:
    sys.path.append(str(_SOLVERS_DIR))

from arc_utils import get_bg as _bg_color
from arc_utils import paste

# -----------------------------------------------------------------------------
# Design analysis for combo D6/3 + D11/1
#
# a) Concrete behavior and trace
#    - Base task: the actor-colored triangular sprite jumps between actor-colored
#      single-pixel markers. Old positions disappear; only the final sprite is
#      output.
#    - Fused task: the remaining markers must lie on a one-handed rectangular
#      spiral, and every consumed head segment paints a width-1 trail whose color
#      encodes direction: up=2, right=1, down=3, left=4.
#    - Small trace: head at (6,2), facing right, with future markers on a spiral
#      tail at (6,6)->(10,6)->(10,3)->(8,3). The move to (6,6) paints blue on
#      (6,2..5), then the move to (10,6) paints green on (6..9,6), then the move
#      left paints yellow, and so on. The output keeps the full colored spiral
#      trail plus the final triangle.
#    - Difference from base: the visible state is no longer "latest sprite only".
#      The rightmost frame already stores past motion, and future motion must keep
#      the same quarter-turn sign whenever it turns.
#
# b) Complexity calibration
#    - Direction colors are fixed to four canonical ARC colors. That keeps the
#      mapping small enough to infer, yet repeated enough across examples that the
#      color rule is visible.
#    - Spiral structure is not implemented as a weak "no opposite turn" rule. The
#      generator explicitly builds four-side rectangular coils with paired side
#      lengths, then subdivides each side into 2+ pixel marker jumps. This makes
#      the spiral shape visually obvious rather than an arbitrary bendy path.
#    - The actor, background, and divider colors are sampled only from non-trail
#      colors so that trail cells cannot be mistaken for sprite pixels or dividers.
#
# c) Hidden assumptions audit
#    - Original `_split_frames()` treated any full-height solid non-bg column as a
#      divider. That breaks once a trail column can be solid, so divider detection
#      must ignore the four trail colors.
#    - Original actor extraction assumed a unique non-bg non-divider color. In the
#      fused task there are four more non-bg colors, so actor detection must remove
#      trail colors first.
#    - Original `_simulate_to_end()` only rejected 180-degree reversals. The fused
#      solver also has to track the sign of the first non-straight turn and enforce
#      that every later 90-degree turn has the same sign.
#    - Rendering a clean background at the end is no longer correct because the
#      trail is part of the final answer. The solver must preserve existing trail
#      from the latest frame and extend it.
#
# d) Placement legality
#    - Future marker heads are the future sprite heads, so every generated head and
#      every rotated template placement must fit in-grid after translation.
#    - No marker may touch any earlier sprite body; otherwise `_find_agent()` can
#      become ambiguous. This is checked incrementally with the same half-line
#      compatibility rule as the base task.
#    - Trail legality adds extra requirements: a new trail segment cannot reuse an
#      old trail cell, a later head cannot land on old trail, and a later sprite
#      body cannot cover old trail. The generator validates all three.
#
# e) Visual observability
#    - A human can see both fused dimensions directly: the trail is colored by
#      direction, and the whole route curls clockwise or counterclockwise as a
#      rectangular spiral. No hidden ordering by numeric color id is used.
#    - The rightmost frame still suffices for solving because it contains the
#      current arrow, the remaining markers, and the already-painted prefix trail.
#
# f) Visual cleanliness
#    - The trail is head-width 1 instead of a full swept body, which avoids thick
#      blobs around turns.
#    - Only the hollow 3-pixel triangle is used. Thicker templates would hide
#      already-painted trail cells under the sprite body, which would make the
#      last snapshot lose information the solver cannot recover.
#    - Markers remain isolated actor-colored dots, and the generator rejects cases
#      where trail or sprite placements would visually merge with them.
#
# g) Inferability of the color change
#    - Only four trail colors are introduced, each tied to a repeated geometric
#      condition (segment orientation). Across a few examples the solver can infer
#      that vertical/horizontal direction, not palette rank, determines recoloring.
#    - We also require at least two remaining steps and at least two directions in
#      the rightmost-frame suffix so the continuation from input to output still
#      exhibits the fused rule rather than collapsing to a trivial one-segment tail.
#
# Post-change main() assumptions guaranteed by the generator
#    - Divider columns never use trail colors.
#    - Exactly one actor color remains after removing background, divider, and trail.
#    - The rightmost frame contains a valid triangle detection, a legal trail
#      prefix, and a remaining marker suffix whose non-straight turns all share one
#      handedness.
# -----------------------------------------------------------------------------

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
TRAIL_BY_DIR = {0: 2, 1: 1, 2: 3, 3: 4}
TRAIL_COLORS = set(TRAIL_BY_DIR.values())
NON_TRAIL_COLORS = (0, 5, 6, 7, 8, 9)
ACTOR_COLORS = (5, 6, 7, 8, 9)

# --- CFG: canvas / frame bounds ---
N_FRAMES_OPTIONS = (2, 3, 4)
FRAME_MAX = 16           # sample h,w from [8,15] / [6,15]
H_FLOOR, W_FLOOR = 8, 6

# --- CFG: spiral complexity ---
STEPS_RANGE = (4, 13)    # 4..12
STEP_DIST_MIN = 2
MIN_SUFFIX_STEPS = 2
MIN_SUFFIX_DIR_DIVERSITY = 2
SPIRAL_SIGNS = (-1, 1)
SPIRAL_BACKBONES = (
    (7, 7, 4, 4),
    (8, 8, 5, 5),
    (9, 9, 6, 6),
    (10, 10, 6, 6),
    (11, 11, 7, 7),
)
MAX_PLANS_PER_BUCKET = 24

# --- CFG: retry budget ---
MAX_ATTEMPTS = 3000
TRAIL_SAFE_TEMPLATES = ("hollow3",)


def _split_frames(g: np.ndarray) -> tuple[list[np.ndarray], int]:
    """
    Input is multiple frames concatenated horizontally. Divider columns are
    full-height solid columns whose color is neither the background nor a trail color.
    """
    h, w = g.shape
    bg = _bg_color(g)

    solid_cols = [c for c in range(w) if len(set(map(int, g[:, c]))) == 1]
    div_cols = [
        c for c in solid_cols
        if int(g[0, c]) != bg and int(g[0, c]) not in TRAIL_COLORS
    ]

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


def _signed_quarter_turn(prev_dir: int, next_dir: int) -> int | None:
    delta = (next_dir - prev_dir) % 4
    if delta == 0:
        return 0
    if delta == 1:
        return 1
    if delta == 3:
        return -1
    return None


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
            elif dir_idx == 1 and y == hy and x > hx:
                out.append((x - hx, (y, x)))
            elif dir_idx == 2 and x == hx and y > hy:
                out.append((y - hy, (y, x)))
            elif dir_idx == 3 and y == hy and x < hx:
                out.append((hx - x, (y, x)))
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


def _segment_cells(
    a: tuple[int, int], b: tuple[int, int], *, include_end: bool
) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    dr = 0 if br == ar else (1 if br > ar else -1)
    dc = 0 if bc == ac else (1 if bc > ac else -1)
    dist = abs(br - ar) + abs(bc - ac)
    stop = dist + 1 if include_end else dist
    return [(ar + dr * k, ac + dc * k) for k in range(stop)]


def _extract_trail(frame: np.ndarray) -> dict[tuple[int, int], int]:
    trail: dict[tuple[int, int], int] = {}
    for y, x in np.argwhere(np.isin(frame, list(TRAIL_COLORS))):
        trail[(int(y), int(x))] = int(frame[y, x])
    return trail


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


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)
    trail = _extract_trail(frame)

    found = _find_agent(frame, actor)
    if found is None:
        out = np.full((h, w), bg, dtype=int)
        for (y, x), color in trail.items():
            out[y, x] = color
        return out

    tname, d, head, _agent, markers = found
    markers = {(int(y), int(x)) for y, x in markers}
    spiral_sign: int | None = None
    agent = _sprite_cells(head, tname, d)

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

        turn = _signed_quarter_turn(d, nd)
        assert turn is not None, f"head={head} dir={d}: 180-degree reversal to {nd}"
        if turn != 0:
            if spiral_sign is None:
                spiral_sign = turn
            else:
                assert turn == spiral_sign, (
                    f"head={head} dir={d}: turn sign {turn} breaks spiral handedness "
                    f"{spiral_sign}"
                )

        for cell in _segment_cells(head, target, include_end=False):
            trail[cell] = TRAIL_BY_DIR[nd]

        head = target
        d = nd
        agent = _sprite_cells(head, tname, d)
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for (y, x), color in trail.items():
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = color
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue the triangle sprite simulation with a direction-colored spiral trail.

    Input structure:
    - One to four frames are concatenated left-to-right, optionally separated by solid divider columns whose color is distinct from the background and the four trail colors.
    - Each frame contains an actor-colored triangular sprite, actor-colored remaining marker dots, and any already-painted trail cells:
      red for upward motion, blue for rightward motion, green for downward motion, yellow for leftward motion.

    Rule:
    - Read the rightmost frame as the latest state. Keep the trail already visible there.
    - Repeatedly find the unique reachable marker half-line, move the sprite head to the nearest marker on that half-line, and color every vacated head-path cell by the direction of that move.
    - The remaining path is constrained to a rectangular spiral: every non-straight turn is a 90-degree turn with the same handedness throughout the continuation. Stop when all markers are consumed and output the accumulated trail plus the final sprite body.

    Key insight: the triangle tip reveals the current facing, while the colored trail reveals past directions; together with the single-handed turn pattern of the remaining markers, the rightmost snapshot is enough to continue the unfinished spiral.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {
        int(g[0, c]) for c in solid_cols
        if int(g[0, c]) != bg and int(g[0, c]) not in TRAIL_COLORS
    }
    colors = set(map(int, g.ravel()))
    actor_candidates = colors - {bg} - div_colors - TRAIL_COLORS
    assert len(actor_candidates) == 1, f"expected 1 actor color, got {actor_candidates}"
    actor = next(iter(actor_candidates))

    return _simulate_to_end(frames[-1], actor)


def _sample_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int | None, str]:
    n_frames = int(rng.choice(N_FRAMES_OPTIONS))
    gaps = sorted({
        n // n_frames
        for n in range(*STEPS_RANGE)
        if n % n_frames == 0 and n // n_frames >= MIN_SUFFIX_STEPS
    })
    gap = int(rng.choice(gaps))
    n_steps = gap * n_frames

    bg = int(rng.choice(NON_TRAIL_COLORS))
    actor = int(rng.choice([c for c in ACTOR_COLORS if c != bg]))
    divider = None
    if n_frames > 1:
        divider_choices = [c for c in NON_TRAIL_COLORS if c not in {bg, actor}]
        divider = int(rng.choice(divider_choices))

    tname = str(rng.choice(TRAIL_SAFE_TEMPLATES))
    return n_frames, n_steps, gap, bg, actor, divider, tname


def _step_count_compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]
    out: list[tuple[int, ...]] = []
    for cuts in itertools.combinations(range(1, total), parts - 1):
        prev = 0
        cur = []
        for c in cuts + (total,):
            cur.append(c - prev)
            prev = c
        out.append(tuple(cur))
    return out


@lru_cache(maxsize=None)
def _segment_partitions(total_len: int, count: int) -> tuple[tuple[int, ...], ...]:
    if count <= 0:
        return ()
    if count == 1:
        return ((total_len,),) if total_len >= STEP_DIST_MIN else ()

    out: list[tuple[int, ...]] = []

    def rec(remaining: int, slots: int, cur: list[int]) -> None:
        if slots == 1:
            if remaining >= STEP_DIST_MIN:
                out.append(tuple(cur + [remaining]))
            return
        hi = remaining - STEP_DIST_MIN * (slots - 1)
        for step in range(STEP_DIST_MIN, hi + 1):
            cur.append(step)
            rec(remaining - step, slots - 1, cur)
            cur.pop()

    rec(total_len, count, [])
    return tuple(out)


def _materialize_plan(
    backbone: tuple[int, ...],
    parts: tuple[tuple[int, ...], ...],
    start_dir: int,
    turn_seed: int,
    reverse: bool,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    seg_dirs = [int((start_dir + turn_seed * i) % 4) for i in range(len(backbone))]
    heads: list[tuple[int, int]] = [(0, 0)]
    step_dirs: list[int] = []
    dists: list[int] = []

    for seg_dir, seg_parts in zip(seg_dirs, parts):
        dr, dc = DIRS4[seg_dir]
        for dist in seg_parts:
            hr, hc = heads[-1]
            heads.append((hr + dr * dist, hc + dc * dist))
            step_dirs.append(seg_dir)
            dists.append(int(dist))

    if reverse:
        heads = list(reversed(heads))
        step_dirs = []
        dists = []
        for a, b in zip(heads, heads[1:]):
            d = _direction(a, b)
            assert d is not None
            step_dirs.append(d)
            dists.append(_dist(a, b))
    return heads, step_dirs, dists


def _segment_profile(step_dirs: list[int], dists: list[int]) -> tuple[list[int], list[int]]:
    seg_dirs = [step_dirs[0]]
    seg_lens = [0]
    for d, dist in zip(step_dirs, dists):
        if d != seg_dirs[-1]:
            seg_dirs.append(d)
            seg_lens.append(0)
        seg_lens[-1] += dist
    return seg_dirs, seg_lens


def _validate_spiral_path(
    tname: str,
    heads: list[tuple[int, int]],
    step_dirs: list[int],
    dists: list[int],
) -> tuple[bool, int | None, int | None, list[int], list[tuple[tuple[int, int], int]], set[tuple[int, int]]]:
    n_steps = len(step_dirs)
    if not (STEPS_RANGE[0] <= n_steps < STEPS_RANGE[1]):
        return False, None, None, [], [], set()
    if any(dist < STEP_DIST_MIN for dist in dists):
        return False, None, None, [], [], set()

    seg_dirs, seg_lens = _segment_profile(step_dirs, dists)
    if len(seg_dirs) != 4 or len(set(seg_dirs)) != 4:
        return False, None, None, [], [], set()

    turn_sign = _signed_quarter_turn(seg_dirs[0], seg_dirs[1])
    if turn_sign not in SPIRAL_SIGNS:
        return False, None, None, [], [], set()
    for prev_dir, next_dir in zip(seg_dirs, seg_dirs[1:]):
        turn = _signed_quarter_turn(prev_dir, next_dir)
        if turn != turn_sign:
            return False, None, None, [], [], set()

    if seg_lens[0] != seg_lens[1] or seg_lens[2] != seg_lens[3]:
        return False, None, None, [], [], set()
    if abs(seg_lens[0] - seg_lens[2]) < STEP_DIST_MIN:
        return False, None, None, [], [], set()
    trend = 1 if seg_lens[2] > seg_lens[0] else -1

    state_dirs = [step_dirs[0]]
    fit_placements: list[tuple[tuple[int, int], int]] = [(heads[0], state_dirs[0])]
    built_heads = [heads[0]]
    used_trace: set[tuple[int, int]] = set()

    for i, nxt in enumerate(heads[1:]):
        cur = built_heads[-1]
        cur_dir = state_dirs[-1]
        next_dir = step_dirs[i]

        if next_dir == (cur_dir + 2) % 4:
            return False, None, None, [], [], set()
        if nxt in used_trace:
            return False, None, None, [], [], set()
        if not _marker_compatible_with_existing(nxt, built_heads, state_dirs, step_dirs, tname):
            return False, None, None, [], [], set()

        seg_trace = _segment_cells(cur, nxt, include_end=False)
        if any(cell in used_trace for cell in seg_trace):
            return False, None, None, [], [], set()
        if any(cell in built_heads[:-1] for cell in seg_trace[1:]):
            return False, None, None, [], [], set()

        nxt_sprite = _sprite_cells(nxt, tname, next_dir)
        if nxt_sprite & used_trace:
            return False, None, None, [], [], set()

        built_heads.append(nxt)
        state_dirs.append(next_dir)
        fit_placements.extend([(nxt, cur_dir), (nxt, next_dir)])
        used_trace.update(seg_trace)

    return True, turn_sign, trend, state_dirs, fit_placements, used_trace


def _build_safe_plan_catalog() -> dict[str, dict[int, list[dict[str, object]]]]:
    catalog: dict[str, dict[int, list[dict[str, object]]]] = {
        tname: {n_steps: [] for n_steps in range(*STEPS_RANGE)} for tname in TEMPLATES
    }

    for tname in TEMPLATES:
        for n_steps in range(*STEPS_RANGE):
            bucket = catalog[tname][n_steps]
            for backbone in SPIRAL_BACKBONES:
                if len(bucket) >= MAX_PLANS_PER_BUCKET:
                    break
                seg_count = len(backbone)
                if n_steps < seg_count:
                    continue
                for counts in _step_count_compositions(n_steps, seg_count):
                    if len(bucket) >= MAX_PLANS_PER_BUCKET:
                        break
                    part_lists = [_segment_partitions(side_len, cnt) for side_len, cnt in zip(backbone, counts)]
                    if any(not part_list for part_list in part_lists):
                        continue
                    for parts in itertools.product(*part_lists):
                        if len(bucket) >= MAX_PLANS_PER_BUCKET:
                            break
                        for reverse in (False, True):
                            valid_orients: list[tuple[int, int]] = []
                            for start_dir in range(4):
                                for turn_seed in SPIRAL_SIGNS:
                                    heads, step_dirs, dists = _materialize_plan(
                                        backbone, parts, start_dir, turn_seed, reverse
                                    )
                                    ok, turn_sign, trend, _state_dirs, _fit, used_trace = _validate_spiral_path(
                                        tname, heads, step_dirs, dists
                                    )
                                    if ok and turn_sign is not None and trend is not None:
                                        valid_orients.append((start_dir, turn_seed))
                            if valid_orients:
                                bucket.append(
                                    {
                                        "backbone": backbone,
                                        "parts": tuple(tuple(seg) for seg in parts),
                                        "counts": tuple(len(seg) for seg in parts),
                                        "reverse": reverse,
                                        "valid_orients": tuple(valid_orients),
                                    }
                                )
            assert bucket, f"no safe spiral plans for template={tname} steps={n_steps}"
    return catalog


SAFE_SPIRAL_PLANS = _build_safe_plan_catalog()


def _pick_frame_shape(
    rng: np.random.Generator,
    tname: str,
    fit_placements: list[tuple[tuple[int, int], int]],
) -> tuple[int, int, tuple[int, int, int, int]] | None:
    feasible: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for h in range(H_FLOOR, FRAME_MAX):
        for w in range(W_FLOOR, FRAME_MAX):
            shift = _shift_interval(fit_placements, h, w, tname)
            if shift is not None:
                feasible.append((h, w, shift))
    if not feasible:
        return None
    return feasible[int(rng.integers(0, len(feasible)))]


def _trail_map(
    heads: list[tuple[int, int]], step_dirs: list[int], upto_steps: int
) -> dict[tuple[int, int], int]:
    trail: dict[tuple[int, int], int] = {}
    for i in range(upto_steps):
        for cell in _segment_cells(heads[i], heads[i + 1], include_end=False):
            trail[cell] = TRAIL_BY_DIR[step_dirs[i]]
    return trail


def _render_frame(
    h: int,
    w: int,
    bg: int,
    actor: int,
    head: tuple[int, int],
    d: int,
    tname: str,
    markers: list[tuple[int, int]],
    trail: dict[tuple[int, int], int],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    for (y, x), color in trail.items():
        frame[y, x] = color
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


def _has_suffix_variety(step_dirs: list[int], gap: int) -> bool:
    suffix = step_dirs[-gap:]
    return len(suffix) >= MIN_SUFFIX_STEPS and len(set(suffix)) >= MIN_SUFFIX_DIR_DIVERSITY


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, gap, bg, actor, divider, tname = _sample_params(rng)

    candidates: list[dict[str, object]] = []
    for plan in SAFE_SPIRAL_PLANS[tname][n_steps]:
        for orient in plan["valid_orients"]:
            start_dir, turn_seed = int(orient[0]), int(orient[1])
            heads, step_dirs, dists = _materialize_plan(
                tuple(plan["backbone"]),
                tuple(tuple(seg) for seg in plan["parts"]),
                start_dir,
                turn_seed,
                bool(plan["reverse"]),
            )
            if not _has_suffix_variety(step_dirs, gap):
                continue
            ok, turn_sign, trend, state_dirs, fit_placements, used_trace = _validate_spiral_path(
                tname, heads, step_dirs, dists
            )
            if not ok or turn_sign is None or trend is None:
                continue
            candidates.append(
                {
                    "plan": plan,
                    "heads": heads,
                    "step_dirs": step_dirs,
                    "dists": dists,
                    "turn_sign": turn_sign,
                    "trend": trend,
                    "state_dirs": state_dirs,
                    "fit_placements": fit_placements,
                    "used_trace": used_trace,
                }
            )
    if not candidates:
        return None

    available_signs = sorted({int(cand["turn_sign"]) for cand in candidates})
    desired_sign = int(rng.choice(available_signs))
    sign_pool = [cand for cand in candidates if int(cand["turn_sign"]) == desired_sign]
    available_trends = sorted({int(cand["trend"]) for cand in sign_pool})
    desired_trend = int(rng.choice(available_trends))
    pool = [cand for cand in sign_pool if int(cand["trend"]) == desired_trend]

    chosen = pool[int(rng.integers(0, len(pool)))]
    plan = chosen["plan"]
    heads = list(chosen["heads"])
    step_dirs = list(chosen["step_dirs"])
    dists = list(chosen["dists"])
    turn_sign = int(chosen["turn_sign"])
    trend = int(chosen["trend"])
    state_dirs = list(chosen["state_dirs"])
    fit_placements = list(chosen["fit_placements"])
    used_trace = set(chosen["used_trace"])

    shape_pick = _pick_frame_shape(rng, tname, fit_placements)
    if shape_pick is None:
        return None
    h, w, shift = shape_pick
    tr_lo, tr_hi, tc_lo, tc_hi = shift
    tr = int(rng.integers(tr_lo, tr_hi + 1))
    tc = int(rng.integers(tc_lo, tc_hi + 1))
    heads = [(r + tr, c + tc) for r, c in heads]
    placements = [(heads[i], state_dirs[i]) for i in range(n_steps + 1)]

    frames = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        trail = _trail_map(heads, step_dirs, si)
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, heads[si + 1:], trail))

    fhead, fdir = placements[-1]
    out_trail = _trail_map(heads, step_dirs, n_steps)
    out_frame = _render_frame(h, w, bg, actor, fhead, fdir, tname, [], out_trail)
    inp = _assemble_input(frames, bg, divider)

    last_idx = (n_frames - 1) * gap
    found = _find_agent(frames[-1], actor)
    if found is None:
        return None
    f_tname, f_dir, f_head, _f_agent, f_markers = found
    if (f_tname, f_dir, f_head) != (tname, placements[last_idx][1], placements[last_idx][0]):
        return None
    if f_markers != set(heads[last_idx + 1:]):
        return None
    expected_last_trail = _trail_map(heads, step_dirs, last_idx)
    if _extract_trail(frames[-1]) != expected_last_trail:
        return None
    if not np.array_equal(main(inp), out_frame):
        return None

    seg_dirs, seg_lens = _segment_profile(step_dirs, dists)
    handedness = "clockwise" if turn_sign == 1 else "counterclockwise"
    trend_name = "expanding" if trend == 1 else "contracting"

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
            "heads": list(heads),
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "step_distances": list(dists),
            "segment_dirs": list(seg_dirs),
            "segment_lengths": list(seg_lens),
            "segment_step_counts": list(plan["counts"]),
            "spiral_handedness": handedness,
            "spiral_turn_sign": int(turn_sign),
            "spiral_trend": trend_name,
            "trail_cells_last_input_frame": len(expected_last_trail),
            "trail_cells_output": len(out_trail),
            "trail_dir_diversity_output": len(set(out_trail.values())),
            "count_in_first_frame": n_steps,
            "count_in_last_input_frame": gap,
            "suffix_dir_diversity": len(set(step_dirs[-gap:])),
            "used_trace_cells": len(used_trace),
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate fused spiral-trail input after {MAX_ATTEMPTS} attempts")
