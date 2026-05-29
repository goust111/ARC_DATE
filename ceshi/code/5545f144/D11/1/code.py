# Variant of 5545f144
# Dimension: D11 — 路径复杂度约束 (meta)
# Change: total_steps 4-12，turn_ratio 0.4-0.9，至少使用3个基数方向，至少2次转弯，步距2-7且... → 螺旋路径约束

from __future__ import annotations

import importlib.util
from functools import lru_cache
import itertools
from pathlib import Path
import sys

import numpy as np

_SOLVERS_DIR = Path(__file__).resolve().parents[5] / "solvers"
if str(_SOLVERS_DIR) not in sys.path:
    sys.path.append(str(_SOLVERS_DIR))

from arc_utils import get_bg as _bg_color
from arc_utils import paste

"""
Design analysis for the D11 spiral variant
=========================================

a) Concrete behavior + trace
- The visible objects stay the same as the base task: one triangle sprite, one actor
  color shared by the sprite and the single-pixel markers, optional full-height
  divider columns, and an output that contains only the final sprite.
- The changed part is the marker path. Instead of an arbitrary non-reversing
  polyline, the future marker heads must lie on a rectangular spiral. The
  generator now builds a four-side spiral backbone whose parallel arms are
  spaced apart, then subdivides each side into 1-4 marker steps. Consecutive
  segment directions keep one fixed quarter-turn sign (clockwise or
  counterclockwise), while the backbone side lengths follow a paired monotone
  pattern such as 7,7,4,4 or 4,4,7,7.
- Small trace:
  old path could be R2, D3, R2, U4. It turns right, then left, so its handedness
  flips and the polyline is just a free zig-zag.
  new path can use R7, D7, L2+2, U2+2. The per-step moves are still axis-aligned,
  but the distinct sides are R7, D7, L4, U4, so every turn is clockwise and the
  markers visibly trace one rectangular coil.
  Solver effect: marker consumption is still "move to the nearest aligned marker",
  but now every non-straight turn must keep the same sign for the whole rollout.

b) Complexity calibration
- The spiral should be obvious to a human from 2-4 examples, so the generator does
  not merely forbid reverse turns; it uses an explicit four-side spiral backbone
  with matched opposite sides, then places multiple markers along each side when
  more total steps are needed. This keeps the path regular instead of degenerating
  into a generic one-sided walk.
- To avoid degeneration, the generator still varies:
  total steps 4-12, frame count 1-4, template choice, start direction, clockwise
  vs counterclockwise handedness, inward vs outward traversal, backbone size, and
  the per-side marker subdivision.
- The path necessarily uses all four cardinal directions, and higher-step examples
  still show varied move distances because long sides are broken into different
  2+ pixel chunks.

c) Hidden assumptions audit
- `_split_frames`, template matching, and actor-color detection do not depend on
  path geometry, so they remain valid.
- `_choose_target` assumes that from the current head there is at most one cardinal
  half-line containing reachable markers. The spiral generator must still satisfy
  this older assumption so the task stays visually parsable and the solver does not
  need unrelated disambiguation logic.
- The original `_simulate_to_end` only rejected 180-degree reversals. That is no
  longer sufficient: a path like R,D,R,U would pass the old check but violates the
  spiral rule. The new solver therefore tracks the sign of the first quarter-turn
  it observes and asserts that every later turn has the same sign.
- `_find_agent` can fail if a marker touches the sprite and changes the best-matched
  template. The generator still uses marker-safety checks and an oracle round-trip
  to prevent that.

d) Placement legality
- Every future marker is a future head position, so generator legality must cover:
  no marker overlap with any sprite body in frames where that marker is still
  present; no out-of-bounds sprite placement before or after each turn; and no
  duplicate or touching marker heads that would visually merge.
- The generator enumerates only prevalidated spiral backbones, expands them into
  marker-level steps, validates each new head against all earlier heads using the
  same half-line ambiguity checks as the base generator, then searches frame sizes
  8-15 x 6-15 that admit a valid translation for every orientation encountered.

e) Visual observability
- The new rule is visible without relying on hidden metadata or color ids:
  the marker trail itself curls clockwise or counterclockwise with regular
  side-length changes. A human can compare examples and see that the sprite is
  following the remaining spiral tail, not an arbitrary turn sequence.
- Handedness is randomized between clockwise and counterclockwise, and traversal
  direction is randomized between expanding and contracting, so the dataset does
  not collapse into one memorized picture.

f) Visual cleanliness
- Markers stay isolated single pixels and are prevented from touching the sprite,
  so the triangle remains easy to spot even though it shares the same color with
  the marker trail.
- The spiral is drawn by sparse head markers only; there are no extra decoration
  pixels that would clutter the frames.

g) Color reasoning
- Colors do not participate in any ordering or mapping. The only color logic is the
  original one: background is the most common color, divider is a solid non-bg
  color, and the actor color is the remaining non-bg non-divider color.

Post-change main() assumptions
- Input frames still obey the unique-half-line marker property from the base task.
- Remaining moves from the rightmost frame form a one-handed quarter-turn sequence.
- The generator guarantees both assumptions by explicit spiral construction and by
  rejecting any case that fails `_find_agent` or `main(input) == output`.
"""

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left
DISTRACTOR_COLOR = 8

_ORIGINAL_5545F144 = None


def _matches_original(inp: np.ndarray, out: np.ndarray) -> bool:
    global _ORIGINAL_5545F144
    if _ORIGINAL_5545F144 is None:
        spec = importlib.util.spec_from_file_location(
            "orig_5545f144", "/home/lixinyu/snap/new/arc2_solver_select/5545f144.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        _ORIGINAL_5545F144 = module
    try:
        original_out = np.asarray(_ORIGINAL_5545F144.main(np.asarray(inp, dtype=int)), dtype=int)
    except Exception:
        return False
    return np.array_equal(original_out, np.asarray(out, dtype=int))


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


def _detect_actor(frames: list[np.ndarray], bg: int, div_colors: set[int]) -> int:
    colors = set(map(int, np.concatenate([frame.ravel() for frame in frames])))
    best: tuple[tuple[int, int], int] | None = None
    for color in sorted(colors - {bg} - div_colors):
        detections = [_find_agent(frame, color) for frame in frames]
        count = sum(found is not None for found in detections)
        if count == 0:
            continue
        marker_total = sum(len(found[4]) for found in detections if found is not None)
        score = (count, marker_total)
        if best is None or score > best[0]:
            best = (score, color)
    assert best is not None, "No actor-colored sprite detected"
    return best[1]


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


def _sprite_cells(head: tuple[int, int], tname: str, d: int) -> set[tuple[int, int]]:
    hr, hc = head
    return {(hr + dr, hc + dc) for dr, dc in TEMPLATES[tname][d]}


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


def _add_distractors(
    rng: np.random.Generator, frame: np.ndarray, bg: int, color: int, count: int = 1
) -> np.ndarray | None:
    out = frame.copy()
    occupied = set(map(tuple, np.argwhere(frame != bg)))
    candidates = [
        (r, c)
        for r in range(frame.shape[0])
        for c in range(frame.shape[1])
        if frame[r, c] == bg and not any(_touching((r, c), cell) for cell in occupied)
    ]
    if len(candidates) < count:
        return None
    picks = rng.choice(len(candidates), size=count, replace=False)
    for idx in np.atleast_1d(picks):
        r, c = candidates[int(idx)]
        out[r, c] = color
    return out


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


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, agent, markers = found
    markers = {(int(y), int(x)) for y, x in markers}
    spiral_sign: int | None = None

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

        head = (ty, tx)
        d = nd
        agent = _sprite_cells(head, tname, d)
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance the triangular sprite along the remaining spiral trail of markers.

    Input structure:
    - One to four snapshots are concatenated left-to-right, with optional full-height
      divider columns in a distinct non-background color.
    - Each frame contains the same triangular sprite template plus the still-unvisited
      single-pixel marker heads, all drawn in one actor color.

    Rule:
    - Read the rightmost snapshot as the current state.
    - From the sprite head, the remaining markers continue on aligned half-lines;
      the sprite always moves to the nearest reachable marker on the unique such
      half-line.
    - After each move, the path must keep a single spiral handedness: every
      non-straight turn is a 90-degree turn with the same sign (clockwise or
      counterclockwise) until all markers are consumed. Output the final sprite only.

    Key insight: the triangle tip gives the current facing, and the marker trail is
    not an arbitrary bendy path but the unfinished tail of a one-handed rectangular spiral.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    actor = _detect_actor(frames, bg, div_colors)

    start = frames[-1]
    return _simulate_to_end(start, actor)


# --------------- generator (spiral path variant) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (1, 5)  # 1..4
FRAME_MAX = 16           # sizes sampled from 8..15 / 6..15
H_FLOOR, W_FLOOR = 8, 6

# --- CFG: spiral complexity ---
STEPS_RANGE = (4, 13)    # 4..12 inclusive
STEP_DIST_MIN = 2
SPIRAL_SIGNS = (-1, 1)
SPIRAL_BACKBONES = (
    (7, 7, 4, 4),
    (8, 8, 5, 5),
    (9, 9, 6, 6),
    (10, 10, 6, 6),
    (11, 11, 7, 7),
)
MAX_PLANS_PER_BUCKET = 24

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 3000

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(1, 5)) for n in range(*STEPS_RANGE)}


def _sample_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int | None, str]:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0]
    weights = np.array([(n ** 3) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    palette = [int(c) for c in rng.permutation(10) if int(c) != DISTRACTOR_COLOR]
    bg, actor = palette[0], palette[1]
    divider = palette[2] if n_frames > 1 else None

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames, n_steps, gap, bg, actor, divider, tname


def _step_count_compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]
    out: list[tuple[int, ...]] = []
    for cuts in itertools.combinations(range(1, total), parts - 1):
        prev = 0
        comp = []
        for c in cuts + (total,):
            comp.append(c - prev)
            prev = c
        out.append(tuple(comp))
    return out


@lru_cache(maxsize=None)
def _segment_partitions(total_len: int, count: int) -> tuple[tuple[int, ...], ...]:
    if count <= 0:
        return ()
    if count == 1:
        return ((total_len,),)

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


def _turn_sequence_sign(dirs: list[int]) -> int | None:
    sign: int | None = None
    for prev_dir, next_dir in zip(dirs, dirs[1:]):
        turn = _signed_quarter_turn(prev_dir, next_dir)
        if turn is None or turn == 0:
            return None
        if sign is None:
            sign = turn
        elif sign != turn:
            return None
    return sign


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


def _validate_spiral_path(
    tname: str,
    heads: list[tuple[int, int]],
    step_dirs: list[int],
    dists: list[int],
) -> tuple[bool, int | None, int | None, list[int], list[tuple[tuple[int, int], int]]]:
    n_steps = len(step_dirs)
    if not (STEPS_RANGE[0] <= n_steps < STEPS_RANGE[1]):
        return False, None, None, [], []
    if any(dist < STEP_DIST_MIN for dist in dists):
        return False, None, None, [], []
    seg_dirs, seg_lens = _segment_profile(step_dirs, dists)
    if len(seg_dirs) != 4 or len(set(seg_dirs)) != 4:
        return False, None, None, [], []

    turn_sign = _turn_sequence_sign(seg_dirs)
    if turn_sign is None:
        return False, None, None, [], []

    if seg_lens[0] != seg_lens[1] or seg_lens[2] != seg_lens[3]:
        return False, None, None, [], []
    if seg_lens[0] == seg_lens[2] or abs(seg_lens[0] - seg_lens[2]) < 2:
        return False, None, None, [], []
    trend = 1 if seg_lens[2] > seg_lens[0] else -1

    state_dirs = [step_dirs[0]]
    fit_placements: list[tuple[tuple[int, int], int]] = [(heads[0], state_dirs[0])]
    built_heads = [heads[0]]

    for i, nxt in enumerate(heads[1:]):
        cur_dir = state_dirs[-1]
        next_dir = step_dirs[i]
        if next_dir == (cur_dir + 2) % 4:
            return False, None, None, [], []
        if not _marker_compatible_with_existing(nxt, built_heads, state_dirs, step_dirs, tname):
            return False, None, None, [], []
        built_heads.append(nxt)
        state_dirs.append(next_dir)
        fit_placements.extend([(nxt, cur_dir), (nxt, next_dir)])

    return True, turn_sign, trend, state_dirs, fit_placements


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
                                    ok, _turn_sign, _trend, _state_dirs, _fit = _validate_spiral_path(
                                        tname, heads, step_dirs, dists
                                    )
                                    if ok:
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
                                break
            assert bucket, f"no spiral plans found for template={tname} steps={n_steps}"
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
    idx = int(rng.integers(0, len(feasible)))
    return feasible[idx]


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
    n_frames, n_steps, gap, bg, actor, divider, tname = _sample_params(rng)

    plans = SAFE_SPIRAL_PLANS[tname][n_steps]
    plan = plans[int(rng.integers(0, len(plans)))]
    start_dir, turn_seed = plan["valid_orients"][int(rng.integers(0, len(plan["valid_orients"])))]
    heads, step_dirs, dists = _materialize_plan(
        tuple(plan["backbone"]),
        tuple(tuple(seg) for seg in plan["parts"]),
        int(start_dir),
        int(turn_seed),
        bool(plan["reverse"]),
    )
    ok, turn_sign, trend, state_dirs, fit_placements = _validate_spiral_path(
        tname, heads, step_dirs, dists
    )
    if not ok or turn_sign is None or trend is None:
        return None

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
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, heads[si + 1:]))
    noisy_frames = []
    for frame in frames:
        noisy = _add_distractors(rng, frame, bg, DISTRACTOR_COLOR, count=1)
        if noisy is None:
            return None
        noisy_frames.append(noisy)
    frames = noisy_frames

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
    if f_markers != set(heads[last_idx + 1:]):
        return None
    if not np.array_equal(main(inp), out_frame):
        return None

    handedness = "clockwise" if turn_sign == 1 else "counterclockwise"
    radial_trend = "expanding" if trend == 1 else "contracting"
    seg_dirs, seg_lens = _segment_profile(step_dirs, dists)

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
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "step_distances": list(dists),
            "segment_dirs": list(seg_dirs),
            "segment_lengths": list(seg_lens),
            "segment_step_counts": list(plan["counts"]),
            "spiral_handedness": handedness,
            "spiral_turn_sign": int(turn_sign),
            "distance_trend": radial_trend,
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
    raise RuntimeError(f"failed to generate a valid spiral input after {MAX_ATTEMPTS} attempts")
