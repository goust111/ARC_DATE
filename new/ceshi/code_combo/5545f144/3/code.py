# Combo variant of 5545f144 (index 3)
# combo_spec: D1/3+D4/0+D14/0
# Name: 行列对齐无方向限定+允许反转+多方向取最近
#
# Design analysis for this combo variant
# --------------------------------------
# a) Concrete behavior and trace
# - The visible objects stay the same as in the source task: 1..4 horizontally
#   concatenated frames, each showing one monochrome triangular sprite plus
#   same-color single-pixel markers. The output is still a single clean frame
#   with only the final sprite.
# - The fused rule removes all direction filtering from target selection. From a
#   head at (5, 5), markers at (5, 2), (2, 5), and (5, 9) are all legal
#   candidates if the translated / landed sprite fits. The next move is whichever
#   aligned marker is uniquely nearest by Manhattan distance.
# - Compared with the original task:
#   original: exactly one cardinal half-line could contain markers, and the move
#   could not be the reverse of the current facing.
#   combo: markers may exist on several directions at once; the triangle may go
#   left, right, up, or down regardless of its current tip; a 180-degree turn is
#   fully legal if that marker is nearest.
# - Example trace:
#   current head faces Right at (5, 5), remaining markers are at (5, 3), (2, 5),
#   and (5, 10). Original logic either rejects the state (multiple directions) or
#   forbids moving left because that is behind the tip. The combo logic chooses
#   (5, 3) because distance 2 beats distances 3 and 5, then the sprite re-faces
#   Left. This single step simultaneously exhibits D1/3, D4/0, and D14/0.
#
# b) Complexity calibration
# - Templates, colors, divider layout, frame cadence, and marker-is-single-pixel
#   appearance are preserved to keep object recognition stable.
# - The generator still enforces turns, direction diversity, and varied distances
#   so the path is not trivial.
# - To prevent degeneration back to the source rule, accepted cases must expose:
#   1. at least one suffix reversal after the rightmost snapshot, and
#   2. the rightmost snapshot itself must have multiple reachable directions and
#      at least one reachable marker behind the current facing.
#   This makes the fused rule visually recoverable from input/output examples.
#
# c) Hidden assumption audit
# - `_find_agent()` survives unchanged because sprite geometry is unchanged.
# - The source `_choose_target()` relied on two assumptions that are both broken:
#   only one direction had candidates, and the reverse direction was illegal.
#   The replacement logic must scan every aligned marker, check fit, and require
#   only a unique global nearest distance.
# - The source generator also encoded those assumptions when sampling path
#   directions and when rejecting "wrong-direction" future markers. In the combo,
#   the only invariant needed for older heads is weaker: any later aligned marker
#   must be strictly farther than that step's true next marker, otherwise the
#   nearest-choice result would change.
# - A new implicit assumption appears in the solver: a candidate marker is treated
#   as reachable only if both the translated current stamp and the landed stamp fit
#   in-grid. The generator therefore verifies the whole translated path under the
#   same rule profile after choosing the final translation.
#
# d) Placement legality
# - Every actual sprite state must fit inside the frame.
# - Markers never overlap sprite cells, and they are kept non-touching with all
#   previously stamped sprite cells so `_find_agent()` can still isolate the body.
# - The generator checks the full suffix with the same reachability predicate used
#   by the solver, so no accepted case can strand markers or create a tied nearest
#   choice.
#
# e) Visual observability
# - No invisible ordering such as color ids is used. All disambiguation comes from
#   visible row/column geometry and distance.
# - The rightmost snapshot is forced to show multiple reachable directions plus an
#   opposite-direction candidate. A human can see that "follow the tip only" is no
#   longer valid, and the output can only be explained by nearest-distance logic.
#
# f) Visual cleanliness
# - Markers remain isolated single pixels and the four source templates are
#   unchanged, so the actor is still easy to recognize even though markers share
#   its color.
# - Candidate paths that crowd older sprite bodies are rejected to avoid clutter.
#
# g) Change inferability
# - There is no extra color mapping. The only changed signal is geometric, which is
#   repeated across accepted cases through visible multi-direction states and
#   suffix reversals.

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_HERE = Path(__file__).resolve()
_ROOT = next((parent for parent in _HERE.parents if (parent / "solvers").is_dir()), _HERE.parent)
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
VERTICAL_DIRS = {0, 2}
HORIZONTAL_DIRS = {1, 3}


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


def _fits_head(head: tuple[int, int], tname: str, d: int, h: int, w: int) -> bool:
    hy, hx = head
    for oy, ox in TEMPLATES[tname][d]:
        ny = hy + oy
        nx = hx + ox
        if not (0 <= ny < h and 0 <= nx < w):
            return False
    return True


def _reachable_targets(
    head: tuple[int, int],
    cur_d: int,
    markers: set[tuple[int, int]],
    tname: str,
    h: int,
    w: int,
) -> list[tuple[int, int, tuple[int, int]]]:
    """
    All aligned markers whose translated current stamp and landed stamp fit.
    """
    out: list[tuple[int, int, tuple[int, int]]] = []
    for pt in markers:
        pt = (int(pt[0]), int(pt[1]))
        dd = _direction(head, pt)
        if dd is None:
            continue
        if not _fits_head(pt, tname, cur_d, h, w):
            continue
        if not _fits_head(pt, tname, dd, h, w):
            continue
        out.append((_dist(head, pt), dd, pt))
    out.sort(key=lambda item: (item[0], item[1], item[2][0], item[2][1]))
    return out


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    tname: str,
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    reachable = _reachable_targets(head, d, markers, tname, h, w)
    if not reachable:
        return None

    if len(reachable) >= 2:
        assert reachable[0][0] < reachable[1][0], (
            f"head={head} dir={d}: closest markers tie at distance "
            f"{reachable[0][0]}, expected a unique nearest marker"
        )

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
    agent = {(head[0] + oy, head[1] + ox) for oy, ox in TEMPLATES[tname][d]}

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        choice = _choose_target(head, d, markers, tname, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        nd, (ty, tx) = choice

        head = (ty, tx)
        d = nd
        agent = {(ty + oy, tx + ox) for oy, ox in TEMPLATES[tname][d]}
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue the triangle sprite from the latest snapshot until every aligned marker is eaten.

    Input structure:
    - One to four equal-sized frames are concatenated left-to-right, optionally separated by full-height divider columns.
    - In each frame, the triangle sprite and all single-pixel markers share one actor color against a dominant background color.

    Rule:
    - Read only the rightmost frame as the current state.
    - From the sprite head, inspect every remaining marker on the same row or the same column, in any direction.
    - Move to the uniquely nearest reachable aligned marker, even if several directions are populated and even if the chosen move is a 180-degree reversal.
    - Consume that marker, rotate the sprite to the travel direction, and repeat until no markers remain; output only the final sprite body on clean background.

    Key insight: the tip still locates the head and current facing, but facing no longer filters legal targets; the future is determined purely by nearest visible row/column distance.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    start = frames[-1]
    return _simulate_to_end(start, actor)


# --------------- generator (combo-specific) ---------------

N_FRAMES_CHOICES = (1, 2, 3, 4)
N_FRAMES_WEIGHTS = (1.0, 2.0, 2.0, 2.0)
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

MIN_GAP_FOR_VARIANT = 2
MIN_TOTAL_REVERSALS = 1
MIN_SUFFIX_REVERSALS = 1

TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

MAX_ATTEMPTS = 3000

_FRAME_AVAILABILITY = {
    n: sum(n % f == 0 and n // f >= MIN_GAP_FOR_VARIANT for f in N_FRAMES_CHOICES)
    for n in range(*STEPS_RANGE)
}
_VALID_STEPS_BY_FRAMES = {
    f: [n for n in range(*STEPS_RANGE) if n % f == 0 and n // f >= MIN_GAP_FOR_VARIANT]
    for f in N_FRAMES_CHOICES
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
) -> tuple[int, int, int, int, int, str]:
    frame_weights = np.array(N_FRAMES_WEIGHTS, dtype=float)
    frame_weights /= frame_weights.sum()
    n_frames_goal = int(rng.choice(N_FRAMES_CHOICES, p=frame_weights))

    multiples = _VALID_STEPS_BY_FRAMES[n_frames_goal]
    weights = np.array([(n**2) * _FRAME_AVAILABILITY[n] for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = rng.permutation(10)
    bg, actor = int(palette[0]), int(palette[1])

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames_goal, n_steps, h, w, bg, actor, tname


def _count_reversals_from_state(start_dir: int, step_dirs: list[int]) -> int:
    cur = start_dir
    total = 0
    for dd in step_dirs:
        if dd == (cur + 2) % 4:
            total += 1
        cur = dd
    return total


def _rollout_path(
    rng: np.random.Generator,
    tname: str,
    start_dir: int,
    n_steps: int,
    h: int,
    w: int,
) -> tuple[list[tuple[int, int]], list[int], list[int], list[tuple[tuple[int, int], int]]] | None:
    heads: list[tuple[int, int]] = [(0, 0)]
    state_dirs: list[int] = [start_dir]
    step_dirs: list[int] = []
    placements: list[tuple[tuple[int, int], int]] = [((0, 0), start_dir)]

    reversals = 0
    for i in range(n_steps):
        cur = heads[-1]
        cur_dir = state_dirs[-1]
        base_dirs = np.array([cur_dir, (cur_dir + 1) % 4, (cur_dir + 3) % 4, (cur_dir + 2) % 4], dtype=int)
        if reversals == 0 and i >= max(1, n_steps - 3):
            probs = np.array([0.14, 0.24, 0.24, 0.38], dtype=float)
        else:
            probs = np.array([0.20, 0.28, 0.28, 0.24], dtype=float)
        dir_order = [int(x) for x in rng.choice(base_dirs, size=4, replace=False, p=probs / probs.sum())]

        chosen = None
        chosen_dir = None
        dists = list(range(*STEP_DIST_RANGE))
        for next_dir in dir_order:
            dr, dc = DIRS4[next_dir]
            if next_dir == (cur_dir + 2) % 4:
                dist_order = list(reversed(dists))
            else:
                dist_order = list(rng.permutation(dists))
            for dist in dist_order:
                nxt = (cur[0] + int(dist) * dr, cur[1] + int(dist) * dc)
                if nxt in heads:
                    continue
                if any(_touching(nxt, m) for m in heads[1:]):
                    continue
                if not _marker_compatible_with_existing(nxt, heads, state_dirs, tname):
                    continue
                trial_placements = placements + [(nxt, next_dir)]
                if _shift_interval(trial_placements, h, w, tname) is None:
                    continue
                chosen = nxt
                chosen_dir = next_dir
                break
            if chosen is not None:
                break

        if chosen is None:
            return None

        if chosen_dir == (cur_dir + 2) % 4:
            reversals += 1
        step_dirs.append(int(chosen_dir))
        heads.append(chosen)
        state_dirs.append(int(chosen_dir))
        placements.append((chosen, int(chosen_dir)))

    return heads, state_dirs, step_dirs, placements


def _marker_compatible_with_existing(
    nxt: tuple[int, int],
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
) -> bool:
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, state_dirs[j])
        if any(_touching(nxt, cell) for cell in sprite):
            return False
        rel_dir = _direction(old_head, nxt)
        if rel_dir is None:
            continue
        if j < len(heads) - 1 and _dist(old_head, nxt) <= _dist(old_head, heads[j + 1]):
            return False
    return True


def _validate_path(
    start_dir: int,
    step_dirs: list[int],
    heads: list[tuple[int, int]],
) -> tuple[bool, int, list[int], int]:
    n_steps = len(step_dirs)
    dir_set = set(step_dirs)
    if len(dir_set) < MIN_DIR_DIVERSITY:
        return False, 0, [], 0
    if not (dir_set & VERTICAL_DIRS and dir_set & HORIZONTAL_DIRS):
        return False, 0, [], 0

    turns = sum(step_dirs[i] != step_dirs[i - 1] for i in range(1, n_steps))
    if turns < MIN_TURNS_ACCEPT or not (TURN_RATIO_LO <= turns / n_steps <= TURN_RATIO_HI):
        return False, turns, [], 0

    dists = [_dist(heads[i], heads[i + 1]) for i in range(n_steps)]
    if len(set(dists)) < MIN_DIST_VARIETY:
        return False, turns, dists, 0

    reversals = _count_reversals_from_state(start_dir, step_dirs)
    if reversals < MIN_TOTAL_REVERSALS:
        return False, turns, dists, reversals
    return True, turns, dists, reversals


def _trace_rule_profile(
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
    h: int,
    w: int,
) -> list[dict] | None:
    n_steps = len(heads) - 1
    profile: list[dict] = []
    for j in range(n_steps):
        markers = set(heads[j + 1 :])
        reachable = _reachable_targets(heads[j], state_dirs[j], markers, tname, h, w)
        if not reachable:
            return None
        if len(reachable) >= 2 and reachable[0][0] >= reachable[1][0]:
            return None
        if reachable[0][2] != heads[j + 1]:
            return None
        if reachable[0][1] != state_dirs[j + 1]:
            return None

        dirs = sorted({dd for _, dd, _ in reachable})
        profile.append(
            {
                "step": j,
                "dir_count": len(dirs),
                "dirs": dirs,
                "chosen_dir": reachable[0][1],
                "chosen_target": reachable[0][2],
                "nearest_dist": reachable[0][0],
                "multi_dir": len(dirs) >= 2,
                "reversal": reachable[0][1] == (state_dirs[j] + 2) % 4,
                "has_opposite_candidate": ((state_dirs[j] + 2) % 4) in dirs,
            }
        )
    return profile


def _frame_options_from_profile(n_steps: int, profile: list[dict]) -> list[dict]:
    out: list[dict] = []
    for n_frames in N_FRAMES_CHOICES:
        if n_steps % n_frames != 0:
            continue
        gap = n_steps // n_frames
        if gap < MIN_GAP_FOR_VARIANT:
            continue
        last_idx = (n_frames - 1) * gap
        suffix = profile[last_idx:]
        suffix_reversals = sum(int(item["reversal"]) for item in suffix)
        if suffix_reversals < MIN_SUFFIX_REVERSALS:
            continue
        if not profile[last_idx]["multi_dir"]:
            continue
        if not profile[last_idx]["has_opposite_candidate"]:
            continue

        out.append(
            {
                "n_frames": n_frames,
                "gap": gap,
                "last_idx": last_idx,
                "suffix_reversals": suffix_reversals,
                "suffix_multi": sum(int(item["multi_dir"]) for item in suffix),
                "last_frame_reversal": int(profile[last_idx]["reversal"]),
                "visible_multi": sum(int(profile[fi * gap]["multi_dir"]) for fi in range(n_frames)),
                "visible_opposite": sum(
                    int(profile[fi * gap]["has_opposite_candidate"]) for fi in range(n_frames)
                ),
            }
        )
    return out


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


def _choose_translation_and_frames(
    rng: np.random.Generator,
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    placements: list[tuple[tuple[int, int], int]],
    tname: str,
    h: int,
    w: int,
    n_frames_goal: int,
) -> tuple[list[tuple[int, int]], list[dict], dict] | None:
    shift = _shift_interval(placements, h, w, tname)
    if shift is None:
        return None
    tr_lo, tr_hi, tc_lo, tc_hi = shift

    translations = [
        (int(tr), int(tc))
        for tr in range(tr_lo, tr_hi + 1)
        for tc in range(tc_lo, tc_hi + 1)
    ]
    rng.shuffle(translations)

    for tr, tc in translations:
        shifted_heads = [(r + tr, c + tc) for r, c in heads]
        profile = _trace_rule_profile(shifted_heads, state_dirs, tname, h, w)
        if profile is None:
            continue
        frame_options = _frame_options_from_profile(len(profile), profile)
        if not frame_options:
            continue
        preferred = [opt for opt in frame_options if opt["n_frames"] == n_frames_goal]
        if preferred:
            frame_options = preferred
        elif n_frames_goal > 1:
            multi_frame = [opt for opt in frame_options if opt["n_frames"] > 1]
            if multi_frame:
                frame_options = multi_frame
        weights = np.array(
            [
                1.0
                + 0.6 * opt["n_frames"]
                + opt["suffix_reversals"]
                + opt["visible_multi"]
                + opt["visible_opposite"]
                + 2.0 * opt["last_frame_reversal"]
                for opt in frame_options
            ],
            dtype=float,
        )
        weights /= weights.sum()
        pick = int(rng.choice(len(frame_options), p=weights))
        return shifted_heads, profile, frame_options[pick]
    return None


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames_goal, n_steps, h, w, bg, actor, tname = _sample_params(rng)

    start_dir = int(rng.integers(0, 4))
    rollout = _rollout_path(rng, tname, start_dir, n_steps, h, w)
    if rollout is None:
        return None
    heads, state_dirs, step_dirs, placements = rollout

    ok, turns, dists, reversals = _validate_path(start_dir, step_dirs, heads)
    if not ok:
        return None

    chosen = _choose_translation_and_frames(
        rng, heads, state_dirs, placements, tname, h, w, n_frames_goal
    )
    if chosen is None:
        return None
    heads, profile, frame_opt = chosen

    n_frames = int(frame_opt["n_frames"])
    gap = int(frame_opt["gap"])
    last_idx = int(frame_opt["last_idx"])

    palette = [c for c in range(10) if c not in {bg, actor}]
    divider = int(rng.choice(palette)) if n_frames > 1 else None
    placements = [(heads[i], state_dirs[i]) for i in range(n_steps + 1)]

    frames = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, heads[si + 1 :]))

    fhead, fdir = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, fhead, fdir, tname, [])
    inp = _assemble_input(frames, bg, divider)

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
            "n_frames_goal": n_frames_goal,
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
            "start_dir": start_dir,
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "step_distances": dists,
            "reversals_total": reversals,
            "reversals_suffix": int(frame_opt["suffix_reversals"]),
            "multi_dir_suffix": int(frame_opt["suffix_multi"]),
            "visible_multi_snapshots": int(frame_opt["visible_multi"]),
            "visible_opposite_snapshots": int(frame_opt["visible_opposite"]),
            "last_frame_multi": int(profile[last_idx]["multi_dir"]),
            "last_frame_reversal": int(profile[last_idx]["reversal"]),
            "last_frame_has_opposite_candidate": int(profile[last_idx]["has_opposite_candidate"]),
            "last_frame_dirs": list(profile[last_idx]["dirs"]),
            "profile_dir_counts": [int(item["dir_count"]) for item in profile],
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
