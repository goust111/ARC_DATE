# Variant of 5545f144
# Dimension: D1 — 匹配条件：方向对齐 (match_cond)
# Change: 精灵头部方向与标记之间的射线对齐——标记必须位于头部所朝方向的同行或同列半线上（direction_matching），... → 行列对齐但非方向限定
#
# Design analysis for this variant:
# - Concrete objects: the input still shows 1..4 horizontally concatenated frames, each with one
#   monochrome triangle sprite plus same-color single-pixel markers. The output is still one clean
#   frame showing only the final sprite.
# - Core behavioral change: the old task consumed markers from a unique half-line direction
#   (up/right/down/left). Here the sprite ignores facing when locating the next marker: it may move
#   to either side on the current row, or either side on the current column. The only global cue is
#   that exactly one axis (row or column) contains reachable markers at each step.
# - Trace difference: if the head is facing right at (4,4) and remaining markers are at (4,2) and
#   (4,8), the old task would be illegal because one marker is behind the head. In this variant the
#   row axis is uniquely active, so the sprite moves to the nearer row marker, here (4,2), then
#   re-faces left. This is the observable D1 change that the generator must force.
# - Complexity calibration: keep the original templates, frame layout, counts, and path-first
#   synthesis. Only the alignment test changes from direction-level to axis-level. To make the
#   difference visible, generated cases require at least one backward move after the last input
#   snapshot; otherwise samples could collapse back to the old solver's behavior.
# - Implicit assumption review in main(): `_find_agent` still works because sprite templates are
#   unchanged. The old `_choose_target` assumed all reachable markers came from one direction and
#   banned 180-degree reversals; both assumptions are removed. The new chooser instead groups by
#   axis, enforces one active axis, and requires a unique nearest marker on that axis.
# - Placement legality: generator rejects markers that touch any earlier sprite placement, that lie
#   on the wrong axis from any earlier head, or that would be at least as close as an already-fixed
#   earlier target. This preserves unique-next-marker behavior for the solver.
# - Visual observability: the triangle orientation remains visible, and at least one suffix move is
#   explicitly opposite the current facing, so the human-visible rule change is "look along the row
#   or column, not only in front of the tip".
# - Visual cleanliness: markers stay isolated single pixels; sprite shapes are unchanged; snapshots
#   remain uncluttered because future markers cannot touch previously stamped sprite cells.

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
VERTICAL_DIRS = {0, 2}
HORIZONTAL_DIRS = {1, 3}


def _axis_of_dir(d: int) -> int:
    return 0 if d in VERTICAL_DIRS else 1


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


def _template_fits(head: tuple[int, int], tname: str, d: int, h: int, w: int) -> bool:
    hy, hx = head
    for oy, ox in TEMPLATES[tname][d]:
        ny = hy + oy
        nx = hx + ox
        if not (0 <= ny < h and 0 <= nx < w):
            return False
    return True


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


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    tname: str,
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    hy, hx = head
    axis_candidates: dict[int, list[tuple[int, int, tuple[int, int]]]] = {0: [], 1: []}

    for y, x in markers:
        rel_dir = _direction((hy, hx), (int(y), int(x)))
        if rel_dir is None:
            continue
        pt = (int(y), int(x))
        if _template_fits(pt, tname, rel_dir, h, w):
            axis_candidates[_axis_of_dir(rel_dir)].append((abs(y - hy) + abs(x - hx), rel_dir, pt))

    active_axes = [axis for axis, vals in axis_candidates.items() if vals]
    if not active_axes:
        return None
    assert len(active_axes) == 1, (
        f"head={head} dir={d}: markers reachable on multiple axes "
        f"{active_axes}, expected exactly one"
    )

    candidates = axis_candidates[active_axes[0]]
    candidates.sort()
    if len(candidates) >= 2:
        assert candidates[0][0] < candidates[1][0], (
            f"head={head} dir={d}: nearest marker on active axis is not unique"
        )

    _, nd, pt = candidates[0]
    return nd, pt


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, agent, markers = found
    markers = set((int(y), int(x)) for y, x in markers)
    agent = set((int(y), int(x)) for y, x in agent)

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
    Advance a triangular sprite until it has eaten every aligned marker.

    Input structure:
    - Several equal-sized frames may be concatenated left-to-right, with solid divider columns between them.
    - In each frame, one actor color is shared by the triangle sprite and all remaining single-pixel markers.

    Rule:
    - Start from the rightmost frame.
    - At each step, look for markers aligned with the head on the same row or the same column.
      Exactly one axis is allowed to contain reachable markers, though markers may lie on either side of that axis.
    - Move the head to the unique nearest marker on that active axis, consume it, rotate the sprite to match the travel direction, and continue until no markers remain.

    Key insight: the tip still tells you the current orientation, but orientation no longer restricts which marker can be chosen; the deciding cue is which axis contains the remaining aligned markers.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    start = frames[-1]
    return _simulate_to_end(start, actor)


# --------------- generator (variant-specific) ---------------

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
MIN_REVERSE_STEPS = 1
MAX_SEQ_ATTEMPTS = 200

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


def _sprite_cells(head: tuple[int, int], tname: str, d: int) -> set[tuple[int, int]]:
    hr, hc = head
    return {(hr + dr, hc + dc) for dr, dc in TEMPLATES[tname][d]}


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


def _suffix_has_backward_step(start_dir: int, step_dirs: list[int], start_idx: int) -> bool:
    facing = start_dir
    for i, step_dir in enumerate(step_dirs):
        if i >= start_idx and step_dir == (facing + 2) % 4:
            return True
        facing = step_dir
    return False


def _rollout_path(
    rng: np.random.Generator, tname: str, start_dir: int, n_steps: int, h: int, w: int, last_idx: int
) -> tuple[list[tuple[int, int]], list[int], list[int], list[tuple[tuple[int, int], int]]] | None:
    heads: list[tuple[int, int]] = [(0, 0)]
    state_dirs: list[int] = [start_dir]
    step_dirs: list[int] = []
    fit_placements: list[tuple[tuple[int, int], int]] = [((0, 0), start_dir)]

    for i in range(n_steps):
        cur = heads[-1]
        cur_dir = state_dirs[-1]
        chosen = None
        base_dirs = np.array([cur_dir, (cur_dir + 1) % 4, (cur_dir + 3) % 4, (cur_dir + 2) % 4], dtype=int)
        if i >= last_idx:
            probs = np.array([0.16, 0.21, 0.21, 0.42], dtype=float)
        else:
            probs = np.array([0.24, 0.27, 0.27, 0.22], dtype=float)
        dir_order = [int(x) for x in rng.choice(base_dirs, size=4, replace=False, p=probs / probs.sum())]

        for next_dir in dir_order:
            dr, dc = DIRS4[next_dir]
            dist_order = list(rng.permutation(np.arange(*STEP_DIST_RANGE)))
            for dist in dist_order:
                nxt = (cur[0] + int(dist) * dr, cur[1] + int(dist) * dc)
                if nxt in heads:
                    continue
                if any(_touching(nxt, m) for m in heads[1:]):
                    continue
                if not _marker_compatible_with_existing(nxt, heads, state_dirs, step_dirs, next_dir, tname):
                    continue
                trial_fit = fit_placements + [(nxt, cur_dir), (nxt, next_dir)]
                if _shift_interval(trial_fit, h, w, tname) is None:
                    continue
                chosen = (next_dir, nxt)
                break
            if chosen is not None:
                break

        if chosen is None:
            return None

        next_dir, nxt = chosen
        step_dirs.append(next_dir)
        heads.append(nxt)
        state_dirs.append(next_dir)
        fit_placements.extend([(nxt, cur_dir), (nxt, next_dir)])

    return heads, state_dirs, step_dirs, fit_placements


def _marker_compatible_with_existing(
    nxt: tuple[int, int],
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    step_dirs: list[int],
    next_dir: int,
    tname: str,
) -> bool:
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, state_dirs[j])
        if any(_touching(nxt, cell) for cell in sprite):
            return False
        rel_dir = _direction(old_head, nxt)
        if rel_dir is None:
            continue
        expected_dir = step_dirs[j] if j < len(step_dirs) else next_dir
        if _axis_of_dir(rel_dir) != _axis_of_dir(expected_dir):
            return False
        if j < len(heads) - 1 and _dist(old_head, nxt) <= _dist(old_head, heads[j + 1]):
            return False
    return True


def _validate_path(
    start_dir: int, step_dirs: list[int], heads: list[tuple[int, int]], last_idx: int
) -> tuple[bool, int, int, list[int]]:
    n_steps = len(step_dirs)
    dir_set = set(step_dirs)
    if len(dir_set) < MIN_DIR_DIVERSITY:
        return False, 0, 0, []
    if not (dir_set & VERTICAL_DIRS and dir_set & HORIZONTAL_DIRS):
        return False, 0, 0, []

    turns = sum(step_dirs[i] != step_dirs[i - 1] for i in range(1, n_steps))
    reverse_steps = sum(step_dirs[i] == (step_dirs[i - 1] + 2) % 4 for i in range(1, n_steps))
    if turns < MIN_TURNS_ACCEPT or not (TURN_RATIO_LO <= turns / n_steps <= TURN_RATIO_HI):
        return False, turns, reverse_steps, []
    dists = [_dist(heads[i], heads[i + 1]) for i in range(n_steps)]
    if len(set(dists)) < MIN_DIST_VARIETY:
        return False, turns, reverse_steps, dists
    if not (dir_set & {0, 2} and dir_set & {1, 3}):
        return False, turns, reverse_steps, dists
    if reverse_steps < MIN_REVERSE_STEPS:
        return False, turns, reverse_steps, dists
    if not _suffix_has_backward_step(start_dir, step_dirs, last_idx):
        return False, turns, reverse_steps, dists
    return True, turns, reverse_steps, dists


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
    last_idx = (n_frames - 1) * gap

    start_dir = int(rng.integers(0, 4))
    rollout = _rollout_path(rng, tname, start_dir, n_steps, h, w, last_idx)
    if rollout is None:
        return None
    heads, state_dirs, step_dirs, fit_placements = rollout

    ok, turns, reverse_steps, dists = _validate_path(start_dir, step_dirs, heads, last_idx)
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
            "n_frames": n_frames,
            "template": tname,
            "bg": bg,
            "actor": actor,
            "divider": divider,
            "total_steps": n_steps,
            "gap": gap,
            "turns": turns,
            "turn_ratio": turns / n_steps,
            "reverse_steps": reverse_steps,
            "direction_diversity": len(set(step_dirs)),
            "start_dir": start_dir,
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "step_distances": dists,
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
